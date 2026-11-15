"""Assembling kernels: PTX text to a cubin with the CUDA toolkit's ``ptxas``.

Every kernel's code is PTX from the Kernel IR to PTX compiler, and ``ptxas`` is the only tool it needs. It is
GPU-free, and every cubin lands in a content-addressed disk cache keyed on the PTX text, the target and ptxas's
version, so a compile pool can warm the cache off the GPU and the runtime loads by path. The kernel's precision
decisions are in its instructions; ptxas takes no flag but the target.

- :func:`compile_to_cubin` — assemble for the text's own target, or an explicit one, into the cache.
- :func:`compile_kernel` — the same, for the cubin THIS process is about to launch: the live device's ISA, with the
  arch-specific suffix when the text's target has it.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from emmy import config

logger = logging.getLogger(__name__)

#: What to do when no ptxas is found.
_INSTALL = (
    "emmy assembles every kernel with the CUDA toolkit's ptxas: install the toolkit (its ptxas ships in NVIDIA's "
    "cuda-nvcc package) and put ptxas on PATH, or point $CUDA_HOME at the toolkit"
)


def cubin_cache_dir() -> Path:
    """Directory holding the content-addressed cubin cache (``EMMY_CUBIN_CACHE``)."""
    return config.cubin_cache_dir()


@functools.cache
def ptxas_path() -> str | None:
    """The ``ptxas`` binary (PATH, then ``$CUDA_HOME``/``$CUDA_PATH``), or ``None`` when there is none. Looked up
    once per process."""
    found = shutil.which("ptxas")
    if found:
        return found
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root and (candidate := Path(root) / "bin" / "ptxas").exists():
            return str(candidate)
    return None


def _ptxas() -> str:
    """The ``ptxas`` binary; an error that says what to install when there is none."""
    found = ptxas_path()
    if found is None:
        raise RuntimeError(f"ptxas not found — {_INSTALL}")
    return found


def _target(source: str) -> str:
    """The target the PTX ``source`` names on its ``.target`` line (``sm_120``, ``sm_90a``): the compiler writes the
    arch-specific one where an instruction needs it."""
    found = re.search(r"^\.target (\S+)", source, re.MULTILINE)
    if found is None:
        raise ValueError("PTX text names no .target")
    return found.group(1)


def _launchable_arch(source: str) -> str:
    """The arch for a cubin THIS process is about to launch: the LIVE device's, with the suffix of the text's target.

    ``--target`` means "make every lowering decision as if on that GPU" (TMA / cp.async gating, atom family,
    goldens), and the text names that target, because an ahead-of-time bake really is building for that card. A
    cubin the current process then launches has a harder constraint: a cubin is not forward-compatible, so one
    assembled for a lower target will not load here at all (``CUDA_ERROR_NO_BINARY_FOR_GPU``). Only the ISA the
    already-chosen instructions are assembled into follows the live device. Falls back to the text's target when no
    device answers (there is nothing to launch on anyway)."""
    from emmy.compiler.backend.cuda.device import compute_capability  # noqa: PLC0415

    target = _target(source)
    cap = compute_capability()
    if cap is None:
        return target
    return f"sm_{cap[0]}{cap[1]}" + ("a" if target.endswith("a") else "")


@functools.cache
def version() -> str:
    """Short digest of ``ptxas --version``: part of the cubin cache's key and of a pack's, so a toolkit upgrade never
    reuses a cubin an older ptxas assembled. Run once per process."""
    ptxas = _ptxas()
    text = subprocess.run([ptxas, "--version"], check=True, capture_output=True, text=True).stdout
    return hashlib.sha1(text.encode()).hexdigest()[:12]


#: The version of how a cubin is assembled from its text, part of the cache's key: a change to it (the one-wave
#: directive dropped where ptxas spills) re-assembles every kernel.
_ASSEMBLY = "2"


def _cubin_key(source: str, arch: str, level: str | None = None) -> str:
    """Content address of a cubin: identical PTX text, target, level and ptxas give the same cubin, so the persistent
    cache is safe to share across (even concurrent) runs."""
    h = hashlib.sha1()
    for part in (source, arch, version(), _ASSEMBLY, *([level] if level else [])):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def compile_to_cubin(source: str, name: str, *, arch: str | None = None, level: str | None = None) -> Path:
    """Assemble the PTX ``source`` of kernel ``name`` for ``arch`` (the text's own target when omitted) with ptxas, at
    its optimization ``level`` (``-O0``; ptxas's default when omitted, which every kernel the compiler runs takes),
    content-addressed in the on-disk cache.
    Idempotent and atomic (assemble to a temp file, then ``os.replace``), so concurrent compilers and the bench
    loader never observe a half-written cubin. GPU-free. Raises ``RuntimeError`` without ptxas and
    ``CalledProcessError`` when ptxas refuses the text."""
    arch = arch or _target(source)
    cache = cubin_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_cubin_key(source, arch, level)}.cubin"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(dir=cache) as td:
        text, cubin = Path(td) / "k.ptx", Path(td) / "k.cubin"
        text.write_text(source)
        if _assemble(text, cubin, arch, level) and _ONE_WAVE in source:
            # Told the launch is one wave, ptxas spends every register a thread may hold, and a kernel whose
            # registers already near that cap then spills: without the directive it keeps its own choice.
            text.write_text(source.replace(_ONE_WAVE, ""))
            _assemble(text, cubin, arch, level)
        os.replace(cubin, out)  # atomic publish
    return out


#: The directive the PTX compiler states for a launch of one wave.
_ONE_WAVE = ".minnctapersm 1\n"


def _assemble(text: Path, cubin: Path, arch: str, level: str | None) -> bool:
    """Assemble ``text`` into ``cubin``; whether ptxas spilled registers to local memory."""
    command = [_ptxas(), *([level] if level else []), f"-arch={arch}", "-m64", "-v", "-o", str(cubin), str(text)]
    run = subprocess.run(command, check=True, capture_output=True)
    return any(int(spill) > 0 for spill in re.findall(rb"(\d+) bytes spill stores", run.stdout + run.stderr))


def compile_kernel(source: str, name: str, *, level: str | None = None) -> Path:
    """Assemble ``name`` (cached) for the cubin THIS process is about to launch, at ``level`` (ptxas's default when
    omitted), and return its path — the runtime loads it from there."""
    try:
        return compile_to_cubin(source, name, arch=_launchable_arch(source), level=level)
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)"
        logger.error("ptxas failed for kernel %r:\n%s", name, detail)
        raise RuntimeError(f"ptxas failed for kernel {name!r}: {detail[-400:]}") from exc
