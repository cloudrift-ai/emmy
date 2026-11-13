"""Offline kernel compilation via the ``nvcc`` binary (ptxas).

``nvcc --cubin`` is the only compile path: it is GPU-free, ~3× faster than a cold NVRTC
compile on the complex tile-search kernels that dominate autotune, and its cubin loads with
no driver JIT. Every kernel lands in a content-addressed disk cache, so a compile **pool** can
warm the cache off the GPU and the runtime loads by path.

- :func:`compile_to_cubin` — ``nvcc --cubin`` for an explicit target into the cache.
- :func:`compile_kernel` — the same, for the cubin THIS process is about to launch: the live
  device's ISA, with the arch-specific suffix when the kernel needs it.

``nvcc`` is required — there is no NVRTC fallback. Install the CUDA toolkit (``nvcc`` on
``$PATH`` or under ``$CUDA_HOME``/``$CUDA_PATH``) or set ``EMMY_NO_NVCC=1`` and accept the
resulting hard error on any kernel.
"""

from __future__ import annotations

import functools
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from emmy import config

logger = logging.getLogger(__name__)


def effective_flags() -> list[str]:
    """The full nvcc flag list: the flags of the live flush and fusion decisions (:func:`precision_flags`) plus any
    extra flags from the ``EMMY_NVCC_FLAGS`` env var (space-separated), which the CLI commands
    set — ``tune``, ``compile`` and ``run`` all default to nvcc's own cicc -O3.
    Read fresh each call so a per-invocation override / the bench-worker
    subprocess (which inherits the env) both see the same value, and so the
    flags fold into the cache key."""
    from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

    return [*precision_flags(precision_pins()), *config.nvcc_flags().split()]


def precision_flags(pins: dict[str, bool]) -> list[str]:
    """The nvcc flags of the two precision decisions among ``pins`` (``space.precision_pins``) that hold for a whole
    kernel: flush to zero and multiply-add fusion, each spelled where it is not nvcc's default (no flush, fusion).

    Every other precision decision is an operation of the kernel body (``lowering/kernel/130_approximate_math``), so
    nvcc compiles at its own defaults for the rest: precise division and square root, the accurate math library. An
    operation the pass did not pick stays exact. No ``--use_fast_math``: that flag is these two decisions and the
    approximate instructions together, and it would approximate what the kernel says is exact."""
    from emmy.compiler.pipeline.search.space import CONTRACT_FMA, FLUSH_TO_ZERO  # noqa: PLC0415

    return [*(["--ftz=true"] if pins[FLUSH_TO_ZERO.name] else []), *([] if pins[CONTRACT_FMA.name] else ["--fmad=false"])]


def cubin_cache_dir() -> Path:
    """Directory holding the content-addressed cubin cache (``EMMY_CUBIN_CACHE``)."""
    return config.cubin_cache_dir()


@functools.cache
def nvcc_path() -> str | None:
    """Resolve the ``nvcc`` binary (PATH, then ``$CUDA_HOME``/``$CUDA_PATH``),
    or ``None`` when unavailable. Cached — looked up once per process."""
    if config.nvcc_disabled():
        return None
    found = shutil.which("nvcc")
    if found:
        return found
    for env in ("CUDA_HOME", "CUDA_PATH"):
        root = os.environ.get(env)
        if root and (cand := Path(root) / "bin" / "nvcc").exists():
            return str(cand)
    return None


def device_arch(arch_specific: bool) -> str:
    """The active compiler target, plus the ``a`` suffix when the kernel needs the ARCH-SPECIFIC ISA.

    Two instruction families need it, for unrelated reasons: TMA, and the block-scaled fp4 mma
    (ptxas assembles that one only for an arch-suffixed sm_120a). So the flag names the need, not
    either instruction."""
    from emmy.compiler.target import compute_capability  # noqa: PLC0415

    major, minor = compute_capability()
    cap = f"{major}{minor}"
    return f"sm_{cap}" + ("a" if arch_specific else "")


def _launchable_arch(arch_specific: bool) -> str:
    """The arch for a cubin THIS process is about to launch — the LIVE device's, not the target's.

    ``--target`` means "make every lowering decision as if on that GPU" (TMA / cp.async gating,
    atom family, goldens), and :func:`device_arch` follows it because an ahead-of-time bake really
    is building for that card. A cubin the current process then launches has a harder constraint:
    a cubin is not forward-compatible, so one assembled for a lower target will not load here at
    all (``CUDA_ERROR_NO_BINARY_FOR_GPU``). Only the ISA the ALREADY-CHOSEN instructions are
    assembled into follows the live device; nothing about the lowering changes. Falls back to the
    target when no device answers (there is nothing to launch on anyway)."""
    from emmy.compiler.backend.cuda.device import compute_capability  # noqa: PLC0415

    cap = compute_capability()
    if cap is None:  # no device to probe; the target is the only answer left
        return device_arch(arch_specific)
    return f"sm_{cap[0]}{cap[1]}" + ("a" if arch_specific else "")


@functools.cache
def _toolkit_tag() -> str:
    """Short digest of the ``nvcc`` toolchain (``nvcc --version``), folded into
    the cache key so a CUDA upgrade never reuses a cubin compiled by an older
    ptxas (which could emit different / worse SASS for the same source). Run
    once per process."""
    nvcc = nvcc_path()
    try:
        ver = subprocess.run([nvcc, "--version"], check=True, capture_output=True, text=True).stdout
    except Exception:  # noqa: BLE001 — fall back to the path; never block a compile on version probing
        ver = nvcc or "?"
    return hashlib.sha1(ver.encode()).hexdigest()[:12]


def _cubin_key(source: str, name: str, arch: str) -> str:
    # Content-addressed: identical (source, name, arch, toolkit, flags) → same
    # cubin, so the persistent cache is safe to share across (even concurrent)
    # runs. Toolkit + flags are in the key so an nvcc / opt-level / flags change
    # recompiles rather than serving a stale or wrong-opt cubin.
    # PTX carries its precision decisions in its instructions and takes no nvcc flag, so the flags are no part of
    # its key: one text is one cubin under every flag list.
    flags = [] if is_ptx(source) else effective_flags()
    h = hashlib.sha1()
    for part in (source, name, arch, _toolkit_tag(), "\x1f".join(flags)):
        h.update(part.encode())
        h.update(b"\0")
    return h.hexdigest()


def is_ptx(source: str) -> bool:
    """Whether ``source`` is PTX text, which opens with its ``.version`` directive, and not CUDA C++."""
    return source.lstrip().startswith(".version")


def compile_to_cubin(source: str, name: str, *, arch: str) -> Path:
    """Compile ``source`` to a cubin with ``nvcc --cubin``, content-addressed in
    the on-disk cache. Idempotent + atomic (compile to a temp file, then
    ``os.replace``) so concurrent compilers / the bench loader never observe a
    half-written cubin. GPU-free — safe to call from a worker pool. Raises
    ``RuntimeError`` if ``nvcc`` is unavailable; ``CalledProcessError`` on a
    compile error (caller decides whether to fall back).

    A ``source`` that is PTX (:func:`is_ptx`) skips the C++ front end: the toolkit's ``ptxas``, beside ``nvcc``,
    assembles it into the same cache. No nvcc flag applies to it; its precision decisions are in its instructions.

    The opt level comes from :func:`effective_flags` (``EMMY_NVCC_FLAGS``). Every command —
    ``tune`` included — defaults to nvcc's own -O3, the deployable regime, so a tuned latency is
    the deployed one. A lower level pinned by hand compiles no faster on current codegen and
    mis-ranks by tile area; see ``backend/cuda/ARCHITECTURE.md``."""
    nvcc = nvcc_path()
    if nvcc is None:
        raise RuntimeError("nvcc unavailable")
    cache = cubin_cache_dir()
    cache.mkdir(parents=True, exist_ok=True)
    out = cache / f"{_cubin_key(source, name, arch)}.cubin"
    if out.exists():
        return out
    with tempfile.TemporaryDirectory(dir=cache) as td:
        tmp_cubin = Path(td) / "k.cubin"
        if is_ptx(source):
            text = Path(td) / "k.ptx"
            command = [str(Path(nvcc).with_name("ptxas")), f"-arch={arch}", "-m64", "-o", str(tmp_cubin), str(text)]
        else:
            text = Path(td) / "k.cu"
            command = [nvcc, "--cubin", f"-arch={arch}", *effective_flags(), "-o", str(tmp_cubin), str(text)]
        text.write_text(source)
        subprocess.run(command, check=True, capture_output=True)
        os.replace(tmp_cubin, out)  # atomic publish
    return out


def compile_kernel(source: str, name: str, *, arch_specific: bool) -> Path:
    """Compile ``name`` (via nvcc, cached) for the cubin THIS process is about to launch and
    return its path — the runtime loads it from there. Raises ``RuntimeError`` if ``nvcc`` is
    unavailable: there is no NVRTC fallback."""
    if nvcc_path() is None:
        raise RuntimeError(
            "nvcc unavailable — emmy requires the CUDA toolkit's nvcc binary on PATH / under $CUDA_HOME "
            "(kernels compile offline into the cubin cache; there is no NVRTC fallback)"
        )
    try:
        return compile_to_cubin(source, name, arch=_launchable_arch(arch_specific))
    except subprocess.CalledProcessError as exc:
        detail = exc.stderr.decode(errors="replace") if exc.stderr else "(no stderr)"
        tool = "ptxas" if is_ptx(source) else "nvcc"
        logger.error("%s failed for kernel %r:\n%s", tool, name, detail)
        raise RuntimeError(f"{tool} compile failed for kernel {name!r}: {detail[-400:]}") from exc
