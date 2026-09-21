"""Measurement freeze — a digest-pinned snapshot of a DB instance's ``perf`` rows.

A tune DB is a live store (tunes and imports write into it), so a model fit or evaluated directly
from it is not reproducible: two runs of the same fitter can see different data. A *freeze* is a
snapshot extracted from a DB instance whose digest pins exactly which measurements a fit saw — the
fit becomes a pure function of (repo, freeze digest).

Freeze v5 is a DIRECTORY of per-GPU YAML files — a ``gpu_name`` / ``compute_cap`` header plus a
``configs`` list — beside a ``manifest.json`` carrying provenance and content digests. Each row is
a ``perf`` row minus what its file header already says: the kernel it measured, the sizes its
symbolic dims were bound to, its knobs (``S_*`` stamps + tunables, exactly as the DB stores them),
the opt level and the residual compiler flags, the status, the latency statistics, ``captured``,
``measured_at`` and a failure's ``error``. Two card-independent files ride beside them when the
instance holds definitions: ``kernels.yaml`` (the ``kernel`` rows of every kernel a frozen row or a
kernel set names — identity, C name, Loop IR wire) and ``kernel_sets.yaml`` (every ``kernel_set``
row), so an import can enumerate from the freeze alone. The manifest lists every file with its
kind (``perf``, ``kernels``, ``kernel_sets``) and its digest. Device ``H_*`` features are never
stored: readers derive them from the card (``data.sample.measured_features``).

What freezes (see :func:`freeze_reason`): every CUDA row measured in the deployable regime on a
card the GPU registry knows, spelled in the current featurizer vocabulary, that passes the
physical-plausibility predicates. ``bench_fail`` rows are kept as durable negatives.

Determinism contract: freezing the same rows twice yields the same digests. Every row
serializes to one canonical JSON line (sorted keys, fixed separators, ``allow_nan=False``);
rows within a file sort by that line (total, content-derived order); the per-file
``sha256`` covers exactly those lines — NOT the YAML bytes, so integrity is content-level
and immune to YAML style — and the manifest's top-level ``sha256`` folds the sorted
per-file digests. ``created_at`` never enters any digest.

:func:`load_freeze` hard-errors — never a silent fallback — on a missing/foreign/corrupt
manifest, a ``freeze_ver`` mismatch, a manifest-listed file missing, or a per-file digest
mismatch. It is not a reader's entry point: ``emmy dataset import`` loads a freeze into a DB
instance (each row's ``source`` naming the freeze's digest), and every reader reads the instance.

One freeze is CHECKED IN, at ``search/freezes/``, and is what ``config.freeze_path()`` resolves
to — the default source of ``emmy dataset import``: the prior's evaluation corpus should be an
artifact, not whatever a machine happens to hold. Its payload YAML is tracked in git LFS (multi-MB
per card); its manifest is plain git so the digest and the version stamps stay diffable. It is
deliberately not wheel package-data — a wheel install never fits or evaluates a prior.

Produced by ``emmy dataset freeze``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
import shutil
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import NamedTuple

import yaml

from emmy.compiler.pipeline.search.db import KernelRow, KernelSetRow, PerfRow, PerfStats, SearchDB
from emmy.compiler.pipeline.search.features import DEPLOYABLE_OPT, FEATURIZER_VERSION

logger = logging.getLogger(__name__)

FREEZE_KIND = "emmy-measurement-freeze"
FREEZE_VER = 5
MANIFEST_NAME = "manifest.json"
KERNELS_NAME = "kernels.yaml"
KERNEL_SETS_NAME = "kernel_sets.yaml"


class Freeze(NamedTuple):
    """What :func:`load_freeze` read: the manifest and the three tables' rows."""

    manifest: dict
    perf: list[PerfRow]
    kernels: list[KernelRow]
    kernel_sets: list[KernelSetRow]


def freeze_reason(row: PerfRow) -> str | None:
    """Why ``row`` is excluded from a measurement freeze and from every measured-pool reader, or
    ``None`` to keep it.

    THE admission filter, and nothing else — keep every row measured in the DEPLOYABLE regime, on a
    card the GPU registry knows, spelled in the current featurizer vocabulary, that passes the shared
    plausibility predicates. ``bench_fail`` rows reach the keep path by construction: both predicates
    return ``None`` for non-``ok`` rows, so failures are kept as negative examples without a special case.

    The regime gate is what keeps a freeze a fair yardstick. A freeze is the corpus a reported
    prior number is computed over, and a measurement taken under a non-deployable opt level
    answers a question nothing asks: nothing trains on it (``Prior.add_rows``) and no deploy
    reads it. Kept, it would put half a card's pools in a lane no one runs, so half the headline
    number would describe a regime that does not exist. The same goes for extra compiler flags: a
    row measured under any was measured in some other regime."""
    from emmy import gpu  # noqa: PLC0415

    if gpu.by_name(row.gpu) is None:
        return "unknown card (not in the GPU registry)"
    if row.feat_ver != FEATURIZER_VERSION:
        return f"stale feat_ver {row.feat_ver} != current {FEATURIZER_VERSION}"
    if row.opt != DEPLOYABLE_OPT:
        return f"non-deployable regime (H_opt={row.opt:g})"
    if row.flags:
        return "non-default compiler flags"
    if not any(k.startswith("S_") for k in row.knobs):
        return "no structural stamps (a whole-slice or kernel-set row)"
    reason = implausible_value_reason(row)
    if reason is not None:
        return f"implausible value: {reason}"
    reason = impossible_kernel_reason(row)
    if reason is not None:
        return f"impossible kernel: {reason}"
    return None


def implausible_value_reason(row: PerfRow) -> str | None:
    """THE physical-plausibility predicate for a ``perf`` row's median — the reason it
    cannot be a real measurement, or ``None`` when it's plausible/ungateable. Part of
    :func:`freeze_reason`, so a freeze and every measured-pool reader drop the same rows.

    The bound is the arithmetic-intensity floor the golden A/B integrity gate uses: the
    throughput a latency implies from the row's stamped shape must stay below the card's
    recorded peak. ``2·free·reduce_max`` FLOPs is the true work ONLY when the reduce axes
    are **disjoint** from the output — the iteration space is then exactly
    ``free_prod × reduce`` (a contraction, or a pure output-shrinking reduce) — which the
    stamps certify as ``S_loop_depth == n_free + n_reduce + n_symbolic`` (every loop of
    the nest is either a counted free/symbolic output axis or a counted reduce axis). A
    norm/softmax kernel fails that equality — its reduced axis is part of the full-size
    output, so ``free_prod`` already contains it and the product overcounts by the reduce
    extent (a cooperative norm legitimately runs ~100x its serial sibling, so a latency
    floor there would flag honest rows); a fused multi-node kernel (attention) fails it
    too. Both stay ungated rather than falsely flagged — the identity was verified
    against every stamp combination in the 2026-07 sweep stores. ``reduce_max``, not
    ``reduce_prod``, keeps the bound a lower estimate of work even off the exact case. A
    symbolic axis is excluded from the stamped products, so the sizes the row was benched at
    (``bindings``) re-enter as one factor, their product; a row recorded before the sizes were
    stored (the converted freeze) is read at the default hint, the size those benches ran
    at. Ungateable rows also pass on: non-``ok`` status
    (a fail sentinel is not a measurement), no stamped shape, unknown card or unrecorded
    peak, and rows outside the current featurizer vocabulary (their stamps aren't trusted
    enough to judge)."""
    if row.status != "ok" or row.stats.median <= 0 or row.feat_ver != FEATURIZER_VERSION:
        return None
    f = row.knobs
    free = float(f.get("S_ext_free_prod") or 0.0)
    red = float(f.get("S_ext_reduce_max") or 0.0)
    if free <= 0 or red <= 0:
        return None
    # Work = free x red only when every loop multiplies the iteration space (disjoint axes).
    depth = float(f.get("S_loop_depth") or 0.0)
    n_sym = float(f.get("S_ext_n_symbolic_axis") or 0.0)
    n_axes = float(f.get("S_ext_n_free_axis") or 0.0) + float(f.get("S_ext_n_reduce_axis") or 0.0) + n_sym
    if depth <= 0 or depth != n_axes:
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    half = any(k.startswith("S_dtype_") and "f16" in k and v for k, v in f.items())  # f16 / bf16
    peak = spec.peak_tflops("fp16" if half else "fp32")
    if not peak:
        return None
    from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

    hint = math.prod(row.bindings.values()) if row.bindings else (DEFAULT_SEQ_HINT if n_sym > 0 else 1)
    implied = 2.0 * free * red * hint / row.stats.median / 1e6  # FLOP / µs -> TFLOP/s
    if implied > peak:
        return f"implies {implied:.0f} TFLOP/s > {peak:.0f} device peak"
    return None


def impossible_kernel_reason(row: PerfRow) -> str | None:
    """The *validity* companion to :func:`implausible_value_reason` — the reason the row's
    stamped kernel could never have launched, or ``None``. A ``cp.async``-staged warp tile
    whose slab (``depth · (tile_m + tile_n) · bk_elems · elem_bytes``, the
    warp stage sizing) exceeds the card's dynamic-smem opt-in cap cannot
    materialize — pre-#330 code stamped such stages anyway, the materializer rejected the
    main kernel, and the bench recorded the surviving combine kernel's cached µs as an
    ``ok`` measurement of the whole op. On shapes too small for the latency floor to
    notice (square.512's combine implies a legal 133 TFLOP/s), THIS check is the only one
    that catches the class: the measurement is of a kernel set that provably didn't
    include the stamped kernel."""
    if row.status != "ok" or row.feat_ver != FEATURIZER_VERSION:
        return None
    f = row.knobs
    tile_spec = next((str(v) for k, v in f.items() if k.startswith("TILE") and v), "")
    stage_spec = next((str(v) for k, v in f.items() if k.startswith("STAGE") and v), "")
    if not tile_spec or not stage_spec.startswith("d"):
        return None
    from emmy.compiler.ir.schedule import Stage, Tile, Work  # noqa: PLC0415

    try:
        work = Work.parse(str(f.get("WORK") or ""))  # the row's unit widths live here, not in TILE
        tp, st = Tile.parse(tile_spec, work), Stage.parse(stage_spec)
    except ValueError:
        return None
    if not tp.is_warp:
        return None
    if st.transport != "smem-async":
        return None
    from emmy import gpu  # noqa: PLC0415

    spec = gpu.by_name(row.gpu) if row.gpu else None
    if spec is None:
        return None
    atom = tp.atom
    tile_m = tp.units_m * tp.reg_m * atom.atom_m
    tile_n = tp.units_n * tp.reg_n * atom.atom_n
    slab = st.depth * (tile_m + tile_n) * tp.bk * atom.atom_k * atom.operand_dtype("a").nbytes
    if slab > spec.smem_optin:
        return f"staged slab {slab} B > {spec.smem_optin} B dynamic-smem cap (kernel cannot launch)"
    return None


def _row_payload(row: PerfRow) -> dict:
    """One freeze row: the ``perf`` row minus what its file header holds (card, compute capability)."""
    s = row.stats
    return {
        "kernel": row.kernel,
        "bindings": row.bindings,
        "knobs": row.knobs,
        "opt": row.opt,
        "flags": row.flags,
        "status": row.status,
        "stats": {"median": s.median, "min": s.min, "max": s.max, "mean": s.mean, "variance": s.variance, "n_samples": s.n_samples},
        "captured": row.captured,
        "measured_at": row.measured_at,
        "error": row.error,
    }


def _row_line(payload: dict) -> bytes:
    """A payload's canonical JSON line (bytes) — sorted keys, fixed separators, no NaN
    tokens (``allow_nan=False`` hard-errors instead of emitting nonstandard JSON). This
    exact spelling is the row sort order AND the digested content, so determinism and
    integrity are content-level, immune to YAML style."""
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()


def _gpu_filename(gpu_name: str, cap: tuple[int, int]) -> str:
    """The per-GPU YAML file name, mirroring the ``goldens/`` convention —
    e.g. ``nvidia_geforce_rtx_4090_sm89.yaml``."""
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "unknown_gpu"
    return f"{slug}_sm{cap[0]}{cap[1]}.yaml"


def _repo_commit() -> str:
    """``git rev-parse HEAD`` of the checkout this module runs from, ``-dirty``-suffixed
    when the tree has uncommitted changes; ``"unknown"`` outside a repo / without git."""
    here = Path(__file__).resolve().parent
    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], cwd=here, capture_output=True, text=True, timeout=10)
        if sha.returncode != 0:
            return "unknown"
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=here, capture_output=True, text=True, timeout=10)
        if dirty.returncode != 0:
            return "unknown"  # dirtiness undeterminable — never stamp a clean-tree claim we can't back
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _write_rows(tmp: Path, name: str, payloads: list[dict], doc: dict, key: str) -> tuple[str, int]:
    """Write one freeze file — ``doc`` with its ``key`` holding ``payloads`` in canonical order — and
    return its content digest and row count."""
    payloads = sorted(payloads, key=_row_line)
    digest = hashlib.sha256()
    for p in payloads:
        digest.update(_row_line(p))
    (tmp / name).write_text(yaml.safe_dump({**doc, key: payloads}, sort_keys=True, width=120))
    return digest.hexdigest(), len(payloads)


def write_freeze(db_path: Path | str, out_dir: Path | str, *, note: str = "") -> dict:
    """Read the DB instance at ``db_path`` read-only — its CUDA ``perf`` rows filtered through
    :func:`freeze_reason`, the ``kernel`` rows those rows and the kernel sets name, every
    ``kernel_set`` row — and atomically write the freeze DIRECTORY at ``out_dir``: one
    per-``(gpu, compute_cap)`` YAML file, the two definition files when there is anything to put in
    them, plus ``manifest.json``. Returns the manifest dict (so a caller reports counts + digest
    without re-reading). Hard-errors when nothing survives the filter — a zero-row freeze means the
    wrong DB, not an empty dataset. An existing ``out_dir`` is replaced only when it is itself a
    freeze (has a manifest) — anything else is refused rather than deleted."""
    db = SearchDB.open_readonly(db_path)
    try:
        rows = list(db.iter_perf_rows(backend="cuda"))
        kernels = list(db.iter_kernels())
        kernel_sets = list(db.iter_kernel_sets())
    finally:
        db.close()
    kept = []
    dropped: Counter[str] = Counter()
    for row in rows:
        reason = freeze_reason(row)
        if reason is None:
            kept.append(row)
        else:
            dropped[reason.split(":")[0]] += 1
    for reason, n in dropped.most_common():
        logger.info("[freeze] dropped %d row(s): %s", n, reason)
    if not kept:
        raise RuntimeError(
            f"no freezable rows in {db_path} — wrong DB, or its rows predate the card key or the current featurizer vocabulary"
        )

    by_card: dict[tuple[str, tuple[int, int]], list] = {}
    for row in kept:
        by_card.setdefault((row.gpu, divmod(row.cc, 10)), []).append(row)

    out = Path(out_dir)
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)

    files: dict[str, dict] = {}
    for (gpu, cap), card_rows in sorted(by_card.items(), key=lambda kv: kv[0]):
        name = _gpu_filename(gpu, cap)
        if name in files:
            raise RuntimeError(f"freeze file name collision: {name} (cards {files[name]['gpu_name']!r} and {gpu!r})")
        digest, n = _write_rows(tmp, name, [_row_payload(r) for r in card_rows], {"gpu_name": gpu, "compute_cap": list(cap)}, "configs")
        files[name] = {"kind": "perf", "gpu_name": gpu, "compute_cap": list(cap), "rows": n, "sha256": digest}
    # Definitions are card-independent: the kernels the frozen rows and the kernel sets name, and
    # every kernel set. A kernel nothing names is not part of what the freeze pins.
    named = {r.kernel for r in kept} | {s.parent for s in kernel_sets} | {c for s in kernel_sets for c in s.children}
    definitions = (
        (KERNELS_NAME, "kernels", [{"identity": k.identity, "name": k.name, "wire": k.wire} for k in kernels if k.identity in named]),
        (
            KERNEL_SETS_NAME,
            "kernel_sets",
            [{"parent": s.parent, "decision": s.decision, "children": list(s.children)} for s in kernel_sets],
        ),
    )
    for name, kind, payloads in definitions:
        if payloads:
            digest, n = _write_rows(tmp, name, payloads, {}, kind)
            files[name] = {"kind": kind, "rows": n, "sha256": digest}

    top = hashlib.sha256()
    for name in sorted(files):
        top.update(files[name]["sha256"].encode())
    per_gpu = Counter(r.gpu for r in kept)
    manifest = {
        "kind": FREEZE_KIND,
        "freeze_ver": FREEZE_VER,
        "feat_ver": FEATURIZER_VERSION,
        "knob_ver": FEATURIZER_VERSION,
        "encoding_ver": FEATURIZER_VERSION,
        "repo_commit": _repo_commit(),
        "source_db": str(Path(db_path).resolve()),
        "policy_note": note,
        "counts": {
            "rows": len(kept),
            "ok": sum(1 for r in kept if r.status == "ok"),
            "bench_fail": sum(1 for r in kept if r.status == "bench_fail"),
            "per_gpu": {g: per_gpu[g] for g in sorted(per_gpu)},
            "kernels": files.get(KERNELS_NAME, {}).get("rows", 0),
            "kernel_sets": files.get(KERNEL_SETS_NAME, {}).get("rows", 0),
        },
        "files": files,
        "created_at": datetime.now(UTC).isoformat(),
        "sha256": top.hexdigest(),
    }
    (tmp / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    if out.exists():
        if not (out.is_dir() and (out / MANIFEST_NAME).exists()):
            shutil.rmtree(tmp)
            raise RuntimeError(f"{out} exists and is not a measurement freeze — refusing to replace it")
        shutil.rmtree(out)
    tmp.replace(out)
    return manifest


_FILE_KEYS = {"perf": "configs", "kernels": "kernels", "kernel_sets": "kernel_sets"}


def _read_rows(p: Path, name: str, info: dict, regen: str) -> tuple[dict, list[dict]]:
    """One freeze file, verified: its document and its rows, or a hard error naming what is wrong."""
    key = _FILE_KEYS.get(info.get("kind"))
    if key is None:
        raise RuntimeError(f"measurement freeze {p}: {name} has the unknown file kind {info.get('kind')!r} — {regen}")
    fpath = p / name
    if not fpath.exists():
        raise RuntimeError(f"measurement freeze {p} is missing {name} (listed in the manifest) — {regen}")
    text = fpath.read_text()
    if text.startswith("version https://git-lfs.github.com/spec/v1"):
        # The payload files are LFS-tracked. A clone or CI checkout without LFS leaves a
        # three-line pointer here, and a pointer is valid YAML — it parses to a string and the
        # first key lookup below fails as ``TypeError: string indices must be integers``, which
        # says nothing about the real problem. Name it instead.
        raise RuntimeError(
            f"measurement freeze {p}: {name} is a git-LFS pointer, not the data. Run `git lfs install && "
            f"git lfs pull`; in CI, check out with `lfs: true`."
        )
    doc = yaml.safe_load(text)
    if not isinstance(doc, dict) or key not in doc:
        raise RuntimeError(f"measurement freeze {p}: {name} is not a freeze payload document — {regen}")
    payloads = doc.get(key) or []
    file_digest = hashlib.sha256()
    for payload in payloads:
        file_digest.update(_row_line(payload))
    if file_digest.hexdigest() != info.get("sha256"):
        raise RuntimeError(f"measurement freeze {p} is corrupt: {name} row digest != manifest — {regen}")
    return doc, payloads


def load_freeze(path: Path | str) -> Freeze:
    """Parse + verify the freeze directory at ``path``: the manifest, the ``perf`` rows (each keyed by
    its file's card and sourced ``freeze:<digest>``), the ``kernel`` rows and the ``kernel_set`` rows.
    Hard ``RuntimeError`` — never a silent fallback — on any integrity failure."""
    p = Path(path)
    regen = "re-freeze with `emmy dataset freeze`"
    mpath = p / MANIFEST_NAME
    if not mpath.exists():
        raise RuntimeError(f"{p} is not a measurement freeze (no {MANIFEST_NAME}) — {regen}")
    try:
        manifest = json.loads(mpath.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(f"measurement freeze {p} has a corrupt manifest: {e} — {regen}") from e
    if not isinstance(manifest, dict) or manifest.get("kind") != FREEZE_KIND:
        raise RuntimeError(f"{p} is not a measurement freeze (manifest kind != {FREEZE_KIND!r}) — {regen}")
    if manifest.get("freeze_ver") != FREEZE_VER:
        raise RuntimeError(f"measurement freeze {p} has freeze_ver={manifest.get('freeze_ver')!r}, this code reads {FREEZE_VER} — {regen}")

    source = f"freeze:{manifest['sha256'][:12]}"
    frozen = Freeze(manifest, [], [], [])
    for name in sorted(manifest.get("files", {})):
        info = manifest["files"][name]
        doc, payloads = _read_rows(p, name, info, regen)
        if info["kind"] == "kernels":
            for payload in payloads:
                if not (
                    isinstance(payload.get("identity"), str)
                    and isinstance(payload.get("wire"), dict)
                    and isinstance(payload.get("name"), str)
                ):
                    raise RuntimeError(f"measurement freeze {p}: {name} row lacks identity/wire/name — {regen}")
                frozen.kernels.append(KernelRow(identity=payload["identity"], wire=payload["wire"], name=payload["name"]))
            continue
        if info["kind"] == "kernel_sets":
            for payload in payloads:
                if not (
                    isinstance(payload.get("parent"), str)
                    and isinstance(payload.get("decision"), dict)
                    and isinstance(payload.get("children"), list)
                ):
                    raise RuntimeError(f"measurement freeze {p}: {name} row lacks parent/decision/children — {regen}")
                frozen.kernel_sets.append(
                    KernelSetRow(parent=payload["parent"], decision=payload["decision"], children=tuple(payload["children"]))
                )
            continue
        gpu_name, (major, minor) = doc["gpu_name"], doc["compute_cap"]
        cc = major * 10 + minor
        for payload in payloads:
            if not (
                isinstance(payload.get("kernel"), str)
                and isinstance(payload.get("bindings"), dict)
                and isinstance(payload.get("knobs"), dict)
            ):
                raise RuntimeError(f"measurement freeze {p}: {name} row lacks kernel/bindings/knobs — {regen}")
            frozen.perf.append(
                PerfRow(
                    gpu=gpu_name,
                    cc=cc,
                    opt=int(payload["opt"]),
                    flags=str(payload["flags"]),
                    kernel=payload["kernel"],
                    bindings=payload["bindings"],
                    knobs=payload["knobs"],
                    backend="cuda",
                    status=payload["status"],
                    stats=PerfStats(**payload["stats"]),
                    measured_at=payload["measured_at"],
                    captured=bool(payload["captured"]),
                    error=payload["error"],
                    feat_ver=int(manifest["feat_ver"]),
                    source=source,
                )
            )
    return frozen
