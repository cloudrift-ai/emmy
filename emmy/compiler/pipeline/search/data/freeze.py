"""Measurement freeze — a DB instance's admitted rows as a golden file per card, re-lowered on import.

A tune DB is a live store (tunes and imports write into it), so a model fit or evaluated straight from it is
not reproducible. A *freeze* is the snapshot a reported number is computed over — identical wherever it is
read — and the training data the priors are fit on (``eval prior --dataset db``). It is written in the
golden file's shape: one document per card, its ``loops`` pool holding each kernel's definition, one config
per kernel set and binding, a realization per measured row — its schedule row, the regime it was measured
under (``FAST_MATH`` on or off, the two a golden records) and its median. Nothing the compiler computed is
stored: no identity a reader has to trust, no stamps spelled in one featurizer's vocabulary. ``emmy dataset
import`` re-lowers every kernel from its definition (``golden_import.import_goldens``, entering at the
lowering passes as the tuner runs a slice), so the dataset DB's identities and stamps are the current
compiler's, and a compiler change is a re-import, never a re-collection.

A kernel's definition is its ``kernel`` row's wire: the body it was formed from, which the lowering passes
take back to the kernel (``KernelRow.formed``). A piece carved from a twisted tree has no such body; its rows
are written under the nearest formed ancestor the routing table reaches — a routing entry per decision on the
path, the rows as receipts naming their kernel and listing the entries in ``kernel_set`` — the way a golden
records a kernel set, and nothing else is written that way.

What freezes (:func:`freeze_reason`): every ``ok`` CUDA row measured on a card the GPU registry knows, at the
deployable opt level, that passes the physical-plausibility predicates; the fast-math flag decides which of
the two precision regimes a row is in, and no other compiler flag is stored or gated on. A failed bench is
not a measurement; the tune DB keeps it. A row a compile imported from a golden file is the file's, and is
not frozen again.

Freezing the same rows twice yields the same bytes: rows sort by content and the golden dump is
deterministic. A file's identity is its bytes — ``emmy dataset import`` sources its rows as
``freeze:<sha256[:12]>`` of the file (:func:`freeze_source`), which is what ``commands.dataset.dataset_db``
checks the default dataset DB holds for every file of the checked-in freeze directory
(``config.freeze_path``, payload in git LFS).

Produced by ``emmy dataset freeze``.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
import shutil
from collections import Counter, defaultdict, deque
from pathlib import Path

from emmy.compiler.context import FAST_MATH_FLAG
from emmy.compiler.loop_wire import intern_wire
from emmy.compiler.pipeline.knob import METADATA_PREFIXES
from emmy.compiler.pipeline.search.db import KernelRow, PerfRow, SearchDB, knobs_json
from emmy.compiler.pipeline.search.features import DEPLOYABLE_OPT
from emmy.compiler.specialize import rehint_program

logger = logging.getLogger(__name__)

_LFS_POINTER = "version https://git-lfs.github.com/spec/v1"

#: The two precision regimes a golden records — fast math off, and on (the default since #868) — by the one
#: compiler flag that decides them, each mapped to the input pin a freeze row carries.
REGIME_PINS = {"": {"FAST_MATH": False}, FAST_MATH_FLAG: {"FAST_MATH": True}}


def regime_of(flags: str) -> str:
    """The regime a row's residual compiler flags put it in — a key of :data:`REGIME_PINS`. The fast-math flag is
    the one flag that is a regime; any other flag a row was compiled with is not, and is not what a freeze
    stores."""
    return FAST_MATH_FLAG if FAST_MATH_FLAG in flags.split() else ""


def freeze_reason(row: PerfRow) -> str | None:
    """Why ``row`` is excluded from a measurement freeze and from every measured-pool reader, or
    ``None`` to keep it.

    THE admission filter, and nothing else — keep every ``ok`` row measured at the DEPLOYABLE opt level, on a
    card the GPU registry knows, that passes the shared plausibility predicates.

    The opt-level gate is what keeps a freeze a fair yardstick. A freeze is the corpus a reported
    prior number is computed over, and a measurement taken under a non-deployable opt level
    answers a question nothing asks: nothing trains on it (``Prior.add_rows``) and no deploy
    reads it. Kept, it would put half a card's pools in a lane no one runs, so half the headline
    number would describe a regime that does not exist. Of the other compiler flags only fast math
    is a regime (:func:`regime_of`); the rest a freeze neither stores nor gates on."""
    from emmy import gpu  # noqa: PLC0415

    if gpu.by_name(row.gpu) is None:
        return "unknown card (not in the GPU registry)"
    if row.opt != DEPLOYABLE_OPT:
        return f"non-deployable regime (H_opt={row.opt:g})"
    if row.status != "ok":
        return f"{row.status}: not a measurement"
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
    (``bindings``) re-enter as one factor, their product; a static kernel binds nothing and a
    symbolic one with no sizes stored is read at the default hint. Ungateable rows also pass on:
    non-``ok`` status (a fail sentinel is not a measurement), no stamped shape, unknown card or
    unrecorded peak."""
    if row.status != "ok" or row.stats.median <= 0:
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
    if row.status != "ok":
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


def _gpu_filename(gpu_name: str, cap: tuple[int, int]) -> str:
    """The per-GPU YAML file name, mirroring the ``golden/`` convention —
    e.g. ``nvidia_geforce_rtx_4090_sm89.yaml``."""
    slug = re.sub(r"[^a-z0-9]+", "_", gpu_name.lower()).strip("_") or "unknown_gpu"
    return f"{slug}_sm{cap[0]}{cap[1]}.yaml"


def _path_to(kernel: str, kernels: dict[str, KernelRow], parents: dict[str, list[tuple[str, dict]]]) -> tuple | None:
    """The decisions from the nearest formed kernel down to ``kernel``, as ``(parent, arm)`` pairs: ``()`` when the
    kernel is formed itself, ``None`` when no formed kernel reaches it through the routing table."""
    if kernels[kernel].formed:
        return ()
    seen = {kernel}
    queue: deque[tuple[str, tuple]] = deque([(kernel, ())])
    while queue:
        child, path = queue.popleft()
        for parent, arm in parents.get(child, ()):
            if parent in seen:
                continue
            step = ((parent, arm), *path)
            if kernels[parent].formed:
                return step
            seen.add(parent)
            queue.append((parent, step))
    return None


def _schedule_row(row: PerfRow) -> dict[str, str]:
    return {str(k): str(v) for k, v in row.knobs.items() if not str(k).startswith(METADATA_PREFIXES)}


def _document(gpu_name: str, cap: tuple[int, int], rows: list[PerfRow], kernels: dict[str, KernelRow], parents, dropped: Counter) -> dict:
    """One card's golden document: a config per kernel set, size and regime, in content order."""
    sets: dict[tuple, list[PerfRow]] = defaultdict(list)
    paths: dict[tuple, tuple] = {}
    for row in rows:
        path = _path_to(row.kernel, kernels, parents)
        if path is None:
            dropped["no formed kernel reaches it"] += 1
            continue
        root = path[0][0] if path else row.kernel
        key = (root, tuple((parent, knobs_json(arm)) for parent, arm in path), knobs_json(row.bindings), regime_of(row.flags))
        sets[key].append(row)
        paths[key] = path
    loops: list[dict] = []
    configs: list[dict] = []
    for key in sorted(sets):
        root, _route, _bindings, regime = key
        path, members = paths[key], sorted(sets[key], key=lambda r: (r.kernel, knobs_json(r.knobs)))
        # The sizes the rows were benched at are the program's hints; a golden's ``bindings`` would make them static.
        # A parent axis a piece dropped keeps the stored hint: the piece's measurement does not depend on it.
        try:
            program = rehint_program(kernels[root].loop_ir, dict(members[0].bindings))
        except ValueError:
            dropped["sizes the program cannot bind"] += len(members)
            continue
        pins = REGIME_PINS[regime]
        realizations: list[dict] = []
        for step, (parent, arm) in enumerate(path):
            name = f"{kernels[parent].name}.{parent[:12]}.route{step}"
            identity = kernels[parent].structural_identity
            realizations.append({"name": name, "bindings": {}, "pins": pins, "knobs": dict(arm), "identity": identity})
        routes = [entry["name"] for entry in realizations]
        for n, row in enumerate(members):
            realizations.append(
                {
                    "name": f"{kernels[row.kernel].name}.{row.kernel[:12]}.{n}",
                    "bindings": {},
                    "pins": pins,
                    "knobs": _schedule_row(row),
                    "identity": kernels[row.kernel].structural_identity,
                    "measurements": {"emmy_us": row.stats.median},
                    **({"kernel_set": routes} if routes else {}),
                }
            )
        configs.append({"target": {"loop": intern_wire(loops, program)}, "realizations": realizations})
    return {"gpu_name": gpu_name, "compute_cap": list(cap), "loops": loops, "configs": configs}


def freeze_documents(db: SearchDB) -> tuple[dict[str, dict], Counter]:
    """The DB's admitted rows as one golden document per card, keyed by the card's file name, and the count
    of the rows left out, by reason."""
    kernels = {k.exact_identity: k for k in db.iter_kernels()}
    parents: dict[str, list[tuple[str, dict]]] = defaultdict(list)
    for decision in db.iter_routing():
        for child in decision.children:
            parents[child].append((decision.parent, decision.arm))
    dropped: Counter[str] = Counter()
    by_card: dict[tuple[str, tuple[int, int]], list[PerfRow]] = defaultdict(list)
    for row in db.iter_perf_rows(backend="cuda"):
        if row.source.startswith("golden:"):
            dropped["a golden file's row"] += 1
            continue
        reason = freeze_reason(row)
        if reason is not None:
            dropped[reason.split(":")[0]] += 1
            continue
        by_card[(row.gpu, divmod(row.cc, 10))].append(row)
    documents = {}
    for (gpu_name, cap), rows in sorted(by_card.items()):
        document = _document(gpu_name, cap, rows, kernels, parents, dropped)
        if document["configs"]:
            documents[_gpu_filename(gpu_name, cap)] = document
    return documents, dropped


def write_freeze(db_path: Path | str, out_dir: Path | str) -> dict[str, str]:
    """Read the DB instance at ``db_path`` read-only and atomically write the freeze DIRECTORY at
    ``out_dir``: one golden document per card. Returns each file's name and digest. Hard-errors when nothing
    survives the filter — a zero-row freeze means the wrong DB, not an empty dataset. An existing ``out_dir``
    is replaced only when it is itself a freeze (holds golden files and nothing else) — anything else is
    refused rather than deleted."""
    from emmy.compiler.pipeline.search.golden import dump_golden_file  # noqa: PLC0415

    db = SearchDB.open_readonly(db_path)
    try:
        documents, dropped = freeze_documents(db)
    finally:
        db.close()
    for reason, n in dropped.most_common():
        logger.info("[freeze] left out %d row(s): %s", n, reason)
    if not documents:
        raise RuntimeError(f"no freezable rows in {db_path} — wrong DB, or its rows are in another regime")
    out = Path(out_dir)
    if out.exists() and not (out.is_dir() and all(p.suffix == ".yaml" for p in out.iterdir())):
        raise RuntimeError(f"{out} exists and is not a measurement freeze — refusing to replace it")
    tmp = out.with_name(out.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    digests = {}
    for name, document in sorted(documents.items()):
        digests[name] = hashlib.sha256(dump_golden_file(document, tmp / name).read_bytes()).hexdigest()
    if out.exists():
        shutil.rmtree(out)
    tmp.replace(out)
    return digests


def freeze_source(path: Path | str) -> str:
    """The ``source`` an import files a freeze file's rows under: the file's own bytes, digested."""
    return f"freeze:{hashlib.sha256(Path(path).read_bytes()).hexdigest()[:12]}"


def is_lfs_pointer(path: Path | str) -> bool:
    """Whether the payload at ``path`` is a git-LFS pointer rather than the data — what a clone or CI checkout
    without LFS leaves, three lines that are valid YAML and parse to a string, so the first key lookup would
    fail with a type error that says nothing about the real problem."""
    with Path(path).open("r") as fh:
        return fh.read(len(_LFS_POINTER)) == _LFS_POINTER
