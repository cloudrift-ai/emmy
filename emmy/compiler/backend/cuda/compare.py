"""The comparison a bench worker job runs: one program timed as eager torch, ``torch.compile`` and Emmy.

:func:`bench_full_model_real` times a traced module against its compiled program on the module's own inputs;
:func:`bench_lowered_vs_torch` times a lowered graph against the Torch twin built from its frontend graph, on
random inputs drawn here. Both interleave the backends inside CUDA-graph-captured windows, so one table never
mixes timing semantics. Nothing here prints: the caller owns the table.
"""

from __future__ import annotations

import logging

from emmy import config
from emmy.compiler.backend.correctness import (
    QUANTIZED_FORMATS,
    check_accuracy,
    comparison_outputs,
    eager_outputs_by_name,
    quantize_nodes,
    quantized_twin,
    reference_precision,
    strict_eager_proof,
    to_cuda_kwargs,
    to_cuda_tensor,
)
from emmy.compiler.loader.binder import flatten_tensors

logger = logging.getLogger(__name__)


def _reset_persisting_l2_cache() -> None:
    """Drop every persisting-L2 carveout on the current CUDA context.

    Wraps ``cudaCtxResetPersistingL2Cache`` (CUDA 11+). cuBLAS, cuDNN,
    and other libraries can install ``cudaAccessPolicyWindow`` hints
    that pin specific gmem regions in L2 across kernel launches; once
    set, the carveout persists until the context tears down (it is NOT
    released by ``torch.cuda.empty_cache`` or stream sync). That eats
    L2 bandwidth from later cp.async-staged kernels — the article
    matmul drops from 1660 µs to 330 µs after the reset. Best-effort:
    silently skipped if the runtime symbol isn't resolvable (CUDA < 11,
    non-Linux, etc.); a non-zero return is logged at DEBUG so the
    bench path doesn't fail loud on driver quirks.

    Loads ``libcudart`` from the already-mapped image so the call
    targets the SAME runtime torch is using (the system
    ``libcudart.so`` may belong to a different CUDA install and would
    operate on a different driver context — its reset returns success
    but doesn't touch our context's L2 carveout). Walks
    ``/proc/self/maps`` for the first mapped ``libcudart.so.*`` path.
    """
    import ctypes

    from emmy.compiler.backend.cuda.cache import cuda_runtime

    try:
        cudart = cuda_runtime()
        cudart.cudaCtxResetPersistingL2Cache.restype = ctypes.c_int
        cudart.cudaCtxResetPersistingL2Cache.argtypes = []
        err = cudart.cudaCtxResetPersistingL2Cache()
        if err != 0:
            logger.debug("cudaCtxResetPersistingL2Cache returned %d (continuing)", err)
    except (OSError, AttributeError) as e:
        logger.debug("cudaCtxResetPersistingL2Cache unavailable: %s", e)


#: The standard deviation (std) of each synthesized tensor: :func:`_random_source_values` draws unquantized weights
#: at 0.02, :func:`_random_input_values` draws activations at 1. The key is the suffix of the checkpoint key of the
#: NVFP4 per-tensor scale that calibrates that tensor.
_SYNTH_WEIGHT_STD = 0.02


_SYNTH_STD = {"_scale_2": _SYNTH_WEIGHT_STD, "input_scale": 1.0}


#: How many std from zero a synthesized tensor's calibrated amax (largest absolute value) sits. Real activations
#: carry outliers far past their std, and inside a program an activation outgrows the unit std of the inputs: a
#: gated MLP multiplies two projections. The headroom keeps the e4m3 block scales of such an activation well under
#: the e4m3 maximum of 448.
_SYNTH_AMAX_STDS = 64.0


def _random_source_values(rng, shape, dtype, *, name: str | None = None):
    """Return nontrivial deterministic values in a constant's declared storage dtype.

    Packed 4-bit pairs are uniform random bytes, so every code appears in both halves of a byte, as in a real
    checkpoint. A source whose ``name`` (the checkpoint key) ends in a scale leaf is positive, like every calibrated
    scale. NVFP4 scales follow how calibration sets them for a tensor of the synthesized std: a per-tensor scale
    (``*_scale_2``, ``input_scale``) is ``amax / (6 * 448)``, 6 and 448 being the largest e2m1 and e4m3 values.
    An 8-bit float scale (in checkpoints, only NVFP4 weight block scales are one) is ``block amax / (6 * per-tensor
    scale)``, so that each 16-value weight block has an amax of one to four std. A code times its block scale times
    the per-tensor scale then reproduces a weight of the 0.02 std the unquantized sources have, and an activation
    quantized with the drawn ``input_scale`` gets block scales well under 448. A block-FP8 weight's scale
    (``*_scale_inv``) is its block's amax over 448, an amax of one to four std, so that the weight it scales has
    the same std; larger scales overflow an FP16 trunk's projections. Any other scale gets a
    log-uniform draw from 1e-4 to 1e-1. A scale drawn from a normal distribution can be negative or near zero, and its
    reciprocal then overflows the 8-bit block scales computed from it."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import F4_VALUES, decode_f8  # noqa: PLC0415
    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415

    canonical = get_dtype(dtype or "f32").name
    leaf = name.rsplit(".", 1)[-1] if name is not None else ""
    e4m3_max, e2m1_max = float(np.nanmax(decode_f8(np.arange(256, dtype=np.uint8), "f8e4m3"))), max(F4_VALUES)
    std = next((v for suffix, v in _SYNTH_STD.items() if leaf.endswith(suffix)), None)
    if canonical == "f4e2m1x2":
        return rng.integers(0, 256, shape, dtype=np.uint8)
    if canonical in {"f8e4m3", "f8e5m2"}:
        if "scale" in leaf:
            codes = np.arange(256, dtype=np.uint8)
            values = decode_f8(codes, canonical)
            lo, hi = e4m3_max / _SYNTH_AMAX_STDS, 4 * e4m3_max / _SYNTH_AMAX_STDS
            return rng.choice(codes[(values >= lo) & (values <= hi)], size=shape)
        bits = rng.integers(0, 256, shape, dtype=np.uint8)
        bits[~np.isfinite(decode_f8(bits, canonical))] = np.uint8(0)
        return bits
    if std is not None:
        return np.full(shape, _SYNTH_AMAX_STDS * std / (e2m1_max * e4m3_max), dtype=np.float32)
    if leaf.endswith("_scale_inv"):
        return (rng.uniform(1, 4, shape) * _SYNTH_WEIGHT_STD / e4m3_max).astype(np.float32)
    if "scale" in leaf:
        return np.exp(rng.uniform(np.log(1e-4), np.log(1e-1), shape)).astype(np.float32)
    return rng.standard_normal(shape, dtype=np.float32) * 0.02


def _random_input_values(rng, shape, dtype, *, name: str | None = None):
    """Return random runtime values, preserving bit carriers and finite E8M0 scales."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415
    from emmy.compiler.trace.huggingface import EXPERT_WEIGHT_INPUTS  # noqa: PLC0415
    from emmy.serving.lora import WEIGHT_INPUTS  # noqa: PLC0415

    canonical = get_dtype(dtype).name
    if name == "lora_mask":
        return rng.integers(0, 2, shape, dtype=np.uint8).astype(np.float32)
    if name in WEIGHT_INPUTS:
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    if canonical in QUANTIZED_FORMATS:
        return _random_source_values(rng, shape, dtype, name=name)
    if canonical == "u8" and name is not None and "scale" in name:
        # MXFP4 boundary scales are E8M0 bytes with bias 127. Arbitrary u8 codes span
        # exponents that overflow a two-linear random reproducer before correctness can run.
        return rng.integers(120, 123, shape, dtype=np.uint8)
    if name in EXPERT_WEIGHT_INPUTS and canonical in {"f16", "bf16", "f32"}:
        # A routed expert's weights are program inputs. At unit scale its two FP16 projections overflow, so the
        # reference holds infinities and no correctness check or isolated re-bench can run.
        return rng.standard_normal(shape, dtype=np.float32) * 0.02
    return rng.standard_normal(shape, dtype=np.float32)


def _block_scale_inputs(dtypes: dict) -> set[str]:
    """The inputs, among ``dtypes`` (each input's dtype by name), that are the block scale of an 8-bit float weight: a
    ``<weight>_scale`` beside its ``<weight>_bits``. A stored kernel reads its constants as inputs, and these are drawn
    as a checkpoint's are (:func:`_random_source_values`). Drawn at unit std the weight is ten thousand times a real
    one: a gated MLP's product of two such projections is infinity or zero in FP16, but for the few elements whose
    gate is near zero, and on those two orders of the same FP32 sum disagree past any tolerance."""
    from emmy.compiler.dtype import get as get_dtype  # noqa: PLC0415

    weights = {
        name.removesuffix("_bits")
        for name, dtype in dtypes.items()
        if name.endswith("_bits") and get_dtype(dtype).name in {"f8e4m3", "f8e5m2"}
    }
    return {f"{weight}_scale" for weight in weights} & dtypes.keys()


def _chunk_position(graph, static) -> int | None:
    """The ``position`` a bench gives a native step's program, whose rows start there in its cache (``q``, ``keys``):
    the window's last chunk, the one its attention is priced for. Drawn at random the rows sit at the cache's start,
    where a causal chunk stops after a few keys, and its time says nothing of the window. ``None`` for any other
    program; ``static`` gives a shape's sizes at the bench's bindings."""
    if not {"q", "keys", "position"} <= graph.nodes.keys():
        return None
    return static(graph.nodes["keys"].output.shape)[0] - static(graph.nodes["q"].output.shape)[0]


async def bench_lowered_vs_torch(
    frontend,
    lowered,
    backend,
    *,
    seed,
    do_bench,
    warmup,
    iters,
    bench_backends,
    capture_graphs=True,
    ref_out=None,
    ref_us_out=None,
    sym_env_out=None,
    strict_accuracy=False,
    return_reference=False,
):
    """Run + (optionally) benchmark a lowered graph against its torch reference on
    shared random inputs. The bench primitive behind ``run --ir`` and a golden target's ``run --bench``.

    ``frontend`` is the pristine frontend-dialect snapshot (must be
    ``torch_ref``-runnable); pass ``None`` for a non-frontend / unsupported graph to
    bench emmy-only. One random source is drawn per distinct constant
    ``source_path`` and each side replays its own ``load_ops`` (so a weight lowering
    transposed + renamed stays the same underlying tensor on both sides). The lowered
    graph runs once for a non-fatal accuracy check vs the torch eager reference; then,
    when ``do_bench``, the selected backends are timed — interleaved when a torch ref
    exists, else emmy-only, both with the requested ``warmup``/``iters``.

    ``capture_graphs`` (default on — this function's callers are the per-kernel
    reproducer paths, where the torch side replays the frontend graph op-by-op and
    would otherwise be dispatch-bound) wraps every timed backend in a CUDA graph so
    the event windows measure pure GPU time. All-or-nothing: if any torch backend
    or the emmy launch loop fails to capture, the whole bench retries
    uncaptured (warning logged) so one table never mixes timing semantics. The
    accuracy check always runs uncaptured, before any capture.

    Returns ``(results, bench, torch_available, captured, accuracy_error)``: ``results``
    is the ``{backend: latency_us}`` dict (``None`` when ``do_bench`` is False), ``bench``
    the emmy ``BenchmarkResult`` (``None`` when ``do_bench`` is False),
    ``torch_available`` whether an eager/torch.compile reference was built, ``captured``
    whether the timings came from graph-captured (pure-GPU) windows, and
    ``accuracy_error`` the accuracy verdict or reference exception (``None`` = passed or no frontend;
    also logged here — returned so a worker-side run can ship it back to the parent, whose
    child logs are invisible). With ``return_reference``, appends the strict correctness
    proof and ``(input_data, eager_outputs_by_name)`` for same-input pinned replay. When
    ``ref_out`` is a list, that reference is also stored before repeated timing so a later
    watchdog cannot discard completed same-input evidence; reference-free Loop IR instead
    stores its greedy Emmy output. ``ref_us_out`` receives the initial Emmy execution timing.
    Does no printing / dumping — callers own that."""
    import numpy as np
    import torch

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.loader.binder import bind_constants

    rng = np.random.default_rng(seed)

    # Symbolic reproducers bench at each dim's hint (``DEFAULT_SEQ_HINT`` for a
    # bare seq axis) — the same size the backend resolves a symbolic graph to
    # when no inputs are supplied — so the random inputs built here get concrete
    # hint-sized shapes. Atomic dims carry their own hint; composite expressions
    # (e.g. ``S * 2``) eval over the collected env.
    from emmy.compiler.dim import Dim

    sym_env = collect_sym_env(([frontend] if frontend is not None else []) + [lowered])
    if sym_env_out is not None:
        sym_env_out.append(dict(sym_env))

    def _static(shape):
        return tuple((d.as_static() if d.is_static else int(d.expression.eval(sym_env))) if isinstance(d, Dim) else int(d) for d in shape)

    # One random source array per distinct constant ``source_path`` — shared by
    # emmy (lowered graph) and the torch ref (frontend graph). ``bind_constants``
    # replays each constant's ``load_ops`` on it, so a weight that lowering transposed +
    # renamed (linear: out×in → in×out) stays the same underlying tensor on both sides.
    sources: dict[str, np.ndarray] = {}
    for gph in ([frontend] if frontend is not None else []) + [lowered]:
        for node in gph.nodes.values():
            op = node.op
            if not (isinstance(op, ConstantOp) and op.value is None):
                continue
            if op.source_path and op.source_path not in sources:
                shp = _static(op.source_shape or node.output.shape)
                sources[op.source_path] = _random_source_values(rng, shp, op.source_dtype, name=op.source_path)
            # A merged (source_parts) constant draws one random source PER PART, keyed by the
            # part path — the same tensors the pre-merge frontend reference binds its separate
            # weights from, so emmy's concat and the torch ref stay numerically aligned.
            for path, shp in op.source_parts:
                if path not in sources:
                    sources[path] = _random_source_values(rng, _static(shp), op.source_dtype, name=path)

    input_data: dict[str, object] = {}
    input_tensors: dict[str, object] = {}
    position = _chunk_position(lowered, _static)
    block_scales = _block_scale_inputs({nid: node.output.dtype for nid, node in lowered.nodes.items() if isinstance(node.op, InputOp)})
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, InputOp):
            # The checkpoint key's leaf is what names a block scale to the draw of a source.
            draw, name = (_random_source_values, "weight_scale_inv") if nid in block_scales else (_random_input_values, nid)
            arr = draw(rng, _static(node.output.shape), node.output.dtype, name=name)
            if nid == "position" and position is not None:
                arr = np.full(arr.shape, position)
            # Keep the ndarray shape (no flatten) — a symbolic graph's launch
            # reads the runtime seq_len off the input array's shape.
            input_data[nid] = arr
            # Reference-free embedded Loop targets may intentionally expose a
            # price-probe boundary tensor whose logical broadcast shape is far
            # larger than device memory. The Emmy backend already owns the one
            # device materialization needed to execute that target; a second
            # Torch copy is useful only when a frontend reference exists.
            if frontend is not None:
                input_tensors[nid] = to_cuda_tensor(arr, node.output.dtype)
        elif isinstance(node.op, ConstantOp) and node.op.value is not None:
            input_data[nid] = [float(node.op.value)]

    input_data.update(bind_constants(lowered, sources))
    if frontend is not None:
        for fid, arr in bind_constants(frontend, sources).items():
            input_tensors[fid] = to_cuda_tensor(arr, frontend.nodes[fid].output.dtype)

    # Fallback for any value-None constant with no source_path (synthetic).
    for nid, node in lowered.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.value is None and nid not in input_data:
            arr = rng.standard_normal(_static(node.output.shape), dtype=np.float32) * 0.02
            input_data[nid] = arr.flatten().tolist()

    quantized = quantize_nodes(frontend, lowered) if strict_accuracy and frontend is not None else ()
    result, _ = backend.run(lowered, input_data=input_data, taps=quantized)
    result_outputs = comparison_outputs(result.outputs, lowered)
    if frontend is None and ref_out is not None:
        ref_out.append((input_data, result_outputs))
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    if ref_us_out is not None and not cold_cache():
        ref_us_out.append(result.time_ms * 1000)
    for nid, arr in result_outputs.items():
        finite = np.isfinite(arr).all()
        logger.info("Output %s: shape=%s finite=%s mean=%.4f", nid, arr.shape, bool(finite), float(arr.mean()))

    # Build the torch reference from the frontend snapshot, fed the same inputs.
    torch_fn = torch_inputs = accuracy_error = correctness = reference = None
    if frontend is not None:
        try:
            torch_fn, torch_inputs = torch_ref.build_callable(frontend, input_tensors)
            exact = None
            if strict_accuracy:
                correctness, eager_values, exact_values = strict_eager_proof(
                    frontend, input_tensors, result_outputs, lowered, result.taps if quantized else {}
                )
                if correctness["status"] != "pass":
                    accuracy_error = f"strict eager correctness failed: {correctness.get('error', 'tolerance exceeded')}"
                if ref_out is not None or return_reference:
                    exact = exact_values()  # the pinned rows weigh their own strict failures against it
            else:
                with torch.no_grad(), reference_precision():
                    eager_values = eager_outputs_by_name(result_outputs, torch_fn(*torch_inputs), lowered)
                accuracy_error = check_accuracy(result_outputs, eager_values)
            reference = (input_data, eager_values) if exact is None else (input_data, eager_values, exact)
            if ref_out is not None:
                ref_out.append(reference)
            if accuracy_error is not None:
                qualifier = "fatal when strict correctness is requested" if strict_accuracy else "non-fatal (random-input reproducer)"
                logger.warning("%s — %s; benching anyway", accuracy_error, qualifier)
        except Exception as exc:  # noqa: BLE001 — torch ref is best-effort
            reference_kind = "strict eager correctness" if strict_accuracy else "torch reference"
            accuracy_error = f"{reference_kind} unavailable: {type(exc).__name__}: {exc}"
            logger.warning("%s — skipping vs-torch comparison", accuracy_error)
            torch_fn = None

    if strict_accuracy and frontend is None:
        accuracy_error = "strict eager correctness unavailable: frontend IR is not runnable"
    torch_available = torch_fn is not None
    if not do_bench:
        base = (None, None, torch_available, False, accuracy_error)
        return (*base, correctness, reference) if return_reference else base

    if torch_available:
        backends = resolve_backends(bench_backends)

        async def timed(torch_fns):
            if capture_graphs:
                return await _bench_interleaved_captured(
                    torch_fn, torch_inputs, {}, backend, lowered, warmup, iters, torch_fns=torch_fns, input_data=input_data
                )
            results, bench = await _bench_interleaved(
                torch_fn,
                torch_inputs,
                {},
                backend,
                lowered,
                warmup,
                iters,
                torch_fns=torch_fns,
                capture_graphs=False,
                input_data=input_data,
            )
            return results, bench, False

        # A quantized twin's codes are what two correct evaluations land one apart on, so torch.compile is timed
        # unchecked, like Emmy's row; and its quantize can trip a device-side assert that poisons the context, so it
        # is built and timed in a window of its own after Emmy's, whose number then survives it.
        deferred = "tcompile" in backends and quantized_twin(frontend)
        torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends=backends - ({"tcompile"} if deferred else set()))
        results, bench, captured = await timed(torch_fns)
        if deferred:
            try:
                torch_fns = _build_torch_fns(torch_fn, torch_inputs, {}, warmup, backends={"tcompile"}, check=False)
                torch.cuda.synchronize()  # a device-side assert in the twin surfaces here, as torch.compile's failure
                compiled = torch_fns["torch.compile"]
                results["torch.compile"] = (
                    compiled if isinstance(compiled, BackendFailure) else (await timed(torch_fns))[0]["torch.compile"]
                )
            except RuntimeError as exc:
                results["torch.compile"] = BackendFailure(f"{type(exc).__name__}: {exc}")
        base = (results, bench, True, captured, accuracy_error)
        return (*base, correctness, reference) if return_reference else base
    # Emmy-only: a capture failure falls back inside ``benchmark_program``
    # (warned + reported via ``bench.captured``) — nothing to de-mix.
    bench = await backend.benchmark_async(lowered, warmup=warmup, num_iters=iters, capture_graphs=capture_graphs, input_data=input_data)
    base = ({"Emmy": bench.time_ms * 1000}, bench, False, bench.captured, accuracy_error)
    return (*base, correctness, reference) if return_reference else base


async def bench_full_model_real(module, args_t, kwargs, lowered, backend, *, warmup, iters, bench_backends, input_data=None):
    """End-to-end full-model bench against the **real torch module** — eager /
    ``torch.compile`` / Emmy — using the all-or-nothing CUDA-graph-captured
    interleaved bench (real modules occasionally resist capture; those fall back
    to uncaptured wall timing, flagged in ``captured``). The module + its
    trace-time inputs come from ``load_or_trace``'s bundle; for a symbolic
    graph the torch closures run on hint-tiled inputs (``_hint_sized_inputs``)
    so both sides bench the hint shape. ``input_data`` (the module's own inputs bound to the
    emmy program, :func:`bind_inputs`) times emmy on the values torch reads; a symbolic graph
    keeps the synthetic fill, since its torch side runs tiled hint-sized copies instead.
    Returns ``(results, bench, captured)``."""
    import torch

    cuda_module = module.to("cuda")
    cuda_args = tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args_t)
    cuda_kwargs = to_cuda_kwargs(kwargs)
    cuda_args, cuda_kwargs, sym_env = _hint_sized_inputs(lowered, cuda_args, cuda_kwargs)
    input_data = None if sym_env else input_data
    backends = resolve_backends(bench_backends)
    torch_fns = _build_torch_fns(cuda_module, cuda_args, cuda_kwargs, warmup, backends=backends)
    if not torch_fns:
        # Emmy-only bench (``--bench-backends emmy``): any prior torch activity in this
        # process (the accuracy probe's eager forward, the warmups above) left
        # cuBLAS-installed ``cudaAccessPolicyWindow`` carveouts pinned in L2, which steal
        # L2 bandwidth from cp.async-staged kernels (~4x on the article tile — 1660 µs
        # with the window live, 330 µs after the reset). With torch backends in the table
        # the state must stay alive and the comparison absorbs it intrinsically; emmy-only,
        # reset it. (Caller-held module refs keep allocator blocks alive — the persisting-L2
        # window is the dominant term and resets regardless.)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        _reset_persisting_l2_cache()
    return await _bench_interleaved_captured(
        cuda_module, cuda_args, cuda_kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, input_data=input_data
    )


def collect_sym_env(graphs) -> dict[str, int]:
    """Every symbolic dim var appearing in ``graphs`` bound to the size the backend resolves it to
    when benching without supplied inputs (:func:`symbolic_bindings`, the tune DB's rule too)."""
    from emmy.compiler.wire import symbolic_bindings

    return symbolic_bindings(node.output for gph in graphs for node in gph.nodes.values())


def _tile_to(tensor, axis: int, size: int):
    """Resize ``tensor`` along ``axis`` to ``size`` by repeating its values
    (ceil) and slicing — keeps every element drawn from the original (token
    ids stay in-vocab, masks keep their value range), unlike fresh randoms."""
    old = tensor.shape[axis]
    if old == size:
        return tensor
    reps = [1] * tensor.dim()
    reps[axis] = -(-size // old)
    return tensor.repeat(*reps).narrow(axis, 0, size)


def _map_tensors(value, fn):
    """Rebuild ``value``'s nested list/tuple structure with ``fn`` applied to
    each tensor leaf, in :func:`flatten_tensors` order."""
    import torch

    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(_map_tensors(v, fn) for v in value)
    return value


def _hint_sized_inputs(lowered, example_args, example_kwargs):
    """Tile a symbolic graph's example inputs out to its ``Dim`` hints.

    The emmy side of the full-model bench runs a symbolic graph at each
    dim's hint (``backend.benchmark`` builds hint-sized synthetic inputs when
    none are supplied), while the torch closures close over the trace-time
    example tensors — different shapes, so the table would compare e.g. a
    seq-512 emmy program against seq-32 eager. Grow every symbolic input
    axis to its hint (positional ``graph.inputs`` ↔ flattened example tensors
    pairing, same as :func:`bind_inputs`) so both sides run the hint shape.
    Values are tiled repeats of the trace inputs — valid for latency, which is
    all this table compares (accuracy runs on the original trace inputs).

    Returns ``(args, kwargs, sym_env)`` — the inputs unchanged and ``{}`` for
    a static graph.
    """
    sym_env = collect_sym_env([lowered])
    if not sym_env:
        return example_args, example_kwargs, {}
    from emmy.compiler.dim import Dim

    flat: list = []
    for v in example_args:
        flat.extend(flatten_tensors(v))
    for v in example_kwargs.values():
        flat.extend(flatten_tensors(v))
    input_ids = list(lowered.inputs)
    if len(input_ids) != len(flat):
        logger.warning("Input arity mismatch (graph %d vs example %d) — benching torch on trace-sized inputs", len(input_ids), len(flat))
        return example_args, example_kwargs, sym_env
    resized = []
    for nid, tensor in zip(input_ids, flat, strict=True):
        for axis, d in enumerate(lowered.buffer(nid).shape):
            if isinstance(d, Dim) and not d.is_static:
                tensor = _tile_to(tensor, axis, int(d.expression.eval(sym_env)))
        resized.append(tensor)
    it = iter(resized)
    args = tuple(_map_tensors(v, lambda _: next(it)) for v in example_args)
    kwargs = {k: _map_tensors(v, lambda _: next(it)) for k, v in example_kwargs.items()}
    return args, kwargs, sym_env


_BACKEND_ALIASES = {
    "eager": "eager",
    "emmy": "emmy",
    "tcompile": "tcompile",
    "torch.compile": "tcompile",
    "compile": "tcompile",
}


def resolve_backends(cli_value: str | None) -> set[str]:
    """Pick which bench backends to time. Precedence:

    1. ``--bench-backends`` CLI arg (comma-separated).
    2. ``EMMY_BENCH_BACKENDS`` env var (same syntax).
    3. Default ``eager,emmy`` — torch.compile is excluded so the
       per-case wall time isn't dominated by a ~0.8 s Inductor JIT
       that most users don't need on every run.

    ``emmy`` is always included even if omitted (the kernel under
    test is the point of the bench). Returns the canonical backend
    keys ``{"eager", "tcompile", "emmy"}``.
    """
    from emmy.compiler.pipeline.search.space import cold_cache  # noqa: PLC0415

    raw = config.bench_backends_raw(cli_value, default="emmy" if cold_cache() else "eager,emmy")
    selected: set[str] = {"emmy"}
    for tok in raw.split(","):
        tok = tok.strip().lower()
        if not tok:
            continue
        canonical = _BACKEND_ALIASES.get(tok)
        if canonical is None:
            raise ValueError(f"unknown bench backend {tok!r} — choose from {sorted(set(_BACKEND_ALIASES.values()))}")
        selected.add(canonical)
    if cold_cache() and selected != {"emmy"}:
        raise ValueError(
            "cold-cache mode times individual Emmy kernels; Torch forwards retain internal cache reuse. Use --bench-backends emmy."
        )
    return selected


class BackendFailure(str):
    """A requested torch backend that could not be built, in the closure dict and then in the
    results dict under its name, where its message stands where its latency would."""


def _build_torch_fns(module, args, kwargs, warmup, *, backends: set[str], check: bool = True):
    """Pre-build the per-backend ``torch_fns`` dict, including the
    ``torch.compile`` JIT step when requested: the JIT (mostly Inductor CPU work plus a few warmup launches)
    runs here, before the measurement iterations.

    ``check`` holds the compiled output to eager on the dtype-scaled verdict; a quantized twin
    (:func:`quantized_twin`) is timed without it, as Emmy's row is.

    Returns only the torch-side closures; emmy's bench loop is
    driven separately by ``backend.benchmark`` in ``_bench_interleaved``.
    """
    import torch

    torch_fns: dict[str, callable] = {}
    if "eager" in backends:
        torch_fns["Eager PyTorch"] = lambda: module(*args, **kwargs)
    if "tcompile" in backends:
        try:
            # Persistent/repeated bench workers compile a fresh closure of
            # the SAME ``torch_ref`` ``fn`` code object per row; dynamo's
            # recompile limit is keyed per code object, so after ~8 rows it
            # silently stops compiling and the tcompile column quietly measures
            # eager (observed: 8 µs rows reading 55-80 µs). Reset dynamo so
            # every bench compiles fresh.
            import torch._dynamo  # noqa: PLC0415

            torch._dynamo.reset()
            # The benchmark owns one outer CUDA graph for every backend. Letting
            # Inductor create its own graph here makes that capture nested and
            # invalidates it on recent PyTorch releases.
            compiled_torch_module = torch.compile(module, fullgraph=True, mode="max-autotune-no-cudagraphs")
            for _ in range(warmup + 5):
                with torch.no_grad():
                    compiled_torch_module(*args, **kwargs)
            if check:
                with torch.no_grad():
                    eager_output = module(*args, **kwargs)
                    compiled_output = compiled_torch_module(*args, **kwargs)
                # The same dtype-scaled verdict the Emmy output gets: a flat 1e-3 rejected Inductor's own
                # FP16 GEMM against cuBLAS on three of the six Gemma 4 projections on an RTX 4090.
                outputs = compiled_output if isinstance(compiled_output, (tuple, list)) else (compiled_output,)
                verdict = check_accuracy({f"out{i}": t.detach().cpu().double().numpy() for i, t in enumerate(outputs)}, eager_output)
                if verdict:
                    raise ValueError(verdict)
            torch_fns["torch.compile"] = lambda: compiled_torch_module(*args, **kwargs)
        except Exception as e:  # noqa: BLE001
            # The column is reported failed, never silently dropped: this runs in the bench worker,
            # whose log the parent only shows on a crash.
            logger.warning("torch.compile failed: %s", e)
            torch_fns["torch.compile"] = BackendFailure(f"{type(e).__name__}: {e}")
    return torch_fns


def _capture_torch_fn(fn, batch_size=1):
    """Capture a complete batch of fixed-input forwards; one replay submits the whole window."""
    import torch

    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side), torch.no_grad():
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(side)
    g = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(g):
        for _ in range(batch_size):
            fn()
    return g.replay


def _capture_torch_fns(torch_fns: dict, batch_size=1) -> dict | None:
    """Capture every backend closure into a CUDA graph; ``None`` if ANY fails.

    All-or-nothing by design: a comparison table must not mix captured
    (pure-GPU) and uncaptured (dispatch-inclusive) timings, so a single
    backend resisting capture (e.g. torch.compile guard machinery) falls the
    whole invocation back to uncaptured timing."""
    captured: dict = {}
    for name, fn in torch_fns.items():
        if isinstance(fn, BackendFailure):
            captured[name] = fn
            continue
        try:
            captured[name] = _capture_torch_fn(fn, batch_size)
        except Exception as exc:  # noqa: BLE001 — capture is best-effort, fallback covers
            logger.warning("CUDA graph capture failed for %s (%s) — benching all backends uncaptured", name, exc)
            return None
    return captured


async def _bench_interleaved_captured(module, args, kwargs, backend, lowered, warmup, iters, *, torch_fns, input_data=None):
    """All-or-nothing CUDA-graph-captured interleaved bench.

    A capture failure at any batch size re-runs every backend uncaptured. Other benchmark
    failures propagate. Returns ``(results, bench, captured)`` on the same live inputs."""
    from emmy.compiler.backend.cuda.program import GraphCaptureError

    try:
        results, bench = await _bench_interleaved(
            module, args, kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=True, input_data=input_data
        )
        if not torch_fns:
            return results, bench, bench.captured  # cold Emmy-only timing deliberately has no whole-program window
        if bench.captured and (bench.num_launches <= 1 or bench.e2e_ms is not None):
            return results, bench, True
    except GraphCaptureError:
        pass
    logger.warning("CUDA graph capture failed — re-benching all backends uncaptured")
    results, bench = await _bench_interleaved(
        module, args, kwargs, backend, lowered, warmup, iters, torch_fns=torch_fns, capture_graphs=False, input_data=input_data
    )
    return results, bench, False


async def _bench_interleaved(
    module, args, kwargs, backend, compiled_graph, warmup, iters, *, torch_fns, capture_graphs=False, input_data=None
):
    """Time the selected backends by alternating one iter of each per
    loop step. All backends see the same warm GPU state across the
    measurement window — same clocks, same caches, same thermal drift
    — instead of running in sequential phases that each get a
    different steady state.

    Driven by ``backend.benchmark_async(on_iter=...)``: emmy is the
    backbone, ``on_iter`` runs each torch closure and records its
    cuda events, and the same call returns per-launch emmy
    timings — so the kernel-stats breakdown shares the same warm
    state as the comparison numbers.

    Per-iter ``torch.cuda.Event``s queue on the (legacy) default
    stream, and the emmy program's launches are issued on that same stream
    (``CompiledProgram.on_stream``), so events see all preceding work.

    ``torch_fns`` is the pre-built backend closure dict from
    :func:`_build_torch_fns`, built before this measurement loop so the ``torch.compile`` JIT stays out of it.

    ``capture_graphs`` captures each calibrated batch before its first event window on both
    sides. :func:`_bench_interleaved_captured` owns the all-or-nothing fallback.

    ``input_data`` binds the emmy program to the same values the torch closures read. Without
    it emmy times a synthetic fill while torch times real inputs, and a kernel whose cost
    depends on its values (a division's slow path, an exp near overflow) reads differently.
    """
    import torch

    from emmy.compiler.backend.cuda.program import GraphCaptureError

    # Each entry: (start_event, stop_event, batch_size_used). The callback supplies the largest
    # per-launch batch; whole-program Emmy windows calibrate their batch from the summed latency.
    # Each captured window submits one graph, with elapsed time normalized by its own batch size.
    failed = {name: fn for name, fn in torch_fns.items() if isinstance(fn, BackendFailure)}
    torch_fns = {name: fn for name, fn in torch_fns.items() if name not in failed}
    torch_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event, int]]] = {name: [] for name in torch_fns}
    captures = {}

    def on_iter(batch_size: int = 1) -> None:
        if capture_graphs and batch_size not in captures:
            captures[batch_size] = _capture_torch_fns(torch_fns, batch_size)
            if captures[batch_size] is None:
                raise GraphCaptureError("torch batch capture failed")
        for name, fn in torch_fns.items():
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                if capture_graphs:
                    captures[batch_size][name]()
                else:
                    for _ in range(batch_size):
                        fn()
                stop.record()
            torch_events[name].append((start, stop, batch_size))

    bench = await backend.benchmark_async(
        compiled_graph, warmup=warmup, num_iters=iters, on_iter=on_iter, capture_graphs=capture_graphs, input_data=input_data
    )
    torch.cuda.synchronize()

    results: dict[str, float] = {}
    measured_count = len(bench.per_launch[0].samples or ()) if bench.per_launch else 0
    for name, evt in torch_events.items():
        measured = evt[-measured_count:] if measured_count else []
        if measured:
            # ``elapsed_time`` is in ms across the whole batch; divide
            # by the batch size and multiply by 1000 to get per-call us.
            # ``min`` (best-case iter) for both torch and emmy — the
            # least-noise latency, matching tune's min-over-variants reporting
            # so tune and run numbers are comparable.
            per_iter_us = [s.elapsed_time(e) * 1000.0 / b for s, e, b in measured]
            results[name] = min(per_iter_us)
    # Whole-program (e2e) time when available — windows around replays of one
    # all-launches CUDA graph, the same semantics the captured torch closures
    # above get. The fallback sums per-launch solo windows, which is not an
    # end-to-end number (no cross-kernel cache effects).
    results["Emmy"] = bench.total_us
    results.update(failed)
    return results, bench
