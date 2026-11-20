"""Whether a compiled program computed the right answer: its outputs against a reference's on the same inputs.

Two references and two strengths. The scaled check (:func:`check_accuracy`) compares against eager torch under
tolerances wide enough for fp16 accumulation; the strict proof (:func:`strict_correctness_proof`,
:func:`strict_eager_proof`) holds every element to ``rtol=atol=1e-3`` or to one rounding step of the output's
format. A pinned bench row is compared against the greedy pick's own outputs (:func:`wrong_answer_flag`), and
:func:`resolve_reference_disagreement` says when that reference does not reproduce.
"""

from __future__ import annotations

import contextlib
import logging

logger = logging.getLogger(__name__)


@contextlib.contextmanager
def reference_precision():
    """Torch as the CORRECTNESS reference: its GEMMs without the reduced-precision reductions it enables by
    default. Those trade accuracy for speed inside the reference itself — at K=15360 the default FP16 GEMM
    leaves 9.6% of its own elements outside ``rtol=atol=1e-3`` of an FP64 product, 0.1% with them off — so a
    kernel nearer the truth than eager failed ``--strict`` for eager's error. The TIMED eager forward keeps
    torch's defaults: that is the library a user runs."""
    import torch

    matmul = torch.backends.cuda.matmul
    # Newer Torch versions pair the precision flag with an independent split-K flag.
    # The native getter retains both; the public getter returns only the first.
    fp16 = torch._C._get_cublas_allow_fp16_reduced_precision_reduction()
    bf16 = torch._C._get_cublas_allow_bf16_reduced_precision_reduction()
    try:
        matmul.allow_fp16_reduced_precision_reduction = (False, fp16[1]) if isinstance(fp16, tuple) else False
        matmul.allow_bf16_reduced_precision_reduction = (False, bf16[1]) if isinstance(bf16, tuple) else False
        yield
    finally:
        matmul.allow_fp16_reduced_precision_reduction = fp16
        matmul.allow_bf16_reduced_precision_reduction = bf16


def wrong_answer_flag(outputs: dict, ref_outputs: dict, quantized: frozenset[str]) -> str | None:
    """Output-correctness gate for a pinned A/B row: compare the pinned kernel's outputs
    against the greedy run's on the SAME inputs. Both sides are emmy kernels over one
    graph, so they agree to reduction-reorder noise — a large deviation means the pinned
    config computed the wrong answer (the ``g2a`` atomic-split re-bench class: a skipped
    zero-init / finalize benches fast and silently wrong). Returns a flag string or
    ``None``; loose 5% relative tolerance so split-K / atomic reorders never trip it.

    ``quantized`` names the outputs whose buffers are packed 4-bit codes (``f4e2m1x2``, compared decoded). Such an
    output flips a code wherever a value sits on a rounding boundary, which moves that element by a whole
    quantization step, so its worst element alone proves nothing: it flags only when its mean error also exceeds 0.5%
    of its peak. That bound comes from the Qwen3.8-27B-NVFP4 MLP on an RTX 5090, whose encoded output differs from
    both the NumPy and the Torch reference on 1.2% of its codes with mean error 0.17% of peak, under every schedule
    tried, the scalar one included. Every other output flags on its worst element, so a sparse fault (one bad row,
    a wrong tail element) still fails."""
    import numpy as np  # noqa: PLC0415

    assert isinstance(quantized, frozenset), f"quantized must be a frozenset of output names, got {type(quantized).__name__}"
    worst = 0.0
    for nid, ref in ref_outputs.items():
        got = outputs.get(nid)
        if got is None:
            return f"wrong-answer: output {nid!r} missing"
        a = np.asarray(got, dtype=np.float64)
        b = np.asarray(ref, dtype=np.float64)
        if a.shape != b.shape:
            return f"wrong-answer: output {nid!r} shape {a.shape} != greedy {b.shape}"
        if _nonfinite_mismatch(a, b):
            return f"wrong-answer: output {nid!r} has non-finite values the greedy output does not"
        finite = np.isfinite(b)
        if finite.any():
            denom = float(np.abs(b[finite]).max()) or 1.0
            diff = np.abs(a[finite] - b[finite]) / denom
            if nid not in quantized or float(diff.mean()) > 0.005:
                worst = max(worst, float(diff.max()))
    if worst > 0.05:
        return f"wrong-answer: rel err {worst:.3f} vs greedy output"
    return None


def quantized_outputs(outputs: dict, graph) -> frozenset[str]:
    """The names in ``outputs`` whose buffers in ``graph`` are packed 4-bit codes — the ones
    :func:`comparison_outputs` decodes, under the same rule."""
    if not hasattr(graph, "buffer"):
        return frozenset()
    return frozenset(name for name in outputs if graph.buffer(name).dtype.name == "f4e2m1x2")


def _nonfinite_mismatch(actual, expected) -> bool:
    """Whether a NaN or an infinity sits where the other side does not hold the same value.

    A mask legitimately holds ``-inf``, so an infinity of one sign at the same position on both sides passes;
    so does a NaN on both sides. Both arguments are float64 arrays of one shape."""
    import numpy as np  # noqa: PLC0415

    finite = np.isfinite(expected)
    return not np.array_equal(np.isfinite(actual), finite) or not np.array_equal(actual[~finite], expected[~finite], equal_nan=True)


REFERENCE_SELF_DISAGREES = (
    "wrong-answer reference unusable: a row realizing the greedy's own config disagrees with the "
    "greedy output, so no row's comparison against it carries information"
)


def resolve_reference_disagreement(verdicts, ref_knobs) -> list[str | None]:
    """The wrong-answer flags a session can actually justify, one per benched row.

    ``verdicts`` is one ``(realized_knob_dicts, wrong_answer_flag_or_None)`` per row and
    ``ref_knobs`` is the greedy's own realized knobs — the configuration whose outputs every row is
    compared against by :func:`wrong_answer_flag`.

    A row that realized the reference configuration computes the reference. If such a row is flagged
    as disagreeing with it, the reference is not reproducible, and then NO row's comparison against
    it means anything: a genuinely wrong row and a correct one are indistinguishable. Emitting a
    wrong-answer flag per row there is worse than emitting none, because a flag that fires on
    everything is one people learn to skip past — which is how a real deviation gets waved through.

    So in that case every wrong-answer verdict is dropped and the session says once, on the row that
    demonstrates it, that the reference itself is unusable. Every other kind of flag is untouched,
    and where the reference does reproduce the per-row verdicts stand exactly as before.
    """
    verdicts = list(verdicts)
    if not ref_knobs:
        return [flag for _, flag in verdicts]
    witness = next((i for i, (knobs, flag) in enumerate(verdicts) if flag and knobs == ref_knobs), None)
    if witness is None:
        return [flag for _, flag in verdicts]
    return [REFERENCE_SELF_DISAGREES if i == witness else None for i in range(len(verdicts))]


def comparison_outputs(outputs: dict, graph) -> dict:
    """Decode physical carriers before command-layer correctness checks."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import decode_bf16  # noqa: PLC0415

    if not hasattr(graph, "buffer"):
        return outputs
    decoded = outputs
    for name, value in outputs.items():
        arr = np.asarray(value)
        dtype = graph.buffer(name).dtype
        if dtype.name == "bf16" and arr.dtype == np.uint16:
            if decoded is outputs:
                decoded = dict(outputs)
            decoded[name] = decode_bf16(arr)
        elif dtype.name in QUANTIZED_FORMATS and arr.dtype == np.uint8:
            if decoded is outputs:
                decoded = dict(outputs)
            decoded[name] = _quantized_values(arr, dtype.name)
    return decoded


#: Mantissa bits of the IEEE float formats, by dtype name: the fewer, the coarser the rounding step.
_FLOAT_MANTISSA_BITS = {"bf16": 7, "f16": 10, "f32": 23, "f64": 52}


def computed_from_formats(frontend) -> dict:
    """The coarsest IEEE float format each output of ``frontend`` is computed in: the coarsest format an op of its
    producing chain rounds into. A layout op (the trace writes a cast as one) moves its first input's values and rounds
    them only into a coarser format, so a bf16 input sliced and cast to f32 adds no bf16 step; every other op rounds
    into its own format. Graph inputs and constants do not count (the FP64 evaluation reads them exactly), nor do
    quantized values (the math decodes them to a float first). ``None`` for an output no op rounds into a float."""
    from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
    from emmy.compiler.ir.frontend.ir import CatOp, ReshapeOp, SliceOp, TransposeOp, UnsqueezeOp  # noqa: PLC0415
    from emmy.compiler.ir.tensor.ir import GatherOp, IndexMapOp, ScatterOp  # noqa: PLC0415

    layout = (CatOp, GatherOp, IndexMapOp, ReshapeOp, ScatterOp, SliceOp, TransposeOp, UnsqueezeOp)
    bits = _FLOAT_MANTISSA_BITS

    def rounds(node, fmt: str) -> bool:
        source = frontend.buffer(node.inputs[0]).dtype.name if isinstance(node.op, layout) and node.inputs else None
        return source not in bits or bits[source] > bits[fmt]

    def coarsest(output: str) -> str | None:
        found, stack, seen = [], [output], set()
        while stack:
            buf = stack.pop()
            node = frontend.producer(buf)
            if buf in seen or node is None or isinstance(node.op, (ConstantOp, InputOp)):
                continue
            seen.add(buf)
            fmt = frontend.buffer(buf).dtype.name
            if fmt in bits and rounds(node, fmt):
                found.append(fmt)
            stack.extend(node.inputs)
        return min(found, key=bits.get, default=None)

    return {output: coarsest(output) for output in frontend.outputs}


def output_formats(outputs: dict, graph, computed_from: dict | None = None) -> dict:
    """The format whose rounding step each of ``outputs`` that ``graph`` keeps as a buffer may differ by (see
    ``strict_correctness_proof``): a quantized output's storage format; for a float output, the coarser of its
    storage format and the format it is computed in (``computed_from``, from ``computed_from_formats``)."""
    if not hasattr(graph, "buffer"):
        return {}
    stored = {name: graph.buffer(name).dtype.name for name in outputs if graph.buffer(name) is not None}
    computed_from = computed_from or {}
    return {
        name: min(fmt, computed_from.get(name) or fmt, key=_FLOAT_MANTISSA_BITS.__getitem__) if fmt in _FLOAT_MANTISSA_BITS else fmt
        for name, fmt in stored.items()
    }


def strict_correctness_proof(
    outputs: dict,
    reference_out,
    *,
    reference="eager",
    rtol: float = 1e-3,
    atol: float = 1e-3,
    exact_out=None,
    formats: dict | None = None,
) -> dict:
    """Return a tolerance verdict against one named reference with reproducible error statistics.

    The pass rule is the same elementwise rule used by ``torch.testing.assert_close`` for
    compiler baselines: ``abs(actual - expected) <= atol + rtol * abs(expected)``, and a non-finite
    value must match the reference exactly. Reference outputs may be tensors, a positional tensor
    sequence, or an output-name mapping.

    ``exact_out`` is a high-precision (FP64) evaluation of the same graph on the same inputs. The
    reference is itself a low-precision evaluation, so a different summation order can round a few
    elements one step apart from it — visible past the elementwise rule after a later cancellation —
    without being any less accurate. When the elementwise rule fails and ``exact_out`` is given, the
    proof passes only if every output passes one of two rules against the exact values. Both require
    the candidate's mean absolute error to sit at most ``_EXACT_MEAN_SLACK`` above the reference's.

    - Largest error: the candidate's largest absolute error is at most the reference's plus one
      rounding step of the output's format (``formats``) at the exact value of the candidate's worst
      element (one ULP of a float format, one code of a quantized one; ``_rounding_step``). The two
      sides' worst elements sit at different magnitudes, so the same half-step rounding error is a
      larger absolute error where the values are larger; the step absorbs that and nothing more.
    - One code (quantized outputs only): every candidate value is the reference's or one of the two
      codes beside it (``_adjacent_codes``), and the candidate misses the exact code on no more
      elements than the reference. A value near the midpoint between two codes rounds either way,
      and one code step is a large jump that no float slack covers.

    Without ``exact_out`` (two compiled programs compared on the same inputs, with no FP64
    evaluation to weigh them against), a quantized output whose every value is the reference's or one
    code beside it passes, and ``one_code_outputs`` counts its differing values. This is the rule a
    strict check applies at the quantize nodes inside a program (``strict_eager_proof``).

    ``formats`` names each output's format (a dtype name, ``output_formats``): a quantized output's
    storage format; for a float output, the coarser of its storage format and the coarsest float format
    it is computed in. An f32 output cast from bf16 math carries bf16 rounding errors, and two summation
    orders of that bf16 math land up to a bf16 step apart, thousands of f32 steps. An output without a
    format gets no step, only the largest-error rule, and no one-code pass. A real fault — a wrong element beyond the reference's own
    error and one step, a code two steps off, or a bias across the output — still fails.
    """
    import numpy as np  # noqa: PLC0415

    def _array(value):
        if hasattr(value, "detach"):
            value = value.detach().float().cpu().numpy()
        return np.asarray(value, dtype=np.float64)

    names = list(outputs)
    if isinstance(reference_out, dict):
        missing = [name for name in names if name not in reference_out]
        extra = [name for name in reference_out if name not in outputs]
        if missing or extra:
            return {
                "status": "fail",
                "reference": reference,
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output names differ: missing={missing}, extra={extra}",
            }
        refs = [reference_out[name] for name in names]
    else:
        refs = list(reference_out) if isinstance(reference_out, (tuple, list)) else [reference_out]
        if len(refs) != len(names):
            return {
                "status": "fail",
                "reference": reference,
                "rtol": rtol,
                "atol": atol,
                "max_abs_error": None,
                "mean_abs_error": None,
                "max_rel_error": None,
                "error": f"output count differs: Emmy={len(names)}, {reference}={len(refs)}",
            }

    max_abs = 0.0
    max_rel = 0.0
    abs_sum = 0.0
    count = 0
    failure = None
    structural = False
    one_code = {}
    for name, ref in zip(names, refs, strict=True):
        actual = _array(outputs[name])
        expected = _array(ref)
        if actual.shape != expected.shape:
            failure = f"output {name!r} shape {actual.shape} != {reference} {expected.shape}"
            structural = True
            break
        if _nonfinite_mismatch(actual, expected):
            failure = f"output {name!r} has non-finite values the {reference} output does not"
            structural = True
            break
        finite = np.isfinite(expected)
        actual, expected = actual[finite], expected[finite]
        absolute = np.abs(actual - expected)
        tolerance = atol + rtol * np.abs(expected)
        if absolute.size:
            max_abs = max(max_abs, float(absolute.max()))
            max_rel = max(max_rel, float((absolute / np.maximum(np.abs(expected), atol)).max()))
            abs_sum += float(absolute.sum())
            count += int(absolute.size)
            fmt = (formats or {}).get(name)
            if (
                exact_out is None
                and fmt in QUANTIZED_FORMATS
                and not np.all(absolute <= tolerance)
                and np.all(_adjacent_codes(actual, expected, fmt))
            ):
                one_code[name] = int((actual != expected).sum())
            elif failure is None and not np.all(absolute <= tolerance):
                worst = int(np.argmax(absolute - tolerance))
                failure = (
                    f"output {name!r} exceeds rtol={rtol:g}, atol={atol:g}: "
                    f"{int((absolute > tolerance).sum())}/{absolute.size} elements, "
                    f"worst at flat index {worst}: emmy={actual.flat[worst]:.6g} {reference}={expected.flat[worst]:.6g}"
                )

    exact = None
    if failure is not None and not structural and exact_out is not None and count:
        values = exact_out() if callable(exact_out) else exact_out
        exact = _exact_error_comparison(outputs, refs, values, names, _array, formats or {}) if values is not None else None
        if exact is not None and exact["candidate_no_less_accurate"]:
            failure = None
    proof = {
        "status": "fail" if failure else "pass",
        "reference": reference,
        "rtol": rtol,
        "atol": atol,
        "max_abs_error": max_abs,
        "mean_abs_error": abs_sum / count if count else 0.0,
        "max_rel_error": max_rel,
    }
    if exact is not None:
        proof["exact_comparison"] = exact
    if one_code:
        proof["one_code_outputs"] = one_code
    if failure:
        proof["error"] = f"{failure} (max_abs={max_abs:.3g}, mean_abs={proof['mean_abs_error']:.3g}, max_rel={max_rel:.3g})"
    return proof


def _exact_outputs(frontend, input_tensors: dict, result_outputs: dict) -> dict:
    """The frontend graph evaluated in FP64 on the same inputs, by output name: the exact values a strict
    proof weighs the candidate and the low-precision reference against."""
    import torch  # noqa: PLC0415

    from emmy.compiler.backend import torch_ref  # noqa: PLC0415

    try:
        fn, tensors = torch_ref.build_callable(frontend, input_tensors, compute_dtype=torch.float64)
        with torch.no_grad():
            out = fn(*tensors)
    except Exception as exc:  # noqa: BLE001 — no FP64 evaluation leaves the elementwise verdict standing
        logger.warning("FP64 reference unavailable (%s) — the strict check stays elementwise", exc)
        return None
    return eager_outputs_by_name(result_outputs, out[: len(result_outputs)] if isinstance(out, tuple) else out, frontend)


#: How far above the reference's mean absolute error, relative to it, a candidate's may sit against the exact values
#: and still count as no less accurate: rounding-level reorderings move it by hundredths of a percent.
_EXACT_MEAN_SLACK = 0.01


def _rounding_step(value: float, fmt: str | None) -> float:
    """One rounding step of storage format ``fmt`` at ``value``: the ULP of a float format, the wider of the two gaps
    beside it on a quantized format's grid of codes. Zero for an unknown format or a non-finite value."""
    import numpy as np  # noqa: PLC0415

    if fmt is None or not np.isfinite(value):
        return 0.0
    magnitude = abs(float(value))
    if fmt in QUANTIZED_FORMATS:
        grid = _quantized_values(np.arange(256, dtype=np.uint8), fmt)
        grid = np.unique(np.abs(grid[np.isfinite(grid)]))
        below = grid[max(int(np.searchsorted(grid, magnitude, "left")) - 1, 0)]
        above = grid[min(int(np.searchsorted(grid, magnitude, "right")), len(grid) - 1)]
        return float(above - below) if magnitude not in grid else float(max(above - magnitude, magnitude - below))
    if fmt == "bf16":
        # 7 stored mantissa bits; below the smallest normal (2**-126) the spacing stays 2**-133.
        return float(np.ldexp(1.0, max(int(np.frexp(magnitude)[1]), -125) - 8)) if magnitude else 2.0**-133
    floats = {"f16": np.float16, "f32": np.float32, "f64": np.float64}
    return float(np.spacing(floats[fmt](magnitude))) if fmt in floats else 0.0


def _exact_error_comparison(outputs: dict, refs: list, exact_out, names: list, to_array, formats: dict) -> dict:
    """The candidate's and the reference's largest and mean absolute errors against a high-precision evaluation,
    per output, and whether the candidate is no less accurate on every one (see ``strict_correctness_proof``)."""
    import numpy as np  # noqa: PLC0415

    exacts = (
        [exact_out[name] for name in names]
        if isinstance(exact_out, dict)
        else (list(exact_out) if isinstance(exact_out, (tuple, list)) else [exact_out])
    )
    per_output, ok = {}, len(exacts) == len(names)
    for name, ref, ex in zip(names, refs, exacts, strict=False):
        exact = to_array(ex)
        actual, expected = to_array(outputs[name]).reshape(exact.shape), to_array(ref).reshape(exact.shape)
        finite = np.isfinite(exact)
        cand, base = np.abs(actual - exact)[finite], np.abs(expected - exact)[finite]
        fmt = formats.get(name)
        stats = {
            "candidate_max": float(cand.max()) if cand.size else 0.0,
            "candidate_mean": float(cand.mean()) if cand.size else 0.0,
            "reference_max": float(base.max()) if base.size else 0.0,
            "reference_mean": float(base.mean()) if base.size else 0.0,
            "rounding_step": _rounding_step(exact[finite].flat[int(cand.argmax())], fmt) if cand.size else 0.0,
        }
        largest = stats["candidate_max"] <= stats["reference_max"] + stats["rounding_step"]
        if fmt in QUANTIZED_FORMATS:
            stats["candidate_misses"] = int((cand > 0).sum())
            stats["reference_misses"] = int((base > 0).sum())
            stats["beyond_one_code"] = int((~_adjacent_codes(actual[finite], expected[finite], fmt)).sum())
            one_code = stats["beyond_one_code"] == 0 and stats["candidate_misses"] <= stats["reference_misses"]
        else:
            one_code = False
        per_output[name] = stats
        ok = ok and (largest or one_code) and stats["candidate_mean"] <= stats["reference_mean"] * (1 + _EXACT_MEAN_SLACK)
    return {"per_output": per_output, "candidate_no_less_accurate": bool(ok)}


#: The storage formats a frontend node can round a computed value into. Two correct evaluations of such a node land one
#: representable value apart wherever the value sits near the midpoint between two codes, and the layers after it
#: amplify that step past any elementwise tolerance.
QUANTIZED_FORMATS = frozenset({"f4e2m1x2", "f8e4m3", "f8e5m2"})


def quantized_twin(frontend) -> bool:
    """Whether the twin rounds a computed value into a quantized storage format (a quantized input is data, not a
    rounding). Two correct evaluations of such a twin land codes apart, so no elementwise verdict holds torch.compile's
    output to eager's: the column is timed unchecked, and the record says so."""
    from emmy.compiler.provenance import is_boundary  # noqa: PLC0415

    return frontend is not None and any(
        node.output.dtype.name in QUANTIZED_FORMATS and not is_boundary(node.op) for node in frontend.nodes.values()
    )


def quantize_nodes(frontend, lowered) -> tuple[str, ...]:
    """The quantize nodes at which a strict check adopts the candidate's values (see ``strict_eager_proof``).

    A quantize node is a frontend node that rounds a computed value into a quantized storage format and is not a graph
    output. The lowered program must keep it as a buffer of the same name, dtype and shape. The check compares a
    quantize node the lowered program does not keep like the rest of the graph, without adoption."""
    from emmy.compiler.provenance import is_boundary  # noqa: PLC0415

    def kept(name: str) -> bool:
        buffer, tensor = lowered.buffer(name), frontend.nodes[name].output
        return buffer is not None and buffer.dtype == tensor.dtype and tuple(buffer.shape) == tuple(tensor.shape)

    return tuple(
        name
        for name in frontend.topological_order()
        if frontend.nodes[name].output.dtype.name in QUANTIZED_FORMATS
        and not is_boundary(frontend.nodes[name].op)
        and name not in frontend.outputs
        and kept(name)
    )


def _graft_candidate(frontend, nodes: tuple[str, ...]):
    """``frontend`` with every consumer of each of ``nodes`` reading a new input ``<node>_candidate`` instead. The graft
    appends the nodes to the graph outputs, so the eager graph's own quantized value at each one comes out after the
    original outputs."""
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.tensor import Tensor  # noqa: PLC0415

    graph = frontend.copy()
    for name in nodes:
        tensor = graph.nodes[name].output
        fed = graph.add_node(op=InputOp(), inputs=[], output=Tensor(f"{name}_candidate", tensor.shape, tensor.dtype.name))
        for user in graph.consumers(name):
            graph.replace_input(user, name, fed)
    graph.outputs = [*frontend.outputs, *nodes]
    return graph


def _quantized_values(raw, fmt: str):
    """The numbers the raw bytes of a quantized storage format encode (two values per byte for ``f4e2m1x2``)."""
    import numpy as np  # noqa: PLC0415

    from emmy.compiler.dtype import decode_f4x2, decode_f8  # noqa: PLC0415

    raw = np.asarray(raw, dtype=np.uint8)
    return decode_f4x2(raw) if fmt == "f4e2m1x2" else decode_f8(raw, fmt)


def _adjacent_codes(ours, theirs, fmt: str):
    """Where ``ours`` is ``theirs`` or one of the two values beside it on the format's grid of representable values:
    the one step a value near the midpoint between two codes rounds either way across. A non-finite reference value
    accepts only itself."""
    import numpy as np  # noqa: PLC0415

    grid = _quantized_values(np.arange(256, dtype=np.uint8), fmt)
    grid = np.unique(grid[np.isfinite(grid)])
    at = np.searchsorted(grid, theirs)
    beside = (ours == grid[np.maximum(at - 1, 0)]) | (ours == grid[np.minimum(at + 1, len(grid) - 1)])
    return (ours == theirs) | (np.isfinite(theirs) & beside)


def strict_eager_proof(frontend, input_tensors: dict, outputs: dict, lowered, taps: dict) -> tuple[dict, dict, object]:
    """The ``strict_correctness_proof`` verdict on the compiled program's ``outputs`` (the candidate) against the
    eager evaluation of ``frontend`` on the same inputs.

    ``taps`` holds the candidate's values at the quantize nodes (``quantize_nodes``). The eager reference adopts them
    and computes everything after each quantize node from them, so the tolerance check covers every computation
    except the rounding at those nodes, which the proof judges code by code instead: the candidate's value must be the
    reference's or one of the two values beside it on the format's grid (``_adjacent_codes``). Comparing bytes there
    would fail correct programs — a value near the midpoint between two codes rounds either way, and the layers after
    it amplify that one step past the tolerance — while a value further away fails the proof like any other mismatch,
    naming the node and the count. The verdict's ``quantize_nodes`` reports, per node, how many values the reference
    rounds differently from the candidate and how many of those lie beyond one code. Returns the verdict, the
    reference outputs by name, and a zero-argument callable for their FP64 evaluation."""
    import numpy as np  # noqa: PLC0415
    import torch  # noqa: PLC0415

    from emmy.compiler.backend import torch_ref  # noqa: PLC0415

    graph = _graft_candidate(frontend, tuple(taps)) if taps else frontend
    device = next((t.device for t in input_tensors.values() if torch.is_tensor(t)), torch.device("cpu"))
    tensors = {**input_tensors, **{f"{name}_candidate": torch.from_numpy(np.ascontiguousarray(v)).to(device) for name, v in taps.items()}}
    fn, args = torch_ref.build_callable(graph, tensors)
    with torch.no_grad(), reference_precision():
        out = fn(*args)
    program_out = out[: len(outputs)] if taps else out
    eager_values = eager_outputs_by_name(outputs, program_out, lowered)
    memo: list = []

    def exact_values():
        if not memo:
            memo.append(_exact_outputs(graph, tensors, outputs))
        return memo[0]

    formats = output_formats(outputs, lowered, computed_from_formats(graph))
    proof = strict_correctness_proof(outputs, eager_values, exact_out=exact_values, formats=formats)
    if taps:
        rounding, faults = {}, []
        for name, reference in zip(taps, out[len(outputs) :], strict=True):
            fmt = frontend.nodes[name].output.dtype.name
            ours, theirs = _quantized_values(taps[name], fmt), _quantized_values(reference.cpu().numpy(), fmt)
            differ, beyond = ours != theirs, ~_adjacent_codes(ours, theirs, fmt)
            rounding[name] = {
                "values": int(differ.size),
                "rounded_differently": int(differ.sum()),
                "beyond_one_code": int(beyond.sum()),
                "max_abs_difference": float(np.abs(ours - theirs)[differ].max()) if differ.any() else 0.0,
            }
            logger.info("quantize node %s: %d/%d values rounded differently", name, rounding[name]["rounded_differently"], differ.size)
            if beyond.any():
                faults.append(f"quantize node {name!r}: {int(beyond.sum())}/{differ.size} values more than one code from the eager value")
        proof["quantize_nodes"] = rounding
        if faults:
            proof["status"] = "fail"
            proof["error"] = "; ".join([proof["error"], *faults] if "error" in proof else faults)
    return proof, eager_values, exact_values


def eager_outputs_by_name(outputs: dict, eager_out, graph=None) -> dict:
    """Map eager outputs to lowered names, preserving packed carriers for decoding."""
    refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    if len(refs) != len(outputs):
        return {}
    mapped = {}
    for name, ref in zip(outputs, refs, strict=True):
        if hasattr(ref, "detach"):
            tensor = ref.detach()
            if graph is not None and graph.buffer(name).dtype.name in QUANTIZED_FORMATS:
                import torch  # noqa: PLC0415

                tensor = tensor.contiguous().view(torch.uint8)
            try:
                ref = tensor.cpu().numpy()
            except TypeError:
                ref = tensor.float().cpu().numpy()
        mapped[name] = ref
    return comparison_outputs(mapped, graph) if graph is not None else mapped


def eager_output(module, args, kwargs, *, device: str = "cuda"):
    """Eager reference forward on ``device``. Returns the module's output — a Tensor, or a
    TUPLE of Tensors for a multi-output module (``check_accuracy`` compares
    each graph output buffer against the tuple positionally)."""
    import torch

    if device == "cuda":
        module, args, kwargs = (
            module.to("cuda"),
            tuple(a.to("cuda") if isinstance(a, torch.Tensor) else a for a in args),
            to_cuda_kwargs(kwargs),
        )
    with torch.no_grad():
        out = module(*args, **kwargs)
    if isinstance(out, tuple):
        tensors = tuple(t for t in out if isinstance(t, torch.Tensor))
        return tensors if len(tensors) > 1 else tensors[0] if tensors else out[0]
    return out


def to_cuda_tensor(arr, dtype):
    """numpy array → CUDA torch tensor in the node's dtype (default fp32).

    A BF16 value bound from its bits (the ``uint16`` carrier a folded table constant evaluates to) is reinterpreted,
    not converted: converting would read each bit pattern as a number."""
    import numpy as np
    import torch

    from emmy.compiler.backend.torch_ref import torch_dtype

    target = torch_dtype(dtype) or torch.float32
    if target is torch.bfloat16 and arr.dtype == np.uint16:
        return torch.from_numpy(np.ascontiguousarray(arr).view(np.int16)).to("cuda").view(torch.bfloat16)
    return torch.from_numpy(arr).to("cuda").to(target)


def to_cuda_kwargs(kwargs):
    import torch

    cuda_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, torch.Tensor):
            cuda_kwargs[k] = v.to("cuda")
        elif isinstance(v, tuple):
            cuda_kwargs[k] = tuple(t.to("cuda") if isinstance(t, torch.Tensor) else t for t in v)
        else:
            cuda_kwargs[k] = v
    return cuda_kwargs


def check_accuracy(outputs, eager_out) -> str | None:
    """Compare backend ``outputs`` against the eager reference — a **pure verdict**:
    ``None`` when every output passes, else a message naming what failed. No printing
    and no exit here: the check runs inside the bench worker child, whose stdout is the
    pickle protocol channel, so the verdict must travel as a return value and fatality
    is the caller's call — the ``--code`` / model path aborts on it, while ``run --ir``
    treats it as informational (a sliced reproducer is fed random *boundary* inputs
    that can be out-of-domain for the op, e.g. a kernel expecting a mean-of-squares
    gets random signed data → NaN from a downstream rsqrt)."""
    import numpy as np  # noqa: PLC0415

    # Multi-output: the eager reference may be a tuple — compare each backend
    # output against its positional eager counterpart (graph-output order).
    # Historic single-output behavior (every output vs THE eager tensor) is the
    # degenerate case of the fallback-to-first pairing below.
    if isinstance(eager_out, dict):
        if set(eager_out) != set(outputs):
            return f"CORRECTNESS FAIL: output names differ: Emmy={list(outputs)}, eager={list(eager_out)}"
        eager_refs = [eager_out[name] for name in outputs]
    else:
        eager_refs = list(eager_out) if isinstance(eager_out, (tuple, list)) else [eager_out]
    # Arrays, never Python lists: a list holds one float object per element. Measured at 2M cells the lists
    # peaked at 13.3 f64 copies of the output against 4.3 here, which at an LM-head output (134M cells) is 14 GB.
    eager_flats = [
        t.detach().cpu().double().flatten().numpy() if hasattr(t, "detach") else np.asarray(t, dtype=np.float64).ravel() for t in eager_refs
    ]
    if any(np.isnan(flat).any() for flat in eager_flats):
        return "eager reference contains NaN (reproducer inputs out of domain)"
    failures: list[str] = []
    for pos, (buf_name, arr) in enumerate(outputs.items()):
        eager_flat = eager_flats[pos] if len(eager_flats) == len(outputs) else eager_flats[0]
        values = np.asarray(arr, dtype=np.float64).ravel()
        if np.isnan(values).any():
            return f"CORRECTNESS FAIL: output {buf_name} contains NaN"
        if len(values) == len(eager_flat):
            diffs = np.abs(values - eager_flat)
            max_diff = float(diffs.max())
            mean_diff = float(diffs.mean())
            # Scale tolerance by max|eager| and by output dtype.
            #
            # fp32: matmul reduction-order drift grows with both K and
            # output magnitude. A fixed threshold flags benign drift on
            # randn×randn at large K as a failure (cp.async + split-K
            # atomic-add ordering vs eager's pairwise sum). Dual check:
            # ``max_diff <= 8% of peak`` (tight ceiling for the typical
            # case) OR ``mean_diff <= 0.5% of peak`` (escape hatch for
            # the long tail — splitK atomic-reduce on randn×randn
            # produces a handful of outliers at K=1024 / 2048 even when
            # the bulk of the output is accurate to 4+ decimals).
            # Codegen bugs that systematically corrupt the output (e.g.
            # the matmul_add fusion adding the residual per-K_s CTA)
            # fail both clauses: mean_diff lifts to ~2-3% of peak.
            #
            # fp16: every step has ~3 fewer decimal digits than fp32. The
            # split-K matmul path is dominated by atomicAdd into an
            # ``__half*`` buffer — each per-CTA partial converts to fp16
            # at the atomic boundary and loses ~11 bits per write. After
            # 1024 K-partials that's RMS error on the order of
            # ``peak * 0.3``. The proper fix (f32 scratch for split-K +
            # separate cast pass) is a future architectural change; for
            # now ``emmy run --bench`` needs to remain usable on
            # legitimate fp16 graphs, so the rtol budget tracks the
            # achievable accuracy of the current path. Bugs that
            # actually corrupt outputs still fail (whole-row mismatch /
            # NaN / order-of-magnitude wrong).
            is_fp16 = arr.dtype == np.float16
            # Three-part verdict — each clause with ONE job:
            #
            #   PASS iff (max_ok AND mean_diff ≤ mean_tol) OR escape_ok
            #   max_ok    = max_diff ≤ tol                                  (fp32)
            #             = max_diff ≤ 4·tol AND count(diff > tol) ≤ budget (fp16; budget
            #               EARNED by a heavy-tailed output — 0 on a gaussian one)
            #   escape_ok = mean_diff ≤ escape_tol [fp16: AND max_diff ≤ 4·tol AND count ≤ budget]
            #
            # - ``tol`` (the MAX clause) is the loose per-cell OUTLIER ceiling: fp16
            #   atomic-reduce accumulation can drift a cell up to ``peak`` in pathological
            #   cancellation cases (random-signed partials), fp32 stays at 8% of peak.
            # - ``mean_tol`` is the SYSTEMATIC-CORRUPTION gate and must stay tight: a
            #   permuted/scrambled output lands at mean_diff ≈ E|a−e| of independent
            #   same-distribution draws (≈ 15% of peak on a gaussian matmul output), while a
            #   correct kernel — including the half-atomic fp16 split-K path — measures
            #   ≤ ~0.1% of peak. The historical fp16 form (``max ≤ peak OR mean ≤ peak``)
            #   PASSED a matmul whose A slab was scrambled by a fill/drain swizzle mismatch
            #   (mean_diff 52 under tol 340, values of scale ~60 — the PR #354 sync-fill
            #   incident); 3% of peak keeps ~300× headroom over measured correct runs and
            #   fails any permutation-class bug REGARDLESS of its max (the old OR let any
            #   output with max_diff ≤ peak pass without ever consulting the mean).
            #   On HEAVY-TAILED outputs 3% of peak over-scales (a gemma-4 layer output:
            #   peak ≈ 24·RMS — a permutation scores mean_diff ≈ 1.04× that gate, a coin
            #   flip), so the gate is additionally floored by the output's own
            #   PERMUTATION SCORE: ``floor = mean|e_i − e_perm(i)|`` over a fixed shuffle
            #   of the eager values — what a scramble of THIS draw would measure. The
            #   ``min(3%·peak, 0.7·floor)`` keeps gaussian outputs on the 3%-of-peak gate
            #   (floor ≈ 1.13·σ ≫ it) and gives heavy-tailed ones ≥ 1.4× margin on BOTH
            #   sides (measured gemma layer-0: correct 0.52×, scramble 1.43× the gate).
            # - ``escape_tol`` (0.5% of peak) preserves the split-K outlier escape hatch: a
            #   handful of atomic-reorder outliers may exceed the max ceiling on randn×randn
            #   at K=1024/2048 while the bulk stays accurate to 4+ decimals — a near-zero
            #   mean vouches for them.
            #
            # fp16 MAX clause = outlier BUDGET + hard garbage ceiling, not a single-cell
            # max. HEAVY-TAILED fp16 outputs (a gemma-4 layer output: peak ≈ 24·RMS, the
            # layer_scalar / outlier-channel dynamic range) put single-cell diffs at
            # O(peak) IN-DISTRIBUTION on the current legitimate fp16 path (residual-
            # cancellation cells + the half-atomic split-K partials): measured correct
            # layer-0 runs show max_diff 0.7–0.9× peak with a tail draw occasionally
            # crossing 1.0× — a binary ``max ≤ peak`` intermittently aborts the bench on
            # a rerun-passes coin flip. The budget separates "a handful of tail cells"
            # (≤ ~0.0015% of cells over ``tol``, each under the 4× garbage ceiling) from
            # the real bug classes: a garbage/saturated cell (uninitialized read, Inf
            # math) blows the HARD ceiling regardless of count; a corrupt tile / row /
            # tail-guard window (the historical incidents: thousands of cells) blows the
            # BUDGET on either pass path (the escape hatch consults it too — a near-zero
            # mean must not vouch for a low-amplitude mass corruption); a permutation
            # blows the mean gate. The budget itself is EARNED by the output's tail shape
            # (``peak > 8·RMS`` — gemma layers sit at ~24×, a gaussian at ~5×): only there
            # do in-distribution cells legitimately land past ``tol``. A gaussian output
            # keeps the strict pre-budget max clause, and its handful of split-K reorder
            # outliers pass only through the escape hatch (near-zero mean AND ≤ budget).
            # Residual blind spot, accepted: ≤ budget cells at ≤ 4·tol with a near-zero
            # mean is statistically indistinguishable from the legit reorder class.
            # NOTE the mean gate's scramble
            # margin THINS on heavy-tailed outputs (a permuted gemma layer output scores
            # ≈ 1.04× mean_tol — 4% clear, vs ~5× on gaussian matmuls): that margin is
            # bounded by the current fp16 path's honest noise (mean_diff ≈ 0.4× mean_tol
            # on correct gemma runs), and widens only when split-K gets its f32 scratch.
            rel_tol = 1.0 if is_fp16 else 0.08
            abs_tol = 1e-1 if is_fp16 else 1e-3
            peak = float(np.abs(eager_flat).max())
            tol = max(abs_tol, rel_tol * peak)
            perm_floor = float(np.abs(np.random.default_rng(0).permutation(eager_flat) - eager_flat).mean())
            mean_tol = max(abs_tol, min(0.03 * peak, 0.7 * perm_floor))
            escape_tol = max(abs_tol, 0.005 * peak)
            outliers = int((diffs > tol).sum())
            budget = max(4, len(diffs) // 65536)
            rms = float(np.sqrt(np.mean(np.square(eager_flat))))
            heavy_tailed = peak > 8.0 * rms
            if is_fp16:
                # The hard 4·tol garbage ceiling bounds BOTH pass paths: the escape hatch
                # vouches for reorder outliers (peak-bounded by construction), never for a
                # garbage/saturated cell a near-zero mean would otherwise hide.
                max_ok = max_diff <= 4 * tol and outliers <= (budget if heavy_tailed else 0)
                escape_ok = mean_diff <= escape_tol and max_diff <= 4 * tol and outliers <= budget
            else:
                max_ok = max_diff <= tol
                escape_ok = mean_diff <= escape_tol
            verdict = "PASS" if (max_ok and mean_diff <= mean_tol) or escape_ok else "FAIL"
            if verdict == "FAIL":
                failures.append(
                    f"output {buf_name}: max_diff={max_diff:.6f} mean_diff={mean_diff:.6f} tol={tol:.6f}"
                    + (f" outliers={outliers}/budget {budget}" if is_fp16 else "")
                )
            else:
                logger.info(
                    "Accuracy vs eager: max_diff=%.6f mean_diff=%.6f tol=%.6f outliers=%d/%d PASS",
                    max_diff,
                    mean_diff,
                    tol,
                    outliers,
                    budget,
                )
        else:
            failures.append(f"output {buf_name}: size {len(values)} does not match eager {len(eager_flat)}")
    return f"accuracy check failed vs eager: {'; '.join(failures)}" if failures else None
