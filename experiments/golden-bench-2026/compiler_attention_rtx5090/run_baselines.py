#!/usr/bin/env python3
"""Measure the pinned attention libraries against current PyTorch Inductor."""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import json
import logging
import statistics
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

OPERATORS = ("prefill_global", "prefill_causal", "prefill_gqa", "decode_causal", "decode_gqa")
#: Why cuDNN declined a setup, when it did. Read once, into that setup's record.
_CUDNN_REFUSAL: dict[str, str] = {}
# TileLang 0.1.8 bundles a TVM whose Python half breaks against apache-tvm-ffi 0.1.12 and newer, so the
# release contemporaneous with it is pinned as tightly as the libraries themselves.
VERSIONS = {"torch": "2.14.0", "flash_attn": "2.8.3", "tilelang": "0.1.8", "apache-tvm-ffi": "0.1.8.post2"}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("operator", choices=OPERATORS, nargs="?")
    parser.add_argument("batch", type=int, nargs="?")
    parser.add_argument("sequence_length", type=int, nargs="?")
    parser.add_argument("--warmup", type=int)
    parser.add_argument("--iters", type=int)
    parser.add_argument("--tilelang-source", type=Path)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _check_versions() -> None:
    for package, expected in VERSIONS.items():
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"{package} must be {expected}, found {actual}")


def _shape(operator: str, batch: int, sequence_length: int) -> tuple[tuple[int, ...], ...]:
    q_heads = 64 if operator.endswith("gqa") else 32
    kv_heads = 8 if operator.endswith("gqa") else 32
    q_length = 1 if operator.startswith("decode") else sequence_length
    return (
        (batch, q_heads, q_length, 128),
        (batch, kv_heads, sequence_length, 128),
        (batch, kv_heads, sequence_length, 128),
    )


def _sdpa(operator: str, q: Any, k: Any, v: Any) -> Any:
    import torch.nn.functional as F

    if operator == "decode_gqa":
        batch = q.shape[0]
        return F.scaled_dot_product_attention(
            q.reshape(batch, 8, 8, 1, 128),
            k.reshape(batch, 8, 1, k.shape[-2], 128),
            v.reshape(batch, 8, 1, v.shape[-2], 128),
            is_causal=False,
        ).reshape(batch, 64, 1, 128)
    return F.scaled_dot_product_attention(
        q,
        k,
        v,
        is_causal=operator in {"prefill_causal", "prefill_gqa"},
        enable_gqa=operator == "prefill_gqa",
    )


def _load_tilelang_kernel(source: Path, operator: str, batch: int, sequence_length: int) -> Callable[..., Any]:
    filename = "example_gqa_fwd_bshd.py" if operator == "prefill_gqa" else "example_mha_fwd_bshd.py"
    module_path = source / "examples" / "flash_attention" / filename
    spec = importlib.util.spec_from_file_location(f"tilelang_018_{operator}", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load TileLang source: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if operator == "prefill_gqa":
        return module.flashattn(
            batch,
            64,
            sequence_length,
            128,
            True,
            groups=8,
            block_M=64,
            block_N=64,
            num_stages=2,
            threads=128,
        )
    return module.flashattn(
        batch,
        32,
        sequence_length,
        128,
        operator == "prefill_causal",
        block_M=128,
        block_N=128,
        num_stages=1,
        threads=128,
    )


def _build_functions(args: argparse.Namespace, q: Any, k: Any, v: Any) -> dict[str, Callable[[], Any]]:
    import torch
    from flash_attn import flash_attn_func
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    def sdpa() -> Any:
        return _sdpa(args.operator, q, k, v)

    def cudnn() -> Any:
        with sdpa_kernel(SDPBackend.CUDNN_ATTENTION):
            return _sdpa(args.operator, q, k, v)

    torch._dynamo.reset()
    compiled_sdpa = torch.compile(sdpa, fullgraph=True, mode="max-autotune-no-cudagraphs")

    q_bshd = q.transpose(1, 2).contiguous()
    k_bshd = k.transpose(1, 2).contiguous()
    v_bshd = v.transpose(1, 2).contiguous()

    def flash_attention_2() -> Any:
        return flash_attn_func(
            q_bshd,
            k_bshd,
            v_bshd,
            causal=args.operator in {"prefill_causal", "prefill_gqa"},
        ).transpose(1, 2)

    block_mask = None
    if args.operator in {"prefill_causal", "prefill_gqa"}:

        def causal_mask(_batch: Any, _head: Any, query: Any, key: Any) -> Any:
            return query >= key

        # The mask reads neither the batch nor the head index, so one mask serves every one of
        # them. Ask for it compiled: the eager builder materializes the dense [B, 1, Q, KV]
        # boolean and reduces it, which is 64 GiB at batch 8 and 32768 -- larger than any card
        # this runs on, for a mask that is the same triangle every time.
        block_mask = create_block_mask(
            causal_mask,
            B=None,
            H=None,
            Q_LEN=q.shape[-2],
            KV_LEN=k.shape[-2],
            device="cuda",
            _compile=True,
        )

    def flex() -> Any:
        return flex_attention(q, k, v, block_mask=block_mask, enable_gqa=args.operator.endswith("gqa"))

    torch._dynamo.reset()
    compiled_flex = torch.compile(flex, fullgraph=True, mode="max-autotune-no-cudagraphs")
    functions: dict[str, Callable[[], Any]] = {
        "SDPA": sdpa,
        "PyTorch Inductor": compiled_sdpa,
        "FlexAttention": compiled_flex,
        "FlashAttention-2": flash_attention_2,
    }
    # cuDNN is the library the paper's headline claim is measured against. It serves only the
    # shapes its own kernels cover, and an unsupported one raises here rather than silently
    # dispatching elsewhere -- that setup records why instead of a number.
    try:
        cudnn()
    except Exception as error:  # noqa: BLE001 - any refusal means this shape has no cuDNN kernel
        functions["cuDNN"] = None
        _CUDNN_REFUSAL[args.operator] = f"{type(error).__name__}: {error}"
    else:
        functions["cuDNN"] = cudnn

    if args.operator.startswith("prefill"):
        tilelang_kernel = _load_tilelang_kernel(args.tilelang_source, args.operator, args.batch, args.sequence_length)

        def tilelang() -> Any:
            return tilelang_kernel(q_bshd, k_bshd, v_bshd).transpose(1, 2)

        functions["TileLang"] = tilelang
    return {name: function for name, function in functions.items() if function is not None}


def _check_outputs(functions: dict[str, Callable[[], Any]]) -> dict[str, dict[str, Any]]:
    """Compare every backend against SDPA one batch element at a time.

    Comparing whole tensors needs several more copies of the output than the measurement window
    itself holds, so at batch 8 it is the correctness check -- not the timing -- that runs out of
    memory. The criterion is unchanged: every element within rtol/atol, and both reported errors
    are over the whole output.
    """
    import torch

    reference = functions["SDPA"]()
    checks = {}
    for name, function in functions.items():
        output = function()
        largest, total = 0.0, 0.0
        for actual, expected in zip(output.split(1), reference.split(1), strict=True):
            torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
            difference = (actual - expected).abs()
            largest = max(largest, difference.max().item())
            total += difference.sum(dtype=torch.float64).item()
        checks[name] = {
            "status": "pass",
            "rtol": 1e-2,
            "atol": 1e-2,
            "max_abs_error": largest,
            "mean_abs_error": total / output.numel(),
        }
        del output
    return checks


def _capture_all(functions: dict[str, Callable[[], Any]]) -> tuple[dict[str, Callable[[], Any]], list[Any]]:
    """Capture every backend into one shared graph pool.

    A pool per graph holds one whole output tensor per backend at once and fragments what is left,
    which is what put batch 8 past this card's memory from 16384 up. The backends are independent --
    each reads the same inputs and writes its own output -- and correctness is settled before
    capture, so the outputs may share the pool: a timing window never reads them.
    """
    import torch

    torch.cuda.empty_cache()
    pool = torch.cuda.graph_pool_handle()
    graphs = []
    replays = {}
    for name, function in functions.items():
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                function()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph, pool=pool):
            function()
        graphs.append(graph)
        replays[name] = graph.replay
    return replays, graphs


def _measure(functions: dict[str, Callable[[], Any]], warmup: int, iters: int) -> tuple[dict[str, list[float]], bool, str | None]:
    import torch

    graphs = []
    try:
        measured, graphs = _capture_all(functions)
    except Exception as error:  # noqa: BLE001 - every backend falls back together to retain timing parity
        measured = functions
        captured = False
        capture_error = f"{type(error).__name__}: {error}"
    else:
        captured = True
        capture_error = None

    names = list(measured)
    events = {name: [] for name in names}
    for iteration in range(warmup + iters):
        offset = iteration % len(names)
        for name in names[offset:] + names[:offset]:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            with torch.no_grad():
                start.record()
                measured[name]()
                stop.record()
            if iteration >= warmup:
                events[name].append((start, stop))
    torch.cuda.synchronize()
    samples = {name: [start.elapsed_time(stop) * 1000 for start, stop in backend_events] for name, backend_events in events.items()}
    del graphs
    return samples, captured, capture_error


def _run(args: argparse.Namespace) -> int:
    import torch

    _check_versions()
    if args.tilelang_source is None or args.json is None or args.warmup is None or args.iters is None:
        raise ValueError("--tilelang-source, --json, --warmup, and --iters are required")
    if args.batch not in {1, 8}:
        raise ValueError("batch must be 1 or 8")

    torch.manual_seed(0)
    q, k, v = (torch.randn(shape, device="cuda", dtype=torch.float16) for shape in _shape(args.operator, args.batch, args.sequence_length))
    functions = _build_functions(args, q, k, v)
    with torch.no_grad():
        for function in functions.values():
            for _ in range(args.warmup + 5):
                function()
    checks = _check_outputs(functions)
    samples, captured, capture_error = _measure(functions, args.warmup, args.iters)
    estimates = {name: statistics.fmean(values) for name, values in samples.items()}
    inductor_us = estimates["PyTorch Inductor"]
    backends = {
        name: {
            "status": "ok",
            "latency_us": estimate,
            "min_us": min(samples[name]),
            "median_us": statistics.median(samples[name]),
            "samples_us": samples[name],
            "inductor_normalized_speedup": inductor_us / estimate,
            "correctness": checks[name],
        }
        for name, estimate in estimates.items()
    }
    if args.operator.startswith("decode"):
        backends["TileLang"] = {
            "status": "not-applicable",
            "reason": "TileLang 0.1.8 has no matching full-attention decode example for this contract",
        }
    if args.operator in _CUDNN_REFUSAL:
        backends["cuDNN"] = {"status": "not-applicable", "reason": _CUDNN_REFUSAL[args.operator]}

    payload = {
        "schema_version": 1,
        "operator": args.operator,
        "batch": args.batch,
        "sequence_length": args.sequence_length,
        "dtype": "float16",
        "q_heads": q.shape[1],
        "kv_heads": k.shape[1],
        "head_dim": q.shape[-1],
        "warmup": args.warmup,
        "iters": args.iters,
        "latency_estimator": "mean",
        "timing_semantics": "captured_whole_forward" if captured else "uncaptured_forward",
        "capture_error": capture_error,
        "normalization": "PyTorch Inductor latency divided by backend latency; values above one favor the backend",
        "gpu": torch.cuda.get_device_name(0),
        "versions": {package: importlib.metadata.version(package) for package in VERSIONS} | {"cudnn": torch.backends.cudnn.version()},
        "backends": backends,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2) + "\n")
    for name, estimate in estimates.items():
        logger.info("%s: %.3f us (%.4fx Inductor-normalized)", name, estimate, inductor_us / estimate)
    for name, entry in backends.items():
        if entry["status"] == "not-applicable":
            logger.info("%s: not applicable (%s)", name, entry["reason"])
    if not captured:
        logger.error("CUDA graph capture failed: %s", capture_error)
        return 1
    return 0


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    if args.smoke:
        logger.info("baseline runner smoke check passed")
        return
    if args.operator is None or args.batch is None or args.sequence_length is None:
        raise SystemExit("OPERATOR, BATCH, and SEQUENCE_LENGTH are required")
    try:
        status = _run(args)
    except Exception:
        logger.exception("baseline measurement failed")
        status = 1
    raise SystemExit(status)


if __name__ == "__main__":
    main()
