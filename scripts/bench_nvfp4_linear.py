#!/usr/bin/env python3
"""Benchmark native PyTorch and vLLM NVFP4 linear paths on identical operands.

By default, activation and weight quantization happen once outside the timed
region.  Both operators consume the same packed E2M1 bytes and the same
CUTLASS-layout E4M3 block scales.  This is the direct, prequantized matmul
comparison used by the NVFP4 showcase investigation.

An exact lowered Emmy single-kernel JSON can be supplied for an independent
same-input output check.  The JSON must have the corresponding M/K/N and output
dtype; Emmy consumes the logical scale layout while PyTorch and vLLM consume
the swizzled layout derived from those same values.

With ``--dynamic-activation``, weight quantization remains outside the timed
region while activation quantization is timed with each matmul.  This matches
vLLM's CUTLASS NVFP4 linear path.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def _args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--dtype", choices=("float16", "bfloat16"), default="bfloat16")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--samples", type=int, default=10)
    p.add_argument("--calls", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--emmy-ir", type=Path)
    p.add_argument("--json", type=Path, help="Also write the complete machine-readable report to this path")
    p.add_argument("--show-kernels", action="store_true")
    p.add_argument(
        "--dynamic-activation",
        action="store_true",
        help="Time activation quantization plus matmul, matching vLLM's CUTLASS NVFP4 linear path",
    )
    return p.parse_args()


def _time_cuda(fn, *, warmup: int, samples: int, calls: int) -> list[float]:
    import torch

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        begin.record()
        for _ in range(calls):
            fn()
        end.record()
        end.synchronize()
        values.append(begin.elapsed_time(end) * 1000.0 / calls)
    return values


def _summary(values: list[float]) -> dict[str, object]:
    return {
        "median_us": statistics.median(values),
        "min_us": min(values),
        "max_us": max(values),
        "samples_us": values,
    }


def _kernel_names(fn) -> list[str]:
    import torch

    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
        fn()
        torch.cuda.synchronize()
    return sorted({event.name for event in prof.events() if event.device_type == torch.autograd.DeviceType.CUDA})


def _emmy_output(path: Path, x, a, scale_a, b, scale_b, dtype, *, warmup: int, samples: int):
    """Run lowered Emmy IR on the tensors used by the stock ops."""
    import numpy as np
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import benchmark_program
    from emmy.compiler.graph import Graph
    from emmy.compiler.loader.binder import bind_constants

    graph = Graph.from_dict(json.loads(path.read_text()))
    direct_inputs = {
        "input_static_fp4_bits": (a.shape[0], a.shape[1]),
        "input_static_fp4_scale_bits": (a.shape[0], scale_a.shape[1], 1),
    }
    full_inputs = {"input": tuple(x.shape)}
    actual_inputs = {name: tuple(d.as_static() for d in graph.buffer(name).shape) for name in graph.inputs}
    if actual_inputs not in (direct_inputs, full_inputs):
        raise ValueError(f"Emmy IR input shapes {actual_inputs} match neither {direct_inputs} nor {full_inputs}")

    sources = {
        "l0.input_scale": np.asarray(1.0, dtype=np.float32),
        "l0.weight": b.detach().cpu().numpy(),
        "l0.weight_scale": scale_b.detach().cpu().view(torch.uint8).numpy(),
        "l0.weight_scale_2": np.asarray([1.0], dtype=np.float32),
    }
    inputs = bind_constants(graph, sources)
    if actual_inputs == direct_inputs:
        inputs.update(
            {
                "input_static_fp4_bits": a.detach().cpu().numpy(),
                "input_static_fp4_scale_bits": scale_a.detach().cpu().view(torch.uint8).numpy()[..., None],
            }
        )
    elif dtype is torch.bfloat16:
        inputs["input"] = x.detach().cpu().contiguous().view(torch.uint16).numpy()
    else:
        inputs["input"] = x.detach().cpu().numpy()
    backend = CudaBackend()
    compiled = backend.compile(graph)
    result, _ = backend.run(compiled, input_data=inputs)
    bench = benchmark_program(compiled, input_data=inputs, warmup=warmup, num_iters=samples)
    launches = bench.per_launch or []
    if len(launches) == 1:
        emmy_timing = _summary([value * 1000.0 for value in launches[0].samples or ()])
    else:
        min_ms = bench.e2e_min_ms if bench.e2e_min_ms is not None else bench.min_ms
        emmy_timing = {
            "median_us": (bench.e2e_ms if bench.e2e_ms is not None else bench.time_ms) * 1000.0,
            "min_us": (min_ms if min_ms is not None else bench.time_ms) * 1000.0,
            "captured": bench.captured,
            "num_launches": bench.num_launches,
            "per_launch": [{"kernel": launch.kernel_name, "median_us": launch.time_ms * 1000.0} for launch in launches],
        }
    out = result.outputs[graph.outputs[0]]
    if dtype is torch.bfloat16:
        decoded = torch.from_numpy(out.view(np.uint16)).view(torch.bfloat16).float()
    else:
        decoded = torch.from_numpy(out).float()
    return decoded, emmy_timing


def main() -> None:
    args = _args()

    import torch
    import vllm._custom_ops as vllm_ops
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import swizzle_blockscale

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    dtype = getattr(torch, args.dtype)
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    w = torch.randn(args.n, args.k, device="cuda", dtype=dtype)
    one = torch.ones((), device="cuda", dtype=torch.float32)

    # Produce one logical W4A4 problem. Weight quantization always remains
    # outside the timed region, as it does after model loading.  The prepared
    # activation is used by the direct mode and by the same-input output checks.
    a, scale_a = vllm_ops.scaled_fp4_quant(x, one, is_sf_swizzled_layout=False)
    b, scale_b = vllm_ops.scaled_fp4_quant(w, one, is_sf_swizzled_layout=False)
    blocked_scale_a = swizzle_blockscale(scale_a)
    blocked_scale_b = swizzle_blockscale(scale_b)
    b_torch = b.view(torch.float4_e2m1fn_x2).t()

    def activation():
        if args.dynamic_activation:
            return vllm_ops.scaled_fp4_quant(
                x,
                one,
                is_sf_swizzled_layout=True,
                backend="cutlass",
            )
        return a, blocked_scale_a

    def pytorch_call():
        a_call, scale_a_call = activation()
        return torch._scaled_mm(
            a_call.view(torch.float4_e2m1fn_x2),
            b_torch,
            scale_a_call,
            blocked_scale_b,
            out_dtype=dtype,
        )

    def vllm_call():
        a_call, scale_a_call = activation()
        return vllm_ops.cutlass_scaled_fp4_mm(
            a_call,
            b,
            scale_a_call,
            blocked_scale_b,
            one,
            dtype,
        )

    torch_out, vllm_out = pytorch_call(), vllm_call()
    delta = (torch_out.float() - vllm_out.float()).abs()
    report = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "dtype": args.dtype,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "protocol": {
            "warmup": args.warmup,
            "samples": args.samples,
            "calls_per_sample": args.calls,
            "dynamic_activation": args.dynamic_activation,
            "timed_operation": "activation_quantize_and_matmul" if args.dynamic_activation else "prequantized_matmul",
            "weight_quantization_timed": False,
        },
        "pytorch": _summary(_time_cuda(pytorch_call, warmup=args.warmup, samples=args.samples, calls=args.calls)),
        "vllm": _summary(_time_cuda(vllm_call, warmup=args.warmup, samples=args.samples, calls=args.calls)),
        "pytorch_vs_vllm": {
            "max_abs": delta.max().item(),
            "mean_abs": delta.mean().item(),
            "exact_fraction": (torch_out == vllm_out).float().mean().item(),
        },
    }
    if args.show_kernels:
        report["pytorch"]["cuda_kernels"] = _kernel_names(pytorch_call)
        report["vllm"]["cuda_kernels"] = _kernel_names(vllm_call)
    if args.emmy_ir:
        emmy_out, emmy_timing = _emmy_output(
            args.emmy_ir,
            x,
            a,
            scale_a,
            b,
            scale_b,
            dtype,
            warmup=args.warmup,
            samples=args.samples,
        )
        emmy_delta = (emmy_out - torch_out.cpu().float()).abs()
        report["emmy"] = emmy_timing
        report["emmy_vs_pytorch"] = {
            "max_abs": emmy_delta.max().item(),
            "mean_abs": emmy_delta.mean().item(),
            "exact_fraction": (emmy_out == torch_out.cpu().float()).float().mean().item(),
        }
    rendered = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
