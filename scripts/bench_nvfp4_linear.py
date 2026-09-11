#!/usr/bin/env python3
"""Fairly compare complete dynamic NVFP4 linears on one shape.

Each backend captures exactly one BF16 -> NVFP4 quantize -> matmul call in a
CUDA graph.  Measurement windows interleave repeated graph replays, matching
Emmy's whole-program benchmark semantics without amortizing one giant graph
over many logical calls.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--samples", type=int, default=20)
    p.add_argument("--calls", type=int, default=200)
    p.add_argument("--profile-calls", type=int, default=0)
    p.add_argument("--emmy-ir", type=Path, required=True)
    p.add_argument("--json", type=Path)
    args = p.parse_args()

    import numpy as np
    import torch
    from vllm._custom_ops import scaled_fp4_quant
    from vllm.model_executor.kernels.linear import init_nvfp4_linear_kernel
    from vllm.model_executor.layers.quantization.utils.nvfp4_utils import (
        pad_nvfp4_activation_for_cutlass,
        pad_nvfp4_weight_for_cutlass,
        swizzle_blockscale,
    )
    from vllm.utils.flashinfer import flashinfer_scaled_fp4_mm

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.graph import Graph
    from emmy.compiler.loader.binder import bind_constants

    def capture_one(fn):
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side), torch.no_grad():
            for _ in range(3):
                fn()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(graph):
            output = fn()
        return graph, output

    def time_interleaved(replays):
        names = list(replays)
        for i in range(args.warmup):
            for name in names[i % len(names) :] + names[: i % len(names)]:
                replays[name]()
        torch.cuda.synchronize()
        samples = {name: [] for name in names}
        for sample in range(args.samples):
            rotation = sample % len(names)
            order = names[rotation:] + names[:rotation]
            if (sample // len(names)) % 2:
                order.reverse()
            for name in order:
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.calls):
                    replays[name]()
                end.record()
                end.synchronize()
                samples[name].append(begin.elapsed_time(end) * 1000.0 / args.calls)
        return samples

    def profile_cuda(fn):
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as prof:
            for _ in range(args.profile_calls):
                fn()
            torch.cuda.synchronize()
        by_name = {}
        for event in prof.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                by_name.setdefault(event.name, []).append(event.time_range.elapsed_us())
        return {
            "per_call_us": sum(sum(values) for values in by_name.values()) / args.profile_calls,
            "per_kernel": {name: {"median_us": statistics.median(values), "count": len(values)} for name, values in by_name.items()},
        }

    dtype = torch.bfloat16
    selected_default = type(init_nvfp4_linear_kernel()).__name__
    if selected_default != "FlashInferCutlassNvFp4LinearKernel":
        raise RuntimeError(f"vLLM selected {selected_default}; this comparison requires its FlashInfer CUTLASS default")
    torch.manual_seed(0)
    x = torch.randn(args.m, args.k, device="cuda", dtype=dtype)
    w = torch.randn(args.n, args.k, device="cuda", dtype=dtype)
    one = torch.ones((), device="cuda", dtype=torch.float32)
    alpha = torch.ones(1, device="cuda", dtype=torch.float32)
    b, scale_b = scaled_fp4_quant(w, one, is_sf_swizzled_layout=False)
    blocked_scale_b = swizzle_blockscale(scale_b)
    b_pad, pad_cols = pad_nvfp4_weight_for_cutlass(b)
    outputs = {}
    graphs = {}
    for quant_backend, mm_backend in (("flashinfer-cutlass", "cutlass"), ("b12x", "b12x")):

        def quant(backend=quant_backend):
            a, sa = scaled_fp4_quant(
                x,
                one,
                is_sf_swizzled_layout=True,
                backend=backend,
                padded_n=args.k + pad_cols * 2,
            )
            return pad_nvfp4_activation_for_cutlass(a, pad_cols), sa

        def dynamic_call(quant=quant, backend=mm_backend):
            a, sa = quant()
            return flashinfer_scaled_fp4_mm(a, b_pad, sa, blocked_scale_b, alpha, dtype, backend=backend)

        outputs[mm_backend] = dynamic_call().detach().clone()
        graphs[mm_backend] = capture_one(dynamic_call)

    emmy_graph = Graph.from_dict(json.loads(args.emmy_ir.read_text()))
    sources = {
        "l0.input_scale": np.asarray(1.0, dtype=np.float32),
        "l0.weight": b.detach().cpu().numpy(),
        "l0.weight_scale": scale_b.detach().cpu().view(torch.uint8).numpy(),
        "l0.weight_scale_2": np.asarray([1.0], dtype=np.float32),
    }
    emmy_inputs = bind_constants(emmy_graph, sources)
    emmy_inputs["input"] = x.detach().cpu().contiguous().view(torch.uint16).numpy()
    emmy_graph = CudaBackend().compile(emmy_graph)
    with gpu_lock():
        program = CompiledProgram.build(emmy_graph, input_data=emmy_inputs)
        program.capture_program_graph()
        program.replay_program_graph()
        torch.cuda.synchronize()
        emmy_raw = program.outputs()[emmy_graph.outputs[0]]
        emmy_out = torch.from_numpy(emmy_raw.view(np.uint16)).view(torch.bfloat16).float()

        replay = {
            "emmy": program.replay_program_graph,
            "flashinfer_cutlass": graphs["cutlass"][0].replay,
            "flashinfer_b12x": graphs["b12x"][0].replay,
        }
        samples = time_interleaved(replay)
        profiles = {name: profile_cuda(fn) for name, fn in replay.items()} if args.profile_calls else None

    reference = outputs["cutlass"].float().cpu()
    report = {
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        "flashinfer": importlib.metadata.version("flashinfer-python"),
        "references": {
            "vllm_selected_default": selected_default,
            "faster_second_choice": "FlashInferB12xNvFp4LinearKernel",
        },
        "pad_cols": pad_cols,
        "protocol": {
            "warmup_rounds": args.warmup,
            "samples": args.samples,
            "graph_replays_per_sample": args.calls,
            "captured_calls_per_graph": 1,
            "interleaved": True,
            "timed_operation": "activation_quantize_and_matmul",
            "weight_quantization_timed": False,
        },
        "timing_us": {
            name: {
                "median": statistics.median(values),
                "min": min(values),
                "max": max(values),
                "samples": values,
            }
            for name, values in samples.items()
        },
        "correctness": {
            "flashinfer_b12x_vs_cutlass_max_abs": (outputs["b12x"].float() - outputs["cutlass"].float()).abs().max().item(),
            "flashinfer_b12x_vs_cutlass_exact_fraction": (outputs["b12x"] == outputs["cutlass"]).float().mean().item(),
            "emmy_vs_cutlass_max_abs": (emmy_out - reference).abs().max().item(),
            "emmy_vs_cutlass_exact_fraction": (emmy_out == reference).float().mean().item(),
        },
    }
    if profiles is not None:
        report["profiles"] = profiles
    rendered = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
