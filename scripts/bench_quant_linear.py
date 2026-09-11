#!/usr/bin/env python3
"""Fairly compare complete dynamically quantized linears on one shape.

Each backend captures exactly one BF16 -> quantize -> matmul call in a CUDA
graph.  Measurement windows interleave repeated graph replays, matching
Emmy's whole-program benchmark semantics without amortizing one giant graph
over many logical calls.

``--format nvfp4`` times vLLM's selected FlashInfer CUTLASS route and the
FlashInfer B12x route. ``--format fp8-block`` (e4m3 weights under one scale
per 128x128 block, per-token-group-of-128 dynamic e4m3 activations) times
vLLM's selected CUTLASS route and its Triton route, which quantize the
activation the same way, plus the Marlin and Humming weight-only kernels,
which skip the activation quantization.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import statistics
from pathlib import Path


def _nvfp4(args, x, w):
    """vLLM's NVFP4 references, the weight bytes Emmy binds, and the report's reference facts."""
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

    selected_default = type(init_nvfp4_linear_kernel()).__name__
    if selected_default != "FlashInferCutlassNvFp4LinearKernel":
        raise RuntimeError(f"vLLM selected {selected_default}; this comparison requires its FlashInfer CUTLASS default")
    one = torch.ones((), device="cuda", dtype=torch.float32)
    alpha = torch.ones(1, device="cuda", dtype=torch.float32)
    b, scale_b = scaled_fp4_quant(w, one, is_sf_swizzled_layout=False)
    blocked_scale_b = swizzle_blockscale(scale_b)
    b_pad, pad_cols = pad_nvfp4_weight_for_cutlass(b)
    calls = {}
    for quant_backend, mm_backend, name in (("flashinfer-cutlass", "cutlass", "flashinfer_cutlass"), ("b12x", "b12x", "flashinfer_b12x")):

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
            return flashinfer_scaled_fp4_mm(a, b_pad, sa, blocked_scale_b, alpha, torch.bfloat16, backend=backend)

        calls[name] = dynamic_call
    sources = {
        "l0.input_scale": np.asarray(1.0, dtype=np.float32),
        "l0.weight": b.detach().cpu().numpy(),
        "l0.weight_scale": scale_b.detach().cpu().view(torch.uint8).numpy(),
        "l0.weight_scale_2": np.asarray([1.0], dtype=np.float32),
    }
    facts = {
        "references": {"vllm_selected_default": selected_default, "faster_second_choice": "FlashInferB12xNvFp4LinearKernel"},
        "pad_cols": pad_cols,
        "flashinfer": importlib.metadata.version("flashinfer-python"),
    }
    return calls, sources, facts, "flashinfer_cutlass"


def _fp8_block(args, x, w):
    """vLLM's block-scaled FP8 references, the weight bytes Emmy binds, and the report's facts.

    The weight is quantized once, here, and every implementation reads the same bytes and scales.
    The references are built through vLLM's own kernel classes, so each applies the weight layout
    processing its production path applies."""
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.kernels.linear import init_fp8_linear_kernel
    from vllm.model_executor.kernels.linear.scaled_mm.cutlass import CutlassFp8BlockScaledMMKernel
    from vllm.model_executor.kernels.linear.scaled_mm.humming import HummingFP8ScaledMMLinearKernel
    from vllm.model_executor.kernels.linear.scaled_mm.marlin import MarlinFP8ScaledMMLinearKernel
    from vllm.model_executor.kernels.linear.scaled_mm.triton import TritonFp8BlockScaledMMKernel
    from vllm.model_executor.layers.quantization.utils.quant_utils import GroupShape, create_fp8_quant_key

    n, k = w.shape
    blocks = w.float().view(n // 128, 128, k // 128, 128)
    scale = blocks.abs().amax(dim=(1, 3)).clamp_min(1e-12) / 448.0
    bits = (blocks / scale[:, None, :, None]).to(torch.float8_e4m3fn).view(n, k)
    act_key = create_fp8_quant_key(static=False, group_shape=GroupShape(1, 128))
    weight_key = create_fp8_quant_key(static=True, group_shape=GroupShape(128, 128))

    def layer():
        mod = torch.nn.Module()
        mod.weight = torch.nn.Parameter(bits.clone(), requires_grad=False)
        mod.weight_scale_inv = torch.nn.Parameter(scale.clone(), requires_grad=False)
        mod.input_scale = mod.input_scale_ub = mod.bias = None
        mod.has_bias = False
        mod.weight_block_size = [128, 128]
        mod.orig_dtype = mod.params_dtype = torch.bfloat16
        mod.input_size_per_partition = mod.input_size = k
        mod.output_size_per_partition = mod.output_size = n
        mod.logical_widths = mod.output_partition_sizes = [n]
        return mod

    calls = {}
    with set_current_vllm_config(VllmConfig()):
        selected_default = type(init_fp8_linear_kernel(act_key, weight_key, torch.bfloat16, torch.bfloat16, (n, k))).__name__
        if selected_default != "CutlassFp8BlockScaledMMKernel":
            raise RuntimeError(f"vLLM selected {selected_default}; this comparison requires its CUTLASS default")
        for name, cls in (
            ("vllm_cutlass", CutlassFp8BlockScaledMMKernel),
            ("vllm_triton", TritonFp8BlockScaledMMKernel),
            ("vllm_marlin_w8a16", MarlinFP8ScaledMMLinearKernel),
            ("vllm_humming_w8a16", HummingFP8ScaledMMLinearKernel),
        ):
            kernel = init_fp8_linear_kernel(act_key, weight_key, torch.bfloat16, torch.bfloat16, (n, k), force_kernel=cls)
            if type(kernel) is not cls:
                raise RuntimeError(f"{cls.__name__} cannot implement this layer here")
            mod = layer()
            kernel.process_weights_after_loading(mod)
            calls[name] = lambda kernel=kernel, mod=mod: kernel.apply_weights(mod, x)
    sources = {"l0.weight": bits.view(torch.uint8).cpu().numpy(), "l0.weight_scale_inv": scale.cpu().numpy()}
    facts = {
        "references": {
            "vllm_selected_default": selected_default,
            "same_activation_quantization": ["vllm_cutlass", "vllm_triton"],
            "weight_only_no_activation_quantization": ["vllm_marlin_w8a16", "vllm_humming_w8a16"],
        }
    }
    return calls, sources, facts, "vllm_cutlass"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=("nvfp4", "fp8-block"), default="nvfp4")
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

    torch.manual_seed(0)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    w = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    calls, sources, facts, reference_name = (_nvfp4 if args.format == "nvfp4" else _fp8_block)(args, x, w)
    outputs, graphs = {}, {}
    for name, call in calls.items():
        outputs[name] = call().detach().clone()
        graphs[name] = capture_one(call)

    emmy_graph = Graph.from_dict(json.loads(args.emmy_ir.read_text()))
    emmy_inputs = bind_constants(emmy_graph, sources)
    emmy_inputs["input"] = x.detach().cpu().contiguous().view(torch.uint16).numpy()
    emmy_graph = CudaBackend().compile(emmy_graph)
    with gpu_lock():
        program = CompiledProgram.build(emmy_graph, input_data=emmy_inputs)
        program.capture_program_graph()
        program.replay_program_graph()
        torch.cuda.synchronize()
        emmy_raw = program.outputs()[emmy_graph.outputs[0]]
        emmy_out = torch.from_numpy(emmy_raw.view(np.uint16)).view(torch.bfloat16).float().reshape(args.m, args.n)

        replay = {"emmy": program.replay_program_graph, **{name: graph.replay for name, (graph, _) in graphs.items()}}
        samples = time_interleaved(replay)
        profiles = {name: profile_cuda(fn) for name, fn in replay.items()} if args.profile_calls else None

    reference = outputs[reference_name].float().cpu()
    denom = reference.abs().max().item()
    correctness = {}
    for name, out in (("emmy", emmy_out), *((name, o.float().cpu()) for name, o in outputs.items() if name != reference_name)):
        correctness[f"{name}_vs_{reference_name}_max_abs"] = (out - reference).abs().max().item()
        correctness[f"{name}_vs_{reference_name}_max_rel"] = (out - reference).abs().max().item() / denom
        correctness[f"{name}_vs_{reference_name}_exact_fraction"] = (out == reference).float().mean().item()
    report = {
        "format": args.format,
        "shape": {"m": args.m, "k": args.k, "n": args.n},
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": importlib.metadata.version("vllm"),
        **facts,
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
        "correctness": correctness,
    }
    if profiles is not None:
        report["profiles"] = profiles
    rendered = json.dumps(report, indent=2)
    if args.json:
        args.json.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
