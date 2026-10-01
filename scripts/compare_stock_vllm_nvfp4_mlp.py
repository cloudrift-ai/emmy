#!/usr/bin/env python3
"""Compare one Emmy MLP with stock vLLM's ModelOpt NVFP4 MLP on one checkpoint.

Requires vLLM 0.23, a CUDA device, and a local snapshot of
Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462.
Only the chosen layer's twelve MLP tensors are read. No attention or GDN runs.

    python scripts/compare_stock_vllm_nvfp4_mlp.py --checkpoint /path/to/snapshot --layer 3
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import sys
import tempfile
from pathlib import Path

from emmy import config

MODEL = "Inferact/Qwen3.8-27B-NVFP4"
REVISION = "6128240ebaf4eaa7bad2b3d1c72c37d677c5f462"
LEAVES = ("weight", "weight_scale", "weight_scale_2", "input_scale")


def layer_keys(prefix: str, layer: int) -> dict[str, tuple[str, int | None]]:
    """Map checkpoint leaves to stock vLLM's merged gate/up or down parameters."""
    base = f"{prefix}.layers.{layer}.mlp"
    return {
        f"{base}.{projection}.{leaf}": ("down_proj", None) if projection == "down_proj" else ("gate_up_proj", shard)
        for projection, shard in (("gate_proj", 0), ("up_proj", 1), ("down_proj", None))
        for leaf in LEAVES
    }


def checkpoint_tensors(directory: Path, keys: set[str]):
    """Read just this layer from safetensors, keeping FP8 scales in their stored dtype."""
    from safetensors import safe_open

    from emmy.compiler.loader.safetensors import _build_index

    index = _build_index(directory)
    missing = keys - index.keys()
    if missing:
        raise ValueError(f"missing MLP checkpoint tensors: {sorted(missing)}")
    shards: dict[Path, list[str]] = {}
    for key in keys:
        shards.setdefault(index[key], []).append(key)
    tensors = {}
    for shard, names in shards.items():
        with safe_open(shard, framework="pt", device="cpu") as reader:
            for name in names:
                tensors[name] = reader.get_tensor(name)
    return tensors


def load_mlp_leaves(module, mapping, tensors) -> None:
    """Use each stock parameter's loader, including the merged gate/up shard id."""
    for key, (linear_name, shard) in mapping.items():
        linear = getattr(module, linear_name)
        param = getattr(linear, key.rsplit(".", 1)[-1])
        if shard is None:
            param.weight_loader(param, tensors[key])
        else:
            param.weight_loader(param, tensors[key], shard)


def error_metrics(actual, expected, *, atol: float, rtol: float) -> dict:
    """Measure magnitude and BF16 output-rounding distance, including near-zero outliers."""
    import torch

    reference = expected.float()
    delta = (actual.float() - reference).abs()
    n = delta.numel()
    flat = delta.flatten()
    quantiles = torch.quantile(flat, torch.tensor([0.5, 0.9, 0.99, 0.999], device=flat.device))

    def ordered_bits(value):
        bits = value.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
        return torch.where((bits & 0x8000) != 0, 0xFFFF - bits, bits + 0x8000)

    ulps = (ordered_bits(actual) - ordered_bits(expected)).abs()
    bound = atol + rtol * reference.abs()
    return {
        "max_abs": delta.max().item(),
        "mean_abs": delta.mean().item(),
        "rms_abs": torch.sqrt(torch.mean(delta.square())).item(),
        "rms_relative": torch.sqrt(torch.sum(delta.square()) / torch.sum(reference.square())).item(),
        "abs_percentiles": dict(zip(("p50", "p90", "p99", "p99_9"), quantiles.tolist(), strict=True)),
        "max_ulp": ulps.max().item(),
        "p99_ulp": torch.quantile(ulps.float(), 0.99).item(),
        "ulp_counts": {str(distance): torch.count_nonzero(ulps == distance).item() for distance in range(4)}
        | {"4_or_more": torch.count_nonzero(ulps >= 4).item()},
        "outside_tolerance": torch.count_nonzero(delta > bound).item(),
        "outside_fraction": (torch.count_nonzero(delta > bound) / n).item(),
        "reference_max_abs": reference.abs().max().item(),
        "nonzero_reference": torch.count_nonzero(expected).item(),
        "nonzero_emmy": torch.count_nonzero(actual).item(),
    }


def activation_probe(intermediate, stem, stock_codes, stock_scales, width, global_factor) -> dict:
    """Compare producer-launch FP4 carriers and their raw-F32 reconstruction."""
    import torch

    from emmy.compiler.dtype import decode_f4x2, decode_f8

    report = {}
    for suffix, stock_tensor in (("bits", stock_codes), ("scale_bits", stock_scales)):
        # A padded static program materializes every row. The stock producer sees
        # only the active prefix, whose carriers occupy the first bytes.
        emmy_tensor = intermediate[f"{stem}_{suffix}"].contiguous().view(torch.uint8).flatten()
        stock_tensor = stock_tensor.contiguous().view(torch.uint8).flatten()
        if emmy_tensor.numel() < stock_tensor.numel():
            raise AssertionError(f"{stem}_{suffix}: byte counts differ, Emmy {emmy_tensor.numel()}, stock {stock_tensor.numel()}")
        emmy_tensor = emmy_tensor[: stock_tensor.numel()]
        mismatches = emmy_tensor != stock_tensor
        report[suffix] = {
            "bytes": stock_tensor.numel(),
            "different": torch.count_nonzero(mismatches).item(),
            "first_differences": [
                (int(index), int(emmy_tensor[index]), int(stock_tensor[index]))
                for index in torch.nonzero(mismatches).flatten()[:8].tolist()
            ],
        }

    def reconstructed(codes, scales):
        packed = codes.contiguous().view(torch.uint8).flatten()[: width // 2].cpu().numpy().reshape(1, width // 2)
        sf_bits = scales.contiguous().view(torch.uint8).flatten()[: width // 16].cpu().numpy().reshape(1, width // 16)
        fp4 = decode_f4x2(packed).reshape(1, width // 16, 16)
        sf = decode_f8(sf_bits, "f8e4m3").reshape(1, width // 16, 1)
        return (fp4 * sf * global_factor).reshape(1, width)

    stock_x = reconstructed(stock_codes, stock_scales)
    emmy_x = reconstructed(intermediate[f"{stem}_bits"], intermediate[f"{stem}_scale_bits"])
    report["reconstruction_rms_relative"] = float(((emmy_x - stock_x) ** 2).sum() ** 0.5 / (stock_x**2).sum() ** 0.5)
    return report


def stock_mlp(directory: Path, prefix: str, layer: int, hidden: int, intermediate: int):
    """Construct stock Qwen3NextMLP, then use its own shard loaders and quant method."""
    import torch
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.quantization.modelopt import ModelOptNvFp4Config
    from vllm.model_executor.models.qwen3_next import Qwen3NextMLP
    from vllm.utils.torch_utils import set_default_torch_dtype

    mapping = layer_keys(prefix, layer)
    tensors = checkpoint_tensors(directory, set(mapping))
    quant = ModelOptNvFp4Config(quant_method="NVFP4", is_checkpoint_nvfp4_serialized=True, group_size=16)
    with set_current_vllm_config(VllmConfig()), set_default_torch_dtype(torch.bfloat16), torch.device("cuda"):
        module = Qwen3NextMLP(hidden, intermediate, "silu", quant_config=quant, prefix=f"{prefix}.layers.{layer}.mlp")
    for linear in (module.gate_up_proj, module.down_proj):
        if linear.params_dtype != torch.bfloat16 or type(linear.quant_method).__name__ != "ModelOptNvFp4LinearMethod":
            raise AssertionError(f"stock MLP linear differs from BF16 ModelOpt W4A4: {linear.params_dtype}, {type(linear.quant_method)}")
    gpu_tensors = {key: tensor.cuda() for key, tensor in tensors.items()}
    load_mlp_leaves(module, mapping, gpu_tensors)
    for linear in (module.gate_up_proj, module.down_proj):
        linear.quant_method.process_weights_after_loading(linear)
    return module.eval(), tensors


def emmy_mlp(
    directory: Path,
    prefix: str,
    layer: int,
    hidden: int,
    intermediate: int,
    *,
    static_only: bool = False,
    static_rows: int = 1,
    projection_only: bool = False,
):
    """Compile only the selected layer with the same path as MLPPrograms."""
    from types import MethodType

    import numpy as np
    import torch

    from emmy.compiler.backend.cuda.program import BufferArena
    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.serving.gen_runner import _compile_split
    from emmy.serving.mlp import logical_mlp, parameter_keys

    module = logical_mlp(hidden, intermediate, torch.bfloat16)
    if projection_only:
        del module.down_proj

        def projections(self, x):
            return self.gate_proj(x), self.up_proj(x)

        module.forward = MethodType(projections, module)
    ckpt = (str(directory), parameter_keys(module, layer, prefix=prefix))
    arena, cache, constants = BufferArena(), PlanTemplateCache(), {}
    symbolic = None
    if not static_only:
        symbolic, _ = _compile_split(
            module,
            [torch.zeros(8, hidden, dtype=torch.bfloat16)],
            ["x"],
            np.dtype("float32"),
            dev_consts=constants,
            arena=arena,
            capacity=64,
            ckpt=ckpt,
            plan_cache=cache,
        )
    one, _ = _compile_split(
        module,
        [torch.zeros(static_rows, hidden, dtype=torch.bfloat16)],
        None,
        np.dtype("float32"),
        dev_consts=constants,
        arena=arena,
        ckpt=ckpt,
        plan_cache=cache,
    )
    return one, symbolic, arena


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="local snapshot of the pinned ModelOpt NVFP4 checkpoint")
    parser.add_argument("--layer", type=int, default=3)
    parser.add_argument("--static-only", action="store_true", help="diagnose decode M=1 without compiling the symbolic prefill graph")
    parser.add_argument("--static-rows", type=int, default=1, help="static Emmy program rows; probe padded decode with 16")
    parser.add_argument("--rows", type=int, default=1, help="input rows for a static-only stock MLP comparison")
    parser.add_argument("--compile-only", action="store_true", help="inspect Emmy's selected kernels without constructing stock vLLM")
    parser.add_argument("--quant-probe", action="store_true", help="compare stock and Emmy activation FP4 bytes for M=1")
    parser.add_argument("--projection-probe", action="store_true", help="compare the independent gate/up projection outputs")
    parser.add_argument("--force-stock-quant", action="store_true", help="feed stock FP4 bytes into the Emmy projection diagnostic")
    parser.add_argument("--padding-probe", action="store_true", help="check stale M=16 rows cannot affect a later M=1 decode")
    parser.add_argument("--atol", type=float, default=0.05, help="provisional diagnostic threshold, not a qualified error bound")
    parser.add_argument("--rtol", type=float, default=0.05, help="provisional diagnostic threshold, not a qualified error bound")
    parser.add_argument("--emmy-knobs", default=config.knobs_aggregate(), help="explicit Emmy compile pins, including FAST_MATH=true/false")
    args = parser.parse_args()

    entries = [item.strip() for item in (args.emmy_knobs or "").split(",") if item.strip()]
    if any("=" not in item for item in entries):
        parser.error("--emmy-knobs must be comma-separated KEY=VALUE pins")
    pins = dict(item.split("=", 1) for item in entries)
    if pins.get("FAST_MATH", "").lower() not in {"0", "false", "1", "true"} or len(pins) < 2:
        parser.error("--emmy-knobs must include FAST_MATH=true/false and at least one explicit schedule/placement pin")
    if not 1 <= args.static_rows <= 64 or (args.static_rows != 1 and not args.static_only):
        parser.error("--static-rows requires --static-only and a row count between 1 and 64")
    if args.static_only and not 1 <= args.rows <= args.static_rows:
        parser.error("--rows must be between 1 and --static-rows")
    if args.projection_probe and (not args.static_only or args.quant_probe):
        parser.error("--projection-probe requires --static-only and cannot combine with --quant-probe")
    if args.force_stock_quant and not args.projection_probe:
        parser.error("--force-stock-quant requires --projection-probe")
    if args.padding_probe and (not args.static_only or args.static_rows == 1 or args.projection_probe):
        parser.error("--padding-probe requires a full MLP, --static-only and --static-rows greater than one")
    import torch
    from transformers import AutoConfig
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import (
        destroy_distributed_environment,
        destroy_model_parallel,
        ensure_model_parallel_initialized,
        init_distributed_environment,
    )

    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.serving.mlp import text_prefix

    version = importlib.metadata.version("vllm")
    if not version.startswith("0.23."):
        raise RuntimeError(f"this oracle pins vLLM 0.23, found {version}")
    if not torch.cuda.is_available():
        raise RuntimeError("this oracle requires CUDA")
    directory = args.checkpoint.resolve()
    snapshot_verified = directory.parent.name == "snapshots" and directory.name == REVISION
    if directory.parent.name == "snapshots" and not snapshot_verified:
        raise ValueError(f"expected {MODEL}@{REVISION}, got snapshot {directory.name}")
    hf_config = AutoConfig.from_pretrained(directory)
    text = hf_config.text_config
    quantization = getattr(hf_config, "quantization_config", None) or getattr(text, "quantization_config", None) or {}
    if quantization.get("quant_algo") != "NVFP4":
        raise ValueError(f"expected ModelOpt NVFP4 checkpoint, got {quantization}")
    if getattr(text, "model_type", None) != "qwen3_5_text" or not 0 <= args.layer < text.num_hidden_layers:
        raise ValueError("expected a Qwen3.5 dense text layer in range")
    prefix = text_prefix(directory)
    hidden, intermediate = text.hidden_size, text.intermediate_size
    if args.compile_only:
        with pinned_knobs(pins):
            one, _, _arena = emmy_mlp(
                directory,
                prefix,
                args.layer,
                hidden,
                intermediate,
                static_only=args.static_only,
                static_rows=args.static_rows,
                projection_only=args.projection_probe,
            )
        print(json.dumps({"static_rows": args.static_rows, "kernels": len(one.program.plan.kernels), "emmy_knobs": pins}, indent=2))
        return
    torch.cuda.set_device(0)

    def cleanup_distributed():
        destroy_model_parallel()
        destroy_distributed_environment()

    with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{temporary}/init", backend="gloo")
        ensure_model_parallel_initialized(1, 1, backend="gloo")
        try:
            stock, tensors = stock_mlp(directory, prefix, args.layer, hidden, intermediate)
            with pinned_knobs(pins):
                print("Emmy compile start", file=sys.stderr, flush=True)
                one, symbolic, _arena = emmy_mlp(
                    directory,
                    prefix,
                    args.layer,
                    hidden,
                    intermediate,
                    static_only=args.static_only,
                    static_rows=args.static_rows,
                    projection_only=args.projection_probe,
                )
                print("Emmy compile done", file=sys.stderr, flush=True)
            torch.cuda.synchronize()
            report = {
                "checkpoint": str(directory),
                "expected_revision": REVISION,
                "snapshot_revision_verified": snapshot_verified,
                "layer": args.layer,
                "vllm": version,
                "stock_kernel": type(stock.gate_up_proj.quant_method.kernel).__name__,
                "stock_linear_dtype": str(stock.gate_up_proj.params_dtype),
                "emmy_knobs": pins,
                "emmy_static_rows": args.static_rows,
                "atol": args.atol,
                "rtol": args.rtol,
                "rows": {},
            }
            generator = torch.Generator(device="cpu").manual_seed(20260930)
            failures = []
            if args.padding_probe:
                first = torch.randn((1, hidden), generator=generator).to(device="cuda", dtype=torch.bfloat16)
                stale = torch.randn((args.static_rows, hidden), generator=generator).to(device="cuda", dtype=torch.bfloat16)
                one.run_device([stale])
                after_stale = one.run_device([first])[0]
                zero_padded = torch.cat((first, torch.zeros_like(stale[1:])), dim=0)
                from_zero = one.run_device([zero_padded])[0][:1]
                report["padding_prefix_equal"] = bool(torch.equal(after_stale, from_zero))
                if not report["padding_prefix_equal"]:
                    failures.append("stale M=16 rows changed the next M=1 output")
            with torch.inference_mode():
                for rows in (args.rows,) if args.static_only else (1, 2, 64):
                    x = torch.randn((rows, hidden), generator=generator).to(device="cuda", dtype=torch.bfloat16)
                    print(f"rows={rows}: stock forward start", file=sys.stderr, flush=True)
                    if args.projection_probe:
                        merged, _bias = stock.gate_up_proj(x)
                        expected = torch.chunk(merged, 2, dim=-1)
                    else:
                        expected = stock(x)
                    print(f"rows={rows}: Emmy forward start", file=sys.stderr, flush=True)
                    snapshots = {}
                    if rows == 1 and args.quant_probe:
                        from emmy.compiler.backend.gpu_lock import gpu_lock

                        target = {
                            "x_static_fp4_bits",
                            "x_static_fp4_scale_bits",
                            "mul_static_fp4_bits",
                            "mul_static_fp4_scale_bits",
                        }

                        def after_launch(_index, launch, names=target, captured=snapshots):
                            for name in names.intersection(launch.writes):
                                captured[name] = one.program.buffer_view(name).clone()

                        with gpu_lock(), one.program.on_stream(torch.cuda.current_stream()):
                            one.program.upload_prefix_device({"x": x})
                            one.program.iter_once(per_launch_hook=after_launch)
                        actual = one.program.output_prefix_device()[one.output_names[0]][:rows].clone()
                        if set(snapshots) != target:
                            raise AssertionError(f"activation buffers were not materialized by a launch: {set(snapshots)}, {target}")
                    elif rows == 1 and args.projection_probe:
                        from emmy.compiler.backend.gpu_lock import gpu_lock

                        stock_bytes = {}
                        if args.force_stock_quant:
                            from vllm._custom_ops import scaled_fp4_quant

                            codes, scales = scaled_fp4_quant(
                                x,
                                stock.gate_up_proj.input_global_scale_inv,
                                is_sf_swizzled_layout=False,
                                backend="flashinfer-cutlass",
                                padded_n=hidden,
                            )
                            stock_bytes = {
                                "x_static_fp4_bits": codes.contiguous().view(torch.uint8).flatten(),
                                "x_static_fp4_scale_bits": scales.contiguous().view(torch.uint8).flatten(),
                            }

                        def replace_quant(_index, launch, replacements=stock_bytes):
                            for name in replacements.keys() & launch.writes:
                                one.program.buffer_view(name).view(torch.uint8).flatten().copy_(replacements[name])

                        with gpu_lock(), one.program.on_stream(torch.cuda.current_stream()):
                            one.program.upload_prefix_device({"x": x})
                            one.program.iter_once(per_launch_hook=replace_quant if stock_bytes else None)
                            views = one.program.output_prefix_device()
                            actual = tuple(views[name][:rows].clone() for name in one.output_names)
                    else:
                        outputs = one.run_device([x]) if args.static_only or rows == 1 else symbolic.run_device_sym([x])
                        actual = tuple(outputs) if args.projection_probe else outputs[0]
                    torch.cuda.synchronize()
                    if rows == 1 and args.quant_probe:
                        from vllm._custom_ops import scaled_fp4_quant

                        stock_codes, stock_scales = scaled_fp4_quant(
                            x,
                            stock.gate_up_proj.input_global_scale_inv,
                            is_sf_swizzled_layout=False,
                            backend="flashinfer-cutlass",
                            padded_n=hidden,
                        )
                        scale_key = f"{prefix}.layers.{args.layer}.mlp.gate_proj.input_scale"
                        report["activation_quantization"] = activation_probe(
                            snapshots,
                            "x_static_fp4",
                            stock_codes,
                            stock_scales,
                            hidden,
                            float(tensors[scale_key].reshape(-1)[0]),
                        )
                        merged, _bias = stock.gate_up_proj(x)
                        stock_z = stock.act_fn(merged)
                        down_codes, down_scales = scaled_fp4_quant(
                            stock_z,
                            stock.down_proj.input_global_scale_inv,
                            is_sf_swizzled_layout=False,
                            backend="flashinfer-cutlass",
                            padded_n=intermediate,
                        )
                        scale_key = f"{prefix}.layers.{args.layer}.mlp.down_proj.input_scale"
                        report["down_activation_quantization"] = activation_probe(
                            snapshots,
                            "mul_static_fp4",
                            down_codes,
                            down_scales,
                            intermediate,
                            float(tensors[scale_key].reshape(-1)[0]),
                        )
                    print(f"rows={rows}: comparisons start", file=sys.stderr, flush=True)
                    report["rows"][rows] = (
                        {
                            name: error_metrics(value, ref, atol=args.atol, rtol=args.rtol)
                            for name, value, ref in zip(("gate", "up"), actual, expected, strict=True)
                        }
                        if args.projection_probe
                        else error_metrics(actual, expected, atol=args.atol, rtol=args.rtol)
                    )
                    if not args.projection_probe:
                        bits = actual.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
                        report["rows"][rows]["emmy_sha256"] = hashlib.sha256(bits).hexdigest()
                    if args.projection_probe:
                        for name, value, ref in zip(("gate", "up"), actual, expected, strict=True):
                            if not torch.isfinite(value).all() or not torch.isfinite(ref).all():
                                failures.append(f"rows={rows} {name}: nonfinite projection output")
                            else:
                                try:
                                    torch.testing.assert_close(value, ref, atol=args.atol, rtol=args.rtol)
                                except AssertionError as exc:
                                    failures.append(f"rows={rows} {name}: {exc}")
                        continue
                    if not torch.isfinite(expected).all() or not torch.isfinite(actual).all():
                        failures.append(f"rows={rows}: nonfinite MLP output")
                    elif not torch.count_nonzero(expected) or not torch.count_nonzero(actual):
                        failures.append(f"rows={rows}: zero MLP output")
                    else:
                        try:
                            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)
                        except AssertionError as exc:
                            failures.append(f"rows={rows}: {exc}")
            gate = f"{prefix}.layers.{args.layer}.mlp.gate_proj.input_scale"
            up = f"{prefix}.layers.{args.layer}.mlp.up_proj.input_scale"
            report["gate_up_input_scales_equal"] = bool(torch.equal(tensors[gate], tensors[up]))
            gate = f"{prefix}.layers.{args.layer}.mlp.gate_proj.weight_scale_2"
            up = f"{prefix}.layers.{args.layer}.mlp.up_proj.weight_scale_2"
            report["gate_up_weight_scales_equal"] = bool(torch.equal(tensors[gate], tensors[up]))
            print(json.dumps(report, indent=2))
            if failures:
                raise AssertionError("\n".join(failures))
        finally:
            cleanup_distributed()


if __name__ == "__main__":
    main()
