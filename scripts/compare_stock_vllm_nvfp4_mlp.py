#!/usr/bin/env python3
"""Compare one Emmy MLP with stock vLLM's ModelOpt NVFP4 MLP on one checkpoint.

Requires vLLM 0.23, a CUDA device, and a local snapshot of
Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462.
Only the chosen layer's twelve MLP tensors are read. No attention or GDN runs.

    python scripts/compare_stock_vllm_nvfp4_mlp.py --checkpoint /path/to/snapshot --layer 3
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
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


def emmy_mlp(directory: Path, prefix: str, layer: int, hidden: int, intermediate: int):
    """Compile only the selected layer with the same path as MLPPrograms."""
    import numpy as np
    import torch

    from emmy.compiler.backend.cuda.program import BufferArena
    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.serving.gen_runner import _compile_split
    from emmy.serving.mlp import logical_mlp, parameter_keys

    module = logical_mlp(hidden, intermediate, torch.bfloat16)
    ckpt = (str(directory), parameter_keys(module, layer, prefix=prefix))
    arena, cache, constants = BufferArena(), PlanTemplateCache(), {}
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
        [torch.zeros(1, hidden, dtype=torch.bfloat16)],
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
    parser.add_argument("--atol", type=float, default=0.05, help="provisional diagnostic threshold, not a qualified error bound")
    parser.add_argument("--rtol", type=float, default=0.05, help="provisional diagnostic threshold, not a qualified error bound")
    parser.add_argument("--emmy-knobs", default=config.knobs_aggregate(), help="explicit Emmy compile pins, including FAST_MATH=0")
    args = parser.parse_args()

    entries = [item.strip() for item in (args.emmy_knobs or "").split(",") if item.strip()]
    if any("=" not in item for item in entries):
        parser.error("--emmy-knobs must be comma-separated KEY=VALUE pins")
    pins = dict(item.split("=", 1) for item in entries)
    if pins.get("FAST_MATH", "").lower() not in {"0", "false"} or len(pins) < 2:
        parser.error("--emmy-knobs must include FAST_MATH=0/false and at least one explicit schedule/placement pin")
    import torch
    from transformers import AutoConfig
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.distributed import ensure_model_parallel_initialized, init_distributed_environment

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
    torch.cuda.set_device(0)
    with tempfile.TemporaryDirectory() as temporary, set_current_vllm_config(VllmConfig()):
        init_distributed_environment(world_size=1, rank=0, local_rank=0, distributed_init_method=f"file://{temporary}/init", backend="nccl")
        ensure_model_parallel_initialized(1, 1)
        stock, tensors = stock_mlp(directory, prefix, args.layer, hidden, intermediate)
    with pinned_knobs(pins):
        one, symbolic, _arena = emmy_mlp(directory, prefix, args.layer, hidden, intermediate)
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
        "atol": args.atol,
        "rtol": args.rtol,
        "rows": {},
    }
    generator = torch.Generator(device="cpu").manual_seed(20260930)
    failures = []
    with torch.inference_mode():
        for rows in (1, 2, 64):
            x = torch.randn((rows, hidden), generator=generator).to(device="cuda", dtype=torch.bfloat16)
            expected = stock(x)
            actual = one.run_device([x])[0] if rows == 1 else symbolic.run_device_sym([x])[0]
            torch.cuda.synchronize()
            report["rows"][rows] = error_metrics(actual, expected, atol=args.atol, rtol=args.rtol)
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


if __name__ == "__main__":
    main()
