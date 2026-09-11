"""Quantize a TRACED module's weights into a real checkpoint, and repoint its graph at it.

The spellers (:mod:`emmy.compiler.loader.quant`) turn a checkpoint's stored tensors into graph
algebra at model-read time, and everything downstream — recognition, staging, the block-scaled
cell — only ever sees that algebra. So the way to compile a quantized version of an inline
expression is not to synthesize the algebra directly: it is to write the checkpoint the spellers
already read, and then read it. That keeps ONE producer of quantized graphs. A second one would
be free to drift from the loader (which modules a config marks, how ``input_scale`` is found,
what ``load_quantized_split`` passes through) while still looking right in a dump.

What this decides, and what it cannot: the weight side is fully determined — :func:`quantize_nvfp4`
derives ``scale_2`` from the tensor's own amax and the block scales from it. The ACTIVATION side
is not, because a checkpoint's ``input_scale`` is calibrated over a dataset and an expression has
no dataset. Here it is calibrated over the ONE example input the trace already carries, by the
same formula modelopt uses (``amax / (6 · 448)``), and written into the checkpoint so the number
is inspectable rather than implied. Calibration on one sample is a real calibration, and a poor
one; the checkpoint says so by existing.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from emmy.compiler.dtype import F8E4M3, encode_f8
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import ConstantOp
from emmy.compiler.loader.quant import _E4M3_MAX, _F4_MAX, NVFP4_BLOCK, quantize_nvfp4

logger = logging.getLogger(__name__)

#: The scheme names ``--quantize`` accepts. The two NVFP4 ones differ ONLY in what the config
#: declares about activations, which is the whole difference between the two deployable NVFP4
#: programs: with a static 4-bit declaration the activation speller marks each linear and the native
#: block-scaled cell becomes reachable (W4A4); without it the weights stay packed against 16-bit
#: activations and the packed byte-slab drain is the path (W4A16). Same weights, same checkpoint
#: layout, so the two are directly comparable on one shape. ``fp8-block`` is the official FP8 form
#: (Qwen, DeepSeek and GLM releases): e4m3 weights under one f32 scale per 128x128 block, with
#: per-token dynamic e4m3 activations.
SCHEMES = ("nvfp4", "nvfp4-w4a16", "fp8-block")

#: The weight block one ``fp8-block`` scale covers, on both axes.
FP8_BLOCK = 128

#: The official FP8 declaration, the shape Qwen/Qwen3-0.6B-FP8 ships.
_FP8_BLOCK_CONFIG = {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [FP8_BLOCK, FP8_BLOCK]}

#: modelopt's NVFP4 declaration, the shape ``_fp4_quant_config`` and the static activation reader
#: both recognize — the same one nvidia/Qwen3-8B-NVFP4 ships.
_NVFP4_CONFIG = {
    "quant_algo": "NVFP4",
    "quant_method": "modelopt",
    "config_groups": {
        "group_0": {
            "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": NVFP4_BLOCK},
            "weights": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": NVFP4_BLOCK},
            "targets": ["Linear"],
        }
    },
    "ignore": [],
}

#: The same declaration minus the activation half — nvidia/Qwen3.6-27B-NVFP4's shape, where the
#: MLP is packed but its activations stay 16-bit.
_NVFP4_W4A16_CONFIG = {
    **{k: v for k, v in _NVFP4_CONFIG.items() if k != "config_groups"},
    "config_groups": {"group_0": {"weights": _NVFP4_CONFIG["config_groups"]["group_0"]["weights"], "targets": ["Linear"]}},
}


def _linear_weights(graph: Graph) -> list[tuple[str, str]]:
    """Every ``(weight buffer, module path)`` a ``LinearOp`` reads, in graph order.

    The module path is the weight constant's own ``source_path`` minus its ``.weight`` suffix —
    for a bare ``nn.Linear`` the trace names it just ``weight``, so the path is empty and
    ``get_submodule("")`` is the module itself.
    """
    from emmy.compiler.ir.frontend.ir import LinearOp  # noqa: PLC0415

    out: list[tuple[str, str]] = []
    for node in graph.nodes.values():
        if not isinstance(node.op, LinearOp) or len(node.inputs) < 2:
            continue
        weight = graph.producer(node.inputs[1])
        if weight is None or not isinstance(weight.op, ConstantOp) or not weight.op.source_path:
            continue
        path = weight.op.source_path
        out.append((node.inputs[1], path[: -len(".weight")] if path.endswith(".weight") else path.removesuffix("weight").rstrip(".")))
    return out


def _quantize_fp8_block(w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(e4m3 bits, f32 scales)`` for one ``[N, K]`` weight: each 128x128 block's scale is its
    amax over the e4m3 maximum, floored like the dynamic activation scale, and the stored scale is
    the dequant multiplier (what the checkpoint names ``weight_scale_inv``)."""
    n, k = w.shape
    blocks = w.reshape(n // FP8_BLOCK, FP8_BLOCK, k // FP8_BLOCK, FP8_BLOCK)
    scale = (np.maximum(np.abs(blocks).max(axis=(1, 3)), 1e-12) / _E4M3_MAX).astype(np.float32)
    bits = encode_f8(blocks / scale[:, None, :, None], F8E4M3.name)
    return bits.reshape(n, k), scale


def _activation_amax(module, args, kwargs, paths: set[str]) -> dict[str, float]:
    """Each named submodule's input amax over the trace's ONE example input.

    A forward pre-hook rather than a graph evaluation: the module is right there, it is what the
    trace came from, and its inputs are the tensors the linear actually sees.
    """
    import torch  # noqa: PLC0415

    seen: dict[str, float] = {}
    handles = []
    for path in paths:
        sub = module.get_submodule(path) if path else module

        def hook(_mod, inputs, _path=path):
            if inputs and isinstance(inputs[0], torch.Tensor):
                seen[_path] = max(seen.get(_path, 0.0), float(inputs[0].detach().abs().max()))

        handles.append(sub.register_forward_pre_hook(hook))
    try:
        with torch.no_grad():
            module(*args, **(kwargs or {}))
    finally:
        for h in handles:
            h.remove()
    return seen


def write_quantized_checkpoint(graph: Graph, bundle, out_dir: str | Path, *, scheme: str = "nvfp4") -> Path:
    """Quantize every linear weight of a TRACED graph, write the checkpoint, repoint the graph.

    Returns the checkpoint directory. The graph's weight constants come out naming
    ``l<i>.weight`` in it, so the caller runs the ordinary spellers over that directory and gets
    exactly the program a real checkpoint of the same shape would give.

    Declines nothing quietly: a linear whose weight is not a pristine trace constant, or whose
    shape the scheme's block cannot tile, is left unquantized and logged.
    """
    if scheme not in SCHEMES:
        raise ValueError(f"unknown quantization scheme {scheme!r} (have {', '.join(SCHEMES)})")
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    module, args, kwargs = bundle
    weights = _linear_weights(graph)
    if not weights:
        raise ValueError(
            "no linear whose weight is a module parameter. Quantization applies to a STORED weight, "
            "and an expression like `a @ b` over two tensors has none — both operands are inputs. "
            "Use a module that owns its weight, e.g. nn.Linear(512, 128, bias=False)(torch.randn(64, 512))."
        )
    state = module.state_dict()
    amax = _activation_amax(module, args, kwargs, {path for _buf, path in weights})

    tensors: dict[str, torch.Tensor] = {}
    for i, (buf, path) in enumerate(weights):
        key = f"{path}.weight" if path else "weight"
        value = state.get(key)
        if value is None:
            logger.warning("--quantize: no state_dict entry %r; leaving that linear unquantized", key)
            continue
        w = value.detach().to(torch.float32).numpy()
        blocks = (FP8_BLOCK, FP8_BLOCK) if scheme == "fp8-block" else (1, NVFP4_BLOCK)
        if w.ndim != 2 or any(d % b for d, b in zip(w.shape, blocks, strict=True)):
            logger.warning("--quantize: %s has shape %s, which the %s block cannot carry", key, w.shape, "x".join(map(str, blocks)))
            continue
        name = f"l{i}"
        if scheme == "fp8-block":
            bits, scale = _quantize_fp8_block(w)
            tensors[f"{name}.weight"] = torch.from_numpy(bits).view(torch.float8_e4m3fn)
            tensors[f"{name}.weight_scale_inv"] = torch.from_numpy(scale)
        else:
            packed, scale_bits, scale_2 = quantize_nvfp4(w)
            tensors[f"{name}.weight"] = torch.from_numpy(packed)
            tensors[f"{name}.weight_scale"] = torch.from_numpy(np.ascontiguousarray(scale_bits)).view(torch.float8_e4m3fn)
            tensors[f"{name}.weight_scale_2"] = torch.from_numpy(scale_2)
            # The activation's calibrated level, in modelopt's units. A zero-amax input would divide
            # the block quantize by zero, so it floors the same way the weight side's amax does.
            tensors[f"{name}.input_scale"] = torch.tensor(max(amax.get(path, 0.0), 1e-12) / (_F4_MAX * _E4M3_MAX), dtype=torch.float32)
        graph.nodes[graph.producer(buf).id].op = ConstantOp(
            name=graph.producer(buf).op.name,
            source_path=f"{name}.weight",
            source_shape=tuple(int(d) for d in w.shape),
            source_dtype=graph.producer(buf).op.source_dtype,
        )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(out / "model.safetensors"))
    config = {"nvfp4": _NVFP4_CONFIG, "nvfp4-w4a16": _NVFP4_W4A16_CONFIG, "fp8-block": _FP8_BLOCK_CONFIG}[scheme]
    (out / "config.json").write_text(json.dumps({"model_type": "synthetic", "quantization_config": config}, indent=1))
    logger.info("wrote a %s checkpoint for %d linear(s): %s", scheme, len({k.split(".")[0] for k in tensors}), out)
    return out


def quantize_and_spell(graph: Graph, bundle, out_dir: str | Path, *, scheme: str = "nvfp4") -> tuple[Path, int, int]:
    """Write the checkpoint for ``graph``'s linear weights and spell the graph against it —
    returning ``(checkpoint path, weights spelled, linears marked for quantized activations)``.

    The pairing is the whole point of the module: writing a checkpoint and then reading it back
    through the ORDINARY spellers is what keeps one producer of quantized graphs. Callers that did
    the two halves themselves would be free to drift from `emmy compile --quantize`, which is the
    drift this module exists to prevent. Each activation speller is a no-op on the other scheme's
    checkpoint."""
    from emmy.compiler.loader.quant import (  # noqa: PLC0415
        spell_dynamic_fp8_activations,
        spell_quantized_constants,
        spell_static_fp4_activations,
    )

    ckpt = write_quantized_checkpoint(graph, bundle, out_dir, scheme=scheme)
    spelled = spell_quantized_constants(graph, str(ckpt))
    return ckpt, spelled, spell_dynamic_fp8_activations(graph, str(ckpt)) + spell_static_fp4_activations(graph, str(ckpt))


def summarize(out: Path) -> str:
    """One line per quantized linear — what the written checkpoint says, for the log."""
    from safetensors import safe_open  # noqa: PLC0415

    with safe_open(str(out / "model.safetensors"), framework="numpy") as f:
        keys = set(f.keys())
        rows = []
        for n in sorted({k.split(".")[0] for k in keys}):
            weight = tuple(f.get_slice(f"{n}.weight").get_shape())
            if f"{n}.weight_scale_inv" in keys:
                rows.append(f"  {n}: e4m3 {weight}  block scales {tuple(f.get_slice(f'{n}.weight_scale_inv').get_shape())}")
                continue
            s2 = float(np.asarray(f.get_tensor(f"{n}.weight_scale_2"), dtype=np.float32).reshape(-1)[0])
            i_s = float(np.asarray(f.get_tensor(f"{n}.input_scale"), dtype=np.float32).reshape(-1)[0])
            rows.append(f"  {n}: packed {weight}  weight_scale_2 {s2:.6g}  input_scale {i_s:.6g}")
    return "\n".join(rows)


__all__ = ["SCHEMES", "quantize_and_spell", "summarize", "write_quantized_checkpoint"]
