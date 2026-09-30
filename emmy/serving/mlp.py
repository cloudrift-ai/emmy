"""Checkpoint-backed, stateless dense MLP programs for mixed vLLM serving.

The logical module is used only to trace the same graph for serving and tuning.
vLLM retains ownership of the decoder layers, their mixers, and their state.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_LEAVES = ("weight", "weight_scale", "weight_scale_2", "input_scale")
MLP_STATIC_ROWS = 16
MLP_PREFILL_ROWS = 64


def checkpoint_keys(layer: int, *, prefix: str = "model") -> frozenset[str]:
    """The twelve exact tensors owned by one packed dense MLP."""
    return frozenset(f"{prefix}.layers.{layer}.mlp.{proj}.{leaf}" for proj in _PROJECTIONS for leaf in _LEAVES)


def logical_mlp(hidden: int, intermediate: int, dtype):
    """One temporary shape-only module; its dense parameters never reach the GPU."""
    import torch
    from torch import nn

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(hidden, intermediate, bias=False)
            self.up_proj = nn.Linear(hidden, intermediate, bias=False)
            self.down_proj = nn.Linear(intermediate, hidden, bias=False)

        def forward(self, x):
            return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))

    with torch.device("meta"):
        module = MLP().to(dtype=dtype)
    return module.to_empty(device="cpu")


def parameter_keys(module, layer: int, *, prefix: str = "model") -> dict[int, str]:
    return {id(t): f"{prefix}.layers.{layer}.mlp.{name}" for name, t in module.named_parameters()}


def text_prefix(model_dir: str | Path) -> str:
    from emmy.compiler.loader.safetensors import _build_index

    index = _build_index(Path(model_dir))
    for prefix in ("model.language_model", "model", "language_model"):
        if f"{prefix}.layers.0.mlp.gate_proj.weight" in index:
            return prefix
    raise ValueError("checkpoint has no dense text MLP at layer 0")


def layer_profiles(model_dir: str | Path, count: int, *, prefix: str | None = None) -> dict[tuple, list[int]]:
    """Group only layers whose stored shapes and activation-scale sharing agree.

    Scale values remain per-layer constants even when the compiled structure shares.
    """
    import numpy as np
    from safetensors import safe_open

    from emmy.compiler.loader.safetensors import _build_index

    index = _build_index(Path(model_dir))
    prefix = prefix or text_prefix(model_dir)
    groups: dict[tuple, list[int]] = defaultdict(list)
    for layer in range(count):
        keys = checkpoint_keys(layer, prefix=prefix)
        missing = keys - index.keys()
        if missing:
            raise ValueError(f"MLP layer {layer} is missing checkpoint tensors: {sorted(missing)}")
        details = []
        scales = []
        for proj in _PROJECTIONS:
            base = f"{prefix}.layers.{layer}.mlp.{proj}"
            for leaf in _LEAVES:
                key = f"{base}.{leaf}"
                with safe_open(index[key], framework="numpy") as shard:
                    slice_ = shard.get_slice(key)
                    shape = tuple(slice_.get_shape())
                    stored_dtype = slice_.get_dtype()
                    scale = shard.get_tensor(key) if leaf == "input_scale" else None
                if leaf in ("weight_scale_2", "input_scale") and shape == ():
                    shape = (1,)
                details.append((proj, leaf, shape, stored_dtype))
                if scale is not None:
                    scales.append(np.asarray(scale, dtype=np.float32).tobytes())
        groups[(tuple(details), scales[0] == scales[1])].append(layer)
    return dict(groups)


def _validate_packed_profiles(groups: dict[tuple, list[int]], hidden: int, intermediate: int) -> None:
    if hidden % 16 or intermediate % 16:
        raise ValueError("NVFP4 MLP dimensions must be divisible by 16")
    for (details, _same_input_scale), members in groups.items():
        expected = []
        for proj in _PROJECTIONS:
            n, k = (hidden, intermediate) if proj == "down_proj" else (intermediate, hidden)
            expected.extend(
                (
                    (proj, "weight", (n, k // 2), "U8"),
                    (proj, "weight_scale", (n, k // 16), "F8_E4M3"),
                    (proj, "weight_scale_2", (1,), "F32"),
                    (proj, "input_scale", (1,), "F32"),
                )
            )
        if details != tuple(expected):
            raise ValueError(f"MLP layer {members[0]} does not match the expected packed NVFP4 layout")


def capture_mlp_graphs(
    model_dir: str | Path,
    hidden: int,
    intermediate: int,
    layers: int,
    *,
    dtype="bfloat16",
    static_rows: int = MLP_STATIC_ROWS,
    prefill_rows: int | None = MLP_PREFILL_ROWS,
):
    """Capture the runtime's padded decode and prefill MLP graphs."""
    import torch

    from emmy.compiler.loader.quant import spell_quantized_constants, spell_static_fp4_activations
    from emmy.serving.gen_runner import _retarget_constants, trace_split

    td = getattr(torch, dtype)
    groups = layer_profiles(model_dir, layers)
    _validate_packed_profiles(groups, hidden, intermediate)
    prefix = text_prefix(model_dir)
    graphs = {}
    for profile_index, (_profile, members) in enumerate(groups.items()):
        layer = members[0]
        module = logical_mlp(hidden, intermediate, td)
        keys = parameter_keys(module, layer, prefix=prefix)
        prefill = ("mlp-sym", 8, ["x"]) if prefill_rows is None else (f"mlp{prefill_rows}", prefill_rows, None)
        for label, rows, argnames in ((f"mlp{static_rows}", static_rows, None), prefill):
            graph = trace_split(module, [torch.zeros(rows, hidden, dtype=td)], argnames)
            _retarget_constants(graph, module, keys)
            if not spell_quantized_constants(graph, str(model_dir)):
                raise ValueError(f"MLP layer {layer} has no NVFP4 constants")
            spell_static_fp4_activations(graph, str(model_dir))
            suffix = "" if len(groups) == 1 else f"-profile{profile_index}"
            graphs[f"{label}{suffix}@nvfp4"] = graph
        del module
    return graphs


class MLPPrograms:
    """Per-layer constants with one shared activation arena and compilation cache."""

    def __init__(
        self,
        model_dir: str | Path,
        hidden: int,
        intermediate: int,
        layers: int,
        *,
        dtype,
        capacity: int = 64,
        static_rows: int = MLP_STATIC_ROWS,
        prefill_rows: int | None = MLP_PREFILL_ROWS,
    ):
        import numpy as np
        import torch

        from emmy import config
        from emmy.compiler.backend.cuda.program import BufferArena
        from emmy.compiler.backend.plan_cache import PlanTemplateCache
        from emmy.compiler.pipeline.knob import scoped_knob_spec
        from emmy.serving.gen_runner import _compile_split

        if capacity < 2:
            raise ValueError(f"MLP capacity must be at least 2, got {capacity}")
        if static_rows < 1:
            raise ValueError(f"MLP static rows must be positive, got {static_rows}")
        if prefill_rows is not None and prefill_rows < capacity:
            raise ValueError(f"MLP prefill rows must cover capacity {capacity}, got {prefill_rows}")
        if dtype not in (torch.bfloat16, torch.float16):
            raise ValueError(f"MLP dtype must be bfloat16 or float16, got {dtype}")
        self.hidden = hidden
        self.capacity = capacity
        self.static_rows = static_rows
        self.prefill_rows = prefill_rows
        self.dtype = dtype
        self.programs = []
        arena = BufferArena()
        plan_cache = PlanTemplateCache()
        np_dtype = np.dtype("float32") if dtype == torch.bfloat16 else np.dtype("float16")
        _validate_packed_profiles(layer_profiles(model_dir, layers), hidden, intermediate)
        prefix = text_prefix(model_dir)
        for layer in range(layers):
            module = logical_mlp(hidden, intermediate, dtype)
            ckpt = (str(model_dir), parameter_keys(module, layer, prefix=prefix))
            constants = {}
            with scoped_knob_spec(config.mlp_prefill_knobs()):
                prefill, _ = _compile_split(
                    module,
                    [torch.zeros(prefill_rows or 8, hidden, dtype=dtype)],
                    None if prefill_rows is not None else ["x"],
                    np_dtype,
                    dev_consts=constants,
                    arena=arena,
                    capacity=capacity if prefill_rows is None else None,
                    ckpt=ckpt,
                    plan_cache=plan_cache,
                )
            with scoped_knob_spec(config.mlp_static_knobs()):
                one, _ = _compile_split(
                    module,
                    [torch.zeros(static_rows, hidden, dtype=dtype)],
                    None,
                    np_dtype,
                    dev_consts=constants,
                    arena=arena,
                    ckpt=ckpt,
                    plan_cache=plan_cache,
                )
            self.programs.append((one, prefill))
            del module

    def forward(self, layer: int, x):
        import torch

        if x.ndim != 2 or x.shape[1] != self.hidden or x.dtype != self.dtype or not x.is_cuda:
            raise ValueError(f"MLP input must be CUDA {self.dtype}[T,{self.hidden}], got {x.shape}, {x.dtype}, {x.device}")
        tokens = x.shape[0]
        if tokens > self.capacity:
            raise ValueError(f"MLP token width {tokens} exceeds compiled capacity {self.capacity}")
        if tokens == 0:
            return torch.empty_like(x)
        one, prefill = self.programs[layer]
        if tokens == 1:
            return one.run_device([x])[0]
        if self.prefill_rows is None:
            return prefill.run_device_sym([x])[0]
        return prefill.run_device([x])[0]
