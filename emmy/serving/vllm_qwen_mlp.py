"""Stock Qwen3.5 hybrid model with only its dense text MLPs compiled by Emmy."""

from __future__ import annotations

import gc
import logging
import os
from itertools import chain

import torch
from torch import nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

from emmy.serving.mlp import MLPPrograms, checkpoint_keys, text_prefix

logger = logging.getLogger(__name__)


def _require_explicit_pins() -> None:
    """Keep this lane off the known unusable unpinned prior route."""
    from emmy import config
    from emmy.compiler.pipeline.knob import parse_knob_spec

    pins = parse_knob_spec(config.knobs_aggregate())
    fast_math = pins.get("FAST_MATH", config.knob_raw("FAST_MATH"))
    if fast_math is None or fast_math.lower() not in ("0", "false"):
        raise ValueError("mixed Qwen MLP serving requires FAST_MATH=false in EMMY_KNOBS or EMMY_FAST_MATH")
    schedule = any(key.split("@", 1)[0] in {"PLACE", "WORK", "TILE", "STAGE", "REDUCE", "RASTER"} for key in pins)
    if not schedule and not os.environ.get(config.GOLDEN_FILE):
        raise ValueError("mixed Qwen MLP serving requires explicit EMMY_KNOBS schedule pins or EMMY_GOLDEN_FILE")


class _CompiledMLP(nn.Module):
    def __init__(self, layer: int, owner):
        super().__init__()
        self.layer = layer
        # Do not register the whole vLLM model as a child of an MLP module.
        object.__setattr__(self, "owner", owner)

    def forward(self, x):
        programs = self.owner._emmy_mlp_programs
        if programs is None:
            raise RuntimeError("Emmy MLP weights have not been bound")
        return programs.forward(self.layer, x)


class EmmyQwen35MlpModel(Qwen3_5ForConditionalGeneration):
    """Keep the upstream constructor, hybrid state methods, and non-MLP loader."""

    def __init__(self, *, vllm_config, prefix: str = "model"):
        parallel = vllm_config.parallel_config
        model = vllm_config.model_config
        scheduler = vllm_config.scheduler_config
        if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
            raise ValueError("mixed Qwen MLP serving requires TP1 and PP1")
        if model.dtype != torch.bfloat16:
            raise ValueError(f"mixed Qwen MLP serving requires checkpoint BF16, got {model.dtype}")
        if scheduler.max_num_seqs != 1 or not 2 <= scheduler.max_num_batched_tokens <= 64:
            raise ValueError("mixed Qwen MLP serving requires one active request and 2..64 scheduled tokens")
        if model.max_model_len > 4096:
            raise ValueError("mixed Qwen MLP serving requires at most 4096 context tokens")
        if not model.multimodal_config or not model.multimodal_config.language_model_only:
            raise ValueError("mixed Qwen MLP serving requires --language-model-only")
        if not model.enforce_eager:
            raise ValueError("mixed Qwen MLP serving requires --enforce-eager")
        if vllm_config.cache_config.enable_prefix_caching:
            raise ValueError("mixed Qwen MLP serving requires prefix caching disabled")
        if vllm_config.speculative_config is not None:
            raise ValueError("mixed Qwen MLP serving does not support speculation")
        text = model.hf_text_config
        if getattr(text, "model_type", None) != "qwen3_5_text" or getattr(text, "hidden_act", None) != "silu":
            raise ValueError("mixed Qwen MLP serving requires a dense Qwen3.5 text model with SiLU")
        _require_explicit_pins()
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self._emmy_mlp_programs = None
        layers = self.language_model.model.layers
        if len(layers) != text.num_hidden_layers:
            raise ValueError("mixed Qwen MLP serving requires all decoder layers on one device")
        for index, layer in enumerate(layers):
            old = layer.mlp
            layer.mlp = _CompiledMLP(index, self)
            del old
        # vLLM linear parameters retain bound weight-loader methods. Those form
        # cycles back to the displaced MLP modules, whose packed GPU placeholders
        # must be collected before Emmy allocates its own packed weight buffers.
        gc.collect()
        torch.cuda.empty_cache()
        self._emmy_model = model.model
        self._emmy_revision = model.revision
        self._emmy_download_dir = vllm_config.load_config.download_dir
        self._emmy_hidden = text.hidden_size
        self._emmy_intermediate = text.intermediate_size
        self._emmy_layers = text.num_hidden_layers
        self._emmy_capacity = scheduler.max_num_batched_tokens
        logger.info("Emmy MLP adapter replaced %d stock MLP modules; vLLM retains all attention and GDN layers", len(layers))

    def load_weights(self, weights):
        from emmy.compiler.loader.safetensors import _resolve_model_dir

        # vLLM's checkpoint download starts only when its lazy weight iterator advances.
        iterator = iter(weights)
        try:
            first = next(iterator)
        except StopIteration as exc:
            raise ValueError("mixed Qwen MLP checkpoint has no weights") from exc
        model_dir = _resolve_model_dir(self._emmy_model, self._emmy_revision, cache_dir=self._emmy_download_dir, local_files_only=True)
        prefix = text_prefix(model_dir)
        required = frozenset().union(*(checkpoint_keys(i, prefix=prefix) for i in range(self._emmy_layers)))
        seen = set()
        text_layers_prefix = f"{prefix}.layers."

        def stock_weights():
            for name, tensor in chain((first,), iterator):
                if name in required:
                    if name in seen:
                        raise ValueError(f"duplicate MLP checkpoint tensor: {name}")
                    seen.add(name)
                    continue
                if name.startswith(text_layers_prefix) and ".mlp." in name:
                    raise ValueError(f"unclaimed MLP checkpoint tensor: {name}")
                yield name, tensor

        loaded = super().load_weights(stock_weights())
        missing = required - seen
        if missing:
            raise ValueError(f"missing MLP checkpoint tensors: {sorted(missing)[:4]} ({len(missing)} total)")
        self._emmy_mlp_programs = MLPPrograms(
            model_dir,
            self._emmy_hidden,
            self._emmy_intermediate,
            self._emmy_layers,
            dtype=torch.bfloat16,
            capacity=self._emmy_capacity,
        )
        logger.info("Emmy bound %d packed BF16 MLP programs at token capacity %d", self._emmy_layers, self._emmy_capacity)
        return loaded
