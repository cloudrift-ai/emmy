"""A Qwen3.5 / Qwen3.8 vision-language checkpoint served with its images: vLLM's vision tower, Emmy's trunk.

``EmmyGenQwenVLModel`` is vLLM's own ``Qwen3_5ForConditionalGeneration`` with ``EmmyGenHybridModel`` as its language
model. Everything about an image is vLLM's and runs unchanged: the multimodal processor, the vision tower and its
merger (``Qwen3_VisionTransformer``), the merge of the image embeddings into the token embeddings, and the M-RoPE
positions of an image prompt. Every decoder layer is Emmy's: the forward hands the merged embeddings to the trunk.

vLLM feeds a multimodal model embeddings on every step, text steps too, so the token-embedding gather runs before
the captured decode graph instead of inside it.

``EMMY_GEN_VISION_HOST`` keeps the tower's weights in pinned, mapped host memory: a text request never runs the
tower, so its weights would hold device memory at rest for nothing. The tower's own code reads them over the link
through each parameter's device view when an image prefill runs.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import StageMissingLayer, maybe_prefix

from emmy import config as emmy_config
from emmy.serving.vllm_model_gen import EmmyGenHybridModel

logger = logging.getLogger(__name__)

#: The tower's tensors in the checkpoint, and in the module.
_CHECKPOINT_PREFIX = "model.visual."


class EmmyGenQwenVLModel(Qwen3_5ForConditionalGeneration):
    def __init__(self, *, vllm_config, prefix: str = "model"):
        # The stock constructor, with the Emmy trunk in place of vLLM's language model.
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_config
        self.config = config
        self.model_config = vllm_config.model_config
        self.multimodal_config = vllm_config.model_config.multimodal_config
        self.use_data_parallel = self.multimodal_config.mm_encoder_tp_mode == "data"
        self.is_multimodal_pruning_enabled = False
        with self._mark_tower_model(vllm_config, {"image", "video"}):
            self.visual = Qwen3_VisionTransformer(
                config.vision_config,
                norm_eps=getattr(config, "rms_norm_eps", 1e-6),
                quant_config=vllm_config.quant_config,
                prefix=maybe_prefix(prefix, "visual"),
            )
        with self._mark_language_model(vllm_config):
            self.language_model = EmmyGenHybridModel(vllm_config=vllm_config, prefix=maybe_prefix(prefix, "language_model"))
        self.make_empty_intermediate_tensors = self.language_model.make_empty_intermediate_tensors

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None, **kwargs):
        return self.language_model(input_ids, positions, intermediate_tensors, inputs_embeds)

    def load_weights(self, weights):
        """The trunk loads as ``EmmyGenHybridModel`` loads it; the tower reads its tensors straight from the checkpoint,
        because the trunk's loader leaves the weight stream unread (a coded head) or reads only what it owns."""
        from emmy.compiler.loader.safetensors import checkpoint_weight_names, load_sources_by_path

        loaded = {f"language_model.{name}" for name in self.language_model.load_weights(weights)}
        if isinstance(self.visual, StageMissingLayer):  # every image and video limit is 0
            return loaded
        model_id = self.language_model._model_id
        paths = [name for name in checkpoint_weight_names(model_id) if name.startswith(_CHECKPOINT_PREFIX)]
        src = load_sources_by_path(model_id, paths)
        stream = ((path.removeprefix(_CHECKPOINT_PREFIX), torch.from_numpy(src.pop(path))) for path in paths)
        loaded |= {f"visual.{name}" for name in self.visual.load_weights(stream)}
        if emmy_config.gen_vision_host():
            self._tower_host = map_to_host(self.visual)
        self.language_model.reclaim_device_memory()
        return loaded

    # The GDN layers' per-request state is the Emmy trunk's, not the stock language model's.
    @classmethod
    def get_mamba_state_shape_from_config(cls, vllm_config):
        return EmmyGenHybridModel.get_mamba_state_shape_from_config(vllm_config)

    @classmethod
    def get_mamba_state_dtype_from_config(cls, vllm_config):
        return EmmyGenHybridModel.get_mamba_state_dtype_from_config(vllm_config)

    @classmethod
    def get_mamba_state_copy_func(cls):
        return EmmyGenHybridModel.get_mamba_state_copy_func()


def map_to_host(module: nn.Module) -> list:
    """Move every parameter of ``module`` into pinned, mapped host memory, its ``data`` the device view of that memory,
    so the module's own code runs unchanged and reads the weights over the link. Returns the host allocations, which
    the views do not keep alive."""
    from emmy.serving.gen_runner import _pinned

    allocations, nbytes = [], 0
    for param in module.parameters():
        raw, view = _pinned(param.data)
        param.data = view.view(param.dtype).view(param.shape)
        allocations.append(raw)
        nbytes += raw.numel()
    logger.info("[EmmyGenQwenVLModel] vision tower (%.3f GiB) mapped in host memory — 0 device bytes", nbytes / 2**30)
    return allocations
