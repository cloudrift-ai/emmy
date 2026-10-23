"""A Qwen3.5 / Qwen3.8 vision-language checkpoint served with its images: vLLM's vision tower, Emmy's trunk.

``EmmyGenQwenVLModel`` is vLLM's own ``Qwen3_5ForConditionalGeneration`` with ``EmmyGenHybridModel`` as its language
model. Everything about an image is vLLM's and runs unchanged: the multimodal processor, the vision tower and its
merger (``Qwen3_VisionTransformer``), the merge of the image embeddings into the token embeddings, and the M-RoPE
positions of an image prompt. Every decoder layer is Emmy's: the forward hands the merged embeddings to the trunk.

vLLM feeds a multimodal model embeddings on every step, text steps too, so the token-embedding gather runs before
the captured decode graph instead of inside it.

``EMMY_GEN_VISION_HOST`` keeps the tower's weights in pinned, mapped host memory: a text request never runs the
tower, so its weights would hold device memory at rest for nothing. An image prefill copies each block's weights to
the device right before the block runs (:func:`map_to_host`).
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


def map_to_host(tower: nn.Module) -> list:
    """Keep every parameter of ``tower`` in pinned, mapped host memory, and run each block from device memory.

    At rest a parameter's ``data`` is the device view of its host copy. Right before a block (a child module, or one
    entry of a child list) runs, its parameters are copied to the device and rebound there; the block before it is
    rebound to its host copy first, so the caching allocator hands the next block the memory the last one released,
    as a pager slot is reused. When the tower returns, every parameter is back on its host copy: the tower holds no
    device memory at rest, and an image prefill moves each weight over the link once, where a matmul reading host
    memory in place reads its weight again for every tile of rows (25 times slower per image on a x16 link). A block
    that never runs as a module (the position table, which an interpolation reads) reads in place. Returns the host
    allocations, which the views do not keep alive."""
    from emmy.serving.gen_runner import _pinned

    host, mapped = {}, {}
    for param in tower.parameters():
        raw, view = _pinned(param.data)
        host[param], mapped[param] = raw, view.view(param.dtype).view(param.shape)
        param.data = mapped[param]
    staged = []

    def restore(*_):
        for param in staged:
            param.data = mapped[param]
        staged.clear()

    def stage(block, args):
        restore()
        for param in block.parameters():
            param.data = host[param].view(param.dtype).view(param.shape).to(param.device, non_blocking=True)
            staged.append(param)

    for child in tower.children():
        for block in child if isinstance(child, nn.ModuleList) else (child,):
            block.register_forward_pre_hook(stage)
    tower.register_forward_hook(restore)
    logger.info(
        "[EmmyGenQwenVLModel] vision tower (%.3f GiB) in host memory, each block staged on the device as it runs",
        sum(raw.numel() for raw in host.values()) / 2**30,
    )
    return list(host.values())
