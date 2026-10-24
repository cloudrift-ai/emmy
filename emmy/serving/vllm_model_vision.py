"""A Qwen3.5 / Qwen3.8 vision-language checkpoint served with its images: Emmy's tower programs, Emmy's trunk.

``EmmyGenQwenVLModel`` is vLLM's own ``Qwen3_5ForConditionalGeneration`` with ``EmmyGenHybridModel`` as its language
model and :class:`~emmy.serving.vision_tower.CompiledTower` in its tower's slot: vLLM calls the tower, the tower runs
compiled programs for the patch embedding, every block's projections and MLP, and the merger, and vLLM's own code for
each block's attention. The rest of an image is vLLM's and runs unchanged: the multimodal processor, the merge of the
image embeddings into the token embeddings, and the M-RoPE positions of an image prompt. Every decoder layer is
Emmy's: the forward hands the merged embeddings to the trunk.

vLLM feeds a multimodal model embeddings on every step, text steps too, so the token-embedding gather runs before
the captured decode graph instead of inside it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.model_executor.models.utils import StageMissingLayer, maybe_prefix

from emmy.serving.vision_tower import CompiledTower
from emmy.serving.vllm_model_gen import EmmyGenHybridModel

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
        model_id, tower = self.language_model._model_id, self.visual
        # The programs bind the blocks' and the merger's weights; vLLM's tower keeps the position table.
        self.visual = CompiledTower(tower, model_id, tower.dtype)
        kept = {f"{_CHECKPOINT_PREFIX}{name}" for name, _ in tower.named_parameters()}
        paths = [name for name in checkpoint_weight_names(model_id) if name in kept]
        src = load_sources_by_path(model_id, paths)
        stream = ((path.removeprefix(_CHECKPOINT_PREFIX), torch.from_numpy(src.pop(path))) for path in paths)
        loaded |= {f"visual.visual.{name}" for name in tower.load_weights(stream)}
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
