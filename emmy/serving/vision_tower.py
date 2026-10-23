"""The Qwen3.5 / Qwen3.8 vision tower as compiled programs: vLLM calls the tower, the tower runs Emmy's programs.

Four traced pieces run as programs at the widths of :data:`IMAGE_TOKEN_BUCKETS`: the patch embedding, each block's
``pre`` (norm1, q/k/v) and ``post`` (the output projection, norm2, the MLP, both residuals), and the merger. vLLM keeps
each block's attention and what depends on the grid alone (the position table's interpolation, the rotary tables).
:func:`tower_twins` traces the same pieces weight-free. The serving ``ARCHITECTURE.md`` (``vision_tower.py``) holds
the design.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

if TYPE_CHECKING:
    from emmy.compiler.graph import Graph

logger = logging.getLogger(__name__)

#: The image sizes the tower is compiled at, in image tokens: a 512 x 512 and a 1024 x 1024 image at Qwen3.5's
#: 16-pixel patches and 2 x 2 merge. An image runs at the smallest that holds it.
IMAGE_TOKEN_BUCKETS = (256, 1024)
#: The traced pieces, in the order an image runs them (``pre`` and ``post`` once per block).
PIECES = ("embed", "pre", "post", "merger")


class _Embed(nn.Module):
    def __init__(self, tower):
        super().__init__()
        self.patch_embed = tower.patch_embed

    def forward(self, patches, positions):
        return self.patch_embed(patches) + positions


class _Pre(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.norm1, self.qkv = block.norm1, block.attn.qkv

    def forward(self, hidden):
        return self.qkv(self.norm1(hidden))


class _Post(nn.Module):
    def __init__(self, block):
        super().__init__()
        self.proj, self.norm2, self.mlp = block.attn.proj, block.norm2, block.mlp

    def forward(self, attn_out, residual):
        hidden = residual + self.proj(attn_out)
        return hidden + self.mlp(self.norm2(hidden))


class _Merger(nn.Module):
    """One row per image token: the token's patches side by side, as the merger reads them after its norm."""

    def __init__(self, tower):
        super().__init__()
        self.merger = tower.merger

    def forward(self, patches):
        return self.merger(patches.reshape(-1, self.merger.norm.normalized_shape[0]))


def hf_tower(config, dtype):
    """The transformers vision tower of a Qwen3.5-family ``config`` on the meta device: the modules the pieces trace,
    with their checkpoint names under ``model.visual.``."""
    from transformers import AutoModel  # noqa: PLC0415

    with torch.device("meta"):
        return AutoModel.from_config(config, dtype=dtype).visual


def tower_pieces(tower, block: int) -> dict:
    """``{piece: (wrapper, example arguments at a width in image tokens)}`` for block ``block`` of ``tower``
    (:func:`hf_tower`); the embedding and the merger are the tower's own, whatever the block."""
    vision = tower.config
    hidden, per_token = vision.hidden_size, vision.spatial_merge_size**2
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    dtype = tower.patch_embed.proj.weight.dtype

    def rows(tokens, width):
        return torch.zeros(tokens * per_token, width, dtype=dtype)

    return {
        "embed": (_Embed(tower), lambda t: [rows(t, patch), rows(t, hidden)]),
        "pre": (_Pre(tower.blocks[block]), lambda t: [rows(t, hidden)]),
        "post": (_Post(tower.blocks[block]), lambda t: [rows(t, hidden), rows(t, hidden)]),
        "merger": (_Merger(tower), lambda t: [torch.zeros(t, per_token * hidden, dtype=dtype)]),
    }


def tower_twins(config, dtype: str) -> dict[str, Graph]:
    """The tower's twins for the Qwen3.5-family model ``config`` in ``dtype`` (a torch name), weight-free: every piece
    at every width of :data:`IMAGE_TOKEN_BUCKETS`, block 0 standing for every block."""
    from emmy.serving.gen_runner import trace_split  # noqa: PLC0415

    graphs = {}
    for piece, (wrapper, example) in tower_pieces(hf_tower(config, getattr(torch, dtype)), 0).items():
        wrapper.to_empty(device="cpu")
        for tokens in IMAGE_TOKEN_BUCKETS:
            graphs[f"tower{tokens}-{piece}"] = trace_split(wrapper, example(tokens), None)
    return graphs


class _Seam(nn.Module):
    """Stands in for a projection vLLM's attention runs: hands back its input, which a program already projected."""

    def forward(self, x):
        return x, None


class CompiledTower(nn.Module):
    """vLLM's vision tower with the blocks' compute as compiled programs, in the tower's slot.

    ``visual`` is vLLM's tower (``Qwen3_VisionTransformer``): it keeps the position table, the rotary tables, the
    metadata of a batch of images (``prepare_encoder_metadata``) and each block's attention, whose projections become
    :class:`_Seam`; its other modules are dropped. The programs trace the transformers tower of the checkpoint
    ``model_id`` and bind their weights from it. Its config is transformers' own class read from the checkpoint, as the
    twins read it: inside vLLM the model type is registered onto vLLM's config class, which transformers' tower does
    not construct from."""

    def __init__(self, visual: nn.Module, model_id: str, dtype: torch.dtype):

        from emmy.compiler.backend.cuda.program import BufferArena  # noqa: PLC0415
        from emmy.compiler.backend.plan_cache import PlanTemplateCache  # noqa: PLC0415
        from emmy.compiler.dtype import get  # noqa: PLC0415
        from emmy.compiler.loader.safetensors import load_sources_by_path, split_revision  # noqa: PLC0415
        from emmy.compiler.trace.huggingface import _auto_config_from_pretrained  # noqa: PLC0415
        from emmy.serving.gen_runner import _WeightScope  # noqa: PLC0415

        super().__init__()
        for block in visual.blocks:
            block.attn.qkv = block.attn.proj = _Seam()
            del block.norm1, block.norm2, block.mlp
        del visual.merger
        self.visual = visual
        model, revision = split_revision(model_id)
        tower = hf_tower(_auto_config_from_pretrained(model, revision=revision), dtype)
        paths = {id(param): path for path, param in tower.named_parameters()}
        self.per_token = tower.config.spatial_merge_size**2
        compiler_dtype = get(str(dtype).removeprefix("torch."))
        arena, plans, fork_less = BufferArena(), PlanTemplateCache(), {}
        # ``programs[piece][tokens]``: the embedding's and the merger's program, or one per block.
        self.programs = {piece: {tokens: [] for tokens in IMAGE_TOKEN_BUCKETS} for piece in PIECES}
        self.weight_bytes = 0
        for block in range(len(tower.blocks)):
            for piece, (wrapper, example) in tower_pieces(tower, block).items():
                if piece in ("embed", "merger") and block:
                    continue
                keys = {name: f"model.visual.{paths[id(param)]}" for name, param in wrapper.named_parameters()}
                values = load_sources_by_path(model_id, list(keys.values()))
                if missing := sorted(set(keys.values()) - set(values)):
                    raise RuntimeError(f"the checkpoint stores no {missing[0]} (of {len(missing)} vision tower tensors)")
                wrapper.to_empty(device="cpu")
                for name, param in wrapper.named_parameters():
                    param.data.copy_(torch.from_numpy(values[keys[name]]))
                scope = _WeightScope()
                for tokens, program in _build(wrapper, example, f"{piece}{block}", scope, compiler_dtype, arena, plans, fork_less):
                    self.programs[piece][tokens].append(program)
                self.weight_bytes += sum(a.numel() * a.element_size() for a in scope.arrays.values())
                wrapper.to_empty(device="meta")  # the host copy of the piece's weights goes once its programs bind them
        logger.info(
            "[CompiledTower] %d blocks as compiled programs at %s image tokens, %.3f GiB of weights on the device",
            len(tower.blocks),
            "/".join(map(str, IMAGE_TOKEN_BUCKETS)),
            self.weight_bytes / 2**30,
        )

    @property
    def dtype(self):
        return self.visual.dtype

    @property
    def device(self):
        return self.visual.device

    @property
    def spatial_merge_size(self):
        return self.visual.spatial_merge_size

    def forward(self, x, grid_thw, **_):
        grid = grid_thw.tolist() if isinstance(grid_thw, torch.Tensor) else grid_thw
        x = x.to(device=self.device, dtype=self.dtype)
        out, start = [], 0
        for group in self._groups(grid):
            patches = sum(t * h * w for t, h, w in group)
            out.append(self._run(x[start : start + patches], group))
            start += patches
        return torch.cat(out)

    def _groups(self, grid):
        """Consecutive images whose patches fit the widest program; an image wider than it is refused."""
        widest = IMAGE_TOKEN_BUCKETS[-1] * self.per_token
        groups, patches = [], widest
        for item in grid:
            size = item[0] * item[1] * item[2]
            if size > widest:
                raise ValueError(
                    f"an image of {size // self.per_token} image tokens is wider than the compiled vision tower's widest "
                    f"program ({IMAGE_TOKEN_BUCKETS[-1]} tokens); bound it with --mm-processor-kwargs "
                    f'\'{{"size":{{"longest_edge":{widest * self.visual.patch_size**2}}}}}\''
                )
            if patches + size > widest:
                groups.append([])
                patches = 0
            groups[-1].append(item)
            patches += size
        return groups

    def _run(self, x, grid):
        tokens = x.shape[0] // self.per_token
        width = next(bucket for bucket in IMAGE_TOKEN_BUCKETS if bucket >= tokens)
        programs = {piece: widths[width] for piece, widths in self.programs.items()}
        meta = self.visual.prepare_encoder_metadata(grid)
        (hidden,) = programs["embed"][0].run_device([x, meta["pos_embeds"]])
        for block, pre, post in zip(self.visual.blocks, programs["pre"], programs["post"], strict=True):
            (qkv,) = pre.run_device([hidden])
            attn = block.attn(
                qkv.unsqueeze(1),
                cu_seqlens=meta["cu_seqlens"],
                rotary_pos_emb_cos=meta["rotary_pos_emb_cos"],
                rotary_pos_emb_sin=meta["rotary_pos_emb_sin"],
                max_seqlen=meta["max_seqlen"],
                sequence_lengths=meta.get("sequence_lengths"),
            )
            (hidden,) = post.run_device([attn.squeeze(1), hidden])
        (merged,) = programs["merger"][0].run_device([hidden.reshape(tokens, -1)])
        return merged


def _build(wrapper, example, name, scope, dtype, arena, plans, fork_less):
    """``(tokens, program)`` for one piece at every width, its weights bound once on the device.

    The widths share ``scope`` and build narrowest first, after the scope settles folded every weight one of them
    reads with no layout fork: the order the trunk's twins build in and
    :func:`~emmy.serving.twins.twins_in_build_order` replays, so the kernels are the golden's."""
    from emmy.compiler.backend.plan_cache import binding_neutral_key  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.serving.gen_runner import _compile_split, _traced_split  # noqa: PLC0415
    from emmy.serving.twins import fork_less_weights  # noqa: PLC0415

    traced = {tokens: _traced_split(wrapper, example(tokens), None, None) for tokens in IMAGE_TOKEN_BUCKETS}
    for tokens, graph in traced.items():
        key = binding_neutral_key(graph)
        if key not in fork_less:
            fork_less[key] = frozenset(fork_less_weights(graph, Context.probe()))
        scope.fix(f"{name}.{tokens}", fork_less[key])
    for tokens in IMAGE_TOKEN_BUCKETS:
        program, _plan = _compile_split(
            wrapper,
            example(tokens),
            None,
            dtype,
            dev_consts=scope,
            arena=arena,
            plan_cache=plans,
            twin=f"{name}.{tokens}",
            traced=traced[tokens],
        )
        yield tokens, program
