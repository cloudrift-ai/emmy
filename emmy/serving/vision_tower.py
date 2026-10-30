"""The Qwen3.5 / Qwen3.8 vision tower as compiled programs: vLLM calls the tower, the tower runs Emmy's programs.

Four traced pieces run as programs at the widths of :data:`IMAGE_TOKEN_BUCKETS`: the patch embedding, each block's
``pre`` (norm1, q/k/v) and ``post`` (the output projection, norm2, the MLP, both residuals), and the merger. vLLM keeps
each block's attention and what depends on the grid alone (the position table's interpolation, the rotary tables).
Below the device budget ``EMMY_GEN_VISION_GPU`` every block's projection weights are pages of the runtime pager
(:class:`_PagedBlocks`). :func:`tower_twins` traces the same pieces weight-free. The serving ``ARCHITECTURE.md``
(``vision_tower.py``) holds the design.
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


def checkpoint_tower(model_id: str, dtype):
    """:func:`hf_tower` of the checkpoint ``model_id`` (a local directory or ``<repo>[@<revision>]``), its config
    transformers' own class read from the checkpoint: inside vLLM the model type is registered onto vLLM's config
    class, which transformers' tower does not construct from."""
    from emmy.compiler.loader.safetensors import split_revision  # noqa: PLC0415
    from emmy.compiler.trace.huggingface import _auto_config_from_pretrained  # noqa: PLC0415

    model, revision = split_revision(model_id)
    return hf_tower(_auto_config_from_pretrained(model, revision=revision), dtype)


def load_piece(wrapper, paths, model_id: str) -> None:
    """Give ``wrapper``, a piece of a meta tower whose parameters ``paths`` names (``{id(param): path in the tower}``),
    its parameters on the CPU, read from the checkpoint ``model_id`` under ``model.visual.``."""
    from emmy.compiler.loader.safetensors import load_sources_by_path  # noqa: PLC0415

    keys = {name: f"model.visual.{paths[id(param)]}" for name, param in wrapper.named_parameters()}
    values = load_sources_by_path(model_id, list(keys.values()))
    if missing := sorted(set(keys.values()) - set(values)):
        raise RuntimeError(f"the checkpoint stores no {missing[0]} (of {len(missing)} vision tower tensors)")
    wrapper.to_empty(device="cpu")
    for name, param in wrapper.named_parameters():
        param.data.copy_(torch.from_numpy(values[keys[name]]))


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

        from emmy import config  # noqa: PLC0415
        from emmy.compiler.backend.cuda.program import BufferArena  # noqa: PLC0415
        from emmy.compiler.backend.plan_cache import PlanTemplateCache  # noqa: PLC0415
        from emmy.compiler.dtype import get  # noqa: PLC0415
        from emmy.serving.gen_runner import _WeightScope, module_sources  # noqa: PLC0415

        super().__init__()
        for block in visual.blocks:
            block.attn.qkv = block.attn.proj = _Seam()
            del block.norm1, block.norm2, block.mlp
        del visual.merger
        self.visual = visual
        tower = checkpoint_tower(model_id, dtype)
        paths = {id(param): path for path, param in tower.named_parameters()}
        self.per_token = tower.config.spatial_merge_size**2
        compiler_dtype = get(str(dtype).removeprefix("torch."))
        arena, plans, fork_less = BufferArena(), PlanTemplateCache(), {}
        # ``programs[piece][tokens]``: the embedding's and the merger's program, or one per block.
        self.programs = {piece: {tokens: [] for tokens in IMAGE_TOKEN_BUCKETS} for piece in PIECES}
        self.weight_bytes = 0
        blocks = len(tower.blocks)
        paging = config.gen_vision_slots(blocks)
        self.pages = None if paging is None else _PagedBlocks(blocks, *paging)
        for block in range(blocks):
            for piece, (wrapper, example) in tower_pieces(tower, block).items():
                if piece in ("embed", "merger") and block:
                    continue
                load_piece(wrapper, paths, model_id)
                scope, paged = _WeightScope(), self.pages is not None and piece in ("pre", "post")
                built = list(_build(wrapper, example, f"{piece}{block}", scope, compiler_dtype, arena, plans, fork_less, paged))
                for tokens, program, _ in built:
                    self.programs[piece][tokens].append(program)
                if paged:
                    programs = [program for _, program, _ in built]
                    self.pages.place(piece, block, programs, built[0][2], module_sources(wrapper, compiler_dtype))
                self.weight_bytes += sum(a.numel() * a.element_size() for a in scope.arrays.values())
                wrapper.to_empty(device="meta")  # the host copy of the piece's weights goes once its programs bind them
        if self.pages is not None:
            self.pages.ready()
        logger.info(
            "[CompiledTower] %d blocks as compiled programs at %s image tokens, %.3f GiB of weights on the device%s",
            blocks,
            "/".join(map(str, IMAGE_TOKEN_BUCKETS)),
            self.weight_bytes / 2**30,
            "" if self.pages is None else f", the blocks' projections paged ({self.pages})",
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
        for index, (block, pre, post) in enumerate(zip(self.visual.blocks, programs["pre"], programs["post"], strict=True)):
            if self.pages is not None:
                self.pages.move_in(index)
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


def traced_widths(wrapper, example, name, scope, fork_less, paged):
    """``{tokens: graph}``, one piece traced at every width of :data:`IMAGE_TOKEN_BUCKETS`, its projection weights
    (every 2-D one) declared paged when ``paged`` (:func:`page_projections`). Before any width compiles, ``scope``
    settles folded every weight one of them reads with no layout fork (``fork_less`` caches that answer by program):
    the widths then build narrowest first in that scope, the order the trunk's twins build in and
    :func:`~emmy.serving.twins.twins_in_build_order` replays, so the kernels are the golden's. The plugin's tower and
    native preparation trace the tower through here alike."""
    from emmy.compiler.backend.plan_cache import binding_neutral_key  # noqa: PLC0415
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.serving.gen_runner import _traced_split  # noqa: PLC0415
    from emmy.serving.twins import fork_less_weights  # noqa: PLC0415

    traced = {tokens: _traced_split(wrapper, example(tokens), None, None) for tokens in IMAGE_TOKEN_BUCKETS}
    for tokens, graph in traced.items():
        key = binding_neutral_key(graph)
        if key not in fork_less:
            fork_less[key] = frozenset(fork_less_weights(graph, Context.probe()))
        scope.fix(f"{name}.{tokens}", fork_less[key])
        if paged:
            page_projections(graph)
    return traced


def _build(wrapper, example, name, scope, dtype, arena, plans, fork_less, paged):
    """``(tokens, program, plan)`` for one piece at every width (:func:`traced_widths`), its weights bound once on the
    device, its projection weights paged instead when ``paged``: one page each, which the programs read where the
    pager put it (``_PagedBlocks``), moving none themselves."""
    from emmy.serving.gen_runner import _compile_split  # noqa: PLC0415

    traced = traced_widths(wrapper, example, name, scope, fork_less, paged)
    for tokens in IMAGE_TOKEN_BUCKETS:
        program, plan = _compile_split(
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
        yield tokens, program, plan


def page_projections(graph):
    """Declare ``graph``'s projection weights (every 2-D one) paged, one page each that the programs read where the
    pager put it (``_PagedBlocks``) and move none of themselves: no slot of their own."""
    from emmy.compiler.dim import as_static  # noqa: PLC0415

    shapes = {nid: graph.buffer(nid).shape for nid, _ in graph.loadable_constants()}
    graph.hints.set("cuda.paged_buffers", tuple((nid, 0, as_static(shape[0]), 0) for nid, shape in shapes.items() if len(shape) == 2))


class _PagedBlocks:
    """The blocks' projection weights as paged buffers of the runtime pager, one page per block: a pool per weight
    of a piece (``pre.linear_wt``, the q/k/v projection, in the layout its kernels read), whose first ``resident``
    pages stay on the device and whose other pages live in pinned, mapped host memory, with ``slots`` device slots
    that hold the host blocks the last image ran. The blocks run in order, so the host lists the coming block's page
    itself: before a host block, the pager's step (``listed_paging``) moves its page into the slot of the least
    recently used one in every pool, and each program reads its block's page through that block's entry of the
    pool's page table. With no slot nothing moves, and every host page is read at its home."""

    def __init__(self, blocks, slots, resident):
        from emmy import emmy_runtime  # noqa: PLC0415

        self.blocks, self.slots, self.resident = blocks, slots, resident
        self.pager = emmy_runtime.Pager()
        # pool -> (its resident pages on the device, its pinned home of the rest, the plan buffer one block reads as its page)
        self.homes = {}
        self.step = None

    def __str__(self):
        nbytes = sum(raw.numel() for _, raw, _ in self.homes.values())
        on_device = self.resident + self.slots
        return f"{on_device} of {self.blocks} blocks on the device ({self.resident} resident), {nbytes / 2**30:.3f} GiB in host memory"

    def place(self, piece, block, programs, plan, sources):
        """Write block ``block``'s paged weights of ``plan`` — every width of ``piece`` reads them in one layout — into
        their pages, and point ``programs`` at those pages' entries of the page tables."""
        import numpy as np  # noqa: PLC0415

        from emmy.compiler.backend.cuda.device import device  # noqa: PLC0415
        from emmy.serving.gen_runner import _bind_plan_constants, _pinned  # noqa: PLC0415

        for nid, values in _bind_plan_constants(plan, sources, None, paged=True).items():
            pool = f"{piece}.{nid}"
            if pool not in self.homes:
                on_device = torch.empty(self.resident * values.nbytes, dtype=torch.uint8, device="cuda")
                raw, _ = _pinned(torch.empty(0, dtype=torch.uint8), (self.blocks - self.resident) * values.nbytes)
                geometry = (self.blocks, values.nbytes, self.slots, self.resident)
                self.pager.declare(device(), pool, (on_device.data_ptr(), raw.data_ptr()), geometry)
                self.homes[pool] = (on_device, raw, next(b for b in plan.buffers if b.name == nid))
            page = torch.from_numpy(np.ascontiguousarray(values).reshape(-1).view(np.uint8))
            home, at = (self.homes[pool][0], block) if block < self.resident else (self.homes[pool][1], block - self.resident)
            home[at * values.nbytes : (at + 1) * values.nbytes].copy_(page)
            table, _ = self.pager.table(pool, "table")
            for program in programs:
                program.program.executor.set_external(f"{nid}__pages", table + 8 * block, 8)

    def ready(self):
        """Build the pager's step over every pool, once every block is placed."""
        import numpy as np  # noqa: PLC0415

        from emmy.compiler.backend.cuda.program import CompiledProgram  # noqa: PLC0415
        from emmy.compiler.backend.gpu_lock import gpu_lock  # noqa: PLC0415
        from emmy.compiler.backend.paging import listed_paging  # noqa: PLC0415
        from emmy.serving.gen_runner import _Program  # noqa: PLC0415

        if not self.slots:
            return
        pools = {pool: (self.slots, spec.resolve_shape({}), spec.dtype) for pool, (_, _, spec) in self.homes.items()}
        with gpu_lock():
            program = CompiledProgram.build_from_plan(listed_paging(self.blocks, pools), {"marks": np.zeros(self.blocks + 1, np.int32)})
            for pool in pools:
                self.pager.bind(program.executor, pool)
        self.step = _Program(program, ["marks"], [])
        self.lists = torch.eye(self.blocks + 1, dtype=torch.int32, device="cuda")  # row b lists block b's page

    def move_in(self, block):
        """Move block ``block``'s pages into the device slots ahead of its programs, on the current stream: nothing for
        a resident block."""
        if self.step is not None and block >= self.resident:
            self.step.run_device([self.lists[block]])
