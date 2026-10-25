"""Native generation's image path: the vision tower's own programs, the M-RoPE columns of the trunk's rotary program,
and the merge of a prompt's image rows into its embedded rows. Native preparation exports the tower as one program per
width (:func:`tower_programs`); the ``ARCHITECTURE.md`` beside this module holds the design."""

from __future__ import annotations

import math
from functools import partial

#: How many positions the vision tower's rotary tables hold, as vLLM's tower keeps them.
VISION_ROPE_POSITIONS = 8192
#: A padding patch's additive score in the vision tower's attention, far below any image key's.
VISION_MASK = -30000.0


def mrope_axes(rope, rotary_dim):
    """Which M-RoPE position (0 temporal, 1 height, 2 width) each of a rotary table's ``rotary_dim`` columns reads, for
    a Qwen3.5 text config's ``rope_parameters``, as transformers' ``recomposition_frequencies`` interleaves them: the
    height and width sections of ``mrope_section`` take every third frequency from the second and the third, the
    temporal one the rest, and a column's two halves share their frequency."""
    sections = rope.get("mrope_section", [11, 11, 10])
    axes = [0] * (rotary_dim // 2)
    for axis in (1, 2):
        for frequency in range(axis, min(3 * sections[axis], rotary_dim // 2), 3):
            axes[frequency] = axis
    return axes * 2


def merge_module():
    """A prompt's image rows merged into its embedded rows before the trunk reads them: a row whose ``image_rows`` entry
    is an image token's index into the tower's output takes that row, every other row (-1) keeps its embedding."""
    import torch

    class Merge(torch.nn.Module):
        def forward(self, hidden, images, image_rows):
            return torch.where((image_rows >= 0).unsqueeze(1), images[image_rows.clamp(min=0)], hidden)

    return Merge()


def vision_rope_tables(head_dim, dtype):
    """The vision tower's rotary tables as vLLM's tower keeps them (``get_rope`` over half of each head, base 10,000,
    :data:`VISION_ROPE_POSITIONS` positions): FP32 angles rounded to the model ``dtype``, ``[positions, head_dim // 4]``
    each, a patch's row half then its column half of every head."""
    import torch

    half = head_dim // 2
    inverse = 1.0 / (10000 ** (torch.arange(0, half, 2, dtype=torch.float) / half))
    angles = torch.outer(torch.arange(VISION_ROPE_POSITIONS, dtype=torch.float), inverse)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def vision_attend_module(heads, head_dim, rows):
    """One tower block's attention over one image's ``rows`` patch rows, the part vLLM's tower runs between the block's
    ``pre`` and ``post`` programs: q and k rotated by each patch's row and column positions in FP32 (vLLM's
    ``ApplyRotaryEmb``, Neox halves over the whole head) and rounded to the model dtype, then attention in the model
    dtype over the image's own ``patches`` rows. The rows past them pad the program's width; a padding key is masked
    through one more score dimension, where every query holds one and a padding key :data:`VISION_MASK`, so the mask
    is a term of the score's own contraction and the expectation folds chunk by chunk, as FlashAttention does. The
    score's head is zero-padded to whole 16-wide steps."""
    import torch
    import torch.nn.functional as F

    half, width = head_dim // 2, -(-(head_dim + 1) // 16) * 16

    class Attend(torch.nn.Module):
        def forward(self, qkv, positions, patches, cosine, sine):
            q, k, v = qkv.view(rows, 3, heads, head_dim).unbind(1)
            c = torch.cat((cosine[positions[:, 0]], cosine[positions[:, 1]]), dim=-1).float().unsqueeze(1)
            s = torch.cat((sine[positions[:, 0]], sine[positions[:, 1]]), dim=-1).float().unsqueeze(1)

            def rotate(x):
                x = x.float()
                return torch.cat((x[..., :half] * c - x[..., half:] * s, x[..., half:] * c + x[..., :half] * s), dim=-1).to(qkv.dtype)

            mask = torch.where(torch.arange(rows) < patches, 0.0, VISION_MASK).to(qkv.dtype).view(rows, 1, 1).expand(rows, heads, 1)
            zeros = torch.zeros(rows, heads, width - head_dim - 1, dtype=qkv.dtype)
            q3 = torch.cat((rotate(q), torch.ones(rows, heads, 1, dtype=qkv.dtype), zeros), dim=-1).transpose(0, 1)
            k3 = torch.cat((rotate(k), mask, zeros), dim=-1).transpose(0, 1)
            out = F.scaled_dot_product_attention(q3.unsqueeze(0), k3.unsqueeze(0), v.transpose(0, 1).unsqueeze(0), scale=head_dim**-0.5)
            return out[0].transpose(0, 1).reshape(rows, heads * head_dim)

    return Attend()


def vision_positions_module(pos_embed):
    """The tower's position rows, its learned grid read at each patch's four interpolation corners, weighted and
    summed (vLLM's ``fast_pos_embed_interpolate``): the host writes the corners and their weights, in the patch rows'
    merged order, since they follow from the image's grid alone."""
    import torch

    class Positions(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.pos_embed = pos_embed

        def forward(self, index, weight):
            return (self.pos_embed(index) * weight.unsqueeze(-1)).sum(0)

    return Positions()


def tower_programs(source, td, cache, shared, scopes, traced=None):
    """The vision tower of the checkpoint ``source`` (a local directory or ``<repo>[@<revision>]``) as native programs
    in the model dtype ``td``, one per width of :data:`~emmy.serving.vision_tower.IMAGE_TOKEN_BUCKETS`
    (``tower<tokens>``), and the contract the runtime and the HTTP adapter read: ``(programs, contract)``.

    A program runs one image, its patch rows padded to the width, from the patch rows the adapter writes to the image
    rows the trunk merges (``images``): the position rows (:func:`vision_positions_module`), the plugin's patch
    embedding, each block's ``pre``, attention (:func:`vision_attend_module`) and ``post``, and the merger. The plugin's
    pieces are its tower twins (:func:`~emmy.serving.vision_tower.traced_widths`, narrowest width first in one layout
    scope per piece), so the recipe golden's tower rows are native's. Below ``EMMY_GEN_VISION_GPU`` blocks each block's
    projection weights are one page of a pool per weight, as the plugin's tower pages them: the program moves a host
    block's pages into the slots right before the block (the pager's step over a page list the program holds, one per
    block), and each block reads its page through its block's entry of the pool's page table, which the contract's
    ``pools`` names and the runtime binds. ``shared``, ``scopes`` and ``traced`` are :class:`~.prepare._Step`'s; a
    traced tower traces one block, every block being the same programs."""
    import numpy as np
    import torch

    from emmy import config
    from emmy.compiler.backend.binding import numpy_storage
    from emmy.compiler.backend.paging import _declare, _step
    from emmy.compiler.backend.plan import BufferSpec
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import I32, I64, get
    from emmy.compiler.loader.safetensors import split_revision
    from emmy.compiler.trace.huggingface import _auto_config_from_pretrained
    from emmy.serving.gen_runner import _bind_plan_constants
    from emmy.serving.native.prepare import _host, _module_sources, _Step, _weight_bytes
    from emmy.serving.twins import LayoutScope
    from emmy.serving.vision_tower import IMAGE_TOKEN_BUCKETS, hf_tower, load_piece, tower_pieces, traced_widths

    model, revision = split_revision(str(source))
    checkpoint = _auto_config_from_pretrained(model, revision=revision)
    tower = hf_tower(checkpoint, td)
    vision, depth = tower.config, len(tower.blocks)
    paths = {id(param): path for path, param in tower.named_parameters()}
    blocks = 1 if traced is not None else depth
    per_token, hidden, heads = vision.spatial_merge_size**2, vision.hidden_size, vision.num_heads
    head_dim, dtype = hidden // heads, get(str(td).removeprefix("torch."))
    patch = vision.in_channels * vision.temporal_patch_size * vision.patch_size**2
    paging = config.gen_vision_slots(depth)

    def load(wrapper):
        if traced is None:
            load_piece(wrapper, paths, str(source))
        else:
            wrapper.to_empty(device="cpu")
        return wrapper

    positions = load(vision_positions_module(tower.pos_embed))
    # Every piece traced at every width once, its weights read from the checkpoint once for both widths.
    fork_less, pieces = {}, {}
    for block in range(blocks):
        for name, (wrapper, example) in tower_pieces(tower, block).items():
            if name in ("embed", "merger") and block:
                continue
            scope = scopes.setdefault(f"tower.{name}{block}", LayoutScope())
            paged = paging is not None and name in ("pre", "post")
            pieces[name, block] = (load(wrapper), traced_widths(wrapper, example, f"{name}{block}", scope, fork_less, paged), scope)
    cosine, sine = vision_rope_tables(head_dim, td)
    # pool -> each block's buffer that reads its page, and each block's (plan, weight, sources) its page binds from.
    pools: dict[str, dict] = {}
    programs = {}
    for tokens in IMAGE_TOKEN_BUCKETS:
        rows = tokens * per_token
        inputs = ["patches", "pos_index", "pos_weight", "positions", "patches_valid"]
        step = _Step(dtype, tokens, None, None, traced, shared, scopes, inputs=inputs, outputs=["images"])
        step.buffer("patches", (rows, patch), role="input")
        step.buffer("pos_index", (4, rows), I64, "input")
        step.buffer("pos_weight", (4, rows), role="input")
        step.buffer("positions", (rows, 2), I64, "input")
        step.buffer("patches_valid", (1,), I64, "input")
        step.buffer("images", (tokens, vision.out_hidden_size), role="output")
        tables = (("cosine", cosine), ("sine", sine))
        rotary = [step.buffer(f"vision.{n}", tuple(t.shape), role="constant", data=partial(_host, t)) for n, t in tables]
        state = step.buffer("vision.positions", (rows, hidden))
        examples = (torch.zeros(4, rows, dtype=torch.int64), torch.zeros(4, rows, dtype=td))
        step.compiled("vpos", positions, examples, ["pos_index", "pos_weight"], [state], cache)

        def piece(name, block, ins, outs, step=step, tokens=tokens):
            wrapper, graphs, scope = pieces[name, block]
            sources = partial(_module_sources, wrapper)
            return step.spliced(f"t{name}{block}", graphs[tokens], ins, outs, cache, sources, scope=scope), sources

        embedded = step.buffer("vision.hidden0", (rows, hidden))
        piece("embed", 0, ["patches", state], [embedded])
        state = embedded
        for block in range(blocks):
            mark = len(step.plan.launches)
            qkv, attention = step.buffer(f"vision.qkv{block}", (rows, 3 * hidden)), step.buffer(f"vision.attention{block}", (rows, hidden))
            following = step.buffer(f"vision.hidden{block + 1}", (rows, hidden))
            pre = piece("pre", block, [state], [qkv])
            examples = (torch.zeros(rows, 3 * hidden, dtype=td), torch.zeros(rows, 2, dtype=torch.int64), torch.zeros(1, dtype=torch.int64))
            examples += (torch.zeros_like(cosine), torch.zeros_like(sine))
            module = vision_attend_module(heads, head_dim, rows)
            step.compiled(f"vattn{block}", module, examples, [qkv, "positions", "patches_valid", *rotary], [attention], cache)
            post = piece("post", block, [attention, state], [following])
            state = following
            if paging is None or traced is not None:
                continue
            # The block's projection weights are pages: name each one's pool, declare it once, and move a host block's
            # pages into the slots ahead of the block's first launch.
            moves = []
            for kind, (plan, sources) in (("pre", pre), ("post", post)):
                for nid in sorted(n for n in plan.paged if n in plan.weights):
                    pool, spec = f"vision.{kind}.{nid}", next(b for b in plan.buffers if b.name == nid)
                    elements = math.prod(spec.resolve_shape({}))
                    entry = pools.setdefault(pool, {"buffers": [], "blocks": [], "dtype": spec.dtype})
                    if len(entry["buffers"]) == block:
                        entry["buffers"].append(f"t{kind}{block}.{nid}")
                        entry["blocks"].append((plan, nid, sources))
                    if pool not in step.plan.paged:
                        step.plan.buffers.append(BufferSpec(pool, (Dim(depth), Dim(elements)), spec.dtype, "input"))
                        step.plan.paged[pool] = (0, 1, paging[0])
                        interface = list(step.plan.inputs)
                        entry["geometry"] = _declare(step.plan, pool, depth, paging[0], elements, spec.dtype)
                        # The tables are the pager's, lent at load: buffers of the program, not inputs the host binds.
                        step.plan.inputs[:] = interface
                    if block >= paging[1]:
                        marks = f"vision.marks{block}"
                        if marks not in {b.name for b in step.plan.buffers}:
                            row = np.zeros(depth + 1, np.int32)
                            row[block] = 1
                            step.buffer(marks, row.shape, I32, "constant", data=lambda row=row: row)
                        moves += _step(step.plan, f"vision.page{block}", pool, marks, entry["geometry"], 1)
            step.plan.launches[mark:mark] = moves
            # The pager's step brings scalar constants of its own, which a compiled program would bind.
            for buffer in step.plan.buffers:
                if buffer.role == "constant" and buffer.name in step.plan.constants and buffer.name not in step.bindings:
                    shape = buffer.resolve_shape({})
                    step.bind(
                        buffer.name,
                        repr((buffer.dtype.name, shape, step.plan.constants[buffer.name])),
                        partial(_weight_bytes, step.plan, buffer.name, shape, None),
                    )
        piece("merger", 0, [state], ["images"])
        programs[f"tower{tokens}"] = step

    def pages(entry):
        weights = (_bind_plan_constants(plan, sources(plan), None, paged=True)[nid] for plan, nid, sources in entry["blocks"])
        return b"".join(numpy_storage(values, entry["dtype"]).tobytes() for values in weights)

    first = programs[f"tower{IMAGE_TOKEN_BUCKETS[0]}"]
    for pool, entry in pools.items():
        first.pages[pool] = (partial(pages, entry), paging[1])
    contract = {
        "widths": list(IMAGE_TOKEN_BUCKETS),
        "image_token_id": checkpoint.image_token_id,
        "vision_start_token_id": checkpoint.vision_start_token_id,
        "patch_size": vision.patch_size,
        "temporal_patch_size": vision.temporal_patch_size,
        "merge_size": vision.spatial_merge_size,
        "position_grid": int(vision.num_position_embeddings**0.5),
        "pools": {pool: entry["buffers"] for pool, entry in pools.items()},
    }
    return programs, contract
