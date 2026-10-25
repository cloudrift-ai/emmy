"""Export Qwen3, Qwen3.5 and Qwen3.5-MoE text models as static decode and chunked prefill programs."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from functools import lru_cache, partial

import numpy as np

from emmy.compiler.backend.binding import numpy_storage
from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, plan_from_graph, splice
from emmy.compiler.dim import DEFAULT_SEQ_HINT, Dim
from emmy.compiler.dtype import F32, I64, get
from emmy.serving.release import DTYPES, KV_CACHE_DTYPES
from emmy.serving.twins import LayoutScope, plan_layouts

#: The longest context native serves: the window of the checkpoints it has served.
MAX_CONTEXT = 262144
GENERATION_VERSION = 13
PREFILL_SIZE = 16
MASK_FILL = -1e9
#: The symbolic axis attention reads the cache to: how many of its leading positions a step attends over, which the
#: runtime sets as the request grows (runtime ``ARCHITECTURE.md``, "Cached generation").
WINDOW = "cache_tokens"
#: e4m3's largest finite value: a cache store saturates to it, as vLLM's does.
FP8_MAX = 448.0
MODEL_TYPES = ("qwen3", "qwen3_5_text", "qwen3_5_moe_text")
LAYER_TYPES = ("full_attention", "linear_attention")


def load_model(model_id, revision=None, dtype="auto"):
    """The CPU model an export traces, in ``dtype`` (a ``release.DTYPES`` name, or ``auto``: BF16 for a
    checkpoint that declares it, FP16 otherwise, as vLLM serves it), and the ``ckpt`` pair its coded
    trunk binds from — ``None`` for an ordinary checkpoint, whose weights come off the module.

    A quantized checkpoint cannot go through ``from_pretrained`` (transformers would engage its own
    quantizer machinery). Its twin is built from the config with the trunk's coded linears left as
    placeholders, and every program binds them from the shards instead — the serving runner's
    checkpoint-sourced lane, which keeps an FP8 trunk at its stored size (weight-only under 16-bit
    activations) rather than doubling it in a decode. Its routed experts stay unread on the meta
    device: the export streams them from the shards a layer at a time (:func:`checkpoint_experts`)."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, GenerationConfig

    from emmy.compiler.trace.huggingface import find_text_decoder, load_quantized_split, quantized_checkpoint_dir
    from emmy.serving.gen_runner import checkpoint_key_map

    if dtype == "auto":
        config = AutoConfig.from_pretrained(model_id, revision=revision)
        declared = getattr(config, "dtype", None) or getattr(getattr(config, "text_config", None), "dtype", None)
        dtype = "bfloat16" if declared in (torch.bfloat16, "bfloat16") else "float16"
    dtype = getattr(torch, DTYPES[dtype])
    qdir = quantized_checkpoint_dir(model_id, revision)
    if qdir is None:
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype).eval().cpu(), None
    model, store = load_quantized_split(qdir, dtype, compress_trunk=True, experts=False)
    # Rotary buffers are non-persistent in transformers, so the meta-built twin has none: rebuild
    # the module on CPU, as the trace twin does, since the export reads its tables.
    decoder = find_text_decoder(model)
    decoder.rotary_emb = type(decoder.rotary_emb)(decoder.config)
    if (qdir / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(qdir)
    return model, (store["dir"], checkpoint_key_map(model, store["keys"]))


def checkpoint_experts(directory, dtype, index):
    """One MoE layer's routed experts read from a quantized checkpoint's shards, as the vLLM plugin's
    fixed-slot program takes them: ``(format, {program input: E-stacked tensor})``, an FP8 layer as its
    codes on the uint8 carrier beside their FP32 block scales (format ``f8e4m3``), any value in the
    model's torch ``dtype``."""
    from emmy.compiler.trace.huggingface import load_quantized_split

    _, store = load_quantized_split(directory, dtype, layer_range=(index, index + 1), trunk=False)
    return store["fmt"], store["layers"][index]


@dataclass(frozen=True)
class Drafter:
    """A checkpoint's multi-token-prediction head, which drafts ``drafts`` tokens a step (:func:`load_drafter`): its
    decoder layer, a full-attention layer of the target's architecture, the layer's final norm, and the prologue that
    feeds it the token's embedding and the target's normalized hidden row. ``experts_of(index)`` reads a quantized MoE
    head's routed experts as :func:`checkpoint_experts` reads a trunk layer's; ``None`` keeps them on the module."""

    layer: object
    norm: object
    prologue: object
    drafts: int
    experts_of: object = None


def load_drafter(model_id, revision, model, drafts, ckpt=None):
    """The multi-token-prediction head of checkpoint ``model_id`` for its loaded target ``model`` (:func:`load_model`),
    as the vLLM plugin's ``EmmyGenRunner.create_mtp`` loads it: the head's layer and final norm fill a one-layer
    decoder of the target's architecture from ``mtp.layers.0.`` and ``mtp.norm.``, through the shard loader, so a coded
    head keeps its codes as the trunk does, and the prologue reads ``mtp.fc`` and its two norms. Returns the
    :class:`Drafter` and the ``ckpt`` pair with the head's checkpoint keys added to the target's, so its programs bind
    from the shards as the trunk's do (``None`` for a target bound off the module)."""
    import copy

    import torch

    from emmy.compiler.loader.safetensors import _resolve_model_dir, load_sources_by_path
    from emmy.compiler.trace.huggingface import (
        _auto_config_from_pretrained,
        build_mtp_prologue,
        find_text_decoder,
        load_quantized_split,
        moe_block_parts,
        quantized_checkpoint_dir,
    )
    from emmy.serving.gen_runner import checkpoint_key_map

    if drafts < 1:
        raise ValueError("a drafter drafts at least one token a step")
    directory = quantized_checkpoint_dir(model_id, revision) or _resolve_model_dir(model_id, revision)
    text = copy.deepcopy(_auto_config_from_pretrained(directory).get_text_config())
    if not getattr(text, "mtp_num_hidden_layers", 0):
        raise ValueError(f"{model_id} has no multi-token-prediction head to draft with")
    text.num_hidden_layers, text.layer_types = 1, ["full_attention"]
    dtype = model.dtype
    head, store = load_quantized_split(
        directory, dtype, compress_trunk=True, experts=False, include_embed=False, hosted_by_engine=True, config=text, mtp=True
    )
    decoder = find_text_decoder(head)
    layer = decoder.layers[0]
    experts_of = None
    if moe_block_parts(layer.mlp) is not None:

        def experts_of(_index):
            _, experts = load_quantized_split(directory, dtype, layer_range=(0, 1), trunk=False, config=text, mtp=True)
            return experts["fmt"], experts["layers"][0]

    prologue = build_mtp_prologue(type(decoder.norm), text.hidden_size, text.rms_norm_eps)
    names = {
        "mtp.fc.weight": prologue.fc.weight,
        "mtp.pre_fc_norm_embedding.weight": prologue.norm_embedding.weight,
        "mtp.pre_fc_norm_hidden.weight": prologue.norm_hidden.weight,
    }
    stored = load_sources_by_path(str(directory), names)
    with torch.no_grad():
        for name, param in names.items():
            param.copy_(torch.from_numpy(stored[name]))
    drafter = Drafter(layer, decoder.norm, prologue.to(dtype).eval(), drafts, experts_of and lru_cache(maxsize=1)(experts_of))
    return drafter, None if ckpt is None else (ckpt[0], {**ckpt[1], **checkpoint_key_map(head, store["keys"])})


def validate_model(model, context_length, ckpt=None):
    """Reject unsupported architectures before tracing or allocating device state; return the
    text decoder, which a Qwen3.5 checkpoint wraps beside a vision tower native never loads.
    Every parameter is on the CPU in one model dtype, FP16 or BF16, which the export runs in.
    A checkpoint-sourced model (``ckpt``, :func:`load_model`) keeps its routed experts on the meta
    device, since the export streams them from the shards."""
    import torch

    from emmy.compiler.trace.huggingface import find_text_decoder, moe_block_parts

    decoder = find_text_decoder(model)
    cfg = decoder.config
    if cfg.model_type not in MODEL_TYPES or any(getattr(c, "quantization_config", None) for c in (cfg, model.config)):
        # A module that still carries its checkpoint's quantizer declaration was loaded through
        # transformers' quantizer, which the trace cannot see through; ``load_model`` is the lane.
        raise ValueError("native generation requires a Qwen3 or Qwen3.5 text model; a quantized checkpoint loads through load_model")
    if not 1 <= context_length <= min(MAX_CONTEXT, cfg.max_position_embeddings):
        raise ValueError("native context length is outside supported model limits")
    if model.training:
        raise ValueError("native export requires evaluation mode")
    rope = cfg.rope_parameters
    rotary_dim = int(cfg.head_dim * rope.get("partial_rotary_factor", 1.0))
    if rope.get("rope_type") != "default" or not 0 < rotary_dim <= cfg.head_dim or rotary_dim % 2:
        raise ValueError("native generation requires the default rotary embedding over an even leading part of each head")
    if any(kind not in LAYER_TYPES for kind in cfg.layer_types):
        raise ValueError("native generation does not support sliding attention")
    if cfg.head_dim <= 0 or cfg.num_key_value_heads <= 0 or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("invalid attention geometry")
    streamed = {id(p) for layer in decoder.layers if ckpt and (parts := moe_block_parts(layer.mlp)) for p in parts[1].parameters()}
    held = [p for p in model.parameters() if id(p) not in streamed]
    if {p.dtype for p in held} not in ({torch.float16}, {torch.bfloat16}) or any(p.device.type != "cpu" for p in held):
        raise ValueError("native export requires an FP16 or BF16 model on CPU")
    return decoder


def rope_module(heads, kv_heads, head_dim, rotary_dim, rows, cache_dtype, axes=None):
    """Rotate q and k at their positions in FP32, round once to the model dtype, and store k and v into the
    cache at those positions in ``cache_dtype``: the model dtype, or FP8 e4m3 saturated to its largest finite
    value, as vLLM stores its FP8 cache with no scale. The caches come in and go out, written in place (the
    outputs are stored over the inputs), so the step replays as one graph at every position. The caches and
    the rotary tables are token-major (``[context, kv_heads, head_dim]``, ``[context, rotary_dim]``) and whole,
    since a store may land anywhere in the context; attention reads only the window. A partial rotary embedding
    (Qwen3.5) rotates only the leading ``rotary_dim`` of each head and passes the rest through.

    With ``axes`` (:func:`mrope_axes`) a row rotates at its M-RoPE position, an ``mrope`` input of ``[rows, 3]``
    (temporal, height, width) positions: each rotary column reads the table at its axis' position. A text row's
    three are one position, so its rotation is the plain one's; an image token's differ. The cache still stores at
    the row's own position in the context."""
    import torch

    def store(x):
        if cache_dtype == x.dtype:
            return x
        return x.float().clamp(-FP8_MAX, FP8_MAX).to(cache_dtype)

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            if axes is not None:
                self.register_buffer("axes", torch.tensor(axes))

        def rotate(self, x, n, c, s):
            dtype, x = x.dtype, x.view(rows, n, head_dim).float()
            rotated, passed = (x[..., :rotary_dim], x[..., rotary_dim:]) if rotary_dim < head_dim else (x, None)
            half = rotary_dim // 2
            paired = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
            rotated = rotated * c + paired * s
            return (rotated if passed is None else torch.cat((rotated, passed), dim=-1)).to(dtype)

        def forward(self, q, k, v, position, cosine, sine, keys, values, *mrope):
            positions = position + torch.arange(rows)
            if axes is None:
                c, s = cosine[positions].unsqueeze(1), sine[positions].unsqueeze(1)
            else:
                index = torch.index_select(mrope[0], 1, self.axes)
                c, s = torch.gather(cosine, 0, index).unsqueeze(1), torch.gather(sine, 0, index).unsqueeze(1)
            rotated = self.rotate(q, heads, c, s).view(rows, heads * head_dim)
            new_keys = store(self.rotate(k, kv_heads, c, s))
            new_values = store(v.view(rows, kv_heads, head_dim))
            # Each element of the chunk names its own position.
            index = positions.view(rows, 1, 1).expand(rows, kv_heads, head_dim)
            return rotated, keys.scatter(0, index, new_keys), values.scatter(0, index, new_values)

    return Rope()


def attend_module(heads, kv_heads, head_dim, rows):
    """Causal grouped-query attention over the window's leading cache positions, every position past a row's own
    masked on the device. The scores and the softmax are FP32; the probabilities are rounded to the model dtype for
    the value expectation, which accumulates in FP32, as attention in the model dtype rounds them, so both products
    are 16-bit contractions; only the output is rounded to the model dtype. Each query head reads its key-value
    head's cache, the cache coming token-major (:func:`rope_module`)."""
    import torch

    class Attend(torch.nn.Module):
        def forward(self, q, keys, values, position):
            group, window = heads // kv_heads, keys.shape[0]
            q3 = q.view(rows, heads, head_dim).transpose(0, 1).float()
            k3 = keys.float().transpose(0, 1).repeat_interleave(group, dim=0)
            v3 = values.float().transpose(0, 1).repeat_interleave(group, dim=0)
            positions = position + torch.arange(rows)
            keep = torch.arange(window).unsqueeze(0) <= positions.unsqueeze(1)
            scores = torch.matmul(q3, k3.transpose(1, 2)) * head_dim**-0.5 + torch.where(keep, 0.0, MASK_FILL)
            probabilities = torch.softmax(scores, dim=-1).to(q.dtype).float()
            return torch.matmul(probabilities, v3).to(q.dtype).transpose(0, 1).reshape(rows, heads * head_dim)

    return Attend()


def carry_module(select=False):
    """A Gated DeltaNet layer's next state and history written over the ones the step carries. The
    layer's program is the plugin's, which writes them apart from what it reads, since some of its
    kernels read the old state after another wrote the new one; each is spelled as the carried
    buffer with every row replaced (a scatter), which the compiler stores into its memory. With
    ``select`` the program kept the state and history after each row of a verify step, and the carry
    takes the ones after the last accepted row: ``accepted`` indexes them, as the plugin reads the state
    block of a request's last accepted token."""
    import torch

    class Carry(torch.nn.Module):
        def forward(self, state_next, history_next, state, history):
            rows = torch.arange(state.shape[0])
            return state.index_copy(0, rows, state_next), history.index_copy(0, rows, history_next)

    class SelectCarry(Carry):
        def forward(self, states_next, histories_next, accepted, state, history):
            return super().forward(states_next[accepted], histories_next[accepted], state, history)

    return SelectCarry() if select else Carry()


def accept_module(drafts):
    """How many of a verify step's ``drafts`` draft tokens the target's greedy choice confirms, decided on the device:
    the step's tokens are the token it continues from and the drafts, and draft ``j`` stands when the target's pick
    at every row before it is the draft that follows, so the count is that of the leading matches. A draft that
    matches by chance is the greedy token all the same. Also the positions the drafter's later steps run at: past the
    last accepted row, one more each."""
    import torch

    class Accept(torch.nn.Module):
        def forward(self, picks, tokens, position):
            miss = (picks[:drafts] != tokens[1:]).long()
            # [prefix, row]: the row's miss where it lies inside the prefix, else none; a prefix with no miss stands.
            ids = torch.arange(drafts)
            inside = ids.unsqueeze(0) <= ids.unsqueeze(1)
            accepted = (torch.where(inside, miss.unsqueeze(0), 0).amax(dim=1) == 0).long().sum(dim=0, keepdim=True)
            return (accepted, *(position + accepted + step for step in range(1, drafts)))

    return Accept()


def pick_module(drafts):
    """The row of a verify step's drafter output after its last accepted row, which the drafter's next step reads;
    with the step's ``mrope`` positions, also the M-RoPE positions the drafter's later steps run at, past the last
    accepted row's, one more each: a verify step's rows are text past the prompt, whose three positions move
    together."""
    import torch

    class Pick(torch.nn.Module):
        def forward(self, rows, accepted, *mrope):
            if not mrope:
                return rows[accepted]
            return (rows[accepted], *(mrope[0][accepted] + step for step in range(1, drafts)))

    return Pick()


def router_module(gate, rows):
    """A MoE layer's router over the step's ``[rows, hidden]`` normalized rows, as Transformers'
    softmax top-k router computes it: the gate's linear in the model dtype, a softmax in FP32, the
    top ``k``, and the picked scores normalized to sum to one and rounded to the model dtype. The top
    ``k`` is each expert's rank, counted in one reduction over every other expert of the row (one
    outranks it with a larger logit, or an equal one and a lower ID, so the lowest ID wins a tie),
    then a scatter of every expert ranked below ``k``, and of its probability, to its rank. A logit
    that is not finite outranks every expert ``k`` times over, so such a row ranks none in its top
    ``k`` and picks the last expert in every slot: the gather stays inside the stack, its logits
    come out nonfinite and greedy selection fails the request. It returns the
    ``[rows, k]`` scores and the picks: one row's as a ``[k]`` selector, a chunk's as ``[rows, k]``."""
    import torch
    import torch.nn.functional as F

    experts, k = gate.num_experts, gate.top_k

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = gate.weight

        def forward(self, xn):
            logits = F.linear(xn, self.weight).float()
            probabilities = torch.softmax(logits, dim=-1)
            ids = torch.arange(experts, device=xn.device)
            # [row, expert, other]: whether the other expert outranks the expert. The logits order the experts as
            # their probabilities do, and are compared as stored: a probability recomputed in two places of one
            # kernel can differ in its last bit, and a tie it breaks would rank two experts alike.
            other, expert = logits.unsqueeze(1), logits.unsqueeze(2)
            outranks = (other > expert) | ((other == expert) & (ids.view(1, 1, -1) < ids.view(1, -1, 1)))
            rank = torch.where(other.abs() <= torch.finfo(torch.float32).max, outranks.long(), k).sum(dim=2)
            # Each expert ranked inside the top ``k`` lands in its slot, with its probability; slot ``k`` takes the
            # rest and is dropped. A slot no expert lands in keeps the last expert.
            slot = torch.where(rank < k, rank, k)
            sel = torch.full((rows, k + 1), experts - 1, device=xn.device).scatter(1, slot, ids.expand(rows, experts))[:, :k]
            peaks = torch.zeros(rows, k + 1, device=xn.device).scatter(1, slot, probabilities)[:, :k]
            scores = (peaks / peaks.sum(dim=1, keepdim=True)).to(xn.dtype)
            return scores, sel[0] if rows == 1 else sel

    return Router()


def combine_module():
    """The routed experts' weighted sum added to the mixer's residual: each row's ``[1, k]`` scores
    times its ``[k, hidden]`` partials, then the residual add, all in the model dtype, as the plugin's
    fixed-slot combine computes them. One row's partials come as ``[k, hidden]``, a chunk's as
    ``[rows, k, hidden]``."""
    import torch

    class Combine(torch.nn.Module):
        def forward(self, residual, scores, *partials):
            operands = scores.unsqueeze(1), partials[0] if partials[0].dim() == 3 else torch.stack(partials)
            return residual + torch.matmul(*operands).squeeze(1)

    return Combine()


def greedy_module(vocab):
    """The lowest token ID among the maximum logits, from the reductions the compiler has: the peak,
    then the largest negated ID among the tokens at the peak. A logit whose magnitude is at most the
    largest finite float is finite; an infinite one is not, nor a NaN, which no comparison admits.
    The maximum over those flags is one when any logit is nonfinite, and then the result is the
    vocabulary size, which the runtime rejects. The flag is a select over a finite bound, not
    ``x - x`` or a NaN literal: normalization folds a value minus itself to zero, and a nonfinite
    literal has no spelling on the runtime's wire."""
    import torch

    class Greedy(torch.nn.Module):
        def forward(self, logits):
            nonfinite = torch.where(logits.abs() <= torch.finfo(torch.float32).max, 0.0, 1.0).amax(dim=1)
            peak = logits.amax(dim=1, keepdim=True)
            ids = torch.arange(vocab).unsqueeze(0)
            lowest = -torch.where(logits == peak, -ids, -vocab).amax(dim=1)
            return torch.where(nonfinite < 0.5, lowest, vocab)

    return Greedy()


def _host(t, dtype=None):
    """A tensor's values as NumPy holds them, BF16 widened to the FP32 values it holds exactly (NumPy has no BF16);
    with a buffer ``dtype``, their storage in it, a BF16 tensor's own bits for a BF16 buffer: the float32 copy of a
    vocabulary-sized table would hold twice its size again."""
    import torch

    t = t.detach().cpu()
    if t.dtype == torch.bfloat16 and getattr(dtype, "name", None) == "bf16":
        return t.contiguous().view(torch.int16).numpy().view(np.uint16)
    values = (t.float() if t.dtype == torch.bfloat16 else t).numpy()
    return values if dtype is None else numpy_storage(values, dtype)


def _leading_rows(program, seam):
    """Whether a program buffer of shape ``program`` holds the leading rows of the step buffer of shape ``seam``: rows
    of the same shape, no more of them. A decode step runs native's programs at one row and the plugin's at the
    width it decodes one token at, so the seams between them hold that width and a one-row program reads or writes
    their first row; the rows past it are padding, which the plugin's programs compute row by row and nothing
    reads. A program buffer of the step buffer's every element in another shape reads it whole, reshaped, as the
    vision tower's merger reads each image token's patches side by side."""
    tail = len(seam) - 1
    leading = tuple(program[len(program) - tail :]) == tuple(seam[1:]) and math.prod(program[: len(program) - tail]) <= seam[0]
    return leading or math.prod(program) == math.prod(seam)


def _module_sources(wrapper, plan):
    """The tensors of ``wrapper`` the weights of ``plan`` bind from, by their paths in it, as NumPy holds their values."""
    paths = {p for w in plan.weights.values() for p in (w.source_path, *(p for p, _ in w.source_parts))}
    named = [*wrapper.named_parameters(remove_duplicate=False), *wrapper.named_buffers(remove_duplicate=False)]
    return {path: _host(t) for path, t in named if path in paths}


def _weight_bytes(plan, nid, shape, sources):
    """The bytes of plan constant ``nid`` at ``shape``: its weight read from ``sources(plan)`` (the program's sources,
    for a plan holding that one weight), or its scalar."""
    weight = plan.weights.get(nid)
    dtype = next(b.dtype for b in plan.buffers if b.name == nid)
    if weight is None:
        value = plan.constants[nid]
    else:
        from emmy.serving.gen_runner import _bind_plan_constants

        one = replace(plan, weights={nid: weight})
        value = _bind_plan_constants(one, sources(one), None)[nid]
    return np.broadcast_to(numpy_storage(value, dtype), shape).tobytes()


class _Step:
    def __init__(self, dtype, rows, context_length, ckpt=None, traced=None, shared=None, scopes=None, inputs=(), outputs=()):
        # A decode step (one row) ends at the logits and the greedy token; the runtime hands the token it
        # selects back as the next step's input. A prefill chunk writes the cache and nothing else.
        # ``ckpt`` (``load_model``) puts every program on the checkpoint-sourced lane: the trunk's
        # coded weights are spelled from the shards and bound from them, never off the module.
        # ``dtype`` is the model's, which every 16-bit buffer takes. With ``traced``, a dict, each
        # distinct program is traced into it, named by its kind and width, and nothing is compiled.
        # ``shared`` and ``scopes`` are the export's, across its programs: each constant's bytes and the
        # layouts each of the plugin's wrappers binds its weights in. The window's hint is the context, so
        # the runtime allocates every buffer it sizes for the whole context; the vision tower's step has no window.
        # ``inputs`` and ``outputs`` are the step's: the ones the host binds and reads.
        self.dtype = dtype
        self.rows = rows
        self.ckpt = ckpt
        self.traced = traced
        self.plan = ExecutionPlan("cuda", list(inputs), list(outputs), [], {}, {}, [], {})
        if context_length is not None:
            self.plan.symbolic_hints[WINDOW] = context_length
        # Each constant's bytes, as a function the pack calls when it writes them, so the export never holds the
        # trunk; a constant another program binds alike takes its function, and the pack keeps it once.
        self.bindings = {}
        self.shared = {} if shared is None else shared
        self.scopes = {} if scopes is None else scopes
        # The buffers the pager holds: a function returning their bytes, page after page, and how many of their first
        # pages are resident; the pack keeps them once for every program.
        self.pages = {}
        # The buffers a step writes in place and carries to the next position; a request starts them at zero.
        self.state = []
        # The kernels whose evidence the step's compiles read (``evidence_reads``), which the pack keeps.
        self.reads = set()
        # The buffers :meth:`buffer` declared, by name.
        self.declared = {}

    def bind(self, name, key, read):
        """Bind constant ``name`` to the bytes ``read()`` returns; a constant of that name and ``key`` bound by another
        of the export's programs shares that program's function."""
        self.bindings[name] = self.shared.setdefault((name, key), read)

    def buffer(self, name, shape, dtype=None, role="scratch", data=None):
        """Declare a step buffer; ``data``, a function returning its host values, binds it. A traced step binds
        nothing: it compiles nothing either. A buffer declared again, alike, is the one already declared: two passes of
        a layer in one step share its cache and weights."""
        dtype = dtype or self.dtype
        spec = BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role)
        if name in self.declared:
            if self.declared[name] != spec:
                raise ValueError(f"step buffer {name} declared twice, differently")
            return name
        self.declared[name] = spec
        self.plan.buffers.append(spec)
        if data is not None and self.traced is None:
            self.bind(name, (dtype.name, tuple(shape)), lambda: numpy_storage(data(), dtype).tobytes())
        return name

    def compiled(
        self,
        prefix,
        wrapper,
        examples,
        inputs,
        outputs,
        cache,
        output_names=(),
        paged=(),
        coded=None,
        twin=False,
        window=(),
        weights=None,
        rows=None,
    ):
        """Trace ``wrapper`` at ``examples`` and splice it in (:meth:`spliced`). ``coded`` spells named weight inputs
        as FP8 codes with a scale input appended after the traced ones (``spell_quantized_inputs``), as the plugin
        spells its expert programs. ``twin`` says the program is one of the plugin's serving twins, traced as the
        plugin traces it: its weights take the layouts of the plugin's wrapper ``weights`` (``prefix`` by default),
        and it is no native twin. ``window`` names the arguments whose leading axis is the window (:data:`WINDOW`):
        the program reads their leading rows, as many as the runtime sets. ``rows``, the program's own rows where
        they are not the step's, names a traced program."""
        import inspect

        import torch

        from emmy.compiler.loader.quant import spell_quantized_inputs
        from emmy.compiler.trace.torch import trace_module
        from emmy.serving.gen_runner import _plan_sources, _traced_split

        if window:
            axis = torch.export.Dim(WINDOW, min=1, max=MAX_CONTEXT)
            shapes = {name: {0: axis} if name in window else {} for name in inspect.signature(wrapper.forward).parameters}
            graph = trace_module(wrapper, tuple(examples), dynamic_shapes=shapes)
        else:
            graph = _traced_split(wrapper, examples, None, self.ckpt)
        if coded:
            spell_quantized_inputs(graph, coded)
        if self.ckpt is not None:
            sources = partial(_plan_sources, wrapper=wrapper, dtype=self.dtype, ckpt_dir=self.ckpt[0], id_to_key=self.ckpt[1])
        else:
            sources = partial(_module_sources, wrapper)
        scope = self.scopes.setdefault(weights or prefix, LayoutScope()) if twin else None
        self.spliced(
            prefix, graph, inputs, outputs, cache, sources, output_names=output_names, paged=paged, scope=scope, weights=weights, rows=rows
        )

    def spliced(
        self, prefix, graph, inputs, outputs, cache, sources, *, output_names=(), paged=(), scope=None, sizes=None, weights=None, rows=None
    ):
        """Compile ``graph`` and splice its plan in: its inputs and outputs take the step's names,
        everything else is scoped by ``prefix``; its constants bind from ``sources(plan)``. ``output_names``
        renames the traced outputs (the tracer names them after their last op) so ``paged`` can address them: it
        names the graph's buffers that are the step's paged buffers, as ``(graph name, axis, page)`` for a
        cache the device holds whole, or ``(graph name, axis, page, slots)`` for one the pager holds
        in that many device slots, and the step buffer takes that paging. A cache a program writes in
        place comes in as one graph buffer and goes out as another stored over it; both take the step
        buffer's name, so its pages are the ones written. A buffer the pager holds brings its tables
        and its view, which take its step name too: every launch that reads it, in every program,
        moves its pages through the same tables. ``scope`` is the plugin wrapper's :class:`LayoutScope` for
        one of its twins: the program compiles under the layouts its other twins settled and settles the rest.
        ``sizes`` gives a symbolic program its sizes. ``weights`` names the program's constants in place of
        ``prefix``, so a module spliced at several sizes into one step, under a prefix each, binds its weights once.
        A program input or output may hold the leading rows of a wider step buffer (:func:`_leading_rows`).
        Everything is spelled in the graph's own names, never the step's, so every layer traces the same graph and
        compiles once. ``rows`` names a traced program as :meth:`compiled` does."""
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.paging import TABLES
        from emmy.compiler.backend.plan_cache import binding_neutral_key
        from emmy.compiler.pipeline.passes.tile._layout import settled_layouts
        from emmy.compiler.pipeline.search.policy.greedy import evidence_reads

        for old, new in zip(list(graph.outputs), output_names, strict=False):
            graph.rename_node(old, new)
        if paged:
            graph.hints.set("cuda.paged_buffers", tuple(paged))
        if self.traced is not None:
            # Every layer of one shape traces one program, which the first names by its kind and width (``-sym`` for a
            # program of any width), and says whether it is the plugin's.
            name = f"{prefix.rsplit('.', 1)[-1].rstrip('0123456789')}{'-sym' if sizes else rows or self.rows}"
            self.traced.setdefault(binding_neutral_key(graph), (name, graph, scope is not None))
            return
        settled = tuple(scope.settled().items()) if scope is not None else ()

        def compile_plan(g):
            with settled_layouts(dict(settled)):
                return plan_from_graph(CudaBackend(tune_db="auto").compile(g))

        with evidence_reads() as read:
            plan = cache.resolve(graph, compile_plan, settled)
        self.reads |= read
        if scope is not None:
            scope.record(prefix, plan_layouts(plan))
        # A serial launch's axes and the paging program's sizes, which the plan hints, are the runtime
        # arguments the standalone runtime resolves itself, and so are the sizes a symbolic program is given and
        # the window, which the runtime sets. A TMA descriptor is encoded once from its source's fixed address, so
        # it may not read pages.
        why = ["symbolic bindings"] if set(plan.symbolic_bindings) - {WINDOW} and sizes is None else []
        why += ["runtime constants"] if plan.runtime_constants else []
        for launch in plan.launches:
            if {t.src_buf for t in launch.tma_descriptors} & plan.paged.keys():
                why.append(f"{launch.node_id} reads a paged buffer through a TMA descriptor")
            given = {name for name, _ in launch.serial} | plan.symbolic_hints.keys() | (sizes or {}).keys()
            if extra := set(launch.runtime_args) - given:
                why.append(f"{launch.node_id} takes runtime arguments {sorted(extra)}")
        if why:
            raise ValueError(f"native generation requires static ordinary-pointer compiled programs; {prefix}: " + "; ".join(why))
        # The tables and view of a buffer the pager holds enter as the inputs the pager lends.
        held = {n: [f"{n}__{part}" for part in (*TABLES, "view")] for n in plan.paged if f"{n}__table" in plan.inputs}
        pager = {part for parts in held.values() for part in parts}
        renames = dict(zip([n for n in plan.inputs if n not in pager], inputs, strict=True))
        renames.update(zip(plan.outputs, outputs, strict=True))
        renames.update({part: renames[n] + part[len(n) :] for n, parts in held.items() for part in parts})
        if weights is not None:
            renames.update({b.name: f"{weights}.{b.name}" for b in plan.buffers if b.role == "constant" and b.name not in renames})
        existing = {b.name: b for b in self.plan.buffers}
        # A program buffer is compared at the window the step's buffers are sized for: the context.
        resolved = {**self.plan.symbolic_hints, **(sizes or {})}
        added = []
        for buffer in plan.buffers:
            name = renames.get(buffer.name, f"{prefix}.{buffer.name}")
            seam = existing.get(name)
            if seam is None:
                added.append((buffer, name))
            elif (
                buffer.name not in plan.paged
                and name not in self.plan.paged
                and (seam.dtype != buffer.dtype or not _leading_rows(buffer.resolve_shape(resolved), seam.resolve_shape({})))
            ):
                raise ValueError(f"incompatible program seam: {name}")
        self.plan.buffers += [replace(buffer, name=name) for buffer, name in added if buffer.name in pager]
        self.plan.launches += splice(self.plan, plan, prefix=prefix, renames=renames, sizes=sizes)
        for buffer, name in added:
            # A paged constant is bound by its page table: its bytes are pages of the pager (the vision tower's blocks).
            if buffer.role == "constant" and buffer.name not in plan.paged:
                if buffer.name not in plan.weights and buffer.name not in plan.constants:
                    raise ValueError(f"unresolved constant: {name}")
                shape = buffer.resolve_shape(sizes or {})
                weight = plan.weights.get(buffer.name)
                spec = plan.constants.get(buffer.name) if weight is None else (weight.source_path, weight.source_parts, weight.generated)
                key = repr((buffer.dtype.name, shape, spec, None if weight is None else weight.load_ops))
                self.bind(name, key, partial(_weight_bytes, plan, buffer.name, shape, sources))
        return plan


def _paged_table(t, dtype, rows):
    """A table's storage page after page, its last page padded with zero rows to ``rows``: the pager's home holds
    whole pages."""
    values = _host(t, dtype)
    pad = -values.shape[0] % rows
    return np.concatenate([values, np.zeros((pad, *values.shape[1:]), values.dtype)]) if pad else values


def _coded_head(model, ckpt):
    """The NVFP4 output head a checkpoint keeps coded, as the plugin keeps it (``CodedHead``): its spec and a function
    reading its codes from the shards, or ``None`` for a head the model holds."""
    from emmy.compiler.loader.safetensors import load_sources_by_path
    from emmy.serving.coded_head import NVFP4_HEAD_LEAVES, Nvfp4HeadSpec, nvfp4_head_stored

    cfg = model.config
    if ckpt is None or not nvfp4_head_stored(load_sources_by_path(ckpt[0], ["lm_head.weight_scale_2"])):
        return None
    spec = Nvfp4HeadSpec.from_config(
        hidden_size=cfg.hidden_size,
        vocab_size=cfg.vocab_size,
        tensor_parallel_size=1,
        tied=getattr(cfg, "tie_word_embeddings", False),
        dtype=str(model.dtype).removeprefix("torch."),
    )
    paths = [f"lm_head.{leaf}" for leaf in NVFP4_HEAD_LEAVES]
    return spec, lru_cache(maxsize=1)(lambda: spec.feed(load_sources_by_path(ckpt[0], paths, bits_paths={"lm_head.weight_scale"})))


def _program(
    kind,
    model,
    decoder,
    context_length,
    rows,
    width,
    page_tokens,
    kv_dtype,
    cache,
    ckpt,
    experts_of,
    shared,
    scopes,
    traced=None,
    drafter=None,
    images=None,
):
    """One program of ``kind``: ``decode``, the one-token step at ``rows == 1``; ``prefill``, a chunk of ``rows``
    prompt tokens that writes the cache and the carried state and computes nothing past the last
    attention layer's keys; or ``verify``, a speculative step over the token it continues from and ``drafter``'s
    drafts, ``rows`` in all, which takes the drafts the target's greedy choice confirms, carries the state after the
    last accepted row and drafts the next step's tokens. The trunk is the vLLM plugin's: its serving twins, in the
    model dtype, the attention halves at ``width`` rows (the width the plugin runs one token at: its decode bucket
    without its M=1 tier; the chunk itself for prefill) and a Gated DeltaNet layer at the step's own rows, as the
    plugin decomposes a step, so the recipe golden's rows are the trunk's. Native's own programs run at
    ``rows`` and read and write the leading rows of the wider seams. The cache holds ``kv_dtype`` values
    (a torch dtype). ``experts_of(index)`` reads a checkpoint-sourced MoE layer's routed experts as
    :func:`checkpoint_experts` does; without it they come off the module. With a ``drafter`` (:class:`Drafter`) the
    decode and prefill programs also write its cache at their rows, from the token after each one, which the host
    uploads, so the prompt's positions are in it when the first verify step drafts. ``shared`` and ``scopes`` are
    :class:`_Step`'s; ``traced`` traces the programs into it instead of compiling them.

    ``images`` is how many image rows the vision tower writes at most (its widest program's), for a step that
    takes images: the step then also takes each row's M-RoPE position (``mrope``, ``[rows, 3]``) and the image row it
    takes in place of its embedding (``image_rows``, -1 for none), which the host uploads with the tokens, and
    reads the tower's output (``images``, a buffer decode owns and the tower writes). A verify step's rows are past
    the prompt: it takes their positions and no image rows. The drafter embeds an image token as its token, as the
    plugin's drafter does."""
    import torch

    from emmy import config
    from emmy.compiler.loader.quant import scale_layout
    from emmy.compiler.trace.huggingface import (
        build_attention_split_wrapper,
        build_gdn_state_wrapper,
        build_moe_slots_wrapper,
        build_moe_split_wrapper,
        moe_block_parts,
        selected_layer_type,
    )
    from emmy.serving.coded_head import head_graph
    from emmy.serving.gen_runner import EMBED_PAGE_ROWS, embed_gather_graph, trace_split

    cfg = decoder.config
    prefill, verify = kind == "prefill", kind == "verify"
    drafts = drafter.drafts if verify else 0
    # The step's position and the tokens it embeds, which the host uploads: the prompt's while it lasts, then the
    # previous step's selection; a verify step's are the token it continues from and the drafts. Past a verify step
    # the drafter needs no token from the host: it embeds the target's picks, then its own drafts.
    inputs = ["position", "tokens", *(["next_tokens"] if drafter is not None and not verify else [])]
    merged = bool(images) and not verify
    inputs += ["mrope", "image_rows"] if merged else ["mrope"] if images else []
    drafted = [f"draft{j}" for j in range(1, drafts + 1)]
    outputs = [] if prefill else ["logits", "token", *(["accepted", *drafted] if verify else [])]
    # The model dtype: every 16-bit activation, cache and weight the step holds, as the plugin serves it.
    td = model.dtype
    step = _Step(get(str(td).removeprefix("torch.")), rows, context_length, ckpt, traced, shared, scopes, inputs, outputs)
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    for name in inputs:
        step.buffer(name, {"position": (1,), "mrope": (rows, 3)}.get(name, (rows,)), I64, "input")
    if merged:
        step.buffer("images", (images, h), role="output")
    if not prefill:
        step.buffer("logits", (rows, vocab), F32, "output")
        step.buffer("token", (rows,), I64, "output")
    for name in outputs[2:]:
        step.buffer(name, (1,), I64, "output")

    # Example inputs are distinct tensors: the tracer folds two arguments that share one into a
    # single aliased input. A windowed one's leading axis is the window, traced at its hint.
    def scalar():
        return torch.zeros(1, dtype=torch.int64)

    def head_rows(n, heads):
        return torch.zeros(n, heads * d, dtype=td)

    def windowed():
        return torch.zeros(DEFAULT_SEQ_HINT, kv, d, dtype=kv_dtype)

    def mixer_outputs(label, out, routed, n, role="scratch"):
        # A MoE layer's mixer leaves its residual and the normalized rows its routed experts read; the
        # experts write the layer's output.
        if routed:
            return [step.buffer(f"{label}.h", (n, h), role=role), step.buffer(f"{label}.xn", (n, h), role=role)]
        return [step.buffer(out, (width, h), role=role)]

    def experts(label, home, prefix, routed, expert, mixed, normed, out, n, read, weights=None, paged=True):
        """The routed half of a MoE layer after its mixer, which leaves the residual ``mixed``
        with the shared expert folded in and the normalized rows ``normed``, over ``n`` rows: the router, the
        fixed-slot program of the ``expert`` form over the layer's E-stacked expert inputs, every
        row's picks at once, and the combine, which writes the layer's output ``out``. ``routed`` is the layer's
        ``(router, experts)`` pair; ``label`` scopes the pass's activations, ``home`` the layer's expert inputs,
        ``prefix(kind)`` names each program and ``weights(kind)`` its weights (by default the program's name).
        ``read()`` reads a checkpoint-sourced layer's experts (:func:`checkpoint_experts`); ``None`` takes them off
        the module. FP8 experts stay coded: each weight is its codes beside its block scales, which the program
        decodes in-graph as the plugin's does. The inputs are
        constants of the step when the device holds every expert; below that budget they are paged one expert per
        page (``config.gen_expert_slots``): the budget's first experts resident on the device, the rest in
        host memory, and a token's picks of device slots holding the most recently routed of those; ``paged`` off
        keeps them on the device whatever the budget. One row runs the plugin's own one-row fixed-slot program."""
        gate, stacked = routed
        k = gate.top_k
        if read is None:
            fmt, tensors = None, {"w_gate_up": stacked.gate_up_proj.detach(), "w_down": stacked.down_proj.detach()}
        else:
            fmt, tensors = read()
        if fmt not in (None, "f8e4m3", "f8e5m2"):
            raise ValueError(f"native generation runs 16-bit or FP8 routed experts, not {fmt}")
        coded = {n: (fmt, tuple(tensors[f"{n}_scale"].shape), "f32") for n in ("w_gate_up", "w_down")} if fmt else {}
        paging = config.gen_expert_slots(gate.num_experts, k) if paged else None
        paged = paging is not None
        names = ["w_gate_up", "w_down", *(f"{name}_scale" for name in coded)]
        inputs = [f"{home}.{name}" for name in names]
        for weight, name in zip(inputs, names, strict=True):
            t = tensors[name]
            if name in coded:
                shape, dtype = tuple(t.shape), get(fmt)
            elif name.endswith("_scale"):
                # Declared as the decode reads it: the block scale broadcast over the codes' blocks.
                shape, dtype = scale_layout(tuple(tensors[name.removesuffix("_scale")].shape), tuple(t.shape))[0], F32
            else:
                shape, dtype = tuple(t.shape), step.dtype
            # Read when the pack is written, a layer at a time, as the pages are.
            values = partial(_host, t) if read is None else partial(lambda read, name: _host(read()[1][name]), read, name)
            step.buffer(weight, shape, dtype, role="input" if paged else "constant", data=None if paged else values)
            if paged:
                step.pages[weight] = (partial(lambda values, d: numpy_storage(values(), d), values, dtype), paging[1])
        # One row's picks are a ``[k]`` selector, a chunk's ``[rows, k]``.
        picks = (k,) if n == 1 else (n, k)
        scores = step.buffer(f"{label}.scores", (n, k))
        sel = step.buffer(f"{label}.sel", picks, I64)
        examples = (torch.zeros(n, h, dtype=td),)
        step.compiled(
            prefix("router"),
            router_module(gate, n),
            examples,
            [normed],
            [scores, sel],
            cache,
            weights=weights and weights("router"),
            rows=n,
        )
        partials = step.buffer(f"{label}.partials", (*picks, h))
        # The trace sees the weights' values, whatever stores them: the model dtype at the codes' shape.
        values = (torch.zeros(tensors[name].shape, dtype=td) for name in ("w_gate_up", "w_down"))
        examples = (torch.zeros(n, h, dtype=td), torch.zeros(picks, dtype=torch.int64), *values)
        declared = tuple((name, 0, 1, paging[0]) for name in names) if paged else ()
        wrapper = build_moe_slots_wrapper(expert)
        step.compiled(
            prefix("experts"),
            wrapper,
            examples,
            [normed, sel, *inputs],
            [partials],
            cache,
            paged=declared,
            coded=coded,
            twin=n == 1,
            weights=weights and weights("experts"),
            rows=n,
        )
        output = step.buffer(out, (width, h))
        examples = (torch.zeros(n, h, dtype=td), torch.zeros(n, k, dtype=td), torch.zeros(*picks, h, dtype=td))
        step.compiled(prefix("combine"), combine_module(), examples, [mixed, scores, partials], [output], cache, rows=n)
        return output

    # The token-embedding table: a constant, or below the plugin's page budget (``EMMY_GEN_EMBED_GPU``) a buffer the
    # pager holds in that many device slots, its pages in host memory, read in place at zero slots. The gather is the
    # plugin's program over it, its ``embed-sym`` twin when the table is paged.
    table = decoder.embed_tokens.weight
    slots = config.gen_embed_slots(-(-vocab // EMBED_PAGE_ROWS))
    if slots is None:
        weight = step.buffer("embed.weight", tuple(table.shape), role="constant", data=partial(_host, table, step.dtype))
    else:
        weight = step.buffer("embed.weight", tuple(table.shape), role="input")
        step.pages[weight] = (partial(_paged_table, table, step.dtype, EMBED_PAGE_ROWS), slots[1])
    gather = embed_gather_graph(vocab, h, td)
    paged = () if slots is None else (("weight", 0, EMBED_PAGE_ROWS, slots[0]),)

    def embed(prefix, ids, n, out):
        """The rows of the table at ``n`` token IDs ``ids``, written to the leading rows of ``out``."""
        scope = None if slots is None else LayoutScope()
        step.spliced(prefix, gather, [ids, weight], [out], cache, lambda _plan: {}, paged=paged, scope=scope, sizes={"num_tokens": n})

    hidden = step.buffer("hidden0", (width, h))
    embedded = step.buffer("embedded", (width, h)) if merged else hidden
    embed("embed", "tokens", rows, embedded)
    if merged:
        from emmy.serving.native.vision import merge_module

        examples = (torch.zeros(rows, h, dtype=td), torch.zeros(images, h, dtype=td), torch.zeros(rows, dtype=torch.int64))
        step.compiled("merge", merge_module(), examples, [embedded, "images", "image_rows"], [hidden], cache)
    # The rotary tables, one copy for every layer, from the checkpoint's own module in FP32; and the cache, a key and
    # a value buffer per attention layer, token-major and allocated for the whole context. It is paged along its
    # positions only when a page is shorter than the context: a page spanning it is the flat buffer.
    with torch.no_grad():
        tables = [t[0] for t in decoder.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float32), torch.arange(context_length).reshape(1, -1))]
    rotary = [step.buffer(f"rope.{n}", tuple(t.shape), F32, "constant", t.numpy) for n, t in zip(("cosine", "sine"), tables, strict=True)]
    if images:
        from emmy.serving.native.vision import mrope_axes

    axes = mrope_axes(cfg.rope_parameters, tables[0].shape[-1]) if images else None
    pages = page_tokens < context_length
    cache_type = get(str(kv_dtype).removeprefix("torch."))

    def attention(
        label, home, prefix, layer, split, hidden, n, position, out, cache_only=False, read=None, weights=None, paged=True, mrope="mrope"
    ):
        """A full-attention layer's pass over ``n`` rows at ``position``, of ``hidden``: ``label`` scopes its
        activations, ``home`` its cache and a routed layer's expert inputs, ``prefix(kind)`` names each program and
        ``weights(kind)`` its weights. Returns the layer's output, ``out``; ``cache_only`` writes the cache and
        stops. ``paged`` off keeps a routed layer's experts on the device (:func:`experts`). A step that takes
        images rotates the rows at their M-RoPE positions, ``mrope``."""
        routed = moe_block_parts(layer.mlp)
        pre, post = split[:2] if split else build_attention_split_wrapper(layer)
        # Outputs nothing reads take the persistent role rather than a scratch slot nobody consumes: a pass that only
        # writes its cache leaves its rotated query and gate.
        unread = "output" if cache_only else "scratch"
        # Qwen3.5's query projection also carries the attention output gate; ``pre`` emits it as a
        # fourth tensor and ``post`` takes it as a third input.
        gated = (("gate", heads),) if pre.emits_gate else ()
        names = [step.buffer(f"{label}.{name}", (width, m * d)) for name, m in (("q", heads), ("k", kv), ("v", kv))]
        names += [step.buffer(f"{label}.{name}", (width, m * d), role=unread) for name, m in gated]
        step.compiled(
            prefix("pre"), pre, (torch.zeros(width, h, dtype=td),), [hidden], names, cache, twin=True, weights=weights and weights("pre")
        )
        rotated = step.buffer(f"{label}.rotated", (n, heads * d), role=unread)
        # The cache, written a chunk at a time at ``position`` — a scatter into the cache the rope program takes in
        # and hands back — and read to the window.
        keys, values = (step.buffer(f"{home}.{name}", (context_length, kv, d), cache_type, "output") for name in ("keys", "values"))
        paging = ((name, 0, page_tokens) for name in ("keys", "values", "keys_next", "values_next")) if pages else ()
        examples = (head_rows(n, heads), head_rows(n, kv), head_rows(n, kv), scalar(), *map(torch.zeros_like, tables))
        examples += (torch.zeros(context_length, kv, d, dtype=kv_dtype), torch.zeros(context_length, kv, d, dtype=kv_dtype))
        examples += (torch.zeros(n, 3, dtype=torch.int64),) if images else ()
        step.compiled(
            prefix("rope"),
            rope_module(heads, kv, d, tables[0].shape[-1], n, kv_dtype, axes),
            examples,
            [*names[:3], position, *rotary, keys, values, *([mrope] if images else [])],
            [rotated, keys, values],
            cache,
            output_names=("rotated", "keys_next", "values_next"),
            paged=tuple(paging),
            rows=n,
        )
        if cache_only:
            return None
        attended = step.buffer(f"{label}.attention", (width, heads * d))
        step.compiled(
            prefix("attend"),
            attend_module(heads, kv, d, n),
            (head_rows(n, heads), windowed(), windowed(), scalar()),
            [rotated, keys, values, position],
            [attended],
            cache,
            paged=(("keys", 0, page_tokens), ("values", 0, page_tokens)) if pages else (),
            window=("keys", "values"),
            rows=n,
        )
        outputs = mixer_outputs(label, out, routed, width)
        examples = (head_rows(width, heads), torch.zeros(width, h, dtype=td), *(head_rows(width, heads) for _ in gated))
        step.compiled(
            prefix("post"), post, examples, [attended, hidden, *names[3:]], outputs, cache, twin=True, weights=weights and weights("post")
        )
        return experts(label, home, prefix, routed, split[2], *outputs, out, n, read, weights, paged) if routed else outputs[0]

    # A verify step keeps each Gated DeltaNet layer's state after every row and carries the one after the last
    # accepted row once the step has decided.
    carries = []
    for index, layer in enumerate(decoder.layers):
        # The last prefill layer's outputs that nothing reads take the persistent role rather than
        # a scratch slot nobody consumes: a Gated DeltaNet layer's hidden output, since the layer
        # runs whole for its state, and an attention layer's rotated query and gate, since only its
        # keys and values are needed in the cache. A drafter reads the last layer's output.
        last_prefill = prefill and drafter is None and index + 1 == len(decoder.layers)
        unread = "output" if last_prefill else "scratch"
        # A MoE layer's split, built once: its attention ``pre`` and ``post`` (none for a Gated
        # DeltaNet layer) and its experts' form.
        routed = moe_block_parts(layer.mlp)
        split = build_moe_split_wrapper(layer) if routed else None
        read = None if experts_of is None else partial(experts_of, index)
        label, out = f"layer{index}", f"hidden{index + 1}"

        def prefix(kind, index=index):
            return f"{kind}{index}"

        if selected_layer_type(decoder, layer, index) == "linear_attention":
            # A Gated DeltaNet layer always runs whole, the last prefill layer included: its state must advance. The
            # plugin's program writes the next state and history apart from the ones it reads; the carry writes them
            # over the ones the step carries. A verify step runs the plugin's program that keeps them after each row.
            mixer = layer.linear_attn
            shapes = {
                "state": ((1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim), F32, torch.float32),
                "history": ((1, mixer.conv_dim, mixer.conv_kernel_size), step.dtype, td),
            }
            kept = rows if verify else 1
            state = [step.buffer(f"{label}.{name}", shape, dtype, role="output") for name, (shape, dtype, _) in shapes.items()]
            nexts = [step.buffer(f"{label}.{name}_next", (kept, *shape[1:]), dtype) for name, (shape, dtype, _) in shapes.items()]
            step.state += state
            outputs = mixer_outputs(label, out, routed, rows, unread)
            examples = [torch.zeros(1, rows, h, dtype=td), *(torch.zeros(shape, dtype=dtype) for shape, _, dtype in shapes.values())]
            wrapper = build_gdn_state_wrapper(layer, rows=verify)
            step.compiled(f"gdn{index}", wrapper, examples, [hidden, *state], [*outputs, *nexts], cache, twin=True)
            examples = [torch.zeros((kept, *shape[1:]), dtype=dtype) for shape, _, dtype in shapes.values()]
            examples += [*([scalar()] if verify else []), *(torch.zeros(shape, dtype=dtype) for shape, _, dtype in shapes.values())]
            carry = (f"carry{index}", carry_module(select=verify), examples, [*nexts, *(["accepted"] if verify else []), *state], state)
            if verify:
                carries.append(carry)
            else:
                step.compiled(*carry, cache)
            # The last prefill layer's output is read by nothing, so its experts never run.
            hidden = (
                experts(label, label, prefix, routed, split[2], *outputs, out, rows, read) if routed and not last_prefill else outputs[0]
            )
            continue
        hidden = attention(label, label, prefix, layer, split, hidden, rows, "position", out, last_prefill, read)
    if prefill and drafter is None:
        return step

    # The final norm, then the head: the plugin's program over the codes of a head the checkpoint keeps coded, else
    # FP32 scores from 16-bit operands, since rounding the logits to 16 bits could tie distinct scores. A drafter reads
    # the normalized rows: a decode step or a prefill chunk writes its cache from them.
    normed = step.buffer("normed", (rows, h))
    step.compiled("norm", decoder.norm, (torch.zeros(rows, h, dtype=td),), [hidden], [normed], cache)

    def draft(label, tokens, hidden, n, position, cache_only=False, mrope="mrope"):
        """The drafter's layer over ``n`` rows at ``position`` (``mrope`` in a step that takes images): the prologue of
        each row's next token (``tokens``, embedded) and its normalized hidden row (``hidden``: the target's, or the
        drafter's previous step's), then the layer, then its final norm. ``label`` scopes the pass's activations; the
        drafter's cache, weights and expert inputs are ``mtp``'s whatever the pass. Returns the normalized rows;
        ``cache_only`` writes the cache and stops."""

        def prefix(kind):
            return f"{label}.{kind}"

        def weights(kind):
            return f"mtp.{kind}"

        embeds = step.buffer(f"{label}.embeds", (n, h))
        embed(prefix("embed"), tokens, n, embeds)
        x = step.buffer(f"{label}.x", (width, h))
        prologue = trace_split(drafter.prologue, [torch.zeros(8, h, dtype=td), torch.zeros(8, h, dtype=td)], ["embeds", "hidden"])
        scope = step.scopes.setdefault("mtp.prologue", LayoutScope())
        sources = partial(_module_sources, drafter.prologue)
        step.spliced(
            prefix("prologue"),
            prologue,
            [embeds, hidden],
            [x],
            cache,
            sources,
            scope=scope,
            sizes={"num_tokens": n},
            weights="mtp.prologue",
        )
        layer = drafter.layer
        split = build_moe_split_wrapper(layer) if moe_block_parts(layer.mlp) else None
        read = None if drafter.experts_of is None else partial(drafter.experts_of, 0)
        # A MoE head's experts stay on the device: the drafter runs every verify step, more than once.
        out = attention(label, "mtp", prefix, layer, split, x, n, position, f"{label}.out", cache_only, read, weights, False, mrope)
        if cache_only:
            return None
        rows_out = step.buffer(f"{label}.normed", (n, h))
        step.compiled(prefix("norm"), drafter.norm, (torch.zeros(n, h, dtype=td),), [out], [rows_out], cache, weights="mtp.norm", rows=n)
        return rows_out

    if drafter is not None and not verify:
        # Its rotated query and gate take the persistent role, named by the program, which owns them.
        draft(f"mtp.{kind}", "next_tokens", normed, rows, "position", cache_only=True)
    if prefill:
        return step
    coded = _coded_head(model, ckpt)

    def head(prefix, normed, logits, n):
        """The head's FP32 logits of ``n`` normalized rows; every pass binds the head's weights once."""
        if coded is None:

            class Head(torch.nn.Module):
                def __init__(self):
                    super().__init__()
                    self.lm_head = model.lm_head

                def forward(self, hidden):
                    return torch.mm(hidden, self.lm_head.weight.transpose(0, 1), out_dtype=torch.float32)

            step.compiled(prefix, Head(), (torch.zeros(n, h, dtype=td),), [normed], [logits], cache, weights="head", rows=n)
            return
        spec, feed = coded
        graph = head_graph(spec)
        inputs = [normed]
        for name in graph.inputs[1:]:
            shape = tuple(dim if isinstance(dim, int) else int(dim.expr.eval({})) for dim in graph.buffer(name).shape)
            inputs.append(step.buffer(f"head.{name}", shape, get(graph.buffer(name).dtype), "constant", partial(lambda n: feed()[n], name)))
        step.spliced(prefix, graph, inputs, [logits], cache, lambda _plan: {}, scope=LayoutScope(), sizes={"num_tokens": n})

    def greedy(prefix, logits, token, n):
        step.compiled(prefix, greedy_module(vocab), (torch.zeros(n, vocab, dtype=torch.float32),), [logits], [token], cache, rows=n)

    head("head", normed, "logits", rows)
    greedy("greedy", "logits", "token", rows)
    if not verify:
        return step
    # The drafts the target confirms, decided on the device, then each Gated DeltaNet layer's state after the last
    # accepted row, and the next step's drafts: the drafter's first step over every row, as its cache needs the rows
    # the target confirmed, from its row past the last accepted one; each later step over that row's draft.
    positions = [step.buffer(f"position{j}", (1,), I64) for j in range(2, drafts + 1)]
    examples = (torch.zeros(rows, dtype=torch.int64), torch.zeros(rows, dtype=torch.int64), scalar())
    step.compiled("accept", accept_module(drafts), examples, ["token", "tokens", "position"], ["accepted", *positions], cache)
    for carry in carries:
        step.compiled(*carry, cache)
    rows_out = draft("mtp1", "token", normed, rows, "position")
    picked = step.buffer("mtp1.picked", (1, h))
    mropes = [step.buffer(f"mrope{j}", (1, 3), I64) if images else None for j in range(2, drafts + 1)]
    examples = (torch.zeros(rows, h, dtype=td), scalar(), *([torch.zeros(rows, 3, dtype=torch.int64)] if images else []))
    inputs = [rows_out, "accepted", *(["mrope"] if images else [])]
    step.compiled("mtp1.pick", pick_module(drafts), examples, inputs, [picked, *filter(None, mropes)], cache, rows=rows)
    for j, token in enumerate(drafted, start=1):
        if j > 1:
            picked = draft(f"mtp{j}", drafted[j - 2], picked, 1, positions[j - 2], mrope=mropes[j - 2])
        logits = step.buffer(f"mtp{j}.logits", (1, vocab), F32)
        head(f"mtp{j}.head", picked, logits, 1)
        greedy(f"mtp{j}.greedy", logits, token, 1)
    return step


def export_model(
    model,
    destination,
    *,
    context_length=None,
    page_tokens=None,
    eos_ids=(),
    provenance=None,
    prefill_size=None,
    ckpt=None,
    kv_cache_dtype="auto",
    drafter=None,
    vision=None,
):
    """Bundle one-token decode and fixed-width prefill, and with a ``drafter`` (:func:`load_drafter`) the verify step
    that drafts with it; preparation owns every model operation.

    ``context_length`` defaults to the model's window, up to :data:`MAX_CONTEXT`. ``page_tokens`` is how
    many tokens of the KV cache one page holds; the default, one page spanning the whole context, is the
    flat buffer. ``kv_cache_dtype`` is the cache's, a :data:`KV_CACHE_DTYPES` name. ``prefill_size`` is the
    chunk width the prompt is consumed at; 1 disables chunking. ``ckpt`` is the checkpoint pair
    :func:`load_model` hands back beside a quantized checkpoint's twin; the trunk's coded weights are then
    spelled and bound from the shards, and its routed experts are read from them a layer at a time. A MoE
    layer's device holds ``EMMY_GEN_EXPERTS_GPU`` of its routed experts (default: all); below its
    expert count the pack keeps every expert's page for the runtime's pager, once, written a layer at
    a time, so the host never holds every expert, and the worker loads the resident ones straight onto
    the device. ``vision`` is the checkpoint (a directory or ``<repo>[@<revision>]``) whose vision tower the pack
    also holds, for a pack that answers images (``serving/native/vision.py``); the key then carries the tower's
    contract (``vision``)."""
    if any(not 0 <= token < model.config.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    steps = _steps(model, context_length, page_tokens, prefill_size, ckpt, kv_cache_dtype, drafter=drafter, vision=vision)
    programs, context_length, prefill_size, contract = steps
    return save_executable(
        destination,
        {name: step.plan for name, step in programs.items()},
        bindings={name: step.bindings for name, step in programs.items()},
        reads={name: frozenset(step.reads) for name, step in programs.items()},
        pages={name: page for step in programs.values() for name, page in step.pages.items()},
        key={
            "generation": {
                "version": GENERATION_VERSION,
                "context_length": context_length,
                "vocab_size": model.config.vocab_size,
                "eos_ids": list(eos_ids),
                "prefill_size": prefill_size,
                # The buffers each step writes in place and carries to the next; a request zeroes them.
                "state": programs["decode"].state,
                # The tokens the verify step drafts; none without one.
                "drafts": 0 if drafter is None else drafter.drafts,
                **({"vision": contract} if contract else {}),
            }
        },
        provenance=provenance,
    )


def export_checkpoint(
    model_id,
    revision,
    destination,
    *,
    dtype="auto",
    golden=None,
    strict=None,
    context_length=None,
    page_tokens=None,
    prefill_size=None,
    kv_cache_dtype="auto",
    images=False,
    drafts=0,
):
    """Load ``model_id`` in ``dtype`` (:func:`load_model`) and export it to ``destination`` (:func:`export_model`) under
    the GPU lock, deploying from the rows of the golden file ``golden`` in the regime they were recorded in, as the
    plugin's boot does (``pins.golden_regime``), with ``strict`` evidence when set; with ``images``, its vision tower
    too. ``drafts`` above zero drafts that many tokens a step with the checkpoint's multi-token-prediction head
    (:func:`load_drafter`)."""
    from emmy import config
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.pipeline.search.golden import GoldenFile
    from emmy.compiler.pipeline.search.pins import golden_regime, pinned_knobs

    model, ckpt = load_model(model_id, revision, dtype)
    drafter, ckpt = load_drafter(model_id, revision, model, drafts, ckpt) if drafts else (None, ckpt)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    regime = golden_regime(GoldenFile.load(golden)) if golden else {}
    vision = (str(ckpt[0]) if ckpt else f"{model_id}@{revision}" if revision else model_id) if images else None
    with gpu_lock(), config.golden_file_override(golden), config.strict_evidence_override(strict), pinned_knobs(regime):
        export_model(
            model,
            destination,
            context_length=context_length,
            page_tokens=page_tokens,
            eos_ids=eos,
            prefill_size=prefill_size,
            ckpt=ckpt,
            kv_cache_dtype=kv_cache_dtype,
            drafter=drafter,
            vision=vision,
        )


def default_context(config):
    """The context an export takes when none is named: the model's window (its text config's), up to
    :data:`MAX_CONTEXT`."""
    return min(MAX_CONTEXT, config.get_text_config().max_position_embeddings)


def trace_programs(model, *, prefill_size=None, decode_width=None, ckpt=None, kv_cache_dtype="auto", drafter=None, vision=None):
    """The programs :func:`export_model` compiles for ``model`` that are native generation's own, traced and not
    compiled: its serving twins, ``{name: graph}``; the rest are the plugin's. A program is named
    ``<kind><width>-native`` after its kind and rows (``router1-native``, ``rope64-native``), numbered after the first
    of a kind and width; the layers of one shape trace one program, and so does the ``drafter``'s layer where its
    shape is theirs. They are traced at the model's window (:func:`default_context`), where the rotary program's
    cache is the whole context; attention reads it to the window at any context. ``vision`` is :func:`export_model`'s:
    an export that answers images adds its image path."""
    traced = {}
    _steps(model, None, None, prefill_size, ckpt, kv_cache_dtype, traced, decode_width, drafter=drafter, vision=vision)
    twins = {}
    for name, graph, plugin in traced.values():
        if plugin:
            continue
        twins[f"{name}-native{sum(n.startswith(f'{name}-native') for n in twins) or ''}"] = graph
    return twins


def _steps(
    model, context_length, page_tokens, prefill_size, ckpt, kv_cache_dtype="auto", traced=None, decode_width=None, drafter=None, vision=None
):
    """The decode program, at a prefill width above one the prefill program, and with a ``drafter`` the verify
    program, the context, the prefill width and the vision tower's contract; with ``traced``, their programs traced
    into it instead (:class:`_Step`). The context defaults to the model's window, up to :data:`MAX_CONTEXT`. Both
    widths default to the plugin's (``EMMY_GEN_DECODE_BUCKET``, ``EMMY_GEN_M1_TIER``, ``EMMY_GEN_PREFILL_BUCKET``):
    ``decode_width`` is the width its trunk runs one token at, and a verify step's rows, as the plugin's, fit it;
    ``prefill_size`` its prefill chunk, :data:`PREFILL_SIZE` when it has none. With ``vision`` (:func:`export_model`)
    the steps take images and the tower's programs join them (``vision.tower_programs``); the contract is ``None``
    without."""
    import torch

    from emmy import config
    from emmy.compiler.backend.plan_cache import PlanTemplateCache

    if kv_cache_dtype not in KV_CACHE_DTYPES:
        raise ValueError(f"native generation stores its cache in {sorted(KV_CACHE_DTYPES)}, not {kv_cache_dtype!r}")
    if context_length is None:
        context_length = default_context(model.config)
    decoder = validate_model(model, context_length, ckpt)
    kv_dtype = model.dtype if KV_CACHE_DTYPES[kv_cache_dtype] is None else getattr(torch, KV_CACHE_DTYPES[kv_cache_dtype])
    page_tokens = context_length if page_tokens is None else page_tokens
    if not 0 < page_tokens <= context_length or context_length % page_tokens:
        raise ValueError(f"page_tokens={page_tokens} must divide the context capacity {context_length}")
    if prefill_size is None:
        prefill_size = config.gen_prefill_bucket() if config.gen_prefill_bucket() > 0 else PREFILL_SIZE
    if type(prefill_size) is not int or not 1 <= prefill_size <= MAX_CONTEXT:
        raise ValueError("prefill size must be within supported context capacity")
    prefill_size = min(prefill_size, context_length)
    if decode_width is None:
        decode_width = 1 if config.gen_m1_tier() or config.gen_decode_bucket() < 1 else config.gen_decode_bucket()
    # One layer at a time: a program reads a layer's experts as it compiles the layer, and the pack reads
    # them again as it writes their pages, so below the budget the host holds one layer's experts at most.
    experts_of = None if ckpt is None else lru_cache(maxsize=1)(partial(checkpoint_experts, ckpt[0], model.dtype))
    cache, shared, scopes = PlanTemplateCache(), {}, {}

    if drafter is not None and drafter.drafts + 1 > decode_width:
        raise ValueError(f"a verify step of {drafter.drafts + 1} rows does not fit the decode width {decode_width}")
    towers, contract = {}, None
    if vision is not None:
        from emmy.serving.native.vision import tower_programs

        towers, contract = tower_programs(vision, model.dtype, cache, shared, scopes, traced)
    images = max(contract["widths"]) if contract else None

    def program(kind, rows, width):
        return _program(
            kind,
            model,
            decoder,
            context_length,
            rows,
            width,
            page_tokens,
            kv_dtype,
            cache,
            ckpt,
            experts_of,
            shared,
            scopes,
            traced,
            drafter,
            images,
        )

    # Decode first: the plugin builds a wrapper's twins narrowest first, and the first to decide a weight's layout
    # settles it for the rest; it builds the programs that keep a verify step's states last.
    programs = {"decode": program("decode", 1, decode_width)}
    if prefill_size > 1:
        programs["prefill"] = program("prefill", prefill_size, prefill_size)
    if drafter is not None:
        programs["verify"] = program("verify", drafter.drafts + 1, decode_width)
    return {**programs, **towers}, context_length, prefill_size, contract
