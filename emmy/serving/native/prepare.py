"""Export Qwen3, Qwen3.5 and Qwen3.5-MoE text models as static decode and chunked prefill programs."""

from __future__ import annotations

import math
from dataclasses import replace
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
GENERATION_VERSION = 11
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


def rope_module(heads, kv_heads, head_dim, rotary_dim, rows, cache_dtype):
    """Rotate q and k at their positions in FP32, round once to the model dtype, and store k and v into the
    cache at those positions in ``cache_dtype``: the model dtype, or FP8 e4m3 saturated to its largest finite
    value, as vLLM stores its FP8 cache with no scale. The caches come in and go out, written in place (the
    outputs are stored over the inputs), so the step replays as one graph at every position. The caches and
    the rotary tables are token-major (``[context, kv_heads, head_dim]``, ``[context, rotary_dim]``) and whole,
    since a store may land anywhere in the context; attention reads only the window. A partial rotary embedding
    (Qwen3.5) rotates only the leading ``rotary_dim`` of each head and passes the rest through."""
    import torch

    def store(x):
        if cache_dtype == x.dtype:
            return x
        return x.float().clamp(-FP8_MAX, FP8_MAX).to(cache_dtype)

    class Rope(torch.nn.Module):
        def rotate(self, x, n, c, s):
            dtype, x = x.dtype, x.view(rows, n, head_dim).float()
            rotated, passed = (x[..., :rotary_dim], x[..., rotary_dim:]) if rotary_dim < head_dim else (x, None)
            half = rotary_dim // 2
            paired = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
            rotated = rotated * c + paired * s
            return (rotated if passed is None else torch.cat((rotated, passed), dim=-1)).to(dtype)

        def forward(self, q, k, v, position, cosine, sine, keys, values):
            positions = position + torch.arange(rows)
            c, s = cosine[positions].unsqueeze(1), sine[positions].unsqueeze(1)
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


def carry_module():
    """A Gated DeltaNet layer's next state and history written over the ones the step carries. The
    layer's program is the plugin's, which writes them apart from what it reads, since some of its
    kernels read the old state after another wrote the new one; each is spelled as the carried
    buffer with every row replaced (a scatter), which the compiler stores into its memory."""
    import torch

    class Carry(torch.nn.Module):
        def forward(self, state_next, history_next, state, history):
            rows = torch.arange(state.shape[0])
            return state.index_copy(0, rows, state_next), history.index_copy(0, rows, history_next)

    return Carry()


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
    reads."""
    tail = len(seam) - 1
    return tuple(program[len(program) - tail :]) == tuple(seam[1:]) and math.prod(program[: len(program) - tail]) <= seam[0]


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
    def __init__(self, dtype, rows, context_length, ckpt=None, traced=None, shared=None, scopes=None):
        # A decode step (one row) ends at the logits and the greedy token; the runtime hands the token it
        # selects back as the next step's input. A prefill chunk writes the cache and nothing else.
        # ``ckpt`` (``load_model``) puts every program on the checkpoint-sourced lane: the trunk's
        # coded weights are spelled from the shards and bound from them, never off the module.
        # ``dtype`` is the model's, which every 16-bit buffer takes. With ``traced``, a dict, each
        # distinct program is traced into it, named by its kind and width, and nothing is compiled.
        # ``shared`` and ``scopes`` are the export's, across its programs: each constant's bytes and the
        # layouts each of the plugin's wrappers binds its weights in. The window's hint is the context, so
        # the runtime allocates every buffer it sizes for the whole context.
        self.dtype = dtype
        self.rows = rows
        self.ckpt = ckpt
        self.traced = traced
        self.plan = ExecutionPlan("cuda", ["position", "tokens"], [] if rows > 1 else ["logits", "token"], [], {}, {}, [], {})
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

    def bind(self, name, key, read):
        """Bind constant ``name`` to the bytes ``read()`` returns; a constant of that name and ``key`` bound by another
        of the export's programs shares that program's function."""
        self.bindings[name] = self.shared.setdefault((name, key), read)

    def buffer(self, name, shape, dtype=None, role="scratch", data=None):
        """Declare a step buffer; ``data``, a function returning its host values, binds it. A traced step binds
        nothing: it compiles nothing either."""
        dtype = dtype or self.dtype
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None and self.traced is None:
            self.bind(name, (dtype.name, tuple(shape)), lambda: numpy_storage(data(), dtype).tobytes())
        return name

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache, output_names=(), paged=(), coded=None, twin=False, window=()):
        """Trace ``wrapper`` at ``examples`` and splice it in (:meth:`spliced`). ``coded`` spells named weight inputs
        as FP8 codes with a scale input appended after the traced ones (``spell_quantized_inputs``), as the plugin
        spells its expert programs. ``twin`` says the program is one of the plugin's serving twins, traced as the
        plugin traces it: its weights take the layouts of the plugin's wrapper ``prefix``, and it is no native twin.
        ``window`` names the arguments whose leading axis is the window (:data:`WINDOW`): the program reads their
        leading rows, as many as the runtime sets."""
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
        scope = self.scopes.setdefault(prefix, LayoutScope()) if twin else None
        self.spliced(prefix, graph, inputs, outputs, cache, sources, output_names=output_names, paged=paged, scope=scope)

    def spliced(self, prefix, graph, inputs, outputs, cache, sources, *, output_names=(), paged=(), scope=None, sizes=None):
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
        ``sizes`` gives a symbolic program its sizes. A program input or output may hold the leading rows of a
        wider step buffer (:func:`_leading_rows`). Everything is spelled in the graph's own names, never the
        step's, so every layer traces the same graph and compiles once."""
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
            # Every layer of one shape traces one program, which the first names by its width (``-sym`` for a program of
            # any width), and says whether it is the plugin's.
            name = f"{prefix.rstrip('0123456789.')}{'-sym' if sizes else self.rows}"
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
            if buffer.role == "constant":
                if buffer.name not in plan.weights and buffer.name not in plan.constants:
                    raise ValueError(f"unresolved constant: {name}")
                shape = buffer.resolve_shape(sizes or {})
                weight = plan.weights.get(buffer.name)
                spec = plan.constants.get(buffer.name) if weight is None else (weight.source_path, weight.source_parts, weight.generated)
                key = repr((buffer.dtype.name, shape, spec, None if weight is None else weight.load_ops))
                self.bind(name, key, partial(_weight_bytes, plan, buffer.name, shape, sources))


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


def _program(model, decoder, context_length, rows, width, page_tokens, kv_dtype, cache, ckpt, experts_of, shared, scopes, traced=None):
    """One program: the one-token decode step at ``rows == 1``, else a prefill chunk of ``rows``
    prompt tokens that writes the cache and the carried state and computes nothing past the last
    attention layer's keys. The trunk is the vLLM plugin's: its serving twins, in the model dtype, the
    attention halves at ``width`` rows (the width the plugin runs one token at: its decode bucket without
    its M=1 tier; the chunk itself for prefill) and a Gated DeltaNet layer at the step's own rows, as the
    plugin decomposes a step, so the recipe golden's rows are the trunk's. Native's own programs run at
    ``rows`` and read and write the leading rows of the wider seams. The cache holds ``kv_dtype`` values
    (a torch dtype). ``experts_of(index)`` reads a checkpoint-sourced MoE layer's routed experts as
    :func:`checkpoint_experts` does; without it they come off the module. ``shared`` and ``scopes`` are
    :class:`_Step`'s; ``traced`` traces the programs into it instead of compiling them."""
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
    from emmy.serving.gen_runner import EMBED_PAGE_ROWS, embed_gather_graph

    cfg = decoder.config
    prefill = rows > 1
    # The model dtype: every 16-bit activation, cache and weight the step holds, as the plugin serves it.
    td = model.dtype
    step = _Step(get(str(td).removeprefix("torch.")), rows, context_length, ckpt, traced, shared, scopes)
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    # The step's position and the tokens it embeds, which the host uploads: the prompt's while it lasts, then
    # the previous step's selection.
    step.buffer("position", (1,), I64, "input")
    step.buffer("tokens", (rows,), I64, "input")
    if not prefill:
        step.buffer("logits", (1, vocab), F32, "output")
        step.buffer("token", (1,), I64, "output")

    # Example inputs are distinct tensors: the tracer folds two arguments that share one into a
    # single aliased input. A windowed one's leading axis is the window, traced at its hint.
    def scalar():
        return torch.zeros(1, dtype=torch.int64)

    def head_rows(n, heads):
        return torch.zeros(n, heads * d, dtype=td)

    def windowed():
        return torch.zeros(DEFAULT_SEQ_HINT, kv, d, dtype=kv_dtype)

    def mixer_outputs(index, routed, n, role="scratch"):
        # A MoE layer's mixer leaves its residual and the normalized rows its routed experts read; the
        # experts write the layer's output.
        if routed:
            return [step.buffer(f"layer{index}.h", (n, h), role=role), step.buffer(f"layer{index}.xn", (n, h), role=role)]
        return [step.buffer(f"hidden{index + 1}", (width, h), role=role)]

    def experts(index, routed, expert, mixed, normed):
        """The routed half of a MoE layer after its mixer, which leaves the residual ``mixed``
        with the shared expert folded in and the normalized rows ``normed``: the router, the
        fixed-slot program of the ``expert`` form over the layer's E-stacked expert inputs, every
        row's picks at once, and the combine, which writes the layer's output. ``routed`` is the layer's
        ``(router, experts)`` pair. FP8 experts stay coded: each weight is its codes beside its block
        scales, which the program decodes in-graph as the plugin's does. The inputs are constants of
        the step when the device holds every expert; below that budget they are paged one expert per
        page (``config.gen_expert_slots``): the budget's first experts resident on the device, the rest in
        host memory, and a token's picks of device slots holding the most recently routed of those. A decode step
        runs the plugin's own one-row fixed-slot program."""
        gate, stacked = routed
        k = gate.top_k
        if experts_of is None:
            fmt, tensors = None, {"w_gate_up": stacked.gate_up_proj.detach(), "w_down": stacked.down_proj.detach()}
        else:
            fmt, tensors = experts_of(index)
        if fmt not in (None, "f8e4m3", "f8e5m2"):
            raise ValueError(f"native generation runs 16-bit or FP8 routed experts, not {fmt}")
        coded = {n: (fmt, tuple(tensors[f"{n}_scale"].shape), "f32") for n in ("w_gate_up", "w_down")} if fmt else {}
        paging = config.gen_expert_slots(gate.num_experts, k)
        paged = paging is not None
        names = ["w_gate_up", "w_down", *(f"{n}_scale" for n in coded)]
        weights = [f"layer{index}.{n}" for n in names]
        for name, n in zip(weights, names, strict=True):
            t = tensors[n]
            if n in coded:
                shape, dtype = tuple(t.shape), get(fmt)
            elif n.endswith("_scale"):
                # Declared as the decode reads it: the block scale broadcast over the codes' blocks.
                shape, dtype = scale_layout(tuple(tensors[n.removesuffix("_scale")].shape), tuple(t.shape))[0], F32
            else:
                shape, dtype = tuple(t.shape), step.dtype
            # Read when the pack is written, a layer at a time, as the pages are.
            read = partial(_host, t) if experts_of is None else partial(lambda i, n: _host(experts_of(i)[1][n]), index, n)
            step.buffer(name, shape, dtype, role="input" if paged else "constant", data=None if paged else read)
            if paged:
                step.pages[name] = (partial(lambda read, d: numpy_storage(read(), d), read, dtype), paging[1])
        # One row's picks are a ``[k]`` selector, a chunk's ``[rows, k]``.
        picks = (k,) if rows == 1 else (rows, k)
        scores = step.buffer(f"layer{index}.scores", (rows, k))
        sel = step.buffer(f"layer{index}.sel", picks, I64)
        step.compiled(f"router{index}", router_module(gate, rows), (torch.zeros(rows, h, dtype=td),), [normed], [scores, sel], cache)
        partials = step.buffer(f"layer{index}.partials", (*picks, h))
        # The trace sees the weights' values, whatever stores them: the model dtype at the codes' shape.
        values = (torch.zeros(tensors[n].shape, dtype=td) for n in ("w_gate_up", "w_down"))
        examples = (torch.zeros(rows, h, dtype=td), torch.zeros(picks, dtype=torch.int64), *values)
        declared = tuple((n, 0, 1, paging[0]) for n in names) if paged else ()
        wrapper = build_moe_slots_wrapper(expert)
        step.compiled(
            f"experts{index}", wrapper, examples, [normed, sel, *weights], [partials], cache, paged=declared, coded=coded, twin=not prefill
        )
        output = step.buffer(f"hidden{index + 1}", (width, h))
        examples = (torch.zeros(rows, h, dtype=td), torch.zeros(rows, k, dtype=td), torch.zeros(*picks, h, dtype=td))
        step.compiled(f"combine{index}", combine_module(), examples, [mixed, scores, partials], [output], cache)
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
    hidden = step.buffer("hidden0", (width, h))
    gather = embed_gather_graph(vocab, h, td)
    paged = () if slots is None else (("weight", 0, EMBED_PAGE_ROWS, slots[0]),)
    scope = None if slots is None else LayoutScope()
    sizes = {"num_tokens": rows}
    step.spliced("embed", gather, ["tokens", weight], [hidden], cache, lambda _plan: {}, paged=paged, scope=scope, sizes=sizes)
    # The rotary tables, one copy for every layer, from the checkpoint's own module in FP32; and the cache, a key and
    # a value buffer per attention layer, token-major and allocated for the whole context. It is paged along its
    # positions only when a page is shorter than the context: a page spanning it is the flat buffer.
    with torch.no_grad():
        tables = [t[0] for t in decoder.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float32), torch.arange(context_length).reshape(1, -1))]
    rotary = [step.buffer(f"rope.{n}", tuple(t.shape), F32, "constant", t.numpy) for n, t in zip(("cosine", "sine"), tables, strict=True)]
    pages = page_tokens < context_length
    cache_type = get(str(kv_dtype).removeprefix("torch."))
    for index, layer in enumerate(decoder.layers):
        # The last prefill layer's outputs that nothing reads take the persistent role rather than
        # a scratch slot nobody consumes: a Gated DeltaNet layer's hidden output, since the layer
        # runs whole for its state, and an attention layer's rotated query and gate, since only its
        # keys and values are needed in the cache.
        last_prefill = prefill and index + 1 == len(decoder.layers)
        unread = "output" if last_prefill else "scratch"
        # A MoE layer's split, built once: its attention ``pre`` and ``post`` (none for a Gated
        # DeltaNet layer) and its experts' form.
        routed = moe_block_parts(layer.mlp)
        split = build_moe_split_wrapper(layer) if routed else None
        if selected_layer_type(decoder, layer, index) == "linear_attention":
            # A Gated DeltaNet layer always runs whole, the last prefill layer included: its state must advance. The
            # plugin's program writes the next state and history apart from the ones it reads; the carry writes them
            # over the ones the step carries.
            mixer = layer.linear_attn
            shapes = {
                "state": ((1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim), F32, torch.float32),
                "history": ((1, mixer.conv_dim, mixer.conv_kernel_size), step.dtype, td),
            }
            state = [step.buffer(f"layer{index}.{name}", shape, dtype, role="output") for name, (shape, dtype, _) in shapes.items()]
            nexts = [step.buffer(f"layer{index}.{name}_next", shape, dtype) for name, (shape, dtype, _) in shapes.items()]
            step.state += state
            outputs = mixer_outputs(index, routed, rows, unread)
            examples = [torch.zeros(1, rows, h, dtype=td), *(torch.zeros(shape, dtype=dtype) for shape, _, dtype in shapes.values())]
            step.compiled(f"gdn{index}", build_gdn_state_wrapper(layer), examples, [hidden, *state], [*outputs, *nexts], cache, twin=True)
            examples = [torch.zeros(shape, dtype=dtype) for _ in range(2) for shape, _, dtype in shapes.values()]
            step.compiled(f"carry{index}", carry_module(), examples, [*nexts, *state], state, cache)
            # The last prefill layer's output is read by nothing, so its experts never run.
            hidden = experts(index, routed, split[2], *outputs) if routed and not last_prefill else outputs[0]
            continue
        pre, post = split[:2] if split else build_attention_split_wrapper(layer)
        # Qwen3.5's query projection also carries the attention output gate; ``pre`` emits it as a
        # fourth tensor and ``post`` takes it as a third input.
        gated = (("gate", heads),) if pre.emits_gate else ()
        names = [step.buffer(f"layer{index}.{name}", (width, n * d)) for name, n in (("q", heads), ("k", kv), ("v", kv))]
        names += [step.buffer(f"layer{index}.{name}", (width, n * d), role=unread) for name, n in gated]
        step.compiled(f"pre{index}", pre, (torch.zeros(width, h, dtype=td),), [hidden], names, cache, twin=True)
        rotated = step.buffer(f"layer{index}.rotated", (rows, heads * d), role=unread)
        # The cache, written a chunk at a time at ``position`` — a scatter into the cache the rope program takes in
        # and hands back — and read to the window.
        keys, values = (step.buffer(f"layer{index}.{name}", (context_length, kv, d), cache_type, "output") for name in ("keys", "values"))
        paging = ((name, 0, page_tokens) for name in ("keys", "values", "keys_next", "values_next")) if pages else ()
        examples = (head_rows(rows, heads), head_rows(rows, kv), head_rows(rows, kv), scalar(), *map(torch.zeros_like, tables))
        examples += (torch.zeros(context_length, kv, d, dtype=kv_dtype), torch.zeros(context_length, kv, d, dtype=kv_dtype))
        step.compiled(
            f"rope{index}",
            rope_module(heads, kv, d, tables[0].shape[-1], rows, kv_dtype),
            examples,
            [*names[:3], "position", *rotary, keys, values],
            [rotated, keys, values],
            cache,
            output_names=("rotated", "keys_next", "values_next"),
            paged=tuple(paging),
        )
        if last_prefill:
            continue
        attention = step.buffer(f"layer{index}.attention", (width, heads * d))
        step.compiled(
            f"attend{index}",
            attend_module(heads, kv, d, rows),
            (head_rows(rows, heads), windowed(), windowed(), scalar()),
            [rotated, keys, values, "position"],
            [attention],
            cache,
            paged=(("keys", 0, page_tokens), ("values", 0, page_tokens)) if pages else (),
            window=("keys", "values"),
        )
        outputs = mixer_outputs(index, routed, width)
        examples = (head_rows(width, heads), torch.zeros(width, h, dtype=td), *(head_rows(width, heads) for _ in gated))
        step.compiled(f"post{index}", post, examples, [attention, hidden, *names[3:]], outputs, cache, twin=True)
        hidden = experts(index, routed, split[2], *outputs) if routed else outputs[0]
    if prefill:
        return step

    # The final norm, then the head: the plugin's program over the codes of a head the checkpoint keeps coded, else
    # FP32 scores from 16-bit operands, since rounding the logits to 16 bits could tie distinct scores.
    normed = step.buffer("normed", (1, h))
    step.compiled("norm", decoder.norm, (torch.zeros(1, h, dtype=td),), [hidden], [normed], cache)
    coded = _coded_head(model, ckpt)
    if coded is None:

        class Head(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.lm_head = model.lm_head

            def forward(self, hidden):
                return torch.mm(hidden, self.lm_head.weight.transpose(0, 1), out_dtype=torch.float32)

        step.compiled("head", Head(), (torch.zeros(1, h, dtype=td),), [normed], ["logits"], cache)
    else:
        spec, feed = coded
        graph = head_graph(spec)
        inputs = [normed]
        for name in graph.inputs[1:]:
            shape = tuple(dim if isinstance(dim, int) else int(dim.expr.eval({})) for dim in graph.buffer(name).shape)
            inputs.append(step.buffer(f"head.{name}", shape, get(graph.buffer(name).dtype), "constant", partial(lambda n: feed()[n], name)))
        step.spliced("head", graph, inputs, ["logits"], cache, lambda _plan: {}, scope=LayoutScope(), sizes={"num_tokens": 1})
    step.compiled("greedy", greedy_module(vocab), (torch.zeros(1, vocab, dtype=torch.float32),), ["logits"], ["token"], cache)
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
):
    """Bundle one-token decode and fixed-width prefill; preparation owns every model operation.

    ``context_length`` defaults to the model's window, up to :data:`MAX_CONTEXT`. ``page_tokens`` is how
    many tokens of the KV cache one page holds; the default, one page spanning the whole context, is the
    flat buffer. ``kv_cache_dtype`` is the cache's, a :data:`KV_CACHE_DTYPES` name. ``prefill_size`` is the
    chunk width the prompt is consumed at; 1 disables chunking. ``ckpt`` is the checkpoint pair
    :func:`load_model` hands back beside a quantized checkpoint's twin; the trunk's coded weights are then
    spelled and bound from the shards, and its routed experts are read from them a layer at a time. A MoE
    layer's device holds ``EMMY_GEN_EXPERTS_GPU`` of its routed experts (default: all); below its
    expert count the pack keeps every expert's page for the runtime's pager, once, written a layer at
    a time, so the host never holds every expert, and the worker loads the resident ones straight onto
    the device."""
    if any(not 0 <= token < model.config.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    programs, context_length, prefill_size = _steps(model, context_length, page_tokens, prefill_size, ckpt, kv_cache_dtype)
    return save_executable(
        destination,
        {name: step.plan for name, step in programs.items()},
        bindings={name: step.bindings for name, step in programs.items()},
        reads={name: frozenset(step.reads) for name, step in programs.items()},
        pages=programs["decode"].pages,
        key={
            "generation": {
                "version": GENERATION_VERSION,
                "context_length": context_length,
                "vocab_size": model.config.vocab_size,
                "eos_ids": list(eos_ids),
                "prefill_size": prefill_size,
                # The buffers each step writes in place and carries to the next; a request zeroes them.
                "state": programs["decode"].state,
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
):
    """Load ``model_id`` in ``dtype`` (:func:`load_model`) and export it to ``destination`` (:func:`export_model`) under
    the GPU lock, deploying from the rows of the golden file ``golden`` in the regime they were recorded in, as the
    plugin's boot does (``pins.golden_regime``), with ``strict`` evidence when set."""
    from emmy import config
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.pipeline.search.golden import GoldenFile
    from emmy.compiler.pipeline.search.pins import golden_regime, pinned_knobs

    model, ckpt = load_model(model_id, revision, dtype)
    eos = model.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    regime = golden_regime(GoldenFile.load(golden)) if golden else {}
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
        )


def default_context(config):
    """The context an export takes when none is named: the model's window (its text config's), up to
    :data:`MAX_CONTEXT`."""
    return min(MAX_CONTEXT, config.get_text_config().max_position_embeddings)


def trace_programs(model, *, prefill_size=None, decode_width=None, ckpt=None, kv_cache_dtype="auto"):
    """The programs :func:`export_model` compiles for ``model`` that are native generation's own, traced and not
    compiled: its serving twins, ``{name: graph}``; the rest are the plugin's. A program is named
    ``<kind><width>-native`` after its step (``router1-native``, ``rope64-native``), numbered after the first of a kind
    and width; the layers of one shape trace one program. They are traced at the model's window
    (:func:`default_context`), where the rotary program's cache is the whole context; attention reads it to the
    window at any context."""
    traced = {}
    _steps(model, None, None, prefill_size, ckpt, kv_cache_dtype, traced, decode_width)
    twins = {}
    for name, graph, plugin in traced.values():
        if plugin:
            continue
        twins[f"{name}-native{sum(n.startswith(f'{name}-native') for n in twins) or ''}"] = graph
    return twins


def _steps(model, context_length, page_tokens, prefill_size, ckpt, kv_cache_dtype="auto", traced=None, decode_width=None):
    """The decode program and, at a prefill width above one, the prefill program, the context and that width; with
    ``traced``, their programs traced into it instead (:class:`_Step`). The context defaults to the model's window, up
    to :data:`MAX_CONTEXT`. Both widths default to the plugin's (``EMMY_GEN_DECODE_BUCKET``, ``EMMY_GEN_M1_TIER``,
    ``EMMY_GEN_PREFILL_BUCKET``): ``decode_width`` is the width its trunk runs one token at, ``prefill_size`` its
    prefill chunk, :data:`PREFILL_SIZE` when it has none."""
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

    def program(rows, width):
        return _program(model, decoder, context_length, rows, width, page_tokens, kv_dtype, cache, ckpt, experts_of, shared, scopes, traced)

    # Decode first: the plugin builds a wrapper's twins narrowest first, and the first to decide a weight's layout
    # settles it for the rest.
    programs = {"decode": program(1, decode_width)}
    if prefill_size > 1:
        programs["prefill"] = program(prefill_size, prefill_size)
    return programs, context_length, prefill_size
