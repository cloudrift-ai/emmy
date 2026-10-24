"""Export Qwen3, Qwen3.5 and Qwen3.5-MoE text models as static decode and chunked prefill programs."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache, partial

import numpy as np

from emmy.compiler.backend.binding import numpy_storage
from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, plan_from_graph, splice
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32, I64, get
from emmy.serving.release import DTYPES

MAX_CONTEXT = 4096
GENERATION_VERSION = 8
PREFILL_SIZE = 16
MASK_FILL = -1e9
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


def embed_module(weight, rows):
    """The tokens a step embeds are the prompt's at its ``rows`` positions while the prompt lasts,
    else the previous step's selection: a gather the device decides, so the host binds one
    position scalar per step and never sees a prompt token. A prefill row past the prompt's last
    token computes alongside the others and writes cache rows the decode program overwrites when
    it reaches that position."""
    import torch

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = weight

        def forward(self, prompt, prompt_length, position, next_token):
            positions = position + torch.arange(rows)
            return self.weight[torch.where(positions < prompt_length, prompt[positions], next_token)].float()

    return Embed()


def rope_module(cosine, sine, heads, kv_heads, head_dim, rows):
    """Rotate q and k at their positions in FP32, round once to the model dtype, and store k and v into the
    cache at those positions: the caches come in and go out, written in place (the outputs are stored
    over the inputs), so the step replays as one graph at every position. A partial rotary embedding
    (Qwen3.5) rotates only the leading ``cosine.shape[-1]`` of each head and passes the rest through."""
    import torch

    rotary_dim = cosine.shape[-1]

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cosine", cosine)
            self.register_buffer("sine", sine)

        def rotate(self, x, n, c, s):
            dtype, x = x.dtype, x.view(rows, n, head_dim).float()
            rotated, passed = (x[..., :rotary_dim], x[..., rotary_dim:]) if rotary_dim < head_dim else (x, None)
            half = rotary_dim // 2
            paired = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
            rotated = rotated * c + paired * s
            return (rotated if passed is None else torch.cat((rotated, passed), dim=-1)).to(dtype)

        def forward(self, q, k, v, position, keys, values):
            positions = position + torch.arange(rows)
            c, s = self.cosine[positions].unsqueeze(1), self.sine[positions].unsqueeze(1)
            rotated = self.rotate(q, heads, c, s).view(rows, heads * head_dim)
            new_keys = self.rotate(k, kv_heads, c, s).transpose(0, 1).unsqueeze(0)
            new_values = v.view(rows, kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
            return rotated, keys.index_copy(2, positions, new_keys), values.index_copy(2, positions, new_values)

    return Rope()


def attend_module(heads, kv_heads, head_dim, context_length, rows):
    """Causal grouped-query attention over the whole cache in FP32, rounding only its output:
    every position past a row's own is masked on the device, so the launch is the same at every
    position."""
    import torch
    import torch.nn.functional as F

    class Attend(torch.nn.Module):
        def forward(self, q, keys, values, position):
            group = heads // kv_heads
            q4 = q.view(rows, heads, head_dim).transpose(0, 1).unsqueeze(0).float()
            k4 = keys.repeat_interleave(group, dim=1).float()
            v4 = values.repeat_interleave(group, dim=1).float()
            positions = position + torch.arange(rows)
            keep = torch.arange(context_length).unsqueeze(0) <= positions.unsqueeze(1)
            mask = torch.where(keep, 0.0, MASK_FILL).view(1, 1, rows, context_length)
            out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask)
            return out.to(q.dtype).squeeze(0).transpose(0, 1).reshape(rows, heads * head_dim)

    return Attend()


def gdn_module(layer):
    """One Gated DeltaNet layer over the step's ``[rows, hidden]`` FP32 seam: the state wrapper in
    its FP32-residual form, over one batch row. The next state and history are written over the
    ones it reads: each is spelled as its input with every row replaced (a scatter), which the
    compiler stores into the input's memory. A layer with routed experts returns the mixer's
    residual and the normalized rows the experts read in place of its output."""
    import torch

    from emmy.compiler.trace.huggingface import build_gdn_state_wrapper

    class Carried(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapper = build_gdn_state_wrapper(layer, float32_residual=True)

        def forward(self, hidden, state, history):
            *outputs, state_next, history_next = self.wrapper(hidden.unsqueeze(0), state, history)
            rows = torch.arange(state.shape[0])
            return *(y.squeeze(0) for y in outputs), state.index_copy(0, rows, state_next), history.index_copy(0, rows, history_next)

    return Carried()


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
    """The routed experts' weighted sum added to the mixer's FP32 residual: each row's ``[1, k]``
    scores times its ``[k, hidden]`` partials in the model dtype, as the fixed-slot combine
    computes it, then the residual add in FP32. One row's partials come as ``[k, hidden]``, a
    chunk's as ``[rows, k, hidden]``."""
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
    with a buffer ``dtype``, their storage in it."""
    import torch

    t = t.detach().cpu()
    values = (t.float() if t.dtype == torch.bfloat16 else t).numpy()
    return values if dtype is None else numpy_storage(values, dtype)


class _Step:
    def __init__(self, dtype, rows, ckpt=None, traced=None):
        # A decode step (one row) ends at the logits and the greedy token; the runtime hands the token it
        # selects back as the next step's input. A prefill chunk writes the cache and nothing else.
        # ``ckpt`` (``load_model``) puts every program on the checkpoint-sourced lane: the trunk's
        # coded weights are spelled from the shards and bound from them, never off the module.
        # ``dtype`` is the model's, which every 16-bit buffer takes. With ``traced``, a dict, each
        # distinct program is traced into it, named by its kind and width, and nothing is compiled.
        self.dtype = dtype
        self.rows = rows
        self.ckpt = ckpt
        self.traced = traced
        self.plan = ExecutionPlan(
            "cuda",
            ["prompt", "prompt_length", "position", "next_token"],
            [] if rows > 1 else ["logits", "token"],
            [],
            {},
            {},
            [],
            {},
        )
        self.bindings = {}
        # The host bytes of the buffers the pager holds, page after page; the pack keeps them once for every program.
        self.pages = {}
        # The buffers a step writes in place and carries to the next position; a request starts them at zero.
        self.state = []
        # The kernels whose evidence the step's compiles read (``evidence_reads``), which the pack keeps.
        self.reads = set()

    def buffer(self, name, shape, dtype=None, role="scratch", data=None):
        """Declare a step buffer; ``data``, a function returning its host values, binds it. A traced step binds
        nothing: it compiles nothing either."""
        dtype = dtype or self.dtype
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None and self.traced is None:
            self.bindings[name] = numpy_storage(data(), dtype).tobytes()
        return name

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache, output_names=(), paged=(), coded=None):
        """Compile ``wrapper`` and splice its plan in: its inputs and outputs take the step's names,
        everything else is scoped by ``prefix``. ``output_names`` renames the traced outputs (the
        tracer names them after their last op) so ``paged`` can address them: it names the
        wrapper's buffers that are the step's paged buffers, as ``(graph name, axis, page)`` for a
        cache the device holds whole, or ``(graph name, axis, page, slots)`` for one the pager holds
        in that many device slots, and the step buffer takes that paging. A cache a program writes in
        place comes in as one graph buffer and goes out as another stored over it; both take the step
        buffer's name, so its pages are the ones written. A buffer the pager holds brings its tables
        and its view, which take its step name too: every launch that reads it, in every program,
        moves its pages through the same tables. ``coded`` spells named weight inputs as FP8 codes with
        a scale input appended after the traced ones (``spell_quantized_inputs``), as the plugin spells
        its expert programs. Everything is spelled in the graph's own names, never the step's, so every
        layer traces the same graph and compiles once."""
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.compiler.backend.paging import TABLES
        from emmy.compiler.backend.plan_cache import binding_neutral_key
        from emmy.compiler.loader.quant import spell_quantized_inputs
        from emmy.compiler.pipeline.search.policy.greedy import evidence_reads
        from emmy.serving.gen_runner import _bind_plan_constants, _plan_sources, _traced_split

        graph = _traced_split(wrapper, examples, None, self.ckpt)
        if coded:
            spell_quantized_inputs(graph, coded)
        for old, new in zip(list(graph.outputs), output_names, strict=False):
            graph.rename_node(old, new)
        if paged:
            graph.hints.set("cuda.paged_buffers", tuple(paged))
        if self.traced is not None:
            # Every layer of one shape traces one program, which the first names.
            self.traced.setdefault(binding_neutral_key(graph), (f"{prefix.rstrip('0123456789.')}{self.rows}", graph))
            return
        with evidence_reads() as read:
            plan = cache.resolve(graph, lambda g: plan_from_graph(CudaBackend(tune_db="auto").compile(g)))
        self.reads |= read
        # A serial launch's axes and the paging program's sizes, which the plan hints, are the runtime
        # arguments the standalone runtime resolves itself. A TMA descriptor is encoded once from its
        # source's fixed address, so it may not read pages.
        why = ["symbolic bindings"] if plan.symbolic_bindings else []
        why += ["runtime constants"] if plan.runtime_constants else []
        for launch in plan.launches:
            if {t.src_buf for t in launch.tma_descriptors} & plan.paged.keys():
                why.append(f"{launch.node_id} reads a paged buffer through a TMA descriptor")
            if extra := set(launch.runtime_args) - {name for name, _ in launch.serial} - plan.symbolic_hints.keys():
                why.append(f"{launch.node_id} takes runtime arguments {sorted(extra)}")
        if why:
            raise ValueError(f"native generation requires static ordinary-pointer compiled programs; {prefix}: " + "; ".join(why))
        if self.ckpt is not None:
            sources = _plan_sources(plan, wrapper, self.dtype, *self.ckpt)
        else:
            sources = {
                name: _host(t)
                for name, t in list(wrapper.named_parameters(remove_duplicate=False)) + list(wrapper.named_buffers(remove_duplicate=False))
            }
        constants = _bind_plan_constants(plan, sources, None)
        # The tables and view of a buffer the pager holds enter as the inputs the pager lends.
        held = {n: [f"{n}__{part}" for part in (*TABLES, "view")] for n in plan.paged if f"{n}__table" in plan.inputs}
        pager = {part for parts in held.values() for part in parts}
        renames = dict(zip([n for n in plan.inputs if n not in pager], inputs, strict=True))
        renames.update(zip(plan.outputs, outputs, strict=True))
        renames.update({part: renames[n] + part[len(n) :] for n, parts in held.items() for part in parts})
        existing = {b.name: b for b in self.plan.buffers}
        added = []
        for buffer in plan.buffers:
            name = renames.get(buffer.name, f"{prefix}.{buffer.name}")
            seam = existing.get(name)
            if seam is None:
                added.append((buffer, name))
            elif buffer.name not in plan.paged and name not in self.plan.paged and (seam.shape, seam.dtype) != (buffer.shape, buffer.dtype):
                raise ValueError(f"incompatible program seam: {name}")
        self.plan.buffers += [replace(buffer, name=name) for buffer, name in added if buffer.name in pager]
        self.plan.launches += splice(self.plan, plan, prefix=prefix, renames=renames)
        for buffer, name in added:
            if buffer.role == "constant":
                value = constants.get(buffer.name, plan.constants.get(buffer.name))
                if value is None:
                    raise ValueError(f"unresolved constant: {name}")
                self.bindings[name] = np.broadcast_to(numpy_storage(value, buffer.dtype), buffer.resolve_shape({})).copy().tobytes()


def _program(model, decoder, context_length, rows, page_tokens, cache, ckpt, experts_of, traced=None):
    """One program: the one-token decode step at ``rows == 1``, else a prefill chunk of ``rows``
    prompt tokens that writes the cache and the carried state and computes nothing past the last
    attention layer's keys. ``experts_of(index)`` reads a checkpoint-sourced MoE layer's routed experts
    as :func:`checkpoint_experts` does; without it they come off the module. ``traced`` traces the
    programs into it instead of compiling them (``_Step``)."""
    import torch

    from emmy import config
    from emmy.compiler.loader.quant import scale_layout
    from emmy.compiler.trace.huggingface import (
        build_attention_split_wrapper,
        build_moe_slots_wrapper,
        build_moe_split_wrapper,
        moe_block_parts,
        selected_layer_type,
    )

    cfg = decoder.config
    prefill = rows > 1
    # The model dtype: every 16-bit activation, cache and weight the step holds, as the plugin serves it.
    td = model.dtype
    step = _Step(get(str(td).removeprefix("torch.")), rows, ckpt, traced)
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    step.buffer("prompt", (context_length,), I64, "input")
    step.buffer("prompt_length", (1,), I64, "input")
    step.buffer("position", (1,), I64, "input")
    step.buffer("next_token", (1,), I64, "input")
    if not prefill:
        step.buffer("logits", (1, vocab), F32, "output")
        step.buffer("token", (1,), I64, "output")

    # Example inputs are distinct tensors: the tracer folds two arguments that share one into a
    # single aliased input.
    def scalar():
        return torch.zeros(1, dtype=torch.int64)

    def head_rows(width):
        return torch.zeros(rows, width * d, dtype=td)

    def cache_rows():
        return torch.zeros(1, kv, context_length, d, dtype=td)

    def mixer_outputs(index, routed, role="scratch"):
        # A MoE layer's mixer leaves its FP32 residual and the normalized rows its routed experts
        # read; the experts write the layer's output.
        if routed:
            return [step.buffer(f"layer{index}.h", (rows, h), F32, role=role), step.buffer(f"layer{index}.xn", (rows, h), role=role)]
        return [step.buffer(f"hidden{index + 1}", (rows, h), F32, role=role)]

    def experts(index, routed, expert, mixed, normed):
        """The routed half of a MoE layer after its mixer, which leaves the FP32 residual ``mixed``
        with the shared expert folded in and the normalized rows ``normed``: the router, the
        fixed-slot program of the ``expert`` form over the layer's E-stacked expert inputs, every
        row's picks at once, and the combine, which writes the layer's output. ``routed`` is the layer's
        ``(router, experts)`` pair. FP8 experts stay coded: each weight is its codes beside its block
        scales, which the program decodes in-graph as the plugin's does. The inputs are constants of
        the step when the device holds every expert; below that budget they are paged one expert per
        page, their pages stay in host memory, and the budget's device slots hold the most recently
        routed experts."""
        gate, stacked = routed
        k = gate.top_k
        if experts_of is None:
            fmt, tensors = None, {"w_gate_up": stacked.gate_up_proj.detach(), "w_down": stacked.down_proj.detach()}
        else:
            fmt, tensors = experts_of(index)
        if fmt not in (None, "f8e4m3", "f8e5m2"):
            raise ValueError(f"native generation runs 16-bit or FP8 routed experts, not {fmt}")
        coded = {n: (fmt, tuple(tensors[f"{n}_scale"].shape), "f32") for n in ("w_gate_up", "w_down")} if fmt else {}
        slots = config.gen_expert_slots(gate.num_experts)
        paged = slots is not None
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
            step.buffer(name, shape, dtype, role="input" if paged else "constant", data=None if paged else partial(_host, t))
            if paged:
                read = partial(lambda i, n, d: _host(experts_of(i)[1][n], d), index, n, dtype)
                step.pages[name] = partial(_host, t, dtype) if experts_of is None else read
        # One row's picks are a ``[k]`` selector, a chunk's ``[rows, k]``.
        picks = (k,) if rows == 1 else (rows, k)
        scores = step.buffer(f"layer{index}.scores", (rows, k))
        sel = step.buffer(f"layer{index}.sel", picks, I64)
        step.compiled(f"router{index}", router_module(gate, rows), (torch.zeros(rows, h, dtype=td),), [normed], [scores, sel], cache)
        partials = step.buffer(f"layer{index}.partials", (*picks, h))
        # The trace sees the weights' values, whatever stores them: the model dtype at the codes' shape.
        values = (torch.zeros(tensors[n].shape, dtype=td) for n in ("w_gate_up", "w_down"))
        examples = (torch.zeros(rows, h, dtype=td), torch.zeros(picks, dtype=torch.int64), *values)
        paging = tuple((n, 0, 1, slots) for n in names) if paged else ()
        wrapper = build_moe_slots_wrapper(expert)
        step.compiled(f"experts{index}", wrapper, examples, [normed, sel, *weights], [partials], cache, paged=paging, coded=coded)
        output = step.buffer(f"hidden{index + 1}", (rows, h), F32)
        examples = (torch.zeros(rows, h), torch.zeros(rows, k, dtype=td), torch.zeros(*picks, h, dtype=td))
        step.compiled(f"combine{index}", combine_module(), examples, [mixed, scores, partials], [output], cache)
        return output

    hidden = "hidden0"
    step.compiled(
        "embed",
        embed_module(decoder.embed_tokens.weight, rows),
        (torch.zeros(context_length, dtype=torch.int64), scalar(), scalar(), scalar()),
        ["prompt", "prompt_length", "position", "next_token"],
        [hidden],
        cache,
    )
    with torch.no_grad():
        cosine, sine = decoder.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float32), torch.arange(context_length).reshape(1, -1))
    cosine, sine = cosine[0].contiguous(), sine[0].contiguous()
    example = torch.zeros(rows, h, dtype=torch.float32)
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
        split = build_moe_split_wrapper(layer, float32_residual=True) if routed else None
        if selected_layer_type(decoder, layer, index) == "linear_attention":
            # A Gated DeltaNet layer always runs whole, the last prefill layer included: its state
            # must advance. It writes its state and history over the ones it reads.
            mixer = layer.linear_attn
            shapes = {
                "state": ((1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim), F32, torch.float32),
                "history": ((1, mixer.conv_dim, mixer.conv_kernel_size), step.dtype, td),
            }
            state = [step.buffer(f"layer{index}.{name}", shape, dtype, role="output") for name, (shape, dtype, _) in shapes.items()]
            step.state += state
            outputs = mixer_outputs(index, routed, unread)
            step.compiled(
                f"gdn{index}",
                gdn_module(layer),
                (example, *(torch.zeros(shape, dtype=dtype) for shape, _, dtype in shapes.values())),
                [hidden, *state],
                [*outputs, *state],
                cache,
            )
            # The last prefill layer's output is read by nothing, so its experts never run.
            hidden = experts(index, routed, split[2], *outputs) if routed and not last_prefill else outputs[0]
            continue
        pre, post = split[:2] if split else build_attention_split_wrapper(layer, float32_residual=True)
        # Qwen3.5's query projection also carries the attention output gate; ``pre`` emits it as a
        # fourth tensor and ``post`` takes it as a third input.
        gated = (("gate", heads),) if pre.emits_gate else ()
        names = [step.buffer(f"layer{index}.{name}", (rows, width * d)) for name, width in (("q", heads), ("k", kv), ("v", kv))]
        names += [step.buffer(f"layer{index}.{name}", (rows, width * d), role=unread) for name, width in gated]
        step.compiled(f"pre{index}", pre, (example,), [hidden], names, cache)
        rotated = step.buffer(f"layer{index}.rotated", (rows, heads * d), role=unread)
        # The cache: paged along its token axis by the programs below, written a chunk at a time
        # at ``position`` — a scatter into the cache the rope program takes in and hands back.
        keys = step.buffer(f"layer{index}.keys", (1, kv, context_length, d), role="output")
        values = step.buffer(f"layer{index}.values", (1, kv, context_length, d), role="output")
        step.compiled(
            f"rope{index}",
            rope_module(cosine, sine, heads, kv, d, rows),
            (head_rows(heads), head_rows(kv), head_rows(kv), scalar(), cache_rows(), cache_rows()),
            [*names[:3], "position", keys, values],
            [rotated, keys, values],
            cache,
            output_names=("rotated", "keys_next", "values_next"),
            paged=(("keys", 2, page_tokens), ("values", 2, page_tokens), ("keys_next", 2, page_tokens), ("values_next", 2, page_tokens)),
        )
        if last_prefill:
            continue
        attention = step.buffer(f"layer{index}.attention", (rows, heads * d))
        step.compiled(
            f"attend{index}",
            attend_module(heads, kv, d, context_length, rows),
            (head_rows(heads), cache_rows(), cache_rows(), scalar()),
            [rotated, keys, values, "position"],
            [attention],
            cache,
            paged=(("keys", 2, page_tokens), ("values", 2, page_tokens)),
        )
        outputs = mixer_outputs(index, routed)
        step.compiled(
            f"post{index}",
            post,
            (torch.zeros(rows, heads * d, dtype=td), example, *(head_rows(heads) for _ in gated)),
            [attention, hidden, *names[3:]],
            outputs,
            cache,
        )
        hidden = experts(index, routed, split[2], *outputs) if routed else outputs[0]
    if prefill:
        return step

    class Head(torch.nn.Sequential):
        # FP32 scores from 16-bit operands: rounding the logits to 16 bits could tie distinct scores.
        def forward(self, hidden):
            return torch.mm(self[0](hidden).to(self[1].weight.dtype), self[1].weight.transpose(0, 1), out_dtype=torch.float32)

    head = Head(decoder.norm, model.lm_head)
    step.compiled("head", head, (example,), [hidden], ["logits"], cache)
    step.compiled("greedy", greedy_module(vocab), (torch.zeros(1, vocab, dtype=torch.float32),), ["logits"], ["token"], cache)
    return step


def export_model(
    model, destination, *, context_length=MAX_CONTEXT, page_tokens=None, eos_ids=(), provenance=None, prefill_size=None, ckpt=None
):
    """Bundle one-token decode and fixed-width prefill; preparation owns every model operation.

    ``page_tokens`` is how many tokens of the KV cache one page holds. The cache is always a table
    of pages the runtime owns; the default — one page spanning the whole context — addresses
    exactly like the single contiguous array it replaces. ``prefill_size`` is the chunk width the
    prompt is consumed at; 1 disables chunking. ``ckpt`` is the checkpoint pair :func:`load_model`
    hands back beside a quantized checkpoint's twin; the trunk's coded weights are then spelled
    and bound from the shards, and its routed experts are read from them a layer at a time. A MoE
    layer's device holds ``EMMY_GEN_EXPERTS_GPU`` of its routed experts (default: all); below its
    expert count the pack keeps every expert's page for the runtime's pager, once, written a layer at
    a time, so the host never holds every expert."""
    if any(not 0 <= token < model.config.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    programs, prefill_size = _steps(model, context_length, page_tokens, prefill_size, ckpt)
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
    context_length=MAX_CONTEXT,
    page_tokens=None,
    prefill_size=None,
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
            model, destination, context_length=context_length, page_tokens=page_tokens, eos_ids=eos, prefill_size=prefill_size, ckpt=ckpt
        )


def trace_programs(model, *, context_length=MAX_CONTEXT, page_tokens=None, prefill_size=None, ckpt=None):
    """The programs :func:`export_model` compiles for ``model``, traced and not compiled: native generation's serving
    twins, ``{name: graph}``. A program is named ``<kind><width>-native`` after its step (``router1-native``,
    ``gdn16-native``), numbered after the first of a kind and width; the layers of one shape trace one program."""
    traced = {}
    _steps(model, context_length, page_tokens, prefill_size, ckpt, traced)
    twins = {}
    for name, graph in traced.values():
        twins[f"{name}-native{sum(n.startswith(f'{name}-native') for n in twins) or ''}"] = graph
    return twins


def _steps(model, context_length, page_tokens, prefill_size, ckpt, traced=None):
    """The decode program and, at a prefill width above one, the prefill program, and that width; with ``traced``,
    their programs traced into it instead (:class:`_Step`)."""
    from emmy.compiler.backend.plan_cache import PlanTemplateCache

    decoder = validate_model(model, context_length, ckpt)
    page_tokens = context_length if page_tokens is None else page_tokens
    if not 0 < page_tokens <= context_length or context_length % page_tokens:
        raise ValueError(f"page_tokens={page_tokens} must divide the context capacity {context_length}")
    prefill_size = PREFILL_SIZE if prefill_size is None else prefill_size
    if type(prefill_size) is not int or not 1 <= prefill_size <= MAX_CONTEXT:
        raise ValueError("prefill size must be within supported context capacity")
    prefill_size = min(prefill_size, context_length)
    # One layer at a time: a program reads a layer's experts as it compiles the layer, and the pack reads
    # them again as it writes their pages, so below the budget the host holds one layer's experts at most.
    experts_of = None if ckpt is None else lru_cache(maxsize=1)(partial(checkpoint_experts, ckpt[0], model.dtype))
    cache = PlanTemplateCache()
    programs = {"decode": _program(model, decoder, context_length, 1, page_tokens, cache, ckpt, experts_of, traced)}
    if prefill_size > 1:
        programs["prefill"] = _program(model, decoder, context_length, prefill_size, page_tokens, cache, ckpt, experts_of, traced)
    return programs, prefill_size
