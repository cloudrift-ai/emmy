"""Export Qwen3, Qwen3.5 and Qwen3.5-MoE text models as static decode and chunked prefill programs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, I64

MAX_CONTEXT = 4096
GENERATION_VERSION = 7
PREFILL_SIZE = 16
MASK_FILL = -1e9
MODEL_TYPES = ("qwen3", "qwen3_5_text", "qwen3_5_moe_text")
LAYER_TYPES = ("full_attention", "linear_attention")


def load_model(model_id, revision=None):
    """The FP16 CPU model an export traces, and the ``ckpt`` pair its coded trunk binds from —
    ``None`` for an ordinary checkpoint, whose weights come off the module.

    A quantized checkpoint cannot go through ``from_pretrained`` (transformers would engage its own
    quantizer machinery). Its twin is built from the config with the trunk's coded linears left as
    placeholders, and every program binds them from the shards instead — the serving runner's
    checkpoint-sourced lane, which keeps an FP8 trunk at its stored size (weight-only under FP16
    activations) rather than doubling it in a decode."""
    import torch
    from transformers import AutoModelForCausalLM, GenerationConfig

    from emmy.compiler.trace.huggingface import find_text_decoder, load_quantized_split, quantized_checkpoint_dir
    from emmy.serving.gen_runner import checkpoint_key_map

    qdir = quantized_checkpoint_dir(model_id, revision)
    if qdir is None:
        return AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=torch.float16).eval().cpu(), None
    model, store = load_quantized_split(qdir, torch.float16, compress_trunk=True)
    # Rotary buffers are non-persistent in transformers, so the meta-built twin has none: rebuild
    # the module on CPU, as the trace twin does, since the export reads its tables.
    decoder = find_text_decoder(model)
    decoder.rotary_emb = type(decoder.rotary_emb)(decoder.config)
    if (qdir / "generation_config.json").exists():
        model.generation_config = GenerationConfig.from_pretrained(qdir)
    return model, (store["dir"], checkpoint_key_map(model, store["keys"]))


def validate_model(model, context_length):
    """Reject unsupported architectures before tracing or allocating device state; return the
    text decoder, which a Qwen3.5 checkpoint wraps beside a vision tower native never loads."""
    import torch

    from emmy.compiler.trace.huggingface import find_text_decoder

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
    if any(p.dtype != torch.float16 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("native export requires an FP16 model on CPU")
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
    """Rotate q and k at their positions in FP32, round once to FP16, and store k and v into the
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
            x = x.view(rows, n, head_dim).float()
            rotated, passed = (x[..., :rotary_dim], x[..., rotary_dim:]) if rotary_dim < head_dim else (x, None)
            half = rotary_dim // 2
            paired = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
            rotated = rotated * c + paired * s
            return (rotated if passed is None else torch.cat((rotated, passed), dim=-1)).to(torch.float16)

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
            return out.to(torch.float16).squeeze(0).transpose(0, 1).reshape(rows, heads * head_dim)

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
    top ``k`` as ``k`` masked maximum reductions, and the picked scores normalized to sum to one and
    rounded to the model dtype. Each reduction takes the peak and the lowest expert ID at it, as
    greedy selection does, then masks that expert below every probability. A row with no element at
    its peak (a NaN) picks the last expert, so the gather stays inside the stack; its logits come out
    nonfinite and greedy selection fails the request. It returns the
    ``[rows, k]`` scores and each row's picks as a ``[k]`` selector; with more than one row, also
    each row as a ``[1, hidden]`` copy, since the fixed-slot program runs once per row on whole
    buffers."""
    import torch
    import torch.nn.functional as F

    experts, k = gate.num_experts, gate.top_k

    class Router(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = gate.weight

        def forward(self, xn):
            remaining = torch.softmax(F.linear(xn, self.weight).float(), dim=-1)
            ids = torch.arange(experts).unsqueeze(0)
            picks, peaks = [], []
            for _ in range(k):
                peak = remaining.amax(dim=1, keepdim=True)
                pick = -torch.where(remaining == peak, -ids, 1 - experts).amax(dim=1, keepdim=True)
                remaining = torch.where(ids == pick, -1.0, remaining)
                picks.append(pick)
                peaks.append(peak)
            peaks = torch.cat(peaks, dim=1)
            scores = (peaks / peaks.sum(dim=1, keepdim=True)).to(xn.dtype)
            sel = torch.cat(picks, dim=1)
            return scores, *(sel[r] for r in range(rows)), *(xn[r : r + 1] for r in range(rows) if rows > 1)

    return Router()


def combine_module():
    """The routed experts' weighted sum added to the mixer's FP32 residual: each row's ``[1, k]``
    scores times its ``[k, hidden]`` partials in the model dtype, as the fixed-slot combine
    computes it, then the residual add in FP32."""
    import torch

    class Combine(torch.nn.Module):
        def forward(self, residual, scores, *partials):
            return residual + torch.matmul(scores.unsqueeze(1), torch.stack(partials)).squeeze(1)

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


class _Step:
    def __init__(self, prefill=False, ckpt=None):
        # A decode step ends at the logits and the greedy token; the runtime hands the token it
        # selects back as the next step's input. A prefill chunk writes the cache and nothing else.
        # ``ckpt`` (``load_model``) puts every program on the checkpoint-sourced lane: the trunk's
        # coded weights are spelled from the shards and bound from them, never off the module.
        self.ckpt = ckpt
        self.plan = ExecutionPlan(
            "cuda",
            ["prompt", "prompt_length", "position", "next_token"],
            [] if prefill else ["logits", "token"],
            [],
            {},
            {},
            [],
            {},
        )
        self.bindings = {}
        # The buffers a step writes in place and carries to the next position; a request starts them at zero.
        self.state = []

    def buffer(self, name, shape, dtype=F16, role="scratch", data=None):
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None:
            self.bindings[name] = np.ascontiguousarray(data).tobytes()
        return name

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache, output_names=(), paged=()):
        """Compile ``wrapper`` and splice its plan in: its inputs and outputs take the step's names,
        everything else is scoped by ``prefix``. ``output_names`` renames the traced outputs (the
        tracer names them after their last op) so ``paged`` can address them: it names the
        wrapper's buffers that are the step's paged caches, as ``(graph name, axis, page tokens)``,
        and the step buffer takes that paging. A cache a program writes in place comes in as one graph
        buffer and goes out as another stored over it; both take the step buffer's name, so its pages
        are the ones written. Everything is spelled in the graph's own names, never the step's, so every
        layer traces the same graph and compiles once."""
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.serving.gen_runner import _bind_plan_constants, _plan_sources, _traced_split

        graph = _traced_split(wrapper, examples, None, self.ckpt)
        for old, new in zip(list(graph.outputs), output_names, strict=False):
            graph.rename_node(old, new)
        if paged:
            graph.hints.set("cuda.paged_buffers", tuple(paged))
        plan = cache.resolve(graph, lambda g: plan_from_graph(CudaBackend(tune_db="auto").compile(g)))
        # A serial launch's axes are the one runtime argument the standalone runtime resolves itself.
        # A TMA descriptor is encoded once from its source's fixed address, so it may not read pages.
        pages = {name for name, *_ in paged}
        why = ["symbolic bindings"] if plan.symbolic_bindings else []
        why += ["runtime constants"] if plan.runtime_constants else []
        for launch in plan.launches:
            if {t.src_buf for t in launch.tma_descriptors} & pages:
                why.append(f"{launch.node_id} reads a paged buffer through a TMA descriptor")
            if extra := set(launch.runtime_args) - {name for name, _ in launch.serial}:
                why.append(f"{launch.node_id} takes runtime arguments {sorted(extra)}")
        if why:
            raise ValueError(f"native generation requires static ordinary-pointer compiled programs; {prefix}: " + "; ".join(why))
        if self.ckpt is not None:
            sources = _plan_sources(plan, wrapper, F16, *self.ckpt)
        else:
            sources = {
                name: t.detach().cpu().numpy()
                for name, t in list(wrapper.named_parameters(remove_duplicate=False)) + list(wrapper.named_buffers(remove_duplicate=False))
            }
        constants = _bind_plan_constants(plan, sources, None)
        names = {b.name: f"{prefix}.{b.name}" for b in plan.buffers}
        names.update(zip(plan.inputs, inputs, strict=True))
        names.update(zip(plan.outputs, outputs, strict=True))
        self.plan.paged.update({names[n]: paging for n, paging in plan.paged.items()})
        existing = {b.name: b for b in self.plan.buffers}
        for buffer in plan.buffers:
            name = names[buffer.name]
            if name in existing:
                if name not in self.plan.paged and (existing[name].shape != buffer.shape or existing[name].dtype != buffer.dtype):
                    raise ValueError(f"incompatible program seam: {name}")
                continue
            role = "constant" if buffer.role == "constant" else "scratch"
            self.plan.buffers.append(replace(buffer, name=name, role=role))
            if role == "constant":
                value = constants.get(buffer.name, plan.constants.get(buffer.name))
                if value is None:
                    raise ValueError(f"unresolved constant: {name}")
                self.bindings[name] = np.broadcast_to(np.asarray(value, dtype=buffer.dtype.np), buffer.resolve_shape({})).copy().tobytes()
        for name, kernel in plan.kernels.items():
            previous = self.plan.kernels.get(name)
            if previous is not None and previous != kernel:
                raise ValueError(f"conflicting compiled kernel: {name}")
            self.plan.kernels[name] = kernel

        def bound(n):
            # A paged operand names its page table; the table follows the buffer's step name.
            return f"{names[n[: -len('__pages')]]}__pages" if n.endswith("__pages") else names[n]

        for launch in plan.launches:
            # A descriptor argument keeps its name, which is local to its launch; its source follows the step.
            descriptors = {t.name for t in launch.tma_descriptors}
            self.plan.launches.append(
                replace(
                    launch,
                    node_id=f"{prefix}.{launch.node_id}",
                    arg_names=tuple(n if n in descriptors else bound(n) for n in launch.arg_names),
                    tma_descriptors=tuple(replace(t, src_buf=names[t.src_buf]) for t in launch.tma_descriptors),
                    zero_outputs=tuple(names[n] for n in launch.zero_outputs),
                    zero_prologues=tuple(names[n] for n in launch.zero_prologues),
                    writes=tuple(names[n] for n in launch.writes),
                )
            )


def _program(model, decoder, context_length, rows, page_tokens, cache, ckpt):
    """One program: the one-token decode step at ``rows == 1``, else a prefill chunk of ``rows``
    prompt tokens that writes the cache and the carried state and computes nothing past the last
    attention layer's keys."""
    import torch

    from emmy.compiler.trace.huggingface import (
        build_attention_split_wrapper,
        build_moe_slots_wrapper,
        build_moe_split_wrapper,
        moe_block_parts,
        selected_layer_type,
    )

    cfg = decoder.config
    prefill = rows > 1
    step = _Step(prefill=prefill, ckpt=ckpt)
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
        return torch.zeros(rows, width * d, dtype=torch.float16)

    def cache_rows():
        return torch.zeros(1, kv, context_length, d, dtype=torch.float16)

    def mixer_outputs(index, routed, role="scratch"):
        # A MoE layer's mixer leaves its FP32 residual and the normalized rows its routed experts
        # read; the experts write the layer's output.
        if routed:
            return [step.buffer(f"layer{index}.h", (rows, h), F32, role=role), step.buffer(f"layer{index}.xn", (rows, h), role=role)]
        return [step.buffer(f"hidden{index + 1}", (rows, h), F32, role=role)]

    def experts(index, routed, expert, mixed, normed):
        """The routed half of a MoE layer after its mixer, which leaves the FP32 residual ``mixed``
        with the shared expert folded in and the normalized rows ``normed``: the router, the
        fixed-slot program of the ``expert`` form over the layer's resident expert weights once per
        row, and the combine, which writes the layer's output. ``routed`` is the layer's
        ``(router, experts)`` pair."""
        gate, stacked = routed
        k = gate.top_k
        tensors = {"w_gate_up": stacked.gate_up_proj.detach(), "w_down": stacked.down_proj.detach()}
        weights = [step.buffer(f"layer{index}.{n}", tuple(t.shape), role="constant", data=t.numpy()) for n, t in tensors.items()]
        scores = step.buffer(f"layer{index}.scores", (rows, k))
        sel = [step.buffer(f"layer{index}.sel{r}", (k,), I64) for r in range(rows)]
        # One row reads the normalized rows whole; a chunk's rows are copied out one per buffer.
        x = [step.buffer(f"layer{index}.x{r}", (1, h)) for r in range(rows)] if rows > 1 else []
        normed_rows = (torch.zeros(rows, h, dtype=torch.float16),)
        step.compiled(f"router{index}", router_module(gate, rows), normed_rows, [normed], [scores, *sel, *x], cache)
        x = x or [normed]
        partials = [step.buffer(f"layer{index}.partials{r}", (k, h)) for r in range(rows)]
        slots = build_moe_slots_wrapper(expert)
        examples = (torch.zeros(1, h, dtype=torch.float16), torch.zeros(k, dtype=torch.int64), *tensors.values())
        for r in range(rows):
            step.compiled(f"experts{index}.{r}", slots, examples, [x[r], sel[r], *weights], [partials[r]], cache)
        output = step.buffer(f"hidden{index + 1}", (rows, h), F32)
        examples = (
            torch.zeros(rows, h),
            torch.zeros(rows, k, dtype=torch.float16),
            *(torch.zeros(k, h, dtype=torch.float16) for _ in partials),
        )
        step.compiled(f"combine{index}", combine_module(), examples, [mixed, scores, *partials], [output], cache)
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
                "history": ((1, mixer.conv_dim, mixer.conv_kernel_size), F16, torch.float16),
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
            (torch.zeros(rows, heads * d, dtype=torch.float16), example, *(head_rows(heads) for _ in gated)),
            [attention, hidden, *names[3:]],
            outputs,
            cache,
        )
        hidden = experts(index, routed, split[2], *outputs) if routed else outputs[0]
    if prefill:
        return step

    class Head(torch.nn.Sequential):
        # FP32 scores from FP16 operands: rounding the logits to FP16 could tie distinct scores.
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
    and bound from the shards."""
    from emmy.compiler.backend.plan_cache import PlanTemplateCache

    decoder = validate_model(model, context_length)
    page_tokens = context_length if page_tokens is None else page_tokens
    if not 0 < page_tokens <= context_length or context_length % page_tokens:
        raise ValueError(f"page_tokens={page_tokens} must divide the context capacity {context_length}")
    prefill_size = PREFILL_SIZE if prefill_size is None else prefill_size
    if type(prefill_size) is not int or not 1 <= prefill_size <= MAX_CONTEXT:
        raise ValueError("prefill size must be within supported context capacity")
    if any(not 0 <= token < model.config.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    prefill_size = min(prefill_size, context_length)
    cache = PlanTemplateCache()
    programs = {"decode": _program(model, decoder, context_length, 1, page_tokens, cache, ckpt)}
    if prefill_size > 1:
        programs["prefill"] = _program(model, decoder, context_length, prefill_size, page_tokens, cache, ckpt)
    return save_executable(
        destination,
        {name: step.plan for name, step in programs.items()},
        bindings={name: step.bindings for name, step in programs.items()},
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
