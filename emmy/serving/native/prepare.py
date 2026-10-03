"""Export dense FP16 Qwen3 and Qwen3.5 text models as static decode and chunked prefill programs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, I64

MAX_CONTEXT = 4096
GENERATION_VERSION = 6
PREFILL_SIZE = 16
MASK_FILL = -1e9
MODEL_TYPES = ("qwen3", "qwen3_5_text")
LAYER_TYPES = ("full_attention", "linear_attention")


def validate_model(model, context_length):
    """Reject unsupported architectures before tracing or allocating device state; return the
    text decoder, which a Qwen3.5 checkpoint wraps beside a vision tower native never loads."""
    import torch

    from emmy.compiler.trace.huggingface import find_text_decoder

    decoder = find_text_decoder(model)
    cfg = decoder.config
    if cfg.model_type not in MODEL_TYPES or any(getattr(c, "quantization_config", None) for c in (cfg, model.config)):
        raise ValueError("native generation requires an unquantized dense Qwen3 or Qwen3.5 text model")
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
    """Rotate q and k at their positions in FP32, round once to FP16, and hand k and v to the
    cache: the two cache outputs are ``rows`` tokens wide and land at ``position`` through the
    page tables. A partial rotary embedding (Qwen3.5) rotates only the leading ``cosine.shape[-1]``
    of each head and passes the rest through."""
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

        def forward(self, q, k, v, position):
            positions = position + torch.arange(rows)
            c, s = self.cosine[positions].unsqueeze(1), self.sine[positions].unsqueeze(1)
            rotated = self.rotate(q, heads, c, s).view(rows, heads * head_dim)
            keys = self.rotate(k, kv_heads, c, s).transpose(0, 1).unsqueeze(0)
            values = v.view(rows, kv_heads, head_dim).transpose(0, 1).unsqueeze(0)
            return rotated, keys, values

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
    its FP32-residual form, over one batch row. The state and history it reads and the ones it
    writes are distinct buffers — a kernel's output never shares memory with its inputs — paged
    one page per batch row, so the runtime hands the step its pages through tables."""
    import torch

    from emmy.compiler.trace.huggingface import build_gdn_state_wrapper

    class Carried(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.wrapper = build_gdn_state_wrapper(layer, float32_residual=True)

        def forward(self, hidden, state, history):
            y, state_next, history_next = self.wrapper(hidden.unsqueeze(0), state, history)
            return y.squeeze(0), state_next, history_next

    return Carried()


def greedy_module(vocab):
    """The lowest token ID among the maximum logits, from the reductions the compiler has: the peak,
    then the largest negated ID among the tokens at the peak. A nonfinite logit leaves no token at
    the peak, and the result is the vocabulary size, which the runtime rejects."""
    import torch

    class Greedy(torch.nn.Module):
        def forward(self, logits):
            peak = logits.amax(dim=1, keepdim=True)
            ids = torch.arange(vocab).unsqueeze(0)
            return -torch.where(logits == peak, -ids, -vocab).amax(dim=1)

    return Greedy()


class _Step:
    def __init__(self, prefill=False):
        # A decode step ends at the logits and the greedy token; the runtime hands the token it
        # selects back as the next step's input. A prefill chunk writes the cache and nothing else.
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
        # The (read, write) pairs of paged buffers the runtime carries from one step to the next.
        self.carried = []

    def buffer(self, name, shape, dtype=F16, role="scratch", data=None):
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None:
            self.bindings[name] = np.ascontiguousarray(data).tobytes()
        return name

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache, output_names=(), paged=()):
        """Compile ``wrapper`` and splice its plan in: its inputs and outputs take the step's names,
        everything else is scoped by ``prefix``. ``output_names`` renames the traced outputs (the
        tracer names them after their last op) so ``paged`` can address them: it names the
        wrapper's buffers that are the step's paged caches, as ``(graph name, axis, page tokens,
        start)``, and the step buffer takes that paging. Their launches address the step buffer's
        pages, so their own shape — a chunk of the cache, or all of it — need not equal the step
        buffer's. Both are spelled in the graph's own names, never the step's, so every layer
        traces the same graph and compiles once."""
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.serving.gen_runner import _bind_plan_constants, trace_split

        graph = trace_split(wrapper, examples, None)
        for old, new in zip(list(graph.outputs), output_names, strict=False):
            graph.rename_node(old, new)
        if paged:
            graph.hints.set("cuda.paged_buffers", tuple(paged))
        plan = cache.resolve(graph, lambda g: plan_from_graph(CudaBackend(tune_db="auto").compile(g)))
        # A serial launch's axes are the one runtime argument the standalone runtime resolves itself.
        if (
            plan.symbolic_bindings
            or plan.runtime_constants
            or any(
                launch.tma_descriptors or launch.indirect_args or set(launch.runtime_args) - {name for name, _ in launch.serial}
                for launch in plan.launches
            )
        ):
            raise ValueError("native generation requires static ordinary-pointer compiled programs")
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
            self.plan.launches.append(
                replace(
                    launch,
                    node_id=f"{prefix}.{launch.node_id}",
                    arg_names=tuple(bound(n) for n in launch.arg_names),
                    zero_outputs=tuple(names[n] for n in launch.zero_outputs),
                    zero_prologues=tuple(names[n] for n in launch.zero_prologues),
                    writes=tuple(names[n] for n in launch.writes),
                )
            )


def _program(model, decoder, context_length, rows, page_tokens, cache):
    """One program: the one-token decode step at ``rows == 1``, else a prefill chunk of ``rows``
    prompt tokens that writes the cache and the carried state and computes nothing past the last
    attention layer's keys."""
    import torch

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper, selected_layer_type

    cfg = decoder.config
    prefill = rows > 1
    step = _Step(prefill=prefill)
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
        if selected_layer_type(decoder, layer, index) == "linear_attention":
            # A Gated DeltaNet layer always runs whole, the last prefill layer included: its state
            # must advance. The state it reads and the state it writes are two paged buffers, one
            # page each, that the runtime exchanges after every step.
            mixer = layer.linear_attn
            shapes = {
                "state": ((1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim), F32, torch.float32),
                "history": ((1, mixer.conv_dim, mixer.conv_kernel_size), F16, torch.float16),
            }
            reads, writes = [], []
            for name, (shape, dtype, _) in shapes.items():
                reads.append(step.buffer(f"layer{index}.{name}", shape, dtype, role="output"))
                writes.append(step.buffer(f"layer{index}.{name}_next", shape, dtype, role="output"))
                step.carried.append((reads[-1], writes[-1]))
            output = step.buffer(f"hidden{index + 1}", (rows, h), F32, role=unread)
            step.compiled(
                f"gdn{index}",
                gdn_module(layer),
                (example, *(torch.zeros(shape, dtype=dtype) for shape, _, dtype in shapes.values())),
                [hidden, *reads],
                [output, *writes],
                cache,
                output_names=("y", "state_next", "history_next"),
                paged=tuple((name, 0, 1, None) for name in ("state", "history", "state_next", "history_next")),
            )
            hidden = output
            continue
        pre, post = build_attention_split_wrapper(layer, float32_residual=True)
        # Qwen3.5's query projection also carries the attention output gate; ``pre`` emits it as a
        # fourth tensor and ``post`` takes it as a third input.
        gated = (("gate", heads),) if pre.emits_gate else ()
        names = [step.buffer(f"layer{index}.{name}", (rows, width * d)) for name, width in (("q", heads), ("k", kv), ("v", kv))]
        names += [step.buffer(f"layer{index}.{name}", (rows, width * d), role=unread) for name, width in gated]
        step.compiled(f"pre{index}", pre, (example,), [hidden], names, cache)
        rotated = step.buffer(f"layer{index}.rotated", (rows, heads * d), role=unread)
        # The cache: paged along its token axis by the programs below, written a chunk at a time
        # at ``position``.
        keys = step.buffer(f"layer{index}.keys", (1, kv, context_length, d), role="output")
        values = step.buffer(f"layer{index}.values", (1, kv, context_length, d), role="output")
        step.compiled(
            f"rope{index}",
            rope_module(cosine, sine, heads, kv, d, rows),
            (head_rows(heads), head_rows(kv), head_rows(kv), scalar()),
            [*names[:3], "position"],
            [rotated, keys, values],
            cache,
            output_names=("rotated", "keys", "values"),
            paged=(("keys", 2, page_tokens, "position"), ("values", 2, page_tokens, "position")),
        )
        if last_prefill:
            continue
        output = step.buffer(f"hidden{index + 1}", (rows, h), F32)
        attention = step.buffer(f"layer{index}.attention", (rows, heads * d))
        step.compiled(
            f"attend{index}",
            attend_module(heads, kv, d, context_length, rows),
            (head_rows(heads), cache_rows(), cache_rows(), scalar()),
            [rotated, keys, values, "position"],
            [attention],
            cache,
            paged=(("keys", 2, page_tokens, None), ("values", 2, page_tokens, None)),
        )
        step.compiled(
            f"post{index}",
            post,
            (torch.zeros(rows, heads * d, dtype=torch.float16), example, *(head_rows(heads) for _ in gated)),
            [attention, hidden, *names[3:]],
            [output],
            cache,
        )
        hidden = output
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


def export_model(model, destination, *, context_length=MAX_CONTEXT, page_tokens=None, eos_ids=(), provenance=None, prefill_size=None):
    """Bundle one-token decode and fixed-width prefill; preparation owns every model operation.

    ``page_tokens`` is how many tokens of the KV cache one page holds. The cache is always a table
    of pages the runtime owns; the default — one page spanning the whole context — addresses
    exactly like the single contiguous array it replaces. ``prefill_size`` is the chunk width the
    prompt is consumed at; 1 disables chunking."""
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
    programs = {"decode": _program(model, decoder, context_length, 1, page_tokens, cache)}
    if prefill_size > 1:
        programs["prefill"] = _program(model, decoder, context_length, prefill_size, page_tokens, cache)
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
                # The runtime carries each pair's state across steps: it reads the first buffer
                # and writes the second, through page tables it exchanges after every step.
                "carried": [{"read": read, "write": write} for read, write in programs["decode"].carried],
            }
        },
        provenance=provenance,
    )
