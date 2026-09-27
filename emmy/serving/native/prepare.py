"""Export dense FP16 Qwen3 as one static, stateful token-step execution plan."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, I64

MAX_CONTEXT = 4096
GENERATION_VERSION = 3
MASK_FILL = -1e9


def validate_model(model, context_length):
    """Reject unsupported architectures before tracing or allocating device state."""
    import torch

    cfg = model.config
    if cfg.model_type != "qwen3" or getattr(cfg, "quantization_config", None):
        raise ValueError("native generation requires unquantized dense Qwen3")
    if not 1 <= context_length <= min(MAX_CONTEXT, cfg.max_position_embeddings):
        raise ValueError("native context length is outside supported model limits")
    if model.training:
        raise ValueError("native export requires evaluation mode")
    rope = cfg.rope_parameters
    if rope.get("rope_type") != "default" or rope.get("partial_rotary_factor", 1.0) != 1.0:
        raise ValueError("native generation requires default full rotary embedding")
    if any(kind != "full_attention" for kind in cfg.layer_types):
        raise ValueError("native generation does not support sliding attention")
    if cfg.head_dim <= 0 or cfg.head_dim % 2 or cfg.num_key_value_heads <= 0 or cfg.num_attention_heads % cfg.num_key_value_heads:
        raise ValueError("invalid Qwen3 attention geometry")
    if any(p.dtype != torch.float16 or p.device.type != "cpu" for p in model.parameters()):
        raise ValueError("native export requires an FP16 model on CPU")


def embed_module(weight):
    """The token a step embeds is the prompt's at this position while the prompt lasts, else the
    previous step's selection: a gather the device decides, so the host binds one position scalar
    per step and never sees a prompt token."""
    import torch

    class Embed(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = weight

        def forward(self, prompt, prompt_length, position, next_token):
            token = torch.where(position < prompt_length, prompt[position], next_token)
            return self.weight[token].float()

    return Embed()


def rope_module(cosine, sine, heads, kv_heads, head_dim):
    """Rotate q and k at this position in FP32, round once to FP16, and hand k and v to the cache:
    the two cache outputs are one token wide and land at ``position`` through the page tables."""
    import torch

    class Rope(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("cosine", cosine)
            self.register_buffer("sine", sine)

        def rotate(self, x, n, c, s):
            x = x.view(1, n, 1, head_dim).float()
            half = head_dim // 2
            paired = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
            return (x * c + paired * s).to(torch.float16)

        def forward(self, q, k, v, position):
            c, s = self.cosine[position], self.sine[position]
            return self.rotate(q, heads, c, s).view(1, heads * head_dim), self.rotate(k, kv_heads, c, s), v.view(1, kv_heads, 1, head_dim)

    return Rope()


def attend_module(heads, kv_heads, head_dim, context_length):
    """Causal grouped-query attention over the whole cache in FP32, rounding only its output: every
    position past this one is masked on the device, so the launch is the same at every position."""
    import torch
    import torch.nn.functional as F

    class Attend(torch.nn.Module):
        def forward(self, q, keys, values, position):
            group = heads // kv_heads
            q4 = q.view(1, heads, 1, head_dim).float()
            k4 = keys.repeat_interleave(group, dim=1).float()
            v4 = values.repeat_interleave(group, dim=1).float()
            mask = torch.where(torch.arange(context_length) <= position, 0.0, MASK_FILL).view(1, 1, 1, context_length)
            out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask)
            return out.to(torch.float16).view(1, heads * head_dim)

    return Attend()


class _Step:
    def __init__(self):
        # The step ends at the logits; the runtime selects the token on the host and hands it back.
        self.plan = ExecutionPlan("cuda", ["prompt", "prompt_length", "position", "next_token"], ["logits"], [], {}, {}, [], {})
        self.bindings = {}

    def buffer(self, name, shape, dtype=F16, role="scratch", data=None, page_tokens=None):
        """One step-level buffer; ``page_tokens`` declares a cache paged along its token axis (2)."""
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None:
            self.bindings[name] = np.ascontiguousarray(data).tobytes()
        if page_tokens is not None:
            self.plan.paged[name] = (2, page_tokens, None)
        return name

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache, output_names=(), paged=()):
        """Compile ``wrapper`` and splice its plan in: its inputs and outputs take the step's names,
        everything else is scoped by ``prefix``. ``output_names`` renames the traced outputs (the
        tracer names them after their last op) so ``paged`` can address them: it names the
        wrapper's buffers that are the step's paged caches, as ``(graph name, axis, page tokens,
        start)``. Their launches address the step buffer's pages, so their own shape — one token of
        the cache, or all of it — need not equal the step buffer's. Both are spelled in the graph's
        own names, never the step's, so every layer traces the same graph and compiles once."""
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.serving.gen_runner import _bind_plan_constants, trace_split

        graph = trace_split(wrapper, examples, None)
        for old, new in zip(list(graph.outputs), output_names, strict=False):
            graph.rename_node(old, new)
        if paged:
            graph.hints.set("cuda.paged_buffers", tuple(paged))
        plan = cache.resolve(graph, lambda g: plan_from_graph(CudaBackend(tune_db="auto").compile(g)))
        if (
            plan.symbolic_bindings
            or plan.runtime_constants
            or any(launch.tma_descriptors or launch.indirect_args or launch.runtime_args for launch in plan.launches)
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


def export_model(model, destination, *, context_length=MAX_CONTEXT, page_tokens=None, eos_ids=(), provenance=None):
    """Compile and bundle a dense Qwen3 checkpoint; no Python operation is needed after export.

    ``page_tokens`` is how many tokens of the KV cache one page holds. The cache is always a table
    of pages the runtime owns; the default — one page spanning the whole context — addresses
    exactly like the single contiguous array it replaces."""
    import torch

    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    validate_model(model, context_length)
    page_tokens = context_length if page_tokens is None else page_tokens
    if not 0 < page_tokens <= context_length or context_length % page_tokens:
        raise ValueError(f"page_tokens={page_tokens} must divide the context capacity {context_length}")
    cfg = model.config
    if any(not 0 <= token < cfg.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    cache = PlanTemplateCache()
    step = _Step()
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    step.buffer("prompt", (context_length,), I64, "input")
    step.buffer("prompt_length", (1,), I64, "input")
    step.buffer("position", (1,), I64, "input")
    step.buffer("next_token", (1,), I64, "input")
    step.buffer("logits", (1, vocab), F16, "output")

    # Example inputs are distinct tensors: the tracer folds two arguments that share one into a
    # single aliased input.
    def scalar():
        return torch.zeros(1, dtype=torch.int64)

    hidden = "hidden0"
    step.compiled(
        "embed",
        embed_module(model.model.embed_tokens.weight),
        (torch.zeros(context_length, dtype=torch.int64), scalar(), scalar(), scalar()),
        ["prompt", "prompt_length", "position", "next_token"],
        [hidden],
        cache,
    )
    with torch.no_grad():
        cosine, sine = model.model.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float32), torch.arange(context_length).reshape(1, -1))
    cosine, sine = cosine[0].contiguous(), sine[0].contiguous()
    example = torch.zeros(1, h, dtype=torch.float32)

    def head_rows(width):
        return torch.zeros(1, width * d, dtype=torch.float16)

    def cache_rows():
        return torch.zeros(1, kv, context_length, d, dtype=torch.float16)

    for index, layer in enumerate(model.model.layers):
        pre, post = build_attention_split_wrapper(layer, float32_residual=True)
        names = [step.buffer(f"layer{index}.{name}", (1, width * d)) for name, width in (("q", heads), ("k", kv), ("v", kv))]
        step.compiled(f"pre{index}", pre, (example,), [hidden], names, cache)
        rotated = step.buffer(f"layer{index}.rotated", (1, heads * d))
        # The cache: paged along its token axis, written one token at a time at ``position``.
        keys = step.buffer(f"layer{index}.keys", (1, kv, context_length, d), role="output", page_tokens=page_tokens)
        values = step.buffer(f"layer{index}.values", (1, kv, context_length, d), role="output", page_tokens=page_tokens)
        step.compiled(
            f"rope{index}",
            rope_module(cosine, sine, heads, kv, d),
            (head_rows(heads), head_rows(kv), head_rows(kv), scalar()),
            [*names, "position"],
            [rotated, keys, values],
            cache,
            output_names=("rotated", "keys", "values"),
            paged=(("keys", 2, page_tokens, "position"), ("values", 2, page_tokens, "position")),
        )
        attention = step.buffer(f"layer{index}.attention", (1, heads * d))
        step.compiled(
            f"attend{index}",
            attend_module(heads, kv, d, context_length),
            (head_rows(heads), cache_rows(), cache_rows(), scalar()),
            [rotated, keys, values, "position"],
            [attention],
            cache,
            paged=(("keys", 2, page_tokens, None), ("values", 2, page_tokens, None)),
        )
        output = step.buffer(f"hidden{index + 1}", (1, h), F32)
        step.compiled(f"post{index}", post, (torch.zeros(1, heads * d, dtype=torch.float16), example), [attention, hidden], [output], cache)
        hidden = output

    class Head(torch.nn.Sequential):
        def forward(self, hidden):
            return self[1](self[0](hidden).to(self[1].weight.dtype))

    head = Head(model.model.norm, model.lm_head)
    step.compiled("head", head, (example,), [hidden], ["logits"], cache)
    return save_executable(
        destination,
        {"decode": step.plan},
        bindings={"decode": step.bindings},
        key={
            "generation": {"version": GENERATION_VERSION, "context_length": context_length, "vocab_size": vocab, "eos_ids": list(eos_ids)}
        },
        provenance=provenance,
    )
