"""Export dense FP16 Qwen3 as one static, stateful token-step execution plan."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec, plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, I64
from emmy.serving.native.kernels import SOURCE

MAX_CONTEXT = 4096
CUDA_THREADS = 128
GENERATION_VERSION = 1


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


class _Step:
    def __init__(self):
        self.plan = ExecutionPlan("cuda", ["prompt", "prompt_length", "position"], ["logits", "next_token"], [], {}, {}, [], {})
        self.bindings = {}

    def buffer(self, name, shape, dtype=F16, role="scratch", data=None):
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None:
            self.bindings[name] = np.ascontiguousarray(data).tobytes()
        return name

    def launch(self, kernel, args, source, *, writes, blocks=1, shared=0, threads=CUDA_THREADS):
        self.plan.kernels[kernel] = KernelSpec(source=source)
        self.plan.launches.append(
            LaunchSpec(kernel, kernel, tuple(args), ((blocks,), (1,), (1,)), ((threads,), (1,), (1,)), shared, (), writes=tuple(writes))
        )

    def compiled(self, prefix, wrapper, examples, inputs, outputs, cache):
        from emmy.compiler.backend.cuda.backend import CudaBackend
        from emmy.serving.gen_runner import _bind_plan_constants, trace_split

        graph = trace_split(wrapper, examples, None)
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
                if existing[name].shape != buffer.shape or existing[name].dtype != buffer.dtype:
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
        for launch in plan.launches:
            self.plan.launches.append(
                replace(
                    launch,
                    node_id=f"{prefix}.{launch.node_id}",
                    arg_names=tuple(names[n] for n in launch.arg_names),
                    zero_outputs=tuple(names[n] for n in launch.zero_outputs),
                    zero_prologues=tuple(names[n] for n in launch.zero_prologues),
                    writes=tuple(names[n] for n in launch.writes),
                )
            )


def export_model(model, destination, *, context_length=MAX_CONTEXT, eos_ids=(), provenance=None):
    """Compile and bundle a dense Qwen3 checkpoint; no Python operation is needed after export."""
    import torch

    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    validate_model(model, context_length)
    cfg = model.config
    if any(not 0 <= token < cfg.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    cache = PlanTemplateCache()
    step = _Step()
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    source = (
        "\n".join(
            f"#define {key} {value}"
            for key, value in {
                "HIDDEN": h,
                "HEADS": heads,
                "KV_HEADS": kv,
                "HEAD_DIM": d,
                "VOCAB": vocab,
                "SCALE": f"{d**-0.5:.17g}f",
            }.items()
        )
        + "\n"
        + SOURCE
    )
    step.buffer("prompt", (context_length,), I64, "input")
    step.buffer("prompt_length", (1,), I64, "input")
    step.buffer("position", (1,), I64, "input")
    step.buffer("next_token", (1,), I64, "output")
    step.buffer("logits", (1, vocab), F16, "output")
    step.buffer("embedding", (vocab, h), role="constant", data=model.model.embed_tokens.weight.detach().numpy())
    with torch.no_grad():
        cosine, sine = model.model.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float16), torch.arange(context_length).reshape(1, -1))
    step.buffer("cosine", (context_length, d), role="constant", data=cosine.numpy())
    step.buffer("sine", (context_length, d), role="constant", data=sine.numpy())
    hidden = step.buffer("hidden0", (1, h))
    step.launch(
        "native_embed",
        ["prompt", "prompt_length", "position", "next_token", "embedding", hidden],
        source,
        writes=[hidden],
        blocks=(h + CUDA_THREADS - 1) // CUDA_THREADS,
    )
    example = torch.zeros(1, h, dtype=torch.float16)
    for index, layer in enumerate(model.model.layers):
        pre, post = build_attention_split_wrapper(layer)
        names = [step.buffer(f"layer{index}.{name}", (1, width * d)) for name, width in (("q", heads), ("k", kv), ("v", kv))]
        step.compiled(f"pre{index}", pre, (example,), [hidden], names, cache)
        rotated = step.buffer(f"layer{index}.rotated", (heads * d,))
        keys = step.buffer(f"layer{index}.keys", (context_length, kv, d), role="output")
        values = step.buffer(f"layer{index}.values", (context_length, kv, d), role="output")
        step.launch(
            "native_rope_cache",
            [*names, "cosine", "sine", "position", rotated, keys, values],
            source,
            writes=[rotated, keys, values],
            blocks=(heads * d + CUDA_THREADS - 1) // CUDA_THREADS,
        )
        attention = step.buffer(f"layer{index}.attention", (1, heads * d))
        step.launch(
            "native_attention",
            [rotated, keys, values, "position", attention],
            source,
            writes=[attention],
            blocks=heads,
            shared=context_length * 4,
        )
        output = step.buffer(f"hidden{index + 1}", (1, h))
        step.compiled(f"post{index}", post, (torch.zeros(1, heads * d, dtype=torch.float16), example), [attention, hidden], [output], cache)
        hidden = output
    head = torch.nn.Sequential(model.model.norm, model.lm_head)
    step.compiled("head", head, (example,), [hidden], ["logits"], cache)
    step.launch("native_greedy", ["logits", "next_token"], source, writes=["next_token"], threads=1)
    return save_executable(
        destination,
        {"decode": step.plan},
        bindings={"decode": step.bindings},
        key={
            "generation": {"version": GENERATION_VERSION, "context_length": context_length, "vocab_size": vocab, "eos_ids": list(eos_ids)}
        },
        provenance=provenance,
    )
