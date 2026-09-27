"""Export dense FP16 Qwen3 as static decode and chunked prefill programs."""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec, plan_from_graph
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F16, F32, F64, I64, U32, U64
from emmy.serving.native.kernels import SOURCE

MAX_CONTEXT = 4096
CUDA_THREADS = 128
GENERATION_VERSION = 3
PREFILL_SIZE = 16
SAMPLING_BINS = 65536


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
    def __init__(self, prefill=False):
        self.plan = ExecutionPlan(
            "cuda",
            ["prompt", "prompt_length", "position"] + ([] if prefill else ["sampling", "seed"]),
            [] if prefill else ["logits", "next_token"],
            [],
            {},
            {},
            [],
            {},
        )
        self.bindings = {}

    def buffer(self, name, shape, dtype=F16, role="scratch", data=None):
        self.plan.buffers.append(BufferSpec(name, tuple(Dim(n) for n in shape), dtype, role))
        if data is not None:
            self.bindings[name] = np.ascontiguousarray(data).tobytes()
        return name

    def launch(self, kernel, args, source, *, writes, blocks=1, rows=1, shared=0, threads=CUDA_THREADS, zero_outputs=()):
        self.plan.kernels[kernel] = KernelSpec(source=source)
        self.plan.launches.append(
            LaunchSpec(
                kernel,
                kernel,
                tuple(args),
                ((blocks,), (rows,), (1,)),
                ((threads,), (1,), (1,)),
                shared,
                tuple(zero_outputs),
                writes=tuple(writes),
            )
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


def _program(model, context_length, rows, cache):
    import torch

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    cfg = model.config
    step = _Step(prefill=rows > 1)
    h, heads, kv, d, vocab = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.vocab_size
    source = (
        "\n".join(
            f"#define {key} {value}"
            for key, value in {
                "PREFILL": int(rows > 1),
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
    if rows == 1:
        step.buffer("sampling", (2,), F64, "input")
        step.buffer("seed", (1,), U64, "input")
        step.buffer("sampling_histogram", (SAMPLING_BINS,), U32)
        step.buffer("logits", (1, vocab), F16, "output")
    step.buffer("embedding", (vocab, h), role="constant", data=model.model.embed_tokens.weight.detach().numpy())
    with torch.no_grad():
        cosine, sine = model.model.rotary_emb(torch.zeros(1, 1, h, dtype=torch.float32), torch.arange(context_length).reshape(1, -1))
    step.buffer("cosine", (context_length, d), F32, role="constant", data=cosine.numpy())
    step.buffer("sine", (context_length, d), F32, role="constant", data=sine.numpy())
    hidden = step.buffer("hidden0", (rows, h), F32)
    step.launch(
        "native_embed",
        ["prompt", "prompt_length", "position", "next_token", "embedding", hidden],
        source,
        writes=[hidden],
        blocks=(h + CUDA_THREADS - 1) // CUDA_THREADS,
        rows=rows,
    )
    example = torch.zeros(rows, h, dtype=torch.float32)
    for index, layer in enumerate(model.model.layers):
        pre, post = build_attention_split_wrapper(layer, float32_residual=True)
        names = [step.buffer(f"layer{index}.{name}", (rows, width * d)) for name, width in (("q", heads), ("k", kv), ("v", kv))]
        step.compiled(f"pre{index}", pre, (example,), [hidden], names, cache)
        # The last layer only needs KV; its rotary query output has no scratch consumer.
        last_prefill = rows > 1 and index + 1 == len(model.model.layers)
        rotated = step.buffer(f"layer{index}.rotated", (rows, heads * d), role="output" if last_prefill else "scratch")
        keys = step.buffer(f"layer{index}.keys", (context_length, kv, d), role="output")
        values = step.buffer(f"layer{index}.values", (context_length, kv, d), role="output")
        step.launch(
            "native_rope_cache",
            [*names, "cosine", "sine", "position", "prompt_length", rotated, keys, values],
            source,
            writes=[rotated, keys, values],
            blocks=(heads * d + CUDA_THREADS - 1) // CUDA_THREADS,
            rows=rows,
        )
        if last_prefill:
            continue
        attention = step.buffer(f"layer{index}.attention", (rows, heads * d))
        step.launch(
            "native_attention",
            [rotated, keys, values, "position", "prompt_length", attention],
            source,
            writes=[attention],
            blocks=heads,
            rows=rows,
            shared=context_length * 4,
        )
        output = step.buffer(f"hidden{index + 1}", (rows, h), F32)
        step.compiled(
            f"post{index}", post, (torch.zeros(rows, heads * d, dtype=torch.float16), example), [attention, hidden], [output], cache
        )
        hidden = output

    if rows > 1:
        return step

    class Head(torch.nn.Sequential):
        def forward(self, hidden):
            return self[1](self[0](hidden).to(self[1].weight.dtype))

    head = Head(model.model.norm, model.lm_head)
    step.compiled("head", head, (example,), [hidden], ["logits"], cache)
    step.launch(
        "native_histogram",
        ["logits", "sampling", "position", "prompt_length", "sampling_histogram"],
        source,
        writes=["sampling_histogram"],
        blocks=(vocab + CUDA_THREADS - 1) // CUDA_THREADS,
        zero_outputs=["sampling_histogram"],
    )
    step.launch(
        "native_sample",
        ["logits", "sampling_histogram", "sampling", "seed", "position", "prompt_length", "next_token"],
        source,
        writes=["next_token"],
        shared=CUDA_THREADS * 4,
    )
    return step


def export_model(model, destination, *, context_length=MAX_CONTEXT, eos_ids=(), provenance=None, prefill_size=None):
    """Bundle one-token decode and fixed-width prefill; preparation owns every model operation."""
    from emmy.compiler.backend.plan_cache import PlanTemplateCache

    validate_model(model, context_length)
    prefill_size = PREFILL_SIZE if prefill_size is None else prefill_size
    if type(prefill_size) is not int or not 1 <= prefill_size <= MAX_CONTEXT:
        raise ValueError("prefill size must be within supported context capacity")
    if any(not 0 <= token < model.config.vocab_size for token in eos_ids):
        raise ValueError("EOS token outside vocabulary")
    prefill_size = min(prefill_size, context_length)
    cache = PlanTemplateCache()
    programs = {"decode": _program(model, context_length, 1, cache)}
    if prefill_size > 1:
        programs["prefill"] = _program(model, context_length, prefill_size, cache)
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
            }
        },
        provenance=provenance,
    )
