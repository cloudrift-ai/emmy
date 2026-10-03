"""The serving-generation lane's model table, its golden, and the scope the tests compile under.

These tests exercise vLLM integration and materialization — whether the runner stitches, dispatches
and allocates correctly — not which schedule a cold search would pick. Cold deploy is the prior's
contract and has its own tests, so a runner built here compiles under :func:`evidence_scope`: the
checked-in golden is the compile's ONLY evidence, strictly. Two things follow.

* **It is fast.** A cold boot re-derives every pick by descent sampling — ~1M ``schedule()`` calls
  per model — which is ~90% of what this lane used to cost. A replay skips all of it.
* **It is the same everywhere.** A cold pick resolves through the machine-local tune DB and online
  prior, so what these tests compiled depended on the box they ran on. A golden does not.

Strict evidence is what keeps the golden honest: a fork no row decides is an ``EvidenceError``
naming the kernel, so an incomplete or stale golden fails loudly here instead of silently
reintroducing the search. Regenerate with ``python -m tests.serving.regen``.

:data:`RUNNERS` is the one place a shape is spelled. The tests read it and so does the regen, so
the golden covers exactly the runners the tests build — a new runner shape is one entry plus a
regen, and strict evidence reports it if you forget the regen.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path

GOLDEN = Path(__file__).parent / "goldens" / "serving.golden.json"


def per_expert(run_expert):
    """``combine_routed_experts``'s batch callback from a per-expert ``run_expert(e, rows)``: each hit
    expert runs on its own block of rows, and the outputs come back concatenated in the same order."""
    import torch

    def run(rows, counts):
        outputs, start = [], 0
        for e, count in enumerate(counts):
            if count:
                outputs.append(run_expert(e, rows[start : start + count]))
            start += count
        return torch.cat(outputs)

    return run


def qwen3_config(layers: int):
    """The lane's dense model: a 1-head-group Qwen3 small enough to compile in seconds."""
    import transformers

    return transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        use_sliding_window=False,
    )


def olmoe_config(layers: int, *, experts: int = 8, max_position_embeddings: int = 64):
    """The lane's MoE model — the routed/expert paths the dense one cannot reach."""
    import transformers

    return transformers.OlmoeConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=32,
        num_hidden_layers=layers,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=experts,
        num_experts_per_tok=2,
        norm_topk_prob=False,
        max_position_embeddings=max_position_embeddings,
    )


def qwen3_model(layers: int):
    import torch
    import transformers

    torch.manual_seed(0)
    return transformers.Qwen3ForCausalLM(qwen3_config(layers)).eval()


#: The tiny Qwen3.5 text geometry: a Gated DeltaNet layer beside a full-attention layer, with the
#: family's partial rotary embedding and fused query/gate projection.
QWEN3_5_TINY = dict(
    vocab_size=64,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    linear_key_head_dim=16,
    linear_value_head_dim=16,
    linear_num_key_heads=2,
    linear_num_value_heads=4,
    linear_conv_kernel_dim=4,
    max_position_embeddings=64,
    layer_types=["linear_attention", "full_attention"],
)


def qwen3_5_model(**overrides):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**(QWEN3_5_TINY | overrides))).eval()


#: The re-minted width of layer 1's experts in the mixed-shape model — EXL3's per-layer bit
#: allocation is the live case (K varies per layer, so the codes shape does).
WIDE_EXPERT = 48


def olmoe_model(
    layers: int,
    *,
    experts: int = 8,
    max_position_embeddings: int = 64,
    flat_router: bool = False,
    wide_layer1: bool = False,
):
    import torch
    from transformers.models.olmoe.modeling_olmoe import OlmoeForCausalLM

    config = olmoe_config(layers, experts=experts, max_position_embeddings=max_position_embeddings)
    torch.manual_seed(0)
    model = OlmoeForCausalLM(config).eval()
    if flat_router:
        # A router scoring every expert alike routes every token to the same one — the degenerate
        # case the profiling run's identical dummy rows produce whatever the weights are.
        for layer in model.model.layers:
            layer.mlp.gate.weight.data.zero_()
    if wide_layer1:
        # Layer 1's experts at a DIFFERENT width, so the runner must build one program set per
        # expert shape rather than sharing one across layers.
        e, h = config.num_experts, config.hidden_size
        experts1 = model.model.layers[1].mlp.experts
        experts1.gate_up_proj = torch.nn.Parameter(torch.randn(e, 2 * WIDE_EXPERT, h) * 0.05)
        experts1.down_proj = torch.nn.Parameter(torch.randn(e, h, WIDE_EXPERT) * 0.05)
    return model


def llama_model(layers: int = 1):
    """The lane's untied-embedding model — the rider split's shape, and the one architecture here
    whose lm_head is a weight of its own."""
    import torch
    from transformers import LlamaConfig
    from transformers.models.llama.modeling_llama import LlamaForCausalLM

    torch.manual_seed(0)
    return LlamaForCausalLM(
        LlamaConfig(
            vocab_size=64,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            tie_word_embeddings=False,
        )
    ).eval()


def qwen3_5_gdn_model(layers: int, *, dtype: str = "float32"):
    """The lane's GDN model: a Qwen3.5 whose layers are all linear attention (gated DeltaNet), so a runner test
    needs no attention stitch between its layer programs."""
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    from tests.compiler.trace.test_huggingface import _QWEN3_5_TINY

    tiny = {**_QWEN3_5_TINY, "num_hidden_layers": layers, "layer_types": ["linear_attention"] * layers}
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**tiny)).eval().to(getattr(torch, dtype))


#: Every runner shape this lane builds: ``id -> (model factory, from_model kwargs)``. The golden
#: covers exactly these, so a test asks for one by id rather than spelling a config of its own.
RUNNERS: dict[str, tuple] = {
    "qwen3.l3": (lambda: qwen3_model(3), {"dtype_str": "float32"}),
    "qwen3.l3.b16": (lambda: qwen3_model(3), {"dtype_str": "float32", "decode_bucket": 16}),
    "qwen3.l2.b16": (lambda: qwen3_model(2), {"dtype_str": "float32", "decode_bucket": 16}),
    "qwen3.l1.b16": (lambda: qwen3_model(1), {"dtype_str": "float32", "decode_bucket": 16}),
    "olmoe.l3.b16": (lambda: olmoe_model(3), {"dtype_str": "float32", "decode_bucket": 16, "max_tokens": 64}),
    "olmoe.l2.b16": (lambda: olmoe_model(2), {"dtype_str": "float32", "decode_bucket": 16, "max_tokens": 64}),
    "olmoe.l2.e4.b8.wide": (
        lambda: olmoe_model(2, experts=4, wide_layer1=True),
        {"dtype_str": "float32", "decode_bucket": 8, "max_tokens": 64},
    ),
    "olmoe.l1.b4": (lambda: olmoe_model(1), {"dtype_str": "float32", "decode_bucket": 4, "max_tokens": 64}),
    "llama.l1.rider": (
        llama_model,
        {"dtype_str": "float32", "decode_bucket": 16, "max_tokens": 64, "prefill_bucket": 32},
    ),
    "olmoe.l1.rider": (
        lambda: olmoe_model(1, max_position_embeddings=128, flat_router=True),
        {"dtype_str": "float32", "decode_bucket": 16, "max_tokens": 512, "prefill_bucket": 512},
    ),
    "qwen3_5.gdn.l2": (
        lambda: qwen3_5_gdn_model(2),
        {"dtype_str": "float32", "decode_bucket": 4, "max_tokens": 64, "prefill_bucket": 16},
    ),
    "qwen3_5.gdn.l2.bf16": (
        lambda: qwen3_5_gdn_model(2, dtype="bfloat16"),
        {"dtype_str": "bfloat16", "decode_bucket": 4, "max_tokens": 64, "prefill_bucket": 16},
    ),
}


def gemma3_block_config():
    """Tiny Gemma-3/4 text config, layer 0 forced global (``sliding_window_pattern=1``) — the
    4-norm carve target (extra pre/post-feedforward norms)."""
    import transformers

    return transformers.Gemma3TextConfig(
        vocab_size=64,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=16,
        sliding_window_pattern=1,
    )


def _block(kind: str):
    """One decoder layer plus its config — the attention-split wrappers' input."""
    import torch
    import transformers

    if kind == "qwen3":
        config = qwen3_config(1)
        torch.manual_seed(0)
        return config, transformers.Qwen3ForCausalLM(config).eval().model.layers[0]
    config = gemma3_block_config()
    torch.manual_seed(0)
    return config, transformers.Gemma3ForCausalLM(config).eval().model.layers[0]


#: Every attention-split wrapper this lane compiles: ``id -> (block kind, "pre" | "post")``. These
#: trace with a SYMBOLIC token axis, so their programs are their own — not the runner shapes'.
WRAPPERS: dict[str, tuple[str, str]] = {
    "qwen3.pre": ("qwen3", "pre"),
    "qwen3.post": ("qwen3", "post"),
    "gemma3.post": ("gemma3", "post"),
}


def wrapper_case(case_id: str):
    """``(wrapper, example_args, argnames, config)`` for one :data:`WRAPPERS` entry.

    The example widths carry the trace; the token axis is symbolic, so a case compiled here runs at
    any count. One spelling for the tests and the regen, the same rule :data:`RUNNERS` follows.
    """
    import torch

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper

    kind, half = WRAPPERS[case_id]
    config, block = _block(kind)
    pre, post = build_attention_split_wrapper(block)
    h = config.hidden_size
    if half == "pre":
        return pre, [torch.randn(8, h)], ["hidden"], config
    width = config.num_attention_heads * (config.head_dim or h // config.num_attention_heads)
    return post, [torch.randn(8, width), torch.randn(8, h)], ["attn_out", "residual"], config


def wrapper_graph(case_id: str):
    """The traced graph a wrapper case compiles — what the regen records for it."""
    from emmy.compiler.trace.dynamic import build_torch_dynamic_shapes, parse_position_specs
    from emmy.compiler.trace.torch import trace_module

    wrapper, args, argnames, _ = wrapper_case(case_id)
    specs = [f"num_tokens@{name}:0" for name in argnames]
    return trace_module(wrapper, tuple(args), dynamic_shapes=build_torch_dynamic_shapes(parse_position_specs(specs)))


def golden_document():
    """The lane's golden as the compile's evidence: every row standing in as a measured one — these authored scalar
    schedules carry no device measurements — and the file scoped to the live card, so strict replay can validate
    them there whatever card traced them."""
    from dataclasses import replace

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements

    document = GoldenFile.load(GOLDEN)
    stand_in = Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="serving-lane")
    rows = [replace(row, measurements=row.measurements or stand_in) for row in document.rows]
    return replace(document, compute_cap=tuple(Context.probe().compute_capability), gpu_name=None, rows=rows)


def build(runner_id: str, *, model=None, plan_cache=None, **overrides):
    """Build one :data:`RUNNERS` shape under the lane's golden.

    ``plan_cache`` is the session's when a caller has one; leave it ``None`` where a boot must be
    a real one — the pack round-trip asserts that its FIRST boot compiles and its second loads the
    pack, and a warm template cache would decide the first boot for it. ``model`` reuses a module
    the caller already built, for the tests that boot the same weights twice.
    """
    from emmy.serving.gen_runner import EmmyGenRunner

    make_model, kwargs = RUNNERS[runner_id]
    with evidence_scope():
        return EmmyGenRunner.from_model(make_model() if model is None else model, plan_cache=plan_cache, **{**kwargs, **overrides})


@contextmanager
def evidence_scope():
    """The lane's golden as the compile's only evidence, strictly (:func:`golden.sole_evidence`)."""
    from emmy.compiler.pipeline.search.golden import sole_evidence
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    document = golden_document()
    with pinned_knobs(document.shared_regime()), sole_evidence([document]):
        yield
