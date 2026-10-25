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

GOLDENS = Path(__file__).parent / "goldens"


def golden_path(compute_capability: tuple[int, int] | None = None) -> Path:
    """The lane's golden for a compute capability (the live card's by default): ``serving_sm<NN>.golden.json``.

    Rows are realized schedules, and a schedule one architecture offers another may not (a TMA stage on a card
    with no TMA), so each capability the lane runs on keeps its own file, written by the regen on that card.
    """
    if compute_capability is None:
        from emmy.compiler.context import Context

        compute_capability = tuple(Context.probe().compute_capability)
    major, minor = compute_capability
    return GOLDENS / f"serving_sm{major}{minor}.golden.json"


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

# Qwen3.8-Flash-Next (`qwen4_exp`): three Gated DeltaNet layers, then sparse attention; layer 2 (one-indexed) reads
# the n-gram embedding. Tiny but valid under the config's own checks (rotary width fits the indexer head).
QWEN4_EXP_TINY = dict(
    vocab_size=64, hidden_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, head_dim=16,
    max_position_embeddings=64, linear_key_head_dim=8, linear_value_head_dim=8, linear_num_key_heads=2,
    linear_num_value_heads=4, moe_intermediate_size=16, shared_expert_intermediate_size=16, num_experts_per_tok=2,
    num_experts=4, hc_count=2, hc_lowrank=4, ple_layer_ids=[2], ple_embed_dim=16, heads_per_ngram=2,
    ngram_vocab_size_base=101, make_ngram_vocab_size_divisible_by=8, indexer_n_heads=2, indexer_kv_heads=1,
    indexer_head_dim=8, indexer_budget=8, indexer_compress_ratio=4, eos_token_id=1,
    layer_types=["linear_attention"] * 3 + ["full_attention"],
    rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "partial_rotary_factor": 0.5},
)  # fmt: skip


def save_tiny_qwen4_exp(path, layer_types) -> None:
    """A bfloat16 Qwen3.8-Flash-Next checkpoint at ``path`` with the published sparse-attention and Gated DeltaNet
    geometry, everything else shrunk, for the 1Cat fork's kernels on the card. Its layers are ``layer_types``, written
    the way the published file spells them: Transformers saves its internal ``indexed_attention``, which the fork's
    config does not read. Layer 2 reads the n-gram PLE, as in the published model. Transformers writes its table as
    BF16 row shards; they are stored the published way, FP8 with one scale, the only form the fork takes on a card
    older than Ampere."""
    import json

    import torch
    import transformers
    from safetensors.torch import load_file, save_file

    text = dict(
        hidden_size=512, hc_count=4, hc_lowrank=64, num_attention_heads=24, num_key_value_heads=2, head_dim=256,
        indexer_n_heads=4, indexer_kv_heads=1, indexer_head_dim=128, indexer_budget=2048, indexer_compress_ratio=4,
        linear_num_key_heads=4, linear_num_value_heads=12, linear_key_head_dim=128, linear_value_head_dim=128,
        linear_conv_kernel_dim=4, output_gate_type="sigmoid",
        partial_rotary_factor=0.25, rope_parameters={"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                                     "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                     "rope_type": "default"},
        num_experts=4, num_experts_per_tok=2, moe_intermediate_size=128, shared_expert_intermediate_size=128,
        vocab_size=1024, max_position_embeddings=4096, mtp_num_hidden_layers=0, ple_layer_ids=[2],
        ngram_vocab_size_base=1000, split_ngram_parts=4, eos_token_id=1,
        num_hidden_layers=len(layer_types), layer_types=list(layer_types), tie_word_embeddings=False,
    )  # fmt: skip
    vision = dict(depth=1, hidden_size=64, intermediate_size=128, num_heads=2, out_hidden_size=text["hidden_size"])
    config = transformers.Qwen4ExpConfig(text_config=text, vision_config=vision, tie_word_embeddings=False)
    torch.manual_seed(0)
    transformers.Qwen4ExpForConditionalGeneration(config).eval().to(torch.bfloat16).save_pretrained(path)
    config_path = Path(path) / "config.json"
    saved = json.loads(config_path.read_text())
    saved["text_config"]["layer_types"] = text["layer_types"]
    saved["text_config"]["ple_embedding_dtype"] = "float8_e4m3fn"
    config_path.write_text(json.dumps(saved))
    weights_path = Path(path) / "model.safetensors"
    tensors = load_file(weights_path)
    site = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding."
    shards = [key for key in tensors if key.startswith(f"{site}shard_")]
    scale = max(tensors[key].float().abs().max() for key in shards) / torch.finfo(torch.float8_e4m3fn).max
    for key in shards:
        tensors[key] = (tensors[key].float() / scale).to(torch.float8_e4m3fn)
    tensors[f"{site}weight_scale"] = scale.reshape(1).to(torch.bfloat16)
    save_file(tensors, weights_path, metadata={"format": "pt"})


def qwen3_5_model(**overrides):
    import torch
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForCausalLM

    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(Qwen3_5TextConfig(**(QWEN3_5_TINY | overrides))).eval()


def qwen3_5_moe_model(**overrides):
    """The tiny Qwen3.5 geometry with routed experts and the gated shared expert in every layer. Its
    router weights are ten times the initializer's, which widens the gaps between router logits, so
    two implementations' rounding does not flip a pick."""
    import torch
    from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import Qwen3_5MoeTextConfig
    from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeForCausalLM

    torch.manual_seed(0)
    experts = dict(num_experts=8, num_experts_per_tok=2, moe_intermediate_size=32, shared_expert_intermediate_size=32)
    model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**(QWEN3_5_TINY | experts | overrides))).eval()
    with torch.no_grad():
        for layer in model.model.layers:
            layer.mlp.gate.weight.mul_(10)
    return model


def add_mtp_head(path, *, repeats=False):
    """Write a random one-layer multi-token-prediction head into the Qwen3.5 checkpoint at ``path``, text-only or
    vision-language, under the names a Qwen3.5 checkpoint gives it, in the dtype of its other tensors. With ``repeats``
    the head drafts the target's own pick again: its layer adds nothing (zero output projections), its prologue passes
    the target's normalized row on and its norms are plain, so every draft a target repeating one token is checked
    against stands."""
    import json

    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoConfig, AutoModel

    config_path = path / "config.json"
    raw = json.loads(config_path.read_text())
    text = raw.get("text_config", raw)
    weights = path / "model.safetensors"
    stored = load_file(weights)
    dtype = next(iter(stored.values())).dtype
    config = AutoConfig.for_model(**(text | {"num_hidden_layers": 1, "layer_types": ["full_attention"]}))
    torch.manual_seed(1)
    head = AutoModel.from_config(config).to(dtype)
    hidden = config.hidden_size
    tensors = {f"mtp.layers.0.{name}": value for name, value in head.layers[0].state_dict().items()}
    tensors["mtp.norm.weight"] = torch.randn(hidden, dtype=dtype) * 0.1
    tensors["mtp.pre_fc_norm_embedding.weight"] = torch.randn(hidden, dtype=dtype) * 0.1
    tensors["mtp.pre_fc_norm_hidden.weight"] = torch.randn(hidden, dtype=dtype) * 0.1
    tensors["mtp.fc.weight"] = torch.randn(hidden, 2 * hidden, dtype=dtype) * (2 * hidden) ** -0.5
    if repeats:
        for name, value in tensors.items():
            if name.endswith(("o_proj.weight", "down_proj.weight")) or "norm" in name:
                tensors[name] = torch.zeros_like(value)
        tensors["mtp.fc.weight"] = torch.cat([torch.zeros(hidden, hidden), torch.eye(hidden)], dim=1).to(dtype)
    save_file(stored | {name: value.contiguous() for name, value in tensors.items()}, weights, metadata={"format": "pt"})
    text["mtp_num_hidden_layers"] = 1
    config_path.write_text(json.dumps(raw))


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

    tiny = {**QWEN3_5_TINY, "num_hidden_layers": layers, "layer_types": ["linear_attention"] * layers}
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


def on_volta() -> bool:
    """Whether the live card is a V100 (sm_70) — the condition of the lane's two V100-only skips."""
    import torch

    return torch.cuda.is_available() and torch.cuda.get_device_capability() == (7, 0)


def skip_without_gdn_verify_counts() -> None:
    """Skip a test of speculative decoding on GDN layers on vLLM < 0.25. There no builder class receives a verify
    step's accepted-token counts, so the GDN serving check refuses speculative decoding."""
    import pytest

    from emmy.serving import vllm_model_gen

    if vllm_model_gen._CountingLinearBuilder is vllm_model_gen.LinearAttentionMetadataBuilder:
        pytest.skip("this vLLM passes no GDN verify counts (< 0.25), so speculative decoding on GDN layers is refused")


def assert_same_schedule(actual, expected, context: str = "") -> None:
    """Two runs of one schedule agree to a few units in the last place of their dtype. Not bit for bit: a
    cross-CTA ``atomicAdd`` split (``g<n>a``, which the V100 golden picks on several of these shapes) adds its
    partials in arrival order, and that order changes from run to run. A different schedule, or a wrong one,
    differs by far more than 64 units in the last place."""
    import numpy as np

    expected = np.asarray(expected)
    eps = np.finfo(expected.dtype).eps
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=64 * eps, atol=64 * eps, err_msg=context)


def golden_document():
    """The lane's golden as the compile's evidence: every row standing in as a measured one — these authored scalar
    schedules carry no device measurements — and the file scoped to the live card, so strict replay can validate
    them there whatever card traced them."""
    from dataclasses import replace

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements

    document = GoldenFile.load(golden_path())
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
    """The lane's golden as the compile's only evidence, strictly (:func:`golden.sole_evidence`), picked from a tune
    DB of this process's own (:func:`_lane_tune_db`)."""
    from emmy.compiler.pipeline.search.golden import sole_evidence
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    document = golden_document()
    with pinned_knobs(document.shared_regime()), sole_evidence([document]), _lane_tune_db():
        yield


#: This process's lane tune DB directory, made on first use and removed at exit.
_LANE_TUNE_DB: list = []


@contextmanager
def _lane_tune_db():
    """``EMMY_TUNE_DB`` pointed at a file of this process's own. The session's tune DB is one file for every xdist
    worker (``tests/conftest.py``), and a compile importing its golden scope (the set of golden files it reads
    evidence from) deletes the other scope's golden rows there (``golden.evidence.evidence_db``). Without a file of
    its own, a lane compile in another worker builds its evidence from that scope's rows or none, and strict evidence
    refuses forks the lane's golden decides."""
    import os
    import tempfile

    from emmy import config

    if not _LANE_TUNE_DB:
        _LANE_TUNE_DB.append(tempfile.TemporaryDirectory(prefix="emmy_lane_tune_db_"))
    previous = os.environ.get(config.TUNE_DB)
    os.environ[config.TUNE_DB] = os.path.join(_LANE_TUNE_DB[0].name, "autotune.db")
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop(config.TUNE_DB, None)
        else:
            os.environ[config.TUNE_DB] = previous


def fp8_block_checkpoint(model, path, *, block: int = 32) -> None:
    """Write ``model`` as a dynamic block-FP8 checkpoint — the form Qwen ships: e4m3 codes beside
    one ``weight_scale_inv`` per block, no activation scale — and leave the model holding the
    values the checkpoint stores, so it is the eager reference of what a coded program computes.
    Every linear the block tiles but the output head is quantized, so the small gates stay as Qwen
    publishes them; routed experts are stored one module per expert, as Qwen publishes them too."""
    import json

    import torch
    from safetensors.torch import save_file

    from emmy.compiler.loader.quant import decode_f8, dequantize
    from emmy.compiler.loader.synthesize import _quantize_fp8_block

    path = Path(path)
    model.config.save_pretrained(path)
    document = json.loads((path / "config.json").read_text())
    document["dtype"] = str(model.dtype).removeprefix("torch.")  # as ``save_pretrained`` declares it
    document["quantization_config"] = {
        "quant_method": "fp8",
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "weight_block_size": [block, block],
    }
    (path / "config.json").write_text(json.dumps(document, indent=1))
    tensors = {}

    def quantize(name, weight):
        bits, scale = _quantize_fp8_block(weight.detach().float().numpy(), block)
        tensors[f"{name}.weight"] = torch.from_numpy(bits).view(torch.float8_e4m3fn)
        tensors[f"{name}.weight_scale_inv"] = torch.from_numpy(scale)
        return torch.from_numpy(dequantize(decode_f8(bits, "f8e4m3"), scale)).to(weight.dtype)

    stacked = set()
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and name != "lm_head" and not any(d % block for d in module.weight.shape):
            module.weight.data = quantize(name, module.weight)
        if isinstance(getattr(module, "gate_up_proj", None), torch.nn.Parameter):  # E-stacked routed experts
            half = module.gate_up_proj.shape[1] // 2
            for e, (gate_up, down) in enumerate(zip(module.gate_up_proj.data, module.down_proj.data, strict=True)):
                gate, up = gate_up.split(half)
                gate_up.copy_(torch.cat([quantize(f"{name}.{e}.gate_proj", gate), quantize(f"{name}.{e}.up_proj", up)]))
                down.copy_(quantize(f"{name}.{e}.down_proj", down))
            stacked |= {f"{name}.gate_up_proj", f"{name}.down_proj"}
    for key, tensor in model.state_dict().items():
        if key not in stacked:
            tensors.setdefault(key, tensor.contiguous())
    if model.config.tie_word_embeddings:
        del tensors["lm_head.weight"]  # one storage; safetensors refuses to write it twice
    save_file(tensors, str(path / "model.safetensors"))
