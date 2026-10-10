"""The serving twins' coded arm spells checkpoint siblings through generic tensor algebra.

These tests drive the spelling stage directly off a synthetic storage listing (no network, no
checkpoint, no transformers): the tracing stage is already covered by the drift gate, and what is
new here is the pairing — which traced module gets which checkpoint entry, and how the per-tensor
rate allocation multiplies the twins.

The NVFP4 arm is the exception, and writes a real (tiny) checkpoint: that format has no weight-free
description of itself, so its spelling reads stored shapes and calibrated activation scales off the
directory rather than a listing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from emmy.compiler.dtype import F16
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.frontend.ir import LinearOp
from emmy.compiler.loader.safetensors import split_revision
from emmy.serving.twins import (
    _attention_query_layout,
    _profile_layers,
    _spell_coded_twins,
    _spell_expert_twins,
)


def _entry(base: str, n: int, k: int, bits: int) -> dict:
    """One ``tensor_storage`` entry, shaped exactly as exllamav3 writes it."""
    n_pad, k_pad = -(-n // 128) * 128, -(-k // 128) * 128
    return {
        "quant_format": "exl3",
        "bits_per_weight": bits,
        "stored_tensors": {
            f"{base}.trellis": {"shape": [k_pad // 16, n_pad // 16, 16 * bits]},
            f"{base}.suh": {"shape": [k_pad]},
            f"{base}.svh": {"shape": [n_pad]},
        },
    }


def _twin(mods: dict[str, tuple[int, int]], m: int = 1) -> Graph:
    """A twin-shaped graph: one activation input and one ``F.linear`` per traced module, with the
    wrapper-relative constant paths a split-wrapper trace produces (``q_proj.weight``)."""
    g = Graph()
    hidden = next(iter(mods.values()))[1]
    g.add_node(InputOp(), [], Tensor("x", (m, hidden), "f16"), node_id="x")
    for mod, (n, k) in mods.items():
        nid = mod.replace(".", "_")
        g.add_node(
            ConstantOp(name=nid, source_path=f"{mod}.weight", source_shape=(n, k), source_dtype="f16"), [], Tensor(nid, (n, k), "f16")
        )
        g.add_node(LinearOp(), ["x", nid], Tensor(f"y_{nid}", (m, n), "f16"), node_id=f"y_{nid}")
    g.inputs, g.outputs = ["x"], [f"y_{mod.replace('.', '_')}" for mod in mods]
    return g


def _decodes(g: Graph) -> dict[str, int]:
    """``{codes source path: k_bits}`` for every coded contraction the graph spells."""
    out = {}
    for nd in g.nodes.values():
        if isinstance(nd.op, ConstantOp) and nd.op.source_path and nd.op.source_path.endswith(".trellis"):
            out[nd.op.source_path] = int(nd.output.shape[2].as_static()) // 16
    return out


def test_coded_twin_spells_the_deployed_contraction():
    """Each traced weight is matched to its checkpoint module by dotted suffix within a layer
    (a twin traces ``q_proj.weight`` where the checkpoint says ``…self_attn.q_proj``) and spelled
    by the same generic speller used by the loader."""
    storage = {
        "model.layers.0.self_attn.q_proj": _entry("model.layers.0.self_attn.q_proj", 12288, 4096, 4),
        "model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2),
    }
    twins = _spell_coded_twins({"pre1": _twin({"q_proj": (12288, 4096), "mlp.gate_proj": (10944, 4096)})}, storage)
    assert list(twins) == ["pre1@b2-4"]
    assert _decodes(twins["pre1@b2-4"]) == {
        "model.layers.0.self_attn.q_proj.trellis": 4,
        "model.layers.0.mlp.gate_proj.trellis": 2,
    }


def test_one_twin_per_rate_profile():
    """An "optimized" rung allocates bits per tensor, and the rate is part of the ShapeKey, so one
    traced layer does not represent the trunk: every distinct allocation is emitted. (The pinned
    GLM-4.5-Air 2.25 checkpoint really does ship q/k/v at 4 bits on 42 layers and 3 on 4.)"""
    storage = {}
    for layer, bits in ((0, 4), (1, 4), (2, 3)):
        storage[f"model.layers.{layer}.self_attn.q_proj"] = _entry(f"model.layers.{layer}.self_attn.q_proj", 12288, 4096, bits)
    twins = _spell_coded_twins({"pre1": _twin({"q_proj": (12288, 4096)})}, storage)
    assert sorted(twins) == ["pre1@b3", "pre1@b4"]  # layer 1 repeats layer 0's profile and is dropped
    assert _decodes(twins["pre1@b4"]) == {"model.layers.0.self_attn.q_proj.trellis": 4}
    assert _decodes(twins["pre1@b3"]) == {"model.layers.2.self_attn.q_proj.trellis": 3}


def test_uncoded_twin_passes_through_untouched():
    """A twin holding no coded weight keeps its original name and graph — the coded arm must not
    rename or rewrite the models that make up every other golden file."""
    graph = _twin({"q_proj": (12288, 4096)})
    twins = _spell_coded_twins({"pre1": graph}, {"model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2)})
    assert twins == {"pre1": graph}


def test_an_ambiguous_suffix_names_no_module():
    """``mlp.gate_proj`` must hit the dense MLP and never an expert's ``gate_proj``. Where the
    suffix is genuinely ambiguous the module is left uncoded — guessing spells the wrong rate,
    and an uncoded fork shows up as a GAP rather than as a wrong MATCH."""
    storage = {
        "model.layers.0.mlp.gate_proj": _entry("model.layers.0.mlp.gate_proj", 10944, 4096, 2),
        "model.layers.0.mlp.experts.0.gate_proj": _entry("model.layers.0.mlp.experts.0.gate_proj", 1408, 4096, 2),
        "model.layers.0.mlp.experts.1.gate_proj": _entry("model.layers.0.mlp.experts.1.gate_proj", 1408, 4096, 2),
    }
    dense = _spell_coded_twins({"post1": _twin({"mlp.gate_proj": (10944, 4096)})}, storage)
    assert _decodes(dense["post1@b2"]) == {"model.layers.0.mlp.gate_proj.trellis": 2}
    # A bare ``gate_proj`` matches all three names, so nothing is spelled.
    assert _spell_coded_twins({"post1": (g := _twin({"gate_proj": (1408, 4096)}))}, storage) == {"post1": g}


def test_laguna_singular_shared_expert_matches_plural_transformers_path():
    storage = {"model.layers.1.mlp.shared_expert.gate_proj": _entry("model.layers.1.mlp.shared_expert.gate_proj", 1024, 3072, 4)}
    twins = _spell_coded_twins({"post1": _twin({"mlp.shared_experts.gate_proj": (1024, 3072)})}, storage)
    assert _decodes(twins["post1@b4"]) == {"model.layers.1.mlp.shared_expert.gate_proj.trellis": 4}


def test_laguna_selects_dense_full_sparse_sliding_and_sparse_full_profiles():
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    cfg = transformers.LagunaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=3,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=64,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=16,
        num_experts=4,
        num_experts_per_tok=2,
        layer_types=["full_attention", "sliding_attention", "full_attention"],
        mlp_layer_types=["dense", "sparse", "sparse"],
        num_attention_heads_per_layer=[4, 6, 4],
        sliding_window=16,
    )
    with torch.device("meta"):
        trunk = transformers.AutoModel.from_config(cfg, dtype=torch.float16)
    profiles = _profile_layers(trunk, cfg)
    assert [(i, suffix) for i, _block, suffix in profiles] == [
        (0, "-dense-full"),
        (1, "-sparse-sliding"),
        (2, "-sparse-full"),
    ]


@pytest.mark.parametrize("quantized", [False, True])
def test_gdn_serving_capture_has_explicit_state_inputs_and_outputs(tmp_path, quantized):
    torch = pytest.importorskip("torch")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from emmy.serving.twins import capture_twin_graphs
    from tests.serving.helpers import QWEN3_5_TINY

    config = Qwen3_5TextConfig(**QWEN3_5_TINY)
    if quantized:
        import numpy as np
        from safetensors.torch import save_file

        from emmy.compiler.loader.quant import quantize_nvfp4
        from emmy.compiler.loader.synthesize import _NVFP4_CONFIG

        config.quantization_config = _NVFP4_CONFIG
        packed, scales, scale2 = quantize_nvfp4(np.random.default_rng(0).standard_normal((128, 64), dtype=np.float32))
        base = "model.layers.0.linear_attn.in_proj_qkv"
        save_file(
            {
                f"{base}.weight": torch.from_numpy(packed),
                f"{base}.weight_scale": torch.from_numpy(np.ascontiguousarray(scales)).view(torch.float8_e4m3fn),
                f"{base}.weight_scale_2": torch.from_numpy(scale2),
                f"{base}.input_scale": torch.tensor([0.05]),
                # An unquantized multi-token-prediction layer numbered like the trunk's first: not a twin's layer.
                "mtp.layers.0.linear_attn.in_proj_qkv.weight": torch.zeros(128, 64, dtype=torch.bfloat16),
            },
            str(tmp_path / "model.safetensors"),
        )
    config.save_pretrained(tmp_path)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=1, prefill_bucket=16, symbolic=False)
    suffix = "@nvfp4" if quantized else ""
    assert set(graphs) == {
        f"gdn1{suffix}",
        f"gdn16{suffix}",
        f"gdn64-count{suffix}",
        "pre1-global",
        "post1-global",
        "pre16-global",
        "post16-global",
    }
    for name, graph in graphs.items():
        if not name.startswith("gdn"):
            continue
        rows = int(name.removeprefix("gdn").split("@")[0].split("-")[0])
        if quantized:
            assert set(_packed_weights(graph)) == {"model.layers.0.linear_attn.in_proj_qkv.weight"}
        assert [tuple(graph.buffer(key).shape) for key in graph.inputs] == [(1, rows, 64), (1, 4, 16, 16), (1, 128, 4)] + (
            [(1,)] if "-count" in name else []
        )
        assert [tuple(graph.buffer(key).shape) for key in graph.outputs] == [(1, rows, 64), (1, 4, 16, 16), (1, 128, 4)]
    assert len(graphs["pre1-global"].outputs) == 4
    post = graphs["post1-global"]
    assert [tuple(post.buffer(key).shape) for key in post.inputs] == [(1, 64)] * 3
    # A GDN layer has no any-width program: the any-width capture keeps its static twins and adds only attention's.
    symbolic = capture_twin_graphs(str(tmp_path), decode_bucket=1, prefill_bucket=16)
    assert set(symbolic) == set(graphs) | {"pre-sym-global", "post-sym-global"}
    # Width 1 serves any token count without padding, so a GDN layer has it whatever widths attention gets.
    assert set(capture_twin_graphs(str(tmp_path), decode_bucket=0, prefill_bucket=0)) == {
        f"gdn1{suffix}",
        f"gdn64-count{suffix}",
        "pre-sym-global",
        "post-sym-global",
    }


def test_bf16_serving_config_captures_bf16_twins_with_static_gdn(tmp_path):
    """A config serving ``--dtype bfloat16`` gets BF16 twins: every buffer crossing a twin's boundary is BF16
    except the GDN matrix state, which stays f32. The GDN layer has static twins only, width 1 among them, and
    its width-1 twin carries a row in every lane although attention layers build no width-1 twin."""
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from emmy.serving.release import load_serving_config
    from emmy.serving.twins import capture_serving_twins, twin_realizations
    from tests.serving.helpers import QWEN3_5_TINY

    Qwen3_5TextConfig(**QWEN3_5_TINY).save_pretrained(tmp_path / "model")
    env = tmp_path / "serving.env"
    env.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={tmp_path / 'g.json'}\nSERVE_MAX_NUM_BATCHED_TOKENS=16\n"
        'SERVE_DECODE_BUCKET=4\nSERVE_PREFILL_BUCKET=16\nSERVE_M1_TIER=0\nSERVE_EXTRA_ARGS="--dtype bfloat16"\n'
        'SERVE_WARM_SHAPES=":::fm"\n'
    )
    serving = load_serving_config(env)
    graphs = capture_serving_twins(str(tmp_path / "model"), serving)
    assert [row.name for row in twin_realizations(serving, "gdn1")] == ["m1", "m1.fm"]
    assert [row.name for row in twin_realizations(serving, "gdn64-count")] == ["m64", "m64.fm"]
    assert all(row.bindings == (("num_tokens", 64),) for row in twin_realizations(serving, "gdn64-count"))
    assert twin_realizations(serving, "pre1-global") == ()
    assert set(graphs) == {
        "gdn1",
        "gdn4",
        "gdn16",
        "gdn64-count",
        "pre4-global",
        "post4-global",
        "pre16-global",
        "post16-global",
        "pre-sym-global",
        "post-sym-global",
    }
    for name, graph in graphs.items():
        boundary = {key: graph.buffer(key).dtype.name for key in (*graph.inputs, *graph.outputs)}
        state = {key for key in boundary if len(graph.buffer(key).shape) == 4}  # S[1, heads, k, v]
        if "-count" in name:
            assert boundary.pop(graph.inputs[-1]) == "i64"
        assert {key: dt for key, dt in boundary.items() if key not in state} == dict.fromkeys(set(boundary) - state, "bf16"), name
        assert all(boundary[key] == "f32" for key in state), name


def test_attention_query_layout_accepts_validated_deepseek_low_rank_signature():
    from types import SimpleNamespace

    def proj(width):
        return SimpleNamespace(out_features=width)

    attention = SimpleNamespace(
        head_dim=8,
        num_heads=4,
        q_a_proj=proj(16),
        q_a_norm=object(),
        q_b_proj=proj(32),
        q_b_norm=object(),
        kv_proj=proj(8),
        kv_norm=object(),
        o_a_proj=proj(16),
        o_b_proj=proj(32),
    )
    assert _attention_query_layout(attention) == (4, 32)


def test_attention_query_layout_rejects_partial_or_inconsistent_signature():
    from types import SimpleNamespace

    with pytest.raises(NotImplementedError, match="neither q_proj nor the complete DeepSeek"):
        _attention_query_layout(SimpleNamespace(head_dim=8, num_heads=4, kv_proj=SimpleNamespace(out_features=8)))

    attention = SimpleNamespace(
        head_dim=8,
        num_heads=4,
        q_a_proj=object(),
        q_a_norm=object(),
        q_b_proj=SimpleNamespace(out_features=24),
        q_b_norm=object(),
        kv_proj=SimpleNamespace(out_features=8),
        kv_norm=object(),
        o_a_proj=object(),
        o_b_proj=object(),
    )
    with pytest.raises(ValueError, match="DeepSeek attention shape mismatch"):
        _attention_query_layout(attention)


def test_deepseek_serving_twins_capture_the_hyper_connection_seam_weight_free(tmp_path):
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    from emmy.serving.twins import capture_twin_graphs

    config = transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=32,
        moe_intermediate_size=16,
        num_hidden_layers=3,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=1,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        layer_types=["sliding_attention", "heavily_compressed_attention", "compressed_sparse_attention"],
        mlp_layer_types=["hash_moe", "moe", "moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=4,
        swiglu_limit=10.0,
        max_position_embeddings=64,
    )
    config.save_pretrained(tmp_path)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0)
    # The attention sublayer is the fork's and routing runs outside the twins, so the three attention kinds and two
    # router kinds of this config all compile the same programs: ONE profile, not one per (mlp, attention) pairing.
    # The expert twins add width 1: every MoE boot compiles that program for the fixed-slot tier.
    assert set(graphs) == {f"{half}{width}" for half in ("pre", "post", "expert") for width in ("4", "-sym")} | {"expert1"}
    pre, post, expert = (graphs[f"{half}4"] for half in ("pre", "post", "expert"))
    assert [tuple(pre.nodes[i].output.shape) for i in pre.inputs] == [(4, 64)]
    assert [tuple(pre.nodes[o].output.shape) for o in pre.outputs] == [(4, 32)]
    assert [tuple(post.nodes[i].output.shape) for i in post.inputs] == [(4, 32), (4, 64)]
    assert [tuple(post.nodes[o].output.shape) for o in post.outputs] == [(4, 64), (4, 32), (4, 2)]
    assert [tuple(expert.nodes[i].output.shape) for i in expert.inputs] == [(4, 32), (32, 32), (32, 16)]
    token_dim = graphs["pre-sym"].nodes[graphs["pre-sym"].inputs[0]].output.shape[0]
    assert not token_dim.is_static and token_dim.as_atom_name() == "num_tokens"


@pytest.mark.parametrize(("inter", "slices"), [(32, 1), (64, 2)])
def test_deepseek_expert_twin_records_the_native_mxfp4_program_serving_binds(tmp_path, inter, slices):
    """The pinned checkpoint declares an fp8 TRUNK while storing its routed experts as native MXFP4
    (``expert_dtype: fp4``), and the serving loader keeps those experts packed. The expert twin has
    to follow the experts, not the trunk declaration, or the golden records a program serving never
    runs. ``w_down``'s packed shape also pins the layout the caller passes down: these experts are
    ``F.linear`` parameters, so the blocks lead with ``out``; the ``x @ W`` reading would produce
    (16, 2, 16) here instead. Served across tensor-parallel ranks, each rank holds a slice of every
    expert, so the twin is the sliced program: intermediate 64 over two ranks records the 32 one."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from safetensors.torch import save_file

    from emmy.serving.twins import capture_twin_graphs

    config = transformers.DeepseekV4Config(
        vocab_size=64,
        hidden_size=64,
        moe_intermediate_size=inter,
        num_hidden_layers=2,
        num_attention_heads=4,
        head_dim=8,
        q_lora_rank=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        n_shared_experts=1,
        o_groups=1,
        o_lora_rank=16,
        index_n_heads=2,
        index_head_dim=4,
        index_topk=2,
        hc_mult=2,
        hc_sinkhorn_iters=2,
        layer_types=["sliding_attention", "heavily_compressed_attention"],
        mlp_layer_types=["hash_moe", "moe"],
        compress_rates={"compressed_sparse_attention": 4, "heavily_compressed_attention": 4},
        sliding_window=4,
        swiglu_limit=10.0,
        max_position_embeddings=64,
    )
    config.quantization_config = {
        "activation_scheme": "dynamic",
        "fmt": "e4m3",
        "quant_method": "fp8",
        "scale_fmt": "ue8m0",
        "weight_block_size": [128, 128],
    }
    config.expert_dtype = "fp4"
    config.save_pretrained(tmp_path)
    # The fp8 trunk declaration sends the capture to the shards for the trunk's coded weights. This
    # one stores none, so the trunk twins stay plain and only the expert program is under test.
    save_file({"embed.weight": torch.zeros(64, 64, dtype=torch.float16)}, str(tmp_path / "model.safetensors"))
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0, expert_slices=slices)
    assert set(graphs) == {"pre4", "pre-sym", "post4", "post-sym", "expert1@mxfp4", "expert4@mxfp4", "expert-sym@mxfp4"}
    expert = graphs["expert4@mxfp4"]
    assert set(expert.inputs) >= {"w_gate_up", "w_gate_up_scale", "w_down", "w_down_scale"}
    by_id = {i: expert.nodes[i].output for i in expert.inputs}
    assert {out.dtype.name for i, out in by_id.items() if i.startswith("w_")} == {"u8"}
    shape = tuple(d.as_static() for d in by_id["w_down"].shape)
    assert shape == (64, 1, 16), f"expected (out, in/32, 16) for an F.linear expert, got {shape}"
    assert tuple(d.as_static() for d in by_id["w_gate_up"].shape) == (64, 2, 16)


@pytest.mark.skip(reason="global greedy ranking still traverses the full coded-expert schedule space")
def test_laguna_coded_expert_inputs_are_spelled_per_allocation_profile():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from emmy.serving.gen_runner import trace_split

    class Expert(nn.Module):
        def forward(self, x, w_gate, w_up, w_down):
            return nn.functional.linear(nn.functional.silu(nn.functional.linear(x, w_gate)) * nn.functional.linear(x, w_up), w_down)

    # The deployed Laguna shape. Shape-only compilation allocates no checkpoint tensors and
    # catches primitives which tiny padded examples can accidentally fold away.
    h, inter = 3072, 1024
    graph = trace_split(
        Expert(),
        (
            torch.zeros(1, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(h, inter, dtype=torch.float16),
        ),
        None,
    )
    storage = {}
    for proj, n, k in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
        base = f"model.layers.1.mlp.experts.0.{proj}"
        storage[base] = _entry(base, n, k, 2)
    twins = _spell_expert_twins("expert1-sparse-sliding", graph, storage)
    assert list(twins) == ["expert1-sparse-sliding@b2"]
    spelled = twins["expert1-sparse-sliding@b2"]
    assert spelled.inputs[:4] == ["x", "w_gate", "w_up", "w_down"]
    assert {spelled.nodes[name].output.dtype.name for name in ("w_gate", "w_up", "w_down")} == {"i16"}
    from emmy.compiler.backend.plan import plan_from_dict, plan_from_graph, plan_to_dict
    from emmy.compiler.context import Context
    from emmy.compiler.ir.base import ConstantOp, InputOp
    from emmy.compiler.ir.cuda import CudaOp
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline

    lowered = Pipeline.build(CUDA_PASSES).run(spelled, ctx=Context(compute_capability=(12, 0)))
    cuda = [node.op for node in lowered.nodes.values() if isinstance(node.op, CudaOp)]
    assert cuda and all(op.kernel_source for op in cuda)
    assert all(isinstance(node.op, (InputOp, ConstantOp, CudaOp)) for node in lowered.nodes.values())
    plan = plan_from_graph(lowered)
    assert plan_to_dict(plan_from_dict(plan_to_dict(plan))) == plan_to_dict(plan)
    assert plan.launches and plan.weights
    assert all(weight.generated is not None and weight.load_ops == () for weight in plan.weights.values())
    for weight in ("w_gate", "w_up", "w_down"):
        assert not {f"{weight}_decoded_{table}" for table in ("bit_start", "word_idx", "next_word")} & set(plan.weights)
        assert {f"{weight}_decoded_shift_step", f"{weight}_decoded_tile_step"} <= set(plan.weights)
    factors = [spec.generated for spec in plan.weights.values() if spec.generated is not None and spec.generated[1] == (128, 128)]
    assert len(factors) == 3 and {factor[0] for factor in factors} == {"<f4"}
    from emmy.serving.gen_runner import _bind_plan_constants

    assert set(_bind_plan_constants(plan, {}, cache=None)) == set(plan.weights)
    active_ir = "\n".join(f"{nid} {type(node.op).__module__} {type(node.op).__name__}" for nid, node in lowered.nodes.items())
    assert "trellis" not in active_ir.lower() and "exl3" not in active_ir.lower()


def test_symbolic_laguna_coded_expert_preserves_the_token_dim():
    """The any-width serving twin keeps ``num_tokens`` symbolic through all three factors."""
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from emmy.compiler.ir.frontend.ir import MatmulOp, ReshapeOp
    from emmy.serving.gen_runner import trace_split

    class Expert(nn.Module):
        def forward(self, x, w_gate, w_up, w_down):
            return nn.functional.linear(nn.functional.silu(nn.functional.linear(x, w_gate)) * nn.functional.linear(x, w_up), w_down)

    h = inter = 128
    graph = trace_split(
        Expert(),
        (
            torch.zeros(8, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(inter, h, dtype=torch.float16),
            torch.zeros(h, inter, dtype=torch.float16),
        ),
        ["x"],
    )
    traced_token_dim = graph.nodes["x"].output.shape[0]
    assert not traced_token_dim.is_static and traced_token_dim.as_atom_name() == "num_tokens"

    storage = {}
    for proj, n, k in (("gate_proj", inter, h), ("up_proj", inter, h), ("down_proj", h, inter)):
        base = f"model.layers.1.mlp.experts.0.{proj}"
        storage[base] = _entry(base, n, k, 2)
    spelled = _spell_expert_twins("expert-sym-sparse-sliding", graph, storage)["expert-sym-sparse-sliding@b2"]
    token_dim = spelled.nodes["x"].output.shape[0]
    assert token_dim == traced_token_dim

    dynamic_factors = [
        node
        for node in spelled.nodes.values()
        if isinstance(node.op, (MatmulOp, ReshapeOp)) and node.output.shape and not node.output.shape[0].is_static
    ]
    assert dynamic_factors
    assert all(node.output.shape[0] == token_dim for node in dynamic_factors)
    for prefix in ("linear", "linear_1", "linear_2"):
        factored = [
            spelled.nodes[f"{prefix}_{suffix}"]
            for suffix in ("x32", "left_blocks", "left_factor", "left_flat32", "core", "right_blocks", "right_factor", "right_flat")
        ]
        assert len({id(node.output.shape[0]) for node in factored}) == 1
    assert spelled.nodes[spelled.outputs[0]].output.shape[0] == token_dim


def test_laguna_serving_twin_capture_includes_symbolic_coded_expert(monkeypatch, tmp_path):
    """The public weight-free capture path inventories the any-width coded expert."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    import emmy.serving.twins as twins

    cfg = transformers.LagunaConfig(
        vocab_size=32,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        max_position_embeddings=64,
        moe_intermediate_size=128,
        shared_expert_intermediate_size=128,
        num_experts=2,
        num_experts_per_tok=1,
        layer_types=["full_attention"],
        mlp_layer_types=["sparse"],
        num_attention_heads_per_layer=[4],
        sliding_window=16,
    )
    cfg.save_pretrained(tmp_path)
    storage = {}
    for proj, n, k in (("gate_proj", 128, 128), ("up_proj", 128, 128), ("down_proj", 128, 128)):
        base = f"model.layers.0.mlp.experts.0.{proj}"
        storage[base] = _entry(base, n, k, 2)
    monkeypatch.setattr(twins, "coded_tensor_storage", lambda *_args, **_kwargs: storage)

    graphs = twins.capture_twin_graphs(str(tmp_path), decode_bucket=1, prefill_bucket=0)
    assert "expert-sym@b2" in graphs
    token_dim = graphs["expert-sym@b2"].nodes["x"].output.shape[0]
    assert not token_dim.is_static and token_dim.as_atom_name() == "num_tokens"


@pytest.mark.parametrize(
    ("spec", "want"),
    [
        ("turboderp/GLM-4.5-Air-exl3@2.25bpw", ("turboderp/GLM-4.5-Air-exl3", "2.25bpw")),
        ("google/gemma-3-12b-it", ("google/gemma-3-12b-it", None)),
    ],
)
def test_model_tag_may_pin_the_rung(spec, want):
    """A coded checkpoint's rung lives on a branch, and the rungs differ in exactly the bit
    allocation the keys carry — so the ``model:`` tag may pin one."""
    assert split_revision(spec) == want


def test_fp8_expert_twins_spell_bits_and_block_scales_and_keep_the_unconverted_profile():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from emmy.compiler.loader.quant import fp8_weight_profile
    from emmy.serving.gen_runner import trace_split
    from emmy.serving.twins import _spell_fp8_expert_twins

    class Expert(nn.Module):
        def forward(self, x, w_gate_up, w_down):
            gate, up = nn.functional.linear(x, w_gate_up).chunk(2, dim=-1)
            return nn.functional.linear(nn.functional.silu(gate) * up, w_down)

    h, inter = 3072, 1024
    graph = trace_split(
        Expert(),
        tuple(torch.zeros(*shape, dtype=torch.float16) for shape in ((1, h), (2 * inter, h), (h, inter))),
        None,
    )
    config = type("Cfg", (), {})()
    config.quantization_config = {
        "quant_method": "fp8",
        "weight_block_size": [128, 128],
        "ignored_layers": ["lm_head", "model.layers.44.mlp.experts"],
    }
    profile = fp8_weight_profile(config)
    assert profile == ("f8e4m3", (128, 128), ["lm_head", "model.layers.44.mlp.experts"])

    twins = _spell_fp8_expert_twins("expert-sparse-sliding", graph, profile, {1, 44})
    assert set(twins) == {"expert-sparse-sliding@f8e4m3", "expert-sparse-sliding"}
    spelled = twins["expert-sparse-sliding@f8e4m3"]
    spelled.validate()
    assert spelled.inputs == ["x", "w_gate_up", "w_down", "w_gate_up_scale", "w_down_scale"]
    assert spelled.nodes["w_gate_up"].output.dtype.name == "f8e4m3"
    # Block scales are declared at the interleaved (grid, 1) layout the dequant cone broadcasts.
    assert tuple(d.as_static() for d in spelled.nodes["w_gate_up_scale"].output.shape) == (16, 1, 24, 1)
    assert tuple(d.as_static() for d in spelled.nodes["w_down_scale"].output.shape) == (24, 1, 8, 1)
    assert twins["expert-sparse-sliding"] is graph
    # Every layer converted: no plain twin. A config without fp8 spells nothing.
    assert list(_spell_fp8_expert_twins("e", graph, profile, {1, 2})) == ["e@f8e4m3"]
    assert fp8_weight_profile(type("Cfg", (), {"quantization_config": {"quant_method": "exl3"}})()) is None


def test_mxfp4_expert_twins_spell_native_blocks_and_scales():
    torch = pytest.importorskip("torch")
    import torch.nn as nn

    from emmy.compiler.loader.quant import mxfp4_weight_profile
    from emmy.serving.gen_runner import trace_split
    from emmy.serving.twins import _spell_mxfp4_expert_twins

    class Expert(nn.Module):
        def forward(self, x, w_gate_up, w_down, b_gate_up, b_down):
            gate, up = (x @ w_gate_up + b_gate_up).chunk(2, dim=-1)
            return gate * up @ w_down + b_down

    hidden, inter = 64, 32
    graph = trace_split(
        Expert(),
        tuple(
            torch.zeros(*shape, dtype=torch.float16)
            for shape in ((1, hidden), (hidden, 2 * inter), (inter, hidden), (2 * inter,), (hidden,))
        ),
        None,
    )
    config = type("Cfg", (), {"quantization_config": {"quant_method": "mxfp4", "modules_to_not_convert": ["lm_head"]}})()
    profile = mxfp4_weight_profile(config)
    twins = _spell_mxfp4_expert_twins("expert1", graph, profile, {0, 1}, True)
    assert set(twins) == {"expert1@mxfp4"}
    spelled = twins["expert1@mxfp4"]
    spelled.validate()
    assert spelled.inputs == ["x", "w_gate_up", "w_down", "b_gate_up", "b_down", "w_gate_up_scale", "w_down_scale"]
    assert tuple(d.as_static() for d in spelled.nodes["w_gate_up"].output.shape) == (64, 2, 16)
    assert tuple(d.as_static() for d in spelled.nodes["w_down"].output.shape) == (64, 1, 16)
    assert tuple(d.as_static() for d in spelled.nodes["w_gate_up_scale"].output.shape) == (64, 2)
    assert profile == ["lm_head"]

    with pytest.raises(NotImplementedError, match=r"requires every routed-expert layer.*layer\(s\) \[1\]"):
        _spell_mxfp4_expert_twins("expert1", graph, ["model.layers.1.mlp.experts"], {0, 1}, True)


def _nvfp4_checkpoint(path: Path) -> None:
    """A tiny two-layer Qwen3 checkpoint whose every linear is stored as the NVFP4 packed trio,
    with the per-linear ``input_scale`` that declares the activation half.

    Written with the real quantizer rather than by hand: the twin arm reads stored shapes and
    calibrated scale VALUES off this directory, so a fixture that only looked right in an index
    would not exercise what the spellers do with it.
    """
    import numpy as np
    import torch
    import transformers
    from safetensors.torch import save_file

    from emmy.compiler.loader.quant import quantize_nvfp4
    from emmy.compiler.loader.synthesize import _NVFP4_CONFIG

    hidden, inter, heads, kv, head_dim, layers = 64, 128, 4, 2, 16, 2
    config = transformers.Qwen3Config(
        vocab_size=64,
        hidden_size=hidden,
        intermediate_size=inter,
        num_hidden_layers=layers,
        num_attention_heads=heads,
        num_key_value_heads=kv,
        head_dim=head_dim,
        max_position_embeddings=64,
    )
    config.save_pretrained(path)
    document = json.loads((path / "config.json").read_text())
    document["quantization_config"] = _NVFP4_CONFIG
    (path / "config.json").write_text(json.dumps(document, indent=1))

    shapes = {
        "self_attn.q_proj": (heads * head_dim, hidden),
        "self_attn.k_proj": (kv * head_dim, hidden),
        "self_attn.v_proj": (kv * head_dim, hidden),
        "self_attn.o_proj": (hidden, heads * head_dim),
        "mlp.gate_proj": (inter, hidden),
        "mlp.up_proj": (inter, hidden),
        "mlp.down_proj": (hidden, inter),
    }
    rng = np.random.default_rng(0)
    tensors: dict[str, object] = {}
    for layer in range(layers):
        for module, (n, k) in shapes.items():
            packed, scale_bits, scale_2 = quantize_nvfp4(rng.standard_normal((n, k), dtype=np.float32))
            base = f"model.layers.{layer}.{module}"
            tensors[f"{base}.weight"] = torch.from_numpy(packed)
            tensors[f"{base}.weight_scale"] = torch.from_numpy(np.ascontiguousarray(scale_bits)).view(torch.float8_e4m3fn)
            tensors[f"{base}.weight_scale_2"] = torch.from_numpy(scale_2)
            # One calibrated activation level per linear, as modelopt writes it. ``q_proj``'s differs
            # from ``k_proj`` / ``v_proj``'s so the twin exercises both halves of the sharing rule:
            # equal levels over one activation share a single quantize, unequal ones get their own.
            tensors[f"{base}.input_scale"] = torch.tensor([0.03 if module.startswith("self_attn.q") else 0.05], dtype=torch.float32)
    save_file(tensors, str(path / "model.safetensors"))


def _static_fp8_checkpoint(path: Path, *, dynamic: bool = False) -> None:
    """A tiny two-layer Qwen3 checkpoint in the official FP8 form with STATIC activations: every
    linear stores its e4m3 bits, one per-tensor ``weight_scale_inv`` and one calibrated
    ``activation_scale``. ``q_proj``'s differs from ``k_proj`` / ``v_proj``'s, as in the NVFP4
    fixture, so both halves of the sharing rule are exercised.

    ``dynamic`` writes the block form Qwen and DeepSeek publish instead: dynamic activations, one
    ``weight_scale_inv`` per 32 x 32 weight block, no activation scale."""
    import torch
    import transformers
    from safetensors.torch import save_file

    hidden, inter, heads, kv, head_dim, layers = 64, 128, 4, 2, 16, 2
    config = transformers.Qwen3Config(
        vocab_size=64, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
        num_attention_heads=heads, num_key_value_heads=kv, head_dim=head_dim, max_position_embeddings=64,
    )  # fmt: skip
    config.save_pretrained(path)
    document = json.loads((path / "config.json").read_text())
    document["quantization_config"] = (
        {"quant_method": "fp8", "activation_scheme": "dynamic", "fmt": "e4m3", "weight_block_size": [32, 32]}
        if dynamic
        else {"quant_method": "fp8", "activation_scheme": "static", "weight_block_size": None}
    )
    (path / "config.json").write_text(json.dumps(document, indent=1))

    shapes = {
        "self_attn.q_proj": (heads * head_dim, hidden),
        "self_attn.k_proj": (kv * head_dim, hidden),
        "self_attn.v_proj": (kv * head_dim, hidden),
        "self_attn.o_proj": (hidden, heads * head_dim),
        "mlp.gate_proj": (inter, hidden),
        "mlp.up_proj": (inter, hidden),
        "mlp.down_proj": (hidden, inter),
    }
    generator = torch.Generator().manual_seed(0)
    tensors: dict[str, object] = {}
    for layer in range(layers):
        for module, shape in shapes.items():
            base = f"model.layers.{layer}.{module}"
            bits = torch.randint(0, 0x7F, shape, dtype=torch.uint8, generator=generator)  # finite e4m3 codes
            tensors[f"{base}.weight"] = bits.view(torch.float8_e4m3fn)
            if dynamic:
                blocks = (shape[0] // 32, shape[1] // 32)
                tensors[f"{base}.weight_scale_inv"] = (torch.rand(blocks, generator=generator) * 0.004).to(torch.bfloat16)
                continue
            tensors[f"{base}.weight_scale_inv"] = torch.tensor(0.002, dtype=torch.bfloat16)
            tensors[f"{base}.activation_scale"] = torch.tensor(0.03 if module.startswith("self_attn.q") else 0.05, dtype=torch.bfloat16)
    save_file(tensors, str(path / "model.safetensors"))


def test_static_fp8_serving_twins_carry_the_declared_w8a8_program(tmp_path):
    """A twin of a static-FP8 checkpoint records the checkpoint's own program: FP8 weight bits
    under their checkpoint keys, and one FP8 encode per calibrated activation scale. Nothing is
    exported beside the model's outputs — the codes stay an interior value of the kernel."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.serving.twins import capture_twin_graphs

    _static_fp8_checkpoint(tmp_path)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0, symbolic=False)
    assert set(graphs) == {"pre4@fp8", "post4@fp8"}

    def weights(graph):
        return {n.op.source_path for n in graph.nodes.values() if isinstance(n.op, ConstantOp) and n.output.dtype.name == "f8e4m3"}

    def encodes(graph):
        return [n for n in graph.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.op.name == "to_f8e4m3"]

    assert weights(graphs["pre4@fp8"]) == {f"model.layers.0.self_attn.{m}_proj.weight" for m in "qkv"}
    assert weights(graphs["post4@fp8"]) == {
        f"model.layers.0.{module}.weight" for module in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }
    # q reads the normed hidden state through its own scale; k and v share theirs. The post half
    # quantizes the attention output, the normed residual (gate and up share it) and the MLP product.
    assert len(encodes(graphs["pre4@fp8"])) == 2
    assert len(encodes(graphs["post4@fp8"])) == 3
    for graph in graphs.values():
        assert not any(node.hints.get("trace.materialize") for node in graph.nodes.values())
        graph.validate()


def test_dynamic_fp8_serving_twins_carry_coded_weights_under_16_bit_activations(tmp_path):
    """A twin of a dynamic block-FP8 checkpoint records what serving compiles from it: the FP8
    weight bits and their block scales under their checkpoint keys, and no activation encode. Serving
    runs that checkpoint weight-only, so a card without FP8 arithmetic holds the trunk at its stored
    size."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.ir.tensor.ir import ElementwiseOp
    from emmy.serving.twins import capture_twin_graphs

    _static_fp8_checkpoint(tmp_path, dynamic=True)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0, symbolic=False)
    assert set(graphs) == {"pre4@fp8", "post4@fp8"}

    def constants(graph, dtype):
        return {
            n.op.source_path: tuple(d.as_static() for d in n.output.shape)
            for n in graph.nodes.values()
            if isinstance(n.op, ConstantOp) and n.output.dtype.name == dtype
        }

    assert set(constants(graphs["pre4@fp8"], "f8e4m3")) == {f"model.layers.0.self_attn.{m}_proj.weight" for m in "qkv"}
    assert constants(graphs["pre4@fp8"], "f32")["model.layers.0.self_attn.q_proj.weight_scale_inv"] == (2, 1, 2, 1)
    for graph in graphs.values():
        assert not [n for n in graph.nodes.values() if isinstance(n.op, ElementwiseOp) and n.op.op.name.startswith("to_f8")]
        graph.validate()


def _packed_weights(graph: Graph) -> dict[str, tuple[int, ...]]:
    """``{checkpoint key: packed shape}`` for every NVFP4 weight constant the twin carries."""
    return {
        node.op.source_path: tuple(d.as_static() for d in node.output.shape)
        for node in graph.nodes.values()
        if isinstance(node.op, ConstantOp) and node.output.dtype.name == "f4e2m1x2"
    }


def test_nvfp4_serving_twins_carry_the_declared_w4a4_program(tmp_path):
    """A twin of an NVFP4 checkpoint must record what serving compiles: packed weights AND the
    static 4-bit activation encode the checkpoint's calibration declares. A float16 twin over
    dequantized weights records a program serving never runs, and tuning it transfers nothing."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.serving.twins import capture_twin_graphs

    _nvfp4_checkpoint(tmp_path)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0, symbolic=False)
    assert set(graphs) == {"pre4@nvfp4", "post4@nvfp4"}

    # The traced wrapper-relative paths are re-addressed to the representative layer's checkpoint
    # keys, which is what lets the checkpoint-driven spellers fire at all.
    assert _packed_weights(graphs["pre4@nvfp4"]) == {
        "model.layers.0.self_attn.q_proj.weight": (64, 32),
        "model.layers.0.self_attn.k_proj.weight": (32, 32),
        "model.layers.0.self_attn.v_proj.weight": (32, 32),
    }
    assert set(_packed_weights(graphs["post4@nvfp4"])) == {
        f"model.layers.0.{module}.weight" for module in ("self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
    }

    for name, graph in graphs.items():
        graph.validate()
        encodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "ElementwiseOp" and n.op.op.name == "to_f4e2m1"]
        packed_activations = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and not isinstance(n.op, ConstantOp)]
        assert encodes, f"{name}: no to_f4e2m1 encode — the twin still runs 16-bit activations over packed weights"
        assert packed_activations, f"{name}: no packed f4e2m1x2 activation buffer"
        # The capture's delivery form is one JSON per twin, so a packed twin that cannot round-trip
        # cannot be recorded.
        assert _structure(Graph.from_dict(json.loads(json.dumps(graph.to_dict())))) == _structure(graph)


@pytest.mark.parametrize("dynamic", [False, True])
def test_fp8_trunk_stays_coded_on_the_serving_lane(tmp_path, dynamic):
    """The serving loader leaves an FP8 trunk linear undecoded — a placeholder at the declared
    shape — and says so in the store, which is what sends the runner to the checkpoint for the
    bits. The default lane still decodes the same checkpoint to values."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    from emmy.compiler.trace.huggingface import load_quantized_split

    _static_fp8_checkpoint(tmp_path, dynamic=dynamic)
    model, store = load_quantized_split(tmp_path, torch.float16, compress_trunk=True)
    assert store["trunk"] == "codes" and store["dir"] == str(tmp_path)
    weight = model.state_dict()["model.layers.0.self_attn.q_proj.weight"]
    assert not weight.is_meta and weight.dtype == torch.float16 and tuple(weight.shape) == (64, 64)
    _model, decoded = load_quantized_split(tmp_path, torch.float16)
    assert decoded["trunk"] == "values"


@pytest.mark.parametrize("scheme", ["nvfp4", "fp8", "fp8-dynamic"])
def test_checkpoint_spelled_twin_is_the_graph_serving_stamps(tmp_path, scheme):
    """The transfer property, asserted directly: the captured twin and the graph
    ``gen_runner._compile_split`` stamps on the same wrapper at the same width are the same graph.

    Tuning evidence is keyed by kernel identity, so evidence recorded against the twin reaches
    serving only while these two agree. They can only agree by construction — one trace path, one
    re-addressing, one spell sequence — which is why the twin arm reuses all three rather than
    reproducing the program from the checkpoint's declaration."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")

    import transformers

    from emmy.compiler.loader.quant import strip_engine_quant_config
    from emmy.compiler.trace.huggingface import build_attention_split_wrapper
    from emmy.serving.gen_runner import _compile_split
    from emmy.serving.twins import capture_twin_graphs

    if scheme == "fp8-dynamic":
        _static_fp8_checkpoint(tmp_path, dynamic=True)
    else:
        {"nvfp4": _nvfp4_checkpoint, "fp8": _static_fp8_checkpoint}[scheme](tmp_path)
    suffix = scheme.removesuffix("-dynamic")
    twins = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=0, symbolic=False)

    config = transformers.AutoConfig.from_pretrained(tmp_path)
    strip_engine_quant_config(config)
    with torch.device("meta"):
        trunk = transformers.AutoModel.from_config(config, dtype=torch.float16).eval()
    pre_w, post_w = build_attention_split_wrapper(trunk.layers[0])
    pre_w.to_empty(device="cpu").to(torch.float16)
    post_w.to_empty(device="cpu").to(torch.float16)
    # What the runner threads as ``ckpt``: the checkpoint dir plus parameter identity → its key.
    id_to_key = {id(t): f"model.{path}" for path, t in trunk.named_parameters(remove_duplicate=False)}

    class _Stamped(Exception):
        def __init__(self, graph):
            self.graph = graph

    class _CaptureBackend:
        def __init__(self, **_kwargs):
            pass

        def compile(self, graph, *, ctx=None):
            raise _Stamped(graph)

    hidden, attn_width = config.hidden_size, config.num_attention_heads * config.head_dim
    examples = {
        "pre": [torch.zeros(4, hidden, dtype=torch.float16)],
        "post": [torch.zeros(4, attn_width, dtype=torch.float16), torch.zeros(4, hidden, dtype=torch.float16)],
    }
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("emmy.compiler.backend.cuda.backend.CudaBackend", _CaptureBackend)
        for half, wrapper in (("pre", pre_w), ("post", post_w)):
            with pytest.raises(_Stamped) as caught:
                _compile_split(wrapper, examples[half], None, F16, ckpt=(str(tmp_path), id_to_key))
            assert _structure(caught.value.graph) == _structure(twins[f"{half}4@{suffix}"])


def _structure(graph: Graph):
    """A graph's node structure, ignoring the values behind it: every node's op kind, operands,
    output dtype and output shape, plus the program's own input/output lists."""
    return (
        list(graph.inputs),
        list(graph.outputs),
        {
            nid: (type(node.op).__name__, tuple(node.inputs), node.output.dtype.name, tuple(str(d) for d in node.output.shape))
            for nid, node in graph.nodes.items()
        },
    )
