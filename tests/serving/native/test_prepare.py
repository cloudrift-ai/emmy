"""Native preparation accepts Qwen3 and Qwen3.5 text models, dense or MoE, and rejects the rest before CUDA work."""

import contextlib

import numpy as np
import pytest

from emmy.serving.native.prepare import load_model, validate_model
from tests.compiler.helpers import requires_cuda
from tests.serving.helpers import fp8_block_checkpoint, qwen3_5_model, qwen3_5_moe_model, qwen3_model


def test_configuration_rejections():
    model = qwen3_model(2).half()
    validate_model(model, 8)
    model.config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="load_model"):
        validate_model(model, 8)
    del model.config.quantization_config
    for length in (0, 65, 4097):
        with pytest.raises(ValueError, match="context"):
            validate_model(model, length)
    model.float()
    with pytest.raises(ValueError, match="FP16"):
        validate_model(model, 8)
    # One model dtype, FP16 or BF16.
    model.bfloat16()
    validate_model(model, 8)
    model.model.norm.half()
    with pytest.raises(ValueError, match="BF16"):
        validate_model(model, 8)
    model.half()
    model.config.layer_types[0] = "sliding_attention"
    with pytest.raises(ValueError, match="sliding"):
        validate_model(model, 8)


def test_native_cli_modes_reject_unsupported_sampling():
    import argparse

    from emmy.commands.generate import handle_generate, register_generate_command

    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    for mode in ("--native-pack", "--export-native"):
        args = parser.parse_args(["generate", "unused", mode, "artifact", "--top-k", "5"])
        with pytest.raises(ValueError, match="top-k"):
            handle_generate(args)
    args = parser.parse_args(["generate", "unused", "--native-pack", "artifact", "--max-new-tokens", "-1"])
    with pytest.raises(ValueError, match="nonnegative"):
        handle_generate(args)


def test_native_only_arguments_fail_before_loading_a_model():
    import argparse

    from emmy.commands.generate import handle_generate, register_generate_command

    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    for options in (
        ["--prefill-size", "16"],
        ["--capture"],
        ["--json", "unused"],
        ["--timeout", "60"],
        ["--context-length", "8"],
        ["--golden", "unused"],
        ["--strict-evidence"],
    ):
        with pytest.raises(ValueError, match="require"):
            handle_generate(parser.parse_args(["generate", "unused", *options]))


@pytest.mark.parametrize(
    "temperature,top_p,seed", [(-1, 1, 0), (float("nan"), 1, 0), (1, 0, 0), (1, 1.1, 0), (1, 1, -1), (1, 1, 2**64), (1, 1, True)]
)
def test_invalid_sampling_controls(temperature, top_p, seed):
    from emmy.serving.native.client import validate_sampling

    with pytest.raises(ValueError):
        validate_sampling(temperature, top_p, seed)


@pytest.mark.parametrize("timeout", ["0", "-1", "nan", "inf"])
def test_native_cli_rejects_invalid_deadlines(timeout):
    import argparse

    from emmy.commands.generate import handle_generate, register_generate_command

    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    args = parser.parse_args(["generate", "unused", "--native-pack", "artifact", "--timeout", timeout])
    with pytest.raises(ValueError, match="finite and positive"):
        handle_generate(args)


def test_qwen3_5_text_model_is_accepted_and_its_decoder_returned():
    """Partial rotary embedding and Gated DeltaNet layers are this family's shape, not a rejection;
    the rotated part of a head must still be even, as the rotation pairs its halves."""
    model = qwen3_5_model().half()
    assert validate_model(model, 8) is model.model
    model.config.rope_parameters["partial_rotary_factor"] = 3 / 16
    with pytest.raises(ValueError, match="rotary"):
        validate_model(model, 8)


def test_quantized_checkpoint_loads_as_a_coded_twin(tmp_path):
    """An FP8 checkpoint loads onto the serving runner's checkpoint-sourced lane: the twin passes
    validation with its coded linears as FP16 placeholders, and the ``ckpt`` pair names the
    checkpoint and each parameter's key in it — what the exporter spells and binds from. An
    ordinary checkpoint keeps the module lane. Either loads in the checkpoint's dtype unless one is asked for."""
    import torch

    fp8_block_checkpoint(qwen3_model(2).half(), tmp_path / "fp8")
    model, (directory, id_to_key) = load_model(str(tmp_path / "fp8"))
    assert validate_model(model, 8) is model.model
    assert directory == str(tmp_path / "fp8")
    assert id_to_key[id(model.model.layers[0].self_attn.q_proj.weight)] == "model.layers.0.self_attn.q_proj.weight"

    qwen3_model(2).half().save_pretrained(tmp_path / "plain")
    model, ckpt = load_model(str(tmp_path / "plain"))
    assert ckpt is None and validate_model(model, 8) is model.model

    fp8_block_checkpoint(qwen3_model(2).bfloat16(), tmp_path / "fp8-bf16")
    qwen3_model(2).bfloat16().save_pretrained(tmp_path / "bf16")
    for path in ("fp8-bf16", "bf16"):
        model, ckpt = load_model(str(tmp_path / path))
        assert model.dtype == torch.bfloat16 and validate_model(model, 8, ckpt) is model.model
        assert load_model(str(tmp_path / path), dtype="float16")[0].dtype == torch.float16
    qwen3_model(2).float().save_pretrained(tmp_path / "fp32")
    assert load_model(str(tmp_path / "fp32"))[0].dtype == torch.float16, "auto serves an FP32 checkpoint in FP16, as vLLM does"


def test_fp8_moe_checkpoint_streams_its_routed_experts(tmp_path):
    """An FP8 MoE checkpoint's twin leaves its routed experts unread on the meta device, which only the
    checkpoint-sourced lane admits; the export reads them a layer at a time as the plugin's fixed-slot
    program takes them: E-stacked e4m3 codes and their block scales, which decode to the checkpoint's
    experts."""
    from emmy.compiler.loader.quant import decode_f8, dequantize
    from emmy.serving.native.prepare import checkpoint_experts

    model = qwen3_5_moe_model().half()
    fp8_block_checkpoint(model, tmp_path / "fp8")
    twin, ckpt = load_model(str(tmp_path / "fp8"))
    assert validate_model(twin, 8, ckpt) is twin.model
    with pytest.raises(ValueError, match="FP16"):
        validate_model(twin, 8)
    for index, layer in enumerate(model.model.layers):
        assert twin.model.layers[index].mlp.experts.gate_up_proj.is_meta
        fmt, inputs = checkpoint_experts(ckpt[0], twin.dtype, index)
        assert fmt == "f8e4m3"
        for name, expected in (("w_gate_up", layer.mlp.experts.gate_up_proj), ("w_down", layer.mlp.experts.down_proj)):
            codes, scales = inputs[name].numpy(), inputs[f"{name}_scale"].numpy()
            values = np.stack([dequantize(decode_f8(c, "f8e4m3"), s) for c, s in zip(codes, scales, strict=True)])
            np.testing.assert_array_equal(values.astype(np.float16), expected.detach().numpy())


def test_router_picks_and_scores_as_transformers_does():
    """The router program's ranks, counted per expert and scattered to their slots, pick Transformers'
    top-k experts in its order and score them as it does, eager and through the traced graph; a chunk's
    rows also come out one per buffer for the per-row fixed-slot launches. A tie goes to the lowest
    expert ID, and a NaN row picks the last expert in every slot, so its gather stays inside the stack,
    while the other rows keep their picks."""
    import torch

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.trace.torch import trace_module_with_constants
    from emmy.serving.native.prepare import router_module

    model = qwen3_5_moe_model().half()
    assert validate_model(model, 8) is model.model
    gate, rows = model.model.layers[1].mlp.gate, 3
    xn = torch.randn(rows, 64).half()
    with torch.no_grad():
        _, scores, picks = gate(xn)
        module = router_module(gate, rows)
        eager = module(xn)
    expected = (scores, *picks, *(xn[r : r + 1] for r in range(rows)))
    assert len(eager) == len(expected)
    for actual, value in zip(eager, expected, strict=True):
        torch.testing.assert_close(actual, value, rtol=0, atol=0)
    graph, targets = trace_module_with_constants(module, (xn,))
    tensors = dict(module.named_parameters())
    inputs = {graph.inputs[0]: xn.numpy(), **{name: tensors[path].detach().numpy() for name, path in targets.items()}}
    result, _ = NumpyBackend().run(graph, input_data=inputs)
    for name, value in zip(graph.outputs, expected, strict=True):
        np.testing.assert_allclose(result.outputs[name], value.numpy(), rtol=1e-3, atol=1e-3)
    # A NaN row picks the last expert in every slot, so the slot program's gather stays inside the stack;
    # the other rows keep their picks.
    keep = [[int(e) for e in row] for row in expected[1 : 1 + rows]]
    keep[1] = [gate.num_experts - 1] * gate.top_k
    xn[1, 5] = float("nan")
    with torch.no_grad():
        eager = module(xn)
    result, _ = NumpyBackend().run(graph, input_data=inputs | {graph.inputs[0]: xn.numpy()})
    for picks in (eager[1 : 1 + rows], [result.outputs[name] for name in graph.outputs[1 : 1 + rows]]):
        assert [[int(e) for e in row] for row in picks] == keep
    # Equal probabilities rank by ID: the lowest IDs win every row.
    xn[1, 5] = 0.0
    with torch.no_grad():
        gate.weight.zero_()
        eager = module(xn)
    weights = {name: tensors[path].detach().numpy() for name, path in targets.items()}
    result, _ = NumpyBackend().run(graph, input_data={graph.inputs[0]: xn.numpy(), **weights})
    for picks in (eager[1 : 1 + rows], [result.outputs[name] for name in graph.outputs[1 : 1 + rows]]):
        assert [[int(e) for e in row] for row in picks] == [list(range(gate.top_k))] * rows


@pytest.mark.parametrize("rows", [1, 3])
def test_routed_gdn_layer_leaves_its_fp32_residual_and_fp16_rows(rows):
    """A Gated DeltaNet MoE layer's program, as native generation traces it from an FP16 model: the
    FP32 residual with the shared expert folded in, the FP16 normalized rows the router reads, and
    the next state and history, eager and through the traced graph."""
    import torch

    from emmy.compiler.backend.numpy import NumpyBackend
    from emmy.compiler.dtype import F16, F32
    from emmy.compiler.trace.torch import trace_module_with_constants
    from emmy.serving.native.prepare import gdn_module

    layer = qwen3_5_moe_model().half().model.layers[0]
    mixer = layer.linear_attn
    module = gdn_module(layer)
    examples = (
        torch.randn(rows, 64),
        torch.randn(1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim) * 0.1,
        (torch.randn(1, mixer.conv_dim, mixer.conv_kernel_size) * 0.1).half(),
    )
    with torch.no_grad():
        expected = module(*examples)
    graph, targets = trace_module_with_constants(module, examples)
    assert [graph.buffer(name).dtype for name in graph.outputs] == [F32, F16, F32, F16]
    tensors = dict(module.named_parameters()) | dict(module.named_buffers())
    inputs = {name: value.numpy() for name, value in zip(graph.inputs, examples, strict=True)}
    inputs.update({name: tensors[path].detach().numpy() for name, path in targets.items()})
    result, _ = NumpyBackend().run(graph, input_data=inputs)
    for name, value in zip(graph.outputs, expected, strict=True):
        np.testing.assert_allclose(result.outputs[name].astype(np.float32), value.float().numpy(), rtol=2e-3, atol=2e-5)


@pytest.mark.parametrize("backend", ["loop", pytest.param("cuda", marks=requires_cuda)])
@pytest.mark.parametrize("value", [0.0, np.inf, -np.inf, np.nan], ids=["finite", "inf", "-inf", "nan"])
def test_greedy_token_is_the_lowest_peak_and_no_token_on_a_nonfinite_logit(backend, value):
    """The greedy module returns the lowest token ID among the maximum logits, and the vocabulary
    size, which the runtime rejects, when any logit is nonfinite — eager and compiled alike. The
    compiled maximum skips NaN (``fmaxf``), so a NaN must not leave a finite peak behind."""
    import torch

    from emmy.compiler.ir.base import ConstantOp
    from emmy.compiler.loader.binder import evaluate_source_graph
    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.native.prepare import greedy_module

    vocab = 8
    logits = np.array([[0.5, -2.0, 1.0, 0.25, -1.0, 1.0, value, 0.75]], np.float32)
    expected = 2 if np.isfinite(value) else vocab
    module = greedy_module(vocab)
    assert module(torch.from_numpy(logits)).tolist() == [expected]

    graph = trace_module(module, (torch.zeros(1, vocab, dtype=torch.float32),))
    if backend == "loop":
        from emmy.compiler.backend.loop import LoopBackend as Backend
    else:
        from emmy.compiler.backend.cuda.backend import CudaBackend as Backend
    compiled = Backend().compile(graph)
    # The folded token IDs are a generated constant the plain interpreter does not evaluate itself.
    data = {"logits": logits}
    for nid, node in compiled.nodes.items():
        if isinstance(node.op, ConstantOp) and node.op.source_graph is not None:
            data[nid] = evaluate_source_graph(node.op.source_graph, {})
    (token,) = Backend().run(compiled, input_data=data)[0].outputs.values()
    assert token.tolist() == [expected]


def test_the_expert_budget_pages_a_layer_only_below_its_expert_count(monkeypatch):
    from emmy import config

    monkeypatch.delenv("EMMY_GEN_EXPERTS_GPU", raising=False)
    assert config.gen_expert_slots(8) is None
    for budget, slots in (("4", 4), ("8", None)):
        monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", budget)
        assert config.gen_expert_slots(8) == slots
    monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", "0")
    with pytest.raises(ValueError, match="at least one expert"):
        config.gen_expert_slots(8)


def test_native_programs_are_the_twins_a_native_serving_config_adds(tmp_path):
    """The programs an export compiles, traced and not compiled, one per kind, width and shape, in the model dtype:
    the twins ``SERVE_NATIVE`` adds to a serving config, each reached in the config's lane at its own width."""
    import torch

    from emmy.compiler.dtype import BF16
    from emmy.serving.native.prepare import _Step, trace_programs
    from emmy.serving.release import load_serving_config
    from emmy.serving.twins import twin_realizations

    twins = trace_programs(qwen3_5_moe_model().to(torch.bfloat16), context_length=8, prefill_size=3)
    # The last prefill layer, an attention layer, only writes its cache; every row of a chunk runs the one-row experts.
    decode = ["embed", "gdn", "router", "experts", "combine", "pre", "rope", "attend", "post", "head", "greedy"]
    assert sorted(twins) == sorted(
        [f"{k}1-native" for k in decode] + [f"{k}3-native" for k in ("embed", "gdn", "router", "combine", "pre", "rope")]
    )
    router = twins["router1-native"]
    assert router.buffer(router.inputs[0]).dtype == BF16
    # Tracing binds no weights: a traced step never calls for a buffer's host values.
    _Step(BF16, 1, traced={}).buffer("weights", (2,), data=lambda: pytest.fail("a traced step binds nothing"))
    env = tmp_path / "serving.env"
    env.write_text(
        "SERVE_MODEL=org/model\nSERVE_GPU=gpu\nSERVE_GOLDEN_FILE=golden.json\nSERVE_MAX_NUM_BATCHED_TOKENS=64\nSERVE_DECODE_BUCKET=4\n"
        "SERVE_M1_TIER=0\nSERVE_WARM_SHAPES=:::fm\nSERVE_NATIVE=1\n"
    )
    serving = load_serving_config(env)
    assert serving.native and serving.experts_gpu is None and serving.dtype is None
    # Native compiles in one regime, its golden's, so a fast-math lane adds no native row.
    for name, width in (("router3-native", 3), ("head1-native", 1), ("experts1-native", 1), ("router4-native", 4)):
        assert [(row.name, row.bindings) for row in twin_realizations(serving, name)] == [(f"m{width}", (("num_tokens", width),))]


def test_a_native_serving_config_traces_native_twins_as_its_export_would(tmp_path, monkeypatch):
    """``SERVE_NATIVE`` adds native's programs to the twins, traced with the config's expert budget and in the dtype
    ``emmy serve --native`` exports: the checkpoint's when ``--dtype`` names none, where the plugin's twins take FP16."""
    from emmy import config
    from emmy.serving import twins
    from emmy.serving.release import load_serving_config

    seen = {}
    monkeypatch.setattr(twins, "capture_twin_graphs", lambda _model, **kw: seen.update(plugin=kw["dtype"]) or {"pre4": "plugin"})
    monkeypatch.setattr(
        "emmy.serving.native.prepare.load_model", lambda _model, _revision, dtype: seen.update(native=dtype) or (None, None)
    )

    def trace(_model, ckpt=None):
        seen["budget"] = config.gen_experts_gpu()
        return {"router1-native": "native"}

    monkeypatch.setattr("emmy.serving.native.prepare.trace_programs", trace)
    monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", "7")
    env = tmp_path / "serving.env"
    env.write_text(
        "SERVE_MODEL=org/model\nSERVE_GPU=gpu\nSERVE_GOLDEN_FILE=golden.json\nSERVE_MAX_NUM_BATCHED_TOKENS=64\nSERVE_DECODE_BUCKET=4\n"
        "SERVE_NATIVE=1\nSERVE_EXPERTS_GPU=4\n"
    )
    serving = load_serving_config(env)
    assert twins.capture_serving_twins("org/model", serving) == {"pre4": "plugin", "router1-native": "native"}
    assert seen == {"plugin": "float16", "native": "auto", "budget": 4}
    assert config.gen_experts_gpu() == 7, "the budget is the config's for the trace alone"


def _standard_lane_golden(path):
    """A golden whose one row was recorded with fast math off."""
    import torch

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden.working import write_trace_inventories
    from emmy.compiler.trace.torch import trace_module

    realization = {"name": "std", "bindings": {}, "pins": {"FAST_MATH": False}}
    graph = trace_module(torch.nn.ReLU(), (torch.zeros(4),))
    write_trace_inventories({"relu": graph}, path, ctx=Context.from_target((12, 0)), realizations=[realization])
    return path


def test_an_export_from_a_golden_runs_in_its_rows_regime(monkeypatch, tmp_path):
    """A golden's rows are evidence only in the regime they were recorded in, so an export from one compiles in it,
    as the plugin's boot runs its engine, through ``generate --export-native`` and native serving's preparation alike;
    a hand pin at another value, as ``EMMY_<KNOB>`` or in ``EMMY_KNOBS``, is refused."""
    import argparse
    from types import SimpleNamespace

    from emmy import config
    from emmy.commands.generate import handle_generate, register_generate_command
    from emmy.serving.native import launch

    golden = _standard_lane_golden(tmp_path / "golden.json")
    seen = []
    model = SimpleNamespace(generation_config=SimpleNamespace(eos_token_id=1))
    tokenizer = SimpleNamespace(is_fast=True, chat_template="", backend_tokenizer=SimpleNamespace(save=lambda _path: None))
    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *_a, **_k: tokenizer)
    monkeypatch.setattr("emmy.serving.native.prepare.load_model", lambda *_a: (model, None))
    monkeypatch.setattr("emmy.serving.native.prepare.export_model", lambda *_a, **_k: seen.append(config.knob_raw("FAST_MATH")))
    monkeypatch.setattr("emmy.compiler.backend.gpu_lock.gpu_lock", contextlib.nullcontext)
    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    args = parser.parse_args(["generate", "unused", "--export-native", str(tmp_path / "artifact"), "--golden", str(golden)])
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    monkeypatch.delenv("EMMY_KNOBS", raising=False)
    handle_generate(args)
    launch.prepare("unused", None, tmp_path, 8, str(golden), False)
    assert seen == ["False", "False"] and config.knob_raw("FAST_MATH") is None
    for name, value in (("EMMY_FAST_MATH", "1"), ("EMMY_KNOBS", "FAST_MATH=1")):
        monkeypatch.setenv(name, value)
        with pytest.raises(ValueError, match="FAST_MATH"):
            handle_generate(args)
        with pytest.raises(ValueError, match="FAST_MATH"):
            launch.prepare("unused", None, tmp_path, 8, str(golden), False)
        monkeypatch.delenv(name)
