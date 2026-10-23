"""Native preparation accepts Qwen3 and Qwen3.5 text models, dense or MoE, and rejects the rest before CUDA work."""

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
    ordinary checkpoint keeps the module lane."""
    fp8_block_checkpoint(qwen3_model(2).half(), tmp_path / "fp8")
    model, (directory, id_to_key) = load_model(str(tmp_path / "fp8"))
    assert validate_model(model, 8) is model.model
    assert directory == str(tmp_path / "fp8")
    assert id_to_key[id(model.model.layers[0].self_attn.q_proj.weight)] == "model.layers.0.self_attn.q_proj.weight"

    qwen3_model(2).half().save_pretrained(tmp_path / "plain")
    model, ckpt = load_model(str(tmp_path / "plain"))
    assert ckpt is None and validate_model(model, 8) is model.model


def test_router_picks_and_scores_as_transformers_does():
    """The router program's ``k`` masked maximum reductions pick Transformers' top-k experts in its
    order and score them as it does, eager and through the traced graph; a chunk's rows also come
    out one per buffer for the per-row fixed-slot launches, and a NaN row still picks experts."""
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
    # A NaN row has no element at its peak: its picks must still name experts, or the slot program's
    # gather would read past the stack.
    xn[1, 5] = float("nan")
    with torch.no_grad():
        eager = module(xn)
    result, _ = NumpyBackend().run(graph, input_data=inputs | {graph.inputs[0]: xn.numpy()})
    for picks in (eager[1 : 1 + rows], [result.outputs[name] for name in graph.outputs[1 : 1 + rows]]):
        assert all(0 <= int(e) < gate.num_experts for row in picks for e in row)


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
