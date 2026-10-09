"""Count-aware GDN staging preserves carried state across short chunks."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from emmy.compiler.trace.huggingface import build_gdn_capacity_wrapper, build_gdn_state_wrapper
from emmy.serving.gen_runner import EmmyGenRunner, trace_split
from tests.serving.helpers import QWEN3_5_TINY


@pytest.fixture
def block():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5DecoderLayer

    torch.manual_seed(43)
    return Qwen3_5DecoderLayer(Qwen3_5TextConfig(**QWEN3_5_TINY), 0).eval()


@pytest.mark.parametrize("count", [1, 2, 20, 63, 64, 65, 84])
@pytest.mark.parametrize("continued", [False, True])
def test_capacity_staging_matches_installed_block(block, count, continued):
    mixer = block.linear_attn
    x = torch.randn(count, block.input_layernorm.weight.numel()) * 0.1
    state = torch.randn(1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim) * (0.1 if continued else 0)
    history = torch.randn(1, mixer.conv_dim, mixer.conv_kernel_size) * (0.1 if continued else 0)
    old_state, old_history = state.clone(), history.clone()
    calls = []

    class EagerProgram:
        def __init__(self, wrapper):
            self.wrapper = wrapper

        def run_device(self, inputs, *, out):
            calls.append((inputs[0].shape[1], inputs[-1].item() if len(inputs) == 4 else 1))
            result = self.wrapper(*inputs)
            for target, source in zip(out, result, strict=True):
                target.copy_(source)

    runner = SimpleNamespace(_gdn=[{1: EagerProgram(build_gdn_state_wrapper(block)), 64: EagerProgram(build_gdn_capacity_wrapper(block))}])
    with torch.no_grad():
        expected = build_gdn_state_wrapper(block)(x[None], old_state, old_history)
        actual = EmmyGenRunner.forward_layer_gdn_device(runner, 0, x, state, history)
    torch.testing.assert_close(actual, expected[0][0], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(state, expected[1], atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(history, expected[2], atol=2e-6, rtol=2e-5)
    assert calls == [(1 if min(count - start, 64) == 1 else 64, min(count - start, 64)) for start in range(0, count, 64)]


def test_capacity_trace_keeps_count_as_runtime_input(block):
    mixer = block.linear_attn
    graph = trace_split(
        build_gdn_capacity_wrapper(block),
        [
            torch.zeros(1, 64, 64),
            torch.zeros(1, mixer.num_v_heads, mixer.head_k_dim, mixer.head_v_dim),
            torch.zeros(1, mixer.conv_dim, mixer.conv_kernel_size),
            torch.tensor([20], dtype=torch.int64),
        ],
        None,
    )
    assert [tuple(graph.buffer(key).shape) for key in graph.inputs] == [(1, 64, 64), (1, 4, 16, 16), (1, 128, 4), (1,)]
    assert graph.buffer(graph.inputs[-1]).dtype.name == "i64"
    assert [tuple(graph.buffer(key).shape) for key in graph.outputs] == [(1, 64, 64), (1, 4, 16, 16), (1, 128, 4)]
