"""Coded output heads (EXL3, NVFP4): eligibility, the head program's values on CPU, and its compiled run on CUDA."""

import numpy as np
import pytest

from emmy.compiler.loader.quant import dequantize_nvfp4, quantize_nvfp4
from emmy.serving.coded_head import Exl3HeadSpec, Nvfp4HeadSpec, head_graph
from tests.compiler.helpers import requires_cuda
from tests.support.checkpoints import exl3_linear_tensors


def _source(*, bits=6, hidden=128, vocab=256, marker="mul1"):
    source = {
        "lm_head.trellis": np.zeros((hidden // 16, vocab // 16, 16 * bits), dtype=np.int16),
        "lm_head.suh": np.ones(hidden, dtype=np.float16),
        "lm_head.svh": np.ones(vocab, dtype=np.float16),
    }
    if marker:
        source[f"lm_head.{marker}"] = np.array(0, dtype=np.int32)
    return source


def test_coded_head_is_architecture_independent_and_keeps_logical_vocab():
    spec = Exl3HeadSpec.from_source(
        _source(vocab=512),
        hidden_size=128,
        vocab_size=500,
        tensor_parallel_size=1,
        tied=False,
    )
    assert spec == Exl3HeadSpec(bits=6, codebook=2, hidden_size=128, stored_vocab_size=512, vocab_size=500)


@pytest.mark.parametrize(
    ("kwargs", "source"),
    [
        ({"tensor_parallel_size": 2}, _source()),
        ({"tied": True}, _source()),
        ({}, _source(marker=None)),
    ],
)
def test_unsupported_coded_head_keeps_decoded_fallback(kwargs, source):
    base = dict(hidden_size=128, vocab_size=256, tensor_parallel_size=1, tied=False)
    assert Exl3HeadSpec.from_source(source, **(base | kwargs)) is None


def test_coded_head_rejects_ambiguous_marker_and_bad_storage():
    source = _source()
    source["lm_head.mcg"] = np.array(0, dtype=np.int32)
    with pytest.raises(ValueError, match="both mcg and mul1"):
        Exl3HeadSpec.from_source(
            source,
            hidden_size=128,
            vocab_size=256,
            tensor_parallel_size=1,
            tied=False,
        )


@requires_cuda
def test_coded_head_matches_decoded_weight_and_keeps_checkpoint_pointers():
    import torch

    from emmy.serving.coded_head import CodedHead

    tensors, decoded = exl3_linear_tensors("lm_head", 256, 128, K=2, cb=2)
    source = {name: tensor.numpy() for name, tensor in tensors.items()}
    spec = Exl3HeadSpec.from_source(
        source,
        hidden_size=128,
        vocab_size=251,
        tensor_parallel_size=1,
        tied=False,
    )
    assert spec is not None
    head = CodedHead(spec, source, capacity=2)
    assert head.program.plan.launches
    assert all("exl3" not in launch.kernel_name.lower() for launch in head.program.plan.launches)
    pointers = {name: head.program.executor.buffer(name)[0] for name in ("lm_head_trellis", "lm_head_suh", "lm_head_svh")}
    x = (torch.randn(2, 128, device="cuda") * 0.1).half()
    got = head(x)
    # ``exl3_linear_tensors`` hands back the reference in checkpoint ``(out, in)`` orientation,
    # so the logical-vocab slice takes the ROW axis and the matmul takes the transpose.
    reference = x.float() @ torch.from_numpy(np.ascontiguousarray(decoded[:251].T)).float().cuda()
    torch.testing.assert_close(got, reference, rtol=2e-2, atol=2e-2)
    assert pointers == {name: head.program.executor.buffer(name)[0] for name in pointers}

    source = _source()
    source["lm_head.svh"] = source["lm_head.svh"].astype(np.float32)
    with pytest.raises(ValueError, match="invalid svh"):
        Exl3HeadSpec.from_source(
            source,
            hidden_size=128,
            vocab_size=256,
            tensor_parallel_size=1,
            tied=False,
        )


def _nvfp4_source(vocab, hidden, seed=0):
    """A synthetic NVFP4 head in checkpoint storage, and the decoded head's bf16 weights as f32."""
    import torch

    rng = np.random.default_rng(seed)
    packed, scale_bits, scale_2 = quantize_nvfp4(rng.normal(scale=0.05, size=(vocab, hidden)).astype(np.float32))
    source = {"lm_head.weight": packed, "lm_head.weight_scale": scale_bits, "lm_head.weight_scale_2": scale_2.reshape(())}
    return source, torch.from_numpy(dequantize_nvfp4(packed, scale_bits, scale_2)).to(torch.bfloat16).float()


def test_nvfp4_head_spec_comes_from_the_config():
    base = dict(hidden_size=128, vocab_size=250, dtype="bfloat16")
    spec = Nvfp4HeadSpec.from_config(tensor_parallel_size=1, tied=False, **base)
    assert spec == Nvfp4HeadSpec(hidden_size=128, vocab_size=250, dtype="bfloat16")
    assert Nvfp4HeadSpec.from_config(tensor_parallel_size=2, tied=False, **base) is None
    assert Nvfp4HeadSpec.from_config(tensor_parallel_size=1, tied=True, **base) is None


def test_nvfp4_head_program_reads_the_codes_and_computes_the_decoded_logits():
    """The program's inputs are the rows and the stored codes, never a dense weight, any number of rows at once, and
    its f32 logits are the decoded head's within f32 rounding: the in-graph decode rounds :func:`dequantize_nvfp4`'s
    exact values once to the model dtype, as the decoded parameter does."""
    import torch

    from emmy.compiler.backend import torch_ref
    from emmy.compiler.loader.binder import evaluate_source_graph

    source, decoded = _nvfp4_source(250, 128)
    spec = Nvfp4HeadSpec(hidden_size=128, vocab_size=250, dtype="bfloat16")
    graph, feed = head_graph(spec), spec.feed(source)
    graph.validate()
    assert graph.inputs == ["x", "lm_head_weight", "lm_head_weight_scale", "lm_head_weight_scale_2"]
    assert [graph.nodes[n].output.dtype.name for n in graph.inputs] == ["bf16", "f4e2m1x2", "f8e4m3", "f32"]
    assert not graph.nodes["x"].output.shape[0].is_static, "the rows are serving's any-width axis"
    assert feed["lm_head_weight"] is source["lm_head.weight"], "the codes bind as stored"
    assert feed["lm_head_weight_scale_2"].shape == (1, 1), "one global scale for the one stored tensor"

    x = torch.randn(3, 128).to(torch.bfloat16)
    feed["x"] = x
    feed["lm_head_weight_f4_pairs"] = evaluate_source_graph(graph.nodes["lm_head_weight_f4_pairs"].op.source_graph, {})
    fn, inputs = torch_ref.build_callable(graph, {k: v if torch.is_tensor(v) else torch.from_numpy(v) for k, v in feed.items()})
    got = fn(*inputs)
    assert got.dtype == torch.float32 and got.shape == (3, 250)
    torch.testing.assert_close(got, x.float() @ decoded.T, rtol=1e-6, atol=1e-6)


def test_nvfp4_head_refuses_codes_that_do_not_store_the_vocab():
    source, _ = _nvfp4_source(256, 128)
    with pytest.raises(ValueError, match=r"NVFP4 lm_head stores codes uint8\(256, 64\).*but the head is \(250, 128\)"):
        Nvfp4HeadSpec(hidden_size=128, vocab_size=250, dtype="bfloat16").feed(source)


@requires_cuda
def test_nvfp4_coded_head_matches_decoded_weight_and_keeps_code_pointers():
    import torch

    from emmy.serving.coded_head import CodedHead

    source, decoded = _nvfp4_source(250, 128)
    head = CodedHead(Nvfp4HeadSpec(hidden_size=128, vocab_size=250, dtype="bfloat16"), source, capacity=3)
    pointers = {name: head.program.executor.buffer(name)[0] for name in ("lm_head_weight", "lm_head_weight_scale")}
    # One row, a full group, and more rows than one group holds.
    for rows in (1, 3, 5):
        x = torch.randn(rows, 128, device="cuda").to(torch.bfloat16)
        got = head(x)
        assert got.dtype == torch.float32 and got.shape == (rows, 250)
        torch.testing.assert_close(got, x.float() @ decoded.cuda().T, rtol=1e-5, atol=1e-5)
    assert pointers == {name: head.program.executor.buffer(name)[0] for name in pointers}
