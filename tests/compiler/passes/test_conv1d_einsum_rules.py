"""Conv1d, conv2d, conv_transpose1d and einsum reach the tensor dialect with torch's numbers.

Both ops used to stop the tracer dead — ``aten.conv1d`` had no mapping at all, and
``aten.einsum`` fell into the elementwise fallback and failed to broadcast. They are the
two operations Qwen3.8's Gated DeltaNet layers open with, so the interesting cases here
are that model's: a depthwise convolution with a causal left pad, and a batched
contraction.

The weights are graph inputs rather than module parameters so the comparison feeds them
directly instead of going through constant rebinding, which is a different mechanism and
already has its own tests.
"""

from __future__ import annotations

import numpy as np
import pytest


def _decompose(module, inputs):
    """Trace ``module`` and run the frontend decomposition, returning the tensor-dialect graph."""
    from emmy.compiler.pipeline import TENSOR_PASSES, Pipeline
    from emmy.compiler.trace.torch import trace_module

    return Pipeline.build(TENSOR_PASSES).run(trace_module(module, inputs))


def _assert_matches_eager(run_graph, module, tensors, *, tol=2e-6):
    """The decomposed graph and eager torch agree on the same inputs."""
    import torch

    graph = _decompose(module, tensors)
    with torch.no_grad():
        expected = module(*tensors).numpy()
    feed = {name: tensor.numpy() for name, tensor in zip(graph.inputs, tensors, strict=True)}
    outputs = run_graph(graph, feed)
    got = np.asarray(next(iter(outputs.values()))).reshape(expected.shape)
    np.testing.assert_allclose(got, expected, rtol=tol, atol=tol)


@pytest.fixture
def conv_module():
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class Conv(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            self.kwargs = kwargs

        def forward(self, x, w):
            return F.conv1d(x, w, **self.kwargs)

    return Conv


def test_depthwise_conv1d_matches_eager(run_graph, conv_module) -> None:
    """Qwen3.8's linear-attention convolution: depthwise, kernel 4, causal left pad."""
    import torch

    torch.manual_seed(0)
    # Batch 2 on purpose: a batch of 1 hides an index map that wrongly reads the batch
    # coordinate when indexing the per-channel weight.
    x, w = torch.randn(2, 8, 16), torch.randn(8, 1, 4)
    _assert_matches_eager(run_graph, conv_module(groups=8, padding=3), (x, w))


@pytest.mark.parametrize("padding", [(0, 0), (1, 0)])
def test_causal_conv1d_with_chunk_pad_matches_eager(run_graph, padding) -> None:
    """The causal convolution preserves values across empty and nonempty chunk padding."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class CausalConv(nn.Module):
        def forward(self, x, w):
            return F.pad(F.conv1d(x, w, padding=3, groups=8), padding)

    torch.manual_seed(0)
    x, w = torch.randn(2, 8, 16), torch.randn(8, 1, 4)
    _assert_matches_eager(run_graph, CausalConv(), (x, w))


def test_dense_conv1d_im2col_matches_eager(run_graph, conv_module) -> None:
    """The im2col form, with the stride and dilation that make the window map non-trivial."""
    import torch

    torch.manual_seed(0)
    x, w = torch.randn(2, 5, 16), torch.randn(7, 5, 3)
    _assert_matches_eager(run_graph, conv_module(stride=2, padding=1, dilation=2), (x, w))


def test_conv1d_rejects_a_grouped_convolution(conv_module) -> None:
    """Neither form covers 1 < groups < C_in, and the tracer says so instead of guessing."""
    import torch

    x, w = torch.randn(1, 8, 16), torch.randn(8, 2, 4)
    with pytest.raises(NotImplementedError, match="groups=4"):
        _decompose(conv_module(groups=4), (x, w))


@pytest.mark.parametrize(
    ("taps", "stride", "padding", "output_padding"),
    [
        (16, 8, 0, 0),  # Qwen3-Omni code2wav's 8x upsampler: kernel twice the stride
        (2, 2, 0, 0),  # its 2x upsampler: kernel equal to the stride, phases never overlap
        (6, 3, 2, 1),  # padding crops both ends, output padding extends the right one
    ],
)
def test_conv_transpose1d_matches_eager(run_graph, taps, stride, padding, output_padding) -> None:
    """The polyphase GEMM and its interleave give torch's numbers, bias included."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class ConvTranspose(nn.Module):
        def forward(self, x, w, b):
            return F.conv_transpose1d(x, w, b, stride=stride, padding=padding, output_padding=output_padding)

    torch.manual_seed(0)
    x, w, b = torch.randn(2, 5, 9), torch.randn(5, 7, taps), torch.randn(7)
    _assert_matches_eager(run_graph, ConvTranspose(), (x, w, b), tol=1e-5)


def test_conv_transpose1d_rejects_groups() -> None:
    """A grouped transposed convolution has no decomposition, and the tracer says so."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class Grouped(nn.Module):
        def forward(self, x, w):
            return F.conv_transpose1d(x, w, stride=2, groups=2)

    with pytest.raises(NotImplementedError, match="groups=2"):
        _decompose(Grouped(), (torch.randn(1, 4, 8), torch.randn(4, 3, 4)))


@pytest.mark.parametrize(
    ("kwargs", "w_shape"),
    [
        ({"stride": 2, "padding": 1}, (6, 3, 3, 3)),  # Qwen3-Omni's audio stem: 3x3, stride 2, padding 1
        ({"stride": (1, 2), "dilation": (2, 1)}, (6, 3, 2, 3)),  # unequal axes, no padding
    ],
)
def test_conv2d_im2col_matches_eager(run_graph, kwargs, w_shape) -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class Conv(nn.Module):
        def forward(self, x, w, b):
            return F.conv2d(x, w, b, **kwargs)

    torch.manual_seed(0)
    x, w, b = torch.randn(2, 3, 9, 8), torch.randn(*w_shape), torch.randn(w_shape[0])
    _assert_matches_eager(run_graph, Conv(), (x, w, b), tol=1e-5)


def test_conv2d_rejects_groups() -> None:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    class Grouped(nn.Module):
        def forward(self, x, w):
            return F.conv2d(x, w, groups=2)

    with pytest.raises(NotImplementedError, match="groups=2"):
        _decompose(Grouped(), (torch.randn(1, 4, 8, 8), torch.randn(4, 2, 3, 3)))


@pytest.mark.parametrize(
    ("x_shape", "w_shape"),
    [
        ((3, 3, 2, 4, 4), (5, 3, 2, 4, 4)),  # a video patch embedding: (C, T, P, P) blocks, conv3d
        ((3, 2, 4, 4), (5, 2, 4, 4)),  # the image form, conv2d
    ],
)
def test_patch_convolution_matches_eager(run_graph, x_shape, w_shape) -> None:
    """A kernel covering its whole input is captured as the linear layer over flattened patches."""
    import torch
    import torch.nn as nn
    import torch.nn.functional as F  # noqa: N812

    conv = F.conv3d if len(w_shape) == 5 else F.conv2d

    class PatchEmbed(nn.Module):
        def forward(self, x, w, b):
            return conv(x, w, b, stride=w_shape[2:])

    torch.manual_seed(0)
    _assert_matches_eager(run_graph, PatchEmbed(), (torch.randn(*x_shape), torch.randn(*w_shape), torch.randn(w_shape[0])), tol=1e-5)


@pytest.fixture
def einsum_module():
    import torch
    import torch.nn as nn

    class Einsum(nn.Module):
        def __init__(self, equation):
            super().__init__()
            self.equation = equation

        def forward(self, a, b):
            return torch.einsum(self.equation, a, b)

    return Einsum


@pytest.mark.parametrize(
    "equation",
    [
        "bij,bjk->bik",  # the plain batched contraction
        "bij,bjk->bki",  # output labels permuted away from the product order
        "bhld,bhdm->bhlm",  # two batch labels, the delta-rule shape
    ],
)
def test_einsum_matches_eager(run_graph, einsum_module, equation) -> None:
    import torch

    torch.manual_seed(0)
    sizes = {"b": 2, "h": 3, "i": 4, "j": 5, "k": 6, "l": 4, "d": 5, "m": 6}
    a_labels, rest = equation.split(",")
    b_labels = rest.split("->")[0]
    a = torch.randn(*[sizes[label] for label in a_labels])
    b = torch.randn(*[sizes[label] for label in b_labels])
    _assert_matches_eager(run_graph, einsum_module(equation), (a, b))


@pytest.mark.parametrize(
    ("equation", "a_shape", "b_shape", "message"),
    [
        ("bii,bij->bj", (2, 5, 5), (2, 5, 6), "repeats a label"),
        ("...ij,...jk->...ik", (2, 4, 5), (2, 5, 6), "ellipsis"),
        ("bij,bjk->b", (2, 4, 5), (2, 5, 6), "free label"),
        ("bij,bjk->bijk", (2, 4, 5), (2, 5, 6), "one contracted"),
    ],
)
def test_einsum_rejects_forms_it_cannot_lower(einsum_module, equation, a_shape, b_shape, message) -> None:
    """A diagonal, an ellipsis, a reduction, or anything needing a reshape is refused by name."""
    import torch

    torch.manual_seed(0)
    a, b = torch.randn(*a_shape), torch.randn(*b_shape)
    with pytest.raises(NotImplementedError, match=message):
        _decompose(einsum_module(equation), (a, b))
