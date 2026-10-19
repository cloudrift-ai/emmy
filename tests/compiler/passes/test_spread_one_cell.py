"""``lowering/kernel/115_spread_one_cell``: a single-cell kernel's free loops stride over its CTA's threads."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from emmy import config
from tests.compiler.helpers import requires_cuda


class _CountThenSweep(nn.Module):
    """A sum over one axis broadcast into a store over another: one cell, whose sweep is a free loop."""

    def forward(self, x, y):
        count = torch.ones_like(y).sum(0, keepdim=True)
        return torch.where(x < count, x, count)


def _inputs():
    return torch.tensor([5, 40, 1, 30, 2, 9] * 10 + [0, 0, 0, 0, 7]), torch.zeros(33, dtype=torch.int64)


def _compiled():
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.trace.torch import trace_module

    with config.strict_evidence_override(False):
        return CudaBackend().compile(trace_module(_CountThenSweep(), _inputs()))


def test_a_single_cell_kernel_strides_its_free_loop_and_keeps_its_reduce_serial() -> None:
    from emmy.compiler.backend.plan import plan_from_graph

    (spec,) = plan_from_graph(_compiled()).kernels.values()
    assert "a1 = threadIdx.x; a1 < 65; a1 += blockDim.x" in spec.source
    assert "a0 = 0; a0 < 33; a0++" in spec.source
    assert "_gid < 1" not in spec.source


@requires_cuda
def test_a_spread_kernel_computes_what_the_module_does() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend

    x, y = _inputs()
    result, _ = CudaBackend().run(_compiled(), input_data={"x": x.numpy(), "y": y.numpy()})
    (got,) = result.outputs.values()
    np.testing.assert_array_equal(got.reshape(-1), _CountThenSweep()(x, y).numpy())
