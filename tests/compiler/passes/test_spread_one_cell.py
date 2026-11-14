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
    from emmy.compiler.ir.stmt import Accum, StridedLoop
    from tests.compiler.helpers import recording_kernels, statements, text

    with recording_kernels() as kernels_of:
        compiled = _compiled()
    (kernel,) = kernels_of(compiled).values()
    (sweep,) = statements(kernel, StridedLoop)
    assert (sweep.axis.name, str(sweep.start), str(sweep.step), sweep.axis.extent.as_static()) == ("a1", "thread_idx.x", "block_dim.x", 65)
    # Every thread folds the reduce itself, serially, outside the sweep; no thread is turned away.
    folds = statements(kernel, Accum)
    assert len(folds) == 33 and not any(isinstance(stmt, Accum) for stmt in sweep.body.iter())
    assert "thread_idx.x < 1" not in text(kernel)


@requires_cuda
def test_a_spread_kernel_computes_what_the_module_does() -> None:
    from emmy.compiler.backend.cuda.backend import CudaBackend

    x, y = _inputs()
    result, _ = CudaBackend().run(_compiled(), input_data={"x": x.numpy(), "y": y.numpy()})
    (got,) = result.outputs.values()
    np.testing.assert_array_equal(got.reshape(-1), _CountThenSweep()(x, y).numpy())
