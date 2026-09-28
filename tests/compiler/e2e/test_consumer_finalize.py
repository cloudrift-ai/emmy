"""The ``consumer`` finalize of a cross-CTA split (``REDUCE g<n>c``): the partitions' f32 partials are summed
where the next kernel reads the value, so the finalize kernel the ``g<n>k`` arm launches is gone.

The program is two chained GEMVs cut apart at the first one's output; the first piece splits 4 ways."""

from __future__ import annotations

import numpy as np
import torch

from tests.compiler.helpers import requires_cuda

_CODE = (
    "torch.nn.functional.linear(torch.relu(torch.nn.functional.linear(torch.randn(1, 512, dtype=torch.float16), "
    "torch.randn(1024, 512, dtype=torch.float16))), torch.randn(256, 1024, dtype=torch.float16))"
)
_PIECE = "place_efe5c4a75a"


def _compiled(finalize: str):
    from emmy.commands.trace import graph_from_code
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.pipeline.search.pins import pinned_knobs

    # The second GEMV stays unsplit, so the kernel count is the first piece's alone.
    knobs = {"PLACE@inner.2/map": "cut", f"REDUCE@{_PIECE}": f"g4{finalize}", "REDUCE@node_linear_1": "coop-t"}
    knobs["WORK@node_linear_1"] = "t256"
    torch.manual_seed(0)  # both arms trace the same example tensors
    graph, _, (module, args, kwargs) = graph_from_code(_CODE)
    backend = CudaBackend()
    with pinned_knobs(knobs):
        compiled = backend.compile(graph)
    from emmy.commands.run import _bind_inputs

    return backend, compiled, _bind_inputs(compiled, module, args, kwargs, checkpoint=None), module(*args, **kwargs)


def _kernels(compiled) -> list[str]:
    return [node.id for node in compiled.nodes.values() if getattr(node.op, "kernel_source", "")]


@requires_cuda
def test_the_consumer_sums_the_split_partials():
    """The consumer arm launches one kernel fewer: the second GEMV reads the four f32 partials and sums
    them where it reads its operand, in partition order, rounding to f16 where the finalize kernel
    stored. Both arms match eager; they may differ by an f16 step, since the reader is a new kernel
    that may accumulate its own reduce in another order."""
    results = {}
    for finalize, launches in (("k", 3), ("c", 2)):
        backend, compiled, inputs, expected = _compiled(finalize)
        assert len(_kernels(compiled)) == launches, _kernels(compiled)
        (results[finalize],) = backend.run(compiled, input_data=inputs)[0].outputs.values()
    want = expected.float().cpu().numpy()
    for finalize, result in results.items():
        np.testing.assert_allclose(np.asarray(result, dtype=np.float32), want, rtol=0, atol=1e-2 * np.abs(want).max(), err_msg=finalize)
