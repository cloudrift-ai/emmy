"""``torch.compile(model, backend="emmy")``: run each Dynamo graph as one emmy-compiled CUDA program.

The program is a pure function. When the graph writes into an input, the program returns the input's new value
as an extra output, and the runtime writes that output straight into the input's memory after the last kernel.

``options`` takes the settings ``emmy compile`` takes: ``tune_db``, ``golden``, ``strict_evidence``, ``knobs``
and ``nvcc_flags``. ``trace`` names a working golden each graph is added to, for ``emmy run --golden PATH --bench``.
"""

from __future__ import annotations

from pathlib import Path

import torch

from emmy import config
from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.backend.cuda.program import CompiledProgram
from emmy.compiler.pipeline.search.golden.working import append_trace_inventory, write_trace_inventory
from emmy.compiler.pipeline.search.pins import pinned_knobs
from emmy.compiler.trace.torch import trace_module_functional

OPTIONS = frozenset({"tune_db", "golden", "strict_evidence", "knobs", "nvcc_flags", "trace"})


def emmy_backend(gm: torch.fx.GraphModule, example_inputs: list, *, options: dict | None = None):
    """The ``torch_dynamo_backends`` entry point."""
    options = options or {}
    if unknown := options.keys() - OPTIONS:
        raise ValueError(f"unknown emmy options {sorted(unknown)}; emmy takes {sorted(OPTIONS)}")
    # Dynamo records each input's shape on its placeholder; a symbolic dimension means other sizes will reach this
    # program, which emmy compiles for one shape. (Dynamo also passes each symbol as an int input, not a tensor.)
    values = [n.meta.get("example_value") for n in gm.graph.nodes if n.op == "placeholder"]
    if not all(isinstance(v, torch.Tensor) and all(isinstance(d, int) for d in v.shape) for v in values):
        raise NotImplementedError("emmy compiles static shapes: use torch.compile(..., dynamic=False), without mark_dynamic")
    graph, writes = trace_module_functional(gm, tuple(example_inputs))
    if "trace" in options:
        write = append_trace_inventory if Path(options["trace"]).exists() else write_trace_inventory
        write(graph.copy(), options["trace"])
    with (  # the build compiles the kernels, so it runs under the same settings as the pipeline
        config.golden_file_override(options.get("golden")),
        config.strict_evidence_override(options.get("strict_evidence")),
        config.nvcc_flags_override(options.get("nvcc_flags")),
        pinned_knobs(options.get("knobs", {})),
    ):
        # The example inputs are the program's input memory, so it allocates and fills none of its own: every call
        # lends it the caller's tensors in their place.
        program = CompiledProgram.build(
            CudaBackend(tune_db=options.get("tune_db", "auto")).compile(graph),
            {name: t.contiguous() for name, t in zip(graph.inputs, example_inputs, strict=True)},
        )

    inputs, shapes = graph.inputs, {b.name: tuple(d.as_static() for d in b.shape) for b in program.plan.buffers}
    dtypes = [n.meta["example_value"].dtype for n in next(n for n in gm.graph.nodes if n.op == "output").args[0]]
    written = {graph.outputs.index(out): graph.inputs.index(name) for out, name in writes.items()}
    returned = {j: name for j, name in enumerate(graph.outputs) if j not in written}  # kernels write these in place
    copied = [graph.outputs[j] for j in written]  # the runtime copies these into their inputs after the last launch

    def run(*args):
        tensors = [a.contiguous() for a in args]
        outs = {j: torch.empty(shapes[name], dtype=dtypes[j], device=args[0].device) for j, name in returned.items()}
        lent = _spans(inputs, tensors) + _spans(returned.values(), outs.values())
        with program.on_torch_stream():
            program.run_each([({}, lent, [], _spans(copied, [tensors[i] for i in written.values()]))])
        for j, i in written.items():
            if args[i] is not tensors[i]:
                args[i].copy_(tensors[i])  # a non-contiguous input took its write in a contiguous copy
            outs[j] = args[i]  # a returned written value is the input itself, as in eager
        return tuple(outs[j] for j in range(len(dtypes)))

    return run


def _spans(names, tensors) -> list[tuple[str, int, int]]:
    return [(name, t.data_ptr(), t.numel() * t.element_size()) for name, t in zip(names, tensors, strict=True)]
