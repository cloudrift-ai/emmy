"""What the CUDA lowering keeps of a kernel's PTX: its identity without its name, the plan it round-trips through, and
the pack's key. GPU-free but for the pack's key."""

from __future__ import annotations

import pytest

from emmy.compiler.backend.cuda import ptxas
from emmy.compiler.backend.plan import plan_from_dict, plan_from_graph, plan_to_dict
from emmy.compiler.context import Context
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.cuda.ir import renamed
from emmy.compiler.ir.kernel import ptx
from emmy.compiler.ir.tensor.ir import ElementwiseOp
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from tests.compiler.helpers import requires_cuda
from tests.compiler.realization.helpers import CASES_DIR, load_case, lowered

COVERED = "reduce/combine-amax-coop-warp.json"


def _sources(path: str) -> tuple[dict[str, CudaOp], object]:
    """The kernels ``path`` lowers to, and its graph."""
    case = load_case(CASES_DIR / path)
    graph, _taken = lowered(case, case.context())
    return {node.op.kernel_name: node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)}, graph


def test_an_error_of_the_compiler_is_raised(monkeypatch):
    def broken(kernel, interface, **decisions):  # noqa: ANN001, ANN003, ANN202 — the compiler entry's signature
        raise ptx.NoRule(f"no PTX rule for the test's missing rule: `{kernel.name}`")

    monkeypatch.setattr(ptx, "compile_kernel", broken)
    with pytest.raises(ptx.NoRule, match="no PTX rule for"):
        _sources(COVERED)


def test_a_compile_with_no_card_asks_for_one():
    """A context no device answered for and no target named has no card to write PTX for: the compile says to name
    one, and writes no ``.target sm_0``."""
    graph = Graph()
    graph.add_node(op=InputOp(), inputs=[], output=Tensor("x", (64,)), node_id="x")
    graph.add_node(op=ElementwiseOp("negative"), inputs=["x"], output=Tensor("y", (64,)), node_id="y")
    graph.inputs, graph.outputs = ["x"], ["y"]
    with pytest.raises(ValueError, match="no target was named; name one \\(--target sm_NN"):
        Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.from_target(ptx.NO_TARGET))


def test_a_ptx_kernel_keeps_its_name_out_of_its_identity_and_round_trips_in_a_plan():
    sources, graph = _sources(COVERED)
    [(name, op)] = sources.items()
    text = op.kernel_source
    assert text.startswith(".version") and f".entry {name}(" in text
    other = "k_renamed"
    moved = CudaOp(**{**op.__dict__, "kernel_source": renamed(text, name, other), "kernel_name": other})
    assert f".entry {other}(" in moved.kernel_source and f"{other}_param_0" in moved.kernel_source
    assert moved.identity_key() == op.identity_key()
    plan = plan_from_graph(graph)
    assert plan_to_dict(plan_from_dict(plan_to_dict(plan))) == plan_to_dict(plan)
    assert plan.kernels[name].source == text
    # A kernel for a card with no wait on the grid ahead says so, and is launched serialized.
    assert plan.kernels[name].dependent_launch == (load_case(CASES_DIR / COVERED).compute_cap >= ptx.GRID_WAIT_CAPABILITY)


def test_a_rename_touches_the_entry_and_its_symbols_alone():
    text = ".visible .entry k_a(\n\t.param .u64 k_a_param_0\n)\n// k_a_b\n\tld.param.u64 %rd1, [k_a_param_0];\n"
    assert renamed(text, "k_a", "k_c") == text.replace("k_a(", "k_c(").replace("k_a_param", "k_c_param")


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_pack_is_keyed_on_the_ptxas_version():
    from emmy.compiler.backend.pack import _environment

    environment = _environment()
    assert environment["ptxas"] == ptxas.version()
    assert "toolkit" not in environment and "nvcc_flags" not in environment
