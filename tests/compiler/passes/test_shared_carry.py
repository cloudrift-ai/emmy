"""A shared carried state preserves old cells across every chunk of an ordered update."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32, F64
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, TernaryExpr, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Loop, Pre, Select, SelectBranch, Write
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

shared = import_module("emmy.compiler.pipeline.passes.lowering.kernel.035_shared_carry")
_PINS = {"PLACE": "fuse", "WORK": "t512", "REDUCE": "coop-t/n8/v4", "FAST_MATH": True}


def _body(n=40, *, masked=True, extra_reader=False):
    t, b, r, c, k = (Var(name) for name in ("t", "b", "r", "c", "k"))
    row = t + Literal(n - 5, "int")
    predicate = BinaryExpr("&&", BinaryExpr("&&", r.lt(row + Literal(1, "int")), BinaryExpr("<=", row, r)), c.lt(row))
    owner = TernaryExpr(predicate, b, Literal(0, "int")) if masked else b
    reduction = Loop(
        Axis("k", n),
        (
            Pre("left", "state", (owner, row if masked else r, k)),
            Pre("right", "state", (owner, k, c)),
            Assign("product", "multiply", ("left", "right")),
            Accum(name="sum", op="add", value="product"),
        ),
    )
    value = (
        Select("next", (SelectBranch("updated", predicate), SelectBranch("old", Literal(True, "bool"))))
        if masked
        else Assign("next", "copy", ("updated",))
    )
    carry = Carry("state", "next", (b, r, c), "seed")
    cell = (
        reduction,
        Pre("old", "state", (b, r, c)),
        Assign("updated", "add", ("old", "sum")),
        value,
        carry,
        Write("out", (t, b, r, c), "next"),
    )
    if extra_reader:
        cell += (Write("other", (t, b, r, c), "sum"),)
    body = Body((Loop(Axis("t", 3), (Loop(Axis("b", 2), (Loop(Axis("r", n), (Loop(Axis("c", n), cell),)),)),)),))
    return body, carry


def _graph(dtype=F32, *, masked=True, n=40):
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("seed", (2, n, n), dtype), node_id="seed")
    graph.add_node(LoopOp(body=_body(n, masked=masked)[0], name="k_step"), ["seed"], Tensor("out", (3, 2, n, n), dtype), node_id="out")
    graph.inputs, graph.outputs = ["seed"], ["out"]
    return graph


def test_discarded_clamps_do_not_change_the_state_owner():
    body, carry = _body()
    assert shared._owned(body, carry) == ("b",)
    body, carry = _body(extra_reader=True)
    assert shared._owned(body, carry) == (), "an unmasked reader can observe the other batch"


def test_shared_storage_keeps_the_ordered_loop_inside_one_launch():
    with pinned_knobs({**_PINS, "SHARED_CARRY": 2}):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target((7, 0)))
    (op,) = (node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp))
    assert not op.serial
    assert op.arg_order == ("seed", "out")
    assert op.knobs["SHARED_CARRY"] == 2
    assert "_carry_copy" in op.kernel_source
    assert "__acc" not in op.kernel_source
    assert len(compiled.nodes["out"].outputs) == 1


def test_shared_storage_requires_room_for_both_states_and_the_combine():
    target = replace(Context.from_target((7, 0)), max_dynamic_smem=10000)
    with pinned_knobs({**_PINS, "SHARED_CARRY": 2}):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=target)
    (op,) = (node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp))
    assert op.serial and not op.knobs.get("SHARED_CARRY")


def test_recorded_shared_storage_is_picked_again_without_a_pin(tmp_path):
    from emmy import config
    from emmy.compiler.pipeline.search.golden import GoldenFile, record_greedy_pick, sole_evidence
    from tests.compiler.helpers import inventory_document

    card = "Tesla V100-SXM2-16GB"
    document = inventory_document(_graph(), (7, 0), gpu_name=card)
    path = tmp_path / "state.json"
    document.dump(path)
    target = document.targets()[0]
    with pinned_knobs({**_PINS, "SHARED_CARRY": 2}):
        compiled = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((7, 0), gpu_name=card), db=None)
        nodes = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
        record_greedy_pick(
            path, document.rows[0].name, decisions=[], kernels=[(n.op, 1.0, 1.0) for n in nodes], reference_backend="same-input-greedy"
        )
    measured = GoldenFile.load(path)
    assert measured.rows[-1].knobs["SHARED_CARRY"] == "2"
    with sole_evidence([measured]), config.strict_evidence_override(True):
        replay = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((7, 0), gpu_name=card), db=None)
    [op] = [n.op for n in replay.nodes.values() if isinstance(n.op, CudaOp)]
    assert op.knobs["SHARED_CARRY"] == 2 and not op.serial


@requires_cuda
def test_an_exposed_state_port_keeps_every_snapshot():
    graph = Pipeline.build(["tile/lift"]).run(_graph(), ctx=Context.probe())
    (port,) = (t.name for t in graph.nodes["out"].outputs if t.name != "out")
    graph.outputs.append(port)
    with pinned_knobs({**_PINS, "SHARED_CARRY": 2}):
        compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe())
    array = (np.random.default_rng(0).standard_normal((2, 40, 40)) * 0.03).astype(np.float32)
    outputs = CudaBackend().run(compiled, input_data={"seed": array})[0].outputs
    np.testing.assert_array_equal(outputs["out"], outputs[port])
    assert np.isfinite(outputs[port]).all()


@requires_cuda
def test_parallel_snapshot_copy_keeps_intermediate_casts():
    graph = _graph()

    def cast_snapshot(body):
        stmts = []
        for stmt in body:
            if isinstance(stmt, Write):
                stmts.append(Assign("narrowed", "copy", (stmt.value,), dtype=F16))
                stmt = replace(stmt, values=("narrowed",))
            elif stmt.nested():
                stmt = stmt.with_bodies(tuple(cast_snapshot(b) for b in stmt.nested()))
            stmts.append(stmt)
        return Body(stmts)

    graph.nodes["out"].op = replace(graph.nodes["out"].op, body=cast_snapshot(graph.nodes["out"].op.body))
    array = (np.random.default_rng(0).standard_normal((2, 40, 40)) * 0.03).astype(np.float32)
    outputs = []
    for enabled in (0, 2):
        with pinned_knobs({**_PINS, "SHARED_CARRY": enabled}):
            compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe())
        outputs.append(CudaBackend().run(compiled, input_data={"seed": array})[0].outputs["out"])
    np.testing.assert_array_equal(*outputs)
    np.testing.assert_array_equal(outputs[0], outputs[0].astype(np.float16).astype(np.float32))


@requires_cuda
def test_shared_state_does_not_round_the_seed_before_its_first_use():
    graph = _graph(F64)
    array = np.random.default_rng(0).standard_normal((2, 40, 40)) * 0.03
    outputs = []
    for storage in (0, 2):
        with pinned_knobs({**_PINS, "SHARED_CARRY": storage}):
            compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe())
        outputs.append(CudaBackend().run(compiled, input_data={"seed": array})[0].outputs["out"])
    np.testing.assert_array_equal(*outputs)


@requires_cuda
@pytest.mark.parametrize("storage", [1, 2], ids=["dense", "padded"])
@pytest.mark.parametrize("dtype", [F32, F16], ids=["f32", "f16"])
@pytest.mark.parametrize("masked", [True, False], ids=["selected", "full"])
def test_shared_state_matches_the_global_state_on_the_same_inputs(dtype, masked, storage):
    graph = _graph(dtype, masked=masked)
    backend = CudaBackend()
    candidates = []
    for enabled in (0, storage):
        with pinned_knobs({**_PINS, "SHARED_CARRY": enabled}):
            candidates.append(Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe()))
    for seed in (0, 1):
        array = (np.random.default_rng(seed).standard_normal((2, 40, 40)) * 0.03).astype(dtype.np)
        outputs = [backend.run(program, input_data={"seed": array})[0].outputs["out"] for program in candidates]
        np.testing.assert_array_equal(*outputs)
