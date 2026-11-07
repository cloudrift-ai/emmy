"""A shared carried state preserves old cells across every chunk of an ordered update."""

from dataclasses import replace
from importlib import import_module

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32, F64
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel import KernelOp, Sync
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.loop.runner import execute_loop_op_cpp
from emmy.compiler.ir.stmt import Body, Carry, Cond, Load, Write, parse_body
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import requires_cuda

shared = import_module("emmy.compiler.pipeline.passes.lowering.kernel.035_shared_carry")
_PINS = {"PLACE": "fuse", "WORK": "t512", "REDUCE": "coop-t/n8/v4", "FAST_MATH": True}


def _body(n=40, *, masked=True, extra_reader=False):
    row = f"t + {n - 5}"
    predicate = f"r < {row} + 1 && {row} <= r && c < {row}"
    owner = f"{predicate} ? b : 0" if masked else "b"
    cell = "    " * 6
    value = f"next = select:\n{cell}    updated when {predicate}\n{cell}    old when true" if masked else "next = copy(updated)"
    extra = f"\n{cell}other[t, b, r, c] = sum" if extra_reader else ""
    body = parse_body(
        f"""
        for t in 0..3:
            for b in 0..2:
                for r in 0..{n}:
                    for c in 0..{n}:
                        for k in 0..{n}:
                            left = pre state[{owner}, {row if masked else "r"}, k]
                            right = pre state[{owner}, k, c]
                            product = multiply(left, right)
                            sum <- add(sum, product)
                        old = pre state[b, r, c]
                        updated = add(old, sum)
                        {value}
                        state[b, r, c] <- next seed `seed`
                        out[t, b, r, c] = next{extra}
        """
    )
    carry = next(stmt for stmt in body.iter() if isinstance(stmt, Carry))
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
    with pinned_knobs({**_PINS, "SHARED_CARRY": 1}):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target((7, 0)))
    (op,) = (node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp))
    assert not op.serial
    assert op.arg_order == ("seed", "out")
    assert op.knobs["SHARED_CARRY"] == 1
    assert "_carry_copy" in op.kernel_source
    assert "__acc" not in op.kernel_source
    assert len(compiled.nodes["out"].outputs) == 1


def test_shared_storage_requires_room_for_both_states_and_the_combine():
    target = replace(Context.from_target((7, 0)), max_dynamic_smem=10000)
    with pinned_knobs({**_PINS, "SHARED_CARRY": 1}):
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
    with pinned_knobs({**_PINS, "SHARED_CARRY": 1}):
        compiled = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((7, 0), gpu_name=card), db=None)
        nodes = [n for n in compiled.nodes.values() if isinstance(n.op, CudaOp)]
        record_greedy_pick(
            path,
            document.rows[0].name,
            decisions=[],
            kernels=[(n.op, 1.0, 1.0, None) for n in nodes],
            reference_backend="same-input-greedy",
        )
    measured = GoldenFile.load(path)
    assert measured.rows[-1].knobs["SHARED_CARRY"] == "1"
    with sole_evidence([measured]), config.strict_evidence_override(True):
        replay = Pipeline.build(CUDA_PASSES).run(target.program({}), ctx=Context.from_target((7, 0), gpu_name=card), db=None)
    [op] = [n.op for n in replay.nodes.values() if isinstance(n.op, CudaOp)]
    assert op.knobs["SHARED_CARRY"] == 1 and not op.serial


def _selected_stores() -> Body:
    return parse_body(
        """
        f32 a = load x[i]
        f32 neg = negative(a)
        f32 b = load fallback[i]
        chosen = select:
            neg when i < 3
            b when true
        f32 out[i] = chosen
        f32 other[i] = chosen
        """
    )


def test_selected_stores_evaluate_only_the_chosen_private_cone():
    body = _selected_stores()
    changed = shared._guard_stores(body)
    (branch,) = changed
    assert isinstance(branch, Cond)
    assert [s.input for s in branch.body if isinstance(s, Load)] == ["x"]
    assert [s.input for s in branch.else_body if isinstance(s, Load)] == ["fallback"]
    inputs = {"x": np.arange(6, dtype=np.float32), "fallback": np.arange(10, 16, dtype=np.float32)}
    outputs = {"out": (6,), "other": (6,)}
    for candidate in (body, changed):
        op = KernelOp(body=parse_body("for i in 0..6:\n" + "\n".join(f"    {line}" for line in str(candidate).splitlines())))
        for result in execute_loop_op_cpp(op, inputs, outputs):
            np.testing.assert_array_equal(result, np.array([0, -1, -2, 13, 14, 15], dtype=np.float32))


@pytest.mark.parametrize("unsafe", ["reload", "sync", "atomic", "extra_reader", "index_reader"])
def test_selected_stores_keep_memory_effects_and_other_readers(unsafe):
    stmts = list(_selected_stores())
    if unsafe == "reload":
        stmts.insert(2, *parse_body("x[i] = replacement"))
    elif unsafe == "sync":
        stmts.insert(2, Sync())
    elif unsafe == "atomic":
        (stmts[-1],) = parse_body("f32 other[i] += chosen")
    elif unsafe == "extra_reader":
        stmts.extend(parse_body("f32 used = abs(chosen)"))
    else:
        (stmts[-1],) = parse_body("f32 other[chosen] = chosen")
    body = Body(stmts)
    assert shared._guard_stores(body) == body


def test_a_state_the_term_stores_every_step_has_no_port():
    """The step stores its next state into ``out`` at the step and the cell, so that output holds every
    step: the lift adds no buffer of its own, and the ordered launches read ``out`` one step back."""
    lifted = Pipeline.build(["tile/lift"]).run(_graph(), ctx=Context.from_target((7, 0)))
    assert [t.name for t in lifted.nodes["out"].outputs] == ["out"]
    with pinned_knobs({**_PINS, "SHARED_CARRY": 0}):
        compiled = Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target((7, 0)))
    (op,) = (node.op for node in compiled.nodes.values() if isinstance(node.op, CudaOp))
    assert op.serial and op.arg_order == ("seed", "out") and "__acc" not in op.kernel_source
    assert op.kernel_source.count("out[") >= 2, "the lagged read is a read of the term's own output"


@requires_cuda
def test_parallel_snapshot_copy_keeps_intermediate_casts():
    graph = _graph()

    def cast_snapshot(body):
        stmts = []
        for stmt in body:
            if isinstance(stmt, Write):
                stmts.extend(parse_body(f"f16 narrowed = copy({stmt.value})"))
                stmt = replace(stmt, values=("narrowed",))
            elif stmt.nested():
                stmt = stmt.with_bodies(tuple(cast_snapshot(b) for b in stmt.nested()))
            stmts.append(stmt)
        return Body(stmts)

    graph.nodes["out"].op = replace(graph.nodes["out"].op, body=cast_snapshot(graph.nodes["out"].op.body))
    array = (np.random.default_rng(0).standard_normal((2, 40, 40)) * 0.03).astype(np.float32)
    outputs = []
    for enabled in (0, 1):
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
    for storage in (0, 1):
        with pinned_knobs({**_PINS, "SHARED_CARRY": storage}):
            compiled = Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe())
        outputs.append(CudaBackend().run(compiled, input_data={"seed": array})[0].outputs["out"])
    np.testing.assert_array_equal(*outputs)


@requires_cuda
@pytest.mark.parametrize("dtype", [F32, F16], ids=["f32", "f16"])
@pytest.mark.parametrize("masked", [True, False], ids=["selected", "full"])
def test_shared_state_matches_the_global_state_on_the_same_inputs(dtype, masked):
    graph = _graph(dtype, masked=masked)
    backend = CudaBackend()
    candidates = []
    for enabled in (0, 1):
        with pinned_knobs({**_PINS, "SHARED_CARRY": enabled}):
            candidates.append(Pipeline.build(CUDA_PASSES).run(graph, ctx=Context.probe()))
    for seed in (0, 1):
        array = (np.random.default_rng(seed).standard_normal((2, 40, 40)) * 0.03).astype(dtype.np)
        outputs = [backend.run(program, input_data={"seed": array})[0].outputs["out"] for program in candidates]
        np.testing.assert_array_equal(*outputs)
