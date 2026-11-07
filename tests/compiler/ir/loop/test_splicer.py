"""Standalone tests for the Loop IR splicer.

Each test builds producer/consumer ``LoopOp``s by hand and splices them
directly, asserting on the merged body structure. This skips the usual
frontend → tensor-IR → loop-IR pipeline so failures point at the splicer
itself rather than some upstream lowering quirk.
"""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.dtype import F16, DataType
from emmy.compiler.graph import Graph
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.expression import BinaryExpression
from emmy.compiler.ir.loop import Accum, Affine, Assign, Load, Loop, LoopOp, UnfusableStmt, Write, splice_graph, splice_loops
from emmy.compiler.ir.stmt import parse_body
from emmy.compiler.tensor import Tensor

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _loop(text: str) -> LoopOp:
    """A Loop op over the body ``text`` states."""
    return LoopOp(body=parse_body(text))


def _count_type(op: LoopOp, cls: type) -> int:
    return sum(1 for s in op.body.iter() if isinstance(s, cls))


def _elementwise_fns(op: LoopOp) -> list[str]:
    return [s.op.name for s in op.body.iter() if isinstance(s, Assign)]


def _affine_shapes(op: LoopOp) -> list[tuple[tuple[float, ...], float]]:
    """Every Affine's sorted coefficients and offset: the folded form a sum, difference or negation takes."""
    return sorted((tuple(sorted(s.coefficients)), s.offset) for s in op.body.iter() if isinstance(s, Affine))


def _splice_loop_ops(producer: LoopOp, consumer: LoopOp, source: str) -> LoopOp | None:
    """Pairwise fixture over the production N-way entry point."""
    writes = producer.writes
    if len(writes) != 1:
        return None
    return splice_loops(
        loops={"producer": producer, "consumer": consumer},
        splice_edges={("consumer", source): ("producer", writes[0].output)},
    )


def test_splicing_independent_row_sums_keeps_lexically_reused_accumulators_distinct() -> None:
    def rows(buffer: str, extent: int) -> str:
        return f"""
for row in 0..{extent}:
    for k in 0..3:
        value = load {buffer}[row, k]
        total <- add(total, value) over k
    {buffer}_sum[row] = total
"""

    op = _loop(rows("x", 2) + rows("y", 4))
    accums = [stmt.name for stmt in op.body.iter() if isinstance(stmt, Accum)]
    assert len(set(accums)) == 2
    merged = splice_loops({"root": op}, {}, roots=(("root", "x_sum"), ("root", "y_sum")))
    assert merged is not None
    inputs = {"x": np.arange(6, dtype=np.float32).reshape(2, 3), "y": np.arange(12, dtype=np.float32).reshape(4, 3) - 4}
    results = dict(zip(merged.outputs, merged.forward(*(inputs[name] for name in merged.inputs)), strict=True))
    for name, values in inputs.items():
        np.testing.assert_array_equal(results[f"{name}_sum"], values.sum(-1))


def test_normalization_preserves_explicit_state_through_a_conditional_update() -> None:
    op = _loop(
        """
        init f32 total = 10
        for k in 0..4 seed=false:
            x = load x[k]
            if k < 2.0:
                total <- add(total, x) over k
        sum[0] = total
        """
    )
    actual = op.forward(np.array([1, 2, 100, 100], dtype=np.float32))
    np.testing.assert_array_equal(actual, np.array([13], dtype=np.float32))


# Fixtures — shared axes
# ---------------------------------------------------------------------------


def _reduce_producer(*, source: str, output: str, rounds_to: DataType | None = None, op: str = "add") -> LoopOp:
    """A bare ``op`` reduce storing to ``output``, with its store rounding spelled when narrowing.

    ``loop/lowering/090_spell_store_rounding`` is what spells it in the real pipeline; these tests
    splice hand-built kernels, so they state it directly — the splicer's contract is to PRESERVE a
    spelled conversion, never to re-derive one from the buffer.
    """
    acc, stored = f"{output}_sum", f"{output}__st"
    tail = f"{output}[a0] = {acc}"
    if rounds_to is not None:
        tail = f"{rounds_to.name} {stored} = copy({acc})\n    {output}[a0] = {stored}"
    return _loop(
        f"""
for a0 in 0..4:
    for k in 0..16:
        {output}_value = load {source}[a0, k]
        {acc} <- {op}({acc}, {output}_value)
    {tail}
"""
    )


def _splice_graph_producer(
    producer: LoopOp,
    *,
    input_shape: tuple[int, ...],
    output_shape: tuple[int, ...],
    edge_dtype: DataType,
) -> LoopOp:
    consumer = _loop(
        """
        for a0 in 0..4:
            value = load producer[a0]
            activated = silu(value)
            consumer[a0] = activated
        """
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", input_shape, F16), node_id="x")
    graph.add_node(producer, ["x"], Tensor("producer", output_shape, edge_dtype), node_id="producer")
    graph.add_node(consumer, ["producer"], Tensor("consumer", output_shape, edge_dtype), node_id="consumer")
    graph.outputs = ["consumer"]
    result = splice_graph(graph)
    assert result is not None
    merged, external = result
    assert external == ["x"]
    return merged


# ---------------------------------------------------------------------------
# Basic pipeline: pointwise producer → pointwise consumer
# ---------------------------------------------------------------------------


def test_pointwise_chain():
    """producer: y = exp(x)  ;  consumer: z = add(y, y)."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load src_0[a0]
            y = exp(x)
            out_0[a0] = y
        """
    )
    consumer = _loop(
        """
        for a0 in 0..4:
            yv = load src_0[a0]
            z = add(yv, yv)
            out_0[a0] = z
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # exp from producer; the consumer's ``add(y, y)`` folds to one Affine ``2*y``.
    assert "exp" in _elementwise_fns(merged)
    assert _affine_shapes(merged) == [((2.0,), 0.0)]
    # Producer's Load (consumer had no other inputs) survives as source 0.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 1
    assert loads[0].input == "src_0"
    # Single Write, referencing the final add.
    writes = [s for s in merged.body.iter() if isinstance(s, Write)]
    assert len(writes) == 1


# ---------------------------------------------------------------------------
# Shared intermediate: consumer uses producer output twice at same index → one emission
# ---------------------------------------------------------------------------


def test_shared_intermediate_deduped():
    """Two consumer refs to the same producer value under identity σ share one binding."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load src_0[a0]
            y = exp(x)
            out_0[a0] = y
        """
    )
    # Consumer loads producer twice at the same index — both solve σ = {a0: a0}.
    consumer = _loop(
        """
        for a0 in 0..4:
            ya = load src_0[a0]
            yb = load src_0[a0]
            z = add(ya, yb)
            out_0[a0] = z
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # Only one exp in the merged body (producer chain materializes once).
    assert _elementwise_fns(merged).count("exp") == 1


@pytest.mark.parametrize("dependent", [False, True])
@pytest.mark.parametrize("offset", [0, 3])
def test_reductions_share_dependencies_in_cse_form(dependent, offset):
    """Independent reductions share upstream work; a finalized maximum stays outside its consumer's sum."""

    producer = _loop(
        f"""
        for a0 in 0..4:
            for j in 0..{16 + offset}:
                x = load X[a0, j]
                e = exp(x)
                P[a0, j] = e
        """
    )
    left = _reduce_producer(source="P", output="left", op="maximum")
    right = _loop(
        f"""
        for a0 in 0..4:
            {"m = load left[a0]" if dependent else ""}
            for k in 0..16:
                p = load P[a0, k + {offset}]
                {"shifted = subtract(p, m)" if dependent else ""}
                total <- add(total, {"shifted" if dependent else "p"})
            right[a0] = total
        """
    )
    edges = {("left", "P"): ("producer", "P"), ("right", "P"): ("producer", "P")}
    if dependent:
        edges["right", "left"] = ("left", "left")
    merged = splice_loops(
        loops={"producer": producer, "left": left, "right": right},
        splice_edges=edges,
        roots=(("left", "left"), ("right", "right")),
    )
    assert merged is not None
    assert _elementwise_fns(merged).count("exp") == (2 if dependent or offset else 1)
    assert sum(loop.is_reduce for loop in merged.body.iter_of_type(Loop)) == (2 if dependent else 1)
    x = np.linspace(-1, 1, 4 * (16 + offset), dtype=np.float32).reshape(4, 16 + offset)
    results = dict(zip(merged.outputs, merged.forward(x), strict=True))
    values = np.exp(x)
    maximum = values[:, :16].max(axis=1)
    summands = values[:, offset:] - maximum[:, None] if dependent else values[:, offset:]
    np.testing.assert_allclose(results["left"], maximum, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(results["right"], summands.sum(axis=1), rtol=1e-6, atol=1e-6)


# ---------------------------------------------------------------------------
# Different σs: two consumer loads at different indices → two emissions
# ---------------------------------------------------------------------------


def test_different_indices_emit_twice():
    """Consumer loads a 2D producer with two different index permutations —
    the two σs are distinct, so the producer's chain materializes twice."""
    # Producer axes (a0, a1); writes y[a0, a1] = exp(x[a0, a1]).
    # Square axes so the transposed load is well-typed.
    producer = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..4:
                x = load src_0[a0, a1]
                y = exp(x)
                out_0[a0, a1] = y
        """
    )
    # Consumer reads the producer at (a0, a1) and (a1, a0) — transposed.
    consumer = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..4:
                ya = load src_0[a0, a1]
                yb = load src_0[a1, a0]
                z = add(ya, yb)
                out_0[a0, a1] = z
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # Two distinct σs ({a0:a0,a1:a1}, {a0:a1,a1:a0}) → exp materializes twice.
    assert _elementwise_fns(merged).count("exp") == 2


# ---------------------------------------------------------------------------
# Reduction producer: sum over k, consumer is pointwise on the result
# ---------------------------------------------------------------------------


def test_reduction_producer():
    """Producer: s = sum_k x[a0,k]. Consumer: y = exp(s[a0])."""
    producer = _loop(
        """
        for a0 in 0..4:
            for k in 0..16:
                x = load src_0[a0, k]
                s <- add(s, x)
            out_0[a0] = s
        """
    )
    consumer = _loop(
        """
        for a0 in 0..4:
            sv = load src_0[a0]
            y = exp(sv)
            out_0[a0] = y
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    # The Accum survives in the merged body.
    assert _count_type(merged, Accum) == 1
    # Elementwise chain includes the producer-load copy + the consumer's exp.
    assert "exp" in _elementwise_fns(merged)


@pytest.mark.parametrize("rounds_to", [F16, None])
def test_graph_splice_preserves_a_spelled_store_rounding(rounds_to: DataType | None):
    """Fusing a store away must not delete the rounding it spelled — and must invent none."""
    producer = _reduce_producer(source="x", output="producer", rounds_to=rounds_to)
    merged = _splice_graph_producer(producer, input_shape=(4, 16), output_shape=(4,), edge_dtype=F16)
    typed = [c.dtype for c in merged.body.iter() if isinstance(c, Assign) and c.op.name == "copy" and c.dtype is not None]
    assert typed == ([rounds_to] if rounds_to is not None else [])


def test_graph_splice_load_output_needs_no_conversion():
    """A pass-through value is already represented at the loaded input dtype."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load x[a0]
            producer[a0] = x
        """
    )
    merged = _splice_graph_producer(producer, input_shape=(4,), output_shape=(4,), edge_dtype=F16)
    aliases = [stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "copy"]
    assert aliases == []


@pytest.mark.parametrize(("rounds_to", "expected_aliases"), [(F16, 2), (None, 0)])
def test_graph_splice_preserves_multi_reduction_output_dtypes(rounds_to: DataType | None, expected_aliases: int):
    """Sibling reductions each keep their own spelled rounding through one shared projection."""

    consumer = _loop(
        """
        for a0 in 0..4:
            left_value = load left[a0]
            right_value = load right[a0]
            aux_value = load aux[a0]
            activated = silu(left_value)
            product = multiply(activated, right_value)
            result = add(product, aux_value)
            consumer[a0] = result
        """
    )
    pointwise = _loop(
        """
        for a0 in 0..4:
            z_value = load z[a0]
            f16 exp_value = exp(z_value)
            aux[a0] = exp_value
        """
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 16), F16), node_id="x")
    graph.add_node(InputOp(), [], Tensor("y", (4, 16), F16), node_id="y")
    graph.add_node(InputOp(), [], Tensor("z", (4,), F16), node_id="z")
    graph.add_node(_reduce_producer(source="x", output="left", rounds_to=rounds_to), ["x"], Tensor("left", (4,), F16), node_id="left")
    graph.add_node(_reduce_producer(source="y", output="right", rounds_to=rounds_to), ["y"], Tensor("right", (4,), F16), node_id="right")
    graph.add_node(pointwise, ["z"], Tensor("aux", (4,), F16), node_id="aux")
    graph.add_node(consumer, ["left", "right", "aux"], Tensor("consumer", (4,), F16), node_id="consumer")
    graph.outputs = ["consumer"]

    result = splice_graph(graph)
    assert result is not None
    merged, external = result
    assert external == ["x", "y", "z"]
    typed = [c.dtype for c in merged.body.iter() if isinstance(c, Assign) and c.op.name == "copy" and c.dtype is not None]
    assert len(typed) == expected_aliases
    assert set(typed) <= {F16}
    pointwise_result = next(stmt for stmt in merged.body.iter() if isinstance(stmt, Assign) and stmt.op.name == "exp")
    assert pointwise_result.dtype == F16


def test_graph_splice_typed_boundary_roundtrips():
    """A spelled rounding is body state, so it survives Loop IR serialization."""
    producer = _reduce_producer(source="x", output="producer", rounds_to=F16)
    merged = _splice_graph_producer(producer, input_shape=(4, 16), output_shape=(4,), edge_dtype=F16)
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (4, 16), F16), node_id="x")
    graph.add_node(merged, ["x"], Tensor("consumer", (4,), F16), node_id="consumer")
    graph.outputs = ["consumer"]

    restored = Graph.from_dict(graph.to_dict()).nodes["consumer"].op
    typed = [c.dtype for c in restored.body.iter() if isinstance(c, Assign) and c.op.name == "copy" and c.dtype is not None]
    assert typed == [F16]


# ---------------------------------------------------------------------------
# Ordered prefix: a Write of the running accumulator cannot cross the splice
# ---------------------------------------------------------------------------


def _prefix_scan(*, source: str, output: str) -> LoopOp:
    return _loop(
        f"""
        for a0 in 0..4:
            for a1 in 0..8:
                x = load {source}[a0, a1]
                prefix <- add(prefix, x)
                {output}[a0, a1] = prefix
        """
    )


def test_ordered_prefix_root_declines_splice():
    """Rebuilding a root from its Writes must not move a prefix output after the reduce loop."""
    producer = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                x = load X[a0, a1]
                y = negative(x)
                P[a0, a1] = y
        """
    )

    with pytest.raises(UnfusableStmt, match="observes running accumulator"):
        _splice_loop_ops(producer, _prefix_scan(source="P", output="OUT"), source="P")


def test_ordered_prefix_producer_declines_splice():
    """Resolving a producer edge must not replace its prefix output with one final reduction."""
    consumer = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                x = load P[a0, a1]
                y = negative(x)
                OUT[a0, a1] = y
        """
    )

    with pytest.raises(UnfusableStmt, match="observes running accumulator"):
        _splice_loop_ops(_prefix_scan(source="X", output="P"), consumer, source="P")


# ---------------------------------------------------------------------------
# Consumer has extra input: source-remapping puts consumer inputs after producer's
# ---------------------------------------------------------------------------


def test_consumer_extra_input_source_remap():
    """Consumer has producer at source 0 and an unrelated buffer at source 1;
    after splice, that unrelated buffer should load from merged source 1
    (producer's input is at 0)."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load src_0[a0]
            y = exp(x)
            out_0[a0] = y
        """
    )
    # ``yv`` reads the producer; ``b`` is unrelated.
    consumer = _loop(
        """
        for a0 in 0..4:
            yv = load src_0[a0]
            b = load src_1[a0]
            z = add(yv, b)
            out_0[a0] = z
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    loads = {s.name: s for s in merged.body.iter() if isinstance(s, Load)}
    # Producer's Load survives at source 0; consumer's unrelated Load shifts to 1.
    assert any(ld.input == "src_0" for ld in loads.values())
    assert any(ld.input == "src_1" for ld in loads.values())
    assert len(tuple(merged.inputs)) == 2


# ---------------------------------------------------------------------------
# Literal in producer write index: no σ-binding needed, still splices
# ---------------------------------------------------------------------------


def test_live_axes_computed():
    """``BodyAnalysis.live_axes`` captures axes transitively used through each
    dep's Expression subtrees. For Accum, the reduce axis is excluded.

    Normalization canonicalizes SSA and axis names (``Load→in0``,
    ``Accum→acc0``; free axes first then reduce axes), so we assert on
    ``meta.scopes`` (now the post-reduce binding scope for Accum) and
    ``meta.reduce_axes`` rather than pre-normalize names.
    """
    op = _loop(
        """
        for a0 in 0..4:
            for k in 0..16:
                x = load src_0[a0, k]
                s <- add(s, x)
            out_0[a0] = s
        """
    )
    meta = op.analyze()
    [(load_name, load_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Load)]
    [(acc_name, acc_scope)] = [(n, s) for n, s in meta.scopes.items() if isinstance(meta.defs[n], Accum)]
    # Load sees both loop axes, Accum binds post-reduce so its scope is shorter.
    assert {a.name for a in load_scope.enclosing} == meta.live_axes[load_name]
    assert acc_name in meta.reduce_axes
    reduce_axis = meta.reduce_axes[acc_name]
    assert reduce_axis.name not in meta.live_axes[acc_name]
    # Accum's binding scope = load's enclosing minus the reduce axis.
    assert meta.scopes[acc_name].enclosing == tuple(a for a in load_scope.enclosing if a.name != reduce_axis.name)


# ---------------------------------------------------------------------------
# N-way chain fusion via splice_loop_chain
# ---------------------------------------------------------------------------


def test_chain_three_loops():
    """Fuse a → b → c in one splice call. ``a`` is pointwise exp, ``b`` negates
    the result, ``c`` adds a bias. The merged kernel should contain exp, neg,
    add — with one Load for x (a's input) and one for bias (c's extra input)."""
    a = _loop(
        """
        for a0 in 0..4:
            x = load X[a0]
            y = exp(x)
            A[a0] = y
        """
    )
    b = _loop(
        """
        for a0 in 0..4:
            av = load A[a0]
            y = negative(av)
            B[a0] = y
        """
    )
    c = _loop(
        """
        for a0 in 0..4:
            bv = load B[a0]
            bias = load BIAS[a0]
            y = add(bv, bias)
            C[a0] = y
        """
    )

    merged = splice_loops(
        loops={"a": a, "b": b, "c": c},
        splice_edges={("b", "A"): ("a", "A"), ("c", "B"): ("b", "B")},
    )
    assert merged is not None
    assert "exp" in _elementwise_fns(merged)
    # ``-exp(x) + bias``: the negation and the add fold into one Affine.
    assert _affine_shapes(merged) == [((-1.0, 1.0), 0.0)]
    # Two external Loads remain (X, BIAS); the splice edges collapsed into copy aliases.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 2
    assert {ld.input for ld in loads} == {"X", "BIAS"}
    assert len(tuple(merged.inputs)) == 2


# ---------------------------------------------------------------------------
# Multi-output support
# ---------------------------------------------------------------------------


def test_separate_roots_share_upstream_definition():
    """Two terminal loops seed one worklist, so their common producer emits once."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load X[a0]
            p = exp(x)
            P[a0] = p
        """
    )

    def consumer(output: str, op: str) -> LoopOp:
        return _loop(
            f"""
            for a0 in 0..4:
                p = load P[a0]
                out = {op}(p)
                {output}[a0] = out
            """
        )

    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("X", (4,)), node_id="X")
    graph.add_node(producer, ["X"], Tensor("P", (4,)), node_id="P")
    graph.add_node(consumer("LEFT", "negative"), ["P"], Tensor("LEFT", (4,)), node_id="LEFT")
    graph.add_node(consumer("RIGHT", "abs"), ["P"], Tensor("RIGHT", (4,)), node_id="RIGHT")
    graph.outputs = ["LEFT", "RIGHT"]

    result = splice_graph(graph)
    assert result is not None
    merged, external = result
    assert external == ["X"]
    assert {write.output for write in merged.writes} == {"LEFT", "RIGHT"}
    assert _elementwise_fns(merged).count("exp") == 1


def test_output_equivalence_cluster_retargets_reduce_through_copy_chain(monkeypatch):
    """Flat-address-identical output copies collapse without inlining the reduction at their loads."""
    producer = _loop(
        """
        for i in 0..2:
            for k in 0..3:
                xk = load x[i, k]
                total <- add(total, xk) over k
            for j in 0..3:
                xj = load x[i, j]
                scaled = multiply(xj, total)
                y[i, j] = scaled
        """
    )
    first_copy = _loop(
        """
        for p in 0..3:
            for q in 0..2:
                mid_value = load y[(p * 2 + q) / 3, (p * 2 + q) % 3]
                mid[p, q] = mid_value
        """
    )
    second_copy = _loop(
        """
        for r in 0..6:
            out_value = load mid[r / 2, r % 2]
            out[r] = out_value
        """
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (2, 3)), node_id="x")
    graph.add_node(producer, ["x"], Tensor("y", (2, 3)), node_id="y")
    graph.add_node(first_copy, ["y"], Tensor("mid", (3, 2)), node_id="mid")
    graph.add_node(second_copy, ["mid"], Tensor("out", (6,)), node_id="out")
    graph.outputs = ["out"]
    eval_expression = BinaryExpression.eval

    def reject_coordinate_evaluation(expression, env):
        if env:
            pytest.fail("output equivalence must not enumerate coordinates")
        return eval_expression(expression, env)

    monkeypatch.setattr(BinaryExpression, "eval", reject_coordinate_evaluation)

    result = splice_graph(graph)

    assert result is not None
    merged, external = result
    assert external == ["x"]
    assert _count_type(merged, Accum) == 1
    assert [write.output for write in merged.writes] == ["out"]
    assert len(merged.writes[0].index) == 1
    assert "/" not in str(merged.writes[0].index[0]) and "%" not in str(merged.writes[0].index[0])


def test_output_equivalence_retargets_transpose_in_canonical_storage_order():
    """A proven permutation retargets the Write and adopts canonical output-storage order."""
    producer = _loop(
        """
        for i in 0..2:
            for j in 0..3:
                value = load x[i, j]
                y[i, j] = value
        """
    )
    transpose = _loop(
        """
        for p in 0..3:
            for q in 0..2:
                transposed = load y[q, p]
                out[p, q] = transposed
        """
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("x", (2, 3)), node_id="x")
    graph.add_node(producer, ["x"], Tensor("y", (2, 3)), node_id="y")
    graph.add_node(transpose, ["y"], Tensor("out", (3, 2)), node_id="out")
    graph.outputs = ["out"]

    result = splice_graph(graph)

    assert result is not None
    merged, _ = result
    assert [loop.axis.extent for loop in merged.body.loops] == [3, 2]
    load = next(load for load in merged.loads if load.input == "x")
    write = next(write for write in merged.writes if write.output == "out")
    assert load.index == tuple(reversed(write.index))


def test_multi_output_root_preserves_all_writes():
    """A root whose body contains Writes at multiple output indices should
    seed each of them — the merged kernel carries all Writes."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load src_0[a0]
            y = exp(x)
            out_0[a0] = y
        """
    )
    # Consumer produces two outputs: a negation and an absolute value — both
    # reading the producer.
    consumer = _loop(
        """
        for a0 in 0..4:
            yv = load out_0[a0]
            z0 = negative(yv)
            z1 = abs(yv)
            C0[a0] = z0
            C1[a0] = z1
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="out_0")
    assert merged is not None
    writes = [s for s in merged.body.iter() if isinstance(s, Write)]
    assert {w.output for w in writes} == {"C0", "C1"}
    fns = _elementwise_fns(merged)
    # Producer chain materialized once (shared σ); both consumer ops present.
    assert fns.count("exp") == 1
    assert "abs" in fns
    assert _affine_shapes(merged) == [((-1.0,), 0.0)]


def test_multi_output_splice_target():
    """A splice target with two outputs (output="out_0" and output="out_1") — distinct
    splice edges read each output. The target's expression chain reconstructs
    separately for each output via the standard worklist walk."""
    # Target: T0 = exp(x), T1 = neg(x). Two outputs.
    target = _loop(
        """
        for a0 in 0..4:
            x = load X[a0]
            y0 = exp(x)
            y1 = negative(x)
            T0[a0] = y0
            T1[a0] = y1
        """
    )
    # Sink: z = add(T0, T1). Two consumer Loads each pointing at one target output.
    sink = _loop(
        """
        for a0 in 0..4:
            a = load T0[a0]
            b = load T1[a0]
            z = add(a, b)
            OUT[a0] = z
        """
    )

    merged = splice_loops(
        loops={"target": target, "sink": sink},
        splice_edges={
            ("sink", "T0"): ("target", "T0"),
            ("sink", "T1"): ("target", "T1"),
        },
    )
    assert merged is not None
    fns = _elementwise_fns(merged)
    # Both target ops appear in the merged body once each.
    assert fns.count("exp") == 1
    # The negation is read once (its write left with the splice edge), so ``exp(x) + (-x)`` is one Affine.
    assert _affine_shapes(merged) == [((-1.0, 1.0), 0.0)]
    # Only one external Load (target's x); the splice edges replaced sink's Loads.
    loads = [s for s in merged.body.iter() if isinstance(s, Load)]
    assert len(loads) == 1
    assert len(tuple(merged.inputs)) == 1


def test_shared_load_dedup_into_reduce():
    """Producer body has a Load reused across a long expression chain
    (mirrors silu: ``in = load x; v0 = neg(in); v1 = exp(v0); v2 = add(1, v1);
    v3 = recip(v2); v4 = mul(in, v3)``). The first and last uses of ``in`` sit
    at opposite ends of the chain. Splicing into a consumer that reads the
    producer's output inside a reduce loop exercises the dedup + emit-ordering
    path: when the late ``v4 = mul(in, v3)`` lands, ``in`` dedups to the
    existing binding emitted earlier for ``v0``; if the splicer prepends naïvely
    the new stmt lands above its own dep. The merged body must keep SSA in
    topological order across the reduce boundary."""
    producer = _loop(
        """
        for a0 in 0..4:
            in_ = load src_0[a0]
            v0 = negative(in_)
            v1 = exp(v0)
            v2 = add(v1, v1)
            v3 = reciprocal(v2)
            v4 = multiply(in_, v3)
            out_0[a0] = v4
        """
    )
    # Consumer: matmul-style reduce over a1. The producer's output is loaded
    # inside the reduce loop, so the spliced chain has to be ordered correctly
    # within the (a0, a1) reduce scope.
    consumer = _loop(
        """
        for a0 in 0..4:
            for a1 in 0..8:
                pv = load src_0[a1]
                w = load src_1[a1, a0]
                prod = multiply(pv, w)
                acc <- add(acc, prod)
            out_0[a0] = acc
        """
    )
    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None  # would have failed pre-fix with SSA validation error
    # Reduce + full silu chain + final multiply should all materialize once.
    assert _count_type(merged, Accum) == 1
    fns = _elementwise_fns(merged)
    for op in ("exp", "reciprocal", "multiply"):
        assert op in fns, f"missing {op} in merged body: {fns}"
    assert _affine_shapes(merged) == [((-1.0,), 0.0), ((2.0,), 0.0)]


def test_literal_producer_write_index():
    """Producer writes at (a0, 0); consumer reads at (a0, 0). The Literal dim
    contributes no σ binding and shouldn't block splicing."""
    producer = _loop(
        """
        for a0 in 0..4:
            x = load src_0[a0]
            y = exp(x)
            out_0[a0, 0.0] = y
        """
    )
    consumer = _loop(
        """
        for a0 in 0..4:
            yv = load src_0[a0, 0.0]
            z = negative(yv)
            out_0[a0] = z
        """
    )

    merged = _splice_loop_ops(producer, consumer, source="src_0")
    assert merged is not None
    assert "exp" in _elementwise_fns(merged)
    assert _affine_shapes(merged) == [((-1.0,), 0.0)]


# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Construction bound — a recurrence-shaped chain must be declined, not built
# ---------------------------------------------------------------------------


def _affine_recurrence_chain(depth: int) -> tuple[dict[str, LoopOp], dict, tuple]:
    """A chain whose σs compose into exponentially many distinct bindings.

    Stage ``k`` reads stage ``k-1`` at ``2·i + 1`` and at ``3·i + 2``: two non-commuting affine
    maps generate a free monoid, so every path through ``depth`` stages composes to a DISTINCT
    map and stage 0's statements are demanded under ``2^(depth-1)`` different σs. This is the
    DeepSeek-V4 Sinkhorn shape minimized — the merge that made 4.5M distinct bindings from 2,287
    input statements and never finished constructing.
    """
    loops: dict[str, LoopOp] = {}
    edges: dict = {}
    for k in range(depth):
        src = "b0" if k == 0 else f"s{k - 1}"
        loops[f"s{k}"] = _loop(
            f"""
            for i in 0..8:
                x{k} = load {src}[2 * i + 1]
                y{k} = load {src}[3 * i + 2]
                v{k} = add(x{k}, y{k})
                s{k}[i] = v{k}
            """
        )
        if k:
            edges[(f"s{k}", f"s{k - 1}")] = (f"s{k - 1}", f"s{k - 1}")
    roots = ((f"s{depth - 1}", f"s{depth - 1}"),)
    return loops, edges, roots


def test_shallow_affine_recurrence_still_fuses():
    """Depth 4 → at most 8 distinct bindings per stage-0 stmt — the suite-wide legitimate maximum."""
    loops, edges, roots = _affine_recurrence_chain(4)
    merged = splice_loops(loops=loops, splice_edges=edges, roots=roots)
    assert merged is not None
    assert _count_type(merged, Write) == 1  # one root — every stage inlined


def test_deep_affine_recurrence_raises_past_the_binding_ratio():
    """Depth 16 → 32768 bindings per stage-0 stmt: constructing the merge is hopeless, so the
    splicer raises at the construction bound instead of multiplying bindings without end. Fusion
    never catches it — a chain that multiplies is the roller's to roll — so the doom names the loop."""
    loops, edges, roots = _affine_recurrence_chain(16)
    with pytest.raises(UnfusableStmt, match="bindings per source statement"):
        splice_loops(loops=loops, splice_edges=edges, roots=roots)
