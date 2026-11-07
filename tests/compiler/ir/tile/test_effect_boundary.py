"""The 1q effect boundary — ``OutputSpec`` decorations + the ``extract_output_specs`` / ``apply_output_specs``
round-trip.

Projection ``Write``\\ s (and the rms/softmax output-sweep ``Loop``) leave the root Fold lambda for
``TileOp.output_specs``; every consumer reconstitutes the effectful stmt stream via ``apply_output_specs``.
The conversion gate is byte-identity: ``extract_output_specs`` returns a spelling ONLY when
``apply_output_specs`` reproduces the captured stream exactly (the 1o construction-gate pattern), so
these tests pin the round-trip on each supported shape and the ``None`` declines.
"""

from __future__ import annotations

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.stmt import Assign, Body, Loop, parse_body
from emmy.compiler.ir.tile import OutputSpec, apply_output_specs, extract_output_specs


def _ax(name: str, n: int = 64) -> Axis:
    return Axis(name=name, extent=Dim(n))


# --- plain trailing root Write(s) ---------------------------------------------------------------- #


def test_trailing_write_splits_and_round_trips() -> None:
    stmts = parse_body(
        """
        y = relu(x)
        o[m] = y
        """
    )
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign"]
    assert len(stores) == 1 and stores[0].sweep == () and stores[0].write is stmts[1]
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_trailing_write_run_keeps_order() -> None:
    """The register-strip shape: r loads, r computes, r stores — the Writes reattach in order."""
    stmts = parse_body(
        """
        x0 = load x[i]
        x1 = load x[i]
        y0 = relu(x0)
        y1 = relu(x1)
        o[i] = y0
        o[i] = y1
        """
    )
    pure, stores = extract_output_specs(stmts)
    assert len(pure) == 4 and len(stores) == 2
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_atomic_and_vector_write_fields_survive() -> None:
    """``OutputSpec`` holds the ``Write`` whole, so atomic / multi-value spellings are lossless."""
    for write in parse_body(
        """
        o[m] = (a, b)
        o[m] += a
        """
    ):
        pure, stores = extract_output_specs([write])
        assert pure == () and stores[0].write == write
        assert apply_output_specs(pure, stores) == Body((write,))


# --- the output-sweep Loop (rms/softmax projection) ----------------------------------------------- #


def _sweep_shape() -> Body:
    """The rms_norm projection: a scalar epilogue, then the normalize sweep with the root store."""
    return parse_body(
        """
        rs = rsqrt(sacc)
        for n in 0..64:
            xe = load x[m, n]
            we = load w[n]
            xhat = multiply(xe, rs)
            o[m, n] = xhat
        """
    )


def test_sweep_loop_splits_to_a_sweep_store_and_round_trips() -> None:
    stmts = _sweep_shape()
    pure, stores = extract_output_specs(stmts)
    assert [type(s).__name__ for s in pure] == ["Assign", "Load", "Load", "Assign"]
    assert len(stores) == 1 and [axis.name for axis in stores[0].sweep] == ["n"]
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_sweep_loop_with_multiple_writes_round_trips() -> None:
    n = _ax("n")
    stmts = parse_body(
        """
        for n in 0..64:
            x = load i[n]
            o[n] = x
            o[n] = x
        """
    )

    pure, stores = extract_output_specs(stmts)

    assert len(stores) == 2 and all(store.sweep == (n,) for store in stores)
    assert apply_output_specs(pure, stores) == Body(stmts)


def test_sweep_membership_is_the_trailing_axis_reading_run() -> None:
    """The scalar epilogue (no ``n`` read) stays outside the reconstituted loop; everything from
    the first ``n``-reading stmt on goes inside — the trailing-run rule."""
    stmts = _sweep_shape()
    pure, stores = extract_output_specs(stmts)
    rebuilt = apply_output_specs(pure, stores)
    assert isinstance(rebuilt[0], Assign) and rebuilt[0].name == "rs"
    assert isinstance(rebuilt[1], Loop) and len(rebuilt) == 2


def test_sibling_output_loops_become_sweep_specs() -> None:
    """Different local output extents are sibling sweeps: one spec per write, each over its own
    axis, reconstituting the loops in order. Their per-cell projections are not the boundary's —
    the lift forms them as terms declaring the sweep axis before extraction sees the stream."""
    stmts = parse_body(
        """
        for q in 0..4:
            q_out[m, q] = qv
        for kv in 0..2:
            k_out[m, kv] = kvv
            v_out[m, kv] = kvv
        """
    )

    pure, specs = extract_output_specs(stmts)

    assert pure == ()
    assert [axis.name for spec in specs for axis in spec.sweep] == ["q", "kv", "kv"]
    assert apply_output_specs(pure, specs) == Body(stmts)


# --- nested output sweeps (DeepSeek-V4 post4096's boundary shapes) -------------------------------- #


def test_nested_write_only_sweep_round_trips() -> None:
    """A write-only sweep nest with no outer store (post4096's softmax pair): the specs carry the
    axis PATH, outermost first, and reconstitution reopens both loops."""
    stmts = parse_body(
        """
        for a10 in 0..4:
            for a17 in 0..4:
                s_sum[m, a10, a17] = acc
                s_exp[m, a10, a17] = v98
        """
    )

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert pure == ()
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a10", "a17"], ["a10", "a17"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


def test_nested_sweeps_beside_an_outer_write_round_trip() -> None:
    """An outer sweep carrying its own store beside nested write-only sweeps (post4096's gate
    stream: view_8 beside the mul_16 and matmul_1 nests). The outer loop is opened once, by the
    last spec whose path carries it; the nested loops sit inside it, writes trailing."""
    stmts = parse_body(
        """
        for a20 in 0..4:
            for a27 in 0..8:
                mul_16[a20, a27] = v216
            for a27b in 0..4:
                for a35 in 0..8:
                    matmul_1[a20, a27b, a35] = v223
            view_8[a20] = v143
        """
    )

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert pure == ()
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"], ["a20", "a27b", "a35"], ["a20"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


def test_a_store_ahead_of_a_nested_sweep_keeps_its_source_position() -> None:
    """The outer store BEFORE the nest (the source order of post4096's gate stream): the outer
    group carries a prefix a later spec shares, so its write is appended bare and the last group
    carrying the outer axis wraps it together with the nest — store first, nest after, as written."""
    stmts = parse_body(
        """
        for a20 in 0..4:
            view_8[a20] = v143
            for a27 in 0..8:
                mul_16[a20, a27] = v216
        """
    )

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20"], ["a20", "a27"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


def test_sibling_nested_sweeps_share_the_outer_loop() -> None:
    """Two write-only nests under one outer sweep with no outer store: the outer loop is still
    opened exactly once, by the last group whose path carries it."""
    stmts = parse_body(
        """
        for a20 in 0..4:
            for a27 in 0..8:
                o1[a20, a27] = x
            for a27b in 0..4:
                o2[a20, a27b] = y
        """
    )

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"], ["a20", "a27b"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


def test_nested_sweep_pure_prefix_rejoins_the_stream() -> None:
    """A pure prefix inside a nested write-only sweep rejoins the stream, exactly as a single
    sweep's does, and the trailing-run rule re-wraps it on reconstitution."""
    stmts = parse_body(
        """
        for a20 in 0..4:
            for a27 in 0..8:
                x = load src[a20, a27]
                y = relu(x)
                o[a20, a27] = y
        """
    )

    split = extract_output_specs(stmts)

    assert split is not None
    pure, specs = split
    assert [type(stmt).__name__ for stmt in pure] == ["Load", "Assign"]
    assert [[axis.name for axis in spec.sweep] for spec in specs] == [["a20", "a27"]]
    assert apply_output_specs(pure, specs) == Body(stmts)


# --- declines (the caller keeps the computation in Loop IR) -------------------------------------- #


def test_reduce_loop_declines() -> None:
    """A reduce ``Loop`` (an ``Accum`` inside) is loop-carried state, never a sweep store."""
    stmts = parse_body(
        """
        for k in 0..64:
            acc <- add(acc, x) over k
        o[m] = acc
        """
    )
    assert extract_output_specs(stmts) is None


def test_impure_residue_declines() -> None:
    """An ``Init`` seed (030's finalize shape) is not a projection stmt — no pure spelling."""
    stmts = parse_body(
        """
        init f32 acc = 0
        o[m] = acc
        """
    )
    assert extract_output_specs(stmts) is None


def test_already_pure_stream_returns_no_stores() -> None:
    stmts = parse_body("y = relu(x)")
    pure, stores = extract_output_specs(stmts)
    assert pure == tuple(stmts) and stores == ()


def test_store_repr_round_trips_through_eval() -> None:
    """``TileOp.output_specs`` serializes as constructor reprs in graph dumps — eval must rebuild it."""
    from emmy.compiler.graph import _eval_stmt

    (write,) = parse_body("o[m] = y")
    st = OutputSpec(write=write, sweep=(_ax("n"),))
    assert _eval_stmt(repr(st)) == st


# --- the gate compares under construction normalization ------------------------------------------ #


def test_sweep_write_of_a_captured_accumulator_extracts() -> None:
    """A sweep may store an enclosing scope's value unchanged — ``o[j] = acc`` broadcasting an
    already-reduced accumulator. Nothing computes over ``j``, so no term opens it: the store's
    sweep spec is what binds the axis, wrapped around the writes alone."""
    stmts = parse_body(
        """
        for a1 in 0..8:
            x = load src[a1]
            y = relu(x)
            o1[a1] = y
        for a2 in 0..4:
            o2[a2] = acc
        """
    )
    split = extract_output_specs(stmts)
    assert split is not None
    pure, stores = split
    assert [store.write.output for store in stores] == ["o1", "o2"]
    assert [type(stmt).__name__ for stmt in pure] == ["Load", "Assign"], "the first sweep's pure prefix rejoins the stream"
    assert [axis.name for store in stores for axis in store.sweep] == ["a1", "a2"]
    assert apply_output_specs(pure, stores) == Body(stmts)
