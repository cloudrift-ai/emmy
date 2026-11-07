"""Tests for ``_unify_siblings`` (relaxed overlap matching) and
``merge_sibling_reduce_loops`` in ``stmt/normalize.py``.

Builds bodies by hand and asserts on the post-pass structure so failures
point at the pass itself rather than upstream lowering. The motivating
case is the gated-MLP pattern ``silu(x @ Wg) * (x @ Wu)`` where the two
sibling reduce-loops index ``x`` at the same K position but bring in
distinct weight tensors — unify must group them on the shared x slot,
then merge must concatenate the bodies into one K-loop.
"""

from __future__ import annotations

from textwrap import indent

from emmy.compiler.ir.stmt import Accum, Body, Load, Loop, normalize_body, parse_body
from emmy.compiler.ir.stmt.normalize import _unify_siblings, merge_sibling_reduce_loops

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _lines(*lines: str) -> str:
    return "\n".join(lines)


def _loop(header: str, *lines: str) -> str:
    """The text of a loop ``header`` over ``lines``, each line a statement or a nested loop's text."""
    return _lines(header, *(indent(line, "    ") for line in lines))


def _matmul_reduce(axis: str, extent: int, w_buffer: str, out_acc: str, k_load: str = "x") -> str:
    """Build a single reduce-Loop body: ``acc += W[k, n] * x[k]``.

    Used to assemble the gated-MLP / matmul-sibling patterns the new
    passes target. Inner SSA names are namespaced by ``out_acc`` so two
    instances composed in one parent body have disjoint defs.
    """
    return _loop(
        f"for {axis} in 0..{extent}:",
        f"w_{out_acc} = load {w_buffer}[{axis}, n]",
        f"x_{out_acc} = load {k_load}[{axis}]",
        f"m_{out_acc} = multiply(w_{out_acc}, x_{out_acc})",
        f"{out_acc} <- add({out_acc}, m_{out_acc})",
    )


# ---------------------------------------------------------------------------
# _unify_siblings — overlap-based grouping
# ---------------------------------------------------------------------------


def test_unify_groups_loops_with_shared_load_position() -> None:
    """Two reduce-loops over different axis names but both indexing ``x``
    at the same dim must end up sharing one canonical axis name."""
    body = parse_body(_lines(_matmul_reduce("a2", 16, "Wg", "acc0"), _matmul_reduce("a3", 16, "Wu", "acc1")))

    out = _unify_siblings(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2
    assert loops[0].axis.name == loops[1].axis.name == "a2"


def test_unify_skips_when_extents_differ() -> None:
    """Overlap on (source, dim) is necessary but not sufficient: extents
    must also match — different extents mean different iteration spaces
    even if positions overlap."""
    body = parse_body(_lines(_matmul_reduce("a2", 16, "Wg", "acc0"), _matmul_reduce("a3", 32, "Wu", "acc1")))

    out = _unify_siblings(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert loops[0].axis.name == "a2"
    assert loops[1].axis.name == "a3"


def test_unify_skips_disjoint_load_positions() -> None:
    """Two reduce-loops with no shared (source, dim) overlap stay in
    distinct groups — they index unrelated tensors at the reduce axis."""
    body = parse_body(
        """
        for a2 in 0..16:
            p = load A[a2]
            acc0 <- add(acc0, p)
        for a3 in 0..16:
            q = load B[a3]
            acc1 <- add(acc1, q)
        """
    )

    out = _unify_siblings(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert loops[0].axis.name == "a2"
    assert loops[1].axis.name == "a3"


def test_unify_transitively_groups_three_loops() -> None:
    """Union-find on the overlap relation: A overlaps B on (x,0), B
    overlaps C on (y,0), so all three unify to one canonical name."""
    body = parse_body(
        """
        for a in 0..8:
            p = load x[a]
            acc_a <- add(acc_a, p)
        for b in 0..8:
            q = load x[b]
            r = load y[b]
            s = multiply(q, r)
            acc_b <- add(acc_b, s)
        for c in 0..8:
            t = load y[c]
            acc_c <- add(acc_c, t)
        """
    )

    out = _unify_siblings(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert {loop.axis.name for loop in loops} == {"a"}


# ---------------------------------------------------------------------------
# merge_sibling_reduce_loops — concatenation under safety gates
# ---------------------------------------------------------------------------


def test_merge_concatenates_matching_reduce_bodies() -> None:
    """The motivating case: after unify renames both axes to ``a2``,
    merge concatenates the two bodies into one Loop, eliminating the
    duplicate K traversal."""
    body = parse_body(
        _lines(
            _matmul_reduce("k", 16, w_buffer="Wg", out_acc="acc0"),
            _matmul_reduce("k", 16, w_buffer="Wu", out_acc="acc1"),
            "out[n] = acc1",
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1, f"expected one merged Loop, got {len(loops)}"
    inner_accs = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert inner_accs == ["acc0", "acc1"]


def test_merge_skips_when_second_loop_reads_first_loops_accum() -> None:
    """Softmax pattern: first reduce produces ``acc_max``, second reduce
    body reads it inside the iteration. Merging would change the read
    from finalized to in-flight — must be skipped."""
    body = parse_body(
        """
        for k in 0..16:
            x1 = load X[k]
            acc_max <- maximum(acc_max, x1)
        for k in 0..16:
            x2 = load X[k]
            d = subtract(x2, acc_max)
            e = exp(d)
            acc_sum <- add(acc_sum, e)
        """
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "softmax-style sequential reduces must stay distinct"


def test_merge_renames_a_colliding_local_rather_than_refusing() -> None:
    """Two bodies binding one name for UNRELATED values is a collision, not a dependence — and it
    is what alpha-equal copies of a single cone always look like. The incoming body's local is
    renamed apart and the merge proceeds."""
    body = parse_body(
        """
        for k in 0..16:
            v = load A[k]
            acc0 <- add(acc0, v)
        for k in 0..16:
            v = load B[k]
            acc1 <- add(acc1, v)
        """
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1
    loads = {s.input: s.names[0] for s in loops[0].body if isinstance(s, Load)}
    assert loads["A"] != loads["B"], "each half keeps its own binding"
    accums = {s.name: s.value for s in loops[0].body if isinstance(s, Accum)}
    assert accums["acc0"] == loads["A"] and accums["acc1"] == loads["B"]


def test_merge_refuses_a_collision_on_a_name_the_loop_still_binds() -> None:
    """A carrier of the incoming Loop's own scope is declared ahead of that loop and read after
    it, so a rename could not reach its readers. That one collision refuses."""
    body = parse_body(
        """
        for k in 0..16:
            p = load A[k]
            acc <- add(acc, p)
        for k in 0..16:
            q = load B[k]
            acc <- add(acc, q)
        out[n] = acc
        """
    )

    out = merge_sibling_reduce_loops(body)

    assert len([s for s in out if isinstance(s, Loop)]) == 2


def _half(tag: str) -> str:
    """An outer reduce over ``k`` folding an inner reduce over ``d`` of ``tag[k, d]`` into ``out_<tag>``."""
    return _loop(
        "for k in 0..16:",
        _loop("for d in 0..8:", f"v = load {tag}[k, d]", "acc0 <- add(acc0, v)"),
        f"out_{tag} <- add(out_{tag}, acc0)",
    )


def test_merge_renames_an_accum_that_stays_inside() -> None:
    """The counterpart, and the case a blocked reduce hits: an ``Accum`` bound in a NESTED loop
    crosses that loop's boundary but is consumed within this body. Nothing outside reads it, so it
    renames apart like any other local."""
    out = merge_sibling_reduce_loops(parse_body(_lines(_half("A"), _half("B"))))

    assert len([s for s in out if isinstance(s, Loop)]) == 1


def test_unify_groups_loops_indexed_affinely_in_the_reduce_axis() -> None:
    """A BLOCKED reduce reads its stream at ``outer·B + inner``. The axis still walks that
    dimension, so siblings over one block must unify — refusing anything but a bare ``Var`` left
    them unmergeable for no semantic reason."""
    blocked = [
        _loop(f"for {name} in 0..64:", f"v_{name} = load x[o * 64 + {name}]", f"acc_{name} <- add(acc_{name}, v_{name})")
        for name in ("i", "j")
    ]

    out = _unify_siblings(parse_body(_lines(*blocked)))

    names = {s.axis.name for s in out if isinstance(s, Loop)}
    assert len(names) == 1, "affine siblings over one block index the same dimension"


def test_unify_keeps_apart_blocks_at_different_offsets() -> None:
    """``o·B + i`` and ``o·B + 32 + j`` walk different halves of the same dimension. Seeing through
    the offset would unify them and miscompile; the anchor is on the key so they stay distinct."""

    def at(name: str, extra: int) -> str:
        shifted = "o * 64" if not extra else f"o * 64 + {extra}"
        return _loop(f"for {name} in 0..32:", f"v_{name} = load x[{shifted} + {name}]", f"acc_{name} <- add(acc_{name}, v_{name})")

    out = _unify_siblings(parse_body(_lines(at("i", 0), at("j", 32))))

    assert len({s.axis.name for s in out if isinstance(s, Loop)}) == 2


def _packed_stream_reduce(name: str, index, w_buffer: str = "W") -> str:
    """A reduce loop ``acc += W[k] * bits[index(k)]`` reading one byte-packed stream ``bits`` — the
    shape a 4-bit activation is read in, where the axis reaches the index through div and mod.
    ``w_buffer`` names the weight, so two instances compute distinct values."""
    return _loop(
        f"for {name} in 0..256:",
        f"w_{name} = load {w_buffer}[{name}]",
        f"v_{name} = load bits[{index(name)}]",
        f"m_{name} = multiply(w_{name}, v_{name})",
        f"acc_{name} <- add(acc_{name}, m_{name})",
    )


def _two_codes_per_byte(k: str) -> str:
    """``(k / 16) * 8 + (k % 16) / 2`` — the byte holding logical element ``k`` of a stream packed
    two codes per byte in blocks of 16."""
    return f"{k} / 16 * 8 + {k} % 16 / 2"


def test_unify_groups_loops_indexed_through_one_composite_expression() -> None:
    """A packed stream is read through div and mod of the axis, which is not affine in it. Two
    siblings reading one stream through the same expression of their own axis walk the same
    dimension and unify, the same way bare and affine readers do."""
    siblings = (_packed_stream_reduce("i", _two_codes_per_byte), _packed_stream_reduce("j", _two_codes_per_byte))
    out = _unify_siblings(parse_body(_lines(*siblings)))

    names = {s.axis.name for s in out if isinstance(s, Loop)}
    assert len(names) == 1, "composite siblings over one packed stream index the same dimension"


def test_unify_keeps_apart_different_composite_expressions() -> None:
    """The composite key is the whole expression: ``bits[k / 16]`` and ``bits[k % 16]`` both
    mention the axis at the same position and walk different things, so they stay distinct."""
    out = _unify_siblings(
        parse_body(
            _lines(
                _packed_stream_reduce("i", lambda k: f"{k} / 16", "Wg"),
                _packed_stream_reduce("j", lambda k: f"{k} % 16", "Wu"),
            )
        )
    )

    assert len({s.axis.name for s in out if isinstance(s, Loop)}) == 2


def test_unify_then_merge_collapses_two_readers_of_one_packed_stream() -> None:
    """After unification the two packed-stream siblings merge into one loop with both
    accumulators, which is the two-channel contraction a fused gate/up pair over a 4-bit
    activation lowers to."""
    out = normalize_body(
        parse_body(_lines(_packed_stream_reduce("i", _two_codes_per_byte, "Wg"), _packed_stream_reduce("j", _two_codes_per_byte, "Wu")))
    )

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1
    assert len([s for s in loops[0].body if isinstance(s, Accum)]) == 2
    assert len([s for s in loops[0].body if isinstance(s, Load) and s.input == "bits"]) == 1, "the shared stream read is one load"


def test_merge_skips_when_between_stmt_def_used_by_second_loop() -> None:
    """A pure stmt between the two loops defines an SSA name the second
    loop's body reads — merging would move the read above the def."""
    body = parse_body(
        """
        for k in 0..16:
            p = load A[k]
            acc0 <- add(acc0, p)
        bias = exp(acc0)
        for k in 0..16:
            q = load B[k]
            m = multiply(q, bias)
            acc1 <- add(acc1, m)
        """
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "between-stmt def consumed by second loop must block the merge"


def test_merge_keeps_intermediate_pure_stmts_in_place() -> None:
    """Pure stmts between the two reduce-loops that only depend on the
    first loop's Accum (not the second's body) remain valid post-merge
    — first Accum is still in scope above them, and they don't feed the
    second loop's body."""
    body = parse_body(
        _lines(
            _matmul_reduce("k", 16, w_buffer="Wg", out_acc="acc0"),
            "silu_val = silu(acc0)",  # depends only on acc0
            _matmul_reduce("k", 16, w_buffer="Wu", out_acc="acc1"),
            "result = multiply(silu_val, acc1)",
            "out[n] = result",
        )
    )

    out = merge_sibling_reduce_loops(body)

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 1, "intermediate stmt depending only on first Accum must not block the merge"
    inner_accs = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert inner_accs == ["acc0", "acc1"]


def test_merge_recurses_into_nested_loops() -> None:
    """Outer free-loops over (n, m) wrap sibling reduce-loops over k —
    the pass must descend through the outer Loops and merge the inner
    siblings."""
    body = parse_body(
        _loop(
            "for n in 0..4:",
            _matmul_reduce("k", 16, w_buffer="Wg", out_acc="acc0"),
            _matmul_reduce("k", 16, w_buffer="Wu", out_acc="acc1"),
            "o[n] = acc1",
        )
    )

    out = merge_sibling_reduce_loops(body)

    outer = out[0]
    assert isinstance(outer, Loop) and outer.axis.name == "n"
    inner_loops = [s for s in outer.body if isinstance(s, Loop)]
    assert len(inner_loops) == 1, "inner sibling reduces must merge inside the outer Loop body"


def test_unify_then_merge_collapses_gated_mlp_pattern() -> None:
    """End-to-end: the gated-MLP pattern ``silu(x@Wg) * (x@Wu)`` enters
    with distinct axis names ``a2`` / ``a3`` indexing the same x slot.
    After unify + merge, one K-loop carries both matmul reductions and
    the post-loop multiply still references both Accums correctly."""
    body = parse_body(
        _lines(
            _matmul_reduce("a2", 16, w_buffer="Wg", out_acc="acc0"),
            "silu_val = silu(acc0)",
            _matmul_reduce("a3", 16, w_buffer="Wu", out_acc="acc1"),
            "result = multiply(silu_val, acc1)",
            "out[n] = result",
        )
    )

    merged = merge_sibling_reduce_loops(body)

    loops = [s for s in merged if isinstance(s, Loop)]
    assert len(loops) == 1
    accs_inside = [s.name for s in loops[0].body if isinstance(s, Accum)]
    assert accs_inside == ["acc0", "acc1"]
    x_loads = [s for s in loops[0].body if isinstance(s, Load) and s.input == "x"]
    assert len(x_loads) == 2, "dedup happens in a later pass — merge alone leaves both x loads"


def test_normalize_closes_reductions_exposed_by_hoisting() -> None:
    """A reduction hoisted beside an existing sibling merges in the same normalization."""
    body = parse_body(
        """
        for i in 0..2:
            for k0 in 0..4:
                x0 = load X[k0]
                a <- add(a, x0)
            A[i] = a
        for k1 in 0..4:
            x1 = load X[k1]
            b <- add(b, x1)
        B[] = b
        """
    )

    normalized = normalize_body(body)
    reduce_loops = [stmt for stmt in normalized if isinstance(stmt, Loop) and stmt.is_reduce]

    assert len(reduce_loops) == 1
    assert len([stmt for stmt in reduce_loops[0].body if isinstance(stmt, Load)]) == 1
    assert normalize_body(Body(tuple(normalized))) == normalized


def test_normalize_closes_children_exposed_by_parent_merge(monkeypatch) -> None:
    """Close newly exposed child reductions before ordering the canonical body: the order runs once, on the closed
    body, and leaves its own output alone."""
    from emmy.compiler.ir.stmt import normalize

    builds = []
    build = normalize._canonical_order

    def counted(body):
        builds.append(body)
        return build(body)

    monkeypatch.setattr(normalize, "_canonical_order", counted)

    def cone(outer: str, inner: str, tag: str) -> str:
        return _loop(
            f"for {outer} in 0..4:",
            _loop(f"for {inner} in 0..8:", f"load_{tag} = load X[{outer}, {inner}]", f"inner_{tag} <- add(inner_{tag}, load_{tag})"),
            f"outer_{tag} <- add(outer_{tag}, inner_{tag})",
        )

    body = parse_body(_lines(cone("k0", "j0", "a"), cone("k1", "j1", "b"), "Y[] = outer_b"))
    merged = merge_sibling_reduce_loops(body)
    assert len(merged.iter_of_type(Loop)) == 2
    assert len(merged.accums) == 4
    normalized = normalize_body(body)

    assert len(builds) == 1
    assert build(normalized) == normalized
    assert len(tuple(normalized.iter_of_type(Loop))) == 2
    assert len(tuple(normalized.iter_of_type(Load))) == 1
    assert normalize_body(Body(tuple(normalized))) == normalized


def test_merge_collapses_three_alpha_equal_siblings() -> None:
    """Merging is pairwise, so a third copy has to fold into the pair already merged: each binds
    ``acc0`` for its own value, and all three end up in one loop."""
    out = merge_sibling_reduce_loops(parse_body(_lines(_half("A"), _half("B"), _half("C"))))

    assert len([s for s in out if isinstance(s, Loop)]) == 1


def _blocked_channel(block: str, tag: str, *tail: str) -> str:
    """One inner fold of a blocked twisted carrier: re-derive the score over the block, weigh it
    against the block pivot, then accumulate this channel's own contribution.

    Every channel is an alpha-equal copy of the same cone down to the spellings — the weight's
    temps come out of ONE stored combine — and each recomputes the score, which is the recompute
    the merge exists to remove.
    """
    return _loop(
        f"for {block} in 0..64:",
        _loop(
            "for a3 in 0..32:",
            "in1 = load x0[a3]",
            f"in2 = load x1[a2_o * 64 + {block}, a3]",
            "v0 = multiply(in1, in2)",
            "acc0 <- add(acc0, v0)",
        ),
        "v1 = multiply(acc0, scale)",
        "acc1__o__gn = maximum(acc1__blk, v1)",
        "acc1__o__dg_o = subtract(v1, acc1__o__gn)",
        "acc1__o__beta = exp(acc1__o__dg_o)",
        *tail,
        f"{tag}__blk <- add({tag}__blk, {tag}__blk__e)",
    )


def test_merge_collapses_the_channel_loops_of_a_blocked_twisted_carrier() -> None:
    """The shape blocking attention's carrier produces: a block pivot, then one inner loop per
    channel, all over the same block. The pivot must stay apart — the weight reads its finished
    accumulator — and the channels must collapse, which is what takes ``Q·K`` from three passes
    over the block to two.

    The enclosing scope re-binds the very temps the channels use (``acc1__o__gn`` and friends are
    the stored combine's, instantiated in both places), so a merge gate that asks whether a name
    is read anywhere around the pair — instead of what the loop itself still binds — refuses here.
    """
    pivot = _loop(
        "for a2_p in 0..64:",
        _loop(
            "for a3 in 0..32:",
            "in1 = load x0[a3]",
            "in2 = load x1[a2_o * 64 + a2_p, a3]",
            "v0 = multiply(in1, in2)",
            "acc0 <- add(acc0, v0)",
        ),
        "acc1__blk__e = multiply(acc0, scale)",
        "acc1__blk <- add(acc1__blk, acc1__blk__e)",
    )
    expectation = _blocked_channel(
        "a2_c1",
        "acc5__sum",
        "in7 = load x2[a2_o * 64 + a2_p, a5]",
        "acc5__sum__blk__e = multiply(acc1__o__beta, in7)",
    )
    denominator = _blocked_channel("a2_c2", "acc3", "acc3__blk__e = copy(acc1__o__beta)")
    combine = (
        "acc1__o__gn = maximum(acc1, acc1__blk)",
        "acc1__o__dg_o = subtract(acc1__blk, acc1__o__gn)",
        "acc1__o__beta = exp(acc1__o__dg_o)",
        "acc5__sum <- add(acc5__sum, acc5__sum__blk)",
        "acc3 <- add(acc3, acc3__blk)",
        "acc1 <- add(acc1, acc1__o__gn)",
    )

    out = merge_sibling_reduce_loops(parse_body(_lines(pivot, expectation, denominator, *combine)))

    loops = [s for s in out if isinstance(s, Loop)]
    assert len(loops) == 2, "the pivot stays apart; the two channels merge"
    assert [s.name for s in loops[0].body if isinstance(s, Accum)] == ["acc1__blk"]
    assert [s.name for s in loops[1].body if isinstance(s, Accum)] == ["acc5__sum__blk", "acc3__blk"]
    (score_loop,) = [s for s in loops[1].body if isinstance(s, Loop)]
    assert len(score_loop.body.accums) == 2, "both score states share one inner loop"
