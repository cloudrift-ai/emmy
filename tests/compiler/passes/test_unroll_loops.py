"""``117_unroll_loops`` replaces every loop flagged ``unroll`` by one copy of its body per trip, so a loop left in a
kernel body is a loop in its code."""

from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.kernel.ir import KernelOp, MmaSyncPtx, Tile
from emmy.compiler.ir.stmt import Accum, Body, Cond, Init, Load, Loop, Select, StridedLoop, Write, parse_body

_RULE = import_module("emmy.compiler.pipeline.passes.lowering.kernel.117_unroll_loops")


def _unrolled(body: str, lanes: int = 4) -> KernelOp:
    """``body`` over a row axis ``i`` and a lane axis ``lane``, through the pass."""
    tiled = KernelOp(body=Body((Tile(axes=(Axis("i", 8), Axis("lane", lanes)), body=parse_body(body)),)), name="k")
    return _RULE.rewrite(SimpleNamespace(op=tiled))


def _loops(op: KernelOp) -> list:
    return [stmt for stmt in op.body.iter() if isinstance(stmt, (Loop, StridedLoop))]


def test_a_flagged_loop_becomes_one_copy_per_trip() -> None:
    """Each copy reads its trip's cell, the carrier is seeded once ahead of them, and no loop is left."""
    op = _unrolled("for k in 0..4 unroll=true:\n    f32 a = load A[i, k]\n    f32 acc <- add(acc, a)\nf32 O[i] = acc")
    assert not _loops(op)
    assert [str(stmt.index[1]) for stmt in op.body.iter() if isinstance(stmt, Load)] == ["0", "1", "2", "3"]
    assert sum(isinstance(stmt, Init) for stmt in op.body.iter()) == 1
    assert sum(isinstance(stmt, Accum) for stmt in op.body.iter()) == 4


def test_a_guard_the_trip_decides_is_dropped() -> None:
    """A copy holds the branch its trip takes: the first two trips store, the last two do not."""
    op = _unrolled("for k in 0..4 unroll=true:\n    f32 a = load A[i, k]\n    if k < 2:\n        B[i, k] = a\n    C[i, k] = a")
    assert not any(isinstance(stmt, Cond) for stmt in op.body.iter())
    stores = [(stmt.output, str(stmt.index[1])) for stmt in op.body.iter() if isinstance(stmt, Write)]
    assert sorted(stores) == [("B", "0"), ("B", "1"), ("C", "0"), ("C", "1"), ("C", "2"), ("C", "3")]


@pytest.mark.parametrize(("loop", "kept"), [("for k in 0..4 unroll=true:", 3), ("for k in 0..8 step 2 unroll=true:", 2)])
def test_a_copy_reads_its_trip_as_an_integer_in_a_value_field(loop: str, kept: int) -> None:
    """A select's predicate over coordinates is a value field: ``i - k / 3 < 7`` reads ``k / 3`` as the integer
    quotient in every copy: a predicate is ``i < 7`` where the quotient is zero, a select whose quotient is not folds
    away under the row's range, and no predicate holds a float."""
    select = "    v = select:\n        a when i - k / 3 < 7\n        z when true\n"
    op = _unrolled(f"let f32 z = 0.0\n{loop}\n    f32 a = load A[i, k]\n{select}    f32 O[i, k] = v")
    predicates = [str(branch.select) for stmt in op.body.iter() if isinstance(stmt, Select) for branch in stmt.branches]
    assert predicates == ["i < 7", "true"] * kept, predicates


def test_the_copies_of_two_loops_keep_their_names_apart() -> None:
    """Two loops of one scope define the same names, each in its own; unrolled into that scope every definition
    keeps a name of its own."""
    loop = "for k in 0..2 unroll=true:\n    f32 a = load {src}[i, k]\n    w = exp(a)\n    {dst}[i, k] = w"
    op = _unrolled(loop.format(src="A", dst="O") + "\n" + loop.format(src="B", dst="P"))
    names = [name for stmt in op.body.iter() for name in stmt.defines()]
    assert len(names) == len(set(names)) == 8


def test_two_carriers_whose_last_copies_fold_one_value_stay_two_values() -> None:
    """Each copy's fold is one step on what its carrier holds. ``q`` folds ``A[i, 1]`` on every trip and ``p`` only on
    the last; the copies' optimization keeps the two carriers, and every value read from them, apart."""
    op = _unrolled(
        "for k in 0..2 unroll=true:\n    f32 a = load A[i, k]\n    f32 b = load A[i, 1]\n    f32 p <- maximum(p, a)\n"
        "    f32 q <- maximum(q, b)\nf32 r = maximum(p, q)\nf32 s = p - r\nf32 t = q - r\nf32 O[i] = s\nf32 P[i] = t"
    )
    stores = {stmt.output: stmt.values for stmt in op.body.iter() if isinstance(stmt, Write)}
    assert stores["O"] != stores["P"]


def test_storage_declared_around_the_loop_keeps_its_name() -> None:
    """A fragment declared ahead of the loop is one accumulator: every copy updates it, under its own name."""
    body = (
        "RegFragment(name=c, role=c, dtype=f32)\n"
        "RegFragment(name=a, role=a, dtype=f16)\n"
        "for k in 0..32 step 16 unroll=true:\n"
        "    RegFragment(name=b, role=b, dtype=f16)\n"
        "    LdmatrixLoad(frag=b, src_buffer=B, src_index=(k, i), role=b, ldm=8)\n"
        "    MmaSyncPtx(c_frag=c, a_frag=a, b_frag=b, shape=(16, 8, 16))"
    )
    mmas = [stmt for stmt in _unrolled(body).body.iter() if isinstance(stmt, MmaSyncPtx)]
    assert [(stmt.c_frag, stmt.a_frag, stmt.b_frag) for stmt in mmas] == [("c", "a", "b__u0"), ("c", "a", "b__u1")]


def test_a_loop_that_takes_copies_seeds_its_own_folds_only() -> None:
    """A loop seeds the folds that stand in its body. The copies of an inner loop bring folds the outer loop never
    seeded: it keeps seeding its own, as a statement ahead of it, and ``acc`` keeps the one seed it had."""
    body = (
        "init f32 acc = 0.0\n"
        "for c in 0..2:\n"
        "    f32 b = load B[i, c]\n"
        "    f32 total <- add(total, b)\n"
        "    for k in 0..2 unroll=true seed=false:\n"
        "        f32 a = load A[i, c, k]\n"
        "        f32 acc <- add(acc, a)\n"
        "f32 O[i] = acc\n"
        "f32 P[i] = total"
    )
    op = _unrolled(body)
    [loop] = _loops(op)
    assert not loop.seed and sum(isinstance(stmt, Accum) and stmt.name == "acc" for stmt in loop.body) == 2
    assert sorted(stmt.name for stmt in op.body.iter() if isinstance(stmt, Init)) == ["acc", "total"]


@pytest.mark.parametrize(("extent", "copies"), [(8, 2), (16, 4)])
def test_a_lane_loop_unrolls_when_every_lane_runs_the_same_trips(extent: int, copies: int) -> None:
    op = _unrolled(f"for j in lane..{extent} step 4 unroll=true:\n    f32 a = load A[i, j]\n    O[i, j] = a")
    assert not _loops(op)
    cells = [str(stmt.index[1]) for stmt in op.body.iter() if isinstance(stmt, Load)]
    assert cells == ["lane", *(f"lane + {4 * k}" for k in range(1, copies))]


def test_a_trip_only_some_lanes_run_keeps_the_loop_bound_as_its_guard() -> None:
    """Six cells over four lanes: every lane runs the first trip, lanes 0 and 1 a second. Sixteen lanes over six
    cells: a lane runs one trip or none."""
    op = _unrolled("for j in lane..6 step 4 unroll=true:\n    f32 a = load A[i, j]\n    f32 O[i, j] = a")
    assert not _loops(op)
    first, guard = op.body[0].body[:2], op.body[0].body[2]
    assert [type(stmt) for stmt in first] == [Load, Write] and str(guard.cond) == "lane + 4 < 6"
    assert [str(stmt.index[1]) for stmt in guard.body] == ["lane + 4", "lane + 4"]
    [guard] = _unrolled("for j in lane..6 step 16 unroll=true:\n    f32 a = load A[i, j]\n    f32 O[i, j] = a", 16).body[0].body
    assert str(guard.cond) == "lane < 6" and len(guard.body) == 2


def test_a_long_loop_keeps_rolling_over_copies_of_its_body() -> None:
    """A 128-trip K loop runs 32 copies of its body per trip, the ``n``-th at the axis ``n`` steps on, and says it is
    unrolled, so no code generator unrolls it again; the fragment it accumulates into and the one it reloads keep
    their names."""
    body = (
        "RegFragment(name=c, role=c, dtype=f32)\n"
        "RegFragment(name=a, role=a, dtype=f16)\n"
        "RegFragment(name=b, role=b, dtype=f16)\n"
        "for k in 0..2048 step 16 unroll=true:\n"
        "    LdmatrixLoad(frag=b, src_buffer=B, src_index=(k, i), role=b, ldm=8)\n"
        "    MmaSyncPtx(c_frag=c, a_frag=a, b_frag=b, shape=(16, 8, 16))"
    )
    [loop] = _loops(_unrolled(body))
    assert (str(loop.start), str(loop.step), loop.unroll, loop.unrolled) == ("0", "512", False, True)
    loads = [stmt for stmt in loop.body if not isinstance(stmt, MmaSyncPtx)]
    assert [str(stmt.src_index[0]) for stmt in loads] == ["k", *(f"k + {16 * n}" for n in range(1, 32))]
    assert {(stmt.c_frag, stmt.b_frag) for stmt in loop.body if isinstance(stmt, MmaSyncPtx)} == {("c", "b")}


def test_a_loop_with_no_static_bound_on_its_trips_stays_a_loop_and_loses_the_flag() -> None:
    """An early stop is a bound only the kernel's values give: the loop stays, and asks for nothing."""
    op = _unrolled("for j in 0..8 step 4 until j_end unroll=true:\n    f32 a = load A[i, j]\n    f32 O[i, j] = a")
    [loop] = _loops(op)
    assert not loop.unroll


def test_a_lowered_matmul_reaches_code_generation_with_no_loop_flagged() -> None:
    """The scalar tier flags its 64-trip K loop where it lowers it; the kernel passes hand over its copies, and the
    kernel's source asks the compiler for nothing."""
    from emmy.compiler.ir.cuda.ir import CudaOp
    from tests.compiler.realization import helpers

    case = helpers.load_case(helpers.CASES_DIR / "matmul" / "f32-epilogue-bias.json")
    graph, _ = helpers.lowered(case, case.context())
    (cuda,) = [node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    kernel = next(op for op in cuda.source_chain() if isinstance(op, KernelOp))
    assert not _loops(kernel)
    folds = sum(isinstance(stmt, Accum) for stmt in kernel.body.iter())
    assert folds and folds % 64 == 0  # one fold per trip and register cell
    # CUDA spelling: moves to the PTX compiler's tests
    assert "#pragma unroll" not in cuda.kernel_source
