"""The statement protocol: what a statement kind owes, checked over every kind in every IR layer."""

import dataclasses
import importlib
import pickle
import pkgutil

import pytest

import emmy.compiler.ir as ir_pkg
import emmy.compiler.ir.stmt as stmt_pkg
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.elementwise import ElementwiseImpl
from emmy.compiler.ir.expr import BinaryExpr, Expr, Literal, Var
from emmy.compiler.ir.kernel import ir as kernel_ir
from emmy.compiler.ir.stmt import (
    Accum,
    Affine,
    Assign,
    Body,
    Carry,
    Cond,
    Declared,
    Init,
    Let,
    Load,
    Loop,
    Pre,
    Select,
    SelectBranch,
    Stmt,
    StridedLoop,
    Write,
    ZeroPrologue,
    map_exprs,
)


def _kinds() -> list[type]:
    for mod in pkgutil.walk_packages(ir_pkg.__path__, prefix="emmy.compiler.ir."):
        importlib.import_module(mod.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    return sorted((cls for cls in set(subclasses(Stmt)) if cls is not Declared), key=lambda cls: cls.__qualname__)


def test_a_kind_check_is_native() -> None:
    """No statement class has a metaclass of its own: ``isinstance`` on a kind stays the interpreter's check."""
    assert [cls.__qualname__ for cls in (Stmt, *_kinds()) if type(cls) is not type] == []


def test_every_kind_states_the_protocol() -> None:
    """A kind says how it rewrites and prints, and each reading half comes with its rebuilding half.

    A kind that lists child bodies rebuilds from bodies; one that lists expressions rebuilds from expressions.
    """
    missing = []
    for cls in _kinds():
        if cls._rewrite is Stmt._rewrite:
            missing.append(f"{cls.__name__}: _rewrite")
        if cls.pretty is Stmt.pretty:
            missing.append(f"{cls.__name__}: pretty")
        if (cls.nested is Stmt.nested) != (cls.with_bodies is Stmt.with_bodies):
            missing.append(f"{cls.__name__}: nested and with_bodies come together")
        if (cls.exprs is Stmt.exprs) != (cls.with_exprs is Stmt.with_exprs):
            missing.append(f"{cls.__name__}: exprs and with_exprs come together")
    assert not missing, "these statement kinds leave part of the protocol unstated:\n" + "\n".join(missing)


_K = Axis("k", 8)
_BELOW = BinaryExpr("<", Var("i"), Literal(2, "int"))
_EXAMPLES = [
    Load("x", "a", (Var("i"),), dtype=F32),
    Assign("y", "exp", ("x",)),
    Affine("z", ("x", "y"), (2.0, -1.0), offset=0.5),
    Accum("s", "x", "add"),
    Carry("c", "x", (Var("i"),), seed=0.0),
    Pre("p", "c", (Var("i"),)),
    Init("s", 0.0, dtype=F32),
    Let("n", BinaryExpr("+", Var("i"), Literal(1, "int"))),
    Write("out", (Var("i"),), "x"),
    Select("m", (SelectBranch("x", _BELOW), SelectBranch("y", Literal(1, "int")))),
    ZeroPrologue(dst="out", words=16),
    Loop(_K, (Load("x", "a", (Var("k"),)), Accum("s", "x", "add"))),
    StridedLoop(_K, Literal(0, "int"), Literal(2, "int"), (Load("x", "a", (Var("k"),)),), end=Var("n")),
    Cond(_BELOW, (Write("out", (), "x"),), (Write("out", (), "y"),)),
]


_INDEX = (Var("i"), Var("j"))
_COMBINE = (Assign("t", "multiply", ("acc", "acc__o")), Assign("acc", "add", ("acc", "t")))
_OPERANDS = kernel_ir.fragment_args(
    (BinaryExpr(">", Var(kernel_ir.FRAG_COL), Var(kernel_ir.FRAG_ROW)), "u", ("r0", "r1"), "f", ("mask", _INDEX)),
    (kernel_ir.COORD, kernel_ir.UNIFORM, kernel_ir.ROW, kernel_ir.FRAG, kernel_ir.GMEM),
)
_KERNEL_EXAMPLES = [
    kernel_ir.Smem("s", (16, 16)),
    kernel_ir.Sync(),
    kernel_ir.Tile(axes=(Axis("m", 4), Axis("n", 8)), body=(Load("x", "a", (Var("m"), Var("n"))),), raster_axes=("m", "n"), raster_group=2),
    kernel_ir.CpAsyncCopy(smem="s", smem_index=_INDEX, src="a", src_index=_INDEX, lane_index=_INDEX, valid=_BELOW),
    kernel_ir.CpAsyncCommit(),
    kernel_ir.CpAsyncWait(1),
    kernel_ir.TmaDescriptor(name="d", src_buf="a", src_shape=(64, 64), box_extents=(16, 16)),
    kernel_ir.TmaLoad(smem="s", smem_index=_INDEX, desc="d", coords=_INDEX, mbar="mb", mbar_slot=Var("p")),
    kernel_ir.MbarrierInit("mb", slot=Var("p")),
    kernel_ir.MbarrierArriveExpectTx("mb", 64, slot=Var("p")),
    kernel_ir.MbarrierArrive("mb", slot=Var("p")),
    kernel_ir.SetMaxNReg(64, "dec"),
    kernel_ir.MbarrierWait("mb", phase=Var("ph"), slot=Var("p")),
    kernel_ir.TreeHalve(bufs=("b",), state=("acc",), state_b=("acc__o",), combine_states=_COMBINE, length=8, tid_var="t"),
    kernel_ir.WarpShuffle(state=("acc",), state_b=("acc__o",), combine_states=_COMBINE, length=8),
    kernel_ir.WarpBroadcast(state=("acc",), length=8),
    kernel_ir.FragmentApply(out="o", op=ElementwiseImpl("where"), args=_OPERANDS, row_base=Var("q"), col_base=Var("kv")),
    kernel_ir.FragmentRowReduce(top="t0", bot="b0", frags=("f0", "f1"), op=ElementwiseImpl("maximum")),
    kernel_ir.RegFragment(name="f", role="c", shape=(16, 8, 16), dtype=F32),
    kernel_ir.BlockScaleLoad(frag="sf", src_buffer="s", src_index=_INDEX, role="a"),
    kernel_ir.LdmatrixLoad(
        frag="f",
        src_buffer="s",
        src_index=_INDEX,
        role="b",
        gmem_guard=_INDEX,
        k_zero=_INDEX,
        pair_frag="g",
        scale_buffer="sc",
        scale_index=_INDEX,
    ),
    kernel_ir.MmaSyncPtx(c_frag="c", a_frag="a", b_frag="b", shape=(16, 8, 16), sfa_frag="sa", sfb_frag="sb"),
    kernel_ir.FragmentPromote(dst="c", src="h"),
    kernel_ir.FragmentRepack(frag="a", srcs=("c0", "c1")),
    kernel_ir.WgmmaDescriptor(name="bD", smem="s", smem_index=Var("i"), swizzle="B128", lbo_bytes=16, sbo_bytes=1024),
    kernel_ir.WgmmaMma(c_frags=tuple(f"c{i}" for i in range(8)), b_desc="bD", shape=(64, 64, 16), a_desc="aD"),
    kernel_ir.WgmmaFence(),
    kernel_ir.WgmmaCommit(),
    kernel_ir.WgmmaWait(),
    kernel_ir.RegStore(
        dst_buffer="out",
        dst_index=_INDEX,
        frag="c",
        shape=(16, 8, 16),
        epilogue=Body((Load("r", "res", _INDEX), Assign("y", "add", ("acc", "r")))),
        epilogue_params=("acc",),
        epilogue_result="y",
        extra_frags=("c2",),
        m_guard=_INDEX,
        n_guard=_INDEX,
        run=(kernel_ir.RegStore(dst_buffer="out", dst_index=_INDEX, frag="d", shape=(16, 8, 16)),),
    ),
]
_ALL_EXAMPLES = _EXAMPLES + _KERNEL_EXAMPLES


def test_every_exported_kind_has_an_example() -> None:
    exported = {kind for name in stmt_pkg.__all__ if isinstance(kind := getattr(stmt_pkg, name), type) and issubclass(kind, Stmt)}
    assert {type(example) for example in _EXAMPLES} == exported - {Stmt, Declared}
    kernel = {kind for kind in _kinds() if kind.__module__ == kernel_ir.__name__}
    assert {type(example) for example in _KERNEL_EXAMPLES} == kernel


@pytest.mark.parametrize("stmt", _KERNEL_EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_a_declared_kind_states_only_its_fields(stmt) -> None:
    """A Kernel IR kind is declared: its protocol members are the ones its fields' roles give it."""
    stated = set(vars(type(stmt))) & {name for name in vars(Declared) if not name.startswith("__")}
    assert not stated


@pytest.mark.parametrize("stmt", _KERNEL_EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_a_rename_reaches_every_name_and_no_buffer(stmt) -> None:
    """What a declared kind reads and defines follows a rename; the buffers it reads and writes follow a buffer
    rename, the ones a carried statement or an epilogue reads included."""
    renamed = stmt.rewrite(lambda name: f"{name}_")
    assert renamed.deps() == tuple(f"{name}_" for name in stmt.deps())
    assert renamed.defines() == tuple(f"{name}_" for name in stmt.defines())
    assert (renamed.external_reads(), renamed.external_writes(), renamed.local_decls()) == (
        stmt.external_reads(),
        stmt.external_writes(),
        tuple(f"{name}_" if name in stmt.defines() else name for name in stmt.local_decls()),
    )
    buffers = {name: f"{name}_" for name in (*stmt.external_reads(), *stmt.external_writes())}
    moved = stmt.rename_buffers(buffers)
    assert moved.external_reads() == tuple(buffers[name] for name in stmt.external_reads())
    assert moved.external_writes() == tuple(buffers[name] for name in stmt.external_writes())
    assert (moved.deps(), moved.defines()) == (stmt.deps(), stmt.defines())


def test_a_carried_program_hides_its_own_names() -> None:
    """A combine a statement carries reads the state and its shuffled copy and binds temporaries; the statement reads
    the state alone."""
    halve = _KERNEL_EXAMPLES[13]
    assert halve.deps() == ("acc",) and halve.defines() == ()
    assert halve.rewrite(lambda name: f"{name}_").combine_states[1] == Assign("acc_", "add", ("acc_", "t_"))


def test_a_term_keeps_its_names_and_follows_a_substitution() -> None:
    """A store's epilogue names are its own: a rename leaves them and reaches the coordinates its indices read."""
    from emmy.compiler.ir.sigma import Sigma

    store = _KERNEL_EXAMPLES[-1]
    moved = store.rewrite(lambda name: f"{name}_", Sigma({"i": Var("i") + 4}))
    assert moved.epilogue[1] == Assign("y", "add", ("acc", "r")) and moved.epilogue_params == ("acc",)
    assert moved.epilogue[0] == Load("r", "res", moved.dst_index) and moved.dst_index == (Var("i_") + 4, Var("j_"))
    assert (moved.frag, moved.extra_frags, moved.run[0].frag) == ("c_", ("c2_",), "d_")


def test_a_bound_axis_name_follows_its_axis() -> None:
    tile = _KERNEL_EXAMPLES[2].replace(spread=True)
    renamed = tile.rename({"m": "m2", "x": "x2"})
    assert [axis.name for axis in renamed.axes] == ["m2", "n"] and renamed.raster_axes == ("m2", "n")
    assert renamed.body == Body((Load("x2", "a", (Var("m2"), Var("n"))),)) and renamed.spread
    assert renamed.binds_axes() == {"m2", "n"}


@pytest.mark.parametrize("stmt", _ALL_EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_a_statement_rebuilds_as_itself(stmt) -> None:
    """Every rebuilding member, given what its reading half returned, gives back an equal statement."""
    assert stmt.rewrite(lambda name: name) == stmt
    assert stmt.with_bodies(stmt.nested()) == stmt
    assert stmt.with_exprs(tuple(stmt.exprs())) == stmt
    assert map_exprs(stmt, lambda expr: expr) is stmt
    assert stmt.rename_buffers({}) == stmt


def _held_exprs(value, child_bodies=()) -> list:
    """Every expression a statement holds in its fields, found by reflection: the check on what ``exprs`` reports.
    A child body's expressions are its own statements'; any other statement a kind holds is part of it."""
    if isinstance(value, Expr):
        return [value]
    if isinstance(value, (tuple, list)) and not any(value is body for body in child_bodies):
        return [expr for item in value for expr in _held_exprs(item)]
    if isinstance(value, (Stmt, SelectBranch, kernel_ir.FragmentArg)):
        nested = value.nested() if isinstance(value, Stmt) else ()
        return [expr for f in dataclasses.fields(value) for expr in _held_exprs(getattr(value, f.name), nested)]
    return []


@pytest.mark.parametrize("stmt", _ALL_EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_an_expression_map_reaches_every_expression_a_statement_holds(stmt) -> None:
    """``exprs`` lists every expression of the statement and ``with_exprs`` puts each one back in its place."""
    held = _held_exprs(stmt)
    assert sorted(map(repr, stmt.exprs())) == sorted(map(repr, held))
    markers = tuple(Var(f"e{index}") for index in range(len(held)))
    fresh = iter(markers)
    mapped = map_exprs(stmt, lambda _expr: next(fresh))
    assert tuple(mapped.exprs()) == markers
    assert sorted(map(repr, _held_exprs(mapped))) == sorted(map(repr, markers))
    assert mapped.with_exprs(tuple(stmt.exprs())) == stmt


@pytest.mark.parametrize("stmt", _EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_a_statement_survives_the_wire_form_and_pickling(stmt) -> None:
    body = Body((stmt,))
    assert Body.from_wire(body.to_wire()) == body
    assert pickle.loads(pickle.dumps(body)) == body


def test_a_rename_reaches_names_and_expressions() -> None:
    names = {"x": "x2", "i": "j"}
    load, write = Load("x", "a", (Var("i"),)), Write("out", (Var("i"),), "x")
    assert load.rename(names) == Load("x2", "a", (Var("j"),))
    assert write.rename(names) == Write("out", (Var("j"),), "x2")
    assert map_exprs(load, lambda expr: expr.rename(lambda name: names.get(name, name))) == Load("x", "a", (Var("j"),))


def test_a_body_is_an_immutable_sequence_of_its_statements() -> None:
    """What callers may rely on in a ``Body``: it is built from any iterable, a slice or a sum of bodies is a
    body, two bodies with equal statements are equal and hash alike, and a statement placed in one is the same
    object every time it is read — passes key their side tables on that object."""
    load, write = Load("x", "a", (Var("i"),)), Write("out", (Var("i"),), "x")
    body = Body(stmt for stmt in [load, write])
    assert len(body) == 2 and list(body) == [load, write] and write in body
    assert isinstance(body[:1], Body) and isinstance(body + [load], Body) and isinstance([load] + body, Body)
    assert body == Body([load, write]) and hash(body) == hash(Body((load, write))) and body != Body((write, load))
    assert body[0] is load and next(iter(body)) is load and body[-1] is write
    looped = Body((Loop(_K, body),))
    assert [id(stmt) for stmt in looped.iter()] == [id(looped[0]), id(load), id(write)]
    assert looped[0].nested()[0][0] is load
    with pytest.raises(TypeError):
        Body((load, "not a statement"))
