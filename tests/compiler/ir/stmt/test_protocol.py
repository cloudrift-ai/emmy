"""The statement protocol: what a statement kind owes, checked over every kind in every IR layer."""

import importlib
import pickle
import pkgutil

import pytest

import emmy.compiler.ir as ir_pkg
import emmy.compiler.ir.stmt as stmt_pkg
from emmy.compiler.dtype import F32
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.stmt import (
    Accum,
    Affine,
    Assign,
    Body,
    Carry,
    Cond,
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

    return sorted(set(subclasses(Stmt)), key=lambda cls: cls.__qualname__)


def test_every_kind_states_the_protocol() -> None:
    """A kind says how it rewrites and prints, and each reading half comes with its rebuilding half.

    A kind that lists child bodies rebuilds from bodies; one that lists expressions rebuilds from expressions. The
    second half is not asked of Kernel IR's kinds: nothing maps expressions over a Kernel IR body, and the base
    refuses, by name, the day something does.
    """
    missing = []
    for cls in _kinds():
        if cls._rewrite is Stmt._rewrite:
            missing.append(f"{cls.__name__}: _rewrite")
        if cls.pretty is Stmt.pretty:
            missing.append(f"{cls.__name__}: pretty")
        if (cls.nested is Stmt.nested) != (cls.with_bodies is Stmt.with_bodies):
            missing.append(f"{cls.__name__}: nested and with_bodies come together")
        shared = cls.__module__.startswith(stmt_pkg.__name__)
        if shared and (cls.exprs is Stmt.exprs) != (cls.with_exprs is Stmt.with_exprs):
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


def test_every_exported_kind_has_an_example() -> None:
    exported = {kind for name in stmt_pkg.__all__ if isinstance(kind := getattr(stmt_pkg, name), type) and issubclass(kind, Stmt)}
    assert {type(example) for example in _EXAMPLES} == exported - {Stmt}


@pytest.mark.parametrize("stmt", _EXAMPLES, ids=lambda stmt: type(stmt).__name__)
def test_a_statement_rebuilds_as_itself(stmt) -> None:
    """Every rebuilding member, given what its reading half returned, gives back an equal statement."""
    assert stmt.rewrite(lambda name: name) == stmt
    assert stmt.with_bodies(stmt.nested()) == stmt
    assert stmt.with_exprs(tuple(stmt.exprs())) == stmt
    assert map_exprs(stmt, lambda expr: expr) is stmt
    assert stmt.rename_buffers({}) == stmt


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
