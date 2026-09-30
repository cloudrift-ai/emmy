"""Every corpus kernel reuses pure values already available in its lexical scope."""

import json
from dataclasses import replace

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Cond, Load, Loop, Write
from emmy.compiler.ir.stmt.body import free_names
from emmy.compiler.ir.stmt.normalize import _normalize_body, normalize_body
from tests.compiler.realization.helpers import CASES_DIR, case_files


def _assert_scoped_cse(body, available=()):
    """Audit surviving pairs directly; do not call the value-numbering pass or its key builder.

    Only pure bindings are claimed here. Reductions and mutable state have separate semantic
    tests. Writes conservatively invalidate the whole buffer, including a loop's back edge.
    """
    available = list(available)
    for stmt in body:
        # A redefinition changes the value of an operand or of the earlier representative.
        rebound = set(stmt.defines()) | {name for child in stmt.nested() for name in child.carried_names}
        available = [prior for prior in available if not rebound.intersection((*prior.defines(), *free_names(prior)))]
        if stmt.nested():
            writes = {name for child in stmt.nested() for member in child.iter() for name in member.external_writes()}
            for child in stmt.nested():
                shadowed = child.local_defs | stmt.binds_axes()
                inherited = [
                    prior
                    for prior in available
                    if not shadowed.intersection((*prior.defines(), *free_names(prior)))
                    and (isinstance(stmt, Cond) or not writes.intersection(prior.external_reads()))
                ]
                _assert_scoped_cse(child, inherited)
        else:
            writes = set(stmt.external_writes())
            if stmt.pure and stmt.defines() and not (isinstance(stmt, Load) and stmt.carried):
                candidate = stmt
                if isinstance(candidate, Assign) and candidate.op.commutative:
                    candidate = replace(candidate, args=tuple(sorted(candidate.args)))
                for prior in available:
                    renamed = prior.rename(dict(zip(prior.defines(), candidate.defines(), strict=False)))
                    if isinstance(renamed, Assign) and renamed.op.commutative:
                        renamed = replace(renamed, args=tuple(sorted(renamed.args)))
                    assert candidate != renamed, f"available duplicate: {prior.pretty()} followed by {stmt.pretty()}"
                available.append(stmt)
        available = [prior for prior in available if not writes.intersection(prior.external_reads())]


@pytest.mark.parametrize("path", case_files(), ids=lambda path: path.relative_to(CASES_DIR).as_posix())
def test_corpus_kernels_have_no_available_duplicates(path):
    document = json.loads(path.read_text())
    kernels = [node for graph in document["loops"] for node in graph["nodes"] if node["op"] == "loop"]
    assert kernels, "the corpus case must exercise at least one kernel"
    for kernel in kernels:
        normalized = normalize_body(Body.from_wire(kernel["attrs"]["body"]))
        _assert_scoped_cse(normalized)
        # Bypass the Body cache: idempotence is checked, not assumed from cached equality.
        assert _normalize_body(normalized) == normalized


@pytest.mark.parametrize("nested", [False, True])
def test_audit_rejects_an_available_duplicate(nested):
    first = Load("first", "x", (Literal(0, "int"),))
    duplicate = Load("second", "x", (Literal(0, "int"),))
    with pytest.raises(AssertionError, match="available duplicate"):
        _assert_scoped_cse(Body((first, Loop(Axis("i", 4), (duplicate,)) if nested else duplicate)))


def test_audit_respects_writes_and_shadowing():
    first = Load("first", "x", (Var("i"),))
    duplicate = Load("second", "x", (Var("i"),))
    _assert_scoped_cse(Body((first, Write("x", (Var("i"),), "first"), duplicate)))
    _assert_scoped_cse(Body((first, Loop(Axis("i", 4), (duplicate,)))))


def test_audit_respects_carried_state_updates():
    _assert_scoped_cse(
        Body((Assign("before", "exp", ("sum",)), Loop(Axis("k", 4), (Accum("sum", "x"),), seed=False), Assign("after", "exp", ("sum",))))
    )
