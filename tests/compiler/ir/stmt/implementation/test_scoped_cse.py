"""Every corpus kernel reuses pure values already available in its lexical scope."""

import json

import pytest
from emmy.emmy_ir import normalize_uncached

from emmy.compiler.ir.stmt import Assign, Body, Cond, Load, canonical_values, free_names, normalize_body, parse_body
from tests.compiler.realization.helpers import CASES_DIR, case_files


def _spelled(stmt):
    """``stmt`` with its value in the canonical spelling, so two spellings of one computation compare equal."""
    return canonical_values(Body((stmt,)))[0] if isinstance(stmt, Assign) else stmt


def _assert_scoped_cse(body, available=()):
    """Audit surviving pairs directly; do not call the value-numbering pass or its key builder.

    Only pure bindings are claimed here. Reductions and mutable state have separate semantic
    tests. Writes conservatively invalidate the whole buffer, including a loop's back edge.
    """
    available = list(available)
    for stmt in body:
        if stmt.pure and not (isinstance(stmt, Load) and stmt.carried) and not set(stmt.defines()).intersection(stmt.deps()):
            assert stmt not in available, f"available duplicate binding: {stmt}"
        # A redefinition changes the value of an operand or of the earlier representative.
        rebound = set(stmt.defines()) | {name for child in stmt.nested() for name in child._carried_names}
        available = [prior for prior in available if not rebound.intersection((*prior.defines(), *free_names(prior)))]
        if stmt.nested():
            writes = {name for child in stmt.nested() for member in child.iter() for name in member.external_writes()}
            for child in stmt.nested():
                shadowed = child._local_defs | stmt.binds_axes()
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
                candidate = _spelled(stmt)
                for prior in available:
                    renamed = _spelled(prior.rename(dict(zip(prior.defines(), candidate.defines(), strict=False))))
                    assert candidate != renamed, f"available duplicate: {prior} followed by {stmt}"
                available.append(stmt)
        available = [prior for prior in available if not writes.intersection(prior.external_reads())]


@pytest.mark.parametrize("path", case_files(), ids=lambda path: path.relative_to(CASES_DIR).as_posix())
def test_corpus_kernels_have_no_available_duplicates(path):
    document = json.loads(path.read_text())
    kernels = [node for kernel in document["kernels"] for node in kernel["loop_ir"]["nodes"] if node["op"] == "loop"]
    assert kernels, "the corpus case must exercise at least one kernel"
    for kernel in kernels:
        normalized = normalize_body(Body.from_wire(kernel["attrs"]["body"]))
        _assert_scoped_cse(normalized)
        # Bypass the Body cache: idempotence is checked, not assumed from cached equality.
        assert normalize_uncached(normalized) == normalized


@pytest.mark.parametrize("nested", [False, True])
def test_audit_rejects_an_available_duplicate(nested):
    duplicate = "for i in 0..4:\n    second = load x[0]" if nested else "second = load x[0]"
    with pytest.raises(AssertionError, match="available duplicate"):
        _assert_scoped_cse(parse_body(f"first = load x[0]\n{duplicate}"))


def test_audit_respects_writes_and_shadowing():
    _assert_scoped_cse(parse_body("first = load x[i]\nx[i] = first\nsecond = load x[i]"))
    _assert_scoped_cse(parse_body("first = load x[i]\nfor i in 0..4:\n    second = load x[i]"))


def test_audit_rejects_repeated_copies_but_respects_self_updates():
    with pytest.raises(AssertionError, match="duplicate binding"):
        _assert_scoped_cse(parse_body("value = load x[0]\nvalue = load x[0]"))
    _assert_scoped_cse(parse_body("value = exp(value)\nvalue = exp(value)"))


def test_audit_respects_carried_state_updates():
    _assert_scoped_cse(
        parse_body(
            """
            before = exp(sum)
            for k in 0..4 seed=false:
                sum <- add(sum, x)
            after = exp(sum)
            """
        )
    )
