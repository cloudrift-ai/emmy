"""The memos are inspected and cleared from Python: what a test worker or a process pool does to start clean."""

from __future__ import annotations

from emmy.compiler.ir.stmt import canonicalize_identity, clear_memos, memo_stats, normalize_body, parse_body


def _by_name() -> dict[str, dict]:
    return {stats["name"]: stats for stats in memo_stats()}


def test_the_memos_report_their_entries_and_clear_together() -> None:
    body = parse_body("x = load X[a]\nO[a] = x")
    normalize_body(body)
    canonicalize_identity(body)
    body[0]  # a statement read: the wrapper table has an entry
    stats = _by_name()
    assert stats["normal_forms"]["entries"] >= 1
    assert stats["identities"]["entries"] >= 1
    # The body memos are bounded by the statements they hold between them, and report how many they hold now.
    assert stats["normal_forms"]["capacity"] == stats["identities"]["capacity"] == 1_000_000
    assert 0 < stats["normal_forms"]["weight"] <= stats["normal_forms"]["capacity"]
    assert stats["statement_wrappers"]["capacity"] is None and stats["statement_wrappers"]["weight"] is None
    clear_memos()
    stats = _by_name()
    assert stats["normal_forms"]["entries"] == 0
    assert stats["identities"]["entries"] == 0


def test_a_second_normalization_is_a_hit() -> None:
    normalize_body(parse_body("y = load Y[b]\nP[b] = y"))
    before = _by_name()["normal_forms"]["hits"]
    normalize_body(parse_body("y = load Y[b]\nP[b] = y"))
    assert _by_name()["normal_forms"]["hits"] == before + 1
