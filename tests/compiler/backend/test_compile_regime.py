"""The regime a compile is in, as its context keys it: every precision decision. Hermetic: no GPU, no ptxas."""

from __future__ import annotations

from emmy.compiler.context import Context


def test_a_live_compile_keys_its_precision_decisions(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    fast = Context.from_target((8, 0))
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    assert Context.from_target((8, 0)).structural_key() != fast.structural_key()
    assert Context.from_target((8, 0)).structural_key() == Context.from_target((8, 0), precision={}).structural_key()


def test_a_whole_kernel_precision_decision_keys_its_own_regime() -> None:
    """A row measured under another flush or contraction decision timed other machine code, so it never answers for
    the plain regime."""

    def key_for(**pins: bool) -> str:
        return Context.from_target((8, 0), precision=pins).structural_key()

    assert key_for(CONTRACT_FMA=False) != key_for()
    assert key_for(FLUSH_TO_ZERO=True) != key_for()
