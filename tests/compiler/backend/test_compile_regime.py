"""The regime a compile is in, as its context spells it: the two whole-kernel precision decisions, each where it is
not the default. Hermetic: no GPU, no ptxas."""

from __future__ import annotations

from emmy.compiler.context import Context, precision_flags


def test_a_live_compile_spells_only_its_precision_decisions(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    assert Context.from_target((8, 0)).compile_flags == "--ftz=true"
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    assert Context.from_target((8, 0)).compile_flags == ""


def test_precision_flags_spell_the_two_whole_kernel_decisions() -> None:
    """Flush to zero and multiply-add fusion hold for a whole kernel, each spelled where it is not the default. Every
    approximate instruction is an operation of the kernel body, so no other precision decision reaches the spelling."""
    from emmy.compiler.pipeline.search.space import precision_pins

    def flags(**pins) -> list[str]:
        return precision_flags(precision_pins(pins))

    assert flags(FAST_MATH=True) == ["--ftz=true"]
    assert flags(FAST_MATH=False) == []
    assert flags(FAST_MATH=True, FLUSH_TO_ZERO=False) == []
    assert flags(FAST_MATH=True, CONTRACT_FMA=False) == ["--ftz=true", "--fmad=false"]
    assert flags(FAST_MATH=False, FLUSH_TO_ZERO=True, CONTRACT_FMA=False) == ["--ftz=true", "--fmad=false"]
    for knob in ("FAST_EXP", "FAST_DIV", "FAST_SQRT"):
        assert flags(**{"FAST_MATH": True, knob: False}) == ["--ftz=true"]
        assert flags(**{"FAST_MATH": False, knob: True}) == []


def test_a_precision_flag_keys_its_own_regime() -> None:
    """A row measured under another precision flag timed other machine code, so it never answers for the plain
    regime."""

    def key_for(flags: str) -> str:
        return Context.from_target((8, 0), compile_flags=flags).structural_key()

    assert key_for("--fmad=false") != key_for("")
    assert key_for("--ftz=true") != key_for("")
