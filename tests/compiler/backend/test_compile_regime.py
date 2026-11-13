"""The regime a compile is in, as its context spells it: every compile is the deployable one (opt level 3), and the
two whole-kernel precision decisions spell the rest. A stored row names the regime it was measured in by the same
flags, cicc's opt level among them for a row measured under nvcc, and the context key reads every spelling of one
regime as one. Hermetic: no GPU, no ptxas."""

from __future__ import annotations

from emmy.compiler.context import Context, precision_flags, split_opt_level


def test_a_live_compile_spells_only_its_precision_decisions(monkeypatch) -> None:
    monkeypatch.delenv("EMMY_FAST_MATH", raising=False)
    assert Context.from_target((8, 0)).compile_flags == "--ftz=true"
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    assert Context.from_target((8, 0)).compile_flags == ""
    assert Context.from_target((8, 0)).features()["H_opt"] == 3.0


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


def test_a_stored_regime_keys_one_way_however_it_is_spelled() -> None:
    """``""`` and an explicit ``-Xcicc -O3`` are the SAME physical regime and must key the same, so a row measured
    under nvcc with an explicit -O3 pin answers a compile. A row measured at another opt level keys apart: it must
    never answer for a deployable measurement."""

    def key_for(flags: str) -> str:
        return Context.from_target((8, 0), compile_flags=flags).structural_key()

    assert key_for("") == key_for("-Xcicc -O3") == key_for("  -Xcicc   -O3  ")
    assert key_for("-Xcicc -O1") != key_for("")
    assert key_for("--fmad=false") != key_for("")
    assert key_for("--fmad=false -Xcicc -O3") == key_for("--fmad=false")
    assert key_for("--fmad=false -Xcicc -O1") != key_for("--fmad=false")


def test_h_opt_and_context_key_agree_on_the_regime() -> None:
    """The deploy evidence tiers gate on ``H_opt`` and then look rows up by ``structural_key``, so the two must never
    disagree about what regime a stored row is in."""
    for flags, opt in (("", 3.0), ("-Xcicc -O3", 3.0), ("-Xcicc -O1", 1.0), ("--use_fast_math", 3.0)):
        assert Context.from_target((8, 0), compile_flags=flags).features()["H_opt"] == opt
    # The residual keeps every other flag, so a same-opt pair still separates on codegen.
    assert split_opt_level("-Xcicc -O1") == (1, "")
    assert split_opt_level("--use_fast_math -Xcicc -O3") == (3, "--use_fast_math")
    assert split_opt_level("") == (3, "")
