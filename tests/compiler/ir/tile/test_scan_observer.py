"""The scan spelling — a ``Fold`` with a per-step observer.

Pins the observer contract end to end at the IR layer: formation (fresh results, positional
state binding, family observability, no observer where the stream is not a stream), identity (a
cumsum keys apart from a sum; α-invariance holds; observer-free digests are untouched), the
rename lockstep, recognition (a reduce loop with a per-step store lifts to an observed fold plus
a boundary store over the observed name), and reconstitution (the streamed store rides inside
the reduce loop, after the observer stmts)."""

from __future__ import annotations

import dataclasses
from textwrap import indent

import pytest

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.pure.fold import Fold
from emmy.compiler.ir.pure.twist import SOFTMAX, Twist
from emmy.compiler.ir.stmt import Loop, Write, parse_body
from emmy.compiler.ir.tile import OutputSpec, extract_output_specs, observed_result_names
from emmy.compiler.pipeline.passes.tile._fromloop import fold_from_loop, lift_loop_op, scan_from_loop


def _scan_text(axis_name: str = "k", acc: str = "acc", out: str = "out", *, store: bool = True) -> str:
    """The 025_lower_scan shape: a reduce loop that both folds and stores per iteration."""
    lines = [
        f"for {axis_name} in 0..16:",
        f"    x0 = load x[m, {axis_name}]",
        f"    {acc} <- add({acc}, x0) over {axis_name}",
        *([f"    {out}[m, {axis_name}] = {acc}"] if store else []),
    ]
    return "\n".join(lines)


def _scan_loop(axis_name: str = "k", acc: str = "acc", out: str = "out") -> Loop:
    (loop,) = parse_body(_scan_text(axis_name, acc, out))
    return loop


def _sum_fold(axis_name: str = "k", acc: str = "acc") -> Fold:
    (pure,) = parse_body(_scan_text(axis_name, acc, store=False))
    return fold_from_loop(pure)


def _observed(fold: Fold, obs: str | None = None) -> Fold:
    (acc,) = fold.combine.results
    obs = obs or f"{acc}__obs"
    observe = Lambda(params=(fold.axis, acc), body=parse_body(f"{obs} = copy({acc})"), results=(obs,))
    return dataclasses.replace(fold, observe=observe)


# --- formation ----------------------------------------------------------------------------------- #


def test_observer_formation_gates() -> None:
    fold = _sum_fold()
    scan = _observed(fold)
    assert scan.observe is not None and fold.observe is None
    assert scan.exposes == ("acc", "acc__obs")

    with pytest.raises(AssertionError, match="zero-axis"):
        Fold(lift=Lambda(params=(), body=parse_body(""), results=()), observe=scan.observe)
    with pytest.raises(AssertionError, match="positionally"):
        # A closed λ over the same names, bound in the wrong ORDER: the observer's contract is
        # positional — the iteration var first, then the carried state.
        dataclasses.replace(fold, observe=Lambda.closing(("acc", "k"), parse_body("t = copy(acc)"), ("t",)))
    with pytest.raises(AssertionError, match="FRESH"):
        # Observing the state under its own name is ill-formed — the boundary distinguishes a
        # streamed store from a post-fold store by the name.
        dataclasses.replace(fold, observe=Lambda(params=("k", "acc"), body=parse_body(""), results=("acc",)))


def test_exp_family_declines_an_observer() -> None:
    names = ("m_i", "l_i")
    lift = Lambda(
        params=("k",),
        body=parse_body(
            """
            s0 = load s[k]
            e = exp(s0)
            """
        ),
        results=("s0", "e"),
    )
    twist = Twist(recipe=SOFTMAX, channels=(0,))
    fold = Fold(lift=lift, init=(float("-inf"), 0.0), base=Lambda.componentwise(SOFTMAX.base[:2], names), twist=twist)
    observe = Lambda(params=("k", *names), body=parse_body("m__obs = copy(m_i)"), results=("m__obs",))
    with pytest.raises(AssertionError, match="does not support a per-step observer"):
        dataclasses.replace(fold, observe=observe)


# --- identity ------------------------------------------------------------------------------------ #


def test_scan_keys_apart_from_sum_and_alpha_invariantly() -> None:
    plain = _sum_fold()
    scan = _observed(plain)
    assert plain.canonical() != scan.canonical(), "a cumsum is not a sum"
    renamed = _observed(_sum_fold(acc="total"), obs="total__obs")
    assert scan.canonical() == renamed.canonical(), "SSA spelling must not enter identity"
    other_axis = _observed(_sum_fold(axis_name="j"))
    assert scan.canonical() == other_axis.canonical(), "axis spelling must not enter identity"


def test_rewrite_threads_the_observer() -> None:
    scan = _observed(_sum_fold())
    # An SSA rename map carries SSA defines only — the iteration var and the enclosing row
    # coordinate are axis names, and they rename through ``axis_fn`` / σ, never through this.
    ssa = {"acc", "acc__obs", "x0"}
    renamed = scan.rewrite(lambda name: f"{name}_r" if name in ssa else name)
    assert renamed.observe is not None
    assert renamed.observe.params == ("k", "acc_r")
    assert renamed.observe.results == ("acc__obs_r",)
    assert renamed.combine.results == ("acc_r",)


# --- recognition --------------------------------------------------------------------------------- #


def test_scan_from_loop_lifts_the_per_step_store() -> None:
    fold, trailing = scan_from_loop(_scan_loop())
    assert fold.observe is not None and fold.observe.results == ("acc__obs",)
    assert len(trailing) == 1 and trailing[0].values == ("acc__obs",)
    with pytest.raises(ValueError, match="scan, not a pure reduction"):
        fold_from_loop(_scan_loop())
    # The derived step taps AFTER the combine: observer stmts trail the Accum.
    kinds = [type(s).__name__ for s in fold.step()]
    assert kinds.index("Accum") < kinds.index("Assign", kinds.index("Accum"))


def test_scan_from_loop_rejects_a_store_off_the_state() -> None:
    (bad,) = parse_body(
        """
        for k in 0..16:
            x0 = load x[m, k]
            acc <- add(acc, x0) over k
            out[m, k] = x0
        """
    )
    with pytest.raises(ValueError, match="may only observe the carried state"):
        scan_from_loop(bad)


# --- the boundary -------------------------------------------------------------------------------- #


def test_streamed_store_reconstitutes_inside_the_reduce_loop() -> None:
    fold, trailing = scan_from_loop(_scan_loop())
    split = extract_output_specs(trailing)
    assert split is not None
    body, specs = split
    assert body == () and len(specs) == 1, "the trailing store is the boundary's whole decoration"
    assert observed_result_names(fold) == frozenset({"acc__obs"})

    lowered = fold.lower(fold.free_axes, tuple(specs), axes=(Axis("k", 8),))
    (loop,) = [s for s in lowered if isinstance(s, Loop)]
    inner = list(loop.body)
    assert isinstance(inner[-1], Write) and inner[-1].values == ("acc__obs",), "the store rides each iteration, last"
    assert not any(isinstance(s, Write) for s in lowered if not isinstance(s, Loop))


def test_observer_free_reconstitution_is_untouched() -> None:
    """A plain reduce with a post-fold store keeps its kernel-tail position — the observed-name
    membership, not loop-defines, is what streams a store."""
    fold = _sum_fold()
    (write,) = parse_body("out[m] = acc")
    spec = OutputSpec(write=write)
    lowered = fold.lower(fold.free_axes, (spec,), axes=(Axis("k", 8),))
    assert isinstance(lowered[-1], Write), "the post-fold store stays the kernel tail"
    (loop,) = [s for s in lowered if isinstance(s, Loop)]
    assert not any(isinstance(s, Write) for s in loop.body)


# --- the schedule gate --------------------------------------------------------------------------- #


def test_lifted_scan_kernel_shape() -> None:
    """The whole 025 → lift path: free axis peeled, observed fold stored, streamed spec at the
    boundary — the TileOp the schedule walk and the materializer consume."""
    from emmy.compiler.ir.loop import LoopOp

    op = LoopOp(body=parse_body("for m in 0..4:\n" + indent(_scan_text(), "    ")))
    tile = lift_loop_op(op)
    # Canonical renumbering renames the accumulator; the observed name follows in lockstep.
    (observed,) = observed_result_names(tile.op)
    assert observed.endswith("__obs")
    assert len(tile.output_specs) == 1 and tile.output_specs[0].write.values == (observed,)
