"""Which approximate form each exact operation takes under the live precision decisions.

``130_approximate_math`` writes the forms into the body; ``118_unroll_as_nvcc`` reads them to size a loop's body as
it will be compiled, before ``130`` runs.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.pipeline.search.space import FAST_DIV, FAST_EXP, FAST_MATH, FAST_SQRT, precision_pin


def approximate_forms(ctx: Context) -> dict[str, str]:
    """The exact operation to its approximate form, for each class the live precision decisions turn on and the
    target has an instruction for."""
    picks: dict[str, str] = {}
    if precision_pin(FAST_EXP):
        picks["exp"] = "exp_fast"
    if precision_pin(FAST_DIV):
        picks |= {"divide": "divide_fast", "reciprocal": "reciprocal_fast"}
    if precision_pin(FAST_SQRT):
        picks["sqrt"] = "sqrt_fast"
    if precision_pin(FAST_MATH) and precision_pin(FAST_EXP):
        picks |= {name: f"{name}_fast" for name in ("log", "sin", "cos", "pow", *(("tanh",) if ctx.has_tanh_approx else ()))}
    if "exp" in picks and "divide" in picks:
        picks["sigmoid"] = "sigmoid_fast"
    return picks
