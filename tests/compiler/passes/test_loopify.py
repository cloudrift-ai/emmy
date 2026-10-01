"""``LOOPIFY`` re-rolls congruent statement runs into a loop over ``_r{depth}``; a position it cannot
express in terms of that loop var declines the run instead of building an invalid node."""

import importlib

from emmy.compiler.ir.expr import Literal
from emmy.compiler.ir.pure import Lambda
from emmy.compiler.ir.stmt import Body, Load

_loopify = importlib.import_module("emmy.compiler.pipeline.passes.lowering.kernel.100_loopify")


def _scale(by: int) -> Lambda:
    return Lambda(params=("x",), body=Body((Load(name="y", input="x", index=(Literal(by, "int"),)),)), results=("y",))


def test_a_lambda_that_differs_across_the_run_declines_the_run() -> None:
    """A closed lambda binds only its params: templating ``8 * _r0`` into its body would read a name
    it does not bind, so the run is declined rather than raising in ``Lambda.__post_init__``."""
    assert _loopify._reroll([_scale(0), _scale(8)], "_r0", set()) is _loopify._FAIL
    assert _loopify._reroll([_scale(8), _scale(8)], "_r0", set()) == _scale(8), "an invariant lambda stays"
