"""Projection legality for cross-CTA atomic reductions."""

import pytest

from emmy.compiler.ir.stmt.leaves import Assign, Write
from emmy.compiler.ir.stmt.passes import projection_distributes


@pytest.mark.parametrize(
    ("operation", "args", "expected"),
    [
        ("divide", ("state", "count"), True),
        ("divide", ("count", "state"), False),
        ("divide", ("state", "state"), False),
        ("multiply", ("state", "count"), True),
        ("multiply", ("count", "state"), True),
        ("multiply", ("state", "state"), False),
        ("add", ("state", "count"), False),
    ],
)
def test_projection_distributes(operation, args, expected):
    body = [Assign("scaled", operation, args), Write("out", (), value="scaled")]
    assert projection_distributes(body, ("state",)) is expected
