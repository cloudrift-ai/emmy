"""Normalization shares equal loads: the normal form of a body with duplicates, executed. What the driver leaves of
such a body is tested in the crate."""

from emmy.compiler.ir.stmt import normalize_body, parse_body


def test_dedup_loads_preserves_loads_under_a_rebound_coordinate() -> None:
    """A normalization sum must read every channel when its index shadows the output index."""
    import numpy as np

    from emmy.compiler.ir.loop import LoopOp
    from emmy.compiler.ir.loop.runner import execute_loop_op_cpp

    body = parse_body(
        """
        for k in 0..4:
            outer = load x[k]
            for k in 0..4:
                inner = load x[k]
                total <- add(total, inner) over k
            value = divide(outer, total)
            out[k] = value
        """
    )
    values = np.array([1, 2, 4, 8], dtype=np.float32)
    actual = execute_loop_op_cpp(LoopOp(body=normalize_body(body)), {"x": values}, {"out": (4,)})

    np.testing.assert_allclose(actual, values / values.sum(), rtol=1e-6)
