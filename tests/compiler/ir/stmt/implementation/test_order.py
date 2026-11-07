"""The executable order does not follow the Python process's hash seed: the one check that needs a second process."""

from __future__ import annotations

import os
import subprocess
import sys

from emmy.compiler.ir.stmt import Body, normalize_body, parse_body


def test_executable_order_ignores_the_iteration_order_of_a_statements_buffers() -> None:
    """Two projections of one normalized input, symmetric but for their weight buffer: which one
    comes first must not follow the order a set of buffer names iterates in, which is the
    per-process hash seed's. A fused two-projection serving kernel re-spelled on every boot."""

    def projection(weight: str) -> str:
        k = f"k_{weight}"
        return f"""
        for {k} in 0..8:
            x_{weight} = load X[i, {k}]
            n_{weight} = load N[{k}]
            w_{weight} = load {weight}[{k}, j]
            s_{weight} = multiply(scale, x_{weight})
            t_{weight} = multiply(n_{weight}, s_{weight})
            p_{weight} = multiply(t_{weight}, w_{weight})
            acc_{weight} <- add(acc_{weight}, p_{weight}) over {k}"""

    def make() -> Body:  # fresh each time: a body caches its normal form
        weights = ("W_up", "W_gate")
        inner = "".join(projection(w) for w in weights) + "".join(f"\n        O_{w}[i, j] = acc_{w}" for w in weights)
        return parse_body("for i in 0..4:\n    for j in 0..4:" + inner)

    forms = {repr(normalize_body(make()))}
    text = str(make())
    script = f"from emmy.compiler.ir.stmt import normalize_body, parse_body\nprint(repr(normalize_body(parse_body({text!r}))), end='')"
    for seed in ("1", "2"):
        forms.add(
            subprocess.run(
                [sys.executable, "-c", script], check=True, capture_output=True, text=True, env={**os.environ, "PYTHONHASHSEED": seed}
            ).stdout
        )
    assert len(forms) == 1
