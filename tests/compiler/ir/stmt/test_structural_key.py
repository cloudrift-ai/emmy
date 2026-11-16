"""``Body.structural_key()`` at the Python boundary: the key is a hashable ``str``.

Which bodies share an identity is tested in the ``emmy-ir`` crate, beside ``canonicalize_identity``.
"""

from __future__ import annotations

from emmy.compiler.ir.stmt import parse_body


def test_structural_key_is_string_and_hashable() -> None:
    body = parse_body(
        """
        for a in 0..4:
            x = load X[a]
            y = load Y[a]
            z = multiply(x, y)
            O[a] = z
        """
    )
    key = body.structural_key()
    assert isinstance(key, str)
    assert hash(key) == hash(body.structural_key())  # cached, deterministic
