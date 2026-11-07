"""``rename_buffers`` — the setter counterpart of ``external_reads`` / ``external_writes``.

What these pin: every buffer-bearing leaf renames its field (and only it), the body-level
rename reaches leaves through wrappers, and identity normalization covers EVERY external-buffer
field — a kernel-stage body differing
only in a staged source buffer's name must key identically."""

from __future__ import annotations

from emmy.compiler.ir.stmt import Body, parse_body


def _body(x: str = "x", out: str = "o", zero: str = "acc_buf") -> Body:
    return parse_body(
        f"""
        zero {zero} words=16
        for k in 0..8:
            v = load {x}[k]
            w = relu(v)
            acc <- add(acc, w) over k
        {out}[] = acc
        """
    )


def test_body_rename_reaches_nested_leaves_and_every_buffer_field() -> None:
    renamed = _body().rename_buffers({"x": "x9", "o": "o9", "acc_buf": "z9"})
    assert str(renamed) == str(_body("x9", "o9", "z9"))


def test_rename_is_identity_off_the_mapping() -> None:
    body = _body()
    assert body.rename_buffers({"unrelated": "name"}) == body


def test_identity_canonicalization_covers_non_load_write_buffer_fields() -> None:
    a, b = _body(zero="acc_buf"), _body(zero="differently_named")
    assert a.structural_key() == b.structural_key()
