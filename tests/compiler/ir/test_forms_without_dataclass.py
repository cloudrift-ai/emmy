"""The wire form and the structural form of a class that is no dataclass: what it supplies, and what it gets.

A class joins the wire with its tag, its two wire functions and its parts, and enters an identity with its own
structural form. Neither form reads a dataclass field of it, and the wire does not ask it to inherit ``Wire``.
"""

from __future__ import annotations

import dataclasses

from emmy.compiler.dtype import F32
from emmy.compiler.ir.expr import Literal, Var
from emmy.compiler.ir.stmt import Body, Load, Stmt, Write
from emmy.compiler.structural import form, record_form
from emmy.compiler.wire import Wire, decode, encode, register, rewrite


class Span:
    """A value with no base class at all."""

    wire_tag = "test.span"

    def __init__(self, lo: int, hi: int = 0) -> None:
        self.lo, self.hi = lo, hi

    def __eq__(self, other: object) -> bool:
        return type(other) is Span and (self.lo, self.hi) == (other.lo, other.hi)

    def __hash__(self) -> int:
        return hash((self.lo, self.hi))

    def __repr__(self) -> str:
        return f"Span({self.lo}, {self.hi})"

    def to_wire(self) -> list:
        return [self.lo, self.hi]

    @classmethod
    def from_wire(cls, value: object, where: str = "span") -> Span:
        return cls(*value)

    def wire_parts(self) -> dict[str, object]:
        return {"lo": self.lo, "hi": self.hi}

    def structural_form(self) -> tuple:
        return record_form("Span", [(self.lo, dataclasses.MISSING), (self.hi, 0)])


register(Span)


@dataclasses.dataclass(frozen=True)
class SpanData:
    """What ``Span`` would be as a dataclass: the form it has to match."""

    lo: int
    hi: int = 0


@dataclasses.dataclass(frozen=True)
class Holder(Wire):
    """A dataclass on the wire that holds the class by annotation and untyped."""

    span: Span
    spans: tuple[Span, ...] = ()
    anything: object = None


class Scaled(Stmt):
    """A statement kind written by hand: ``name = value`` over a span."""

    wire_tag = "test.scaled"

    def __init__(self, name: str, value: str, span: Span) -> None:
        self.name, self.value, self.span = name, value, span

    def __eq__(self, other: object) -> bool:
        return type(other) is Scaled and (self.name, self.value, self.span) == (other.name, other.value, other.span)

    def __hash__(self) -> int:
        return hash((self.name, self.value, self.span))

    def __repr__(self) -> str:
        return f"Scaled({self.name!r}, {self.value!r}, {self.span!r})"

    def deps(self) -> tuple[str, ...]:
        return (self.value,)

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return Scaled(rename(self.name), rename(self.value), self.span)

    def pretty(self, indent: str = "") -> list[str]:
        return [f"{indent}{self.name} = scaled({self.value}, {self.span!r})"]

    def to_wire(self) -> dict:
        return {"name": self.name, "value": self.value, "span": self.span.to_wire()}

    @classmethod
    def from_wire(cls, value: object, where: str = "scaled") -> Scaled:
        return cls(value["name"], value["value"], Span.from_wire(value["span"]))

    def wire_parts(self) -> dict[str, object]:
        return {"name": self.name, "value": self.value, "span": self.span}

    def structural_form(self) -> tuple:
        return record_form("Scaled", [(part, dataclasses.MISSING) for part in (self.name, self.value, self.span)])


def _body(x: str = "x", y: str = "y", span: Span | None = None) -> Body:
    return Body(
        (
            Load(x, "a", (Var("i"),), dtype=F32),
            Scaled(y, x, span or Span(1, 4)),
            Write(output="out", index=(Var("i"),), value=y),
        )
    )


def test_a_registered_class_round_trips_the_wire_tagged_and_by_annotation() -> None:
    assert encode(Span(1, 4)) == {"test.span": [1, 4]}
    assert decode(encode(Span(1, 4))) == Span(1, 4)
    holder = Holder(Span(1), (Span(2, 3),), anything=(Span(5), 7))
    assert holder.to_wire() == {"span": [1, 0], "spans": [[2, 3]], "anything": {"tuple": [{"test.span": [5, 0]}, 7]}}
    assert Holder.from_wire(holder.to_wire()) == holder


def test_a_hand_written_statement_kind_round_trips_the_wire() -> None:
    body = _body()
    wire = encode(body)
    assert wire["body"][1] == {"test.scaled": {"name": "y", "value": "x", "span": [1, 4]}}
    assert decode(wire) == body


def test_the_rewrite_goes_into_a_class_through_its_parts_and_rebuilds_it() -> None:
    def widen(value: object):
        return Span(value.lo, value.hi + 1) if isinstance(value, Span) else None

    assert rewrite(_body(), widen) == _body(span=Span(1, 5))
    assert rewrite(Holder(Span(1), (Span(2, 3),)), widen) == Holder(Span(1, 1), (Span(2, 4),))
    assert rewrite(Span(1, 4), lambda value: 9 if value == 4 else None) == Span(1, 9)
    untouched = _body()
    assert rewrite(untouched, lambda value: None) is untouched


def test_a_class_with_its_own_form_is_keyed_the_way_a_dataclass_is() -> None:
    assert form(Span(1)) == ("Span", 1)
    assert form(Span(1, 4)) == ("Span", 1, 4)
    for span, data in ((Span(1), SpanData(1)), (Span(1, 4), SpanData(1, 4)), (Span(0, 0), SpanData(0, 0))):
        assert form(span)[1:] == form(data)[1:]
    assert form(Scaled("y", "x", Span(1))) == ("Scaled", "y", "x", ("Span", 1))
    assert form((Literal(1, "int"), Span(1))) == (form(Literal(1, "int")), ("Span", 1))


def test_a_hand_written_statement_kind_enters_an_identity() -> None:
    key = _body().structural_key(structural=False)
    assert _body("p", "q").structural_key(structural=False) == key
    assert _body(span=Span(2, 4)).structural_key(structural=False) != key
    assert _body().structural_key() == _body("p", "q").structural_key()
