"""The fields a crate class of ``ir/pure`` declares to the rest of the compiler: what a dataclass's fields give it,
for a class whose data lives in the ``emmy-pure`` crate."""

from __future__ import annotations

from dataclasses import MISSING

from emmy.compiler.structural import record_form
from emmy.compiler.wire import declared_from_wire, declared_to_wire, register


def declare(cls: type, fields: tuple[tuple[str, object, object], ...]) -> None:
    """Give ``cls`` its declared ``fields`` — ``(name, annotation, default)``, ``MISSING`` for none, in the order its
    constructor takes them as keywords: the wire codec, the parts a generic rewrite goes into, and the structural form
    (the class name over the fields, trailing defaults left out). Then put the class on the wire."""

    def to_wire(self):
        return declared_to_wire(self, fields)

    def from_wire(owner, value, where=None):
        return declared_from_wire(owner, value, where or owner.__name__, fields)

    def wire_parts(self):
        return {name: getattr(self, name) for name, _, _ in fields}

    def structural_form(self):
        return record_form(cls.__name__, [(getattr(self, name), default) for name, _, default in fields])

    cls.to_wire = to_wire
    cls.from_wire = classmethod(from_wire)
    cls.wire_parts = wire_parts
    cls.structural_form = structural_form
    register(cls)


__all__ = ["MISSING", "declare"]
