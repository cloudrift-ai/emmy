"""Shared reduction bodies: the two kinds fusion builds a region with. Fusion expands every call before a body
leaves it, so normalization, identity and storage never meet one."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import Expression
from emmy.compiler.ir.stmt.base import Statement
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.declared import role, statement
from emmy.utils import cached_method


@dataclass(frozen=True, eq=False)
class Subroutine:
    """One shared, read-only reduction cone with explicit coordinate parameters.

    Definition identity is sufficient for the cheap call CSE. Semantic identity belongs to the
    expanded normal form, so independently formed definitions need not compare equal here.
    """

    name: str
    axes: tuple[Axis, ...]
    body: Body
    result: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", Body.coerce(self.body))
        if self.body.carries or any(stmt.has_side_effects for stmt in self.body):
            raise ValueError("a fusion subroutine must be read-only")
        # The body may read more than its parameters: an index can name a symbolic dim, which is no coordinate.
        if self.result not in self.body.definitions:
            raise ValueError("a fusion subroutine must define its result")

    def __getstate__(self):
        return {name: self.__dict__[name] for name in self.__dataclass_fields__}

    @cached_property
    def buffers(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(name for stmt in self.body.iter() for name in stmt.external_reads()))

    @cached_method
    def with_buffers(self, buffers: tuple[str, ...]) -> Subroutine:
        """Rebind captured buffers once for all calls sharing the same definition."""
        return replace(self, body=self.body.rename_buffers(dict(zip(self.buffers, buffers, strict=True))))

    @property
    def params(self) -> tuple[str, ...]:
        return tuple(axis.name for axis in self.axes)


@statement
class Call(Statement):
    """A fusion-only scalar call; its definition stays outside generic statement-tree walks.

    Fusion expands its calls before a LoopOp is built. They are not members of Tile IR lambdas, and a body that
    holds one has no normal form.
    """

    name: str = role("defines")
    target: Subroutine
    args: tuple[Expression, ...] = role("expression", "reads")

    deps_deep = True

    def __post_init__(self) -> None:
        if len(self.args) != len(self.target.params):
            raise ValueError("subroutine arguments do not match its parameters")

    def external_reads(self) -> tuple[str, ...]:
        return self.target.buffers

    def rename_buffers(self, rename):  # noqa: ANN001 — see Statement.rename_buffers
        buffers = tuple(rename.get(name, name) for name in self.target.buffers)
        return self if buffers == self.target.buffers else self.replace(target=self.target.with_buffers(buffers))
