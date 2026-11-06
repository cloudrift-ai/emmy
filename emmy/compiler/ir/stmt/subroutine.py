"""Shared reduction bodies: the two kinds fusion builds a region with. Fusion expands every call before a body
leaves it, so normalization, identity and storage never meet one."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.stmt.body import Body
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
        if self.result not in self.body.ssa_defs:
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

    def pretty(self, indent: str = "") -> list[str]:
        return [
            f"{indent}sub {self.name}({', '.join((*self.buffers, *self.params))}):",
            *pretty_body(self.body, indent + "    "),
            f"{indent}    return {self.result}",
        ]


@dataclass(frozen=True)
class Call(Stmt):
    """A fusion-only scalar call; its definition stays outside generic statement-tree walks.

    Fusion expands its calls before a LoopOp is built. They are not members of Tile IR lambdas, and a body that
    holds one has no normal form.
    """

    name: str
    target: Subroutine
    args: tuple[Expr, ...]

    deps_deep = True

    def __post_init__(self) -> None:
        if len(self.args) != len(self.target.params):
            raise ValueError("subroutine arguments do not match its parameters")

    def defines(self) -> tuple[str, ...]:
        return (self.name,)

    def deps(self) -> tuple[str, ...]:
        return tuple(sorted(set().union(*(arg.free_vars() for arg in self.args))))

    def exprs(self) -> tuple[Expr, ...]:
        return self.args

    def with_exprs(self, exprs: tuple[Expr, ...]) -> Stmt:
        return self.replace(args=exprs)

    def external_reads(self) -> tuple[str, ...]:
        return self.target.buffers

    def rename_buffers(self, rename):  # noqa: ANN001 — see Stmt.rename_buffers
        buffers = tuple(rename.get(name, name) for name in self.target.buffers)
        return self if buffers == self.target.buffers else self.replace(target=self.target.with_buffers(buffers))

    def pretty(self, indent: str = "") -> list[str]:
        args = (*self.target.buffers, *(arg.pretty() for arg in self.args))
        return [f"{indent}{self.name} = {self.target.name}({', '.join(args)})"]

    def _rewrite(self, rename, sigma, axis_fn):  # noqa: ANN001, ANN202 — see ``Stmt.rewrite``
        return self.replace(
            name=rename(self.name),
            args=tuple(sigma.apply(arg).rename(rename) for arg in self.args),
        )
