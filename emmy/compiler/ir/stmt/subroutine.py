"""Shared reduction bodies during fusion; calls disappear before executable CSE."""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expr import Expr
from emmy.compiler.ir.stmt.base import Stmt, pretty_body
from emmy.compiler.ir.stmt.body import Body
from emmy.compiler.ir.stmt.passes import _rename_ssa_vars_in_expr, _rewrite_kind
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

    @cached_property
    def reduction_depth(self) -> int:
        return reduction_depths(self.body)[self.result]

    def pretty(self, indent: str = "") -> list[str]:
        return [
            f"{indent}sub {self.name}({', '.join((*self.buffers, *self.params))}):",
            *pretty_body(self.body, indent + "    "),
            f"{indent}    return {self.result}",
        ]


@dataclass(frozen=True)
class Call(Stmt):
    """A fusion-only scalar call; its definition stays outside generic statement-tree walks.

    Calls expand before a LoopOp is validated or lifted. They are not members of Tile IR lambdas.
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

    def external_reads(self) -> tuple[str, ...]:
        return self.target.buffers

    def rename_buffers(self, rename):  # noqa: ANN001 — see Stmt.rename_buffers
        buffers = tuple(rename.get(name, name) for name in self.target.buffers)
        return self if buffers == self.target.buffers else replace(self, target=self.target.with_buffers(buffers))

    def pretty(self, indent: str = "") -> list[str]:
        args = (*self.target.buffers, *(arg.pretty() for arg in self.args))
        return [f"{indent}{self.name} = {self.target.name}({', '.join(args)})"]


@_rewrite_kind.register
def _(stmt: Call, rename, sigma, axis_fn) -> Stmt:
    return replace(
        stmt,
        name=rename(stmt.name),
        args=tuple(_rename_ssa_vars_in_expr(sigma.apply(arg), rename) for arg in stmt.args),
    )


def definitions(body: Body) -> tuple[Subroutine, ...]:
    """Definitions in dependency order, each visited once regardless of the number of calls."""
    seen: set[int] = set()
    out: list[Subroutine] = []

    def visit(stmts: Body) -> None:
        for stmt in stmts.iter():
            if isinstance(stmt, Call) and id(stmt.target) not in seen:
                seen.add(id(stmt.target))
                visit(stmt.target.body)
                out.append(stmt.target)

    visit(body)
    return tuple(out)


def pretty_subroutines(body: Body, indent: str = "") -> list[str]:
    """Compact definitions followed by the calling body."""
    return [line for target in definitions(body) for line in (*target.pretty(indent), "")] + pretty_body(body, indent)


def reduction_depths(body: Body, inputs: dict[str, int] | None = None) -> dict[str, int]:
    """Reduction depth including the reductions held in shared definitions."""
    from emmy.compiler.ir.stmt.leaves import Accum

    inputs = inputs or {}
    memo = body.fold(
        lambda stmt, children, _: (
            max(
                (*(depth for depth in children if depth is not None), *(inputs.get(name, 0) for name in stmt.external_reads())),
                default=0,
            )
            + (stmt.target.reduction_depth if isinstance(stmt, Call) else isinstance(stmt, Accum))
        )
    )
    return {name: memo[id(stmt)] for name, stmt in body.definitions.items()}
