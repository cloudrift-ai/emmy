"""Shared expression AST + coordinate-expression helpers.

Backend-agnostic expression sublanguage used by:

- ``IndexMapOp.coord_map`` (``ir.tensor.ir``): affine output→input coord maps.
- ``Mux.select`` / ``MuxBranch.select`` (``ir.loop``): coord predicates.
- Tile IR (``ir.tile``): array indices, loop bounds, ternary selects.

The expression types are implemented in Rust (the ``emmy-ir`` crate, reached through ``emmy.emmy_ir``) and exported
here: the ``Expression`` base and its eight types, the simplification context and the affine-form helpers. On the wire
an expression is its C-like text, ``((a5 / 128) * 128) + a6`` written as ``a5 / 128 * 128 + a6`` with parentheses only
where precedence needs them (:func:`parse` reads it back), and that text is also its structural identity. The members
that take a Python function or evaluate against host values — ``rename`` by a function, ``rebuild``, ``eval`` — are
written here over the Rust reads; rendering to C / CUDA lives in :mod:`emmy.compiler.ir.render`.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

import emmy.emmy_ir as _ir
from emmy.compiler.wire import register as _register

Expression = _ir.Expression
Var = _ir.Var
Literal = _ir.Literal
Builtin = _ir.Builtin
BinaryExpression = _ir.BinaryExpression
FunctionCallExpression = _ir.FunctionCallExpression
TernaryExpression = _ir.TernaryExpression
CastExpression = _ir.CastExpression
FlatIndex = _ir.FlatIndex
Interval = _ir.Interval
SimplifyCtx = _ir.SimplifyCtx

parse = _ir.parse
spell_name = _ir.spell_name
affine_form = _ir.affine_form
split_anchor = _ir.split_anchor
index_set_size = _ir.index_set_size
placeholder = _ir.placeholder
is_placeholder = _ir.is_placeholder
BUILTINS = _ir.BUILTINS
PLACEHOLDER_PREFIX = _ir.PLACEHOLDER_PREFIX

for _cls in (Expression, Var, Literal, Builtin, BinaryExpression, FunctionCallExpression, TernaryExpression, CastExpression, FlatIndex):
    _register(_cls)


def apply_binop(op: str, lv: object, rv: object) -> object:
    """Apply a binary operator to two already-evaluated values, scalars or numpy arrays; integer floor division is
    used for both ``/`` and ``//``, and ``&&`` / ``||`` fall back to the elementwise logical ops on arrays."""
    if op == "+":
        return lv + rv
    if op == "-":
        return lv - rv
    if op == "*":
        return lv * rv
    if op in ("/", "//"):
        try:
            return int(lv) // int(rv)
        except TypeError:
            return lv // rv
    if op == "%":
        try:
            return int(lv) % int(rv)
        except TypeError:
            return lv % rv
    if op == "<":
        return lv < rv
    if op == "<=":
        return lv <= rv
    if op == ">":
        return lv > rv
    if op == ">=":
        return lv >= rv
    if op == "==":
        return lv == rv
    if op == "&&":
        try:
            return bool(lv) and bool(rv)
        except (TypeError, ValueError):
            return np.logical_and(lv, rv)
    if op == "||":
        try:
            return bool(lv) or bool(rv)
        except (TypeError, ValueError):
            return np.logical_or(lv, rv)
    if op == "&":
        return lv & rv
    if op == "|":
        return lv | rv
    if op == "^":
        return lv ^ rv
    if op == "<<":
        return lv << rv
    if op == ">>":
        return lv >> rv
    raise ValueError(f"Unknown BinOp: {op}")


def _eval(self: Expression, env: dict[str, object]) -> object:
    """Evaluate against a name → value environment of scalars or numpy arrays."""
    if isinstance(self, Var):
        return env[self.name]
    if isinstance(self, Literal):
        return self.value
    if isinstance(self, BinaryExpression):
        return apply_binop(self.op, _eval(self.left, env), _eval(self.right, env))
    if isinstance(self, FunctionCallExpression):
        from emmy.compiler.ir.elementwise import ElementwiseImpl  # noqa: PLC0415 — the elementwise module imports this one

        try:
            op = ElementwiseImpl(self.name)
        except ValueError as exc:
            raise NotImplementedError(f"FunctionCallExpression.eval: unknown intrinsic {self.name!r}") from exc
        return op(*(_eval(a, env) for a in self.args))
    if isinstance(self, TernaryExpression):
        cond = _eval(self.cond, env)
        if np.ndim(cond) == 0:
            return _eval(self.if_true, env) if bool(cond) else _eval(self.if_false, env)
        return np.where(cond, _eval(self.if_true, env), _eval(self.if_false, env))
    if isinstance(self, CastExpression):
        v = _eval(self.expression, env)
        if self.dtype == "int":
            return np.asarray(v).astype(np.int64) if hasattr(v, "__array__") else int(v)
        return v
    if isinstance(self, Builtin):
        raise NotImplementedError(f"Builtin {self.name!r} is GPU-only; cannot eval in numpy")
    raise TypeError("FlatIndex has no host evaluation: the buffer shape is a render-time fact")


def _rename(self: Expression, rename) -> Expression:  # noqa: ANN001 — a mapping or a ``str -> str`` function
    """This expression with every free variable's name mapped through ``rename``; a name it keeps passes through."""
    mapping = dict(rename) if hasattr(rename, "get") else {name: rename(name) for name in self.free_vars()}
    return self._rename({old: new for old, new in mapping.items() if old != new})


def _rebuild(self: Expression, fn: Callable[[Expression], Expression]) -> Expression:
    """Bottom-up map: ``fn`` applied to every subterm once its children are rebuilt."""
    return fn(self.with_children([child.rebuild(fn) for child in self.children()]))


def _repr(self: Expression) -> str:
    """The constructor form, ``Var(name='x')``: what ``repr`` spells and the graph's dict form reads back with ``eval``;
    ``str`` is the text."""
    if isinstance(self, Var):
        return f"Var(name={self.name!r})"
    if isinstance(self, Literal):
        return f"Literal(value={self.value!r}, dtype={self.dtype!r})"
    if isinstance(self, Builtin):
        return f"Builtin(name={self.name!r})"
    if isinstance(self, BinaryExpression):
        return f"BinaryExpression(op={self.op!r}, left={self.left!r}, right={self.right!r})"
    if isinstance(self, FunctionCallExpression):
        return f"FunctionCallExpression(name={self.name!r}, args={self.args!r})"
    if isinstance(self, TernaryExpression):
        return f"TernaryExpression(cond={self.cond!r}, if_true={self.if_true!r}, if_false={self.if_false!r})"
    if isinstance(self, CastExpression):
        return f"CastExpression(dtype={self.dtype!r}, expression={self.expression!r})"
    return f"FlatIndex(buffer={self.buffer!r}, index={self.index!r})"


Expression.__repr__ = _repr
Expression.eval = _eval
Expression.rename = _rename
Expression.rebuild = _rebuild
