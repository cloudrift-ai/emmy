"""A reference interpreter of Loop IR's value form, in numpy: what a body MEANS, stated apart from every compiler.

It executes a parsed body statement by statement, in order, and knows three things the forms it is compared with
spell otherwise:

- A loop walks its range, ``start..end``, where it states one, and ``0..extent`` otherwise.
- A definition ``name[index] <- value [seed s]`` stores one cell of a value the kernel keeps for itself, and a read
  ``pre name[index]`` takes a cell by index, wherever the read stands.
- A read sees the last definition at or before the coordinate read, and the seed where there is none. "Before" runs
  along the step coordinates of the value: the index positions a read of the value lags its own axis at
  (``S[t - 1, i]`` under the loop over ``t``). Every other position is a cell coordinate, matched exactly, so a value
  with no step coordinate is one array updated in place: a read sees what the statements run so far stored, and the
  seed in a cell none of them defined.

The interpreter is slow and total over the small kernels the tests hand it. It reads no normal form and no schedule:
the body is run as parsed.
"""

from __future__ import annotations

import numpy as np

from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.stmt import Accum, Assign, Body, Carry, Cond, Let, Load, Loop, Pre, Select, Write, parse_body

_COMBINE = {
    "add": lambda a, b: a + b,
    "multiply": lambda a, b: a * b,
    "maximum": max,
    "minimum": min,
}


def _lag(read, axis: str) -> bool:
    """Whether ``read`` is ``axis - c`` for a constant ``c >= 1``: a coordinate one or more steps back."""
    if not isinstance(read, BinaryExpression):
        return False
    if read.op == "-" and read.left == Var(axis) and isinstance(read.right, Literal):
        return read.right.value >= 1
    if read.op == "+" and isinstance(read.left, Literal) and read.right == Var(axis):
        return read.left.value <= -1
    return False


class _Value:
    """One value a body defines for itself: its seed, which index positions are step coordinates, and every cell
    stored so far as ``cell -> [(step, value), ...]`` in the order the statements ran."""

    def __init__(self, definition: Carry, reads: list[Pre]) -> None:
        self.name, self.seed = definition.name, definition.seed
        self.steps = tuple(
            position
            for position, at in enumerate(definition.index)
            if isinstance(at, Var) and any(_lag(read.index[position], at.name) for read in reads)
        )
        self.cells: dict[tuple, list[tuple[tuple, float]]] = {}

    def _split(self, index: tuple) -> tuple[tuple, tuple]:
        step = tuple(index[position] for position in self.steps)
        cell = tuple(at for position, at in enumerate(index) if position not in self.steps)
        return step, cell

    def store(self, index: tuple, value: float) -> None:
        step, cell = self._split(index)
        self.cells.setdefault(cell, []).append((step, value))

    def read(self, index: tuple, inputs: dict[str, np.ndarray]) -> float:
        step, cell = self._split(index)
        behind = [(at, order, value) for order, (at, value) in enumerate(self.cells.get(cell, ())) if at <= step]
        if behind:
            return max(behind)[2]
        if self.seed is None:
            raise ValueError(f"{self.name}{list(index)} is read before any statement defined it, and the value has no seed")
        if isinstance(self.seed, str):
            return float(inputs[self.seed][cell])
        return float(self.seed)


def run(text: str | Body, inputs: dict[str, np.ndarray], outputs: dict[str, tuple[int, ...]]) -> dict[str, np.ndarray]:
    """The arrays ``text`` stores into ``outputs`` (each a shape, zero where nothing is stored) when run on ``inputs``."""
    body = parse_body(text) if isinstance(text, str) else text
    stmts = list(body.iter())
    values = {
        stmt.name: _Value(stmt, [read for read in stmts if isinstance(read, Pre) and read.carrier == stmt.name])
        for stmt in stmts
        if isinstance(stmt, Carry)
    }
    out = {name: np.zeros(shape, dtype=np.float32) for name, shape in outputs.items()}
    env: dict[str, float] = {}

    def at(index) -> tuple:
        return tuple(int(entry.eval(env)) for entry in index)

    def execute(stmts: Body) -> None:
        for stmt in stmts:
            if isinstance(stmt, Loop):
                start, end = (0, stmt.axis.extent.as_static()) if stmt.range is None else (int(bound.eval(env)) for bound in stmt.range)
                for name, op in {inner.name: inner.op for inner in stmt.body if isinstance(inner, Accum)}.items():
                    env[name] = float(op.identity)
                for coordinate in range(start, end):
                    env[stmt.axis.name] = coordinate
                    execute(stmt.body)
            elif isinstance(stmt, Cond):
                execute(stmt.body if stmt.cond.eval(env) else stmt.else_body)
            elif isinstance(stmt, Load):
                env[stmt.name] = float(inputs[stmt.input][at(stmt.index)])
            elif isinstance(stmt, Pre):
                env[stmt.name] = values[stmt.carrier].read(at(stmt.index), inputs)
            elif isinstance(stmt, (Assign, Let)):
                env[stmt.name] = stmt.value.eval(env)
            elif isinstance(stmt, Select):
                env[stmt.name] = next(env[branch.value] for branch in stmt.branches if branch.select.eval(env))
            elif isinstance(stmt, Accum):
                env[stmt.name] = _COMBINE[stmt.op.name](env[stmt.name], stmt.value.eval(env))
            elif isinstance(stmt, Carry):
                values[stmt.name].store(at(stmt.index), env[stmt.value])
            elif isinstance(stmt, Write):
                out[stmt.output][at(stmt.index)] = env[stmt.value]
            else:
                raise NotImplementedError(f"the reference model runs no {type(stmt).__name__}")

    execute(body)
    return out
