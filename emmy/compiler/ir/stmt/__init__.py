"""Shared statement primitives — the leaves and control flow used across
every IR layer.

Defined here rather than under any one IR package because all three IRs
(Loop, Tile, Kernel) consume the same leaf vocabulary:

- ``Stmt`` — abstract base for every body statement.
- Leaves: ``Load``, ``Assign``, ``Accum``, ``Init``, ``Write``,
  ``Select``, ``SelectBranch`` — pure compute primitives that read/write SSA
  names and external buffers (in :mod:`.leaves`).
- Block stmts: ``Loop``, ``StridedLoop``, ``Cond`` — carry child bodies
  (in :mod:`.blocks`).
- Tree walks: :meth:`Body.iter` (pre-order recursive) and
  :meth:`Body.map` (flat 1:N transformer) — methods on
  :class:`Body` (in :mod:`.body`).
- Body normalization: the public ``normalize_body`` driver and its internal ordered passes
  (drop-size-one, canonicalize-axis-order, copy-alias-elim,
  reduce-axis-unify, hoist, simplify, dedup-loads, canonical-statement-order, rename-ssa) in
  :mod:`.normalize`.
- Pretty printing + render context: ``RenderCtx``, ``op_to_expr``,
  ``select_to_ternary``, ``render_index`` (in :mod:`.base`).

Each IR layer adds its own scheduling-specific Stmts on top:

- Loop IR: nothing extra — its bodies are exactly Loop / leaves.
- Tile IR: the typed tile flavors and staging constructs — DEMOLISHED,
  pending rebuild.
- Kernel IR: ``Smem``, ``Sync``, ``TreeHalve``, plus the shared
  constructs.

Loop-IR's ``LoopOp``, ``LoopMeta``, and validation stay in ``ir/loop/`` because they enforce
Loop-IR-specific invariants. Shared body normalization and structural identity stay here in
``ir/stmt/``.
"""

from emmy.compiler.ir.stmt.base import (
    INDENT,
    Flat,
    Memory,
    Paged,
    RenderCtx,
    Stmt,
    op_to_expr,
    pretty_body,
    render_body,
    render_index,
    select_to_ternary,
)
from emmy.compiler.ir.stmt.base import (
    _axis_identity as _axis_identity,  # re-export for downstream IR layers,
)
from emmy.compiler.ir.stmt.base import (
    _pad as _pad,  # re-export for ir.kernel.ir,
)
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body, refs_axis, stmt_axis_names
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Assign,
    Carry,
    Init,
    Let,
    Load,
    OutputSpec,
    Pre,
    Select,
    SelectBranch,
    Write,
    ZeroPrologue,
    mask_select_predicate,
)
from emmy.compiler.ir.stmt.normalize import normalize_body

__all__ = [
    "refs_axis",
    "stmt_axis_names",
    "INDENT",
    "Accum",
    "Pre",
    "Carry",
    "Assign",
    "Body",
    "Cond",
    "Flat",
    "Init",
    "Let",
    "Load",
    "Loop",
    "mask_select_predicate",
    "Memory",
    "Paged",
    "RenderCtx",
    "ZeroPrologue",
    "Select",
    "SelectBranch",
    "Stmt",
    "StridedLoop",
    "Write",
    "OutputSpec",
    "normalize_body",
    "op_to_expr",
    "pretty_body",
    "render_body",
    "render_index",
    "select_to_ternary",
]
