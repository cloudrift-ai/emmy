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
- Body normalization: the public ``normalize_body`` driver and its steps (in :mod:`.normalize`).

``__all__`` is the whole interface: the rest of the compiler imports these names and no submodule. Rendering a
statement as C / CUDA source is not statement business and lives in :mod:`emmy.compiler.ir.render`.

Each IR layer adds its own scheduling-specific Stmts on top:

- Loop IR: nothing extra — its bodies are exactly Loop / leaves.
- Tile IR: the typed tile flavors and staging constructs — DEMOLISHED,
  pending rebuild.
- Kernel IR: ``Smem``, ``Sync``, ``TreeHalve``, plus the shared
  constructs.

Loop-IR's ``LoopOp``, its validation and fusion's body assembly stay in ``ir/loop/``. Normalization and structural
identity stay here, with the two kinds fusion assembles a region with (``Call``, ``Subroutine``).
"""

from emmy.compiler.ir.stmt.base import Stmt, dtype_promote
from emmy.compiler.ir.stmt.blocks import Cond, Loop, StridedLoop
from emmy.compiler.ir.stmt.body import Body, exposed_defines, free_names
from emmy.compiler.ir.stmt.declared import Declared, role
from emmy.compiler.ir.stmt.identity import canonicalize_identity
from emmy.compiler.ir.stmt.leaves import (
    Accum,
    Affine,
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
    binary,
    mask_select_predicate,
)
from emmy.compiler.ir.stmt.normalize import fold_affine, normalize_body, prepare_body, simplify_body
from emmy.compiler.ir.stmt.order import ordering_constraints
from emmy.compiler.ir.stmt.passes import map_exprs, rename_free
from emmy.compiler.ir.stmt.subroutine import Call, Subroutine
from emmy.compiler.ir.stmt.text import parse_body, pretty_body

__all__ = [
    "Accum",
    "Affine",
    "Assign",
    "Body",
    "Call",
    "Carry",
    "Cond",
    "Declared",
    "Init",
    "Let",
    "Load",
    "Loop",
    "OutputSpec",
    "Pre",
    "Select",
    "SelectBranch",
    "Stmt",
    "StridedLoop",
    "Subroutine",
    "Write",
    "ZeroPrologue",
    "binary",
    "canonicalize_identity",
    "dtype_promote",
    "exposed_defines",
    "fold_affine",
    "free_names",
    "map_exprs",
    "mask_select_predicate",
    "normalize_body",
    "ordering_constraints",
    "prepare_body",
    "parse_body",
    "pretty_body",
    "rename_free",
    "role",
    "simplify_body",
]
