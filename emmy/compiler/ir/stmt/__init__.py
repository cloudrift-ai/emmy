"""Shared statement primitives — the leaves and control flow used across every IR layer.

The statement kinds, ``Body``, the text form, normalization and identity are implemented in Rust (the ``emmy-ir`` crate,
reached through the ``emmy.emmy_ir`` extension); this package is their one face to the rest of the compiler.
``__all__`` is the whole interface: the rest of the compiler imports these names and no submodule. The kinds every
layer shares are here; the Kernel IR's thirty are exported by ``emmy.compiler.ir.kernel.ir``. Rendering a statement
as C / CUDA source is not statement business and lives in :mod:`emmy.compiler.ir.render`.

The boundary: Rust owns the data and Python reads it through accessors; a field read hands back a held object, so a
statement read twice from a body is one object; only strings, numbers, booleans, tuples of those and the module's own
classes cross it. The few members that take a Python function and keep its results (``rename`` by a function,
``rewrite``, ``Body.fold``) are written here over the Rust reads; ``Body.map`` and ``map_expressions`` call the function
from Rust through the fallible walks and hold nothing of it afterwards.
"""

from __future__ import annotations

import emmy.emmy_ir as _ir
from emmy.compiler.wire import register as _register

Statement = _ir.Statement
Body = _ir.Body
Accum = _ir.Accum
Affine = _ir.Affine
Assign = _ir.Assign
Call = _ir.Call
Carry = _ir.Carry
Cond = _ir.Cond
Init = _ir.Init
Let = _ir.Let
Load = _ir.Load
Loop = _ir.Loop
Pre = _ir.Pre
Select = _ir.Select
SelectBranch = _ir.SelectBranch
StridedLoop = _ir.StridedLoop
Subroutine = _ir.Subroutine
Write = _ir.Write
ZeroPrologue = _ir.ZeroPrologue

parse_body = _ir.parse_body
pretty_body = _ir.pretty_body
normalize_body = _ir.normalize_body
fold_affine = _ir.fold_affine
simplify_body = _ir.simplify_body
prepare_body = _ir.prepare_body
free_names = _ir.free_names
exposed_defines = _ir.exposed_defines
rename_free = _ir.rename_free
binary = _ir.binary
mask_select_predicate = _ir.mask_select_predicate
ordering_constraints = _ir.ordering_constraints
dtype_promote = _ir.dtype_promote

_register(Body)

from emmy.compiler.ir.stmt.glue import Identity, OutputSpec, canonicalize_identity, fields_of  # noqa: E402 — the glue installs members on the classes above

__all__ = [
    "Accum",
    "Affine",
    "Assign",
    "Body",
    "Call",
    "Carry",
    "Cond",
    "Identity",
    "Init",
    "Let",
    "Load",
    "Loop",
    "OutputSpec",
    "Pre",
    "Select",
    "SelectBranch",
    "Statement",
    "StridedLoop",
    "Subroutine",
    "Write",
    "ZeroPrologue",
    "binary",
    "canonicalize_identity",
    "dtype_promote",
    "exposed_defines",
    "fields_of",
    "fold_affine",
    "free_names",
    "mask_select_predicate",
    "normalize_body",
    "ordering_constraints",
    "prepare_body",
    "parse_body",
    "pretty_body",
    "rename_free",
    "simplify_body",
]
