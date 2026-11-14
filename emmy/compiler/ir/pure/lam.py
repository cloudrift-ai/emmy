"""``Lambda`` — the ONE binder kind, shared by every IR level.

A lambda is a PURE term: explicit binders over the reused stmt vocabulary, an A-normal-form
let-chain in ``body``, the returned defs named by ``results``. It binds; it never executes.
Nothing here is a :class:`~emmy.compiler.ir.stmt.Statement` — see ``ir/ARCHITECTURE.md`` for the
invariant and for how a pure term reaches a statement position (it is RENDERED to stmts, never
spliced in as one).

The type is the ``emmy-ir`` crate's (``emmy.emmy_ir.Lambda``): construction normalizes the body and
checks that every statement is pure, every result defined and every read bound; ``closing`` forms a
closed lambda by appending what the body still reads as trailing params; ``rename``, ``canonical``
(α-canonical renumbering, ``_p0…`` / ``_v0…``), ``alpha_eq``, ``cone``, ``componentwise`` /
``components`` and ``replace`` are whole questions the crate answers. Python holds a handle.
"""

from __future__ import annotations

import emmy.emmy_ir as _ir
from emmy.compiler.wire import register

Lambda = _ir.Lambda
register(Lambda)

__all__ = ["Lambda"]
