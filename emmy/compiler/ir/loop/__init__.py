"""Loop IR — post-fusion kernel representation and validation.

``LoopOp`` normalizes its body before validation. The statement vocabulary, body analysis and
reconstruction live in ``ir/stmt``; the Loop IR splicer supplies graph regions to that shared engine.
Common types are re-exported here for Loop IR callers.
"""

from emmy.compiler.ir.loop.ir import (
    Accum,
    Assign,
    Axis,
    BodyAnalysis,
    Cond,
    Load,
    Loop,
    LoopOp,
    Scope,
    Select,
    SelectBranch,
    Stmt,
    Write,
)
from emmy.compiler.ir.loop.splicer import UnfusableStmt, observes_running_accumulator, splice_graph, splice_loops
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt.builder import BodyBuilder

__all__ = [
    "Accum",
    "Assign",
    "Axis",
    "Cond",
    "Load",
    "Loop",
    "BodyBuilder",
    "BodyAnalysis",
    "LoopOp",
    "Scope",
    "Select",
    "SelectBranch",
    "Sigma",
    "Stmt",
    "Write",
    "iter_body",
    "map_body",
    "UnfusableStmt",
    "observes_running_accumulator",
    "splice_graph",
    "splice_loops",
]
