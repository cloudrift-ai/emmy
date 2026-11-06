"""Loop IR — post-fusion kernel representation and validation.

``LoopOp`` normalizes its body before validation. The statement vocabulary lives in ``ir/stmt``; body analysis
and the body assembly fusion drives live here, and the splicer supplies graph regions to that engine.
Common types are re-exported here for Loop IR callers.
"""

from emmy.compiler.ir.loop.ir import (
    Accum,
    Affine,
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
    Statement,
    Write,
)
from emmy.compiler.ir.loop.splicer import UnfusableStmt, observes_running_accumulator, splice_graph, splice_loops
from emmy.compiler.ir.sigma import Sigma

__all__ = [
    "Accum",
    "Affine",
    "Assign",
    "Axis",
    "Cond",
    "Load",
    "Loop",
    "BodyAnalysis",
    "LoopOp",
    "Scope",
    "Select",
    "SelectBranch",
    "Sigma",
    "Statement",
    "Write",
    "iter_body",
    "map_body",
    "UnfusableStmt",
    "observes_running_accumulator",
    "splice_graph",
    "splice_loops",
]
