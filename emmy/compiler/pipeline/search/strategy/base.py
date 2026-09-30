"""SearchStrategy — the ABC for search SHAPES over the engine's loop.

A search strategy owns everything ABOVE the loop: how many resolutions run, over which pass lists,
with which decide callback inside, and what the results mean together. ``GreedyStrategy`` (one
deterministic resolve plus retry orchestration) is the one shape today. Contrast the protocol a
shape composes with: ``pipeline.strategy.PipelineStrategy`` reacts to the events a loop emits
(provenance, identity, the kernel inventory) without steering it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from emmy.compiler.context import Context
    from emmy.compiler.graph import Graph


class SearchStrategy(ABC):
    """One search shape: constructor carries the shape's configuration (backend, DB, the pipeline
    or pass lists it composes); :meth:`run` drives one input graph to the shape's result."""

    @abstractmethod
    def run(self, graph: Graph, ctx: Context | None = None):
        """Drive ``graph`` to this shape's result — a terminal ``Graph``."""
