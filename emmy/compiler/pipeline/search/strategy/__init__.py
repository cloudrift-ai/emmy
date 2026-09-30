"""Search strategies — the search SHAPES over the engine's loop (see :mod:`.base`).

``policy`` answers the question one resolution asks (which option at a fork); ``strategy`` composes
resolutions (retries, the rejection sink, and what the result means)."""

from emmy.compiler.pipeline.search.strategy.base import SearchStrategy
from emmy.compiler.pipeline.search.strategy.greedy import GreedyStrategy

__all__ = ["GreedyStrategy", "SearchStrategy"]
