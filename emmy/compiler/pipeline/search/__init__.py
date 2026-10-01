"""Search infrastructure: the candidate, the greedy pick, the on-disk kernel + perf store, and the
prior that ranks where nothing measured decides.

- :mod:`.candidate` — :class:`Candidate` / :class:`Cursor` data classes.
- :mod:`.policy` — :func:`greedy_decide` (``greedy``, the deterministic ``Run.resolve`` pick).
- :mod:`.db` — :class:`SearchDB` SQLite store (the kernels, the decisions that minted them and their
  measurements).
- :mod:`.bench_record` — the ONE writer for a kernel measurement and for a failed bench.
- :mod:`.inventory` — the splice watcher that hears which kernels a lowering minted and which
  kernel-set decisions it took, and the routing-row writer.
- :mod:`.prior` — the :class:`Prior` a compile ranks with.

Op identity and the rewrite-chain walk live on the ops themselves —
:meth:`~emmy.compiler.ir.base.Op.identity_key`, :attr:`~emmy.compiler.ir.base.Op.dialect`,
:meth:`~emmy.compiler.ir.base.Op.source_chain`.
"""

from emmy.compiler.pipeline.search.candidate import Candidate, Cursor
from emmy.compiler.pipeline.search.db import PerfRow, PerfStats, SearchDB
from emmy.compiler.pipeline.search.policy import greedy_decide

__all__ = [
    "Candidate",
    "Cursor",
    "PerfRow",
    "PerfStats",
    "SearchDB",
    "greedy_decide",
]
