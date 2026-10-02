"""The ``emmy fit`` run harness — full-train fit, cross-validation, and metrics assembly as one
pure function.

:func:`run_fit` owns the *shape* of a fit run and none of its *content*: the trainer arrives as one
configured object, the dataset as pre-built
:class:`GoldenGroup` lists, and every non-deterministic input (dates, repo commit, CLI args)
pre-rendered inside ``header``. No I/O, no clock, no argparse: the same inputs produce the same
``(metrics, fit)`` pair, so the harness is testable on synthetic groups with a stub trainer.

It returns the FIT, not an artifact. Assembling one is shipping policy — which dynamic weight set a
fit with no dynamic groups goes out with, what provenance it carries — and that belongs with the
command layer (:mod:`emmy.commands.fit`), which also keeps what ``pipeline/`` must not import: the
snippet-tracing group builder, the CLI, and the file writing.
"""

from __future__ import annotations

import logging

from emmy.compiler.pipeline.search.dataset.group import GoldenGroup
from emmy.compiler.pipeline.search.prior.fit.cv import build_metrics, run_folds


def run_fit(groups: list[GoldenGroup], skipped: list[tuple[str, str, str]], *, trainer, folds, header):
    """One complete fit run → ``(metrics, fit)``. ``trainer.fit(groups)`` produces the shippable
    model over every group, and the same trainer fits one model per :func:`~.cv.run_folds` fold. ``folds`` is the fold count (``0`` skips
    cross-validation entirely and the metrics carry an empty ``cv`` block).

    The cross-validation block is the folds' own result, not a map keyed by fold axis: there is one axis
    (shape) and there has only ever been one, so the key was a wrapper a reader had to step through."""
    full = trainer.fit(groups)

    # Each fold re-fits — silence the fit package's per-group rank logging for that stage
    # (the full-train fit above already reported it).
    pkg_logger = logging.getLogger(__package__)
    level = pkg_logger.level
    pkg_logger.setLevel(logging.WARNING)
    try:
        cv = run_folds(groups, trainer=trainer, k=folds) if folds else {}
    finally:
        pkg_logger.setLevel(level)

    return build_metrics(header, groups, skipped, full, cv), full
