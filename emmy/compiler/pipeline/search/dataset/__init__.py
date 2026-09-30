"""The training data: the values a ranking question is asked over, and the document they travel in.

- :class:`Group` — one candidate pool packed as a matrix plus one label per row (``group.py``). The base says nothing
  about what the labels mean, which is all a ranking metric needs; :class:`GoldenGroup` is the kind whose labels
  MARK the rows goldens verified (``golden_ids``, and the :class:`GoldenPool` s it was built from), and
  :class:`MeasuredGroup` the kind whose labels ARE the measured microseconds.
- :class:`GoldenPool` — one kernel's schedule space on one card, in one regime, at one set of sizes, with the
  :class:`GoldenRow` s a golden file verified on it, and the :class:`KernelDef` it is enumerated from (``pool.py``,
  ``kernel.py``). The regime vocabulary (:data:`REGIME_PINS`, :func:`regime_of`) lives with the pool.
- :class:`Dataset` — the groups as a directory: ``manifest.json`` plus one ``.npy`` per group (``document.py``),
  written by ``emmy db export`` and read by ``emmy fit`` and ``emmy eval prior``.
- :class:`Sample` — the per-row read-view of a golden record, and the cheap :class:`ShapeKey` structural identity
  (``sample.py``, ``shape.py``).

Nothing here reads a DB, and nothing here may import :mod:`~..prior`: this package describes candidates and their
labels; where they come from is ``db/export.py``'s business, and what a score MEANS is the layer above (both guarded
by ``tests/architecture/test_layering.py``)."""

from __future__ import annotations

from emmy.compiler.pipeline.search.dataset.document import Dataset, repo_commit
from emmy.compiler.pipeline.search.dataset.group import (
    DEFAULT_FEATURES,
    GoldenGroup,
    Group,
    MeasuredGroup,
    feature_view,
    pack_features,
)
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.dataset.pool import REGIME_PINS, GoldenPool, GoldenRow, regime_of
from emmy.compiler.pipeline.search.dataset.sample import Sample, measured_features
from emmy.compiler.pipeline.search.dataset.shape import ShapeKey, is_matmul, op_label

__all__ = [
    "DEFAULT_FEATURES",
    "REGIME_PINS",
    "Dataset",
    "GoldenGroup",
    "GoldenPool",
    "GoldenRow",
    "Group",
    "KernelDef",
    "MeasuredGroup",
    "Sample",
    "ShapeKey",
    "feature_view",
    "is_matmul",
    "measured_features",
    "op_label",
    "pack_features",
    "regime_of",
    "repo_commit",
]
