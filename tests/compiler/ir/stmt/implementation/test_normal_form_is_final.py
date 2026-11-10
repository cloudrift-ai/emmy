"""Normalization runs its steps once and does not check its own result. These tests are the check: the normal form of
every kernel the repository stores is one normalization leaves alone. A form that is not final would give two
spellings of one body two texts, and a stored kernel would read back as another."""

from __future__ import annotations

from pathlib import Path

import pytest

from emmy.compiler.graph import Graph
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import normalize_body, parse_body
from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR, repository_golden_paths
from tests.compiler.realization import helpers as corpus


def _golden_parameters():
    with repository_golden_paths() as paths:
        return [
            pytest.param(path, id=path.stem if path.parent == _RECORDS_DIR else f"{path.parent.parent.name}/{path.stem}") for path in paths
        ]


def _not_final(path: Path) -> list[str]:
    """The kernels of the golden at ``path`` whose normal form normalization changes again."""
    return [
        kernel.ref
        for kernel in GoldenFile.load(path).kernels
        for node in Graph.from_wire(kernel.loop_ir).nodes.values()
        if isinstance(node.op, LoopOp) and normalize_body(parse_body(str(node.op.body))) != node.op.body
    ]


@pytest.mark.parametrize("path", _golden_parameters())
def test_normalization_leaves_the_normal_form_of_every_golden_kernel_alone(path: Path) -> None:
    assert _not_final(path) == []


def test_normalization_leaves_the_normal_form_of_every_corpus_kernel_alone() -> None:
    assert [(path.name, ref) for path in corpus.case_files() for ref in _not_final(path)] == []
