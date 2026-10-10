"""``Dataset`` — the training data as a directory: a manifest beside one matrix file per group, round-tripped exactly;
the same groups twice write the same bytes; a directory that is no dataset is refused rather than replaced; an
export under another featurizer version is refused by name."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.dataset import Dataset, GoldenGroup, GoldenPool, GoldenRow, MeasuredGroup, pack_features
from tests.compiler.pipeline.search.helpers import kernel_row

_STAMP = {"S_ext_n_symbolic_axis": 0.0}


def _dataset() -> Dataset:
    """One golden group over one pool (a second row lacking a feature, so the matrix carries the NaN fill) and one
    measured group, with the bookkeeping an export writes beside them."""
    pool = GoldenPool("gpuA", (12, 0), "", kernel_row("k1", name="k_one"), {"m": 512}, (GoldenRow({"TILE": "a"}, 1.5, "golden:x"),))
    golden = GoldenGroup.over(
        "gpuA/k_one",
        "k_one",
        "warp",
        "gpuA",
        "shape",
        pack_features([{"D_a": 1.0, **_STAMP}, {"D_a": 2.0, "D_b": 3.0, **_STAMP}]),
        1,
        10,
        pools=(pool,),
    )
    measured = MeasuredGroup.from_measured(
        "gpuA/sig@O3", "gpuA", "sig", 3.0, [10.0, 20.0], [{"D_a": 1.0, **_STAMP}, {"D_a": 2.0, **_STAMP}]
    )
    provenance = {
        "source": "test.db",
        "sources": {"golden:x": 1},
        "pool_sample": 0,
        "seed": 0,
        "feat_ver": features.FEATURIZER_VERSION,
        "compiler": "abc",
    }
    return Dataset([golden], [measured], [("gpuA", "k_two", "did not lower")], {"golden": {"bench_fail": 1}, "measured": {}}, provenance)


def test_a_dataset_round_trips_through_its_directory(tmp_path):
    dataset = _dataset()
    out = dataset.dump(tmp_path / "dataset")
    assert sorted(p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()) == [
        "golden/0000.npy",
        "manifest.json",
        "measured/0000.npy",
    ]
    back = Dataset.load(out)
    [g], [m] = back.golden, back.measured
    want = dataset.golden[0]
    assert (g.key, g.name, g.tier, g.gpu, g.shape, g.dynamic, g.feat_names, g.total, g.golden_ids) == (
        want.key,
        want.name,
        want.tier,
        want.gpu,
        want.shape,
        want.dynamic,
        want.feat_names,
        want.total,
        want.golden_ids,
    )
    assert np.array_equal(g.feats, want.feats, equal_nan=True) and g.pools == want.pools
    assert m.latency_us.tolist() == [10.0, 20.0] and m.h_opt == 3.0 and m.key == "gpuA/sig@O3"
    assert (back.skipped, back.dropped, back.provenance) == (dataset.skipped, dataset.dropped, dataset.provenance)
    manifest = json.loads((out / "manifest.json").read_text())
    assert len(manifest["kernels"]) == 1 and manifest["golden"][0]["pools"][0]["kernel"] == 0  # interned by identity


def test_the_same_groups_write_the_same_bytes(tmp_path):
    first, second = _dataset().dump(tmp_path / "a"), _dataset().dump(tmp_path / "b")
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()
    _dataset().dump(tmp_path / "a")  # a dataset replaces a dataset
    assert (first / "manifest.json").read_bytes() == (second / "manifest.json").read_bytes()


def test_a_directory_that_is_no_dataset_is_refused(tmp_path):
    (tmp_path / "notes").mkdir()
    (tmp_path / "notes" / "keep.txt").write_text("mine")
    with pytest.raises(RuntimeError, match="not a dataset"):
        _dataset().dump(tmp_path / "notes")
    assert (tmp_path / "notes" / "keep.txt").read_text() == "mine"
    with pytest.raises(FileNotFoundError, match="no dataset"):
        Dataset.load(tmp_path / "notes")


def test_an_export_with_the_old_placement_labels_is_refused(tmp_path):
    out = _dataset().dump(tmp_path / "old-labels")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["version"] = 2
    (out / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="version 2 dataset.*re-export"):
        Dataset.load(out)


def test_an_export_under_another_featurizer_version_is_refused(tmp_path):
    out = _dataset().dump(tmp_path / "dataset")
    manifest = json.loads((out / "manifest.json").read_text())
    manifest["provenance"]["feat_ver"] = features.FEATURIZER_VERSION - 1
    (out / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="featurizer version"):
        Dataset.load(out)
