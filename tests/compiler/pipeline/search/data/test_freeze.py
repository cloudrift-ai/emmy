"""The measurement freeze (v6) — the admission filter, freeze-twice determinism over the per-GPU YAML
directory, the round trip through a DB instance, and the loader's hard-error contract.

A fit must be a pure function of (repo, pinned data), so one command snapshots a DB instance's
``perf`` rows into a digest-pinned directory of per-GPU YAML files, and ``emmy dataset import`` loads
it back into a DB instance for the readers. These tests never touch a GPU — the plausibility physics
itself is covered by the shared fixtures in ``helpers.py``."""

from __future__ import annotations

import dataclasses
import hashlib
import json

import pytest
import yaml

from emmy.compiler.pipeline.search.data.freeze import (
    FREEZE_KIND,
    FREEZE_VER,
    KERNELS_NAME,
    MANIFEST_NAME,
    ROUTING_NAME,
    _row_line,
    freeze_reason,
    load_freeze,
    write_freeze,
)
from emmy.compiler.pipeline.search.db import RoutingRow, SearchDB
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION
from tests.compiler.pipeline.search.helpers import F16_MATMUL_FEATS, impossible_staged_feats, kernel_row
from tests.compiler.pipeline.search.helpers import GPU_5090 as _GPU
from tests.compiler.pipeline.search.helpers import perf_row as _row

_GPU2 = "NVIDIA GeForce RTX 4090"

# A small fp32 matmul's stamps, plausible at a few hundred µs.
_STAMPS = {
    "S_ext_free_prod": 4096.0,
    "S_ext_free_max": 64.0,
    "S_ext_reduce_max": 64.0,
    "S_ext_n_free_axis": 2.0,
    "S_ext_n_reduce_axis": 1.0,
    "S_loop_depth": 3.0,
    "S_dtype_f32": 3.0,
}


def _feats(**knobs) -> dict:
    """The stamps plus the tunables that must survive the round trip verbatim."""
    return {**_STAMPS, "TILE": "f2x2", "WORK": "t16x16", **knobs}


# ---------------------------------------------------------------------------
# freeze_reason — the admission filter
# ---------------------------------------------------------------------------


def test_reason_keeps_ordinary_row() -> None:
    assert freeze_reason(_row("k", us=500.0, knobs=_feats())) is None


def test_reason_keeps_bench_fail_row_as_negative() -> None:
    # A fail's median is the watchdog sentinel — absurd as a latency, but the row is a durable
    # "doesn't build/launch here" negative and must freeze.
    assert freeze_reason(_row("k", us=9.17, knobs=_feats(), status="bench_fail")) is None


def test_reason_drops_a_card_the_registry_does_not_know() -> None:
    # Its H_* features cannot be derived, so no reader could featurize it.
    row = dataclasses.replace(_row("k", us=500.0, knobs=_feats()), gpu="Mystery GPU")
    assert freeze_reason(row).startswith("unknown card")


def test_reason_drops_a_non_deployable_regime() -> None:
    assert freeze_reason(_row("k", us=500.0, knobs=_feats(), opt=1)) == "non-deployable regime (H_opt=1)"


def test_reason_drops_extra_compiler_flags() -> None:
    # Same card, same opt level, but a flag beside it (fast-math, say) is another regime.
    assert freeze_reason(_row("k", us=500.0, knobs=_feats(), flags="--use_fast_math")) == "non-default compiler flags"


def test_reason_drops_implausible_value() -> None:
    # The shared f16 mlp_down extents at 9.17 µs imply ~6500 TFLOP/s at the default hint of its
    # symbolic axis — and an honest 13 TFLOP/s when the row says it was benched at one token.
    assert "implausible value" in freeze_reason(_row("k", us=9.17, knobs=F16_MATMUL_FEATS))
    assert freeze_reason(_row("k", us=9.17, knobs=F16_MATMUL_FEATS, bindings={"m": 1})) is None


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    assert "impossible kernel" in freeze_reason(_row("k", us=2.02, knobs=impossible_staged_feats()))


# ---------------------------------------------------------------------------
# write_freeze / load_freeze round trip
# ---------------------------------------------------------------------------

# The definitions the rows are of: every measured kernel, a routing row whose parent and pieces have
# rows of their own, and one kernel nothing names. Every kernel carries the same stamps as the rows
# seeded on it, so what a reader gets back is what was written.
_KERNELS = [kernel_row(k, stamps=_STAMPS) for k in ("a3", "a1", "b", "dyn", "fail", "p", "c1", "orphan")]
_ROUTING = [RoutingRow(parent="p", arm={"PLACE@map.1/inner": "cut"}, children=("c1", "a3"))]

_SEED = [
    _row("a3", us=500.0, knobs=_feats()),
    _row("a1", us=900.0, knobs=_feats(), opt=1),  # the non-deployable twin an older store still holds
    _row("b", us=480.0, knobs=_feats(), gpu=_GPU2, cc=89),
    _row("dyn", us=520.0, knobs=_feats(), bindings={"seq_len": 512}),  # a dynamic kernel at the size it was benched
    _row("fail", us=60000.0, knobs=_feats(TILE="f4x4"), status="bench_fail", error="kernel 'k' did not complete"),
]
_N_KEPT = 4


def _seed_db(path, rows, kernels=_KERNELS, routing=()) -> None:
    db = SearchDB(path)
    db.record_kernels(kernels)
    db.record_routings(routing)
    db.record_perf_rows(rows)
    db.close()


def test_write_freeze_round_trip(tmp_path) -> None:
    db_path = tmp_path / "dataset.db"
    _seed_db(db_path, _SEED, _KERNELS, _ROUTING)
    out = tmp_path / "freeze"
    manifest = write_freeze(db_path, out, note="unit-test policy")

    assert manifest["kind"] == FREEZE_KIND
    assert manifest["freeze_ver"] == FREEZE_VER
    assert manifest["feat_ver"] == manifest["knob_ver"] == manifest["encoding_ver"] == FEATURIZER_VERSION
    assert manifest["counts"] == {"rows": _N_KEPT, "ok": 3, "bench_fail": 1, "per_gpu": {_GPU2: 1, _GPU: 3}, "kernels": 6, "routing": 1}
    assert manifest["policy_note"] == "unit-test policy"
    assert manifest["source_db"] == str(db_path.resolve())
    # One YAML per (gpu, cap), the two definition files beside them. A perf row stores its schedule
    # row only: its stamps are its kernel row's, and H_* features readers derive from the card.
    assert {name: info["kind"] for name, info in manifest["files"].items()} == {
        "nvidia_geforce_rtx_5090_sm120.yaml": "perf",
        "nvidia_geforce_rtx_4090_sm89.yaml": "perf",
        KERNELS_NAME: "kernels",
        ROUTING_NAME: "routing",
    }
    doc = yaml.safe_load((out / "nvidia_geforce_rtx_5090_sm120.yaml").read_text())
    assert doc["gpu_name"] == _GPU and doc["compute_cap"] == [12, 0]
    assert all(set(c["knobs"]) == {"TILE", "WORK"} for c in doc["configs"])
    kernels_doc = yaml.safe_load((out / KERNELS_NAME).read_text())
    assert all(k["stamps"] == _STAMPS for k in kernels_doc["kernels"])

    loaded_manifest, rows, kernels, routing = load_freeze(out)
    assert loaded_manifest == manifest
    # The kernels a frozen row or a routing row names travel; the one nothing names does not, and
    # neither does the kernel whose only row did not freeze.
    assert {k.exact_identity for k in kernels} == {"a3", "b", "dyn", "fail", "p", "c1"}
    assert all(k in _KERNELS for k in kernels)
    assert routing == _ROUTING
    by_key = {r.kernel: r for r in rows}
    assert set(by_key) == {"a3", "b", "dyn", "fail"}, "the non-deployable twin does not freeze"
    seeded = {r.kernel: r for r in _SEED}
    for key, row in by_key.items():
        # Everything a live bench wrote comes back — the same key, so an import lands where the
        # measurement would have, and the stamps reassembled from the kernel row — except the
        # source, which now names the freeze.
        assert dataclasses.replace(row, source="measured") == seeded[key]
        assert row.source == f"freeze:{manifest['sha256'][:12]}"
    assert (by_key["b"].gpu, by_key["b"].cc, by_key["b"].opt, by_key["b"].flags) == (_GPU2, 89, 3, "")
    assert by_key["dyn"].bindings == {"seq_len": 512}


def test_freeze_twice_same_digest(tmp_path) -> None:
    db_path = tmp_path / "dataset.db"
    _seed_db(db_path, _SEED)
    m1 = write_freeze(db_path, tmp_path / "f1")
    m2 = write_freeze(db_path, tmp_path / "f2")
    assert m1["sha256"] == m2["sha256"]
    assert m1["files"] == m2["files"]
    for name in m1["files"]:
        assert (tmp_path / "f1" / name).read_bytes() == (tmp_path / "f2" / name).read_bytes()


def test_freeze_digest_insertion_order_independent(tmp_path) -> None:
    _seed_db(tmp_path / "fwd.db", _SEED)
    _seed_db(tmp_path / "rev.db", list(reversed(_SEED)), list(reversed(_KERNELS)))
    assert write_freeze(tmp_path / "fwd.db", tmp_path / "fwd")["sha256"] == write_freeze(tmp_path / "rev.db", tmp_path / "rev")["sha256"]


def test_an_imported_freeze_refreezes_to_the_same_digest(tmp_path) -> None:
    # freeze -> import into a fresh instance -> freeze again: the digest must not drift, so a
    # dataset rebuilt from a checked-in freeze reproduces it exactly.
    _seed_db(tmp_path / "a.db", _SEED, _KERNELS, _ROUTING)
    m1 = write_freeze(tmp_path / "a.db", tmp_path / "f1")
    frozen = load_freeze(tmp_path / "f1")
    _seed_db(tmp_path / "b.db", frozen.perf, frozen.kernels, frozen.routing)
    assert write_freeze(tmp_path / "b.db", tmp_path / "f2")["sha256"] == m1["sha256"]


# ---------------------------------------------------------------------------
# the loader's hard-error contract
# ---------------------------------------------------------------------------


def _frozen(tmp_path):
    db_path = tmp_path / "dataset.db"
    _seed_db(db_path, _SEED, _KERNELS, _ROUTING)
    out = tmp_path / "freeze"
    write_freeze(db_path, out)
    return out


def _restamp(out, name: str, key: str, edit) -> None:
    """Edit one freeze file's rows in place, keeping its digest honest so schema validation (not
    integrity) is what trips."""
    doc = yaml.safe_load((out / name).read_text())
    for payload in doc[key]:
        edit(payload)
    (out / name).write_text(yaml.safe_dump(doc, sort_keys=True, width=120))
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    digest = hashlib.sha256()
    for payload in doc[key]:
        digest.update(_row_line(payload))
    manifest["files"][name]["sha256"] = digest.hexdigest()
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))


def test_load_freeze_digest_mismatch_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    f = out / "nvidia_geforce_rtx_4090_sm89.yaml"
    f.write_text(f.read_text().replace("median: 480.0", "median: 4.0"))
    with pytest.raises(RuntimeError, match="corrupt"):
        load_freeze(out)


@pytest.mark.parametrize("name", ["nvidia_geforce_rtx_4090_sm89.yaml", KERNELS_NAME, ROUTING_NAME])
def test_load_freeze_missing_listed_file_hard_error(tmp_path, name) -> None:
    out = _frozen(tmp_path)
    (out / name).unlink()
    with pytest.raises(RuntimeError, match="missing"):
        load_freeze(out)


def test_load_freeze_malformed_definition_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    _restamp(out, KERNELS_NAME, "kernels", lambda k: k.update(loop_ir="not-a-program"))
    with pytest.raises(RuntimeError, match="lacks a kernel definition"):
        load_freeze(out)


def test_load_freeze_row_without_its_kernel_hard_error(tmp_path) -> None:
    """A perf row's stamps are read off its kernel row, so a freeze whose rows name a kernel its
    kernels file lacks is not self-contained."""
    out = _frozen(tmp_path)
    _restamp(out, "nvidia_geforce_rtx_4090_sm89.yaml", "configs", lambda c: c.update(kernel="ghost"))
    with pytest.raises(RuntimeError, match="names kernel 'ghost'"):
        load_freeze(out)


@pytest.mark.parametrize("field", ["freeze_ver", "feat_ver"])
def test_load_freeze_version_mismatch_hard_error(tmp_path, field) -> None:
    """A freeze from another format or featurizer generation reports itself rather than being read
    under today's code: the rows carry no version of their own, the manifest's is theirs."""
    out = _frozen(tmp_path)
    manifest = json.loads((out / MANIFEST_NAME).read_text())
    manifest[field] += 1
    (out / MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match=field):
        load_freeze(out)


def test_load_freeze_foreign_dir_hard_error(tmp_path) -> None:
    plain = tmp_path / "not-a-freeze"
    plain.mkdir()
    with pytest.raises(RuntimeError, match="not a measurement freeze"):
        load_freeze(plain)


def test_load_freeze_malformed_row_hard_error(tmp_path) -> None:
    out = _frozen(tmp_path)
    _restamp(out, "nvidia_geforce_rtx_5090_sm120.yaml", "configs", lambda c: c.update(knobs="not-a-mapping"))
    with pytest.raises(RuntimeError, match="lacks kernel/bindings/knobs"):
        load_freeze(out)


def test_write_freeze_nothing_survives_hard_error(tmp_path) -> None:
    db_path = tmp_path / "dataset.db"
    _seed_db(db_path, [_row("a3", us=500.0, knobs=_feats(), opt=1)])
    with pytest.raises(RuntimeError, match="no freezable rows"):
        write_freeze(db_path, tmp_path / "freeze")


def test_write_freeze_refuses_to_replace_non_freeze_dir(tmp_path) -> None:
    db_path = tmp_path / "dataset.db"
    _seed_db(db_path, _SEED)
    target = tmp_path / "precious"
    target.mkdir()
    (target / "notes.txt").write_text("do not delete\n")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        write_freeze(db_path, target)
    assert (target / "notes.txt").exists()


def test_open_readonly_names_a_file_that_is_not_a_db(tmp_path) -> None:
    # A freeze file handed to a DB reader must fail at open with a named reason, not a bare
    # sqlite DatabaseError deep inside a PRAGMA.
    garbage = tmp_path / "manifest.json"
    garbage.write_text(json.dumps({"kind": FREEZE_KIND}) + "\n")
    with pytest.raises(RuntimeError, match="not a sqlite database"):
        SearchDB.open_readonly(garbage)


def test_an_lfs_pointer_is_named_rather_than_parsed(tmp_path) -> None:
    """A checkout without LFS leaves a three-line pointer where the payload should be, and a pointer is
    valid YAML — it parses to a string, and the first key lookup fails as ``TypeError: string indices must
    be integers``, which says nothing about the real problem. This is how a checked-in freeze once reached
    ``main`` with red CI: the failure named a type error, not a missing file."""
    out = _frozen(tmp_path)
    name = next(iter(json.loads((out / MANIFEST_NAME).read_text())["files"]))
    (out / name).write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")

    with pytest.raises(RuntimeError, match="git-LFS pointer"):
        load_freeze(out)
