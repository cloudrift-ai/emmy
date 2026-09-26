"""The measurement freeze — the admission filter, the golden file per card a DB instance freezes to, and the
round trip: a freeze re-lowered by ``emmy dataset import`` is the rows it was written from.

The DB a freeze is written from is what a tune of the realization corpus leaves behind (``helpers.tuned_db``),
so every kind of kernel the compiler mints is on the way: a fused kernel, a twisted attention kernel, cut
pieces formed as kernels of their own, and the pieces of an attention split, which no body of their own
re-lowers and which the freeze reaches through their parent's program. These tests never touch a GPU."""

from __future__ import annotations

import dataclasses

import pytest

from emmy.compiler.pipeline.knob import METADATA_PREFIXES
from emmy.compiler.pipeline.search.data.freeze import REGIME_PINS, freeze_documents, freeze_reason, freeze_source, regime_of, write_freeze
from emmy.compiler.pipeline.search.db import SearchDB, knobs_json
from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.golden.evidence import import_file
from tests.compiler.pipeline.search.helpers import F16_MATMUL_FEATS, impossible_staged_feats, tuned_db
from tests.compiler.pipeline.search.helpers import perf_row as _row

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
#: One case per kind of kernel: a fused kernel, a twisted one, a cut into formed pieces, an attention split
#: whose pieces are formed from no loop op.
CASES = (
    "fused/norm-linear-f16-scalar-reduce.json",
    "attention/sdpa-hd128-softmax-v-mma.json",
    "fused/linear-add-place-cut-sm70.json",
    "attention/sdpa-gqa-decode-split-kv.json",
)


def _feats(**knobs) -> dict:
    return {**_STAMPS, "TILE": "f2x2", "WORK": "t16x16", **knobs}


# ---------------------------------------------------------------------------
# freeze_reason — the admission filter
# ---------------------------------------------------------------------------


def test_reason_keeps_a_row_of_either_precision_regime(monkeypatch) -> None:
    """The fast-math flag alone decides a row's regime, spelled once and not read off this shell: any other flag
    a row was compiled with, and whatever ``EMMY_NVCC_FLAGS`` holds when the freeze runs, leave it where its
    context put it."""
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-lineinfo")
    assert freeze_reason(_row("k", us=500.0, knobs=_feats())) is None
    assert freeze_reason(_row("k", us=500.0, knobs=_feats(), flags="--use_fast_math")) is None
    assert freeze_reason(_row("k", us=500.0, knobs=_feats(), flags="-lineinfo --use_fast_math")) is None
    assert REGIME_PINS == {"": {"FAST_MATH": False}, "--use_fast_math": {"FAST_MATH": True}}
    assert regime_of("-lineinfo --use_fast_math") == "--use_fast_math" and regime_of("-lineinfo") == ""


def test_reason_drops_a_failed_bench() -> None:
    # A fail's median is the watchdog sentinel, not a measurement; the tune DB keeps the failure.
    assert freeze_reason(_row("k", us=9.17, knobs=_feats(), status="bench_fail")) == "bench_fail: not a measurement"


def test_reason_drops_a_card_the_registry_does_not_know() -> None:
    # Its H_* features cannot be derived, so no reader could featurize it.
    row = dataclasses.replace(_row("k", us=500.0, knobs=_feats()), gpu="Mystery GPU")
    assert freeze_reason(row).startswith("unknown card")


def test_reason_drops_a_non_deployable_regime() -> None:
    assert freeze_reason(_row("k", us=500.0, knobs=_feats(), opt=1)) == "non-deployable regime (H_opt=1)"


def test_reason_drops_implausible_value() -> None:
    # The shared f16 mlp_down extents at 9.17 µs imply ~6500 TFLOP/s at the default hint of its
    # symbolic axis — and an honest 13 TFLOP/s when the row says it was benched at one token.
    assert "implausible value" in freeze_reason(_row("k", us=9.17, knobs=F16_MATMUL_FEATS))
    assert freeze_reason(_row("k", us=9.17, knobs=F16_MATMUL_FEATS, bindings={"m": 1})) is None


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    assert "impossible kernel" in freeze_reason(_row("k", us=2.02, knobs=impossible_staged_feats()))


# ---------------------------------------------------------------------------
# the freeze and its round trip
# ---------------------------------------------------------------------------


def _measured(db: SearchDB) -> set[tuple]:
    """Every row as a freeze carries it: the kernel, its bindings, its schedule row, its median and its regime."""

    def schedule(row):
        return knobs_json({k: v for k, v in row.knobs.items() if not k.startswith(METADATA_PREFIXES)})

    return {(r.kernel, knobs_json(r.bindings), schedule(r), r.stats.median, r.flags) for r in db.iter_perf_rows() if r.status == "ok"}


def _definitions(db: SearchDB) -> dict[str, tuple]:
    measured = {r.kernel for r in db.iter_perf_rows()}
    return {k.exact_identity: (k.structural_identity, k.stamps, k.formed) for k in db.iter_kernels() if k.exact_identity in measured}


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    """The tune DB the freeze tests are written from, and its path."""
    path = tmp_path_factory.mktemp("tune") / "autotune.db"
    db = tuned_db(path, CASES)
    yield db, path
    db.close()


def test_a_freeze_is_a_golden_file_per_card_that_re_lowers_to_the_rows_it_was_written_from(tuned, tmp_path) -> None:
    tuned, tuned_path = tuned
    """One document per card, valid as a golden file; every formed kernel a config of its own; the attention
    split's pieces, formed from no loop op, under their parent's program with the split as a routing entry and
    their rows as receipts. Imported into a fresh instance, the rows come back with the same kernels, schedule
    rows, medians and regimes, the kernels with the same identities and stamps — the current compiler's, since
    the file stores neither."""

    documents, dropped = freeze_documents(tuned)
    assert dropped == {} and set(documents) == {"nvidia_geforce_rtx_5090_sm120.json", "nvidia_tesla_v100_sxm2_16gb_sm70.json"}
    for document in documents.values():
        GoldenFile.from_wire(document).check()
    definitions = _definitions(tuned)
    unformed = {identity for identity, (_deploy, _stamps, formed) in definitions.items() if not formed}
    assert unformed, "the attention split mints pieces no loop op forms"
    rtx = documents["nvidia_geforce_rtx_5090_sm120.json"]
    routes = [config for config in rtx["configs"] if any("kernel_set" in entry for entry in config["realizations"])]
    [route] = routes
    [decision] = [entry for entry in route["realizations"] if "measurements" not in entry]
    receipts = [entry for entry in route["realizations"] if "measurements" in entry]
    kernels = {kernel.exact_identity: kernel for kernel in tuned.iter_kernels()}
    [split] = [decision_row for decision_row in tuned.iter_routing() if set(decision_row.children) & unformed]
    assert kernels[split.parent].formed and decision["identity"] == kernels[split.parent].structural_identity
    assert decision["knobs"] == split.arm
    assert all(entry["kernel_set"] == [decision["name"]] for entry in receipts)
    assert {entry["identity"] for entry in receipts} == {definitions[identity][0] for identity in unformed}
    assert all("kernel_set" not in entry for config in rtx["configs"] if config is not route for entry in config["realizations"])

    digests = write_freeze(tuned_path, tmp_path / "freeze")
    assert set(digests) == set(documents)
    again = SearchDB()
    for name in sorted(digests):
        counts = import_file(again, tmp_path / "freeze" / name)
        assert not counts["did not lower"] and not counts["identities no kernel carries"], (name, counts)
    assert _measured(again) == _measured(tuned)
    assert _definitions(again) == definitions
    assert set(again.perf_sources()) == {freeze_source(tmp_path / "freeze" / name) for name in digests}
    assert all(source.startswith("freeze:") for source in again.perf_sources())


def test_freezing_the_same_rows_twice_yields_the_same_bytes(tuned, tmp_path) -> None:
    _db, tuned_path = tuned
    first, second = write_freeze(tuned_path, tmp_path / "f1"), write_freeze(tuned_path, tmp_path / "f2")
    assert first == second
    for name in first:
        assert (tmp_path / "f1" / name).read_bytes() == (tmp_path / "f2" / name).read_bytes()


def test_both_precision_lanes_freeze_as_pinned_rows_and_import_apart(tmp_path) -> None:
    """A row measured with fast math on and the same kernel's row with it off are two regimes: each entry
    carries its pin, and the import files each under its own context."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.compiler.pipeline.search.helpers import CARDS

    path = tmp_path / "autotune.db"
    db = tuned_db(path, CASES[:1])
    [row] = list(db.iter_perf_rows())
    with pinned_knobs({"FAST_MATH": True}):
        ctx = Context.from_target((12, 0), gpu_name=CARDS[(12, 0)], compile_flags="--use_fast_math")
        db.record_perf(ctx, row.kernel, bindings=row.bindings, knobs=row.knobs, backend="cuda", status="ok", stats=row.stats)
    assert {r.flags for r in db.iter_perf_rows()} == {"", "--use_fast_math"}
    documents, _dropped = freeze_documents(db)
    [document] = documents.values()
    entries = [entry for config in document["configs"] for entry in config["realizations"]]
    assert sorted(entry["pins"]["FAST_MATH"] for entry in entries) == [False, True]
    records = GoldenFile.from_wire(document).records()
    assert {tuple(name for name, _value in record.pins) for record in records} == {("FAST_MATH",)}
    write_freeze(path, tmp_path / "freeze")
    db.close()
    again = SearchDB()
    import_file(again, next((tmp_path / "freeze").glob("*.json")))
    lanes = sorted((r.flags, r.stats.median) for r in again.iter_perf_rows())
    assert lanes == [("", row.stats.median), ("--use_fast_math", row.stats.median)]


def test_a_kernel_benched_at_two_sizes_freezes_as_two_programs(tmp_path) -> None:
    """A row's sizes travel as the program's hints, never as golden ``bindings``, which would make the dims static
    and name another kernel: the same symbolic kernel benched at two sizes is two loop programs, and each row comes
    back at the size it was benched at."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.wire import symbolic_vars
    from tests.compiler.pipeline.search.helpers import CARDS

    path = tmp_path / "autotune.db"
    db = tuned_db(path, ("reduce/combine-amax-ilp-symbolic.json",))
    [row] = list(db.iter_perf_rows())
    assert row.bindings == {"seq_len": 512}
    with pinned_knobs({"FAST_MATH": row.flags != ""}):
        ctx = Context.from_target((12, 0), gpu_name=CARDS[(12, 0)], compile_flags=row.flags)
        db.record_perf(ctx, row.kernel, bindings={"seq_len": 128}, knobs=row.knobs, backend="cuda", status="ok", stats=row.stats)
    documents, _dropped = freeze_documents(db)
    [document] = documents.values()
    assert all(entry["bindings"] == {} for config in document["configs"] for entry in config["realizations"])
    assert [symbolic_vars(wire) for wire in document["loops"]] == [{"seq_len"}, {"seq_len"}]
    dims = [dim for wire in document["loops"] for node in wire["nodes"] for _n, _d, shape in node["outputs"] for dim in shape]
    hints = sorted(dim["hint"] for dim in dims if isinstance(dim, dict))
    assert hints[0] == 128 and hints[-1] == 512
    write_freeze(path, tmp_path / "freeze")
    db.close()
    again = SearchDB()
    import_file(again, next((tmp_path / "freeze").glob("*.json")))
    assert sorted(r.bindings["seq_len"] for r in again.iter_perf_rows()) == [128, 512]
    assert {r.kernel for r in again.iter_perf_rows()} == {row.kernel}


def test_a_golden_files_rows_and_failures_are_not_frozen(tmp_path) -> None:
    """A row a compile imported from a golden file is the file's, not this machine's; a failed bench is no
    measurement. Neither is written, and the freeze says why."""
    from emmy.compiler.pipeline.search.db import PerfStats

    path = tmp_path / "autotune.db"
    db = tuned_db(path, CASES[:1], source="golden:abcdef012345")
    [row] = list(db.iter_perf_rows())
    failed = PerfStats(median=2e6, min=2e6, max=2e6, mean=2e6, variance=0.0, n_samples=0)
    hung = dataclasses.replace(row, source="measured", knobs={**row.knobs, "WORK": "t8"}, status="bench_fail", stats=failed, error="hung")
    db.record_perf_row(hung)
    documents, dropped = freeze_documents(db)
    assert documents == {} and dropped == {"a golden file's row": 1, "bench_fail": 1}
    with pytest.raises(RuntimeError, match="no freezable rows"):
        write_freeze(path, tmp_path / "freeze")
    db.close()


def test_write_freeze_refuses_to_replace_a_directory_that_is_not_a_freeze(tuned, tmp_path) -> None:
    _db, tuned_path = tuned
    target = tmp_path / "precious"
    target.mkdir()
    (target / "notes.txt").write_text("do not delete\n")
    with pytest.raises(RuntimeError, match="refusing to replace"):
        write_freeze(tuned_path, target)
    assert (target / "notes.txt").exists()
    write_freeze(tuned_path, tmp_path / "freeze")
    write_freeze(tuned_path, tmp_path / "freeze")  # a freeze replaces a freeze
    assert GoldenFile.load(next((tmp_path / "freeze").glob("*.json")))


def test_an_lfs_pointer_is_named_rather_than_parsed(tmp_path) -> None:
    """A checkout without LFS leaves a three-line pointer where the payload should be, and a pointer is
    valid YAML — it parses to a string, and the first key lookup fails with a type error that says nothing
    about the real problem. This is how a checked-in freeze once reached ``main`` with red CI."""

    pointer = tmp_path / "nvidia_geforce_rtx_5090_sm120.json"
    pointer.write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 1\n")
    with pytest.raises(ValueError, match="git-LFS pointer"):
        import_file(SearchDB(), pointer)


@pytest.mark.xdist_group("golden_import_rtx5090")
def test_the_rtx_5090_hardware_goldens_rows_round_trip_through_a_freeze(tmp_path) -> None:
    """A card's recorded rows — attention and softmax kernels, split winners' routing rows, cut receipts — imported
    as a tune of that card would leave them, freeze to one file that re-lowers to the same rows and kernels. The
    same file imported straight from the repository, its traced slices entering at the lowering passes, files the
    same rows too: a golden file is a source ``emmy dataset import`` accepts."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden.evidence import import_goldens
    from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from tests.compiler.pipeline.search.helpers import GPU_5090

    path = _RECORDS_DIR / "rtx5090_sm120.json"
    records = GoldenFile.load(path).records()
    tuned_path = tmp_path / "autotune.db"
    tuned = SearchDB(tuned_path)
    with pinned_knobs({"FAST_MATH": False}):
        counts = import_goldens(tuned, Context.from_target((12, 0), gpu_name=GPU_5090, compile_flags=""), records, source="measured")
    assert counts["perf rows"] >= 30
    documents, dropped = freeze_documents(tuned)
    assert dropped == {} and list(documents) == ["nvidia_geforce_rtx_5090_sm120.json"]
    [name] = write_freeze(tuned_path, tmp_path / "freeze")
    again = SearchDB()
    counts = import_file(again, tmp_path / "freeze" / name)
    assert not counts["did not lower"] and not counts["identities no kernel carries"], counts
    assert _measured(again) == _measured(tuned)
    assert _definitions(again) == _definitions(tuned)
    straight = SearchDB()
    counts = import_file(straight, path)
    assert not counts["did not lower"] and not counts["identities no kernel carries"], counts
    # The file records both precision lanes; the tune above ran in one.
    assert {row for row in _measured(straight) if row[-1] == ""} == _measured(tuned)
    tuned.close()
