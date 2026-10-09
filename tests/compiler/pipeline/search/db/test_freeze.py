"""The measurement freeze — the admission filter, the golden file per card a DB instance freezes to, and the
round trip: a freeze imported by ``emmy db import`` is the rows it was written from.

The DB a freeze is written from is what a tune of the realization corpus leaves behind (``helpers.tuned_db``),
so every kind of kernel the compiler mints is on the way: a fused kernel, a twisted attention kernel, cut
pieces formed as kernels of their own, and the pieces of an attention split, which the freeze reaches through the
decision that minted them. These tests never touch a GPU."""

from __future__ import annotations

import dataclasses

import pytest

from emmy.compiler.pipeline.search.dataset import REGIME_PINS, regime_of
from emmy.compiler.pipeline.search.db import SearchDB, knobs_json
from emmy.compiler.pipeline.search.db.freeze import freeze_documents, freeze_reason, write_freeze
from emmy.compiler.pipeline.search.golden import GoldenFile
from emmy.compiler.pipeline.search.golden.evidence import file_source, import_file
from tests.compiler.pipeline.search.helpers import F16_MATMUL_STAMPS, SQUARE_512_STAMPS, impossible_staged_row, tuned_db
from tests.compiler.pipeline.search.helpers import perf_row as _row

# A small fp32 matmul's stamps — what ``features.stamps`` computes from such a kernel — plausible at a few hundred µs.
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


#: Its schedule row.
_ROW = {"TILE": "f2x2", "WORK": "t16x16"}


# ---------------------------------------------------------------------------
# freeze_reason — the admission filter
# ---------------------------------------------------------------------------


def test_reason_keeps_a_row_of_either_precision_regime(monkeypatch) -> None:
    """The fast-math flag alone decides a row's regime, spelled once and not read off this shell: any other flag
    a row was compiled with, and whatever ``EMMY_NVCC_FLAGS`` holds when the freeze runs, leave it where its
    context put it."""
    monkeypatch.setenv("EMMY_NVCC_FLAGS", "-lineinfo")
    assert freeze_reason(_row("k", us=500.0, knobs=_ROW), _STAMPS) is None
    assert freeze_reason(_row("k", us=500.0, knobs=_ROW, flags="--use_fast_math"), _STAMPS) is None
    assert freeze_reason(_row("k", us=500.0, knobs=_ROW, flags="-lineinfo --use_fast_math"), _STAMPS) is None
    assert REGIME_PINS[""] == {"FAST_MATH": False}
    assert REGIME_PINS["--use_fast_math"] == {"FAST_MATH": True}
    assert regime_of("-lineinfo --use_fast_math") == "--use_fast_math" and regime_of("-lineinfo") == ""


def test_freeze_preserves_cold_and_hot_rows_of_the_same_kernel(tmp_path):
    from emmy.compiler.pipeline.search.golden import import_file

    path = tmp_path / "source.db"
    source = tuned_db(path, CASES[:1], us=500.0)
    hot_rows = list(source.iter_perf_rows())
    for row in hot_rows:
        source.record_perf_row(dataclasses.replace(row, cold_cache=True, stats=dataclasses.replace(row.stats, median=700.0)))
    source.close()
    directory = tmp_path / "freeze"
    write_freeze(path, directory)
    restored = SearchDB()
    for file in directory.glob("*.json"):
        import_file(restored, file, source=f"freeze:{file.name}")
    rows = list(restored.iter_perf_rows())
    assert len(rows) == 2 * len(hot_rows)
    assert {r.stats.median for r in rows if r.cold_cache} == {700.0}
    assert {r.stats.median for r in rows if not r.cold_cache} == {500.0}


def test_reason_drops_a_failed_bench() -> None:
    # A fail's median is the watchdog sentinel, not a measurement; the tune DB keeps the failure.
    assert freeze_reason(_row("k", us=9.17, knobs=_ROW, status="bench_fail"), _STAMPS) == "bench_fail: not a measurement"


def test_reason_drops_a_card_the_registry_does_not_know() -> None:
    # Its H_* features cannot be derived, so no reader could featurize it.
    row = dataclasses.replace(_row("k", us=500.0, knobs=_ROW), gpu="Mystery GPU")
    assert freeze_reason(row, _STAMPS).startswith("unknown card")


def test_reason_drops_a_non_deployable_regime() -> None:
    assert freeze_reason(_row("k", us=500.0, knobs=_ROW, opt=1), _STAMPS) == "non-deployable regime (H_opt=1)"


def test_reason_drops_a_row_whose_kernel_no_longer_lowers() -> None:
    # A row's shape is computed from its kernel's definition; a definition the compiler stopped taking back has none.
    assert freeze_reason(_row("k", us=500.0, knobs=_ROW), None) == "stale kernel: its definition no longer lowers"


def test_reason_drops_implausible_value() -> None:
    # The shared f16 mlp_down extents at 9.17 µs imply ~6500 TFLOP/s at the default hint of its
    # symbolic axis — and an honest 13 TFLOP/s when the row says it was benched at one token.
    assert "implausible value" in freeze_reason(_row("k", us=9.17), F16_MATMUL_STAMPS)
    assert freeze_reason(_row("k", us=9.17, bindings={"m": 1}), F16_MATMUL_STAMPS) is None


def test_reason_drops_impossible_kernel() -> None:
    # The square.512 residue: over-cap cp.async slab -> legal-looking latency, invalid kernel.
    assert "impossible kernel" in freeze_reason(_row("k", us=2.02, knobs=impossible_staged_row()), SQUARE_512_STAMPS)


# ---------------------------------------------------------------------------
# the freeze and its round trip
# ---------------------------------------------------------------------------


def _measured(db: SearchDB) -> set[tuple]:
    """Every row as a freeze carries it: the kernel, its bindings, its schedule row, its median and its regime."""

    return {
        (r.kernel, knobs_json(r.bindings), knobs_json(r.knobs), r.stats.median, r.flags) for r in db.iter_perf_rows() if r.status == "ok"
    }


def _definitions(db: SearchDB) -> dict[str, bool]:
    """Each measured kernel's identity — the key the import computed from its stored body — and whether it is formed."""
    measured = {r.kernel for r in db.iter_perf_rows()}
    return {k.exact_identity: k.formed for k in db.iter_kernels() if k.exact_identity in measured}


@pytest.fixture(scope="module")
def tuned(tmp_path_factory):
    """The tune DB the freeze tests are written from, and its path."""
    path = tmp_path_factory.mktemp("tune") / "autotune.db"
    # A time every one of these kernels could run in: the plausibility gate reads each kernel's own shape.
    db = tuned_db(path, CASES, us=500.0)
    yield db, path
    db.close()


def test_a_freeze_is_a_golden_file_per_card_that_re_lowers_to_the_rows_it_was_written_from(tuned, tmp_path) -> None:
    tuned, tuned_path = tuned
    """One document per card, valid as a golden file: the kernels the rows measured, every decision that reaches one
    of them (the attention split's pieces, formed from no loop op, through the split on their parent) and a row per
    measurement. Imported into a fresh instance, the rows come back with the same kernels, schedule rows, medians
    and regimes: a freeze names its kernels by a ref of its own, and the import computes each one's identity from
    the body the file stores, which is the identity the tune filed it under."""

    documents, dropped = freeze_documents(tuned)
    assert dropped == {} and set(documents) == {"nvidia_geforce_rtx_5090_sm120.json", "nvidia_tesla_v100_sxm2_16gb_sm70.json"}
    for document in documents.values():
        document.check()
    definitions = _definitions(tuned)
    unformed = {identity for identity, formed in definitions.items() if not formed}
    assert unformed, "the attention split mints pieces no loop op forms"
    rtx = documents["nvidia_geforce_rtx_5090_sm120.json"]
    kernels = {kernel.exact_identity: kernel for kernel in tuned.iter_kernels()}
    [split] = [decision_row for decision_row in tuned.iter_routing() if set(decision_row.children) & unformed]
    ref = {identity: ref for ref, identity in rtx.identities().items()}
    assert {split.parent, *split.children} <= set(ref) and kernels[split.parent].formed
    assert dataclasses.replace(split, parent=ref[split.parent], children=tuple(ref[child] for child in split.children)) in rtx.routing
    assert all(row.measured for row in rtx.rows)

    digests = write_freeze(tuned_path, tmp_path / "freeze")
    assert set(digests) == set(documents)
    again = SearchDB()
    for name in sorted(digests):
        import_file(again, tmp_path / "freeze" / name, file_source("freeze", tmp_path / "freeze" / name))
    assert _measured(again) == _measured(tuned)
    assert _definitions(again) == definitions
    assert set(again.perf_sources()) == {file_source("freeze", tmp_path / "freeze" / name) for name in digests}
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
    assert sorted(row.pins["FAST_MATH"] for row in document.rows) == [False, True]
    assert {tuple(row.pins) for row in document.rows} == {("FAST_MATH",)}
    write_freeze(path, tmp_path / "freeze")
    db.close()
    again = SearchDB()
    import_file(again, (frozen := next((tmp_path / "freeze").glob("*.json"))), file_source("freeze", frozen))
    lanes = sorted((r.flags, r.stats.median) for r in again.iter_perf_rows())
    assert lanes == [("", row.stats.median), ("--use_fast_math", row.stats.median)]


def test_a_kernel_benched_at_two_sizes_freezes_as_two_rows_of_one_kernel(tmp_path) -> None:
    """A row's sizes are the row's ``bindings``, the sizes its symbolic dims were benched at: the same symbolic kernel
    benched at two sizes is one kernel with two rows, and each row comes back at the size it was benched at."""
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
    assert sorted(row.bindings["seq_len"] for row in document.rows) == [128, 512]
    [kernel] = document.kernels
    assert symbolic_vars(kernel.loop_ir) == {"seq_len"}
    write_freeze(path, tmp_path / "freeze")
    db.close()
    again = SearchDB()
    import_file(again, (frozen := next((tmp_path / "freeze").glob("*.json"))), file_source("freeze", frozen))
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
        import_file(SearchDB(), pointer, file_source("freeze", pointer))


@pytest.mark.xdist_group("golden_import_rtx5090")
def test_the_rtx_5090_hardware_goldens_rows_round_trip_through_a_freeze(tmp_path) -> None:
    """A card's recorded rows — attention and softmax kernels, split winners' routing rows, cut pieces — imported as a
    tune of that card would leave them, freeze to one file that imports to the same rows and kernels. The same file
    imported straight from the repository files the same rows too: a golden file is a source ``emmy db import``
    accepts."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import import_rows
    from emmy.compiler.pipeline.search.golden.repository import _RECORDS_DIR
    from tests.compiler.pipeline.search.helpers import GPU_5090

    path = _RECORDS_DIR / "rtx5090_sm120.json"
    document = GoldenFile.load(path)
    tuned_path = tmp_path / "autotune.db"
    tuned = SearchDB(tuned_path)
    standard = [row for row in document.rows if row.pins == {"FAST_MATH": False}]
    ctx = Context.from_target((12, 0), gpu_name=GPU_5090, compile_flags="")
    assert import_rows(tuned, ctx, document, standard, source="measured") >= 30
    documents, dropped = freeze_documents(tuned)
    assert dropped == {} and list(documents) == ["nvidia_geforce_rtx_5090_sm120.json"]
    [name] = write_freeze(tuned_path, tmp_path / "freeze")
    again = SearchDB()
    import_file(again, tmp_path / "freeze" / name, file_source("freeze", tmp_path / "freeze" / name))
    assert _measured(again) == _measured(tuned)
    assert _definitions(again) == _definitions(tuned)
    straight = SearchDB()
    import_file(straight, path, file_source("freeze", path))
    # The file records both precision lanes; the tune above ran in one.
    assert {row for row in _measured(straight) if row[-1] == ""} == _measured(tuned)
    tuned.close()
