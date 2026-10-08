"""Working-golden execution tests."""

import argparse
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from emmy.commands import run as run_mod
from emmy.compiler.pipeline.search.db import RoutingRow
from emmy.compiler.pipeline.search.golden import GoldenFile, Kernel, Row


def _parser():
    parser = argparse.ArgumentParser()
    run_mod.register_run_command(parser.add_subparsers())
    return parser


def test_run_golden_schema_matches_compile():
    """``run`` spells golden selection the way every replaying command does: ``--golden PATH`` is
    the file, ``--realization NAME`` the row inside it — one pair, one meaning, on ``run`` /
    ``compile`` / ``tune`` / ``serve`` alike."""
    args = _parser().parse_args(["run", "--golden", "working.json", "--realization", "linear.layer0", "--gpu-arch", "sm_90"])

    assert args.golden == "working.json"
    assert args.realization == "linear.layer0"
    assert args.gpu_arch == "sm_90"
    for removed in ("all_targets", "repeats", "require_kernel_source", "golden_target", "golden_file"):
        assert not hasattr(args, removed)
    with pytest.raises(SystemExit):
        _parser().parse_args(["run", "--golden-file", "working.json"])


def _args(tmp_path, **updates):
    values = {
        "golden": str(tmp_path / "working.json"),
        "realization": None,
        "input": None,
        "code": None,
        "ir": None,
        "json": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def _hex(tag: str) -> str:
    return hashlib.sha256(tag.encode()).hexdigest()


def _kernel(tag: str, *, piece: bool = False) -> Kernel:
    """A fake kernel the file calls ``k_<tag>``, its identity handed in: nothing here lowers its (empty) wire."""
    kernel = Kernel(loop_ir={"inputs": [], "outputs": [], "nodes": []}, name=f"k_{tag}", formed=True, traced=None if piece else 0)
    return kernel.keyed(_hex(tag))


def _document(rows, routing=()) -> GoldenFile:
    """A golden of fake kernels: ``rows`` as ``(name, kernel tag, pins)``, ``routing`` as ``(parent tag, child tags)``
    — a child is a piece, with no program of its own."""
    children = {child for _, kids in routing for child in kids}
    tags = {tag for _, tag, _ in rows} | {parent for parent, _ in routing} | children
    return GoldenFile(
        compute_cap=(8, 9),
        programs=[{"inputs": [], "outputs": [], "nodes": []}],
        kernels=[_kernel(tag, piece=tag in children) for tag in sorted(tags)],
        routing=[RoutingRow(f"k_{parent}", {"PLACE": "cut"}, tuple(f"k_{child}" for child in kids)) for parent, kids in routing],
        rows=[Row(name=name, kernel=f"k_{tag}", pins=dict(pins)) for name, tag, pins in rows],
    )


#: A golden with no rows: the tests below patch the document a load yields.
_EMPTY = GoldenFile(compute_cap=(8, 9))


def _patch_document(monkeypatch, document: GoldenFile) -> None:
    from emmy.compiler.pipeline.search import golden

    monkeypatch.setattr(golden.GoldenFile, "load", classmethod(lambda _cls, _path, **_: document))


def test_golden_runs_every_distinct_target_in_process(monkeypatch, tmp_path):
    _patch_document(monkeypatch, _document([("linear.layer0", "a", {}), ("linear.layer0", "a", {}), ("linear.layer1", "b", {})]))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.realization for args in calls] == ["linear.layer0", "linear.layer1"]
    assert all(args.golden.endswith("working.json") and args._explicit_realization is False for args in calls)


def test_golden_walk_benches_each_target_once_not_its_pieces(monkeypatch, tmp_path):
    """A piece a decision minted runs with its target: the walk names the target once, by the shortest row name of
    the set, and leaves the rows to the evidence pick."""
    rows = [
        ("k_mean.aaaa", "mean", {}),
        ("k_mean.aaaa.c5cd", "mean_piece", {}),
        ("k_lin.bbbb", "lin", {}),
        ("k_lin.bbbb.8270", "lin_a", {}),
        ("k_lin.bbbb.4d7f", "lin_b", {}),
        ("orphan.cccc.dddd", "orphan", {}),
    ]
    _patch_document(monkeypatch, _document(rows, routing=[("mean", ("mean_piece",)), ("lin", ("lin_a", "lin_b"))]))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.realization for args in calls] == ["k_mean.aaaa", "k_lin.bbbb", "orphan.cccc.dddd"]


def test_golden_walk_names_a_target_by_its_own_row_before_a_shorter_piece_row(monkeypatch, tmp_path):
    """The seed row on the target names the walk's run — ``--record`` writes the whole target's latency onto that name
    — even where a piece's row has a shorter name."""
    rows = [("k_layer_seed_long_name", "layer", {}), ("qk.t32", "layer_piece", {})]
    _patch_document(monkeypatch, _document(rows, routing=[("layer", ("layer_piece",))]))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.realization for args in calls] == ["k_layer_seed_long_name"]


def test_golden_walk_without_seeds_names_a_target_by_its_shortest_row(monkeypatch, tmp_path):
    """A file that dropped its seed rows still benches each target once, through the shortest row name of its set —
    a piece's row where the target itself has none."""
    rows = [
        ("post16.k_a.1111.m16.bbbb", "a_piece_b", {}),
        ("post16.k_a.1111.m16.aaaa", "a_piece_a", {}),
        ("pre1.k_b.2222.m1.dddd", "b", {}),
        ("pre1.k_b.2222.m1.eeee", "b", {}),
    ]
    _patch_document(monkeypatch, _document(rows, routing=[("a", ("a_piece_a", "a_piece_b"))]))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.realization for args in calls] == ["post16.k_a.1111.m16.aaaa", "pre1.k_b.2222.m1.dddd"]


def test_golden_walk_keeps_dotted_names_sizes_and_pin_regimes(monkeypatch, tmp_path):
    """Two kernels are two targets whatever their names share; a kernel at two sizes is two kernels; one kernel's
    rows in two precision regimes run once each."""
    rows = [
        ("k_mean", "m", {}),
        ("k_mean.type_as", "t", {}),
        ("dynamic.m16", "d16", {}),
        ("dynamic.m32", "d32", {}),
        ("exact", "p", {"FAST_MATH": False}),
        ("fast", "p", {"FAST_MATH": True}),
    ]
    _patch_document(monkeypatch, _document(rows))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [args.realization for args in calls] == [name for name, _, _ in rows]


def test_golden_walk_reports_every_target_before_failing(monkeypatch, tmp_path):
    """A target that fails does not hide the targets after it: the walk runs them all and exits 1."""
    _patch_document(monkeypatch, _document([(f"linear.layer{i}", f"k{i}", {}) for i in range(3)]))
    calls = []

    def run_once(args):
        calls.append(args.realization)
        if args.realization == "linear.layer0":
            raise SystemExit(1)
        if args.realization == "linear.layer1":
            raise ValueError("Tile.aux_threads requires a cooperative block_threads")

    monkeypatch.setattr(run_mod, "_handle_run_once", run_once)

    with pytest.raises(SystemExit) as exc:
        run_mod._run_golden_targets(_args(tmp_path))

    assert exc.value.code == 1
    assert calls == ["linear.layer0", "linear.layer1", "linear.layer2"]


def test_naming_one_target_skips_the_multi_target_walk(run_cli):
    """``--realization NAME`` goes straight down the single-run path — the walk is for a bare file."""
    rc, stdout, stderr = run_cli("run", "--realization", "linear.layer0", "--code", "torch.randn(4, 4)")

    assert rc == 2
    assert "mutually exclusive" in stdout + stderr


def test_multi_target_json_uses_one_readable_file_per_target(monkeypatch, tmp_path):
    _patch_document(monkeypatch, _document([("linear/layer0", "k0", {}), ("linear/layer1", "k1", {})]))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)
    output = tmp_path / "results"

    run_mod._run_golden_targets(_args(tmp_path, json=str(output)))

    assert [Path(args.json).name for args in calls] == ["000-linear_layer0.json", "001-linear_layer1.json"]
    assert output.is_dir()


def test_strict_result_requires_backends_capture_and_correctness():
    proof = {
        "status": "pass",
        "reference": "eager",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True, num_launches=1, per_launch=[], e2e_min_ms=None)
    args = SimpleNamespace(bench_backends="eager,tcompile,emmy", ab=None)
    results = {"Eager PyTorch": 20.0, "torch.compile": 15.0, "Emmy": 10.0}

    assert run_mod._strict_benchmark_errors(args, results, bench, True, proof, []) == []
    errors = run_mod._strict_benchmark_errors(args, {"Emmy": 10.0}, bench, True, proof, [])
    assert "torch.compile" in " ".join(errors)


def test_strict_result_requires_every_requested_exact_row():
    args = SimpleNamespace(bench_backends="emmy", ab=["TILE=f2x4"])
    proof = {
        "status": "pass",
        "reference": "eager",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True)

    errors = run_mod._strict_benchmark_errors(args, {"Emmy": 10.0}, bench, True, proof, [])

    assert "expected 1 exact --ab row(s), got 0" in errors


def test_strict_result_accepts_same_input_greedy_only_for_reference_free_loop():
    args = SimpleNamespace(bench_backends="emmy", ab=None)
    proof = {
        "status": "pass",
        "reference": "same-input-greedy",
        "rtol": 1e-3,
        "atol": 1e-3,
        "max_abs_error": 0.0,
        "mean_abs_error": 0.0,
        "max_rel_error": 0.0,
    }
    bench = SimpleNamespace(captured=True, num_launches=1, per_launch=[], e2e_min_ms=None)
    results = {"Emmy": 10.0}

    runnable_errors = run_mod._strict_benchmark_errors(args, results, bench, True, proof, [])
    assert "same-input-greedy" not in " ".join(runnable_errors)
    assert "strict eager correctness" in " ".join(runnable_errors)

    missing_errors = run_mod._strict_benchmark_errors(
        args,
        results,
        bench,
        True,
        proof,
        [],
        frontend_runnable=False,
        same_input_reference=False,
    )
    assert "same-input greedy reference is unavailable" in missing_errors

    assert (
        run_mod._strict_benchmark_errors(
            args,
            results,
            bench,
            True,
            proof,
            [],
            frontend_runnable=False,
            same_input_reference=True,
        )
        == []
    )


def test_golden_walk_resolves_one_name_in_two_regimes_to_each_regimes_rows(monkeypatch, tmp_path):
    """A row name both precision regimes share runs once per regime, and each run selects only its regime's rows:
    the compile publishes that regime and ``--record`` attributes the timing to one row."""
    from emmy.commands import compile as compile_mod

    document = _document([("matmul", "p", {"FAST_MATH": False}), ("matmul", "p", {"FAST_MATH": True})])
    _patch_document(monkeypatch, document)
    monkeypatch.setattr(GoldenFile, "executable", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(GoldenFile, "reference_program", lambda *_args, **_kwargs: None)
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert [(args.realization, args._golden_pins) for args in calls] == [("matmul", {"FAST_MATH": False}), ("matmul", {"FAST_MATH": True})]
    for args in calls:
        vars(args).update(input=None, code=None, ir=None, dynamic=None)
        compile_mod.resolve_golden_arg(args)
        assert [row.pins for row in args._golden_rows] == [args._golden_pins]


def test_golden_document_is_parsed_once_for_every_target(monkeypatch, tmp_path):
    """A whole-model inventory must not be re-read and re-validated per target."""
    from emmy.compiler.pipeline.search import golden

    loads = []
    document = _document([(name, name, {}) for name in ("a", "b", "c")])
    monkeypatch.setattr(golden.GoldenFile, "load", classmethod(lambda _cls, _path, **_: loads.append(_path) or document))
    calls = []
    monkeypatch.setattr(run_mod, "_handle_run_once", calls.append)

    run_mod._run_golden_targets(_args(tmp_path))

    assert len(loads) == 1
    assert [args._golden_document for args in calls] == [document] * 3


def test_resolve_golden_arg_prefers_the_document_the_caller_loaded(monkeypatch, tmp_path):
    """With a document supplied, resolution must not touch the file at all."""
    from emmy.commands import compile as compile_mod
    from emmy.compiler.pipeline.search import golden

    def _explode(_cls, _path, **_kwargs):
        raise AssertionError("load_golden_file must not be called when a document is supplied")

    monkeypatch.setattr(golden.GoldenFile, "load", classmethod(_explode))

    args = SimpleNamespace(
        realization="missing",
        golden=str(tmp_path / "absent.json"),
        _golden_document=_EMPTY,
        input=None,
        code=None,
        ir=None,
        dynamic=None,
    )
    # No record matches "missing", so resolution exits 2 — reaching that exit proves it
    # resolved against the supplied document instead of reading the absent file.
    with pytest.raises(SystemExit) as excinfo:
        compile_mod.resolve_golden_arg(args)
    assert excinfo.value.code == 2


def test_record_latency_ignores_a_child_receipt_of_the_same_target():
    """A recorded route's child receipts are rows of the same target, so ``--record`` benches them
    beside the realization it was asked for. Their timing is one kernel of the program and their
    knobs are that kernel's schedule, so attributing the realization's latency to them stores the
    wrong number under knobs that select no row — the write is then refused and the measurement
    lost. Only a row named for the realization itself can carry it."""
    seen = {}
    receipt = SimpleNamespace(
        status="ok",
        bench=object(),
        sample=SimpleNamespace(name="linear.layer0.abcdef123456", knobs={"WORK": "w1x1"}, pins={"FAST_MATH": True}),
    )
    args = SimpleNamespace(golden="working.json", realization="linear.layer0")
    with mock.patch.object(run_mod, "_bench_total_us", side_effect=AssertionError("a receipt's timing is not the program's")):
        with mock.patch("emmy.compiler.pipeline.search.golden.record_latency", lambda *a, **kw: seen.update(kw)):
            run_mod._record_golden_latency(args, {"Emmy": 12.5, "Eager PyTorch": 30.0}, [receipt])

    assert seen["emmy_us"] == 12.5
    assert seen["knobs"] is None and seen["pins"] is None


def test_record_latency_of_two_measured_schedules_lands_on_the_faster():
    """Two measured schedules of one target in one regime: the evidence pick deploys the faster, so its row takes
    the target's latency."""
    seen = {}

    def row(knobs, us):
        return SimpleNamespace(status="ok", bench=us, flags=[], sample=SimpleNamespace(name="mm", knobs=knobs, pins={"FAST_MATH": False}))

    args = SimpleNamespace(golden="working.json", realization="mm")
    with mock.patch.object(run_mod, "_bench_total_us", lambda us: (us, "e2e")):
        with mock.patch("emmy.compiler.pipeline.search.golden.record_latency", lambda *a, **kw: seen.update(kw)):
            run_mod._record_golden_latency(args, {"torch.compile": 9.0}, [row({"WORK": "w4x4"}, 12.0), row({"WORK": "w4x2"}, 10.0)])

    assert seen["emmy_us"] == 10.0 and seen["knobs"] == {"WORK": "w4x2"} and seen["tcompile_us"] == 9.0


def test_record_greedy_is_a_golden_bench_flag(run_cli):
    """``--record-greedy`` writes the greedy pick's kernel set back into the benched golden, so
    like ``--record`` it is refused without the file and the bench that measure it."""
    args = _parser().parse_args(["run", "--golden", "working.json", "--realization", "linear.layer0", "--bench", "--record-greedy"])
    assert args.record_greedy is True

    rc, stdout, stderr = run_cli("run", "--golden", "working.json", "--realization", "linear.layer0", "--record-greedy")

    assert rc == 2
    assert "--record-greedy requires --golden PATH and --bench" in stdout + stderr


def test_recording_into_another_cards_file_is_refused(caplog):
    """A recorded row is evidence only on the card its file names, so ``--record-greedy`` into a file
    seeded for another card is refused before anything is benched, not written where no replay reads it."""
    args = _parser().parse_args(["run", "--golden", "working.json", "--realization", "linear.layer0", "--bench", "--record-greedy"])
    seeded = SimpleNamespace(gpu_name="NVIDIA Tesla V100 SXM2 16GB", compute_cap=(7, 0))
    live = SimpleNamespace(gpu_name="Tesla V100-SXM3-32GB", compute_capability=(7, 0))
    with (
        mock.patch.object(GoldenFile, "load", return_value=seeded),
        mock.patch("emmy.compiler.context.Context.probe", return_value=live),
        mock.patch.object(run_mod, "_handle_run_once", side_effect=AssertionError("benched a row it cannot record")),
        pytest.raises(SystemExit) as exc,
    ):
        run_mod.handle_run(args)
    assert exc.value.code == 2
    assert "working golden targets NVIDIA Tesla V100 SXM2 16GB" in caplog.text


def test_pinned_rows_bench_when_the_greedy_returned_no_outputs():
    """A greedy that cannot be timed must not also block the pinned alternative that escapes it.

    The reference a pinned row is checked against comes from the greedy. When the greedy
    bench_fails there is none, and the question is whether any OTHER reference exists. An exact
    Loop target has no Torch twin, so the greedy is the only reference obtainable and the row would
    be unfalsifiable -- refuse. Where a twin exists the rows bench, flagged unverified, because the
    targets whose greedy hangs are exactly the ones a pinned row exists for.

    The predicate is the twin's existence, not ``same_input_greedy``: those differ by a
    ``strict_correctness`` conjunct, and keying on it would let a non-strict twinless target bench
    with no reference at all.
    """
    fail = "greedy run/bench failed: HungKernelError"

    # A reference is present: nothing to refuse, whatever else is true.
    assert run_mod.pinned_reference_refusal(ab_ref=("in", "out"), torch_twin=False, greedy_fail=None) is None
    assert run_mod.pinned_reference_refusal(ab_ref=("in", "out"), torch_twin=True, greedy_fail=fail) is None

    # No reference and no twin to fall back on: refused, and the reason still names the failure.
    refusal = run_mod.pinned_reference_refusal(ab_ref=None, torch_twin=False, greedy_fail=fail)
    assert refusal is not None
    assert run_mod._NO_GREEDY_REF in refusal
    assert fail in refusal
    assert run_mod.pinned_reference_refusal(ab_ref=None, torch_twin=False, greedy_fail=None) is not None

    # No reference but a twin exists: bench anyway. This is the case the gate used to refuse.
    assert run_mod.pinned_reference_refusal(ab_ref=None, torch_twin=True, greedy_fail=fail) is None
    assert run_mod.pinned_reference_refusal(ab_ref=None, torch_twin=True, greedy_fail=None) is None


def test_record_refuses_a_row_benched_without_a_reference(tmp_path):
    """An unverified row must never become golden evidence -- a miscompiling tile runs at a
    perfectly plausible latency, so a recorded number for an unchecked kernel is worse than none."""
    sample = SimpleNamespace(name="pinned.row", knobs={"WORK": "w2x2"}, pins={}, dynamic=None, shape=None)
    gb = SimpleNamespace(
        status="ok",
        bench=SimpleNamespace(min_ms=1.0, time_ms=1.0, per_launch=[]),
        sample=sample,
        flags=[f"{run_mod.UNVERIFIED_ROW}: greedy run/bench failed"],
    )
    args = SimpleNamespace(golden=str(tmp_path / "g.json"), realization="pinned.row")
    with pytest.raises(SystemExit) as exc:
        run_mod._record_golden_latency(args, {"Emmy": 1000.0}, [gb])
    assert exc.value.code == 2


def test_an_env_pin_that_did_not_realize_is_flagged_like_an_ab_pin(monkeypatch):
    """A pin published through EMMY_KNOBS gates the greedy compile, and nothing used to check it.

    An --ab row has always been gated: benching a row whose pin did not take would measure the planner's
    own pick under the pin's name. The same pin set in the environment was unchecked, so a hand-run sweep
    whose pins did nothing reported the planner's pick under the experiment's name -- not a wrong number
    but an unfalsifiable one, indistinguishable from a flat result.

    The last case is the one measured in practice: a scheduled TILE pinned against a kernel that reached
    the unscheduled per-cell tier, where the pin is simply absent from the realized knobs.
    """
    realized = [{"WORK": "w2x2", "TILE": "mma_m8n8k4_f16_f32/f4x4/k8", "STAGE": "d2/smem"}]

    # Nothing pinned: nothing to refuse.
    assert run_mod.env_pin_refusal(realized) is None

    # A pin the graph realized is silent, exactly as the --ab gate is.
    monkeypatch.setenv("EMMY_WORK", "w2x2")
    assert run_mod.env_pin_refusal(realized) is None

    # A pin the graph contradicts names both sides, in the --ab gate's own wording.
    monkeypatch.setenv("EMMY_WORK", "w4x8")
    flag = run_mod.env_pin_refusal(realized)
    assert flag is not None
    assert "w4x8" in flag and "w2x2" in flag
    monkeypatch.delenv("EMMY_WORK")

    # A scheduled pin against the unscheduled per-cell tier: absent, not contradicted.
    monkeypatch.setenv("EMMY_TILE", "mma_m8n8k4_f16_f32/f4x4/k8")
    flag = run_mod.env_pin_refusal([{"LOOPIFY": "0"}])
    assert flag is not None
    assert "TILE" in flag
    monkeypatch.delenv("EMMY_TILE")

    # Placement is consumed before CUDA emission, so its realized side comes from the
    # greedy resolution trace rather than a kernel knob stamp.
    monkeypatch.setenv("EMMY_PLACE@MAP.1/INNER.2/MAP", "cut")
    flag = run_mod.env_pin_refusal(realized, [{"PLACE": "fuse"}])
    assert flag is not None
    assert "PLACE@map.1/inner.2/map=cut" in flag
    assert run_mod.env_pin_refusal(realized, [{"PLACE@map.1/inner.2/map": "cut"}]) is None


def test_a_greedy_pick_whose_env_pin_did_not_realize_is_never_recorded(monkeypatch):
    """``--record-greedy`` under ``EMMY_KNOBS`` records the pin — so a pin that did not take records the
    planner's own pick under the pin's lane. Measured while hand-recording a fast-math row beside a
    standard one that split the same way: the greedy replayed the standard receipt, the run warned, and
    the recording still filed an f32-accumulate schedule under ``FAST_MATH: true``."""
    realized = [{"WORK": "w4x2", "TILE": "mma_m16n8k16_f16_f32/f2x4/k2", "STAGE": "d2/smem-tma"}]

    assert run_mod.greedy_record_refusal(realized, accuracy_error=None) is None
    assert "accuracy" in run_mod.greedy_record_refusal(realized, accuracy_error="max error 0.3")

    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f16/f4x8/k4")
    refusal = run_mod.greedy_record_refusal(realized, accuracy_error=None)
    assert refusal is not None
    assert "f16_f16/f4x8/k4" in refusal and "f16_f32/f2x4/k2" in refusal


def test_a_reference_that_disagrees_with_itself_is_reported_unusable_not_per_row():
    """A pinned row is checked against the greedy output. A row that realized the greedy's OWN config
    computes that output, so if it is flagged as disagreeing the reference does not reproduce -- and
    then no row's comparison against it distinguishes a wrong answer from a right one.

    Measured on Qwen3.8-27B-W4A16 layer 0: every row of one target was flagged, including the --ab row
    realizing the greedy's own w2x2 f4x4/k8 d2/smem at 1722.4 us against the greedy's 1723.4 us, at
    rel err 15.859. A flag that fires on the reference itself is not evidence about any row, and a
    flag that fires on everything is one readers learn to skip -- which is how a real deviation
    (a sibling slot measured rel err 2.177 on a genuinely miscompiling tile) gets waved through.
    """
    ref = [{"WORK": "w2x2", "TILE": "mma_m8n8k4_f16_f32/f4x4/k8", "STAGE": "d2/smem"}]
    other = [{"WORK": "w4x8", "TILE": "mma_m8n8k4_f16_f32/f4x4", "STAGE": "d2/smem"}]

    # The reference reproduces: a row that differs from it is genuinely suspect, and says so.
    verdict = "wrong-answer: rel err 2.177 vs greedy output"
    assert run_mod.resolve_reference_disagreement([(other, verdict), (ref, None)], ref) == [verdict, None]

    # The reference disagrees with a row that IS the reference: one statement, and nothing per row.
    resolved = run_mod.resolve_reference_disagreement(
        [(ref, "wrong-answer: rel err 15.992 vs greedy output"), (other, "wrong-answer: rel err 15.859 vs greedy output")],
        ref,
    )
    assert resolved == [run_mod.REFERENCE_SELF_DISAGREES, None]
    assert "unusable" in run_mod.REFERENCE_SELF_DISAGREES
    # It is stated once, not once per row.
    assert resolved.count(run_mod.REFERENCE_SELF_DISAGREES) == 1

    # Order does not matter: the witness may be any row, and the others still lose their verdicts.
    resolved = run_mod.resolve_reference_disagreement(
        [(other, "wrong-answer: rel err 15.859 vs greedy output"), (ref, "wrong-answer: rel err 15.992 vs greedy output")],
        ref,
    )
    assert resolved == [None, run_mod.REFERENCE_SELF_DISAGREES]

    # A reference row that AGREES is no witness -- the others keep their verdicts.
    assert run_mod.resolve_reference_disagreement([(ref, None), (other, verdict)], ref) == [None, verdict]

    # Without the greedy's realized knobs there is nothing to compare, so verdicts pass through.
    assert run_mod.resolve_reference_disagreement([(other, verdict)], None) == [verdict]
    assert run_mod.resolve_reference_disagreement([(other, verdict)], []) == [verdict]


def test_the_correctness_oracle_runs_torch_gemms_at_full_precision_and_restores_the_defaults():
    """The reference a kernel is judged against must not carry torch's speed-for-accuracy reductions: at
    K=15360 the default FP16 GEMM misses ``--strict``'s own tolerance against an FP64 product on one element
    in ten, so ``--strict`` rejected an FP32-accumulate kernel for eager's error. The timed eager forward runs
    outside the context and keeps the defaults."""
    import torch

    matmul = torch.backends.cuda.matmul
    before = (matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction)
    with run_mod.correctness_oracle():
        assert not matmul.allow_fp16_reduced_precision_reduction
        assert not matmul.allow_bf16_reduced_precision_reduction
    assert (matmul.allow_fp16_reduced_precision_reduction, matmul.allow_bf16_reduced_precision_reduction) == before
