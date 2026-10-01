from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emmy.compiler.pipeline.search.golden import GoldenFile
from tests.compiler.realization import helpers


def test_built_loads_the_lowered_program_through_nvcc(monkeypatch) -> None:
    source = SimpleNamespace(copy=lambda: "source-copy")
    case = SimpleNamespace(program=lambda: source)
    lowered = object()
    seen = []

    monkeypatch.setattr(helpers, "lowered", lambda case, ctx: (lowered, []))
    monkeypatch.setattr("emmy.compiler.context.Context.probe", staticmethod(lambda: "live"))
    monkeypatch.setattr("emmy.compiler.backend.cuda.program.CompiledProgram.build", lambda graph, feed: seen.append((graph, feed)))
    monkeypatch.setattr("emmy.compiler.backend.gpu_lock.gpu_lock", nullcontext)
    monkeypatch.setattr(helpers, "seeded_inputs", lambda program: {"x": program})

    assert helpers.built(case) is lowered
    assert seen == [(lowered, {"x": source})]


def test_bench_command_replays_the_named_realization_through_the_golden_flags() -> None:
    """The case's own record is the replay: no hand pin rides beside it, so the route, the input
    regime and the schedule row all reach the compile through the one golden mechanism."""
    case = SimpleNamespace(row=SimpleNamespace(name="k_example"), path=Path("case.json"))

    command = helpers.bench_command(case, Path("result.json"))

    assert command[command.index("--golden") + 1] == "case.json"
    assert command[command.index("--realization") + 1] == "k_example"
    assert "--ab" not in command and "--bench" in command


def test_reference_and_lowered_weights_share_the_source_before_transpose() -> None:
    case = helpers.load_case(helpers.CASES_DIR / "matmul" / "f16-mma-splitk-unit-output.json")
    sources = {}
    feed = helpers.seeded_inputs(case.program(), sources=sources)
    reference = helpers.seeded_inputs(case.document.reference_program(case.target), sources=sources)

    np.testing.assert_array_equal(feed["linear_6_wt"], reference["p_mlp_down_proj_weight"].T)


def test_a_regenerated_case_keeps_its_note(tmp_path) -> None:
    """The note is a field of the file, not a comment the dump drops: regeneration and a dump carry it."""
    case = helpers.load_case(helpers.CASES_DIR / "reduce/singleton-softmax-unsqueeze.json")
    assert case.document.note
    path = tmp_path / "case.json"
    helpers.regenerate(case.document).dump(path)
    assert GoldenFile.load(path).note == case.document.note
