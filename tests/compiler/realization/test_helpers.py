from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from tests.compiler.realization import helpers


def test_built_loads_the_lowered_program_through_nvcc(monkeypatch) -> None:
    source = SimpleNamespace(copy=lambda: "source-copy")
    case = SimpleNamespace(pinned={}, record=SimpleNamespace(target_program=source))
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
    case = SimpleNamespace(
        pinned={"FAST_MATH": False, "PLACE@map.1/inner": "cut", "WORK": "w1x1"},
        record=SimpleNamespace(name="k_example"),
        path=Path("case.yaml"),
    )

    command = helpers.bench_command(case, Path("result.json"))

    assert command[command.index("--golden") + 1] == "case.yaml"
    assert command[command.index("--realization") + 1] == "k_example"
    assert "--ab" not in command and "--bench" in command


def test_reference_and_lowered_weights_share_the_source_before_transpose() -> None:
    case = helpers.load_case(helpers.CASES_DIR / "matmul" / "f16-mma-splitk-unit-output.yaml")
    sources = {}
    feed = helpers.seeded_inputs(case.record.target_program, sources=sources)
    reference = helpers.seeded_inputs(case.record.reference_program, sources=sources)

    np.testing.assert_array_equal(feed["linear_6_wt"], reference["p_mlp_down_proj_weight"].T)
