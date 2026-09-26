from __future__ import annotations

from contextlib import nullcontext
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from emmy.compiler.pipeline.search.golden import Config, GoldenFile, Target
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


def test_regeneration_matches_typed_compute_without_a_provenance_name() -> None:
    case = helpers.load_case(helpers.CASES_DIR / "pointwise/relu-vectorized-interleaved-sm89.yaml")
    kernel = case.document.loops[0]
    renamed = deepcopy(kernel)
    compute = next(node for node in renamed["nodes"] if node["op"] == "loop")
    compute["attrs"]["name"] = "renamed_kernel"
    entry = Config(program=0, target=Target(loop=0), realizations=[])
    fresh = GoldenFile(compute_cap=(8, 9), programs=[], configs=[entry], loops=[renamed])
    assert helpers._matching_entry(fresh, entry, kernel) == entry

    compute["outputs"][0][1] = "f16"
    with pytest.raises(helpers.CaseError, match="no kernel"):
        helpers._matching_entry(fresh, entry, kernel)
