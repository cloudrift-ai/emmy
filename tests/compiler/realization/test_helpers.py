from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from emmy.compiler.pipeline.search.golden import GoldenFile
from tests.compiler.realization import helpers


def test_built_assembles_every_kernel_for_the_target_its_text_names(monkeypatch) -> None:
    from emmy.compiler.ir.cuda import CudaOp

    graph = SimpleNamespace(
        nodes={
            "plain": SimpleNamespace(op=CudaOp(kernel_source=".target sm_90\n", kernel_name="k_plain")),
            "tma": SimpleNamespace(op=CudaOp(kernel_source=".target sm_90a\n", kernel_name="k_tma")),
            "input": SimpleNamespace(op=object()),
        }
    )
    case = SimpleNamespace(compute_cap=(9, 0), context=lambda: "declared")
    seen = []

    monkeypatch.setattr(helpers, "lowered", lambda case, ctx: (graph, []) if ctx == "declared" else None)
    monkeypatch.setattr("emmy.compiler.backend.cuda.ptxas.compile_to_cubin", lambda source, name: seen.append((name, source)))

    assert helpers.built(case) is graph
    assert seen == [("k_plain", ".target sm_90\n"), ("k_tma", ".target sm_90a\n")]


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


def test_regenerate_refuses_a_row_that_would_lose_a_key() -> None:
    """A row's scoped key that names no site of the fresh kernel would be dropped by the restamp; a corpus row has no
    measurement whose demotion would show it, so the regen refuses instead of writing a case that is "current" with
    a schedule nobody authored."""
    from dataclasses import replace

    import pytest

    document = GoldenFile.load(helpers.CASES_DIR / "fused/gdn-recurrence-state-kv.json")
    (row,) = document.rows
    nested = max((key for key in row.knobs if key.startswith("TILE@")), key=len).partition("@")[2]
    unmapped = {key.replace(nested, f"{nested}9"): value for key, value in row.knobs.items()}
    with pytest.raises(helpers.CaseError, match="names no site"):
        helpers.regenerate(replace(document, rows=[replace(row, knobs=unmapped)]))
    assert helpers.regenerate(document) == document


def test_a_regenerated_case_keeps_its_note(tmp_path) -> None:
    """The note is a field of the file, not a comment the dump drops: regeneration and a dump carry it."""
    case = helpers.load_case(helpers.CASES_DIR / "reduce/singleton-softmax-unsqueeze.json")
    assert case.document.note
    path = tmp_path / "case.json"
    helpers.regenerate(case.document).dump(path)
    assert GoldenFile.load(path).note == case.document.note
