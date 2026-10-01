"""Tests for ``emmy eval golden`` — the release audit of one canonical golden against its serving
configuration — and ``eval prior`` over an exported dataset."""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import replace
from pathlib import Path

from emmy.compiler.pipeline.search.golden import GoldenFile, Measurements, write_trace_inventories


def test_eval_golden_requires_exact_file_and_serving_config(run_cli, tmp_path):
    rc, stdout, stderr = run_cli("eval", "golden")
    assert rc == 2
    assert "--golden" in stdout + stderr

    golden = tmp_path / "given.json"
    configured = tmp_path / "configured.json"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={configured}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
    )
    rc, stdout, stderr = run_cli("eval", "golden", "--golden", str(golden), "--serving-config", str(config))
    assert rc == 2
    assert "serving config names" in stdout + stderr


def test_serving_config_derives_standard_and_fast_math_realizations(tmp_path):
    from emmy.serving.release import load_serving_config

    golden = tmp_path / "golden.json"
    config = tmp_path / "release.env"
    config.write_text(
        f"SERVE_MODEL=org/model\nSERVE_GPU=NVIDIA-Test\nSERVE_GOLDEN_FILE={golden}\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=72\nSERVE_DECODE_BUCKET=8\nSERVE_PREFILL_CAPACITY=64\n"
        'SERVE_WARM_SHAPES="32:64:96:fm"\n'
    )

    serving = load_serving_config(config)

    got = {(dict(row.bindings).get("num_tokens"), row.pins) for row in serving.realizations}
    assert got == {
        *((width, (("FAST_MATH", False),)) for width in (None, 1, 8, 64)),
        *((width, (("FAST_MATH", True),)) for width in (None, 1, 32, 64)),
    }


def _release_golden(path: Path, twins: dict[str, tuple[int, list[tuple[str, dict, bool]]]]) -> None:
    """A release golden for ``twins``: each twin a relu of its own size, traced into the file under its name, with a
    measured row per ``(template, bindings, fast math)`` the config reaches it at."""
    from emmy.commands.trace import trace_inline_code
    from emmy.compiler.context import Context

    graphs = {twin: trace_inline_code(f"torch.relu(torch.randn({size}))")["graph"] for twin, (size, _rows) in twins.items()}
    templates = {
        twin: [{"name": name, "bindings": bindings, "pins": {"FAST_MATH": fast}} for name, bindings, fast in rows]
        for twin, (_size, rows) in twins.items()
    }
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    write_trace_inventories(graphs, path, model="org/model", ctx=ctx, realizations=templates)
    document = GoldenFile.load(path)
    measured = Measurements(emmy_us=1.0, reference_us=1.0, reference_backend="torch")
    replace(document, rows=[replace(row, knobs={}, measurements=measured) for row in document.rows]).dump(
        path, overwrite=True, repository=True
    )


def _scoped(scope) -> set[tuple]:
    """What the gate's compile sees as its golden evidence: each row's sizes and regime."""
    return {(tuple(sorted(doc.kernel(row.kernel).bindings.items())), tuple(sorted(row.pins.items()))) for doc in scope for row in doc.rows}


def test_eval_golden_audits_file_scoped_static_release(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import emmy.commands.eval as eval_cmd
    import emmy.serving.twins as twins
    from emmy import config as emmy_config
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import Pipeline
    from emmy.compiler.pipeline.search import golden as golden_mod

    golden = tmp_path / "golden.json"
    _release_golden(golden, {"pre1": (8, [("m1", {"num_tokens": 1}, False)])})
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_STATIC_ONLY=1\nSERVE_MAX_NUM_BATCHED_TOKENS=1\nSERVE_DECODE_BUCKET=1\n"
        "SERVE_PREFILL_CAPACITY=1\nSERVE_PREFILL_BUCKET=0\nSERVE_M1_TIER=1\nSERVE_CAPTURE_SIZES=[1]\n"
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))
    captured = {"compiles": []}
    twin = object()

    def fake_capture(source, **kwargs):
        captured["capture"] = (source, kwargs)
        return {"pre1": twin}

    def fake_run(self, graph, *, ctx=None, **_kwargs):
        # What the gate's compile sees: the lane's rows as the only golden scope, strict on, the
        # live card's context.
        captured["compiles"].append((graph, ctx, emmy_config.strict_evidence(), _scoped(golden_mod.repository.SCOPE)))
        return graph

    monkeypatch.setattr(twins, "capture_twin_graphs", fake_capture)
    monkeypatch.setattr(Pipeline, "run", fake_run)

    eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))

    static = {"decode_bucket": 1, "prefill_bucket": 0, "symbolic": False, "static_only": True, "expert_slices": 1}
    assert captured["capture"] == ("org/model", static)
    assert captured["compiles"] == [(twin, ctx, True, {((("num_tokens", 1),), (("FAST_MATH", False),))})]
    assert golden_mod.repository.SCOPE is None and not emmy_config.strict_evidence()


def test_eval_golden_rejects_a_missing_config_realization(monkeypatch, tmp_path):
    from types import SimpleNamespace

    import pytest

    import emmy.commands.eval as eval_cmd
    from emmy.compiler.context import Context

    golden = tmp_path / "golden.json"
    _release_golden(golden, {"pre8": (8, [("m8.fm", {"num_tokens": 8}, True)])})  # the row is in the other lane
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_MAX_NUM_BATCHED_TOKENS=32\nSERVE_DECODE_BUCKET=8\n"
        "SERVE_PREFILL_CAPACITY=32\nSERVE_PREFILL_BUCKET=0\n"
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))

    with pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))
    assert exc.value.code == 1


def test_eval_golden_fails_when_a_twin_is_not_decided_by_the_golden_rows(monkeypatch, tmp_path, caplog):
    """The serving-matrix compile runs under strict evidence: a twin with a fork no golden row
    decides is a gate failure naming the twin and the kernel, however many others deploy."""
    import logging
    from types import SimpleNamespace

    import pytest

    import emmy.commands.eval as eval_cmd
    import emmy.serving.twins as twins
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import Pipeline
    from emmy.compiler.pipeline.search.policy.greedy import EvidenceError

    golden = tmp_path / "golden.json"
    _release_golden(golden, {"pre1": (8, [("m1", {"num_tokens": 1}, False)])})
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_STATIC_ONLY=1\nSERVE_MAX_NUM_BATCHED_TOKENS=1\nSERVE_DECODE_BUCKET=1\n"
        "SERVE_PREFILL_CAPACITY=1\nSERVE_PREFILL_BUCKET=0\nSERVE_M1_TIER=1\nSERVE_CAPTURE_SIZES=[1]\n"
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))
    undecided = object()
    monkeypatch.setattr(twins, "capture_twin_graphs", lambda source, **kwargs: {"pre1": object(), "post1": undecided})

    def fake_run(self, graph, *, ctx=None, **_kwargs):
        if graph is undecided:
            raise EvidenceError("strict evidence: kernel 'k_linear' (node 'n') has no measured evidence")
        return graph

    monkeypatch.setattr(Pipeline, "run", fake_run)

    with caplog.at_level(logging.ERROR), pytest.raises(SystemExit) as exc:
        eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))
    assert exc.value.code == 1
    assert any("post1" in r.message and "k_linear" in r.message for r in caplog.records)


def test_eval_golden_compiles_a_static_twin_only_in_the_lanes_that_warm_its_width(monkeypatch, tmp_path):
    """A warm shape names its lane (``64:::fm``), so a served process in a lane compiles the
    static twins of that lane's widths and nothing wider or narrower; the audit asks the same of
    each lane's rows — a symbolic twin in every lane, a static twin where its width is warmed."""
    from types import SimpleNamespace

    import emmy.commands.eval as eval_cmd
    import emmy.serving.twins as twins
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline import Pipeline

    golden = tmp_path / "golden.json"
    _release_golden(
        golden,
        {
            "pre8": (8, [("m8", {"num_tokens": 8}, False)]),
            "pre64": (64, [("m64.fm", {"num_tokens": 64}, True)]),
            "pre-sym": (16, [("dynamic", {}, False), ("dynamic.fm", {}, True)]),
        },
    )
    config = tmp_path / "release.env"
    config.write_text(
        f'SERVE_MODEL=org/model\nSERVE_GPU="NVIDIA GeForce RTX 4090"\nSERVE_GOLDEN_FILE={golden}\n'
        "SERVE_MAX_NUM_BATCHED_TOKENS=64\nSERVE_DECODE_BUCKET=8\nSERVE_PREFILL_CAPACITY=64\nSERVE_PREFILL_BUCKET=0\n"
        'SERVE_M1_TIER=0\nSERVE_WARM_SHAPES="64:::fm"\n'
    )
    ctx = Context.from_target((8, 9), gpu_name="NVIDIA GeForce RTX 4090")
    monkeypatch.setattr(Context, "probe", staticmethod(lambda: ctx))
    graphs = {"pre8": object(), "pre64": object(), "pre-sym": object()}
    monkeypatch.setattr(twins, "capture_twin_graphs", lambda source, **kwargs: dict(graphs))
    compiled = []

    def fake_run(self, graph, *, ctx=None, **_kwargs):
        from emmy import config as emmy_config

        compiled.append((next(name for name, g in graphs.items() if g is graph), emmy_config.knob_raw("FAST_MATH")))
        return graph

    monkeypatch.setattr(Pipeline, "run", fake_run)

    eval_cmd.handle_eval_golden(SimpleNamespace(golden=str(golden), serving_config=str(config)))

    assert sorted(compiled) == [("pre-sym", "False"), ("pre-sym", "True"), ("pre64", "True"), ("pre8", "False")]


def test_eval_prior_golden_ranks_an_exported_datasets_golden_pools(tmp_path, caplog):
    """``eval prior`` reads the golden pools of a dataset ``emmy db export`` wrote — the positional argument names the
    directory — and its header names the dataset and the golden files its rows came from. The deploy-faithful check
    runs over the same pools, re-lowering each kernel from the definition the dataset carries."""
    from emmy.commands.eval import register_eval_command
    from emmy.compiler.pipeline.search.db.export import export_dataset
    from tests.compiler.pipeline.search.helpers import tuned_db

    db = tuned_db(None, ("matmul/f16-mma-m128n128k128-f32.json",), source="golden:case")
    export_dataset(db, source="test", pool_sample=0, seed=0).dump(tmp_path / "dataset")
    parser = argparse.ArgumentParser()
    register_eval_command(parser.add_subparsers())
    dataset, out = str(tmp_path / "dataset"), str(tmp_path / "r.json")
    args = parser.parse_args(["eval", "prior", dataset, "--json", out])
    with caplog.at_level(logging.INFO):
        args.func(args)
    header = json.loads((tmp_path / "r.json").read_text())["header"]
    assert (header["dataset"], header["source"], header["sources"]) == ("golden", dataset, {"golden:case": 1})
    assert (header["groups"], header["positives"], header["skipped"]) == (1, 1, 0)
    assert "Golden reproduction" in caplog.text and "k_matmul_" in caplog.text
