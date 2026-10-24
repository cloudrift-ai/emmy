"""Packs: a standalone pack resolves storage and publishes only complete bundles, and a pack entry is valid while
the golden rows its compile read are."""

import json
import os
from dataclasses import replace

import numpy as np
import pytest

from emmy.compiler.backend.pack import save_executable
from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan
from emmy.compiler.dim import Dim
from emmy.compiler.dtype import F32


def test_executable_bundles_constants_and_cubins(tmp_path, monkeypatch):
    from emmy.compiler.backend import pack

    cache = tmp_path / "cache"
    cache.mkdir()
    (cache / "ab.cubin").write_bytes(b"compiled")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(cache))
    plan = ExecutionPlan("cuda", [], [], [BufferSpec("w", (Dim(2),), F32, "constant")], {}, {}, [], {})

    def store(root, plans, **kwargs):
        root.mkdir()
        (root / "plan.json").write_text(json.dumps({"kernels": {"k": {"binary_key": "ab"}}}))
        (root / "manifest.json").write_text(json.dumps({"format": 1, "programs": {"p": "plan.json"}}))

    monkeypatch.setattr(pack, "save_pack", store)
    root = tmp_path / "artifact"
    with pytest.raises(ValueError, match="missing constant"):
        save_executable(root, {"p": plan}, bindings={"p": {}}, key={}, reads={"p": frozenset()})
    with pytest.raises(ValueError, match="size mismatch"):
        save_executable(root, {"p": plan}, bindings={"p": {"w": b"x"}}, key={}, reads={"p": frozenset()})
    assert not root.exists()
    save_executable(root, {"p": plan}, bindings={"p": {"w": bytes(8)}}, key={}, reads={"p": frozenset()})
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["standalone"] == 1
    assert (root / manifest["bindings"]["p"]["w"]).read_bytes() == bytes(8)
    assert (root / "cubin/ab.cubin").read_bytes() == b"compiled"
    with pytest.raises(FileExistsError):
        save_executable(root, {"p": plan}, bindings={"p": {"w": bytes(8)}}, key={}, reads={"p": frozenset()})


def test_executable_does_not_publish_failed_export(tmp_path, monkeypatch):
    from emmy.compiler.backend import pack

    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(pack, "save_pack", fail)
    root = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="compile failed"):
        save_executable(root, {}, bindings={}, key={}, reads={})
    assert list(tmp_path.iterdir()) == []


def test_executable_keeps_the_pages_of_a_buffer_the_pager_holds_once(tmp_path, monkeypatch):
    from emmy.compiler.backend import pack

    def store(root, plans, **kwargs):
        root.mkdir()
        (root / "plan.json").write_text(json.dumps({"kernels": {}}))
        (root / "manifest.json").write_text(json.dumps({"format": 1, "programs": dict.fromkeys(plans, "plan.json")}))

    monkeypatch.setattr(pack, "save_pack", store)
    experts = BufferSpec("experts", (Dim(4), Dim(2)), F32, "input")
    plan = ExecutionPlan("cuda", [], [], [experts], {}, {}, [], {}, paged={"experts": (0, 1, 2)})
    data, bindings = np.arange(8, dtype=np.float32).reshape(4, 2), {"decode": {}, "prefill": {}}
    calls = []

    def experts():
        # Called once, as its file is written, so a caller holds one buffer's pages at a time.
        calls.append(sorted(p.name for p in tmp_path.rglob("pages/*.bin")))
        return data

    root = save_executable(
        tmp_path / "paged",
        {"decode": plan, "prefill": plan},
        bindings=bindings,
        key={},
        reads=dict.fromkeys(bindings, frozenset()),
        pages={"experts": experts},
    )
    assert calls == [[]]
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["pages"] == {"experts": "pages/0.bin"} and manifest["bindings"] == {"decode": {}, "prefill": {}}
    assert (root / "pages/0.bin").read_bytes() == data.tobytes()
    root = save_executable(tmp_path / "resident", {"decode": plan}, bindings={"decode": {}}, key={}, reads={"decode": frozenset()})
    assert "pages" not in json.loads((root / "manifest.json").read_text()) and not (root / "pages").exists()


def test_a_golden_edit_recompiles_only_the_programs_that_read_its_rows(tmp_path, monkeypatch):
    """A pack entry is valid while the golden rows its compile read are: programs compiled from a two-file scope, one
    row of one file re-measured, and the next load hands back no plan for the program that read it and for the other
    twin of its wrapper, which settled its weights' layouts with it, while the rest load. A scope that moved without
    moving any program's rows re-keys the pack, so the load after it compares nothing."""
    from emmy.compiler.backend import pack
    from emmy.compiler.backend.plan import KernelSpec, WeightSpec
    from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
    from emmy.compiler.pipeline.search import golden
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.compiler.pipeline.search.policy.greedy import evidence_reads
    from tests.compiler.realization import helpers as corpus

    scan_case, mean_case = (
        corpus.load_case(corpus.CASES_DIR / name) for name in ("scan/f32-cumsum-row.json", "reduce/combine-mean-serial.json")
    )
    scan, mean = corpus._measured(scan_case.document), corpus._measured(mean_case.document)

    def scope(*documents):
        monkeypatch.setenv(
            "EMMY_GOLDEN_FILE", os.pathsep.join(str(document.dump(tmp_path / name, overwrite=True)) for name, document in documents)
        )

    def plan(weight):
        weights = {"w": WeightSpec(source_path=weight)}
        return ExecutionPlan("cuda", [], [], [], {}, {}, [], {"k": KernelSpec(binary_key="ab")}, weights=weights)

    scope(("scan.json", scan), ("mean.json", mean))
    (tmp_path / "cubin").mkdir()
    (tmp_path / "cubin/ab.cubin").write_bytes(b"compiled")
    monkeypatch.setenv("EMMY_CUBIN_CACHE", str(tmp_path / "cubin"))
    monkeypatch.setattr(pack, "_environment", lambda: {"backend": "cuda"})
    monkeypatch.setattr(golden, "live_gpu_key", lambda: ("NVIDIA GeForce RTX 5090", scan_case.compute_cap))
    with pinned_knobs(scan_case.regime):
        compiled = []
        for case in (scan_case, mean_case):
            with evidence_reads() as read:
                Pipeline.build(CUDA_PASSES).run(case.program(), ctx=case.context(), db=None)
            assert case.target.exact_identity in read
            compiled.append(frozenset(read))
        # Two twins of one layer's wrapper, one reading the scan's rows and one the mean's, and another layer's twin.
        programs = {"L0.pre.decode": plan("layers.0.w"), "L0.pre.sym": plan("layers.0.w"), "L1.pre.decode": plan("layers.1.w")}
        reads = {"L0.pre.decode": compiled[0], "L0.pre.sym": compiled[1], "L1.pre.decode": compiled[1]}
        root = pack.save_pack(tmp_path / "pack", programs, key={}, reads=reads)
        assert all(entry.plan is not None and entry.reads == reads[name] for name, entry in pack.load_pack(root, key={}).items())

        [row] = scan.rows
        slower = replace(row, measurements=replace(row.measurements, emmy_us=2 * row.measurements.emmy_us))
        scope(("scan-remeasured.json", replace(scan, rows=[slower])), ("mean.json", mean))
        stale = {name for name, entry in pack.load_pack(root, key={}).items() if entry.plan is None}
        assert stale == {"L0.pre.decode", "L0.pre.sym"}

        scope(("scan-renamed.json", scan), ("mean.json", mean))
        assert all(entry.plan is not None for entry in pack.load_pack(root, key={}).values())
        monkeypatch.setattr(pack, "_rows", lambda reads: pytest.fail("a pack re-keyed to its scope compares no rows"))
        assert all(entry.plan is not None for entry in pack.load_pack(root, key={}).values())
