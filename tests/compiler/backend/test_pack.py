"""Standalone packs resolve storage and publish only complete bundles."""

import json

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
        save_executable(root, {"p": plan}, bindings={"p": {}}, key={})
    with pytest.raises(ValueError, match="size mismatch"):
        save_executable(root, {"p": plan}, bindings={"p": {"w": b"x"}}, key={})
    assert not root.exists()
    save_executable(root, {"p": plan}, bindings={"p": {"w": bytes(8)}}, key={})
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["standalone"] == 1
    assert (root / manifest["bindings"]["p"]["w"]).read_bytes() == bytes(8)
    assert (root / "cubin/ab.cubin").read_bytes() == b"compiled"
    with pytest.raises(FileExistsError):
        save_executable(root, {"p": plan}, bindings={"p": {"w": bytes(8)}}, key={})


def test_executable_does_not_publish_failed_export(tmp_path, monkeypatch):
    from emmy.compiler.backend import pack

    def fail(*args, **kwargs):
        raise RuntimeError("compile failed")

    monkeypatch.setattr(pack, "save_pack", fail)
    root = tmp_path / "artifact"
    with pytest.raises(RuntimeError, match="compile failed"):
        save_executable(root, {}, bindings={}, key={})
    assert list(tmp_path.iterdir()) == []
