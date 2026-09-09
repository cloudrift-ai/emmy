from __future__ import annotations

import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("validate_serve", PROJECT_ROOT / "scripts" / "validate_serve.py")
validate_serve = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(validate_serve)


def test_serving_reference_deploys_the_named_golden_at_the_named_decode_width(monkeypatch):
    monkeypatch.setenv("EMMY_GEN_DECODE_BUCKET", "32")
    args = SimpleNamespace(
        emmy="./venv/bin/emmy",
        model="nvidia/Qwen3-8B-NVFP4",
        max_model_len="4096",
        port="8000",
        gpu_mem_util="0.82",
        max_num_batched_tokens="256",
        golden="sweep.yaml",
        decode_bucket=16,
    )

    cmd, env = validate_serve._serve_invocation(args)

    assert cmd == [
        "./venv/bin/emmy",
        "serve",
        "--generate",
        "nvidia/Qwen3-8B-NVFP4",
        "--max-model-len",
        "4096",
        "--port",
        "8000",
        "--gpu-memory-utilization",
        "0.82",
        "--max-num-batched-tokens",
        "256",
        "--golden",
        "sweep.yaml",
    ]
    assert env["EMMY_GEN_DECODE_BUCKET"] == "16"
    assert os.environ["EMMY_GEN_DECODE_BUCKET"] == "32"


def test_serving_reference_leaves_unspecified_evidence_and_width_alone(monkeypatch):
    monkeypatch.delenv("EMMY_GEN_DECODE_BUCKET", raising=False)
    args = SimpleNamespace(
        emmy="emmy",
        model="org/model",
        max_model_len="1024",
        port="8123",
        gpu_mem_util="0.9",
        max_num_batched_tokens=None,
        golden=None,
        decode_bucket=None,
    )

    cmd, env = validate_serve._serve_invocation(args)

    assert "--golden" not in cmd
    assert "--max-num-batched-tokens" not in cmd
    assert "EMMY_GEN_DECODE_BUCKET" not in env
