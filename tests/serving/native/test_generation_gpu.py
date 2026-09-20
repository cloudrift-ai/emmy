"""Independent cached model logits, device sampling, capture, and request reset."""

import asyncio
import json
import shutil

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.native import NativeWorker
from emmy.serving.native.prepare import export_model
from tests.compiler.helpers import requires_cuda
from tests.serving.helpers import qwen3_model

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]


def _python_reference(root):
    from pathlib import Path

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.plan import plan_from_dict

    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text())
    plan = plan_from_dict(json.loads((root / manifest["programs"]["decode"]).read_text()))
    buffers = {buffer.name: buffer for buffer in plan.buffers}
    data = {
        name: np.fromfile(root / path, dtype=buffers[name].dtype.np).reshape(buffers[name].resolve_shape({}))
        for name, path in manifest["bindings"]["decode"].items()
    }
    data.update({name: np.zeros(buffers[name].resolve_shape({}), dtype=buffers[name].dtype.np) for name in plan.inputs})
    return CompiledProgram.build_from_plan(plan, data, cubin_dir=root / "cubin")


def _reset_python(program, prompt):
    ids = np.zeros(program.arrays["prompt"].shape, dtype=np.int64)
    ids[: len(prompt)] = prompt
    program.arrays["prompt"].set(ids)
    program.arrays["prompt_length"].set(np.array([len(prompt)], np.int64))


def _python_step(program, position):
    program.arrays["position"].set(np.array([position], np.int64))
    program.run_once()
    return program.outputs()["logits"].reshape(-1)


def test_cached_qwen3_logits_and_generation(tmp_path, monkeypatch):
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    model = qwen3_model(2).half()
    model.config._attn_implementation = "eager"
    with gpu_lock():
        root = export_model(model, tmp_path / "pack", context_length=8)
        reference_program = _python_reference(root)
        monkeypatch.setenv("PATH", "/nonexistent")
        model.cuda()
        # Strict reference products use FP32 accumulation, including on Torch versions with split-K defaults.
        from emmy.compiler.backend.cuda._bench_worker import _reference_precision

        async def check():
            worker = NativeWorker(executable=executable)
            path = tmp_path / "prompt.bin"
            logits_path = tmp_path / "logits.bin"
            try:
                await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=30)
                for capture, prompt in ((False, [1, 2, 3]), (None, [8, 9]), (True, [3]), (True, [4, 5, 6, 7])):
                    np.asarray(prompt, np.int64).tofile(path)
                    await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                    _reset_python(reference_program, prompt)
                    prefix = []
                    next_token = None
                    selected = []
                    for position in range(8):
                        prefix.append(prompt[position] if position < len(prompt) else next_token)
                        result = await worker.run_job(
                            {
                                "op": "generation_step",
                                "capture": (position >= len(prompt) if capture is None else capture),
                                "logits": str(logits_path),
                            },
                            wall_timeout_s=30,
                        )
                        with torch.no_grad(), _reference_precision(True):
                            expected = model(torch.tensor([prefix], device="cuda")).logits[0, -1].float().cpu().numpy()
                        actual = np.fromfile(logits_path, np.float16).astype(np.float32)
                        np.testing.assert_array_equal(actual, _python_step(reference_program, position).astype(np.float32))
                        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
                        if position >= len(prompt) - 1:
                            next_token = result["token"]
                            assert next_token == int(expected.argmax())
                            selected.append(next_token)
                        else:
                            assert result["token"] is None
                    output = tmp_path / "generated.bin"
                    await worker.run_job(
                        {
                            "op": "generate",
                            "prompt": str(path),
                            "max_new_tokens": 8 - len(prompt),
                            "capture": bool(capture),
                            "output": str(output),
                        },
                        wall_timeout_s=30,
                    )
                    assert np.fromfile(output, np.int64).tolist() == selected[:-1]
                manifest_path = root / "manifest.json"
                manifest = json.loads(manifest_path.read_text())
                manifest["key"]["generation"]["eos_ids"] = [selected[0]]
                manifest_path.write_text(json.dumps(manifest))
                await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=30)
                for budget in (0, 3):
                    await worker.run_job(
                        {"op": "generate", "prompt": str(path), "max_new_tokens": budget, "capture": True, "output": str(output)},
                        wall_timeout_s=30,
                    )
                    assert np.fromfile(output, np.int64).tolist() == ([selected[0]] if budget else [])
                with pytest.raises(RuntimeError, match="stopped|context"):
                    await worker.run_job({"op": "generation_step", "capture": True}, wall_timeout_s=30)
            finally:
                await worker.aclose()

        asyncio.run(check())


def test_checkpoint_logits_and_completions(request, tmp_path, monkeypatch):
    checkpoint = request.config.getoption("--native-checkpoint")
    artifact = request.config.getoption("--native-artifact")
    if not checkpoint or not artifact:
        pytest.skip("pass local --native-checkpoint and --native-artifact to qualify a checkpoint")
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.fail("checkpoint qualification requires the native worker")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emmy.compiler.backend.cuda._bench_worker import _reference_precision

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(checkpoint, dtype=torch.float16, attn_implementation="eager", local_files_only=True).eval()
    with gpu_lock():
        model.cuda()
        reference_program = _python_reference(artifact)
        monkeypatch.setenv("PATH", "/nonexistent")

        async def check():
            worker = NativeWorker(executable=executable)
            path, logits_path = tmp_path / "prompt.bin", tmp_path / "logits.bin"
            measurements = []
            try:
                await worker.run_job({"op": "load_generation", "root": artifact}, wall_timeout_s=60)
                for capture, text in ((False, "The capital of France is"), (True, "2 + 2 =")):
                    prompt = tokenizer.encode(text)
                    np.asarray(prompt, np.int64).tofile(path)
                    await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                    _reset_python(reference_program, prompt)
                    prefix, next_token, past = [], None, None
                    for position in range(len(prompt) + 15):
                        prefix.append(prompt[position] if position < len(prompt) else next_token)
                        result = await worker.run_job(
                            {"op": "generation_step", "capture": capture, "logits": str(logits_path)}, wall_timeout_s=30
                        )
                        with torch.no_grad(), _reference_precision(True):
                            reference = model(torch.tensor([[prefix[-1]]], device="cuda"), past_key_values=past, use_cache=True)
                            past = reference.past_key_values
                            expected = reference.logits[0, -1].float().cpu().numpy()
                            full_prefix = model(torch.tensor([prefix], device="cuda"), use_cache=False).logits[0, -1].float().cpu().numpy()
                        actual = np.fromfile(logits_path, np.float16).astype(np.float32)
                        np.testing.assert_array_equal(actual, _python_step(reference_program, position).astype(np.float32))
                        measurements.append(
                            {
                                "prompt": text,
                                "position": position,
                                "capture": capture,
                                "max_absolute_error": float(np.max(np.abs(actual - expected))),
                                "relative_l2_error": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
                                "python_native_equal": True,
                                "hf_prefix_max_absolute_error": float(np.max(np.abs(full_prefix - expected))),
                                "hf_prefix_relative_l2_error": float(np.linalg.norm(full_prefix - expected) / np.linalg.norm(expected)),
                                "hf_prefix_close": bool(np.allclose(full_prefix, expected, rtol=2e-2, atol=2e-2)),
                                "hf_prefix_argmax_match": int(full_prefix.argmax()) == int(expected.argmax()),
                                "argmax_match": int(actual.argmax()) == int(expected.argmax()),
                                "native_token": int(actual.argmax()),
                                "reference_token": int(expected.argmax()),
                                "reference_margin": float(np.sort(expected)[-1] - np.sort(expected)[-2]),
                                "native_margin": float(np.sort(actual)[-1] - np.sort(actual)[-2]),
                                "strict_close": bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-3)),
                                "full_model_close": bool(np.allclose(actual, expected, rtol=2e-2, atol=2e-2)),
                            }
                        )
                        if result["token"] is not None:
                            next_token = result["token"]
                            (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
                            assert next_token == int(actual.argmax())
                (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
                # Same FP16 full-model criterion as the existing generation oracle. Tiny-model
                # qualification above retains 1e-3; preserve its stricter checkpoint result in the evidence.
                assert all(row["full_model_close"] and row["argmax_match"] for row in measurements), measurements
            finally:
                await worker.aclose()

        asyncio.run(check())


def test_rotary_preserves_half_precision_operation_boundaries(tmp_path):
    from emmy.compiler.backend.pack import save_executable
    from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16, I64
    from emmy.serving.native.kernels import SOURCE

    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    rng = np.random.default_rng(71)
    q = (rng.normal(size=(4, 128)) * 10).astype(np.float16)
    k = (rng.normal(size=(2, 128)) * 10).astype(np.float16)
    v = rng.normal(size=(2, 128)).astype(np.float16)
    angles = np.tile(rng.normal(size=64), 2)
    cosine, sine = np.cos(angles).astype(np.float16), np.sin(angles).astype(np.float16)
    data = {"q": q, "k": k, "v": v, "cosine": cosine, "sine": sine, "position": np.array([0], np.int64)}
    source = (
        "#define HIDDEN 32\n#define HEADS 4\n#define KV_HEADS 2\n#define HEAD_DIM 128\n"
        "#define VOCAB 32\n#define SCALE 0.08838834764831845f\n" + SOURCE
    )
    buffers = [BufferSpec(n, tuple(Dim(x) for x in a.shape), I64 if n == "position" else F16, "input") for n, a in data.items()]
    outputs = {"rotated": q, "keys": k, "values": v}
    buffers += [BufferSpec(n, tuple(Dim(x) for x in a.shape), F16, "output") for n, a in outputs.items()]
    args = tuple(data) + tuple(outputs)
    plan = ExecutionPlan(
        "cuda",
        list(data),
        list(outputs),
        buffers,
        {},
        {},
        [LaunchSpec("rope", "native_rope_cache", args, ((4,), (1,), (1,)), ((128,), (1,), (1,)), 0, ())],
        {"native_rope_cache": KernelSpec(source=source)},
    )
    with gpu_lock():
        root = save_executable(tmp_path / "rope", {"rope": plan}, bindings={"rope": {n: a.tobytes() for n, a in data.items()}}, key={})

        async def check():
            worker = NativeWorker(executable=executable)
            try:
                await worker.run_job({"op": "load", "root": str(root), "program": "rope"}, wall_timeout_s=30)
                await worker.run_job(
                    {"op": "run", "warmup": 0, "iterations": 1, "capture": False, "outputs": {n: str(tmp_path / n) for n in outputs}},
                    wall_timeout_s=30,
                )
            finally:
                await worker.aclose()

        asyncio.run(check())
        for name, values in (("rotated", q), ("keys", k)):
            rotated = np.concatenate((-values[:, 64:], values[:, :64]), axis=-1)
            expected = (values * cosine + rotated * sine).astype(np.float16)
            np.testing.assert_array_equal(np.fromfile(tmp_path / name, np.float16).reshape(values.shape), expected)
        np.testing.assert_array_equal(np.fromfile(tmp_path / "values", np.float16).reshape(v.shape), v)
