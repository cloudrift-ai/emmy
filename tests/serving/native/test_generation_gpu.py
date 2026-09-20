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


# Per-position absolute budgets and per-prompt RMS comparison, fixed before the
# held-out cases below. FP32 uses the same FP16-rounded weights. The qualification
# report retains the failed exploratory per-position comparison contract.
FULL_MODEL_ERROR = 2e-2
REFERENCE_ERROR_FACTOR = 2.0
CHECKPOINT_CASES = (
    ("france", "The capital of France is", None, 15, False),
    ("arithmetic", "2 + 2 =", None, 15, True),
    ("explanation", "Explain why the sky appears blue in one sentence.", None, 16, False),
    ("code", 'def square(x):\n    """Return x squared."""\n    return', None, 16, True),
    ("german", "Translate to English: Der kleine Hund wartet vor der Tür.", None, 16, True),
    ("json", 'Return JSON with keys "name" and "count" for three apples:', None, 16, False),
    ("long", "The observatory records stars, planets, and comets every clear night. ", 127, 16, True),
    ("boundary", "A library stores books on history, science, art, and travel. ", 240, 16, True),
    ("heldout_spanish", "Translate to English: La estación está cerca del río.", None, 24, True),
    ("heldout_math", "A box has 7 red balls and 5 blue balls. How many balls are in the box?", None, 24, False),
    ("heldout_code", "def is_even(number):\n    return", None, 24, True),
    ("heldout_context", "The train crosses a bridge and stops beside a quiet village. ", 128, 24, True),
)


def _logit_errors(actual, expected):
    def probabilities(values):
        values = values.astype(np.float64)
        exponentials = np.exp(values - values.max())
        return exponentials / exponentials.sum()

    return {
        "max_absolute_error": float(np.max(np.abs(actual - expected))),
        "relative_l2_error": float(np.linalg.norm(actual - expected) / np.linalg.norm(expected)),
        "probability_tv": float(np.abs(probabilities(actual) - probabilities(expected)).sum() / 2),
    }


@pytest.mark.parametrize("name,text,prompt_length,decode_steps,capture", CHECKPOINT_CASES, ids=[case[0] for case in CHECKPOINT_CASES])
def test_checkpoint_logits_and_completions(request, tmp_path, monkeypatch, name, text, prompt_length, decode_steps, capture):
    import copy

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
    precise = copy.deepcopy(model).float()
    prompt = tokenizer.encode(text)
    if prompt_length is not None:
        prompt = (prompt * ((prompt_length + len(prompt) - 1) // len(prompt)))[:prompt_length]
    with gpu_lock():
        model.cuda()
        precise.cuda()
        # Check full-checkpoint dispatcher identity on the original two cases. Other
        # cases independently check the model, including the longer cache histories.
        reference_program = _python_reference(artifact) if name in ("france", "arithmetic") else None
        monkeypatch.setenv("PATH", "/nonexistent")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

        async def check():
            worker = NativeWorker(executable=executable)
            path, logits_path = tmp_path / "prompt.bin", tmp_path / "logits.bin"
            measurements = []
            try:
                await worker.run_job({"op": "load_generation", "root": artifact}, wall_timeout_s=60)
                np.asarray(prompt, np.int64).tofile(path)
                await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                if reference_program is not None:
                    _reset_python(reference_program, prompt)
                prefix, next_token, past, precise_past = [], None, None, None
                for position in range(len(prompt) + decode_steps):
                    prefix.append(prompt[position] if position < len(prompt) else next_token)
                    result = await worker.run_job(
                        {"op": "generation_step", "capture": capture, "logits": str(logits_path)}, wall_timeout_s=30
                    )
                    with torch.no_grad(), _reference_precision(True):
                        ids = torch.tensor([[prefix[-1]]], device="cuda")
                        reference = model(ids, past_key_values=past, use_cache=True)
                        past = reference.past_key_values
                        expected = reference.logits[0, -1].float().cpu().numpy()
                        accurate = precise(ids, past_key_values=precise_past, use_cache=True)
                        precise_past = accurate.past_key_values
                        fp32 = accurate.logits[0, -1].cpu().numpy()
                    actual = np.fromfile(logits_path, np.float16).astype(np.float32)
                    if reference_program is not None:
                        np.testing.assert_array_equal(actual, _python_step(reference_program, position).astype(np.float32))
                    assert np.isfinite(actual).all() and np.isfinite(expected).all() and np.isfinite(fp32).all()
                    native_error, reference_error = _logit_errors(actual, fp32), _logit_errors(expected, fp32)
                    measurements.append(
                        {
                            "case": name,
                            "position": position,
                            "capture": capture,
                            "prompt_length": len(prompt),
                            "native_fp32": native_error,
                            "hf_fp32": reference_error,
                            "native_fp16": _logit_errors(actual, expected),
                            "python_native_equal": True if reference_program is not None else None,
                            "native_token": int(actual.argmax()),
                            "reference_token": int(expected.argmax()),
                            "fp32_token": int(fp32.argmax()),
                            "reference_margin": float(np.sort(expected)[-1] - np.sort(expected)[-2]),
                            "fp32_margin": float(np.sort(fp32)[-1] - np.sort(fp32)[-2]),
                            "strict_close": bool(np.allclose(actual, expected, rtol=1e-3, atol=1e-3)),
                            "full_model_close": bool(np.allclose(actual, expected, rtol=2e-2, atol=2e-2)),
                        }
                    )
                    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
                    if result["token"] is not None:
                        next_token = result["token"]
                        assert next_token == int(actual.argmax())
                # Experimental acceptance budgets, not a theorem about FP16. A per-prompt
                # RMS comparison avoids unstable ratios at nearly exact reference positions;
                # the absolute per-position limits still prohibit hiding an outlier in a mean.
                for row in measurements:
                    for metric in ("relative_l2_error", "probability_tv"):
                        assert row["native_fp32"][metric] <= FULL_MODEL_ERROR, row
                    assert row["native_token"] in (row["reference_token"], row["fp32_token"]), row
                for metric in ("relative_l2_error", "probability_tv"):
                    native_rms = np.sqrt(np.mean([row["native_fp32"][metric] ** 2 for row in measurements]))
                    reference_rms = np.sqrt(np.mean([row["hf_fp32"][metric] ** 2 for row in measurements]))
                    assert native_rms <= max(REFERENCE_ERROR_FACTOR * reference_rms, np.finfo(np.float16).eps), (
                        name,
                        metric,
                        native_rms,
                        reference_rms,
                    )
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


def test_attention_reads_only_the_written_cache_prefix(tmp_path):
    from emmy.compiler.backend.pack import save_executable
    from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan, KernelSpec, LaunchSpec
    from emmy.compiler.dim import Dim
    from emmy.compiler.dtype import F16, I64
    from emmy.serving.native.kernels import SOURCE
    from emmy.serving.native.prepare import MAX_CONTEXT

    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    rng = np.random.default_rng(19)
    query = rng.normal(size=(4, 128)).astype(np.float16)
    keys = rng.normal(size=(MAX_CONTEXT, 2, 128)).astype(np.float16)
    values = rng.normal(size=keys.shape).astype(np.float16)
    source = (
        "#define HIDDEN 32\n#define HEADS 4\n#define KV_HEADS 2\n#define HEAD_DIM 128\n"
        "#define VOCAB 32\n#define SCALE 0.08838834764831845f\n" + SOURCE
    )
    data = {"q": query, "k": keys, "v": values, "position": np.array([0], np.int64)}
    buffers = [BufferSpec(n, tuple(Dim(x) for x in a.shape), I64 if n == "position" else F16, "input") for n, a in data.items()]
    buffers.append(BufferSpec("attention", (Dim(4), Dim(128)), F16, "output"))
    plan = ExecutionPlan(
        "cuda",
        list(data),
        ["attention"],
        buffers,
        {},
        {},
        [
            LaunchSpec(
                "attention",
                "native_attention",
                (*data, "attention"),
                ((4,), (1,), (1,)),
                ((128,), (1,), (1,)),
                MAX_CONTEXT * 4,
                (),
                writes=("attention",),
            )
        ],
        {"native_attention": KernelSpec(source=source)},
    )
    with gpu_lock():
        root = save_executable(
            tmp_path / "pack", {"attention": plan}, bindings={"attention": {n: a.tobytes() for n, a in data.items()}}, key={}
        )

        async def check():
            worker = NativeWorker(executable=executable)
            try:
                await worker.run_job({"op": "load", "root": str(root), "program": "attention"}, wall_timeout_s=30)
                # Ascending and descending lengths catch stale future data after request reset.
                for count in (1, 127, 128, 129, MAX_CONTEXT - 1, MAX_CONTEXT, 2):
                    data["k"], data["v"] = keys.copy(), values.copy()
                    data["k"][count:] = np.nan
                    data["v"][count:] = np.nan
                    data["position"][0] = count - 1
                    for name, array in data.items():
                        array.tofile(tmp_path / name)
                    await worker.run_job({"op": "bind", "inputs": {n: str(tmp_path / n) for n in data}}, wall_timeout_s=30)
                    await worker.run_job(
                        {"op": "run", "warmup": 0, "iterations": 1, "capture": True, "outputs": {"attention": str(tmp_path / "result")}},
                        wall_timeout_s=30,
                    )
                    # Independent float64 reductions, with the eager FP16 storage boundaries.
                    k = keys[:count].repeat(2, axis=1).astype(np.float64)
                    v = values[:count].repeat(2, axis=1).astype(np.float64)
                    scores = np.einsum("hd,thd->ht", query.astype(np.float64), k).astype(np.float16)
                    scores = (scores.astype(np.float32) * np.float32(128**-0.5)).astype(np.float16).astype(np.float64)
                    probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
                    probabilities = (probabilities / probabilities.sum(axis=-1, keepdims=True)).astype(np.float16)
                    expected = np.einsum("ht,thd->hd", probabilities.astype(np.float64), v).astype(np.float16)
                    actual = np.fromfile(tmp_path / "result", np.float16).reshape(query.shape)
                    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
            finally:
                await worker.aclose()

        asyncio.run(check())
