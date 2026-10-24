"""Independent cached model logits, host sampling, capture, and request reset."""

import asyncio
import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import torch

from emmy.compiler.backend.gpu_lock import gpu_lock
from emmy.compiler.backend.native import NativeWorker
from emmy.serving.native.prepare import export_model
from tests.compiler.helpers import requires_cuda
from tests.serving.helpers import qwen3_5_model, qwen3_5_moe_model, qwen3_model

pytestmark = [requires_cuda, pytest.mark.xdist_group("cuda")]


def _python_reference(root):
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
    ids = np.zeros(program.executor.buffer("prompt")[2], dtype=np.int64)
    ids[: len(prompt)] = prompt
    program.upload_prefix({"prompt": ids, "prompt_length": np.array([len(prompt)], np.int64)})


def _python_step(program, position, previous):
    """One step of the same program through the Python executor: the position, and the token the
    host selected from the previous step's logits, are its only per-step inputs."""
    program.upload_prefix({"position": np.array([position], np.int64), "next_token": np.array([previous or 0], np.int64)})
    program.run_once()
    return program.outputs()["logits"].reshape(-1)


def test_cached_qwen3_logits_and_generation(tmp_path, monkeypatch):
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    # Bit-identical replay requires a fixed reduction order, excluding atomic split reductions.
    monkeypatch.setenv("EMMY_REDUCE", "")
    model = qwen3_model(2).half()
    model.config._attn_implementation = "eager"
    with gpu_lock():
        root = export_model(model, tmp_path / "pack", context_length=8, prefill_size=3)
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
                for capture, prompt in (
                    (False, [1, 2, 3, 4, 5, 6, 7, 8]),
                    (False, [1, 2, 3, 4, 5, 6, 7]),
                    (True, [1, 2, 3, 4, 5, 6]),
                    (None, [8, 9]),
                    (True, [3]),
                    (True, [4, 5, 6, 7]),
                ):
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
                        actual = np.fromfile(logits_path, np.float32)
                        np.testing.assert_array_equal(actual, _python_step(reference_program, position, next_token).astype(np.float32))
                        np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
                        if position >= len(prompt) - 1:
                            next_token = result["token"]
                            assert next_token == int(expected.argmax())
                            selected.append(next_token)
                        else:
                            assert result["token"] is None
                    # Chunk dispatch advances only valid prompt rows. Reusing captured graphs
                    # across different tails must neither expose padding nor retain an old cache.
                    await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                    position = 0
                    chunked = []
                    while position < 8:
                        result = await worker.run_job(
                            {"op": "generation_step", "prefill": True, "capture": bool(capture), "logits": str(logits_path)},
                            wall_timeout_s=30,
                        )
                        expected_position = position + (min(3, len(prompt) - position - 1) if position < len(prompt) - 1 else 1)
                        assert result["position"] == expected_position
                        position = expected_position
                        if result["token"] is not None:
                            prefix = prompt + chunked
                            with torch.no_grad(), _reference_precision(True):
                                expected = model(torch.tensor([prefix], device="cuda")).logits[0, -1].float().cpu().numpy()
                            actual = np.fromfile(logits_path, np.float32)
                            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
                            chunked.append(result["token"])
                    assert chunked == selected
                    output = tmp_path / "generated.bin"
                    result = await worker.run_job(
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
                    # One time per token, since the request started: never decreasing.
                    assert len(result["token_ms"]) == len(selected) - 1 and result["token_ms"] == sorted(result["token_ms"])
                # Same seed resets the random stream, including when a graph survives requests.
                sampled = []
                for capture, seed in ((False, 81), (True, 81), (True, 19), (True, 81)):
                    await worker.run_job(
                        {
                            "op": "generate",
                            "prompt": str(path),
                            "max_new_tokens": 4,
                            "capture": capture,
                            "output": str(output),
                            "sampling": {"temperature": 1.5, "top_p": 0.9, "seed": seed},
                        },
                        wall_timeout_s=30,
                    )
                    sampled.append(np.fromfile(output, np.int64).tolist())
                assert sampled[0] == sampled[1] == sampled[3]
                assert sampled[2] != sampled[0]
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


def test_cached_qwen3_5_carries_state_across_steps_and_resets(tmp_path, monkeypatch):
    """The Gated DeltaNet plan's gate on a tiny two-layer Qwen3.5 text model, one layer of each
    type: logits match transformers at prompt lengths 1, 16, 17 and 65, a second request after a
    reset equals the first, and capture on and off select the same tokens.

    Sequential prefill only. Without measured evidence the chunk rule's kernels take no cut, and a
    one-block kernel then recomputes a 64-wide reduction per state cell: 14 s per two-row chunk on
    the smallest geometry, hours at sixteen rows. Chunked prefill of a Gated DeltaNet layer waits
    for recorded schedules; the chunk rule for state is a runtime unit test."""
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    monkeypatch.setenv("EMMY_REDUCE", "")
    context, decode_steps = 80, 4
    model = qwen3_5_model(max_position_embeddings=context).half()
    model.config._attn_implementation = "eager"
    with gpu_lock():
        root = export_model(model, tmp_path / "pack", context_length=context, prefill_size=1)
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["key"]["generation"]["state"] == ["layer0.state", "layer0.history"]
        monkeypatch.setenv("PATH", "/nonexistent")

        def reference(prefix):
            # On the CPU: transformers' Gated DeltaNet fast path would build Triton kernels, which
            # the hidden PATH forbids, and the tiny model needs no card.
            with torch.no_grad():
                return model(torch.tensor([prefix])).logits[0, -1].float().numpy()

        async def check():
            worker = NativeWorker(executable=executable)
            path, logits_path, output = tmp_path / "prompt.bin", tmp_path / "logits.bin", tmp_path / "generated.bin"
            try:
                await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=60)
                for length in (1, 16, 17, 65):
                    prompt = [(7 * i + length) % 64 for i in range(length)]
                    np.asarray(prompt, np.int64).tofile(path)
                    runs = []
                    for capture in (False, True, False):
                        await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=60)
                        prefix, logits, tokens = list(prompt), [], []
                        for position in range(length + decode_steps):
                            result = await worker.run_job(
                                {"op": "generation_step", "capture": capture, "logits": str(logits_path)}, wall_timeout_s=60
                            )
                            assert result["position"] == position + 1
                            if position + 1 < length:
                                assert result["token"] is None
                                continue
                            actual = np.fromfile(logits_path, np.float32)
                            np.testing.assert_allclose(actual, reference(prefix), rtol=1e-3, atol=1e-3)
                            assert result["token"] == int(actual.argmax())
                            logits.append(actual)
                            tokens.append(result["token"])
                            prefix.append(result["token"])
                        runs.append((np.stack(logits), tokens))
                    # Captured and not: the same tokens. A request after a reset equals the first,
                    # bit for bit where the same program ran.
                    assert runs[1][1] == runs[0][1]
                    np.testing.assert_array_equal(runs[2][0], runs[0][0])
                    await worker.run_job(
                        {"op": "generate", "prompt": str(path), "max_new_tokens": decode_steps, "capture": True, "output": str(output)},
                        wall_timeout_s=60,
                    )
                    assert np.fromfile(output, np.int64).tolist() == runs[0][1][:decode_steps]
            finally:
                await worker.aclose()

        asyncio.run(check())


@pytest.mark.parametrize("experts_gpu", [None, 4, 1])
def test_cached_qwen3_5_moe_routes_like_transformers(tmp_path, monkeypatch, experts_gpu):
    """Native MoE's gate on a tiny Qwen3.5-MoE whose two full-attention layers each carry eight
    routed experts (two per token) and the gated shared expert: every logit matches transformers'
    and every greedy token is its own, through sequential and chunked prefill, with capture on and
    off; a request after a reset repeats the first. A chunk runs the fixed-slot program once over
    all its rows. The device holds every expert, or four or one of them per layer: then the pager
    moves the experts the router picks into the slots, and with one slot the other picks are read
    in place from host memory. The experts are wide, so they dominate the worker's memory, and the
    host holds them once: the heap keeps no copy, and the load's peak never passes through one.

    The layers are attention only: without evidence a Gated DeltaNet layer's kernels take minutes
    to compile (the dense Qwen3.5 gate above), and a MoE layer's routed half is the same programs
    after either mixer."""
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    monkeypatch.setenv("EMMY_REDUCE", "")
    if experts_gpu is not None:
        monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", str(experts_gpu))
    context = 10
    model = qwen3_5_moe_model(max_position_embeddings=context, layer_types=["full_attention"] * 2, moe_intermediate_size=16384).half()
    model.config._attn_implementation = "eager"
    experts = sum(t.nbytes for layer in model.model.layers for t in (layer.mlp.experts.gate_up_proj, layer.mlp.experts.down_proj))
    with gpu_lock():
        root = export_model(model, tmp_path / "pack", context_length=context, prefill_size=2)
        monkeypatch.setenv("PATH", "/nonexistent")

        def reference(prefix):
            with torch.no_grad():
                return model(torch.tensor([prefix])).logits[0, -1].float().numpy()

        async def check():
            worker = NativeWorker(executable=executable)
            path, logits_path = tmp_path / "prompt.bin", tmp_path / "logits.bin"
            try:
                await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=60)
                if experts_gpu is not None:
                    status = (line.split(":") for line in (Path("/proc") / str(worker._proc.pid) / "status").read_text().splitlines())
                    memory = {key: int(value.split()[0]) * 1024 for key, value in status if key in ("VmHWM", "VmRSS", "RssAnon")}
                    assert memory["RssAnon"] < experts / 2 and memory["VmHWM"] - memory["VmRSS"] < experts / 2, memory
                for prompt in ([3], [5, 1, 4, 1, 5, 9, 2]):
                    np.asarray(prompt, np.int64).tofile(path)
                    runs = []
                    for capture, prefill in ((False, False), (True, False), (False, False), (True, True)):
                        await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=60)
                        prefix, logits = list(prompt), []
                        while len(prefix) < context:
                            step = {"op": "generation_step", "prefill": prefill, "capture": capture, "logits": str(logits_path)}
                            result = await worker.run_job(step, wall_timeout_s=60)
                            if result["token"] is None:
                                continue
                            actual, expected = np.fromfile(logits_path, np.float32), reference(prefix)
                            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
                            assert result["token"] == int(expected.argmax())
                            logits.append(actual)
                            prefix.append(result["token"])
                        runs.append((prefix, np.stack(logits)))
                    # The same tokens every way; the same program after a reset, the same logits bit for bit.
                    assert runs[0][0] == runs[1][0] == runs[2][0] == runs[3][0]
                    np.testing.assert_array_equal(runs[2][1], runs[0][1])
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
    ("context_1024", "The garden has a pond, a wooden bench, and a path lined with trees. ", 1008, 16, False),
    ("context_4096", "The museum catalog describes paintings, pottery, maps, and tools from different centuries. ", 4080, 16, True),
    # Held out until the FP32 attention-intermediate fix and its independent regression were committed.
    ("heldout_attention", "Why does an ice cube float in water? Give a brief explanation.", None, 24, True),
    ("heldout_context_4096", "The coastal survey records tides, winds, water temperatures, and seabird sightings. ", 4080, 16, False),
    # Selected after the rotary precision fix; also qualifies Python dispatch beyond its old shared-memory limit.
    ("heldout_rotary_context", "The field notebook lists soil samples, rainfall, seed counts, and flowering dates. ", 496, 16, True),
    # Fixed after distinguishing whole-prompt RMS from the much shorter chunked output window.
    ("heldout_prefill_tail", "The laboratory log records sample weights, temperatures, and observation times. ", 1007, 24, True),
    ("heldout_prefill_long", "The archive contains letters, photographs, shipping records, and handwritten notes. ", 4080, 16, False),
)


def test_fp8_checkpoint_exports_its_coded_trunk(tmp_path):
    """A dynamic block-FP8 checkpoint exports onto the checkpoint-sourced lane: the pack carries the
    trunk's e4m3 codes, never decoded weights, and every step computes what the dequantized model
    does — through a prefill chunk as well as the one-token program."""
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    from emmy.compiler.backend.plan import plan_from_dict
    from emmy.serving.native.prepare import load_model
    from tests.serving.helpers import fp8_block_checkpoint

    model = qwen3_model(2).half()
    model.config._attn_implementation = "eager"
    fp8_block_checkpoint(model, tmp_path / "ckpt")  # the model now holds the values the checkpoint stores
    twin, ckpt = load_model(str(tmp_path / "ckpt"))
    with gpu_lock():
        root = export_model(twin, tmp_path / "pack", context_length=8, prefill_size=3, ckpt=ckpt)
    manifest = json.loads((root / "manifest.json").read_text())

    def weights(module):
        return sum(m.weight.numel() for m in module.modules() if isinstance(m, torch.nn.Linear))

    layers = model.model.layers
    # Every trunk weight a program runs is in the pack as its e4m3 code, so none was bound decoded;
    # a prefill chunk stops at the last layer's keys and values, so its post half is not there.
    tail = weights(layers[-1].self_attn.o_proj) + weights(layers[-1].mlp)
    for name, expected in (("decode", weights(layers)), ("prefill", weights(layers) - tail)):
        plan = plan_from_dict(json.loads((root / manifest["programs"][name]).read_text()))
        coded = [b for b in plan.buffers if b.role == "constant" and b.dtype.name == "f8e4m3"]
        assert sum(int(np.prod(b.resolve_shape({}))) for b in coded) == expected
    _computes_the_dequantized_model(model, root, executable, tmp_path)


@pytest.mark.parametrize(
    "experts_gpu,dtype", [(None, torch.float16), (4, torch.float16), (4, torch.bfloat16)], ids=["resident", "paged", "paged-bf16"]
)
def test_fp8_moe_checkpoint_exports_its_coded_experts(tmp_path, monkeypatch, experts_gpu, dtype):
    """A dynamic block-FP8 Qwen3.5-MoE checkpoint exports its routed experts as the plugin's
    fixed-slot program reads them, streamed from the shards: E-stacked e4m3 codes beside their block
    scales, decoded in-graph. On the card they are constants of the step; four of eight per layer,
    codes and scales alike are the pack's pages. Every step computes what the dequantized model
    does, through a prefill chunk as well as the one-token program. Two attention layers, as in the
    FP16 MoE gate above. A BF16 checkpoint exports in BF16, as the plugin serves it."""
    executable = shutil.which("emmy-runtime-worker")
    if not executable:
        pytest.skip("build native worker and add it to PATH")
    from emmy.compiler.backend.plan import plan_from_dict
    from emmy.serving.native.prepare import load_model
    from tests.serving.helpers import fp8_block_checkpoint

    if experts_gpu is not None:
        monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", str(experts_gpu))
    model = qwen3_5_moe_model(max_position_embeddings=8, layer_types=["full_attention"] * 2).to(dtype)
    model.config._attn_implementation = "eager"
    fp8_block_checkpoint(model, tmp_path / "ckpt")
    twin, ckpt = load_model(str(tmp_path / "ckpt"))
    assert twin.dtype == dtype
    with gpu_lock():
        root = export_model(twin, tmp_path / "pack", context_length=8, prefill_size=3, ckpt=ckpt)
    manifest = json.loads((root / "manifest.json").read_text())
    plan = plan_from_dict(json.loads((root / manifest["programs"]["decode"]).read_text()))
    inputs = [f"layer{i}.{n}{part}" for i in range(2) for n in ("w_gate_up", "w_down") for part in ("", "_scale")]
    buffers = {b.name: b for b in plan.buffers}
    assert [buffers[name].dtype.name for name in inputs] == ["f8e4m3", "f32"] * 4
    assert buffers["layer0.xn"].dtype.name == {torch.float16: "f16", torch.bfloat16: "bf16"}[dtype]
    assert set(manifest.get("pages", ())) == (set() if experts_gpu is None else set(inputs))
    # The reference rounds its logits to the model dtype; BF16 keeps 8 bits of precision, FP16 11.
    _computes_the_dequantized_model(model, root, executable, tmp_path, tolerance=1e-3 if dtype == torch.float16 else 8e-3)


def _computes_the_dequantized_model(model, root, executable, tmp_path, tolerance=1e-3):
    """Every logit and greedy token of the pack at ``root`` is the eager model's, within ``tolerance``,
    through the one-token program and through prefill chunks."""
    from emmy.compiler.backend.cuda._bench_worker import _reference_precision

    model.cuda()

    async def check():
        worker = NativeWorker(executable=executable)
        path, logits_path = tmp_path / "prompt.bin", tmp_path / "logits.bin"
        try:
            await worker.run_job({"op": "load_generation", "root": str(root)}, wall_timeout_s=30)
            for prefill, prompt in ((False, [1, 2, 3, 4, 5]), (True, [6, 7, 8, 9])):
                np.asarray(prompt, np.int64).tofile(path)
                await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                generated = []
                while len(prompt) + len(generated) < 8:
                    result = await worker.run_job(
                        {"op": "generation_step", "prefill": prefill, "capture": False, "logits": str(logits_path)}, wall_timeout_s=30
                    )
                    if result["token"] is None:
                        continue
                    with torch.no_grad(), _reference_precision(True):
                        expected = model(torch.tensor([prompt + generated], device="cuda")).logits[0, -1].float().cpu().numpy()
                    np.testing.assert_allclose(np.fromfile(logits_path, np.float32), expected, rtol=tolerance, atol=tolerance)
                    assert result["token"] == int(expected.argmax())
                    generated.append(result["token"])
        finally:
            await worker.aclose()

    asyncio.run(check())


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
@pytest.mark.parametrize("prefill", [False, True], ids=["sequential", "chunked"])
def test_checkpoint_logits_and_completions(request, tmp_path, monkeypatch, name, text, prompt_length, decode_steps, capture, prefill):
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
        # Include a longer cache history in exact dispatcher parity, exercising dynamic shared memory.
        reference_program = (
            _python_reference(artifact) if not prefill and name in ("france", "arithmetic", "heldout_rotary_context") else None
        )
        monkeypatch.setenv("PATH", "/nonexistent")
        monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)

        async def check():
            worker = NativeWorker(executable=executable)
            path, logits_path = tmp_path / "prompt.bin", tmp_path / "logits.bin"
            measurements = []
            precise_outputs = []
            try:
                await worker.run_job({"op": "load_generation", "root": artifact}, wall_timeout_s=60)
                np.asarray(prompt, np.int64).tofile(path)
                await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                if reference_program is not None:
                    _reset_python(reference_program, prompt)
                prefix, next_token, past, precise_past = [], None, None, None
                native_position = 0
                for position in range(len(prompt) + decode_steps):
                    prefix.append(prompt[position] if position < len(prompt) else next_token)
                    if position == native_position:
                        result = await worker.run_job(
                            {"op": "generation_step", "capture": capture, "prefill": prefill, "logits": str(logits_path)},
                            wall_timeout_s=30,
                        )
                        native_position = result["position"]
                    with torch.no_grad(), _reference_precision(True):
                        ids = torch.tensor([[prefix[-1]]], device="cuda")
                        reference = model(ids, past_key_values=past, use_cache=True)
                        past = reference.past_key_values
                        expected = reference.logits[0, -1].float().cpu().numpy()
                        accurate = precise(ids, past_key_values=precise_past, use_cache=True)
                        precise_past = accurate.past_key_values
                        fp32 = accurate.logits[0, -1].cpu().numpy()
                    if prefill and position < len(prompt) - 1:
                        continue
                    if prefill:
                        precise_outputs.append(fp32)
                    actual = np.fromfile(logits_path, np.float32)
                    if reference_program is not None:
                        np.testing.assert_array_equal(actual, _python_step(reference_program, position, next_token).astype(np.float32))
                    assert np.isfinite(actual).all() and np.isfinite(expected).all() and np.isfinite(fp32).all()
                    native_error, reference_error = _logit_errors(actual, fp32), _logit_errors(expected, fp32)
                    measurements.append(
                        {
                            "case": name,
                            "position": position,
                            "capture": capture,
                            "prefill": prefill,
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
                    if position % 64 == 0 or position + 1 == len(prompt) + decode_steps:
                        (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
                    if result["token"] is not None:
                        next_token = result["token"]
                        assert next_token == int(actual.argmax())
                if prefill:
                    # Teacher-force the exact observed prefix through one-token execution. A chunk
                    # exposes no intermediate prompt logits, so its shorter RMS window must also
                    # be compared with the baseline on that same window, not the old whole prompt.
                    np.asarray(prefix, np.int64).tofile(path)
                    await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                    for position in range(len(prefix)):
                        await worker.run_job({"op": "generation_step", "capture": capture, "logits": str(logits_path)}, wall_timeout_s=30)
                        if position >= len(prompt) - 1:
                            index = position - len(prompt) + 1
                            baseline = np.fromfile(logits_path, np.float32)
                            measurements[index]["sequential_fp32"] = _logit_errors(baseline, precise_outputs[index])
                    (tmp_path / "measurements.json").write_text(json.dumps(measurements, indent=2))
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
                    sequential_rms = np.sqrt(np.mean([row["sequential_fp32"][metric] ** 2 for row in measurements])) if prefill else 0
                    assert native_rms <= max(REFERENCE_ERROR_FACTOR * reference_rms, np.finfo(np.float16).eps, sequential_rms), (
                        name,
                        metric,
                        native_rms,
                        reference_rms,
                        sequential_rms,
                    )
                if prompt_length is not None and prompt_length >= 1008:
                    # A short request after a full cache must see only its own overwritten prefix.
                    np.asarray(prompt[:3], np.int64).tofile(path)
                    await worker.run_job({"op": "start_generation", "prompt": str(path)}, wall_timeout_s=30)
                    for _ in range(3):
                        reset = await worker.run_job({"op": "generation_step", "capture": True}, wall_timeout_s=30)
                    with torch.no_grad(), _reference_precision(True):
                        expected = model(torch.tensor([prompt[:3]], device="cuda")).logits[0, -1]
                    assert reset["token"] == int(expected.argmax())
            finally:
                await worker.aclose()

        asyncio.run(check())


@pytest.mark.parametrize("page_tokens", [4096, 128], ids=["one_page", "paged"])
@pytest.mark.parametrize("near_tie", [False, True], ids=["random", "near_tie"])
def test_attention_reads_only_the_written_cache_prefix(near_tie, page_tokens):
    """The compiled attention masks every position past the current one on the device, so the
    unwritten rest of the cache — poisoned here with large finite values — never reaches the
    output, at page boundaries and at both ends of the context alike."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.trace.torch import trace_module
    from emmy.serving.native.prepare import MAX_CONTEXT, attend_module

    heads, kv, d = 4, 2, 128
    rng = np.random.default_rng(19)
    query = rng.normal(size=(1, heads * d)).astype(np.float16)
    keys = rng.normal(size=(1, kv, MAX_CONTEXT, d)).astype(np.float16)
    # The values are stored by head dimension, each dimension's positions in a row.
    values = rng.normal(size=(1, kv, d, MAX_CONTEXT)).astype(np.float16)
    if near_tie:
        # QK is about 1024: FP16 score storage would erase a real 0.125 difference.
        query.fill(1)
        keys.fill(8)
        keys[0, :, 0, 0] += 0.125
        values.fill(-1)
        values[0, :, :, 0] = 1
    # Distinct example tensors: two arguments sharing one trace as a single aliased input.
    examples = (
        torch.zeros(1, heads * d, dtype=torch.float16),
        torch.zeros(1, kv, MAX_CONTEXT, d, dtype=torch.float16),
        torch.zeros(1, kv, d, MAX_CONTEXT, dtype=torch.float16),
        torch.zeros(1, dtype=torch.int64),
    )
    graph = trace_module(attend_module(heads, kv, d, MAX_CONTEXT, 1), examples)
    graph.hints.set("cuda.paged_buffers", (("keys", 2, page_tokens), ("values", 3, page_tokens)))
    compiled = CudaBackend().compile(graph)
    with gpu_lock():
        program = CompiledProgram.build(compiled, {"q": query, "position": np.array([0], np.int64)})
        # Ascending and descending lengths catch stale future data after request reset.
        for count in (1, 127, 128, 129, MAX_CONTEXT - 1, MAX_CONTEXT, 2):
            k, v = keys.copy(), values.copy()
            k[:, :, count:] = 1e4
            v[..., count:] = 1e4
            pages: list = []  # the tables hold raw device pointers: the pages outlive the launch
            program.alias_buffer("keys__pages", _pages(torch, k, page_tokens, pages, axis=2))
            program.alias_buffer("values__pages", _pages(torch, v, page_tokens, pages, axis=3))
            program.upload_prefix({"position": np.array([count - 1], np.int64)})
            program.run_once()
            actual = program.outputs()[compiled.outputs[0]].reshape(heads, d)
            # Independent float64 attention over the written prefix only; inputs and output are FP16.
            group = heads // kv
            kk = keys[0, :, :count].repeat(group, axis=0).astype(np.float64)
            vv = values[0, ..., :count].transpose(0, 2, 1).repeat(group, axis=0).astype(np.float64)
            scores = np.einsum("hd,htd->ht", query.reshape(heads, d).astype(np.float64), kk) * d**-0.5
            probabilities = np.exp(scores - scores.max(axis=-1, keepdims=True))
            probabilities /= probabilities.sum(axis=-1, keepdims=True)
            expected = np.einsum("ht,htd->hd", probabilities, vv).astype(np.float16)
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)


def _pages(torch, cache, page_tokens, keep, axis):
    """A page table over ``cache``'s token ``axis``, its pages appended to ``keep`` so the device
    memory the table points at outlives the call."""
    start = len(keep)
    for offset in range(0, cache.shape[axis], page_tokens):
        page = np.take(cache, range(offset, offset + page_tokens), axis=axis)
        keep.append(torch.from_numpy(np.ascontiguousarray(page)).cuda())
    torch.cuda.synchronize()  # the copies are on torch's stream, which the runtime's launches do not wait on
    return torch.tensor([page.data_ptr() for page in keep[start:]], dtype=torch.int64, device="cuda")
