"""Fast CPU tests for ``gen_runner`` helpers (no GPU/model). The decode-bucket compile +
correctness are covered on GPU by ``test_gen_runner_gpu.py`` / ``test_vllm_plugin_gen_gpu.py``."""

import numpy as np
import pytest

from emmy.compiler.dtype import F32
from emmy.serving.gen_runner import EmmyGenRunner, _pad_rows, _program_config_sha, _static_decode_covers_capacity


class _Config:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        import copy

        return copy.deepcopy(self.data)


def test_program_config_identity_ignores_only_generation_eos_policy():
    base = {
        "model_type": "gemma4_unified_text",
        "text_config": {"hidden_size": 3840, "intermediate_size": 15360},
        "eos_token_id": 1,
    }
    instruction = {**base, "eos_token_id": [1, 106]}
    assert _program_config_sha(_Config(base)) == _program_config_sha(_Config(instruction))

    different_architecture = {**instruction, "model_type": "gemma4_other_text"}
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_architecture))

    different_geometry = {
        **instruction,
        "text_config": {**instruction["text_config"], "hidden_size": 4096},
    }
    assert _program_config_sha(_Config(base)) != _program_config_sha(_Config(different_geometry))


def test_pad_rows_pads_with_zeros_and_preserves_real_rows():
    a = np.arange(6, dtype=np.float16).reshape(3, 2)
    out = _pad_rows(a, 5)
    assert out.shape == (5, 2)
    assert out.dtype == np.float16
    np.testing.assert_array_equal(out[:3], a)  # real rows intact
    assert (out[3:] == 0).all()  # padding is zeros (computed then sliced away)


def test_pad_rows_is_passthrough_when_already_at_bucket():
    a = np.ones((4, 8), dtype=np.float16)
    assert _pad_rows(a, 4) is a  # no copy when t == bucket


@pytest.mark.parametrize(
    ("max_tokens", "decode_bucket", "prefill_bucket", "expected"),
    [
        (None, 16, 0, False),
        (1, 1, 0, True),
        (1, 16, 0, True),
        (16, 16, 0, True),
        (17, 16, 0, False),
        (1, 0, 0, False),
        (1, 16, 32, False),
    ],
)
def test_static_decode_capacity_proof(max_tokens, decode_bucket, prefill_bucket, expected):
    assert _static_decode_covers_capacity(max_tokens, decode_bucket, prefill_bucket) is expected


def test_static_only_runner_counts_layers_without_symbolic_programs():
    runner = EmmyGenRunner(
        embed_weight=np.empty((1, 1), dtype=np.float16),
        norm=None,
        pre=[],
        post=[],
        attn_meta=[(1, 1, 1, 1.0), (1, 1, 1, 1.0)],
        np_dtype=np.dtype("float16"),
        pre_decode=[object(), object()],
        post_decode=[object(), object()],
        decode_bucket=1,
        prefill_capacity=1,
    )
    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 0
    assert runner.global_layer_id(1) == 1
    assert runner.prefill_capacity == 1
    assert runner.has_device_decode
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_pre(0, np.zeros((2, 1), dtype=np.float16))
    with pytest.raises(RuntimeError, match="token width 2 exceeds static-only capacity 1"):
        runner.forward_layer_post(
            0,
            np.zeros((2, 1), dtype=np.float16),
            np.zeros((2, 1), dtype=np.float16),
        )


def test_float32_residual_moe_rider_keeps_normalized_activation_in_float16():
    """The two-output fp32-residual post rider must not allocate ``xn`` in residual dtype."""
    torch = pytest.importorskip("torch")

    class PostProgram:
        output_names = ("hidden", "moe_xn")

        def run_device(self, inputs, *, out):
            _attn_out, residual = inputs
            out[0].copy_(residual)
            out[1].copy_(residual.to(out[1].dtype))

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    runner._post_m1 = None
    runner._post_decode = [PostProgram()]
    runner._post_prefill = [PostProgram()]
    runner._post = []
    runner._pre_decode = [object()]
    runner._pre_prefill = [object()]
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._activation_dtype = torch.float16

    residual = torch.randn(6, 8, dtype=torch.float32)
    hidden, normalized = runner._route_post_device(0, torch.randn(6, 8, dtype=torch.float16), residual)

    assert hidden.dtype == torch.float32
    assert normalized.dtype == torch.float16
    assert torch.nn.Linear(8, 2, dtype=torch.float16)(normalized).dtype == torch.float16


def test_fork_attention_pre_rider_sizes_its_destination_at_hidden_width():
    """A chunk step carrying decode riders splits across two programs into one joint destination.
    The classic seam sizes those from ``(q, k, v)``; the fork-attention seam has no projections
    here — its ``pre`` returns one hidden-width activation, and sizing that from the attention
    metadata instead (``num_heads * head_dim``, 32768 on DeepSeek V4 against a hidden size of
    4096) makes every rider-width step fail its copy."""
    torch = pytest.importorskip("torch")

    class PreProgram:
        output_names = ("x",)

        def run_device(self, inputs, *, out):
            out[0].copy_(inputs[0])

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    runner._pre_m1 = None
    runner._pre_decode = [PreProgram()]
    runner._pre_prefill = [PreProgram()]
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._hidden_size = 8
    runner._activation_dtype = torch.float16
    # The attention metadata a fork-attention layer still carries: 4 heads x head_dim 16 is
    # 64 wide, eight times the seam's real width — the shape the old sizing would have used.
    runner._attn_meta = [(16, 4, 1, 0.5)]

    (x,) = runner.forward_layer_pre_device(0, torch.randn(6, 8, dtype=torch.float16))

    assert x.shape == (6, 8)


def test_pipeline_runner_tracks_absolute_layers_and_boundary_ownership():
    runner = EmmyGenRunner(
        embed_weight=None,
        norm=None,
        hidden_size=8,
        layer_ids=[7, 8],
        pre=[],
        post=[],
        attn_meta=[(2, 4, 1, 0.5), (2, 4, 1, 0.5)],
        np_dtype=np.dtype("float16"),
    )

    assert runner.num_layers == 2
    assert runner.global_layer_id(0) == 7
    assert runner.global_layer_id(1) == 8
    with pytest.raises(RuntimeError, match="does not own the token embedding"):
        runner.embed([0])
    with pytest.raises(RuntimeError, match="does not own the final norm"):
        runner.final_norm(np.zeros((1, 8), dtype=np.float16))


@pytest.mark.parametrize(
    ("quant_method", "coded_trunk"),
    [("exl3", True), ("awq", True), ("fp8", True), ("fp8-static", True), ("modelopt", True), ("bitsandbytes", False)],
)
def test_create_keeps_storage_coded_trunks_packed(tmp_path, monkeypatch, quant_method, coded_trunk):
    """EXL3/AWQ/NVFP4 and FP8, static or dynamic, stay checkpoint-coded; a scheme with no coded
    lane keeps the decoded one.

    NVFP4 (``modelopt``) sat in the decoded column while two defects made a coded trunk compute
    silently wrong numbers — a packed operand that dropped its split-K slice base, and a plan-keyed
    constant read that decoded the e4m3 block scales a second time. Both are fixed, and serving
    parity against eager torch is what moved this row: the coded and decoded trunks agree
    bit-for-bit on a layer's q/k/v, and both match eager torch on the rest of the layer.
    """
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    quant_config = {"quant_method": quant_method}
    if quant_method == "fp8-static":
        quant_config = {"quant_method": "fp8", "activation_scheme": "static"}
    if quant_method == "modelopt":
        quant_config["quant_algo"] = "NVFP4"
    if quant_method == "awq":
        quant_config.update(bits=4, version="gemm", zero_point=True)
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": quant_config}))
    seen = {}
    fake_model = object()
    fake_store = {"fmt": quant_method}

    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, *, compress_trunk=False, layer_range=None, include_embed=True, include_norm=True, expert_slice=None):
        seen.update(
            path=path,
            dtype=dtype,
            compress_trunk=compress_trunk,
            layer_range=layer_range,
            include_embed=include_embed,
            include_norm=include_norm,
            expert_slice=expert_slice,
        )
        return fake_model, fake_store

    monkeypatch.setattr(huggingface, "load_quantized_split", fake_load)

    def fake_from_model(cls, model, **kwargs):
        seen.update(cls=cls, model=model, kwargs=kwargs)
        return "runner"

    monkeypatch.setattr(EmmyGenRunner, "from_model", classmethod(fake_from_model))

    assert EmmyGenRunner.create(str(tmp_path), dtype_str="float16") == "runner"
    assert seen["path"] == tmp_path
    assert seen["compress_trunk"] is coded_trunk
    assert seen["model"] is fake_model
    assert seen["kwargs"]["expert_store"] is fake_store


class _StampedGraph(Exception):
    """Carries the graph out of ``_compile_split`` at the backend seam — the last point where the
    loader's spellings are all in, and the first that would need a GPU."""

    def __init__(self, graph):
        super().__init__("captured the stamped split graph")
        self.graph = graph


def test_compile_split_spells_static_fp4_activations_for_a_marked_nvfp4_checkpoint(tmp_path, monkeypatch):
    """A checkpoint declaring static 4-bit input activations must reach serving's split programs as
    the declared W4A4 algebra: a ``to_f4e2m1`` encode ahead of the marked linear and a packed
    ``f4e2m1x2`` activation buffer beside the packed weight. ``emmy compile`` runs the activation
    speller after the weight speller; serving's stamp inside ``_compile_split`` must do the same,
    or the coded trunk computes 4-bit weights against 16-bit activations — the W4A16 scaffolding,
    which is a different program from the one the checkpoint declares.

    The graph is captured at the CUDA backend seam, so nothing here compiles or runs a kernel."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.loader.synthesize import write_quantized_checkpoint
    from emmy.serving.gen_runner import _compile_split, trace_split

    class _Split(torch.nn.Module):
        """One marked linear — the shape a serving split's ``pre`` carries per projection."""

        def __init__(self, hidden, inner):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden, inner, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    wrapper = _Split(64, 32)
    example = (torch.randn(8, 64),)
    ckpt = write_quantized_checkpoint(trace_split(wrapper, example, None), (wrapper, example, {}), tmp_path / "ckpt")
    # What the runner threads as ``ckpt``: the checkpoint dir plus parameter identity → its key.
    id_to_key = {id(wrapper.q_proj.weight): "l0.weight"}

    class _CaptureBackend:
        def __init__(self, **_kwargs):
            pass

        def compile(self, graph, *, ctx=None):
            raise _StampedGraph(graph)

    monkeypatch.setattr("emmy.compiler.backend.cuda.backend.CudaBackend", _CaptureBackend)
    with pytest.raises(_StampedGraph) as caught:
        _compile_split(wrapper, list(example), None, F32, ckpt=(str(ckpt), id_to_key))
    graph = caught.value.graph

    packed_weights = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and type(n.op).__name__ == "ConstantOp"]
    assert packed_weights, "the weight speller never fired — the fixture is not a marked NVFP4 checkpoint"

    encodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "ElementwiseOp" and n.op.name == "to_f4e2m1"]
    packed_activations = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and n not in packed_weights]
    assert encodes, "no to_f4e2m1 encode ahead of the marked linear: serving still runs 16-bit activations"
    assert packed_activations, "no packed f4e2m1x2 activation buffer: only the weight side is spelled"


def test_compile_split_spells_static_fp4_activations_on_a_symbolic_width_split(tmp_path, monkeypatch):
    """The same W4A4 stamp on the tier serving compiles for every non-bucket width: the symbolic
    program, whose token axis is a ``num_tokens`` Var rather than a static extent.

    NVFP4 packs along the LAST (feature) axis — two codes per byte over K, one block scale per 16
    elements of K — and K comes from the weight, so every dim the packing arithmetic divides is
    static. The token axis only rides along, through elementwise ops, a gather and leading-axis
    reshapes. A speller that resolves the WHOLE shape to ints therefore raises on a graph it could
    have spelled, and every layer's symbolic program takes the runner's engine init down with
    it."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.loader.synthesize import write_quantized_checkpoint
    from emmy.serving.gen_runner import _compile_split, trace_split

    class _Split(torch.nn.Module):
        def __init__(self, hidden, inner):
            super().__init__()
            self.q_proj = torch.nn.Linear(hidden, inner, bias=False)

        def forward(self, x):
            return self.q_proj(x)

    wrapper = _Split(64, 32)
    example = (torch.randn(8, 64),)
    ckpt = write_quantized_checkpoint(trace_split(wrapper, example, None), (wrapper, example, {}), tmp_path / "ckpt")
    id_to_key = {id(wrapper.q_proj.weight): "l0.weight"}

    class _CaptureBackend:
        def __init__(self, **_kwargs):
            pass

        def compile(self, graph, *, ctx=None):
            raise _StampedGraph(graph)

    monkeypatch.setattr("emmy.compiler.backend.cuda.backend.CudaBackend", _CaptureBackend)
    with pytest.raises(_StampedGraph) as caught:
        # ``["x"]`` ties the forward arg's axis 0 to the shared symbolic ``num_tokens`` Dim, and
        # ``capacity`` sizes the build feed — exactly how the runner builds its ``*.sym`` tier.
        _compile_split(wrapper, list(example), ["x"], F32, capacity=64, ckpt=(str(ckpt), id_to_key))
    graph = caught.value.graph

    assert not graph.buffer(graph.inputs[0]).shape[0].is_static, "the fixture traced static — not the symbolic tier"

    packed_weights = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and type(n.op).__name__ == "ConstantOp"]
    assert packed_weights, "the weight speller never fired — the fixture is not a marked NVFP4 checkpoint"

    encodes = [n for n in graph.nodes.values() if type(n.op).__name__ == "ElementwiseOp" and n.op.name == "to_f4e2m1"]
    packed_activations = [n for n in graph.nodes.values() if n.output.dtype.name == "f4e2m1x2" and n not in packed_weights]
    assert encodes, "no to_f4e2m1 encode ahead of the marked linear: the symbolic tier still runs 16-bit activations"
    assert packed_activations, "no packed f4e2m1x2 activation buffer: only the weight side is spelled"
    assert any(not d.is_static for n in packed_activations for d in n.output.shape), (
        "the packed activation lost its symbolic token axis — the tier would only serve one width"
    )


def _qwen3_5_full_attention_config():
    """The tiny Qwen3.5 text config with every layer full attention: the runner has no path for a
    linear-attention layer, and the full-attention layer is the one whose query projection also
    carries the attention output gate."""
    pytest.importorskip("transformers.models.qwen3_5")
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

    from tests.serving.helpers import QWEN3_5_TINY

    return Qwen3_5TextConfig(**(QWEN3_5_TINY | {"layer_types": ["full_attention", "full_attention"]}))


def _traced_runner(monkeypatch, model, **kwargs):
    """``EmmyGenRunner.from_model`` with every program traced but none compiled, so it runs on a CPU.

    Returns the runner and its traced graphs as ``[(wrapper class name, token width or None for the
    symbolic program, graph)]`` in build order."""
    import types

    import torch

    from emmy.serving import gen_runner, roofline

    traced = []

    def trace_only(wrapper, example_args, argnames, *_args, **_kwargs):
        graph = gen_runner.trace_split(wrapper, example_args, argnames)
        traced.append((type(wrapper).__name__, None if argnames else example_args[0].shape[0], graph))
        program = types.SimpleNamespace(buffer_view=lambda _name: torch.empty(0), alias_buffer=lambda *_a: None)
        return gen_runner._Program(program, list(graph.inputs), list(graph.outputs)), None

    monkeypatch.setenv("EMMY_GEN_M1_TIER", "1")  # every tier, the single-token twin included
    monkeypatch.setattr(gen_runner, "_compile_split", trace_only)
    monkeypatch.setattr(gen_runner.EmmyGenRunner, "_ensure_device", lambda self: None)
    monkeypatch.setattr(roofline, "audit_boot_programs", lambda *_a, **_k: {})
    return gen_runner.EmmyGenRunner.from_model(model, **kwargs), traced


def _input_shapes(graph, rows):
    return [tuple(rows if not d.is_static else d.value for d in graph.buffer(name).shape) for name in graph.inputs]


@pytest.mark.parametrize("gated", [True, False])
def test_runner_reads_true_head_count_and_feeds_post_the_output_gate(monkeypatch, gated):
    """Qwen3.5's full-attention ``q_proj`` is twice its query width: the second half is a gate that
    multiplies the attention output before ``o_proj``. The runner must count heads from the query
    half (4, not 8) and give every ``post`` program the gate as a third input; with the width alone
    corrected, ``post`` would trace and run without the gate, a wrong answer and no error. An
    ungated model (tiny Qwen3, same head geometry) keeps its two ``post`` inputs."""
    torch = pytest.importorskip("torch")
    import transformers

    from tests.serving.helpers import qwen3_model

    if gated:
        torch.manual_seed(0)
        model = transformers.Qwen3_5ForCausalLM(_qwen3_5_full_attention_config()).eval()
    else:
        model = qwen3_model(2)
    runner, traced = _traced_runner(monkeypatch, model, dtype_str="float32", decode_bucket=4, prefill_bucket=16)

    assert [runner.layer_meta(i)[:3] for i in range(runner.num_layers)] == [(16, 4, 2)] * 2
    assert runner._output_gates == (gated, gated)
    posts = [(rows, graph) for kind, rows, graph in traced if kind == "Post"]
    # Every tier: symbolic, the single-token twin, the decode bucket, the prefill bucket.
    assert sorted({rows or 0 for rows, _graph in posts}) == [0, 1, 4, 16]
    for rows, graph in posts:
        expected = [(rows or "T", 64)] * (3 if gated else 2)
        assert _input_shapes(graph, rows or "T") == expected


def test_gated_runner_post_is_the_serving_twin(tmp_path, monkeypatch):
    """Measured schedules reach serving only when its kernels are the ones the twin capture
    recorded. For a gated layer the runner's ``post`` must therefore be the twin's graph at every
    width, and lower to the same kernels."""
    torch = pytest.importorskip("torch")
    import transformers

    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.bench_record import kernel_row
    from emmy.compiler.pipeline.search.golden.restamp import lift_targets
    from emmy.serving.twins import capture_twin_graphs

    config = _qwen3_5_full_attention_config()
    config.save_pretrained(tmp_path)
    torch.manual_seed(0)
    model = transformers.Qwen3_5ForCausalLM(config).eval()
    _runner, traced = _traced_runner(monkeypatch, model, dtype_str="float32", decode_bucket=4, prefill_bucket=16)
    twins = capture_twin_graphs(str(tmp_path), decode_bucket=4, prefill_bucket=16, extra_widths=(1,), dtype="float32")

    ctx = Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")
    posts = {rows: graph for kind, rows, graph in traced if kind == "Post"}  # layer 1 repeats layer 0
    assert sorted(posts, key=lambda rows: rows or 0) == [None, 1, 4, 16]
    for rows, graph in posts.items():
        twin = twins["post-sym" if rows is None else f"post{rows}"]
        assert graph.structural_key() == twin.structural_key()
        if rows is not None:  # the symbolic program lowers per bound width; the static ones stand for it
            identities = [{kernel_row(t, "k").exact_identity for t in lift_targets(g, ctx).values()} for g in (graph, twin)]
            assert identities[0] and identities[0] == identities[1]


@pytest.mark.parametrize("prefill_bucket", [0, 8])
def test_lora_runner_programs_match_serving_twins(tmp_path, monkeypatch, prefill_bucket):
    pytest.importorskip("torch")
    from transformers import LlamaConfig, LlamaForCausalLM

    from emmy.serving.twins import capture_twin_graphs

    config = LlamaConfig(
        hidden_size=32, intermediate_size=64, num_attention_heads=4, num_key_value_heads=2, num_hidden_layers=1, vocab_size=64
    )
    config.save_pretrained(tmp_path)
    runner, traced = _traced_runner(
        monkeypatch,
        LlamaForCausalLM(config).eval(),
        dtype_str="float32",
        decode_bucket=4,
        prefill_bucket=prefill_bucket,
        max_tokens=8,
        lora_rank=2,
    )
    twins = capture_twin_graphs(
        str(tmp_path), decode_bucket=4, prefill_bucket=prefill_bucket, extra_widths=(1,), dtype="float32", lora_rank=2
    )
    assert runner._lora_rank == 2
    expected = {"pre1", "post1", "pre4", "post4", "pre-sym", "post-sym"}
    if prefill_bucket:
        expected |= {"pre8", "post8"}
    assert set(twins) == expected
    assert {f"{half.lower()}{'-sym' if rows is None else rows}" for half, rows, _ in traced} == expected
    for half, rows, graph in traced:
        name = f"{half.lower()}{'-sym' if rows is None else rows}"
        assert graph.structural_key() == twins[name].structural_key()


def test_lora_prefill_just_below_static_bucket_uses_static_programs():
    torch = pytest.importorskip("torch")
    from emmy.serving.gen_runner import EmmyGenRunner

    class StaticProgram:
        def __init__(self):
            self.inputs = []

        def run_device(self, inputs):
            self.inputs.append(inputs)
            return [inputs[0]]

    class SymbolicProgram:
        def run_device_sym(self, _inputs):
            raise AssertionError("a nearly full LoRA chunk must use the static program")

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    pre, post = StaticProgram(), StaticProgram()
    runner._lora_rank = 2
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._pre_m1 = runner._post_m1 = None
    runner._pre_decode = runner._post_decode = None
    runner._pre_prefill, runner._post_prefill = [pre], [post]
    runner._pre, runner._post = [SymbolicProgram()], [SymbolicProgram()]
    hidden = torch.zeros(3, 8)
    mask = torch.ones(3, 1)
    weight = torch.ones(2, 8)
    runner.forward_layer_pre_device(0, hidden, lora=(mask, weight))
    runner._route_post_device(0, hidden, hidden, lora=(mask, weight))
    assert pre.inputs[0][2].shape == post.inputs[0][3].shape == (2, 8)


def test_lora_prefill_multiple_chunks_preserves_masks_and_weights():
    torch = pytest.importorskip("torch")
    from emmy.serving.gen_runner import EmmyGenRunner

    class StaticProgram:
        def __init__(self):
            self.inputs = []

        def run_device(self, inputs, *, out):
            self.inputs.append(inputs)
            for target in out:
                target.copy_(inputs[0])

    class SymbolicProgram:
        def run_device_sym(self, _inputs):
            raise AssertionError("the nearly full tail should use the static program")

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    pre, post = StaticProgram(), StaticProgram()
    runner._lora_rank = 2
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._pre_m1 = runner._post_m1 = None
    runner._pre_decode = runner._post_decode = None
    runner._pre_prefill, runner._post_prefill = [pre], [post]
    runner._pre, runner._post = [SymbolicProgram()], [SymbolicProgram()]
    runner._attn_meta = [(2, 4, 4, 1.0)]
    runner._activation_dtype = torch.float32
    hidden = torch.arange(56, dtype=torch.float32).reshape(7, 8)
    mask = torch.arange(7, dtype=torch.float32).reshape(7, 1)
    weight = torch.ones(2, 8)
    assert runner.post_attn_backing(0, 7) is None
    assert all(torch.equal(value, hidden) for value in runner.forward_layer_pre_device(0, hidden, lora=(mask, weight)))
    assert torch.equal(runner._route_post_device(0, hidden, hidden, lora=(mask, weight))[0], hidden)
    for program, mask_index in ((pre, 1), (post, 2)):
        assert len(program.inputs) == 2
        assert [part[mask_index].shape[0] for part in program.inputs] == [4, 3]
        assert torch.equal(torch.cat([part[mask_index] for part in program.inputs]), mask)
        assert all(part[mask_index + 1] is weight for part in program.inputs)


def test_lora_prefill_above_static_bucket_keeps_weight_inputs_whole():
    torch = pytest.importorskip("torch")
    from emmy.serving.gen_runner import EmmyGenRunner

    class StaticProgram:
        def __init__(self):
            self.inputs = []

        def run_device(self, inputs, *, out):
            self.inputs.append(inputs)
            for target in out:
                target.copy_(inputs[0])

    class SymbolicProgram:
        def run_device_sym(self, _inputs):
            raise AssertionError("a two-row tail must use the decode program")

    runner = EmmyGenRunner.__new__(EmmyGenRunner)
    pre, post, pre_decode, post_decode = (StaticProgram() for _ in range(4))
    runner._lora_rank = 2
    runner._decode_bucket = 2
    runner._prefill_bucket = 4
    runner._pre_m1 = runner._post_m1 = None
    runner._pre_decode, runner._post_decode = [pre_decode], [post_decode]
    runner._pre_prefill, runner._post_prefill = [pre], [post]
    runner._pre, runner._post = [SymbolicProgram()], [SymbolicProgram()]
    runner._sym_decode_warned = set()
    runner._attn_meta = [(2, 4, 4, 1.0)]
    runner._activation_dtype = torch.float32
    hidden = torch.zeros(6, 8)
    mask = torch.ones(6, 1)
    weight = torch.ones(2, 8)
    runner.forward_layer_pre_device(0, hidden, lora=(mask, weight))
    runner._route_post_device(0, hidden, hidden, lora=(mask, weight))
    for program, rows, weight_index in ((pre, 4, 2), (pre_decode, 2, 2), (post, 4, 3), (post_decode, 2, 3)):
        assert program.inputs[0][0].shape[0] == rows
        assert program.inputs[0][weight_index] is weight


def test_create_passes_the_expert_slice_through_to_the_loader(tmp_path, monkeypatch):
    """A tensor-parallel rank's expert slice must reach the checkpoint read, not just the programs:
    holding every whole expert is what does not fit the card in the first place."""
    import json

    from emmy.compiler.loader import safetensors
    from emmy.compiler.trace import huggingface
    from emmy.serving.gen_runner import EmmyGenRunner

    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "fp8"}}))
    seen = {}
    monkeypatch.setattr(safetensors, "warn_if_unpinned", lambda _model_id: None)
    monkeypatch.setattr(huggingface, "quantized_checkpoint_dir", lambda _model_id: tmp_path)

    def fake_load(path, dtype, **kwargs):
        seen.update(kwargs)
        return object(), {"fmt": "mxfp4"}

    monkeypatch.setattr(huggingface, "load_quantized_split", fake_load)
    monkeypatch.setattr(EmmyGenRunner, "from_model", classmethod(lambda cls, model, **kwargs: kwargs))

    built = EmmyGenRunner.create(model_id=str(tmp_path), expert_slice=(3, 8))

    assert seen["expert_slice"] == (3, 8), "the slice never reached the checkpoint read"
    assert built["expert_slices"] == 8, "the slice never reached the pack key"


def _offset_norm_pre(family):
    """The ``pre`` wrapper of one layer whose norms scale by ``1 + weight``: Qwen3.5 full attention
    or Gemma-3. The input norm's weight is random, so a zero or ``1`` buffer cannot pass for it."""
    torch = pytest.importorskip("torch")
    import transformers

    from emmy.compiler.trace.huggingface import build_attention_split_wrapper
    from tests.serving.helpers import _block

    torch.manual_seed(0)
    if family == "qwen3_5":
        block = transformers.Qwen3_5ForCausalLM(_qwen3_5_full_attention_config()).eval().model.layers[0]
    else:
        _config, block = _block("gemma3")
    with torch.no_grad():
        block.input_layernorm.weight.normal_()
    return build_attention_split_wrapper(block)[0]


@pytest.mark.parametrize("family", ["qwen3_5", "gemma3"])
def test_single_token_pre_plan_binds_every_constant(family):
    """At one row the compiler folds the input norm's ``1 + weight`` into one constant computed
    from the checkpoint weight. The serving runner binds constants from the execution plan alone,
    so the plan must say how to rebuild that constant from the weight; when it could not, the
    runner skipped it and ``pre`` read a zero buffer. Every constant the plan declares must bind,
    and the folded one must hold ``1 + weight``. Lowering needs no GPU."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.context import Context
    from emmy.serving.gen_runner import _bind_plan_constants, trace_split

    pre = _offset_norm_pre(family)
    graph = trace_split(pre, [torch.zeros(1, 64)], None)
    plan = plan_from_graph(CudaBackend().compile(graph, ctx=Context.from_target((12, 0), gpu_name="NVIDIA GeForce RTX 5090")))
    sources = {path: t.detach().numpy() for path, t in [*pre.named_parameters(), *pre.named_buffers()]}

    bound = _bind_plan_constants(plan, sources, None)

    assert set(bound) == set(plan.weights)
    norm = [nid for nid, w in plan.weights.items() if w.source_path == "input_layernorm.weight"]
    assert len(norm) == 1
    expected = 1 + sources["input_layernorm.weight"]
    np.testing.assert_array_equal(bound[norm[0]].reshape(expected.shape), expected)

    # A pack stores the plan as JSON; the folded constant must rebind from that form too.
    import dataclasses
    import json

    from emmy.compiler.backend.plan import WeightSpec, plan_from_dict, plan_to_dict

    packed = plan_from_dict(json.loads(json.dumps(plan_to_dict(plan))))
    np.testing.assert_array_equal(_bind_plan_constants(packed, sources, None)[norm[0]], bound[norm[0]])
    # And a constant the plan cannot rebuild is an error, never a zero buffer.
    unbindable = dataclasses.replace(plan, weights={**plan.weights, norm[0]: WeightSpec(source_path=None, load_ops=None)})
    with pytest.raises(RuntimeError, match=f"plan constant '{norm[0]}' cannot bind"):
        _bind_plan_constants(unbindable, sources, None)
