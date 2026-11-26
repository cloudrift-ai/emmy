"""``EmmyGenModel.load_weights`` — where the vLLM-owned ``lm_head`` comes from (no GPU).

vLLM owns exactly one weight of this model, and its own strict "was everything loaded?" check
waives any parameter whose quant method defines ``process_weights_after_loading`` — which
``ParallelLMHead``'s does. So an unsourced head is invisible to vLLM and the server would answer
with noise while looking healthy. These tests pin the plain ``lm_head.weight``, tied embedding
alias, EXL3 head and NVFP4 head, and the loud
failure when none of them applies, driving ``load_weights`` unbound against a light double —
constructing the real model needs vLLM's engine config plus CUDA.
"""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")
vllm_model_gen = pytest.importorskip("emmy.serving.vllm_model_gen")

from emmy.compiler.loader.exl3 import decode_exl3_linear  # noqa: E402

VOCAB, HIDDEN = 256, 128


def _model(*, model_id="does-not-exist/nowhere", tied=False, vocab=VOCAB, hidden=HIDDEN):
    """The surface ``load_weights`` reads: the head parameter, the HF config's tie flag, the
    optional attention sinks, the runner's adopt hook, and the model id it re-opens for a coded head."""
    param = torch.nn.Parameter(torch.full((vocab, hidden), float("nan")), requires_grad=False)
    adopted: list = []
    runner = types.SimpleNamespace(
        _embed_scale=1.0,
        adopt_embed_table=lambda w, scale=1.0: adopted.append((w, scale)),
    )
    return (
        types.SimpleNamespace(
            lm_head=types.SimpleNamespace(weight=param),
            config=types.SimpleNamespace(tie_word_embeddings=tied),
            sinks=None,
            runner=runner,
            _model_id=model_id,
            _is_last_rank=True,
            _coded_head_rows=1,
        ),
        adopted,
    )


def _load(model, weights=()):
    # ``reclaim_device_memory`` touches the driver; this suite is CPU-only.
    model.reclaim_device_memory = lambda: None
    return vllm_model_gen.EmmyGenModel.load_weights(model, iter(weights))


def test_plain_lm_head_weight_loads_from_the_stream():
    model, adopted = _model()
    w = torch.arange(VOCAB * HIDDEN, dtype=torch.float32).reshape(VOCAB, HIDDEN)
    assert _load(model, [("model.embed_tokens.weight", w * 0), ("lm_head.weight", w)]) == {"lm_head.weight"}
    assert torch.equal(model.lm_head.weight.data, w)
    assert adopted == []  # untied: the head is NOT the runner's embed table


def test_tied_checkpoint_takes_the_embedding_as_the_head():
    """The multimodal 'unified' spelling nests the alias under ``model.language_model.``. The
    hand-off to the runner (``adopt_embed_table``, which drops its own device table) is guarded on
    a CUDA parameter, so on this CPU double it correctly does not fire."""
    model, adopted = _model(tied=True)
    w = torch.randn(VOCAB, HIDDEN)
    assert _load(model, [("model.language_model.embed_tokens.weight", w)]) == {"lm_head.weight"}
    assert torch.equal(model.lm_head.weight.data, w)
    assert adopted == []


def test_unsourced_head_is_a_loud_failure():
    """The whole point: nothing downstream notices a head left at its construction garbage."""
    model, _ = _model()
    with pytest.raises(ValueError, match="no source for lm_head.weight"):
        _load(model, [("model.embed_tokens.weight", torch.randn(VOCAB, HIDDEN))])
    with pytest.raises(ValueError, match="the config is untied"):
        _load(model, [])
    tied, _ = _model(tied=True)
    with pytest.raises(ValueError, match=r"no tied '\*embed_tokens.weight'"):
        _load(tied, [("model.layers.0.mlp.down_proj.weight", torch.randn(4, 4))])


def _write_coded_head(dirpath, *, kt, nt, k_bits, seed=3):
    """A checkpoint whose only tensor is an EXL3-coded ``lm_head`` — no ``lm_head.weight``,
    exactly what turboderp's GLM-4.5-Air-exl3 carries (there at ``head_bits`` 6)."""
    from safetensors.torch import save_file

    rng = np.random.default_rng(seed)
    trellis = rng.integers(-(2**15), 2**15, (kt, nt, 16 * k_bits)).astype(np.int16)
    suh = (rng.choice([-1.0, 1.0], kt * 16) * 0.02).astype(np.float16)
    svh = rng.choice([-1.0, 1.0], nt * 16).astype(np.float16)
    save_file(
        {
            "lm_head.trellis": torch.from_numpy(trellis),
            "lm_head.suh": torch.from_numpy(suh),
            "lm_head.svh": torch.from_numpy(svh),
        },
        str(dirpath / "model.safetensors"),
    )
    cfg = {"model_type": "test", "quantization_config": {"quant_method": "exl3", "bits": 2.0, "head_bits": k_bits}}
    (dirpath / "config.json").write_text(json.dumps(cfg))
    return decode_exl3_linear(trellis, suh, svh).T  # (out, in) — vLLM's [vocab, hidden] orientation


def test_exl3_coded_head_decodes_from_the_checkpoint(tmp_path):
    ref = _write_coded_head(tmp_path, kt=8, nt=16, k_bits=6)
    model, _ = _model(model_id=str(tmp_path), vocab=256, hidden=128)

    def _explode():
        raise AssertionError("the coded path must not walk vLLM's weight stream (~29 GiB of expert codes)")

    assert _load(model, (_explode() for _ in range(1))) == {"lm_head.weight"}
    got = model.lm_head.weight.data.numpy()
    assert np.isfinite(got).all()
    np.testing.assert_allclose(got.astype(np.float32), ref.astype(np.float32), rtol=1e-3, atol=0)


def test_compiler_exl3_head_loads_codes_without_decoding_weight(tmp_path, monkeypatch):
    _write_coded_head(tmp_path, kt=8, nt=16, k_bits=6)
    model, _ = _model(model_id=str(tmp_path), vocab=256, hidden=128)
    source = vllm_model_gen._coded_lm_head_source(str(tmp_path))
    marker, dense = object(), model.lm_head.weight
    model._coded_head_spec = object()
    model._coded_head_source = source
    monkeypatch.setattr(vllm_model_gen, "CodedHead", lambda spec, coded, capacity: marker)
    assert _load(model) == {"lm_head.weight"}
    assert model.lm_head is marker, "the coded head takes the lm_head slot, which a speculative drafter shares"
    assert model._coded_head_source is None
    assert torch.isnan(dense).all(), "the compiler path must not materialize the dense head"


def test_exl3_coded_head_zero_fills_vocab_padding(tmp_path):
    """EXL3 rounds the out extent to 128 and vLLM's ParallelLMHead to 64, so the decoded head can
    be WIDER than the logical vocab (and, once padded, narrower than the parameter). Copy what
    fits, leave the rest zero — the logits processor slices the logical vocab off the front."""
    _write_coded_head(tmp_path, kt=8, nt=16, k_bits=2)  # decodes 256 rows
    model, _ = _model(model_id=str(tmp_path), vocab=192, hidden=128)  # parameter holds only 192
    assert _load(model) == {"lm_head.weight"}
    assert torch.isfinite(model.lm_head.weight.data).all()

    wide, _ = _model(model_id=str(tmp_path), vocab=320, hidden=128)
    assert _load(wide) == {"lm_head.weight"}
    assert torch.count_nonzero(wide.lm_head.weight.data[256:]) == 0


def test_coded_head_shape_mismatch_raises(tmp_path):
    _write_coded_head(tmp_path, kt=8, nt=16, k_bits=2)  # in_features 128
    model, _ = _model(model_id=str(tmp_path), vocab=256, hidden=64)
    with pytest.raises(ValueError, match="in_features 128"):
        _load(model)


def _write_nvfp4_head(dirpath, rows):
    """A checkpoint whose head is NVFP4: packed 0x21 holds 0.5 and 1.0, block scales 0.5 / 1.0, global scale 2."""
    from safetensors.torch import save_file

    scales = torch.full((rows, 2), 0.5).to(torch.float8_e4m3fn)
    scales[:, 1] = 1.0
    save_file(
        {
            "lm_head.weight": torch.full((rows, 16), 0x21, dtype=torch.uint8),
            "lm_head.weight_scale": scales,
            "lm_head.weight_scale_2": torch.tensor(2.0),
        },
        str(dirpath / "model.safetensors"),
    )
    (dirpath / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}))
    return torch.tensor([0.5, 1.0] * 8 + [1.0, 2.0] * 8).repeat(rows, 1)


@pytest.mark.parametrize("vocab", [1100, 1152])
def test_nvfp4_head_decodes_scales_and_multiple_blocks(tmp_path, vocab):
    """1100 rows span two decode blocks; a parameter padded past the stored vocab keeps its padding zero."""
    expected = _write_nvfp4_head(tmp_path, rows=1100)
    model, _ = _model(model_id=str(tmp_path), vocab=vocab, hidden=32)

    def _explode():
        raise AssertionError("the stream's lm_head.weight holds packed codes; the NVFP4 path must not walk it")

    assert _load(model, (_explode() for _ in range(1))) == {"lm_head.weight"}
    assert torch.equal(model.lm_head.weight[:1100], expected)
    assert torch.count_nonzero(model.lm_head.weight[1100:]) == 0


def test_compiler_nvfp4_head_loads_codes_at_load_time_without_decoding_weight(tmp_path, monkeypatch):
    """Construction reads nothing of an NVFP4 head but its global scale; ``load_weights`` hands the stored codes
    and the raw e4m3 block-scale bits to the compiled head and leaves the dense parameter alone."""
    _write_nvfp4_head(tmp_path, rows=256)
    assert vllm_model_gen.nvfp4_head_dir(str(tmp_path)) == str(tmp_path)
    model, _ = _model(model_id=str(tmp_path), vocab=256, hidden=32)
    model._coded_head_spec, model._coded_head_source, dense = object(), None, model.lm_head.weight
    built = []
    monkeypatch.setattr(vllm_model_gen, "CodedHead", lambda spec, coded, capacity: built.append(coded) or "head")

    def _explode():
        raise AssertionError("the stream's lm_head.weight holds packed codes; the NVFP4 path must not walk it")

    assert _load(model, (_explode() for _ in range(1))) == {"lm_head.weight"}
    assert model.lm_head == "head"
    (coded,) = built
    assert {k: (v.dtype, v.shape) for k, v in coded.items()} == {
        "lm_head.weight": (np.uint8, (256, 16)),
        "lm_head.weight_scale": (np.uint8, (256, 2)),
        "lm_head.weight_scale_2": (np.float32, ()),
    }
    assert torch.isnan(dense).all(), "the compiler path must not materialize the dense head"


def test_nvfp4_head_with_more_rows_than_the_parameter_raises(tmp_path):
    """An NVFP4 head stores exactly the vocab, unlike EXL3's rows rounded up to 128: extra rows are a mismatch."""
    _write_nvfp4_head(tmp_path, rows=1100)
    model, _ = _model(model_id=str(tmp_path), vocab=1088, hidden=32)
    with pytest.raises(ValueError, match="stores 1100 rows"):
        _load(model)


def test_nvfp4_head_missing_block_scale_raises(tmp_path):
    from safetensors.torch import save_file

    save_file(
        {"lm_head.weight": torch.zeros((256, 16), dtype=torch.uint8), "lm_head.weight_scale_2": torch.tensor(1.0)},
        str(tmp_path / "model.safetensors"),
    )
    (tmp_path / "config.json").write_text(json.dumps({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4"}}))
    model, _ = _model(model_id=str(tmp_path), hidden=32)
    with pytest.raises(ValueError, match="missing its packed weight or block scale"):
        _load(model)


def test_plain_checkpoint_is_not_probed_as_coded(tmp_path):
    """An unquantized checkpoint keeps the ordinary stream path — no checkpoint re-open."""
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "test"}))
    assert vllm_model_gen._coded_lm_head_source(str(tmp_path)) is None
    assert vllm_model_gen.nvfp4_head_dir(str(tmp_path)) is None
    model, _ = _model(model_id=str(tmp_path))
    w = torch.randn(VOCAB, HIDDEN)
    assert _load(model, [("lm_head.weight", w)]) == {"lm_head.weight"}
    assert torch.equal(model.lm_head.weight.data, w)


def test_non_last_pipeline_stage_loads_only_its_attention_sinks():
    model, _ = _model()
    model._is_last_rank = False
    model.start_layer, model.end_layer = 2, 4
    model.sinks = torch.nn.ParameterList([torch.nn.Parameter(torch.full((3,), float("nan")), requires_grad=False) for _ in range(2)])
    weights = [
        ("model.layers.1.self_attn.sinks", torch.full((3,), 1.0)),
        ("model.layers.2.self_attn.sinks", torch.full((3,), 2.0)),
        ("model.layers.3.self_attn.sinks", torch.full((3,), 3.0)),
        ("model.layers.4.self_attn.sinks", torch.full((3,), 4.0)),
        ("lm_head.weight", torch.zeros(VOCAB, HIDDEN)),
    ]

    assert _load(model, weights) == {"sinks.0", "sinks.1"}
    assert torch.equal(model.sinks[0], torch.full((3,), 2.0))
    assert torch.equal(model.sinks[1], torch.full((3,), 3.0))
