import shlex
from pathlib import Path

import pytest

from emmy.serving.release import load_serving_config
from emmy.serving.twins import _serving_twin_buckets


def _release_config(path: Path, **overrides: str) -> Path:
    values = {
        "SERVE_MODEL": "org/model",
        "SERVE_GPU": "NVIDIA-Test",
        "SERVE_GOLDEN_FILE": str(path.with_suffix(".golden.json")),
        "SERVE_STATIC_ONLY": "1",
        "SERVE_MAX_NUM_BATCHED_TOKENS": "1",
        "SERVE_DECODE_BUCKET": "1",
        "SERVE_PREFILL_CAPACITY": "1",
        "SERVE_PREFILL_BUCKET": "0",
        "SERVE_M1_TIER": "1",
        "SERVE_CAPTURE_SIZES": "[1]",
    }
    values.update(overrides)
    path.write_text("".join(f"{key}={shlex.quote(value)}\n" for key, value in values.items()))
    return path


def test_standard_serving_twin_scope_keeps_all_widths_and_symbolic():
    assert _serving_twin_buckets(32, 256, (), symbolic=True, static_only=False) == [
        ("32", 32),
        ("256", 256),
        ("-sym", None),
    ]


def test_static_only_serving_twin_scope_is_exactly_m1():
    assert _serving_twin_buckets(1, 0, (), symbolic=True, static_only=True) == [("1", 1)]


@pytest.mark.parametrize(
    ("decode", "prefill", "extra"),
    [(2, 0, ()), (1, 1, ()), (1, 0, (8,))],
)
def test_static_only_serving_twin_scope_rejects_wider_inputs(decode, prefill, extra):
    with pytest.raises(ValueError, match="static-only serving capture requires"):
        _serving_twin_buckets(decode, prefill, extra, symbolic=True, static_only=True)


def test_static_only_release_config_accepts_exact_and_inherited_m1_warm_shapes(tmp_path):
    path = _release_config(tmp_path / "model.env", SERVE_WARM_SHAPES=":: 1:0:1")
    assert load_serving_config(path).static_only


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("SERVE_MAX_NUM_BATCHED_TOKENS", "2"),
        ("SERVE_DECODE_BUCKET", "2"),
        ("SERVE_PREFILL_CAPACITY", "2"),
        ("SERVE_PREFILL_BUCKET", "1"),
        ("SERVE_M1_TIER", "0"),
        ("SERVE_CAPTURE_SIZES", "[1, 2]"),
    ],
)
def test_static_only_release_config_rejects_unsafe_pinned_values(tmp_path, field, value):
    path = _release_config(tmp_path / "model.env", **{field: value})
    with pytest.raises(ValueError, match=field):
        load_serving_config(path)


@pytest.mark.parametrize("warm", ["2::", ":1:", "::2", "1:0:1:fm"])
def test_static_only_release_config_rejects_unsafe_warm_overrides(tmp_path, warm):
    path = _release_config(tmp_path / "model.env", SERVE_WARM_SHAPES=warm)
    with pytest.raises(ValueError, match="static-only release"):
        load_serving_config(path)


@pytest.mark.parametrize(("extra", "dtype"), [("", None), ("--dtype bf16", "bfloat16"), ("--max-model-len 4096 --dtype=half", "float16")])
def test_release_config_traces_twins_in_the_engine_dtype(tmp_path, extra, dtype):
    """Buffer types are part of a kernel's identity, so the twins take the ``--dtype`` the engine serves with; with
    none, each runtime resolves its own default."""
    assert load_serving_config(_release_config(tmp_path / "model.env", SERVE_EXTRA_ARGS=extra)).dtype == dtype


def test_release_config_rejects_a_dtype_the_trunk_cannot_serve(tmp_path):
    with pytest.raises(ValueError, match="--dtype"):
        load_serving_config(_release_config(tmp_path / "model.env", SERVE_EXTRA_ARGS="--dtype auto"))


def test_release_config_declares_the_cache_regime_on_every_lane(tmp_path):
    """``EMMY_COLD_CACHE`` in ``SERVE_ENV`` is the regime the server compiles and records in, so every lane's rows
    carry the ``COLD_CACHE`` pin, the expert fallback row included; a config without it declares hot rows, with no
    pin. The value is the knob's own spelling."""
    widths = {
        "SERVE_STATIC_ONLY": "0",
        "SERVE_MAX_NUM_BATCHED_TOKENS": "64",
        "SERVE_DECODE_BUCKET": "8",
        "SERVE_PREFILL_CAPACITY": "64",
        "SERVE_PREFILL_BUCKET": "0",
        "SERVE_M1_TIER": "0",
        "SERVE_WARM_SHAPES": "64:::fm",
    }
    cold = load_serving_config(_release_config(tmp_path / "cold.env", SERVE_ENV="EMMY_STRICT_EVIDENCE=1 EMMY_COLD_CACHE=1", **widths))
    hot = load_serving_config(_release_config(tmp_path / "hot.env", SERVE_ENV="EMMY_COLD_CACHE=0", **widths))
    assert {row.name for row in cold.realizations} == {row.name for row in hot.realizations} == {"m8", "dynamic", "m64.fm", "dynamic.fm"}
    assert all(dict(row.pins)["COLD_CACHE"] is True for row in cold.realizations)
    assert all("COLD_CACHE" not in dict(row.pins) for row in hot.realizations)
    assert [row.pins for row in cold.realizations_for(1, expert=True)] == [(("COLD_CACHE", True), ("FAST_MATH", False))]
    with pytest.raises(ValueError, match="SERVE_ENV"):
        load_serving_config(_release_config(tmp_path / "bad.env", SERVE_ENV="EMMY_COLD_CACHE=maybe", **widths))


def test_release_config_carries_lora_rank_into_twin_capture(tmp_path):
    assert load_serving_config(_release_config(tmp_path / "model.env", SERVE_LORA_RANK="8")).lora_rank == 8
    with pytest.raises(ValueError, match="SERVE_LORA_RANK"):
        load_serving_config(_release_config(tmp_path / "bad.env", SERVE_LORA_RANK="0"))
