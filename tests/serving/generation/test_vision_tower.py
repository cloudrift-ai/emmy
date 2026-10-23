"""The vision tower as compiled programs (no GPU): its twins, the serving rows they reach, and how images group.

A tiny Qwen3.5 vision-language config stands in for the checkpoint; the twins trace from the config alone.
"""

import pytest

torch = pytest.importorskip("torch")

from emmy.serving.vision_tower import IMAGE_TOKEN_BUCKETS, CompiledTower  # noqa: E402
from tests.serving.helpers import QWEN3_5_TINY  # noqa: E402

#: A tiny tower of Qwen3.5's shape: 8-pixel patches, two blocks, a 2 x 2 merge into the trunk's hidden size.
VISION_TINY = dict(
    depth=2,
    hidden_size=64,
    num_heads=2,
    intermediate_size=96,
    patch_size=8,
    temporal_patch_size=2,
    in_channels=3,
    spatial_merge_size=2,
    out_hidden_size=QWEN3_5_TINY["hidden_size"],
    num_position_embeddings=16,
)


def vl_config():
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config

    return Qwen3_5Config(text_config=QWEN3_5_TINY, vision_config=VISION_TINY, architectures=["Qwen3_5ForConditionalGeneration"])


def test_tower_twins_trace_every_piece_at_every_width(tmp_path):
    from emmy.serving.twins import capture_twin_graphs

    vl_config().save_pretrained(tmp_path)
    text = capture_twin_graphs(str(tmp_path), decode_bucket=1, prefill_bucket=16, symbolic=False)
    graphs = capture_twin_graphs(str(tmp_path), decode_bucket=1, prefill_bucket=16, symbolic=False, images=True)
    tower = {name: graph for name, graph in graphs.items() if name not in text}
    assert set(tower) == {f"tower{tokens}-{piece}" for tokens in IMAGE_TOKEN_BUCKETS for piece in ("embed", "pre", "post", "merger")}
    hidden, patch = VISION_TINY["hidden_size"], 3 * 2 * 8 * 8
    for tokens in IMAGE_TOKEN_BUCKETS:
        rows = 4 * tokens  # a token merges 2 x 2 patches

        def shapes(name, names, tokens=tokens):
            graph = tower[f"tower{tokens}-{name}"]
            return [tuple(graph.buffer(key).shape) for key in getattr(graph, names)]

        assert shapes("embed", "inputs") == [(rows, patch), (rows, hidden)]
        assert shapes("embed", "outputs") == [(rows, hidden)]
        assert shapes("pre", "outputs") == [(rows, 3 * hidden)]
        # The post program reads the attention output and the residual as two inputs, never one aliased one.
        assert tower[f"tower{tokens}-post"].inputs == ["attn_out", "residual"]
        assert shapes("post", "inputs") == [(rows, hidden), (rows, hidden)]
        assert shapes("merger", "inputs") == [(tokens, 4 * hidden)]
        assert shapes("merger", "outputs") == [(tokens, QWEN3_5_TINY["hidden_size"])]


def _env(tmp_path, extra_args):
    path = tmp_path / "serving.env"
    path.write_text(
        "SERVE_MODEL=org/model\nSERVE_GPU='NVIDIA GeForce RTX 5090'\nSERVE_GOLDEN_FILE=golden.json\n"
        "SERVE_MAX_NUM_BATCHED_TOKENS=64\nSERVE_DECODE_BUCKET=16\nSERVE_PREFILL_BUCKET=64\nSERVE_PREFILL_CAPACITY=64\n"
        f'SERVE_WARM_SHAPES=:::fm\nSERVE_EXTRA_ARGS="{extra_args}"\n'
    )
    return path


def test_a_boot_that_answers_images_carries_the_tower_rows_in_every_lane(tmp_path):
    from emmy.serving.release import load_serving_config
    from emmy.serving.twins import twin_realizations

    images = load_serving_config(_env(tmp_path, """--dtype bfloat16 --limit-mm-per-prompt '{\\"image\\":1,\\"video\\":0}'"""))
    assert images.images
    rows = twin_realizations(images, "tower256-post")
    assert {(row.name, row.bindings, dict(row.pins)["FAST_MATH"]) for row in rows} == {
        ("m256", (("num_tokens", 256),), False),
        ("m256.fm", (("num_tokens", 256),), True),
    }
    assert not load_serving_config(_env(tmp_path, "--dtype bfloat16")).images
    assert not load_serving_config(_env(tmp_path, """--limit-mm-per-prompt '{\\"image\\":0}'""")).images


def test_images_run_together_while_they_fit_the_widest_program():
    tower = CompiledTower.__new__(CompiledTower)
    torch.nn.Module.__init__(tower)
    tower.per_token = 4
    tower.visual = torch.nn.Module()
    tower.visual.patch_size = 16
    widest = IMAGE_TOKEN_BUCKETS[-1] * 4
    small, half = [1, 8, 8], [1, 32, widest // 64]
    assert tower._groups([small, small]) == [[small, small]]
    assert tower._groups([half, half, small]) == [[half, half], [small]]
    with pytest.raises(ValueError, match="wider than the compiled vision tower's widest program"):
        tower._groups([[1, 64, widest // 32]])
