"""Native generation's image path without a GPU: the tower's programs are the plugin's tower twins, native's own
programs compute what vLLM's tower and M-RoPE compute, and the launcher's image switch."""

import pytest

torch = pytest.importorskip("torch")

from tests.serving.native.test_images_gpu import _checkpoint  # noqa: E402


def test_native_tower_pieces_are_the_plugins_tower_twins(tmp_path, monkeypatch):
    """Every tower program native traces as the plugin's is, wire for wire, one of the tower twins a serving config
    that answers images traces, at its own width; native's own image programs are its twins."""
    from emmy.compiler.backend.plan_cache import binding_neutral_key
    from emmy.serving.native.prepare import _steps, load_model, trace_programs
    from emmy.serving.twins import capture_twin_graphs

    monkeypatch.delenv("EMMY_GEN_VISION_GPU", raising=False)
    _checkpoint(tmp_path / "checkpoint")
    lm, _ = load_model(str(tmp_path / "checkpoint"))
    plugin = capture_twin_graphs(
        str(tmp_path / "checkpoint"), decode_bucket=4, prefill_bucket=8, symbolic=False, dtype="float16", images=True
    )
    twins = {binding_neutral_key(graph): name for name, graph in plugin.items()}
    traced = {}
    _steps(lm, None, None, 8, None, traced=traced, decode_width=4, vision=str(tmp_path / "checkpoint"))
    tower = {twins.get(key) for key, (name, _graph, theirs) in traced.items() if theirs and name.startswith("t")}
    assert tower == {f"tower{tokens}-{piece}" for tokens in (256, 1024) for piece in ("embed", "pre", "post", "merger")}
    native = trace_programs(lm, prefill_size=8, decode_width=4, vision=str(tmp_path / "checkpoint"))
    assert {"merge1-native", "merge8-native", "vpos256-native", "vattn1024-native"} <= set(native)
    assert native["rope1-native"].inputs[-1].startswith("mrope")


def test_mrope_columns_interleave_as_transformers_recomposes_them():
    """Each rotary column reads the axis transformers' interleaved recomposition gives its frequency, so a rotation
    at three positions equals transformers' M-RoPE angles; three equal positions give the plain rotary's columns."""
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding

    from emmy.serving.native.vision import mrope_axes

    config = Qwen3_5TextConfig(
        head_dim=256, rope_parameters={"rope_type": "default", "partial_rotary_factor": 0.25, "mrope_section": [11, 11, 10]}
    )
    rotary = Qwen3_5TextRotaryEmbedding(config)
    axes = mrope_axes(config.rope_parameters, 64)
    assert len(axes) == 64 and axes[:4] == [0, 1, 2, 0] and axes[30:34] == [0, 1, 0, 1]
    table, _ = rotary(torch.zeros(1), torch.arange(64).view(1, -1))
    positions = torch.tensor([[3, 17, 40]]).view(3, 1, 1)
    expected, _ = rotary(torch.zeros(1), positions)
    gathered = torch.gather(table[0], 0, torch.tensor([[3, 17, 40][a] for a in axes]).view(1, -1))
    torch.testing.assert_close(gathered, expected[0], rtol=0, atol=0)


def test_vision_attention_rotates_and_attends_over_the_images_own_patches():
    """The tower attention program's module equals vLLM's tower attention over the image's patches alone: the padding
    rows past them change no image row, through the mask folded into the score's extra dimension."""
    import torch.nn.functional as F

    from emmy.serving.native.vision import vision_attend_module, vision_rope_tables

    heads, head_dim, rows, patches = 2, 72, 40, 33
    cosine, sine = vision_rope_tables(head_dim, torch.float32)
    torch.manual_seed(0)
    qkv, positions = torch.randn(rows, 3 * heads * head_dim), torch.randint(0, 9, (rows, 2))
    actual = vision_attend_module(heads, head_dim, rows)(qkv, positions, torch.tensor([patches]), cosine, sine)
    q, k, v = qkv[:patches].view(patches, 3, heads, head_dim).unbind(1)
    c = torch.cat((cosine[positions[:patches, 0]], cosine[positions[:patches, 1]]), dim=-1).unsqueeze(1)
    s = torch.cat((sine[positions[:patches, 0]], sine[positions[:patches, 1]]), dim=-1).unsqueeze(1)

    def rotate(x):
        x1, x2 = x[..., : head_dim // 2], x[..., head_dim // 2 :]
        return torch.cat((x1 * c - x2 * s, x2 * c + x1 * s), dim=-1).transpose(0, 1)

    expected = F.scaled_dot_product_attention(rotate(q), rotate(k), v.transpose(0, 1)).transpose(0, 1).reshape(patches, -1)
    torch.testing.assert_close(actual[:patches], expected, rtol=1e-5, atol=1e-5)


def test_the_merge_takes_an_image_row_in_place_of_an_embedding():
    from emmy.serving.native.vision import merge_module

    hidden, images = torch.randn(4, 3), torch.randn(5, 3)
    merged = merge_module()(hidden, images, torch.tensor([-1, 2, 0, -1]))
    torch.testing.assert_close(merged, torch.stack([hidden[0], images[2], images[0], hidden[3]]), rtol=0, atol=0)


@pytest.mark.parametrize(
    ("limits", "images"),
    [(None, False), ('{"image":1,"video":0}', True), ('{"image":0}', False), ('{"image":2}', None), ('{"video":1}', None)],
)
def test_the_launcher_answers_at_most_one_image_per_prompt(limits, images):
    from emmy.serving.native.launch import options

    arguments = [] if limits is None else ["--limit-mm-per-prompt", limits]
    if images is None:
        with pytest.raises(ValueError, match="one image"):
            options(arguments)
    else:
        assert options(arguments).images is images
