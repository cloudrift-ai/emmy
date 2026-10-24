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


def test_a_tower_below_its_block_budget_pages_its_projections(monkeypatch):
    """``EMMY_GEN_VISION_GPU`` counts the blocks the device holds: unset, or at the block count, the tower stays on the
    device. Below it every block's projection weights (the 2-D ones) are one page each of the pager's, in the layout
    the kernels read them in, which no TMA descriptor stages and no inspector reads: the host lists a block's page
    itself. The norms and biases stay constants on the device."""
    from emmy import config
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.dtype import BF16
    from emmy.serving.gen_runner import _bind_plan_constants, _traced_split, module_sources
    from emmy.serving.vision_tower import hf_tower, page_projections, tower_pieces

    monkeypatch.delenv("EMMY_GEN_VISION_GPU", raising=False)
    assert config.gen_vision_slots(2) is None
    for budget, slots in (("0", 0), ("1", 1), ("2", None), ("5", None)):
        monkeypatch.setenv("EMMY_GEN_VISION_GPU", budget)
        assert config.gen_vision_slots(2) == slots
    monkeypatch.setenv("EMMY_GEN_VISION_GPU", "-1")
    with pytest.raises(ValueError, match="EMMY_GEN_VISION_GPU"):
        config.gen_vision_slots(2)

    hidden, inner = VISION_TINY["hidden_size"], VISION_TINY["intermediate_size"]
    pieces = tower_pieces(hf_tower(vl_config(), torch.bfloat16), 0)
    for piece, projections in (
        ("pre", {"qkv.weight": 3 * hidden * hidden}),
        ("post", {"proj.weight": hidden * hidden, "mlp.linear_fc1.weight": inner * hidden, "mlp.linear_fc2.weight": hidden * inner}),
    ):
        wrapper, example = pieces[piece]
        wrapper.to_empty(device="cpu")
        graph = _traced_split(wrapper, example(IMAGE_TOKEN_BUCKETS[0]), None, None)
        page_projections(graph)
        plan = plan_from_graph(CudaBackend().compile(graph))
        assert {plan.weights[name].source_path for name in plan.paged} == set(projections)
        assert all(slots == 0 for _, _, slots in plan.paged.values())
        assert not any(".inspect." in launch.node_id for launch in plan.launches)
        assert not {t.src_buf for launch in plan.launches for t in launch.tma_descriptors} & set(plan.paged)
        pages = _bind_plan_constants(plan, module_sources(wrapper, BF16), None, paged=True)
        assert {plan.weights[name].source_path: values.size for name, values in pages.items()} == projections
        assert not set(_bind_plan_constants(plan, module_sources(wrapper, BF16), None)) & set(plan.paged)


def test_a_text_step_never_runs_the_tower():
    """vLLM runs the tower only for a step that schedules an image (``embed_multimodal``). A text step embeds its tokens
    and runs the trunk; neither may touch the tower, whose pages a text request must never move over the link."""
    vllm_model_vision = pytest.importorskip("emmy.serving.vllm_model_vision")
    model = vllm_model_vision.EmmyGenQwenVLModel.__new__(vllm_model_vision.EmmyGenQwenVLModel)
    torch.nn.Module.__init__(model)
    model.visual = torch.nn.Linear(4, 4)

    def refuse(module, args):
        raise AssertionError("a text step ran the vision tower")

    model.visual.register_forward_pre_hook(refuse)
    calls = []
    model.language_model = _Trunk(calls)
    ids = torch.tensor([1, 2, 3])
    assert model.embed_input_ids(ids).shape == (3, 4)
    model(ids, torch.arange(3).expand(3, -1))
    assert calls == ["embed", "forward"]


class _Trunk(torch.nn.Module):
    def __init__(self, calls):
        super().__init__()
        self.calls = calls

    def embed_input_ids(self, ids):
        self.calls.append("embed")
        return torch.zeros(len(ids), 4)

    def forward(self, input_ids, positions, intermediate_tensors=None, inputs_embeds=None):
        self.calls.append("forward")
        return torch.zeros(len(input_ids), 4)
