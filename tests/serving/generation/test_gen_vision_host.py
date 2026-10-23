"""The vision tower in host memory (``EMMY_GEN_VISION_HOST``), and the text path that never runs it (no GPU).

``map_to_host`` keeps every tower parameter on the device view of its pinned, mapped host copy and moves each block's
parameters to the device only while that block runs, so vLLM's tower code runs unchanged. Here ``_pinned`` returns
two host copies that stand in for the pinned bytes and their device view, which need a GPU.
"""

import pytest

torch = pytest.importorskip("torch")
vllm_model_vision = pytest.importorskip("emmy.serving.vllm_model_vision")


def _fake_pinned(tensor):
    raw = tensor.detach().clone().contiguous().view(-1).view(torch.uint8)
    return raw, raw.clone()


class _Tower(torch.nn.Module):
    """The shape of vLLM's tower: a table read without running it as a module, a list of blocks, a merger."""

    def __init__(self):
        super().__init__()
        self.pos_embed = torch.nn.Embedding(4, 8)
        self.blocks = torch.nn.ModuleList(torch.nn.Linear(8, 8) for _ in range(3))
        self.merger = torch.nn.Linear(8, 2)

    def forward(self, x):
        x = x + self.pos_embed.weight[0]
        for block in self.blocks:
            x = block(x)
        return self.merger(x)


def test_each_block_runs_from_the_device_and_rests_in_host_memory(monkeypatch):
    monkeypatch.setattr("emmy.serving.gen_runner._pinned", _fake_pinned)
    torch.manual_seed(0)
    tower = _Tower()
    x = torch.randn(5, 8)
    expected = tower(x)
    allocations = vllm_model_vision.map_to_host(tower)
    assert len(allocations) == len(list(tower.parameters()))
    at_rest = {param: param.data_ptr() for param in tower.parameters()}
    host = dict(zip(tower.parameters(), (raw.data_ptr() for raw in allocations), strict=True))
    assert not set(at_rest.values()) & set(host.values()), "at rest a parameter reads the device view, not the bytes"
    running = []

    def record(block, args):
        # The block about to run is staged (here the device copy of the host bytes is the bytes themselves); every
        # other parameter reads its view.
        staged = {param for param in tower.parameters() if param.data_ptr() == host[param]}
        running.append(staged == set(block.parameters()))

    for block in [*tower.blocks, tower.merger]:
        block.register_forward_pre_hook(record)
    torch.testing.assert_close(tower(x), expected)
    assert running == [True] * 4
    assert {param: param.data_ptr() for param in tower.parameters()} == at_rest


def test_a_text_step_never_runs_the_tower():
    """vLLM runs the tower only for a step that schedules an image (``embed_multimodal``). A text step embeds its tokens
    and runs the trunk; neither may touch the tower, whose weights a text request must never read over the link."""
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
