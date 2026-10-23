"""The vision tower in mapped host memory (``EMMY_GEN_VISION_HOST``), and the text path that never runs it (no GPU).

``map_to_host`` gives every tower parameter the device view of its own pinned, mapped host copy, so vLLM's tower code
runs unchanged and reads the weights over the link. Here ``_pinned`` is replaced by a host copy that stands in for
the mapping, which needs a GPU.
"""

import pytest

torch = pytest.importorskip("torch")
vllm_model_vision = pytest.importorskip("emmy.serving.vllm_model_vision")


def _fake_pinned(tensor):
    raw = tensor.detach().clone().contiguous().view(-1).view(torch.uint8)
    return raw, raw


def test_every_tower_parameter_reads_its_mapped_copy(monkeypatch):
    monkeypatch.setattr("emmy.serving.gen_runner._pinned", _fake_pinned)
    tower = torch.nn.Sequential(torch.nn.Linear(8, 6), torch.nn.LayerNorm(6)).to(torch.bfloat16)
    before = {name: p.detach().clone() for name, p in tower.named_parameters()}
    allocations = vllm_model_vision.map_to_host(tower)
    assert len(allocations) == len(before)
    for (name, param), raw in zip(tower.named_parameters(), allocations, strict=True):
        assert param.data_ptr() == raw.data_ptr(), f"{name} must read the mapped copy"
        assert param.dtype == torch.bfloat16 and torch.equal(param, before[name])


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
