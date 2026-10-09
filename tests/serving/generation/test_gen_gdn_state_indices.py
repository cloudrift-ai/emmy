"""Per-forward GDN metadata reads; CPU tensors exercise the host-conversion seam."""

from collections import Counter
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


def test_gdn_state_indices_are_read_once_per_tensor_and_refreshed_each_forward(monkeypatch):
    import torch

    from emmy.serving import vllm_model_gen

    model = vllm_model_gen.EmmyGenHybridModel.__new__(vllm_model_gen.EmmyGenHybridModel)
    torch.nn.Module.__init__(model)
    model.fork_attn = None
    model._is_last_rank = True
    indices = [torch.tensor([0, 2]), torch.tensor([1, 3])]
    bounds = torch.tensor([0, 1, 3])
    lengths = torch.tensor([5, 6])
    metadata = {}
    model.gdn_state = []
    for layer, group in enumerate((0, 0, 1, 1)):
        prefix = str(layer)
        state = (torch.arange(4) + layer * 10).float().reshape(4, 1)
        model.gdn_state.append(SimpleNamespace(prefix=prefix, kv_cache=(state.clone(), state)))
        metadata[prefix] = SimpleNamespace(query_start_loc=bounds, seq_lens=lengths, state_indices_tensor=indices[group])
    model.runner = SimpleNamespace(
        num_layers=4,
        forward_layer_gdn_device=lambda layer, hidden, state, history: hidden + state[0, 0],
        final_norm_device=lambda hidden: hidden,
    )
    monkeypatch.setattr(vllm_model_gen, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))
    reads = Counter()
    tolist = torch.Tensor.tolist

    def read_list(tensor):
        reads[id(tensor)] += 1
        return tolist(tensor)

    monkeypatch.setattr(torch.Tensor, "tolist", read_list)
    hidden = torch.zeros(3, 1)
    positions = torch.arange(3)
    first = model._forward_device(hidden, positions)
    torch.testing.assert_close(first, torch.tensor([[62.0], [70.0], [70.0]]))
    assert [reads[id(tensor)] for tensor in indices] == [1, 1]

    indices[0].copy_(torch.tensor([3, 1]))
    indices[1].copy_(torch.tensor([0, 2]))
    second = model._forward_device(hidden, positions)
    torch.testing.assert_close(second, torch.full((3, 1), 66.0))
    assert [reads[id(tensor)] for tensor in indices] == [2, 2]
