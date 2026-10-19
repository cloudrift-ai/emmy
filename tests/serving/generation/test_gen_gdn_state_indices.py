"""Per-step GDN metadata: the request layout comes from vLLM's host copies, the state blocks stay on the device, and
the builder zeroes the blocks of new requests. CPU tensors stand in for both sides."""

from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")


def test_gdn_builder_takes_the_layout_from_the_host_and_zeroes_new_requests():
    """A 1-token prompt is shaped like a decode step; only its sequence length tells it apart. The device copies of
    the token ranges and sequence lengths here disagree with the host ones, so reading them would show."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata

    from emmy.serving.vllm_model_gen import _GdnBackend

    # A decode step, a 1-token prompt, the second chunk of a prompt, and a 3-token prompt.
    query_start_loc = torch.tensor([0, 1, 2, 6, 9], dtype=torch.int32)
    seq_lens = torch.tensor([10, 1, 8, 3], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=torch.zeros(5, dtype=torch.int32),
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.zeros(4, dtype=torch.int32),
        seq_lens_cpu_upper_bound=seq_lens,
        num_reqs=4,
        num_actual_tokens=9,
        max_query_len=4,
        max_seq_len=10,
        block_table_tensor=torch.tensor([[7], [3], [5], [2]], dtype=torch.int32),
        slot_mapping=torch.zeros(9, dtype=torch.int64),
    )
    layers = {name: SimpleNamespace(kv_cache=(torch.ones(8, 2), torch.ones(8, 3))) for name in ("a", "b", "other")}
    builder_cls = _GdnBackend.get_builder_cls()
    builder = builder_cls.__new__(builder_cls)
    builder.kv_cache_spec = None
    builder._blocks = torch.zeros(6, dtype=torch.int64)
    builder.layer_names = ["a", "b"]
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    metadata = builder.build(0, common)
    assert metadata.requests == ((0, 1), (1, 2), (2, 6), (6, 9))
    assert metadata.blocks.dtype == torch.int64 and metadata.blocks.tolist() == [7, 3, 5, 2]
    # In the builder's own buffer, whose address a captured decode step keeps.
    assert metadata.blocks.data_ptr() == builder._blocks.data_ptr()
    assert (metadata.num_decodes, metadata.num_prefills) == (2, 2)
    # Blocks 3 and 2 belong to the new requests: zero in both caches of the group's layers, and nowhere else.
    zeroed = [cache.sum(dim=1).eq(0).nonzero().flatten().tolist() for name in ("a", "b") for cache in layers[name].kv_cache]
    assert zeroed == [[2, 3]] * 4
    assert all(cache.eq(1).all() for cache in layers["other"].kv_cache)


def test_gdn_forward_reads_nothing_back(monkeypatch):
    """Two cache groups, each with its own blocks. Every read of a tensor's values on the host is refused: the layout
    is host data already, and the blocks are only indexed with."""
    import torch

    from emmy.serving import vllm_model_gen

    model = vllm_model_gen.EmmyGenHybridModel.__new__(vllm_model_gen.EmmyGenHybridModel)
    torch.nn.Module.__init__(model)
    model.fork_attn = None
    model._is_last_rank = True
    blocks = [torch.tensor([0, 2]), torch.tensor([1, 3])]
    metadata = {}
    model.gdn_state = []
    for layer, group in enumerate((0, 0, 1, 1)):
        prefix = str(layer)
        state = (torch.arange(4) + layer * 10).float().reshape(4, 1)
        model.gdn_state.append(SimpleNamespace(prefix=prefix, kv_cache=(state.clone(), state)))
        metadata[prefix] = SimpleNamespace(requests=((0, 1), (1, 3)), blocks=blocks[group], verify=(False, False), writes=None)

    def gdn(layer, hidden, state, history, block, out):
        state.index_add_(0, block, torch.ones(1, 1))  # the layer advances the request's state
        torch.add(hidden, state.index_select(0, block)[0, 0], out=out[0])

    model.runner = SimpleNamespace(
        num_layers=4,
        gdn_outputs=lambda layer, hidden: [torch.empty_like(hidden)],
        forward_layer_gdn_device=gdn,
        finish_layer_gdn=lambda layer, out, token_ids: out[0],
        final_norm_device=lambda hidden: hidden,
    )
    monkeypatch.setattr(vllm_model_gen, "get_forward_context", lambda: SimpleNamespace(attn_metadata=metadata))

    def refuse(*_):
        raise AssertionError("a GDN step must not read a tensor back to the host")

    for read in ("tolist", "item", "cpu", "numpy", "__bool__", "__index__", "__int__"):
        monkeypatch.setattr(torch.Tensor, read, refuse)
    out = model._forward_device(torch.zeros(3, 1), torch.arange(3))
    monkeypatch.undo()
    # Request 0 runs on block 0 of layers 0 and 1 and block 1 of layers 2 and 3: (0+1) + (10+1) + (21+1) + (31+1).
    # Request 1 runs on blocks 2, 2, 3, 3: (2+1) + (12+1) + (23+1) + (33+1).
    torch.testing.assert_close(out, torch.tensor([[66.0], [74.0], [74.0]]))
    assert [module.kv_cache[1].flatten().tolist() for module in model.gdn_state] == [
        [1.0, 1.0, 3.0, 3.0],
        [11.0, 11.0, 13.0, 13.0],
        [20.0, 22.0, 22.0, 24.0],
        [30.0, 32.0, 32.0, 34.0],
    ]


def _spec_builder(layers, spec_tokens):
    import torch

    from emmy.serving.vllm_model_gen import _GdnBackend

    builder_cls = _GdnBackend.get_builder_cls()
    builder = builder_cls.__new__(builder_cls)
    builder.kv_cache_spec = None
    builder.num_spec_tokens = spec_tokens
    builder._blocks = torch.zeros(4, dtype=torch.int64)
    builder._writes = torch.zeros(4, 1 + spec_tokens, dtype=torch.int64)
    builder.layer_names = ["a"]
    builder.vllm_config = SimpleNamespace(
        cache_config=SimpleNamespace(mamba_cache_mode="none"),
        compilation_config=SimpleNamespace(static_forward_context=layers),
    )
    return builder


def test_gdn_builder_starts_a_verify_request_from_its_last_accepted_token():
    """Under speculative decoding a request owns ``1 + k`` blocks. A verify request continues from the block of its
    last accepted token, on the device, and keeps the state after each row in its own block; a prompt request (``-1``
    draft tokens) keeps its state in its first block, and a new one starts from zero there."""
    import torch
    from vllm.v1.attention.backend import CommonAttentionMetadata

    # A verify step that accepted 2 of its tokens last step, then a new 2-token prompt.
    query_start_loc = torch.tensor([0, 3, 5], dtype=torch.int32)
    common = CommonAttentionMetadata(
        query_start_loc=query_start_loc,
        query_start_loc_cpu=query_start_loc,
        seq_lens=torch.tensor([12, 2], dtype=torch.int32),
        seq_lens_cpu_upper_bound=torch.tensor([12, 2], dtype=torch.int32),
        num_reqs=2,
        num_actual_tokens=5,
        max_query_len=3,
        max_seq_len=12,
        block_table_tensor=torch.tensor([[4, 5, 6], [1, 2, 3]], dtype=torch.int32),
        slot_mapping=torch.zeros(5, dtype=torch.int64),
    )
    layers = {"a": SimpleNamespace(kv_cache=(torch.ones(8, 2), torch.ones(8, 3)))}
    builder = _spec_builder(layers, spec_tokens=2)
    metadata = builder.build(0, common, num_accepted_tokens=torch.tensor([2, 1]), num_decode_draft_tokens_cpu=torch.tensor([2, -1]))
    assert metadata.verify == (True, False)
    assert metadata.blocks.tolist() == [5, 1]
    assert metadata.writes.tolist() == [[4, 5, 6], [1, 2, 3]]
    assert metadata.writes.data_ptr() == builder._writes.data_ptr()
    assert [cache.sum(dim=1).eq(0).nonzero().flatten().tolist() for cache in layers["a"].kv_cache] == [[1], [1]]


def test_gdn_verify_request_keeps_the_state_after_each_row(monkeypatch):
    """A verify request runs the per-row program on its blocks; a prompt request in the same step runs in place."""
    import torch

    from emmy.serving import vllm_model_gen

    model = vllm_model_gen.EmmyGenHybridModel.__new__(vllm_model_gen.EmmyGenHybridModel)
    torch.nn.Module.__init__(model)
    state = torch.arange(6).float().reshape(6, 1)
    history = -state
    model.gdn_state = [SimpleNamespace(prefix="gdn", kv_cache=(history, state))]

    def step(layer, hidden, state, history, block, out):
        state.index_add_(0, block, torch.ones(1, 1))
        torch.add(hidden, state.index_select(0, block)[0, 0], out=out[0])

    def rows(layer, hidden, state, history, block, writes, out):
        states = state.index_select(0, block)[0, 0] + 10 * torch.arange(1, hidden.shape[0] + 1).float()[:, None]
        state.index_copy_(0, writes, states)
        history.index_copy_(0, writes, -states)
        torch.add(hidden, states, out=out[0])

    model.runner = SimpleNamespace(
        gdn_outputs=lambda layer, hidden: [torch.empty_like(hidden)],
        forward_layer_gdn_device=step,
        forward_layer_gdn_rows_device=rows,
        finish_layer_gdn=lambda layer, out, token_ids: out[0],
    )
    metadata = SimpleNamespace(
        requests=((0, 2), (2, 5)),
        blocks=torch.tensor([0, 4]),
        verify=(False, True),
        writes=torch.tensor([[0, 1, 2], [3, 4, 5]]),
    )
    monkeypatch.setattr(vllm_model_gen, "get_forward_context", lambda: SimpleNamespace(attn_metadata={"gdn": metadata}))
    out = model._forward_gdn(0, torch.zeros(5, 1))
    torch.testing.assert_close(out, torch.tensor([[1.0], [1.0], [14.0], [24.0], [34.0]]))
    torch.testing.assert_close(state[:, 0], torch.tensor([1.0, 1, 2, 14, 24, 34]))
    torch.testing.assert_close(history[:, 0], torch.tensor([0.0, -1, -2, -14, -24, -34]))
