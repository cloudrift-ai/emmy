"""Tensor-parallel routed experts (no GPU, no vLLM).

A DeepSeek V4 rank cannot hold all 256 routed experts whole: one pipeline stage's experts are ~9.4 GB
at MXFP4 against a 32 GB card that also carries attention, arenas and the KV cache. So each rank holds
1/world of EVERY expert, cut along the intermediate axis (``slice_routed_experts``), runs the same
picks, and the group all-reduce sums the ranks' outputs.

These tests pin that the cut is exact — the slices of one expert sum to the whole expert — and the
routing math serving runs, ``combine_routed_experts``.
"""

from __future__ import annotations

import pytest


def _router_return(torch, tokens: int, experts: int, top_k: int, seed: int = 0):
    """One HF-router-shaped ``(scores, indices)`` pair over the GLOBAL expert space."""
    generator = torch.Generator().manual_seed(seed)
    logits = torch.randn(tokens, experts, generator=generator)
    scores, indices = torch.topk(logits.softmax(dim=-1), top_k, dim=-1)
    return scores, indices


def _one_moe_layer(torch, experts: int, hidden: int, inter: int, *, device=None):
    """The smallest model ``slice_routed_experts`` walks: one block whose ``mlp`` has a router and a
    transformers-v5 experts module in the (out, in), concatenated, biasless layout."""
    from types import SimpleNamespace

    from torch import nn

    generator = torch.Generator().manual_seed(5)
    holder = nn.Module()
    holder.gate_up_proj = nn.Parameter(torch.randn(experts, 2 * inter, hidden, generator=generator).to(device or "cpu"))
    holder.down_proj = nn.Parameter(torch.randn(experts, hidden, inter, generator=generator).to(device or "cpu"))
    block = SimpleNamespace(mlp=SimpleNamespace(gate=object(), experts=holder))
    return SimpleNamespace(model=SimpleNamespace(layers=[block]))


def _expert(torch, experts, e, x):
    """The serving expert's algebra (DeepSeek's ``ExpertFFN`` without the clamp): gate/up as chunk
    halves of one projection, SwiGLU, then down."""
    gate, up = torch.nn.functional.linear(x, experts.gate_up_proj[e]).chunk(2, dim=-1)
    return torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, experts.down_proj[e])


def test_the_ranks_slices_of_an_expert_sum_to_the_whole_expert():
    """Gate and up keep the same intermediate rows and down the matching columns, so each rank's
    output is an exact partial of the whole expert's — the sum the all-reduce forms."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import slice_routed_experts

    experts, hidden, inter, world = 3, 8, 12, 4
    x = torch.randn(5, hidden, dtype=torch.float64)
    whole = _one_moe_layer(torch, experts, hidden, inter)
    reference = [_expert(torch, whole.model.layers[0].mlp.experts, e, x.float()).double() for e in range(experts)]
    total = [torch.zeros_like(r) for r in reference]
    for rank in range(world):
        model = _one_moe_layer(torch, experts, hidden, inter)
        slice_routed_experts(model, rank, world)
        cut = model.model.layers[0].mlp.experts
        assert tuple(cut.gate_up_proj.shape) == (experts, 2 * inter // world, hidden)
        assert tuple(cut.down_proj.shape) == (experts, hidden, inter // world)
        for e in range(experts):
            total[e] += _expert(torch, cut, e, x.float()).double()
    for got, want in zip(total, reference, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-5)


def test_a_meta_twin_is_cut_to_the_declared_shapes():
    """The serving-twin capture cuts a weightless (meta) twin: only the declared shapes change."""
    torch = pytest.importorskip("torch")

    from emmy.compiler.trace.huggingface import slice_routed_experts

    model = _one_moe_layer(torch, 4, 64, 2048, device="meta")
    slice_routed_experts(model, 7, 8)
    experts = model.model.layers[0].mlp.experts
    assert experts.gate_up_proj.is_meta and tuple(experts.gate_up_proj.shape) == (4, 512, 64)
    assert tuple(experts.down_proj.shape) == (4, 64, 256)
    with pytest.raises(ValueError, match="not divisible"):
        slice_routed_experts(_one_moe_layer(torch, 1, 8, 6), 0, 4)


def test_a_single_row_routes_without_waiting_on_the_device(monkeypatch):
    """One row's combine reads its picks once and never asks the device how many rows an expert
    got (``unique``, ``where``): each of those is a host wait on every decode step. Its sum is
    bit-identical to the same row routed inside a batch, every pick in ascending order."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import combine_routed_experts

    hidden, experts, top_k = 8, 8, 4
    generator = torch.Generator().manual_seed(3)
    xn = torch.randn(2, hidden, generator=generator)
    weights = [torch.randn(hidden, hidden, generator=generator) for _ in range(experts)]
    gated = _router_return(torch, 2, experts, top_k, seed=4)

    def run_expert(e, rows):
        return rows @ weights[e]

    batch = combine_routed_experts(xn, gated, run_expert)
    launched: list[int] = []

    def run_recorded(e, rows):
        launched.append(e)
        return run_expert(e, rows)

    def refuse(*_args, **_kwargs):
        raise AssertionError("a single row asked the device how its rows are routed")

    monkeypatch.setattr(torch.Tensor, "unique", refuse)
    monkeypatch.setattr(torch, "where", refuse)
    row = combine_routed_experts(xn[:1], tuple(g[:1] for g in gated), run_recorded)

    assert torch.equal(row, batch[:1])
    assert launched == sorted(gated[1][0].tolist())


def test_hash_routing_needs_the_steps_token_ids():
    """A hash router selects experts by token id; the router call must pass the ids through and
    refuse to route without them (silently routing on garbage would serve noise)."""
    pytest.importorskip("torch")

    from emmy.serving.gen_runner import EmmyGenRunner

    seen = []
    moe = {"hash": True, "layer": 7, "gate": lambda xn, ids: seen.append((xn, ids)) or ("logits", "scores", "indices")}
    xn, ids = object(), object()
    assert EmmyGenRunner._route(None, moe, xn, ids) == ("logits", "scores", "indices")
    assert seen == [(xn, ids)]
    with pytest.raises(RuntimeError, match="hash-routed"):
        EmmyGenRunner._route(None, moe, xn, None)

    plain = {"hash": False, "gate": lambda xn: ("l", "s", "i")}
    assert EmmyGenRunner._route(None, plain, xn, None) == ("l", "s", "i")


def test_a_float32_router_scores_float32_rows():
    """DeepSeek V4's reference runtime scores its experts in float32: in float16 a near-tie for the
    last of the top-k flips on rounding alone. That router is kept in float32 and the step's rows
    are cast to meet it; every other router takes the rows as they come."""
    torch = pytest.importorskip("torch")

    from emmy.serving.gen_runner import EmmyGenRunner

    seen = []

    def gate(xn, ids=None):
        seen.append(xn.dtype)
        return ("l", "s", "i")

    xn, ids = torch.zeros(2, 4, dtype=torch.float16), torch.zeros(2, dtype=torch.long)
    assert EmmyGenRunner._route(None, {"hash": False, "router_float32": True, "gate": gate}, xn, None) == ("l", "s", "i")
    EmmyGenRunner._route(None, {"hash": True, "layer": 0, "router_float32": True, "gate": gate}, xn, ids)
    EmmyGenRunner._route(None, {"hash": False, "gate": gate}, xn, None)
    assert seen == [torch.float32, torch.float32, torch.float16]
