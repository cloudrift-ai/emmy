"""Generative split integration for structural plan-template reuse (CPU-only)."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.backend.plan_cache import PlanTemplateCache
from emmy.compiler.dtype import F16, F32
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.base import ConstantOp, InputOp
from emmy.compiler.ir.cuda import CudaOp


def _split_graph() -> Graph:
    g = Graph()
    g.add_node(op=InputOp(), inputs=[], output=Tensor("x", (2, 4)), node_id="x")
    g.add_node(
        op=ConstantOp(name="weight", source_path="weight", source_shape=(4, 4), source_dtype="f32"),
        inputs=[],
        output=Tensor("weight", (4, 4)),
        node_id="weight",
    )
    g.add_node(
        op=CudaOp(
            kernel_source='extern "C" __global__ void k_split() {}',
            kernel_name="k_split",
            arg_order=("x", "weight", "y"),
            grid=((1,), (1,), (1,)),
            block=((32,), (1,), (1,)),
        ),
        inputs=["x", "weight"],
        output=Tensor("y", (2, 4)),
        node_id="y",
    )
    g.inputs = ["x"]
    g.outputs = ["y"]
    return g


def test_compile_split_rejects_multiple_expert_input_formats():
    from emmy.serving.gen_runner import _compile_split

    with pytest.raises(ValueError, match="expert input formats are mutually exclusive"):
        _compile_split(None, [], None, F16, quant_specs={"weight": object()}, mxfp4_specs={"weight": object()})


@pytest.mark.parametrize("input_dtype", ["float32", "bfloat16"])
def test_compile_split_feeds_bf16_input_as_bits(monkeypatch, input_dtype):
    from types import SimpleNamespace

    import torch

    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.plan import WeightSpec
    from emmy.compiler.dtype import BF16
    from emmy.serving.gen_runner import _compile_split

    class Wrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([1.0, -2.0, 3.14], dtype=torch.bfloat16))

    plan = SimpleNamespace(
        inputs=["x"],
        outputs=["y"],
        weights={"weight": WeightSpec(source_path="weight", graph_dtype="bf16")},
        buffers=[SimpleNamespace(name="x", dtype=BF16), SimpleNamespace(name="weight", dtype=BF16)],
    )
    feeds = []

    def build(_cls, _plan, feed, **_kwargs):
        feeds.append(feed)
        return object()

    monkeypatch.setattr(CompiledProgram, "build_from_plan", classmethod(build))
    x = torch.tensor([1.0, -2.0, 3.14], dtype=getattr(torch, input_dtype))
    _compile_split(Wrapper(), [x], None, F16, plan=plan)

    assert feeds[0]["x"].dtype == np.uint16
    np.testing.assert_array_equal(feeds[0]["x"], [0x3F80, 0xC000, 0x4049])
    assert feeds[0]["weight"].dtype == np.uint16
    np.testing.assert_array_equal(feeds[0]["weight"], [0x3F80, 0xC000, 0x4049])


def test_bind_plan_constants_separates_bf16_and_f32_storage_in_wrapper_cache(monkeypatch):
    from types import SimpleNamespace

    import torch

    from emmy.compiler.backend.plan import WeightSpec
    from emmy.compiler.dtype import BF16, F32
    from emmy.serving.gen_runner import _bind_plan_constants

    source = torch.tensor([1.0, -2.0, 3.140625], dtype=torch.bfloat16).float().numpy()
    plan = SimpleNamespace(
        weights={
            "bf16_weight": WeightSpec(source_path="weight"),
            "f32_weight": WeightSpec(source_path="weight"),
        },
        buffers=[
            SimpleNamespace(name="bf16_weight", dtype=BF16),
            SimpleNamespace(name="f32_weight", dtype=F32),
        ],
    )
    uploads = []

    def fake_cuda(tensor):
        uploads.append(tensor)
        return tensor

    monkeypatch.setattr(torch.Tensor, "cuda", fake_cuda)
    cache = {}
    first = _bind_plan_constants(plan, {"weight": source}, cache)
    second = _bind_plan_constants(plan, {"weight": source}, cache)

    assert len(uploads) == len(cache) == 2
    assert second["bf16_weight"] is first["bf16_weight"]
    assert second["f32_weight"] is first["f32_weight"]
    assert first["bf16_weight"] is not first["f32_weight"]
    assert first["bf16_weight"].numpy().dtype == np.uint16
    np.testing.assert_array_equal(first["bf16_weight"].numpy(), [0x3F80, 0xC000, 0x4049])
    assert first["f32_weight"].numpy().dtype == np.float32
    np.testing.assert_array_equal(first["f32_weight"].numpy(), source)


def test_compile_split_reuses_plan_but_builds_fresh_programs_and_weights(monkeypatch):
    import torch

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.serving import gen_runner

    class Wrapper(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.full((4, 4), value))

        def forward(self, x):
            return x @ self.weight.T

    compile_calls = []
    builds = []

    def compile_graph(_backend, graph, *, ctx=None):
        compile_calls.append(graph)
        return graph

    def build_from_plan(_cls, plan, feed, **_kwargs):
        program = object()
        builds.append((program, plan, dict(feed)))
        return program

    monkeypatch.setattr(gen_runner, "trace_split", lambda *_args, **_kwargs: _split_graph())
    monkeypatch.setattr(CudaBackend, "compile", compile_graph)
    monkeypatch.setattr(CompiledProgram, "build_from_plan", classmethod(build_from_plan))

    cache = PlanTemplateCache()
    x = torch.zeros(2, 4)
    first, plan0 = gen_runner._compile_split(Wrapper(1.0), [x], None, F32, plan_cache=cache)
    second, plan1 = gen_runner._compile_split(Wrapper(2.0), [x], None, F32, plan_cache=cache)

    assert len(compile_calls) == 1
    assert (cache.hits, cache.misses) == (1, 1)
    assert first.program is builds[0][0]
    assert second.program is builds[1][0]
    assert first.program is not second.program
    assert plan0 is not plan1
    assert plan0.weights["weight"].source_path == "weight"
    assert plan1.weights["weight"].source_path == "weight"
    np.testing.assert_array_equal(builds[0][2]["weight"], np.ones((4, 4), dtype=np.float32))
    np.testing.assert_array_equal(builds[1][2]["weight"], np.full((4, 4), 2.0, dtype=np.float32))
