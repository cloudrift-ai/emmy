"""A cooperative matrix-vector product reads each lane's run of fp8 weight bytes as one vector load.

At one row a product streams its weight once, so the weight load is the kernel. On a ``coop/v<n>`` band each lane
reads ``n`` adjacent bytes of a K-contiguous weight row: the run reads as one word, each byte decodes straight to
f16, every element of the run reads its block scale at the run's own index, and the run carries no bounds mask
when ``n`` divides K (a mask spells every index as a modulo, which no vector load can cover).
"""

from __future__ import annotations

import re

import numpy as np
import pytest

from emmy.compiler.backend.cuda.backend import CudaBackend
from emmy.compiler.context import Context
from emmy.compiler.dtype import F16, F32, decode_f8
from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.base import InputOp
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.pipeline import CUDA_PASSES, Pipeline
from emmy.compiler.pipeline.search.pins import pinned_knobs
from tests.compiler.helpers import device_compute_capability, requires_cuda

# K divides into 16-byte runs but not into the 128-lane band's 2048-element stride, so the band ends mid-row.
N, K = 256, 3072


def _graph() -> Graph:
    """``out[n] = Σ_k x[k] · f16(scale[n/128, k/128] · decode(w_bits[n, k]))``: a block-scaled fp8 weight's decode
    cone fused into the product, as a width-1 projection reads it."""
    n, k = Var("n"), Var("k")
    block = lambda v: BinaryExpr("/", v, Literal(128, "int"))  # noqa: E731
    cell = (
        Loop(
            Axis("k", K),
            (
                Load("wb", "w_bits", (n, k)),
                Assign("wd", "from_f8e4m3", ("wb",), dtype=F16),
                Load("ws", "w_scale", (block(n), block(k))),
                Assign("w", "multiply", ("ws", "wd"), dtype=F16),
                Load("xv", "x", (k,)),
                Assign("p", "multiply", ("xv", "w"), dtype=F32),
                Accum("sum", "p", axes=("k",)),
            ),
        ),
        Write("out", (n,), "sum"),
    )
    graph = Graph()
    graph.add_node(InputOp(), [], Tensor("w_bits", (N, K), "f8e4m3"), node_id="w_bits")
    graph.add_node(InputOp(), [], Tensor("w_scale", (N // 128, K // 128), "f32"), node_id="w_scale")
    graph.add_node(InputOp(), [], Tensor("x", (K,), "f16"), node_id="x")
    graph.add_node(
        LoopOp(body=Body((Loop(Axis("n", N), cell),)), name="k_fp8_matvec"),
        ["w_bits", "w_scale", "x"],
        Tensor("out", (N,), "f32"),
        node_id="out",
    )
    graph.inputs, graph.outputs = ["w_bits", "w_scale", "x"], ["out"]
    return graph


def _compiled(run: int, cc=(7, 0)):
    with pinned_knobs({"PLACE": "fuse", "WORK": "t128", "REDUCE": f"coop/v{run}", "TILE": "", "STAGE": ""}):
        return Pipeline.build(CUDA_PASSES).run(_graph(), ctx=Context.from_target(cc))


def _kernel(compiled) -> str:
    (src,) = [node.op.kernel_source for node in compiled.nodes.values() if isinstance(node.op, CudaOp)]
    return src[src.index("__global__") :]


@pytest.mark.parametrize(("run", "word"), [(4, "unsigned int"), (8, "uint2"), (16, "uint4")])
def test_a_lane_reads_its_run_of_weight_bytes_as_one_word(run, word):
    src = _kernel(_compiled(run))
    assert re.search(rf"{word} \w+ = \*reinterpret_cast<const {word}\*>\(&w_bits\[", src), "one vector load per run"
    assert "w_bits[" not in src.replace(f"reinterpret_cast<const {word}*>(&w_bits[", ""), "no scalar byte load is left"
    assert f"% {K}" not in src, "a run that divides K needs no bounds mask"
    runs = len(re.findall(rf"reinterpret_cast<const {word}\*>\(&w_bits\[", src))
    assert len(set(re.findall(r"w_scale\[([^\]]*)\]", src))) == runs, "a run inside one 128-wide block reads its scale at one index"


def test_the_decode_converts_each_byte_straight_to_f16():
    src = _kernel(_compiled(16))
    assert "emmy_from_f8e4m3(" in src and "emmy_from_f8e4m3_f32(" not in src


@requires_cuda
@pytest.mark.parametrize("run", [4, 16])
def test_the_run_matches_an_independent_reference(run):
    rng = np.random.default_rng(5)
    bits = rng.integers(0, 256, size=(N, K), dtype=np.uint8)
    bits[(bits & 0x7F) == 0x7F] = 0x38  # no NaN codes: the reference sum would be NaN either way
    scale = rng.uniform(0.5, 2.0, size=(N // 128, K // 128)).astype(np.float32)
    x = rng.standard_normal(K).astype(np.float16)
    weight = (np.repeat(np.repeat(scale, 128, 0), 128, 1) * decode_f8(bits, "f8e4m3").astype(np.float32)).astype(np.float16)
    expected = weight.astype(np.float32) @ x.astype(np.float32)
    result, _ = CudaBackend().run(_compiled(run, device_compute_capability()), input_data={"w_bits": bits, "w_scale": scale, "x": x})
    np.testing.assert_allclose(result.outputs["out"], expected, rtol=1e-3, atol=1e-2)
