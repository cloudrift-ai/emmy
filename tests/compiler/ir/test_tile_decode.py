"""A ``Tile``'s decode: each axis read from the built-ins under the launch's ranges, 64-bit only where a value can pass
2^31 - 1.

A per-cell coop-reduce grid can exceed 2^31 total threads (gemma-4-12B's M=4096 down-proj demote: 4096 * 3840 * 256 =
4.03e9); a 32-bit flat id then wraps, every decoded index goes wild, and the first launch dies with
``cudaErrorIllegalAddress`` (the M=4096 prefill-chunk twin crash). Read from ``blockIdx.x`` alone, the block axes of
that grid stay 32-bit; arithmetic that can still pass 2^31 - 1 (the guard of such a grid, a symbolic size) is 64-bit."""

from emmy.compiler.dim import Dim
from emmy.compiler.ir.axis import Axis
from emmy.compiler.ir.expression import BinaryExpression, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Sync, Tile, decode_tile
from emmy.compiler.ir.kernel.render import _BUILTIN_TO_CUDA
from emmy.compiler.ir.render import RenderCtx, render_stmt
from emmy.compiler.ir.stmt import Body, Cond, Let, pretty_body


def _render(axes, **kw) -> str:
    return "\n".join(render_stmt(Tile(axes=axes, body=Body(()), **kw), RenderCtx(builtins=_BUILTIN_TO_CUDA)))


def test_a_grid_past_int32_reads_its_block_axes_from_the_cta_index():
    src = _render((Axis("m", 4096), Axis("n", 3840), Axis("k_co", 256)), block_threads=256)
    assert "int m = blockIdx.x / 3840;" in src
    assert "int n = blockIdx.x % 3840;" in src
    assert "int k_co = threadIdx.x;" in src
    # 256 divides the cells: no guard, and nothing left to pass 2^31 - 1.
    assert "if (" not in src and "long long" not in src


def test_a_thread_axis_straddling_ctas_reads_both_indices():
    src = _render((Axis("m", 64), Axis("n", 32)))
    assert "int m = blockIdx.x * 8 + threadIdx.x / 32;" in src
    assert "int n = threadIdx.x % 32;" in src
    assert "long long" not in src


def test_the_producer_band_wraps_onto_the_cells():
    src = _render((Axis("m", 4096), Axis("n", 3840), Axis("k_co", 256)), block_threads=256, aux_threads=128)
    assert "int m = blockIdx.x / 3840;" in src
    assert "int k_co = threadIdx.x % 256;" in src


def test_a_guard_past_int32_is_64_bit():
    src = _render((Axis("m", 4097), Axis("n", 3841), Axis("k", 255)), block_threads=256)
    (guard,) = [line for line in src.splitlines() if line.lstrip().startswith("if (")]
    assert "((long long)(blockIdx.x)) * 256" in guard


def test_a_symbolic_grid_guards_in_64_bits_and_decodes_its_block_axes_in_32():
    src = _render((Axis("s", Dim(Var("seq_len"))), Axis("n", 6144), Axis("k_co", 256)), block_threads=256)
    (guard,) = [line for line in src.splitlines() if line.lstrip().startswith("if (")]
    assert "((long long)(seq_len))" in guard
    assert "int s = blockIdx.x / 6144;" in src
    assert "int n = blockIdx.x % 6144;" in src


def test_a_one_cta_grid_reads_the_thread_index_alone():
    src = _render((Axis("m", 100),))
    assert "if (threadIdx.x < 100) {" in src
    assert "int m = threadIdx.x;" in src


def test_the_decode_binds_the_thread_coordinates_the_body_reads():
    """A body that reads a thread's lane gets it as a value of the grid, under the launch's ranges; one that reads
    neither coordinate gets neither."""
    reads_lane = Body((Cond(cond=BinaryExpression("==", Var("lane"), Literal(0, "int")), body=(Sync(),)),))
    grid = decode_tile(Tile(axes=(Axis("m", 4), Axis("t", 64)), body=reads_lane, block_threads=64))
    assert [stmt.name for stmt in grid.body if isinstance(stmt, Let)] == ["lane"]
    assert "let i32 lane = thread_idx.x % 32" in "\n".join(pretty_body(grid.body))
    # One warp a CTA: the optimized body knows the lane is the thread index.
    one_warp = KernelOp(body=Body((decode_tile(Tile(axes=(Axis("m", 4), Axis("t", 32)), body=reads_lane, block_threads=32)),)))
    assert "let i32 lane = thread_idx.x" in "\n".join(pretty_body(one_warp.body))
    bare = decode_tile(Tile(axes=(Axis("m", 4), Axis("t", 64)), body=Body((Sync(),)), block_threads=64))
    assert not any(isinstance(stmt, Let) for stmt in bare.body)
