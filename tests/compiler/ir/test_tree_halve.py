"""The cross-thread smem combine's rendered forms and its rewrite."""

from emmy.compiler.ir.kernel.ir import TreeHalve
from emmy.compiler.ir.sigma import Sigma
from emmy.compiler.ir.stmt import Assign, RenderCtx
from emmy.compiler.ir.stmt.passes import _rewrite_kind

_SUM = (Assign(name="acc0", op="add", args=("acc0", "acc0__o")),)


def _render(halve: TreeHalve) -> str:
    return "\n".join(halve.render(RenderCtx()))


def test_the_warp_partials_fold_in_one_warp_behind_one_barrier() -> None:
    """Sixteen warp partials are a register butterfly for warp 0, not four barrier-separated halvings."""
    src = _render(TreeHalve(bufs=("acc0_smem",), state=("acc0",), state_b=("acc0__o",), combine_states=_SUM, length=16, tid_var="warp"))
    assert "if (warp == 0) {" in src and "acc0_smem[lane & 15]" in src
    assert src.count("__syncthreads();") == 1 and "for (int s" not in src
    assert src.rstrip().endswith("acc0 = acc0_smem[0];")


def test_a_block_slab_keeps_the_halving_tree() -> None:
    """A slab of per-thread partials (not per-warp) is wider than a warp and keeps the tree."""
    src = _render(TreeHalve(bufs=("acc0_smem",), state=("acc0",), state_b=("acc0__o",), combine_states=_SUM, length=64, tid_var="t"))
    assert "for (int s = 32; s > 0; s >>= 1)" in src


def test_a_rewrite_keeps_the_segment_layout() -> None:
    """The transposed combine halves segment by segment; losing ``inner`` in a rewrite folds across outputs."""
    halve = TreeHalve(bufs=("b",), state=("acc0",), state_b=("acc0__o",), combine_states=_SUM, length=4, tid_var="k_co", inner=("n_ln", 32))
    assert _rewrite_kind(halve, lambda name: name, Sigma({}), lambda axis: axis).inner == ("n_ln", 32)
