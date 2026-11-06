"""The deploy join key — ``Op.deploy_identity`` over the schedule-free lowered Loop-IR body.

What these pin: (a) the key hashes the canonical lowered body, not the term — kernels whose
epilogues differ only within one compute-unit op cluster (``relu`` vs ``tanh``) share the key
while their term hashes differ, because their schedule evidence transfers; (b) DISCRIMINATION —
a cross-cluster op change moves the key, and so does an extent change (the extent fingerprint,
folded in beside the body digest)."""

from __future__ import annotations

from dataclasses import replace

from emmy.compiler.graph import Graph, Tensor
from emmy.compiler.ir.loop import LoopOp
from emmy.compiler.ir.stmt import Body, parse_body
from emmy.compiler.pipeline import Pipeline


def _tile(body: Body):
    graph = Graph()
    graph.add_node(LoopOp(body=body), [], Tensor("out", (1,)), node_id="out")
    graph.outputs = ["out"]
    return Pipeline.build(["tile/lift"], select=["lift"]).run(graph).nodes["out"].op


def _matmul_tile(epilogue_op: str | None = None, k_extent: int = 128, n_outer: bool = False):
    result, epilogue = ("acc", "") if epilogue_op is None else ("outv", f"\n                outv = {epilogue_op}(acc)")
    outer, inner = ("n in 0..64", "m in 0..32") if n_outer else ("m in 0..32", "n in 0..64")
    return _tile(
        parse_body(
            f"""
        for {outer}:
            for {inner}:
                for k in 0..{k_extent}:
                    xv = load x[m, k]
                    wv = load w[n, k]
                    prod = multiply(xv, wv)
                    acc <- add(acc, prod) over k{epilogue}
                out[m, n] = {result}
        """
        )
    )


def test_cluster_sibling_epilogues_share_the_structural_key_but_not_the_exact_one() -> None:
    relu, tanh = _matmul_tile("relu"), _matmul_tile("tanh")
    assert relu.identity_key(structural=False) != tanh.identity_key(structural=False)
    assert relu.identity_key(with_io=True) == tanh.identity_key(with_io=True), "schedule evidence transfers within a cluster"
    assert relu.identity_key(structural=False, with_io=True) != tanh.identity_key(structural=False, with_io=True), (
        "the exact kernels differ"
    )


def test_cross_cluster_epilogues_key_apart() -> None:
    assert _matmul_tile("relu").identity_key(with_io=True) != _matmul_tile("abs").identity_key(with_io=True)


def test_the_key_is_the_terms_not_the_source_nests() -> None:
    """The derived body binds the free coordinates with the term's own loops, in the order the
    tree declares them, so the source nest's spelling of its parallel loops does not reach the
    key: ``for m: for n`` and ``for n: for m`` lift to one term and one identity."""
    assert _matmul_tile().loop_body == _matmul_tile(n_outer=True).loop_body
    assert _matmul_tile().identity_key(with_io=True) == _matmul_tile(n_outer=True).identity_key(with_io=True)


def test_extent_moves_the_key() -> None:
    assert _matmul_tile(k_extent=128).identity_key(with_io=True) != _matmul_tile(k_extent=64).identity_key(with_io=True)


def test_identity_key_lattice() -> None:
    """One function, one lattice: the named identities are points of ``identity_key``, and each
    flag folds in exactly one fact."""
    tile = _matmul_tile("relu", k_extent=128)
    assert tile.identity_key() == tile.identity_key()
    assert tile.identity_key(with_io=True) == tile.identity_key(with_io=True)
    assert tile.identity_key(with_io=True, with_knobs=True) == tile.identity_key(with_io=True, with_knobs=True)
    base = _matmul_tile("relu", k_extent=128)
    knobbed = replace(base, knobs={**base.knobs, "TILE": "f4"})
    assert knobbed.identity_key(with_io=True) == tile.identity_key(with_io=True), "knobs stay out without the flag"
    assert knobbed.identity_key(with_io=True, with_knobs=True) != tile.identity_key(with_io=True, with_knobs=True)
