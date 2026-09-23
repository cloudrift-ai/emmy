"""Production classic scheduling obeys the independent-domain contract."""

from dataclasses import replace as dc_replace
from importlib import import_module
from types import SimpleNamespace

import pytest

from emmy.compiler.context import Context
from emmy.compiler.graph import Tensor
from emmy.compiler.ir.axis import Axis, Window
from emmy.compiler.ir.expr import Var
from emmy.compiler.ir.schedule import Reduce, ScheduleContext, ScheduleRefused, Stage, Tile, Work
from emmy.compiler.ir.schedule import schedule as advance_schedule
from emmy.compiler.ir.schedule.catalog import coop_reduce_moves, scalar_tile_moves
from emmy.compiler.ir.schedule.classic import (
    ClassicKernelSite,
    ClassicProblem,
    ClassicScheduleCodec,
    ClassicScheduleContext,
    ReductionSchedule,
)
from emmy.compiler.ir.schedule.classic import refusals as classic
from emmy.compiler.ir.schedule.classic.schedule import output_sweep_works
from emmy.compiler.ir.stmt import Accum, Assign, Body, Load, Loop, Write
from emmy.compiler.ir.tile import OutputSpec, Placement, TileOp
from emmy.compiler.ir.tile.ops import carries_partition
from emmy.compiler.pipeline.fork import iter_leaves
from emmy.compiler.pipeline.passes.lowering.tile._fromloop import fold_from_loop
from tests.compiler.helpers import enumerate_classic_reference
from tests.compiler.terms import contraction, projection

classic_forks = import_module("emmy.compiler.pipeline.passes.lowering.tile.040_schedule").classic_forks


def _signature(codec, schedule) -> tuple[tuple[str, str], ...]:
    return tuple(codec.encode(schedule).items())


def _enumerate_context(context: ScheduleContext):
    for value in advance_schedule(context):
        if isinstance(value, ScheduleContext):
            yield from _enumerate_context(value)
        else:
            yield value


def _row(tile, pins) -> dict[str, str]:
    """Production's pin row: a kernel that consumed a split reads a REDUCE pin without its g<n> half."""
    consumed = carries_partition(tile) or tile.split_consumed
    row = {}
    for family, pairs in (pins or {}).items():
        for key, value in pairs:
            row[key] = "/".join(part for part in value.split("/") if not part.startswith("g")) if consumed and family == "REDUCE" else value
    return row


def _offers(tile, target) -> ClassicProblem:
    """The problem whose sites hold the catalog: no row, no precision policy."""
    return ClassicProblem(tile, target)


def _plain(tile, target):
    """A context over the catalog the sites project, no row and no precision policy."""
    return ClassicScheduleContext(tile, target, _offers(tile, target))


def _context(tile, target, *, pins=None):
    """A context sourced the way production sources it: the pins as the row, the precision policy off."""
    problem = ClassicProblem(
        tile, target, row=_row(tile, pins), allow_f16_accumulate=False, allow_fp8=False, validate_pins=target.validate_pins
    )
    return ClassicScheduleContext(tile, target, problem)


def _reference(tile, target, *, pins=None):
    """Run Algorithm 1(c, p, t) under the same row as production."""
    return enumerate_classic_reference(_context(tile, target, pins=pins))


def _schedule_leaves(tile, name, target):
    """Expand the lazy traversal while retaining its typed schedule leaves."""
    return tuple(iter_leaves(classic_forks(tile, name, {}, target)))


def _pointwise() -> TileOp:
    """``y[n] = x[n] + x[n]`` — one projection site and no reduce axis: the read stays a statement of
    the cell, as the lift leaves it for a root nothing reduces over."""
    n = Axis("n", 8)
    root = projection((), (Load(name="x", input="x", index=(Var("n"),)), Assign("y", "add", ("x", "x"))), ("y",))
    return TileOp(op=root, place=Placement(free=(n,)), axes=(n,))


def _sweep_tile(*sweeps: tuple[Axis, ...]):
    specs = tuple(OutputSpec(Write(output=f"out{i}", index=(), value=f"v{i}"), sweep=sweep) for i, sweep in enumerate(sweeps))
    return SimpleNamespace(family_sites={"TILE": (), "REDUCE": (0,)}, output_specs=specs)


def _sweep_work(*sweeps: tuple[Axis, ...]) -> set[str]:
    """Kernel-level WORK offers for serial reductions with the given output sweeps."""
    tile = _sweep_tile(*sweeps)
    return {work.spell() for work in ClassicKernelSite(SimpleNamespace(tile=tile))._sweep_widths()}


def test_disjoint_output_sweeps_offer_worker_widths() -> None:
    """Sibling output loops can share one lane width without sharing an axis."""
    assert "t128" in _sweep_work((Axis("n", 16),), (Axis("p", 32),))


def test_a_scalar_output_suppresses_sweep_worker_widths() -> None:
    """A scalar sibling would be repeated by lanes, so mixed outputs stay serial."""
    assert _sweep_work((Axis("n", 16),), ()) == set()


def test_a_node_owned_inventory_suppresses_sweep_worker_widths() -> None:
    """A cooperating node's inventory must compose through the ordinary equality relation."""
    tile = _sweep_tile((Axis("n", 16),), (Axis("p", 32),))
    assert not output_sweep_works(tile, Work.parse("t128"))


def _row_sum(k: Axis, n: Axis) -> TileOp:
    """``acc[n] = Σ_k x[k]`` — one reduction site over the kernel's ``k``."""
    root = fold_from_loop(
        Loop(axis=k, body=Body((Load(name="xv", input="x", index=(Var("k"),)), Accum(name="acc", value="xv", op="add", axes=("k",)))))
    )
    return TileOp(op=root, place=Placement(free=(n,)), axes=(n, k))


def _matmul(m: Axis, n: Axis, k: Axis, a=None, **fields) -> TileOp:
    """``acc[m, n] = Σ_k a[m, k] · b[k, n]`` — one contraction site; ``a`` may be a computed edge."""
    a = a if a is not None else Load(name="a_e", input="a", index=(Var("m"), Var("k")))
    root = contraction(k, a, (Load(name="b_e", input="b", index=(Var("k"), Var("n"))), "acc"))
    return TileOp(op=root, place=Placement(free=(m, n)), axes=(m, n, k), **fields)


def test_production_enumeration_is_the_compatible_independent_product() -> None:
    tile = _pointwise()
    target = Context.from_target((12, 0))
    offers = _offers(tile, target)
    codec = ClassicScheduleCodec(_context(tile, target))

    reference = {_signature(codec, schedule) for schedule in _reference(tile, target)}
    leaves = _schedule_leaves(tile, "pointwise", target)

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    # The per-cell form and every register strip dividing the 8-wide axis, ``f8`` included: a map's
    # strip ladder runs past the scalar-contraction one.
    assert {dict(row)["TILE"] for row in reference} == {"", "f2", "f4", "f8"}
    assert len(reference) == offers.bounds[0] == 4
    (materialized,) = leaves[0].expand()
    assert materialized.schedule == leaves[0].schedule
    assert materialized.place == tile.place.on_grid()


def test_a_complete_row_proves_its_singleton_by_selection_alone() -> None:
    tile = _row_sum(Axis("k", 64), Axis("n", 64))
    target = Context.from_target((12, 0))
    offers = _offers(tile, target)
    context = _plain(tile, target)
    codec = ClassicScheduleCodec(context)
    candidates = tuple(enumerate_classic_reference(context))
    wanted = candidates[-1]
    pins = {family: [] for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    for key, value in codec.encode(wanted).items():
        pins[key.partition("@")[0]].append((key, value))
    c = _context(tile, target, pins={family: tuple(values) for family, values in pins.items()})

    site = tile.node_sites[0]
    assert len(candidates) > 1
    assert set(c.problem.node_site(site).nodes) <= set(offers.node_site(site).nodes)  # selects, never adds
    assert tuple(_enumerate_context(c)) == (wanted,)
    assert tuple(enumerate_classic_reference(c)) == (wanted,)


def test_a_swept_map_row_decodes_exactly_with_its_strip_bare() -> None:
    """A map whose store sweeps an axis offers the kernel-level worker widths beside its register
    strips, and its rows spell both (``WORK=t256,TILE=f2``). The strip never carries that inventory
    — it is the sweep's — so the exact decode of a complete row must parse the strip bare, as the
    catalog spells it; folding the WORK into it made every such row equal no enumerated leaf,
    which failed the release audit of the Gemma 4 golden on four QK-norm pieces."""
    m, n = Axis("m", 8), Axis("n", 256)
    root = projection((), (Load(name="x", input="x", index=(Var("m"), Var("n"))), Assign("y", "add", ("x", "x"))), ("y",))
    spec = OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="y"), sweep=(n,))
    tile = TileOp(op=root, place=Placement(free=(m,)), axes=(m, n), output_specs=(spec,))
    target = Context.from_target((12, 0))
    context = _plain(tile, target)
    codec = ClassicScheduleCodec(context)
    rows = {}
    for wanted in enumerate_classic_reference(context):
        row = codec.encode(wanted)
        rows[(row["WORK"], row["TILE"])] = wanted
        complete = {key: row.get(key, "") for key in codec.keys()}
        assert ClassicScheduleCodec(context.narrowed(complete, strict=True)).decode(complete) == wanted, row
    assert ("t256", "f2") in rows and len(rows) == 36


def test_reduction_enumeration_filters_the_independent_product_by_compatibility() -> None:
    tile = _row_sum(Axis("k", 2048), Axis("n", 512))
    target = Context.from_target((12, 0))
    offers = _offers(tile, target)
    context = _plain(tile, target)
    site = context.tile_op.node_sites[0]

    expected_reductions = {Reduce(), *coop_reduce_moves()}
    assert {choice.reduce for choice in offers.node_site(site).nodes if isinstance(choice, ReductionSchedule)} == expected_reductions
    reference = tuple(_reference(tile, target))
    leaves = _schedule_leaves(tile, "reduce", target)
    codec = ClassicScheduleCodec(_context(tile, target))

    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in reference}
    assert len(reference) == len(expected_reductions)
    assert offers.bounds[0] > len(reference)


def test_scalar_contraction_enumeration_is_the_compatible_independent_product(monkeypatch) -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    tile = _matmul(m, n, k)
    target = Context.from_target((12, 0))
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    offers = _offers(tile, target)
    context = _plain(tile, target)
    site = context.tile_op.node_sites[0]

    choices = offers.node_site(site).nodes
    assert {choice.tile for choice in choices if isinstance(choice, ReductionSchedule) and choice.reduce == Reduce()} == set(
        scalar_tile_moves()
    )
    expected_reductions = {Reduce(), *coop_reduce_moves()}
    actual_reductions = {choice.reduce for choice in choices if isinstance(choice, ReductionSchedule) and not choice.tile.is_tiled}
    assert actual_reductions == expected_reductions

    reference = tuple(_reference(tile, target))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in reference}
    assert offers.bounds[0] > len(reference)
    assert {choice.raster.spell() for choice in offers.kernel_site.kernels} == {"", "gm8"}
    assert all(schedule.kernel.raster.is_direct for schedule in reference if not schedule.nodes[site].tile.is_tiled)

    tiled = next(leaf for leaf in leaves if leaf.schedule.nodes[site].tile.is_tiled)
    materialized = tiled.expand()[0]
    assert materialized.materialization.tiles[site].choice == tiled.schedule.nodes[site].tile


def test_overwide_reduction_is_in_the_domain_before_c_restricts_it(monkeypatch) -> None:
    tile = _row_sum(Axis("k", 8), Axis("n", 8))
    target = Context.from_target((12, 0))
    overwide = Reduce.of(coop=128)
    monkeypatch.setattr(classic, "coop_reduce_moves", lambda: [overwide])
    offers = _offers(tile, target)
    c = _context(
        tile,
        target,
        pins={
            "WORK": (("WORK", "t128"),),
            "TILE": (("TILE", ""),),
            "REDUCE": (("REDUCE", "coop"),),
            "STAGE": (("STAGE", ""),),
            "RASTER": (("RASTER", ""),),
        },
    )
    assert any(choice.reduce == overwide for choice in offers.node_site(0).nodes)
    assert len(tuple(_enumerate_context(c))) == 1


def test_multi_channel_contraction_domain_is_per_cell_direct_and_warp_staged() -> None:
    m, n, k = Axis("m", 16), Axis("n", 16), Axis("k", 16)
    root = contraction(
        k,
        Load(name="a_e", input="a", index=(Var("m"), Var("k"))),
        (Load(name="b0_e", input="b0", index=(Var("k"), Var("n"))), "acc0"),
        (Load(name="b1_e", input="b1", index=(Var("k"), Var("n"))), "acc1"),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, k),
        inputs={name: Tensor(name, (16, 16), "f16") for name in ("a", "b0", "b1")},
        outputs={"out": Tensor("out", (16, 16), "f16")},
    )
    target = Context.from_target((12, 0))
    offers = _offers(tile, target)
    c = _plain(tile, target)
    compatible = []
    for pick in c.extensions():
        try:
            compatible.append(c.extend(pick))
        except ScheduleRefused:
            pass

    assert any(not choice.tile.is_tiled for choice in offers.node_site(0).nodes)
    assert any(choice.tile.is_warp for choice in offers.node_site(0).nodes)
    per_cell = [child for child in compatible if not child.schedule.nodes[0].tile.is_tiled]
    warp = [child for child in compatible if child.schedule.nodes[0].tile.is_warp]
    assert per_cell and all(choice.stage.is_direct for child in per_cell for choice in child.schedule.edges.values())
    # The gmem-direct mma leaf folds ONE B out of registers, so no warp choice of a multi-channel
    # node is direct. Staged it is ordinary, and the compute fill stays among its transports beside
    # whatever copy transports the card offers.
    warp_transports = {choice.stage.transport for child in warp for choice in child.schedule.edges.values()}
    assert warp and not any(choice.stage.is_direct for child in warp for choice in child.schedule.edges.values())
    assert "smem" in warp_transports


def test_tensor_core_enumeration_is_the_compatible_independent_product(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 132)
    tile = _matmul(
        m,
        n,
        k,
        inputs={"a": Tensor("a", (128, 132), "f16"), "b": Tensor("b", (132, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    target = Context.from_target((12, 0))
    warp = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x2"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [warp] if warp.atom.name in atoms else [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    offers = _offers(tile, target)
    context = _plain(tile, target)
    site = context.tile_op.node_sites[0]

    warp_choices = tuple(choice for choice in offers.node_site(site).nodes if isinstance(choice, ReductionSchedule) and choice.tile.is_warp)
    assert warp_choices
    assert any(choice.work.kind == "warp" for choice in offers.kernel_site.kernels)

    reference = tuple(_reference(tile, target))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in reference}
    assert offers.bounds[0] > len(reference)
    assert {choice.raster.spell() for choice in offers.kernel_site.kernels} == {"", "gm8"}
    assert all(schedule.kernel.raster.is_direct for schedule in reference if not schedule.nodes[site].tile.is_tiled)

    warp = next(leaf for leaf in leaves if leaf.schedule.nodes[site].tile.is_warp)
    assert warp.expand()[0].materialization.tiles[site].choice == warp.schedule.nodes[site].tile


def test_producer_band_is_a_restricted_kernel_domain_choice(monkeypatch) -> None:
    """Producer bands belong to the fixed kernel factor; edge compatibility admits only TMA."""
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 128)
    tile = _matmul(
        m,
        n,
        k,
        inputs={"a": Tensor("a", (128, 128), "f16"), "b": Tensor("b", (128, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    target = Context.from_target((12, 0))
    plan = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x2"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [plan] if plan.atom.name in atoms else [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage.parse("d2/smem-tma")])
    offers = _offers(tile, target)

    works = {choice.work.spell() for choice in offers.kernel_site.kernels}
    assert {"w2x2", "w2x2+p1", "w2x2+p2"} <= works
    reference = tuple(_reference(tile, target))
    assert any(schedule.kernel.work.producer for schedule in reference)
    assert all(
        all(choice.stage.transport == "smem-tma" for choice in schedule.edges.values())
        for schedule in reference
        if schedule.kernel.work.producer
    )

    monkeypatch.setenv("EMMY_WORK", "w2x2+p1")
    monkeypatch.setenv("EMMY_TILE", plan.spell())
    monkeypatch.setenv("EMMY_REDUCE", "")
    monkeypatch.setenv("EMMY_STAGE", "d2/smem-tma")
    # A site reads the row it is given, never the environment: setting the pins leaves the catalog alone, and
    # only the row the schedule pass builds from them narrows what the sites offer.
    assert _offers(tile, target).kernel_site.kernels == offers.kernel_site.kernels
    leaves = _schedule_leaves(tile, "matmul", target)
    c = _context(
        tile,
        target,
        pins={
            "WORK": (("WORK", "w2x2+p1"),),
            "TILE": (("TILE", plan.spell()),),
            "REDUCE": (("REDUCE", ""),),
            "STAGE": (("STAGE", "d2/smem-tma"),),
        },
    )
    codec = ClassicScheduleCodec(c)
    restricted = tuple(enumerate_classic_reference(c))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in restricted}
    assert restricted and all(schedule.kernel.work.spell() == "w2x2+p1" for schedule in restricted)
    assert leaves[0].expand()[0].workers.producer_warps == 1


def test_schedule_parameters_select_from_the_catalog_without_adding_a_member(monkeypatch) -> None:
    """An exact parameter selects one member of each factor it names and adds none; composition prunes the rest."""
    m, n, k = Axis("m", 128), Axis("n", 128), Axis("k", 128)
    tile = _matmul(
        m,
        n,
        k,
        inputs={"a": Tensor("a", (128, 128), "f16"), "b": Tensor("b", (128, 128), "f16")},
        outputs={"out": Tensor("out", (128, 128), "f16")},
    )
    pinned_plan = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x1"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(
        classic,
        "warp_tile_moves",
        lambda atoms: [pinned_plan] if pinned_plan.atom.name in atoms else [],
    )
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    target = Context.from_target((12, 0))
    catalog = _offers(tile, target)
    site = tile.node_sites[0]
    monkeypatch.setenv("EMMY_WORK", "w2x1")
    monkeypatch.setenv("EMMY_TILE", "mma_m16n8k16_f16_f32/f2x2/k2")
    # A site reads the row it is given, never the environment: the catalog is the same once the pins are set, and
    # only the row the schedule pass builds from them narrows what the sites offer.
    assert _offers(tile, target).node_site(site).nodes == catalog.node_site(site).nodes

    c = _context(
        tile,
        target,
        pins={
            "WORK": (("WORK", "w2x1"),),
            "TILE": (("TILE", "mma_m16n8k16_f16_f32/f2x2/k2"),),
        },
    )
    codec = ClassicScheduleCodec(c)
    assert set(c.problem.node_site(site).nodes) <= set(catalog.node_site(site).nodes)  # selects, never adds
    assert set(c.problem.kernel_site.kernels) <= set(catalog.kernel_site.kernels)
    assert {pick.nodes[0].tile.spell() for pick in c.extensions()} == {"mma_m16n8k16_f16_f32/f2x2/k2"}
    schedules = tuple(_enumerate_context(c))
    assert {schedule.nodes[0].tile.spell() for schedule in schedules} == {"mma_m16n8k16_f16_f32/f2x2/k2"}
    assert {schedule.kernel.work.spell() for schedule in schedules} == {"w2x1"}
    reference = {_signature(codec, schedule) for schedule in enumerate_classic_reference(c)}
    leaves = _schedule_leaves(tile, "matmul", target)
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == reference
    assert reference
    assert all(dict(row)["WORK"] == "w2x1" and dict(row)["TILE"] == "mma_m16n8k16_f16_f32/f2x2/k2" for row in reference)


def test_bare_kernel_parameter_applies_when_scoped_pin_targets_another_kernel() -> None:
    tile = _pointwise()
    target = Context.from_target((12, 0))
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", "w1x1"),)
    pins["TILE"] = (("TILE@map.10/inner", "mma_m16n8k16_f16_f32/f2x2/k2"),)

    c = _context(tile, target, pins=pins)

    assert tuple(enumerate_classic_reference(c)) == ()


def test_union_parameter_ignores_a_global_value_unsupported_by_this_kernel() -> None:
    """A graph-wide pin may target a sibling kernel in a union compile."""
    tile = _pointwise()
    target = dc_replace(Context.from_target((12, 0)), validate_pins=False)
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", "w1x1"),)

    c = _context(tile, target, pins=pins)

    assert tuple(enumerate_classic_reference(c))


def test_schedule_restriction_snapshots_parameter_values() -> None:
    tile = _pointwise()
    target = Context.from_target((12, 0))
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["WORK"] = (("WORK", ""),)
    c = _context(tile, target, pins=pins)
    expected = tuple(enumerate_classic_reference(c))

    pins["WORK"] = (("WORK", "t2"),)

    assert expected
    assert tuple(enumerate_classic_reference(c)) == expected


def test_schedule_restriction_drops_the_structural_split_stage_from_c() -> None:
    # A realized cross-CTA split: the term names ``k``; the kernel's axis table holds the slice
    # (half the parent's extent) and the partition receipt on its ``Window``.
    parent = Axis("k", 2048)
    tile = _row_sum(Axis("k", 1024, window=Window(parent=parent, partition=True)), Axis("n", 512))
    target = Context.from_target((12, 0))
    pins = {family: () for family in ("WORK", "TILE", "REDUCE", "STAGE", "RASTER")}
    pins["REDUCE"] = (("REDUCE", "g2k"),)
    c = _context(tile, target, pins=pins)
    schedules = tuple(enumerate_classic_reference(c))
    site = tile.node_sites[0]

    assert schedules
    assert all(schedule.nodes[site].reduce == Reduce() for schedule in schedules)


def test_staged_edges_are_independent_product_factors(monkeypatch) -> None:
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    tile = _matmul(
        m,
        n,
        k,
        inputs={"a": Tensor("a", (64, 64), "f32"), "b": Tensor("b", (64, 64), "f32")},
        outputs={"out": Tensor("out", (64, 64), "f32")},
    )
    target = Context.from_target((12, 0))
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [Stage.parse("d1/smem-async")])
    offers = _offers(tile, target)
    context = _plain(tile, target)

    assert len(tile.edge_sites) == 2
    assert all({choice.stage.spell() for choice in site.edges} == {"", "d1/smem-async"} for site in offers.node_sites if site.edges)
    reference = tuple(_reference(tile, target))
    leaves = _schedule_leaves(tile, "matmul", target)
    codec = ClassicScheduleCodec(_context(tile, target))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in reference}
    assert offers.bounds[0] > len(reference)
    assert all(len({choice.stage for choice in schedule.edges.values()}) == 1 for schedule in reference)

    staged = next(leaf for leaf in leaves if all(not choice.stage.is_direct for choice in leaf.schedule.edges.values()))
    materialized = staged.expand()[0]
    assert set(materialized.materialization.stages) == set(context.tile_op.edge_sites)
    assert all(stage.choice == staged.schedule.edges[edge].stage for edge, stage in materialized.materialization.stages.items())


def test_compute_fill_edges_remain_independent_product_factors(monkeypatch) -> None:
    monkeypatch.setenv("EMMY_FAST_MATH", "0")
    m, n, k = Axis("m", 64), Axis("n", 64), Axis("k", 64)
    computed_a = projection(
        (), (Load(name="score", input="scores", index=(Var("m"), Var("k"))), Assign(name="prob", op="exp", args=("score",)))
    )
    tile = _matmul(
        m,
        n,
        k,
        a=computed_a,
        inputs={"scores": Tensor("scores", (64, 64), "f16"), "b": Tensor("b", (64, 64), "f16")},
        outputs={"out": Tensor("out", (64, 64), "f16")},
    )
    target = Context.from_target((12, 0))
    warp = Tile.parse("mma_m16n8k16_f16_f32/f1x1", Work.parse("w1x1"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [warp] if warp.atom.name in atoms else [])
    offers = _offers(tile, target)
    context = _plain(tile, target)
    site = context.tile_op.node_sites[0]

    assert all({choice.stage.spell() for choice in site.edges} == {"", "d1/smem", "d2/smem"} for site in offers.node_sites if site.edges)
    reference = tuple(_reference(tile, target))
    leaves = _schedule_leaves(tile, "computed_a", target)
    codec = ClassicScheduleCodec(_context(tile, target))
    assert {_signature(codec, leaf.schedule) for leaf in leaves} == {_signature(codec, schedule) for schedule in reference}
    assert offers.bounds[0] > len(reference)
    warp_schedules = tuple(schedule for schedule in reference if schedule.nodes[site].tile.is_warp)
    assert warp_schedules
    assert all({edge.stage.transport for edge in schedule.edges.values()} == {"smem"} for schedule in warp_schedules)


def _two_root_projection(m: Axis, n: Axis, body, results: tuple[str, ...], outputs: dict, specs: tuple) -> TileOp:
    """Two contraction roots with DISTINCT A operands under one projection — the NVFP4 gate/up shape
    where each channel reads its own quantized activation, spelled over ``body`` / ``results``."""
    k0, k1 = Axis("k0", 132), Axis("k1", 132)

    def channel(k: Axis, a: str, b: str, acc: str):
        a_edge = Load(name=f"{a}_e", input=a, index=(Var("m"), Var(k.name)))
        return contraction(k, a_edge, (Load(name=f"{b}_e", input=b, index=(Var(k.name), Var("n"))), acc))

    root = projection((channel(k0, "a0", "b0", "acc0"), channel(k1, "a1", "b1", "acc1")), body, results)
    shapes = {"a0": (128, 132), "b0": (132, 128), "a1": (128, 132), "b1": (132, 128)}
    return TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, k0, k1),
        inputs={name: Tensor(name, shape, "f16") for name, shape in shapes.items()},
        outputs=outputs,
        output_specs=specs,
    )


def _tiled_root_sets(tile: TileOp, target, monkeypatch) -> set[tuple[int, ...]]:
    """Which root sites each enumerated row output-tiles, over a one-atom catalog."""
    plan = Tile.parse("mma_m16n8k16_f16_f32/f2x2/k2", Work.parse("w2x2"))
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [plan] if plan.atom.name in atoms else [])
    monkeypatch.setattr(classic, "stage_moves", lambda *, warp, ctx=None: [])
    roots = tuple(tile.node_id(edge) for edge in tile.op.operands)
    leaves = _schedule_leaves(tile, "gate_up", target)
    assert leaves
    return {tuple(site for site in roots if leaf.schedule.nodes[site].tile.is_tiled) for leaf in leaves}


def _cooperative_root_sets(tile: TileOp, target, monkeypatch) -> set[tuple[int, ...]]:
    """Which root sites each enumerated row partitions, over one cooperative reduction."""
    cooperative = Reduce.of(coop=4)
    monkeypatch.setattr(classic, "scalar_tile_moves", lambda: [Tile()])
    monkeypatch.setattr(classic, "warp_tile_moves", lambda atoms: [])
    monkeypatch.setattr(classic, "coop_reduce_moves", lambda: [cooperative])
    roots = tuple(tile.node_id(edge) for edge in tile.op.operands)
    leaves = _schedule_leaves(tile, "gate_up", target)
    assert leaves
    return {tuple(site for site in roots if leaf.schedule.nodes[site].reduce == cooperative) for leaf in leaves}


def test_shared_output_projection_refuses_fragment_atoms(monkeypatch) -> None:
    """One output reads both roots, so either fragment would have a serial reduction in its epilogue."""
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import OutputSpec

    m, n = Axis("m", 128), Axis("n", 128)
    tile = _two_root_projection(
        m,
        n,
        (Assign("v", "multiply", ("acc0", "acc1")),),
        ("v",),
        {"out": Tensor("out", (128, 128), "f16")},
        (OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="v")),),
    )
    assert _tiled_root_sets(tile, Context.from_target((12, 0)), monkeypatch) == {()}


def test_shared_output_projection_offers_at_most_one_cooperative_root(monkeypatch) -> None:
    """A cooperative reduction selects the same kernel root as an output tile, so a projection
    that does not partition by root may not offer cooperative plans for two roots at once."""
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import OutputSpec

    m, n = Axis("m", 128), Axis("n", 128)
    tile = _two_root_projection(
        m,
        n,
        (Assign("v", "multiply", ("acc0", "acc1")),),
        ("v",),
        {"out": Tensor("out", (128, 128), "f16")},
        (OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="v")),),
    )
    roots = tuple(tile.node_id(edge) for edge in tile.op.operands)
    assert _cooperative_root_sets(tile, Context.from_target((12, 0)), monkeypatch) == {(), (roots[0],), (roots[1],)}


def test_partitioned_projection_still_offers_both_roots_output_tiled(monkeypatch) -> None:
    """Each output reads its own accumulator: the projection partitions by root, the binder binds
    both output-tiled roots into one kernel, and the offer keeps that row."""
    from emmy.compiler.ir.stmt import Write
    from emmy.compiler.ir.tile import OutputSpec

    m, n = Axis("m", 128), Axis("n", 128)
    tile = _two_root_projection(
        m,
        n,
        (Assign("v0", "copy", ("acc0",)), Assign("v1", "copy", ("acc1",))),
        ("v0", "v1"),
        {"out0": Tensor("out0", (128, 128), "f16"), "out1": Tensor("out1", (128, 128), "f16")},
        (
            OutputSpec(Write(output="out0", index=(Var("m"), Var("n")), value="v0")),
            OutputSpec(Write(output="out1", index=(Var("m"), Var("n")), value="v1")),
        ),
    )
    roots = tuple(tile.node_id(edge) for edge in tile.op.operands)
    assert roots in _tiled_root_sets(tile, Context.from_target((12, 0)), monkeypatch)
    assert roots in _cooperative_root_sets(tile, Context.from_target((12, 0)), monkeypatch)


def test_fragment_domain_includes_the_reduction_in_a_sibling_projection():
    """A matmul result normalized by another reduction is not a straight-line fragment epilogue."""
    from tests.compiler.terms import reduction

    m, n, k, r = Axis("m", 32), Axis("n", 32), Axis("k", 64), Axis("r", 32)
    matmul = contraction(
        k,
        Load(name="av", input="a", index=(Var("m"), Var("k"))),
        (Load(name="bv", input="b", index=(Var("k"), Var("n"))), "acc"),
    )
    statistic = reduction(
        r,
        (Load(name="xv", input="x", index=(Var("m"), Var("r"))),),
        (Assign("sum__v", "multiply", ("xv", "xv")),),
        ("sum",),
    )
    scale = projection((statistic,), (Assign("scale", "rsqrt", ("sum",)),), ("scale",))
    root = projection((matmul, scale), (Assign("y", "multiply", ("acc", "scale")),), ("y",))
    tile = TileOp(
        op=root,
        place=Placement(free=(m, n)),
        axes=(m, n, k, r),
        inputs={name: Tensor(name, shape, "f16") for name, shape in {"a": (32, 64), "b": (64, 32), "x": (32, 32)}.items()},
        outputs={"out": Tensor("out", (32, 32), "f16")},
        output_specs=(OutputSpec(Write(output="out", index=(Var("m"), Var("n")), value="y")),),
    )
    assert not classic._warp_atoms(tile, Context.from_target((8, 9)), matmul)


@pytest.mark.parametrize("heads", [1, 16])
def test_grouped_matvec_does_not_tile_two_axes_of_one_operand(heads):
    """A vector times per-head weights is not a matrix multiply across heads and channels."""
    head, channel, k = Axis("head", heads), Axis("channel", 128), Axis("k", 64)
    root = contraction(
        k,
        Load(name="xv", input="x", index=(Var("k"),)),
        (Load(name="wv", input="w", index=(Var("head"), Var("channel"), Var("k"))), "acc"),
    )
    tile = TileOp(
        op=root,
        place=Placement(free=(head, channel)),
        axes=(head, channel, k),
        inputs={"x": Tensor("x", (64,), "f16"), "w": Tensor("w", (heads, 128, 64), "f16")},
        outputs={"out": Tensor("out", (heads, 128), "f32")},
        output_specs=(OutputSpec(Write(output="out", index=(Var("head"), Var("channel")), value="acc")),),
    )
    assert bool(tile.contractions) == (heads == 1)
    assert bool(tile.family_sites["TILE"]) == (heads == 1)
