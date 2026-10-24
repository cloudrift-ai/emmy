"""The paging program: one step of the pager, as a torch module the compiler lowers like any program.

A paged buffer of ``pages`` pages has ``slots`` device slots. The program takes the marks the inspector
wrote — one per page the coming launches touch — and the pager's tables, and leaves the tables as they
stand after the step: a hit refreshes its slot's age; a miss takes the slot of the least recently used
page nothing marked, is copied there from its host page, and the two pages' table entries swap; a miss
with no slot to take keeps its host entry and is read in place. Every array carries one spare entry past
its end: a store pointed there is skipped (``ScatterOp``), which is how the fixed launch set drops the
work it has no page for, so one compiled program serves every step and a captured graph records it.

The tables, in the order the program takes them (``int64`` unless said otherwise):

- ``marks[pages + 1]`` (``int32``): 1 for every page the launches touch; the inspector's output.
- ``table[pages + 1]``: the address each page is read at — its slot, or its host copy. The page table the
  kernels take.
- ``slot_page[slots + 1]``: the page each slot holds, ``pages`` when none.
- ``page_slot[pages + 1]``: the slot each page sits in, ``slots`` when none.
- ``age[slots + 1]``: the step each slot was last used at.
- ``host[pages]``: each page's host address, fixed at load; the pager adds a spare entry at the spare slot, and the
  copy reads the pages through it.
- ``slot_address[slots]``: each slot's device address, fixed at load.
- ``clock[1]``: the step counter; the program advances it, so every run is a later time than the last.
- ``pages`` (paged, one page per row) and ``slots[slots + 1, elements]``: the pages' bytes and the slots they are
  copied into, by :class:`PageCopy` after the step; the copy is a scatter of a gather over the step's list of
  moved pages, as long as the most pages the launch can touch, skipped past the moves.

The outputs are the tables, the slots and the clock written in place (``Tensor.alias``): nothing is copied
back.

:func:`with_paging` is where the program meets a plan. For every launch that reads or writes a paged buffer
declared with slots, fewer than its pages, it derives that launch's **inspector** from the kernel's own Loop IR —
the statements the index along the paged axis is computed from, re-run to mark the pages the launch touches
— and splices the inspector's launch, over the launch's own sizes, and the paging program's launches ahead of it.
The tables become inputs of the plan named after the buffer (``<name>__table`` …), which the pager lends; the
paging program reads the buffer's pages as ``<name>__view``, a ``[pages, words]`` paged buffer — each page in the
widest word that divides it, since the copy moves bytes — whose page table is the ``host`` table: the copy reads
each page at its home, never through the table the step just pointed at its slot. A buffer declared with no slots takes
neither: nothing can move, so its page table, bound once, points every page at its home.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn


def _ids(x):
    """``0 .. len(x) - 1``, over ``x``'s symbolic length."""
    return torch.arange(x.shape[0])


def _count(x):
    """``len(x)``, as a one-element tensor: the last id, plus one."""
    return _ids(x)[-1:] + 1


class PagingProgram(nn.Module):
    """One pager step's decisions: which slot each hit refreshes, which slot each miss takes, which page each miss
    evicts, and the list of pages the step moves. It reads the tables and writes none, so no kernel of it reads a
    table another of its threads is writing; :class:`PagingUpdate` applies the decisions. Every size is read off
    the tables, so one program serves every geometry: ``pages`` is one less than ``page_slot``'s length, ``slots``
    one less than ``slot_page``'s."""

    def forward(self, marks, slot_page, page_slot, age, clock, moved_page, moved_slot):
        P, S = _count(page_slot) - 1, _count(slot_page) - 1
        page_ids, slot_ids = _ids(page_slot), _ids(slot_page)
        # The spare page is never marked and holds no slot, so it is neither a hit nor a miss.
        marked = marks != 0
        resident = page_slot < S
        # A hit refreshes its slot's age; a miss is a marked page with no slot.
        hit_slot = torch.where(marked & resident, page_slot, S)
        miss = marked & ~resident
        # Each miss's rank among the misses: the misses before it, counted per page so no step waits on the last.
        miss_rank = (miss.unsqueeze(0) & (page_ids.unsqueeze(0) < page_ids.unsqueeze(1))).to(torch.int64).sum(1)
        # The victims: slots whose page is not marked, least recently used first. A slot holding a marked page, and
        # the spare slot, are never victims: their key is pushed past every age.
        held_marked = marks[slot_page] != 0
        blocked = held_marked | (slot_ids == S)
        key = torch.where(blocked, (clock + 1) * (S + 1) + slot_ids, age * (S + 1) + slot_ids)
        victim_rank = (key.unsqueeze(1) > key.unsqueeze(0)).to(torch.int64).sum(1)
        # The slot at each rank: the ranks are a permutation, so each slot stores its id at its rank.
        victim_of_rank = torch.zeros_like(slot_ids).index_copy(0, victim_rank, slot_ids)
        victims = (~blocked).to(torch.int64).sum(0, keepdim=True)
        # The j-th miss takes the j-th victim while victims last; the rest stay in host memory.
        takes = miss & (miss_rank < victims)
        miss_slot = torch.where(takes, victim_of_rank[torch.minimum(miss_rank.clamp(min=0), S)], S)
        evicted_page = torch.where(takes, slot_page[miss_slot], P)
        # The pages the step moves, as a list in miss order, and the slot each moves into: the copy runs over this
        # list, not over every page. The j-th move is the j-th miss, into the j-th victim. An entry past the moves
        # names the spare page and a slot past the spare.
        ranks = _ids(moved_page)
        moves = torch.minimum(miss.to(torch.int64).sum(0, keepdim=True), victims)
        # The page at each rank: each miss that takes a slot stores its page at its rank, every other page at the
        # spare entry; a rank no miss reached keeps the spare page.
        page_of_rank = (torch.zeros_like(page_ids) + P).index_copy(0, torch.where(takes, miss_rank, P), page_ids)
        moved_page = moved_page.index_copy(0, ranks, page_of_rank[ranks])
        moved_slot = moved_slot.index_copy(0, ranks, torch.where(ranks < moves, victim_of_rank[torch.minimum(ranks, S)], S + 1))
        return hit_slot, miss_slot, evicted_page, moved_page, moved_slot


class PagingUpdate(nn.Module):
    """The step's decisions applied to the tables in place. It reads no table it writes — only the decisions, the
    fixed addresses and the clock — so its scatters can share a kernel without one thread reading what another
    writes."""

    def forward(self, table, slot_page, page_slot, age, host, slot_address, clock, hit_slot, miss_slot, evicted_page):
        P, S = _count(page_slot) - 1, _count(slot_page) - 1
        taken_page = torch.where(miss_slot < S, _ids(page_slot), P)
        # The evicted pages lose their slot and point at host memory again; the misses move into their slots.
        table = table.index_copy(0, evicted_page, host[torch.minimum(evicted_page, P - 1)])
        table = table.index_copy(0, taken_page, slot_address[torch.minimum(miss_slot, S - 1)])
        page_slot = page_slot.index_copy(0, evicted_page, torch.zeros_like(evicted_page) + S)
        page_slot = page_slot.index_copy(0, taken_page, miss_slot)
        slot_page = slot_page.index_copy(0, miss_slot, taken_page)
        now = torch.zeros_like(hit_slot) + clock
        age = age.index_copy(0, hit_slot, now)
        age = age.index_copy(0, miss_slot, now)
        # The clock advances in place too: a scatter of the next time into its one cell.
        clock = clock.index_copy(0, torch.zeros(1, dtype=torch.int64), clock + 1)
        return table, slot_page, page_slot, age, clock


class PageCopy(nn.Module):
    """The pager step's copy: every page the step moved into a slot, copied there from its home. It runs over the
    step's list of moved pages, whose length bounds the pages one launch can touch, so its work follows the misses
    and not the pages; an entry past the moves names a slot past the end, so the compiled copy skips the store
    without reading the page."""

    def forward(self, moved_page, moved_slot, pages, slots):
        return slots.index_copy(0, moved_slot, pages[moved_page])


#: The pager's arrays, by the suffix their plan buffer takes after the paged buffer's name, with the program's
#: name for each. ``marks`` is per launch (the inspector writes it); every other one is per paged buffer.
TABLES = ("table", "slot_page", "page_slot", "age", "host", "slot_address", "clock", "slots")

#: What the step decides per page, passed from :class:`PagingProgram` to :class:`PagingUpdate` per launch.
DECISIONS = ("hit_slot", "miss_slot", "evicted_page")

#: The step's list of moved pages, passed from :class:`PagingProgram` to :class:`PageCopy` per launch.
MOVES = ("moved_page", "moved_slot")

#: The paging programs' symbolic sizes; :func:`with_paging` gives them their values per paged buffer and launch.
SYMBOLS = ("n_pages1", "n_slots1", "n_pages", "n_slots", "n_moved", "n_words")

#: The sizes the programs are compiled and their golden rows measured at — one MoE layer's experts as pages: every
#: geometry runs the kernels picked here.
HINTS = dict(zip(SYMBOLS, (65, 33, 64, 32, 8, 1 << 20), strict=True))

_programs: dict[str, tuple[tuple, frozenset[str]]] = {}


def _lower(graph, passes=None):
    """The pager's own kernels, lowered with strict evidence off. The paging programs' rows are in the card's
    hardware golden, but an inspector is a slice of a kernel, with no program a record run could bench, and a kernel
    with nothing to re-bench alone has no row either. Those forks take the prior, under a scope that would refuse
    that for the kernels they serve."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.plan import plan_from_graph  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    with config.strict_evidence_override(False):
        return plan_from_graph(CudaBackend().compile(graph) if passes is None else Pipeline.build(passes).run(graph))


def _paging_plan(word):
    """The paging step's plans — the decisions, the table update, the copy — compiled once per process for pages of
    the dtype ``word``, every size symbolic (:data:`SYMBOLS`, named apart from every buffer): the pages and the slots
    and one more of each (the tables' spare entry), the most pages one step can move and a page's length in words.
    One kernel then serves every geometry, so a golden row measured at one prices them all. The decisions pass to the update under
    :data:`DECISIONS`, the list of moves to the copy under :data:`MOVES`."""
    from torch.export import Dim  # noqa: PLC0415

    from emmy.compiler.pipeline.search.policy.greedy import evidence_reads, note_reads  # noqa: PLC0415
    from emmy.compiler.specialize import rehint_program  # noqa: PLC0415
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    if word.name not in _programs:
        pages1, slots1, pages, slots, moved, words = (Dim(n) for n in SYMBOLS)
        i64, value = torch.int64, torch.from_numpy(np.zeros(1, dtype=word.np)).dtype

        def trace(module, inputs, outputs=()):
            graph = trace_module(
                module, tuple(torch.zeros(*shape, dtype=dt) for _, shape, _, dt in inputs), dynamic_shapes={n: d for n, _, d, _ in inputs}
            )
            for old, new in (
                *zip(graph.inputs, (n for n, *_ in inputs), strict=True),
                *zip(graph.outputs[: len(outputs)], outputs, strict=True),
            ):
                graph.rename_node(old, new)
            return rehint_program(graph, HINTS)

        tables = {
            "marks": ((65,), {0: pages1}, torch.int32),
            "table": ((65,), {0: pages1}, i64),
            "slot_page": ((33,), {0: slots1}, i64),
            "page_slot": ((65,), {0: pages1}, i64),
            "age": ((33,), {0: slots1}, i64),
            "host": ((64,), {0: pages}, i64),
            "slot_address": ((32,), {0: slots}, i64),
            "clock": ((1,), {}, i64),
            **{decision: ((65,), {0: pages1}, i64) for decision in DECISIONS},
            **{move: ((8,), {0: moved}, i64) for move in MOVES},
            "pages": ((64, 16), {0: pages, 1: words}, value),
            "slots": ((33, 16), {0: slots1, 1: words}, value),
        }

        def inputs(*names):
            return tuple((n, *tables[n]) for n in names)

        decide = trace(PagingProgram(), inputs("marks", "slot_page", "page_slot", "age", "clock", *MOVES), DECISIONS)
        update = trace(PagingUpdate(), inputs(*TABLES[:7], *DECISIONS))
        copy = trace(PageCopy(), inputs(*MOVES, "pages", "slots"))
        copy.hints.set("cuda.paged_buffers", (("pages", 0, 1),))
        with evidence_reads() as reads:
            _programs[word.name] = (_lower(decide), _lower(update), _lower(copy)), frozenset(reads)
    plans, reads = _programs[word.name]
    note_reads(reads)  # every program that splices them read what their compile read
    return plans


def _inspector_plan(graph, node, name: str, axis: int, page: int, pages: int):
    """The launch's inspector — the kernel's Loop IR cut to what the index along ``name``'s paged axis is computed
    from, each access replaced by a mark of the page it touches, compiled like any kernel — and the most pages the
    launch can touch: per access, the iterations of the loops its index depends on."""
    from math import prod  # noqa: PLC0415

    from emmy.compiler.dtype import I32  # noqa: PLC0415
    from emmy.compiler.graph import Graph, Tensor  # noqa: PLC0415
    from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
    from emmy.compiler.ir.expr import BinaryExpr, CastExpr, Literal  # noqa: PLC0415
    from emmy.compiler.ir.loop import Load, Loop, LoopOp, Write  # noqa: PLC0415
    from emmy.compiler.ir.stmt import (
        Body,  # noqa: PLC0415
        Let,  # noqa: PLC0415
    )
    from emmy.compiler.ir.tile import TileOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOWERING_PASSES  # noqa: PLC0415

    tile = node.op
    while tile is not None and not isinstance(tile, TileOp):
        tile = tile.source
    if tile is None or tile.loop_body is None:
        raise ValueError(f"launch {node.id!r} reads the paged buffer {name!r} but has no Loop IR to inspect")
    body = tile.loop_body
    marks = f"{node.id}__{name}__marks"
    accesses = [s for s in body.iter() if (isinstance(s, Load) and s.input == name) or (isinstance(s, Write) and s.output == name)]
    # What the indices read, closed over the statements defining it.
    needed = {var for s in accesses for var in s.index[axis].free_vars()}
    changed = True
    while changed:
        changed = False
        for s in body.iter():
            if set(s.defines()) & needed:
                more = set(s.deps()) | {var for expr in s.exprs() for var in expr.free_vars()}
                if more - needed:
                    needed |= more
                    changed = True
    kept = {id(s) for s in body.iter() if set(s.defines()) & needed}

    def mark(stmt):
        page_of = BinaryExpr("//", CastExpr("int", stmt.index[axis]), Literal(page, "int"))
        return (Let("one", Literal(1, "int"), dtype=I32), Write(output=marks, index=(page_of,), value="one"))

    def prune(stmts) -> tuple:
        out: list = []
        for s in stmts:
            if any(s is a for a in accesses):
                out.extend(mark(s))
            elif isinstance(s, Loop):
                inner = prune(s.body)
                # A loop the index does not depend on marks the same pages on every iteration: its body runs once.
                if inner and s.axis.name in needed:
                    out.append(s.with_bodies((Body(inner),)))
                else:
                    out.extend(inner)
            elif id(s) in kept:
                out.append(s)
        return tuple(out)

    def touched(stmts, extents: tuple) -> int:
        count = 0
        for s in stmts:
            if any(s is a for a in accesses):
                count += prod(extents)
            elif isinstance(s, Loop):
                extent = s.axis.extent
                varies = s.axis.name in needed
                if varies and not extent.is_static:
                    return pages
                count += touched(s.body, (*extents, extent.as_static() if varies else 1))
        return count

    inspector = LoopOp(body=Body(prune(body)))
    frag = Graph()
    for buf in inspector.inputs:
        tensor = graph.buffer(buf)
        frag.add_node(InputOp(), [], Tensor(buf, tensor.shape, tensor.dtype), node_id=buf)
    frag.add_node(inspector, list(inspector.inputs), Tensor(marks, (pages + 1,), "i32"), node_id=marks)
    frag.inputs, frag.outputs = list(inspector.inputs), [marks]
    return _lower(frag, LOWERING_PASSES), touched(body, ())


def _declare(plan, name: str, pages: int, slots: int, elements: int, dtype):
    """Give ``plan`` the pager's tables of paged buffer ``name`` — ``pages`` pages of ``elements`` elements of
    ``dtype``, ``slots`` device slots — as inputs named ``<name>__<table>``, and the pages' ``[pages, words]`` view
    ``<name>__view``. Returns the geometry :func:`_step` sizes the paging program by."""
    from emmy.compiler.backend.plan import BufferSpec  # noqa: PLC0415
    from emmy.compiler.dim import Dim  # noqa: PLC0415
    from emmy.compiler.dtype import I16, I32, I64, U8  # noqa: PLC0415

    tables = (
        ("table", (pages + 1,), I64),
        ("slot_page", (slots + 1,), I64),
        ("page_slot", (pages + 1,), I64),
        ("age", (slots + 1,), I64),
        ("host", (pages,), I64),
        ("slot_address", (slots,), I64),
        ("clock", (1,), I64),
        ("slots", (slots + 1, elements), dtype),
    )
    for suffix, table_shape, table_dtype in tables:
        plan.buffers.append(BufferSpec(name=f"{name}__{suffix}", shape=tuple(Dim(d) for d in table_shape), dtype=table_dtype, role="input"))
        plan.inputs.append(f"{name}__{suffix}")
    # The copy moves bytes: it reads and writes each page in the widest word that divides it.
    page_bytes = elements * dtype.nbytes
    word = next(w for w in (I64, I32, I16, U8) if page_bytes % w.nbytes == 0)
    plan.buffers.append(BufferSpec(name=f"{name}__view", shape=(Dim(pages), Dim(page_bytes // word.nbytes)), dtype=word, role="input"))
    plan.paged[f"{name}__view"] = (0, 1, slots)  # the pager's: its table is the host table
    return pages, slots, page_bytes // word.nbytes, word


def _step(plan, prefix: str, name: str, marks: str, geometry, moved: int) -> list:
    """The paging program's launches for paged buffer ``name`` over the page list ``marks``, scoped by ``prefix``:
    at most ``moved`` pages move."""
    from emmy.compiler.backend.plan import splice  # noqa: PLC0415

    pages, slots, words, word = geometry
    decided = {decision: f"{prefix}.{name}.{decision}" for decision in (*DECISIONS, *MOVES)}
    sizes = dict(zip(SYMBOLS, (pages + 1, slots + 1, pages, slots, min(pages, slots, moved), words), strict=True))
    launches = []
    for part, sub in zip(("page", "page.update", "copy"), _paging_plan(word), strict=True):
        renames = {"marks": marks, "pages": f"{name}__view", **{table: f"{name}__{table}" for table in TABLES}, **decided}
        # An output stored over an input takes the input's name: written in place, one buffer.
        alias = {b.name: b.alias for b in sub.buffers}
        for output in sub.outputs:
            root = output
            while alias.get(root) is not None:
                root = alias[root]
            renames[output] = renames[root]
        launches.extend(splice(plan, sub, prefix=f"{prefix}.{name}.{part}", renames=renames, sizes=sizes))
    return launches


def with_paging(plan, graph):
    """``plan`` with the inspector and the paging program spliced ahead of every launch that reads or writes a
    paged buffer declared with one slot or more (``cuda.paged_buffers``'s fourth element). Tables, slots and clock
    enter as inputs named ``<buffer>__<table>``; the buffer's ``[pages, words]`` view is ``<buffer>__view``."""
    from dataclasses import replace  # noqa: PLC0415
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.plan import splice  # noqa: PLC0415

    managed = {name: (axis, page, slots) for name, (axis, page, slots) in plan.paged.items() if slots}
    if not managed:
        return plan
    specs = {b.name: b for b in plan.buffers}
    spliced = replace(
        plan,
        inputs=list(plan.inputs),
        buffers=list(plan.buffers),
        constants=dict(plan.constants),
        runtime_constants=dict(plan.runtime_constants),
        launches=[],
        kernels=dict(plan.kernels),
        weights=dict(plan.weights),
        paged=dict(plan.paged),
        symbolic_hints=dict(plan.symbolic_hints),
    )
    geometry = {}
    for name, (axis, page, slots) in managed.items():
        shape = specs[name].resolve_shape({})
        geometry[name] = _declare(spliced, name, -(-shape[axis] // page), slots, prod(shape) // shape[axis] * page, specs[name].dtype)
    for launch in plan.launches:
        node = graph.nodes[launch.node_id]
        for name, (axis, page, _) in managed.items():
            if f"{name}__pages" not in launch.arg_names:
                continue
            marks = f"{launch.node_id}__{name}__marks"
            inspector, moved = _inspector_plan(graph, node, name, axis, page, geometry[name][0])
            # The inspector reads the plan's own buffers, under their own names.
            renames = {**{buffer: buffer for buffer in inspector.inputs}, inspector.outputs[0]: marks}
            spliced.launches.extend(splice(spliced, inspector, prefix=f"{launch.node_id}.{name}.inspect", renames=renames, zero=(marks,)))
            spliced.launches.extend(_step(spliced, launch.node_id, name, marks, geometry[name], moved))
        spliced.launches.append(launch)
    return spliced


def listed_paging(pages: int, buffers: dict):
    """The paging program alone, for paged buffers whose launches run in an order the host knows (a vision tower's
    blocks), so the host writes their page list and no inspector derives it: a plan whose one input ``marks``
    (``int32[pages + 1]``, 1 for the one page the coming launches touch) moves that page in for every buffer of
    ``buffers``, ``{name: (slots, page shape, dtype)}``, each of ``pages`` pages. Each buffer is a paged input of the
    plan, which the pager binds as it binds a program's (``Pager.bind``)."""
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.plan import BufferSpec, ExecutionPlan  # noqa: PLC0415
    from emmy.compiler.dim import Dim  # noqa: PLC0415
    from emmy.compiler.dtype import I32  # noqa: PLC0415

    plan = ExecutionPlan("cuda", ["marks"], [], [BufferSpec("marks", (Dim(pages + 1),), I32, "input")], {}, {}, [], {})
    for name, (slots, shape, dtype) in buffers.items():
        plan.buffers.append(BufferSpec(name, (Dim(pages), Dim(prod(shape))), dtype, "input"))
        plan.paged[name] = (0, 1, slots)
        plan.launches.extend(_step(plan, "listed", name, "marks", _declare(plan, name, pages, slots, prod(shape), dtype), 1))
    return plan
