"""The paging program: one step of the pager, as a torch module the compiler lowers like any program.

A paged buffer of ``pages`` pages has ``slots`` device slots. The program takes the marks the inspector
wrote — one per page the coming launches touch — and the pager's tables, and leaves the tables as they
stand after the step: a hit refreshes its slot's age; a miss takes the slot of the least recently used
page nothing marked, is copied there from its host page, and the two pages' table entries swap; a miss
with no slot to take keeps its host entry and is read in place. Every array carries one spare entry past
its end: a store pointed there is skipped (``ScatterOp``), which is how the fixed launch set drops the
work it has no page for, so one compiled program serves every step and a captured graph records it.

The tables, in the order the program takes them (``int64`` unless said otherwise):

- ``marks[pages + 1]`` (``int32``): 1 for every page the launches touch; the inspector's output. The step stores
  each page's slot over it (``slots`` for none), which the copy reads.
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
  copied into, by :class:`PageCopy` after the step; the copy is a scatter of a gather, skipped where the step has
  nothing to move.

The outputs are the tables, the slots, the clock and the marks written in place (``Tensor.alias``): nothing is copied
back.

:func:`with_paging` is where the program meets a plan. For every launch that reads or writes a paged buffer
declared with fewer slots than pages, it derives that launch's **inspector** from the kernel's own Loop IR —
the statements the index along the paged axis is computed from, re-run to mark the pages the launch touches
— and splices the inspector's launch and the paging program's launches ahead of it. The tables become
inputs of the plan named after the buffer (``<name>__table`` …), which the pager lends; the paging program
reads the buffer's pages as ``<name>__view``, a ``[pages, elements]`` paged buffer whose page table is the
``host`` table: the copy reads each page at its home, never through the table the step just pointed at its slot.
"""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from torch import nn


class PagingProgram(nn.Module):
    """One pager step's decisions over a buffer of ``pages`` pages and ``slots`` device slots: which slot each hit
    refreshes, which slot each miss takes, which page each miss evicts. It reads the tables and writes none, so no
    kernel of it reads a table another of its threads is writing; :class:`PagingUpdate` applies the decisions."""

    def __init__(self, pages: int, slots: int):
        super().__init__()
        self.pages, self.slots = pages, slots

    def forward(self, marks, slot_page, page_slot, age, clock):
        P, S = self.pages, self.slots
        slot_ids = torch.arange(S)
        marked = marks[:P] != 0
        resident = page_slot[:P] < S
        # A hit refreshes its slot's age; a miss is a marked page with no slot.
        hit_slot = torch.where(marked & resident, page_slot[:P], S)
        miss = marked & ~resident
        miss_rank = torch.cumsum(miss.to(torch.int64), 0) - 1
        # The victims: slots whose page is not marked, least recently used first. A slot holding a marked page is
        # never a victim: its key is pushed past every age. A hit's age counts as this step's.
        held = slot_page[:S]
        held_marked = torch.where(held < P, marks[held.clamp(max=P)] != 0, torch.zeros(S, dtype=torch.bool))
        ages = torch.where(held_marked, clock.expand(S), age[:S])
        key = torch.where(held_marked, (clock + 1) * S + slot_ids, ages * S + slot_ids)
        victim_rank = (key.unsqueeze(1) > key.unsqueeze(0)).to(torch.int64).sum(1)
        # The slot at each rank, the rank order inverted as a one-hot sum; a rank no slot holds reads slot 0.
        ranks = torch.arange(S + 1).unsqueeze(1)
        victim_of_rank = ((victim_rank.unsqueeze(0) == ranks).to(torch.int64) * slot_ids.unsqueeze(0)).sum(1)
        victims = (~held_marked).to(torch.int64).sum(0, keepdim=True)
        # The j-th miss takes the j-th victim while victims last; the rest stay in host memory.
        takes = miss & (miss_rank < victims)
        miss_slot = torch.where(takes, victim_of_rank[miss_rank.clamp(min=0)], S)
        evicted_page = torch.where(takes, slot_page[miss_slot.clamp(max=S)], P)
        return hit_slot, miss_slot, evicted_page


class PagingUpdate(nn.Module):
    """The step's decisions applied to the tables in place. It reads no table it writes — only the decisions, the
    fixed addresses and the clock — so its scatters can share a kernel without one thread reading what another
    writes."""

    def __init__(self, pages: int, slots: int):
        super().__init__()
        self.pages, self.slots = pages, slots

    def forward(self, marks, table, slot_page, page_slot, age, host, slot_address, clock, hit_slot, miss_slot, evicted_page):
        P, S = self.pages, self.slots
        page_ids = torch.arange(P)
        taken_page = torch.where(miss_slot < S, page_ids, P)
        # The evicted pages lose their slot and point at host memory again; the misses move into their slots.
        table = table.index_copy(0, evicted_page, host[evicted_page.clamp(max=P - 1)])
        table = table.index_copy(0, taken_page, slot_address[miss_slot.clamp(max=S - 1)])
        page_slot = page_slot.index_copy(0, evicted_page, torch.full((P,), S, dtype=torch.int64))
        page_slot = page_slot.index_copy(0, taken_page, miss_slot)
        slot_page = slot_page.index_copy(0, miss_slot, taken_page)
        age = age.index_copy(0, hit_slot, clock.expand(P))
        age = age.index_copy(0, miss_slot, clock.expand(P))
        # Each page's slot for this step is stored over the marks: the copy (:class:`PageCopy`) reads it there.
        marks = marks.index_copy(0, page_ids, miss_slot.to(marks.dtype))
        # The clock advances in place too: a scatter of the next time into its one cell.
        clock = clock.index_copy(0, torch.zeros(1, dtype=torch.int64), clock + 1)
        return table, slot_page, page_slot, age, clock, marks


class PageCopy(nn.Module):
    """The pager step's copy: every page the step moved into a slot, copied there from its home. It is a program of its
    own because the copy runs one thread per element of every page and must read each page's slot, which the step
    stored over the marks, rather than compute it: inside one program, maximal fusion recomputed the victim ranking
    for every element copied."""

    def __init__(self, pages: int, slots: int):
        super().__init__()
        self.pages, self.slots = pages, slots

    def forward(self, marks, pages, slots):
        moved = marks[: self.pages].to(torch.int64)
        return slots.index_copy(0, moved, pages[torch.where(moved < self.slots, torch.arange(self.pages), self.pages)])


#: The pager's arrays, by the suffix their plan buffer takes after the paged buffer's name, with the program's
#: name for each. ``marks`` is per launch (the inspector writes it); every other one is per paged buffer.
TABLES = ("table", "slot_page", "page_slot", "age", "host", "slot_address", "clock", "slots")

#: What the step decides, passed from :class:`PagingProgram` to :class:`PagingUpdate` per launch.
DECISIONS = ("hit_slot", "miss_slot", "evicted_page")

_programs: dict[tuple, object] = {}


def _lower(graph, passes=None):
    """The pager's own kernels, lowered with strict evidence off. Nothing measures them: the step is bound by the
    link its copies cross, not by a schedule, and an inspector is a slice of a kernel, with no program a record run
    could bench. Their forks take what a kernel nothing measured takes — a row when the tune DB holds one, else
    the prior — under a scope that would refuse that for the kernels they serve."""
    from emmy import config  # noqa: PLC0415
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415
    from emmy.compiler.backend.plan import plan_from_graph  # noqa: PLC0415
    from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

    with config.strict_evidence_override(False):
        return plan_from_graph(CudaBackend().compile(graph) if passes is None else Pipeline.build(passes).run(graph))


def _paging_plan(pages: int, slots: int, elements: int, dtype):
    """The paging step's plans for one geometry — the decisions, the table update, the copy — compiled once per
    process. The decisions pass to the update under :data:`DECISIONS`."""
    from emmy.compiler.trace.torch import trace_module  # noqa: PLC0415

    key = (pages, slots, elements, dtype.name)
    if key not in _programs:
        i64 = torch.int64
        examples = (
            torch.zeros(pages + 1, dtype=torch.int32),
            torch.zeros(pages + 1, dtype=i64),
            torch.zeros(slots + 1, dtype=i64),
            torch.zeros(pages + 1, dtype=i64),
            torch.zeros(slots + 1, dtype=i64),
            torch.zeros(pages, dtype=i64),
            torch.zeros(slots, dtype=i64),
            torch.zeros(1, dtype=i64),
        )
        decide = trace_module(PagingProgram(pages, slots), (examples[0], *examples[2:5], examples[7]))
        for old, new in (
            *zip(decide.inputs, ("marks", "slot_page", "page_slot", "age", "clock"), strict=True),
            *zip(decide.outputs, DECISIONS, strict=True),
        ):
            decide.rename_node(old, new)
        decisions = tuple(torch.zeros(pages, dtype=i64) for _ in DECISIONS)
        update = trace_module(PagingUpdate(pages, slots), (*examples, *decisions))
        for old, new in zip(update.inputs, ("marks", *TABLES[:7], *DECISIONS), strict=True):
            update.rename_node(old, new)
        value = torch.from_numpy(np.zeros(1, dtype=dtype.np)).dtype
        copy = trace_module(
            PageCopy(pages, slots), (examples[0], torch.zeros(pages, elements, dtype=value), torch.zeros(slots + 1, elements, dtype=value))
        )
        for old, new in zip(copy.inputs, ("marks", "pages", "slots"), strict=True):
            copy.rename_node(old, new)
        copy.hints.set("cuda.paged_buffers", (("pages", 0, 1),))
        _programs[key] = (_lower(decide), _lower(update), _lower(copy))
    return _programs[key]


def _inspector_plan(graph, node, name: str, axis: int, page: int, pages: int):
    """The launch's inspector: the kernel's Loop IR cut to what the index along ``name``'s paged axis is computed
    from, each access replaced by a mark of the page it touches, compiled like any kernel."""
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
                if inner:
                    out.append(s.with_bodies((Body(inner),)))
            elif id(s) in kept:
                out.append(s)
        return tuple(out)

    inspector = LoopOp(body=Body(prune(body)))
    frag = Graph()
    for buf in inspector.inputs:
        tensor = graph.buffer(buf)
        frag.add_node(InputOp(), [], Tensor(buf, tensor.shape, tensor.dtype), node_id=buf)
    frag.add_node(inspector, list(inspector.inputs), Tensor(marks, (pages + 1,), "i32"), node_id=marks)
    frag.inputs, frag.outputs = list(inspector.inputs), [marks]
    return _lower(frag, LOWERING_PASSES)


def with_paging(plan, graph):
    """``plan`` with the inspector and the paging program spliced ahead of every launch that reads or writes a
    paged buffer declared with slots (``cuda.paged_buffers``'s fourth element). Tables, slots and clock enter as
    inputs named ``<buffer>__<table>``; the buffer's ``[pages, elements]`` view is ``<buffer>__view``."""
    from dataclasses import replace  # noqa: PLC0415
    from math import prod  # noqa: PLC0415

    from emmy.compiler.backend.plan import BufferSpec  # noqa: PLC0415
    from emmy.compiler.dim import Dim  # noqa: PLC0415
    from emmy.compiler.dtype import I64  # noqa: PLC0415

    managed = {name: (axis, page, slots) for name, (axis, page, slots) in plan.paged.items() if slots is not None}
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
    )
    geometry: dict[str, tuple[int, int, int]] = {}
    for name, (axis, page, slots) in managed.items():
        shape = specs[name].resolve_shape({})
        pages, elements = -(-shape[axis] // page), prod(shape) // shape[axis] * page
        geometry[name] = (pages, slots, elements)
        tables = (
            ("table", (pages + 1,), I64),
            ("slot_page", (slots + 1,), I64),
            ("page_slot", (pages + 1,), I64),
            ("age", (slots + 1,), I64),
            ("host", (pages,), I64),
            ("slot_address", (slots,), I64),
            ("clock", (1,), I64),
            ("slots", (slots + 1, elements), specs[name].dtype),
        )
        for suffix, table_shape, dtype in tables:
            spliced.buffers.append(
                BufferSpec(name=f"{name}__{suffix}", shape=tuple(Dim(d) for d in table_shape), dtype=dtype, role="input")
            )
            spliced.inputs.append(f"{name}__{suffix}")
        view = BufferSpec(name=f"{name}__view", shape=(Dim(pages), Dim(elements)), dtype=specs[name].dtype, role="input")
        spliced.buffers.append(view)
        spliced.paged[f"{name}__view"] = (0, 1, slots)  # the pager's: its table is the host table
    for launch in plan.launches:
        node = graph.nodes[launch.node_id]
        for name, (axis, page, _) in managed.items():
            if f"{name}__pages" not in launch.arg_names:
                continue
            pages, slots, elements = geometry[name]
            marks = f"{launch.node_id}__{name}__marks"
            inspector = _inspector_plan(graph, node, name, axis, page, pages)
            # The inspector reads the plan's own buffers, under their own names.
            renames = {**{buffer: buffer for buffer in inspector.inputs}, inspector.outputs[0]: marks}
            spliced.launches.extend(_splice(spliced, inspector, prefix=f"{launch.node_id}.{name}.inspect", renames=renames, zero=(marks,)))
            decided = {decision: f"{launch.node_id}.{name}.{decision}" for decision in DECISIONS}
            for part, sub in zip(("page", "page.update", "copy"), _paging_plan(pages, slots, elements, specs[name].dtype), strict=True):
                renames = {"marks": marks, "pages": f"{name}__view", **{table: f"{name}__{table}" for table in TABLES}, **decided}
                # An output stored over an input takes the input's name: written in place, one buffer.
                alias = {b.name: b.alias for b in sub.buffers}
                for output in sub.outputs:
                    root = output
                    while alias.get(root) is not None:
                        root = alias[root]
                    renames[output] = renames[root]
                spliced.launches.extend(_splice(spliced, sub, prefix=f"{launch.node_id}.{name}.{part}", renames=renames))
        spliced.launches.append(launch)
    return spliced


def _splice(plan, sub, *, prefix: str, renames: dict[str, str], zero: tuple[str, ...] = ()) -> list:
    """Merge ``sub``'s buffers, constants, weights and kernels into ``plan`` under ``prefix`` — except the buffers
    ``renames`` maps onto the plan's own — and return its launches with every name mapped."""
    from dataclasses import replace  # noqa: PLC0415

    names = {b.name: renames.get(b.name, f"{prefix}.{b.name}") for b in sub.buffers}
    existing = {b.name for b in plan.buffers}
    for buffer in sub.buffers:
        name = names[buffer.name]
        if name in existing:
            continue
        role = "constant" if buffer.role == "constant" else "scratch"
        alias = None if buffer.alias is None or names[buffer.alias] == name else names[buffer.alias]
        plan.buffers.append(replace(buffer, name=name, role=role, alias=alias))
        existing.add(name)
    plan.constants.update({names[n]: v for n, v in sub.constants.items()})
    plan.runtime_constants.update({names[n]: v for n, v in sub.runtime_constants.items()})
    plan.weights.update({names[n]: w for n, w in sub.weights.items()})
    plan.paged.update({names[n]: paging for n, paging in sub.paged.items() if names[n] not in plan.paged})
    # Launches resolve kernels by name and the first source wins: a kernel the paging program of another
    # geometry already brought in under the same name takes a suffix from its source.
    kernels: dict[str, str] = {}
    for kernel, spec in sub.kernels.items():
        name = kernel
        if kernel in plan.kernels and plan.kernels[kernel] != spec:
            name = f"{kernel}_{hashlib.sha1(spec.source.encode()).hexdigest()[:8]}"
            spec = replace(spec, source=spec.source.replace(kernel, name))
        plan.kernels[name] = spec
        kernels[kernel] = name

    def bound(n: str) -> str:
        return f"{names[n[: -len('__pages')]]}__pages" if n.endswith("__pages") else names[n]

    launches = []
    for launch in sub.launches:
        descriptors = {t.name for t in launch.tma_descriptors}
        launches.append(
            replace(
                launch,
                node_id=f"{prefix}.{launch.node_id}",
                kernel_name=kernels[launch.kernel_name],
                arg_names=tuple(n if n in descriptors else bound(n) for n in launch.arg_names),
                tma_descriptors=tuple(replace(t, src_buf=names[t.src_buf]) for t in launch.tma_descriptors),
                zero_outputs=tuple(dict.fromkeys((*(names[n] for n in launch.zero_outputs), *zero))),
                zero_prologues=tuple(names[n] for n in launch.zero_prologues),
                writes=tuple(names[n] for n in launch.writes),
            )
        )
    return launches
