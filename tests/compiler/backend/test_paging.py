"""The paging step (``backend/paging.py``): the table program then the copy, each compiled like any program.

The reference is the module run eagerly: every array carries a spare entry (the slots two: the copy drops an unused
move one row past the spare), so torch writes the dropped work there, and the two agree on every real entry. Steps feed
their tables back, so the sequence exercises hits, misses with a free slot, evictions, and misses past the
slots that stay in host memory.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from emmy.compiler.backend.paging import DECISIONS, MOVES, PageCopy, PagingProgram, PagingUpdate
from tests.compiler.helpers import requires_cuda

PAGES, SLOTS, ELEMS, MOVED = 12, 3, 4, 4
NAMES = ("marks", "table", "slot_page", "page_slot", "age", "host", "slot_address", "clock", "pages", "slots")


class _Step(torch.nn.Module):
    """One pager step as one module — the decisions, the table update, the copy — which the plan splices in turn."""

    def __init__(self):
        super().__init__()
        self.decide, self.update, self.copy = PagingProgram(), PagingUpdate(), PageCopy()

    def forward(self, marks, table, slot_page, page_slot, age, host, slot_address, clock, pages, slots):
        moves = torch.zeros(MOVED, dtype=torch.int64)
        *decisions, moved_page, moved_slot = self.decide(marks, slot_page, page_slot, age, clock, moves, moves.clone())
        table, slot_page, page_slot, age, clock = self.update(table, slot_page, page_slot, age, host, slot_address, clock, *decisions)
        return table, slot_page, page_slot, age, self.copy(moved_page, moved_slot, pages, slots), clock


def _state(resident=0):
    """Fresh tables: no page in a slot, every page at its home (small integers, exact in f32), the first ``resident``
    on the device in no slot (``page_slot`` -1)."""
    home = 1000 + torch.arange(PAGES + 1, dtype=torch.int64) - 500 * (torch.arange(PAGES + 1) < resident)
    return {
        "table": home.clone(),
        "slot_page": torch.full((SLOTS + 1,), PAGES, dtype=torch.int64),
        "page_slot": torch.where(torch.arange(PAGES + 1) < resident, -1, SLOTS),
        "age": torch.zeros(SLOTS + 1, dtype=torch.int64),
        "host": home[:PAGES].clone(),
        "slot_address": 5000 + torch.arange(SLOTS, dtype=torch.int64),
        "slots": torch.zeros(SLOTS + 2, ELEMS, dtype=torch.float32),
    }


def _marks(rng, step: int):
    marks = torch.zeros(PAGES + 1, dtype=torch.int32)
    marks[torch.as_tensor(rng.choice(PAGES, size=2 + step % 3, replace=False))] = 1
    return marks


def _pages():
    return torch.arange(PAGES + 1, dtype=torch.float32).unsqueeze(1).repeat(1, ELEMS) + torch.arange(ELEMS) / 10


def _inputs(state, marks, step, pages):
    moves = {move: torch.zeros(MOVED, dtype=torch.int64) for move in MOVES}
    return {**state, **moves, "marks": marks, "clock": torch.tensor([step + 1], dtype=torch.int64), "pages": pages}


def _check(got: dict, expected: tuple):
    for (name, value), reference in zip(got.items(), expected, strict=True):
        real = PAGES if reference.shape[0] == PAGES + 1 else SLOTS
        np.testing.assert_array_equal(np.asarray(value).reshape(reference.shape)[:real], reference.numpy()[:real], err_msg=name)


@pytest.mark.parametrize("resident", [0, 3])
def test_the_paging_programs_match_their_eager_reference_over_a_sequence_of_steps(resident):
    """The three programs compiled one by one and run in turn, as the plan splices them, against the eager step. A
    resident page is a hit in no slot: it never takes a slot or leaves its home."""
    from emmy.compiler.backend.loop.backend import LoopBackend
    from emmy.compiler.trace.torch import trace_module

    step, rng, pages, backend = _Step(), np.random.default_rng(0), _pages(), LoopBackend()
    state = _state(resident)
    feed = _inputs(state, _marks(rng, 0), 0, pages)
    decide_args = ("marks", "slot_page", "page_slot", "age", "clock", *MOVES)
    update_args = (
        "table",
        "slot_page",
        "page_slot",
        "age",
        "host",
        "slot_address",
        "clock",
        "hit_slot",
        "miss_slot",
        "evicted_page",
    )
    produced = {"decide": (*DECISIONS, *MOVES), "update": ("table", "slot_page", "page_slot", "age", "clock"), "copy": ("slots",)}
    decisions = dict(zip(produced["decide"], step.decide(*(feed[n] for n in decide_args)), strict=True))
    parts = (("decide", decide_args), ("update", update_args), ("copy", (*MOVES, "pages", "slots")))
    programs = [
        (part, backend.compile(trace_module(getattr(step, part), tuple({**feed, **decisions}[n] for n in names))), names)
        for part, names in parts
    ]
    for i in range(5):
        feed = _inputs(state, _marks(rng, i), i, pages)
        expected = step(*(feed[name] for name in NAMES))
        values = {name: value.numpy() for name, value in feed.items()}
        for part, compiled, names in programs:
            result, _ = backend.run(compiled, input_data={name: values[name] for name in names})
            values.update(zip(produced[part], (result.outputs[o] for o in compiled.outputs), strict=True))
        got = {name: values[name] for name in ("table", "slot_page", "page_slot", "age", "slots", "clock")}
        _check(got, expected)
        state = dict(zip(("table", "slot_page", "page_slot", "age", "slots", "clock"), expected[:6], strict=True))
        state.update(host=feed["host"], slot_address=feed["slot_address"])
    # The sequence exercised an eviction: a slot held a page and then another, never a resident one.
    assert (state["slot_page"][:SLOTS] < PAGES).all() and (state["slot_page"][:SLOTS] >= resident).all()
    assert (state["table"][:resident] == _state(resident)["table"][:resident]).all()
    assert (state["page_slot"][:resident] == -1).all()


def test_a_miss_never_evicts_the_slot_of_a_page_the_same_step_hits():
    """Slots full, one marked page resident and one not: the miss takes the least recently used slot whose page
    nothing marked, never the hit's — however recent the hit's age is next to the others."""
    program, pages, state = _Step(), _pages(), _state()
    held = []
    for step, picks in enumerate(([5, 7], [4, 5], [1, 2], [1, 7], [1, 4])):
        marks = torch.zeros(PAGES + 1, dtype=torch.int32)
        marks[picks] = 1
        feed = _inputs(state, marks, step, pages)
        out = program(*(feed[name] for name in NAMES))
        state = dict(zip(("table", "slot_page", "page_slot", "age", "slots", "clock"), out[:6], strict=True))
        state.update(host=feed["host"], slot_address=feed["slot_address"])
        held.append(state["slot_page"][:SLOTS].tolist())
        assert set(picks) <= set(held[-1]), (picks, held[-1])
    assert held == [[5, 7, PAGES], [5, 7, 4], [2, 1, 4], [2, 1, 7], [4, 1, 7]], held


if __name__ == "__main__":
    pytest.main([__file__])


ROWS, COLS, PAGE_SLOTS = 8, 16, 3


def _toy(slots=PAGE_SLOTS):
    """``y = w[sel] * x`` with ``w`` paged one row per page and three slots (``slots``) for eight rows."""
    import torch.nn as nn

    from emmy.compiler.trace.torch import trace_module

    class Toy(nn.Module):
        def forward(self, w, sel, x):
            return w[sel] * x

    w = torch.arange(ROWS * COLS, dtype=torch.float32).reshape(ROWS, COLS)
    graph = trace_module(Toy(), (w, torch.tensor([5, 1]), torch.full((2, COLS), 2.0)))
    graph.hints.set("cuda.paged_buffers", (("w", 0, 1, slots),))
    return Toy(), w, graph


def test_a_paged_buffer_with_slots_takes_its_inspector_and_the_paging_program_ahead_of_its_launch():
    """The plan: the kernel's inspector (its own index computation, marking pages) runs first, writing a zeroed
    marks buffer; the paging program's launches follow, over tables the plan takes as inputs named after the
    buffer; the kernel comes last, reading the buffer through the one table. The pages are also reachable as a
    ``[pages, words]`` view of the buffer's bytes, which is what the program copies."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph

    _, _, graph = _toy()
    plan = plan_from_graph(CudaBackend().compile(graph))
    kinds = [
        "inspect" if ".inspect." in lc.node_id else "page" if ".page." in lc.node_id or ".copy." in lc.node_id else "kernel"
        for lc in plan.launches
    ]
    assert kinds[0] == "inspect" and kinds[-1] == "kernel" and set(kinds[1:-1]) == {"page"}
    inspector, kernel = plan.launches[0], plan.launches[-1]
    assert inspector.arg_names == ("sel", "mul__w__marks") and inspector.zero_outputs == ("mul__w__marks",)
    assert "w__pages" in kernel.arg_names
    assert [name for name in plan.inputs if name.startswith("w__")] == [
        f"w__{t}" for t in ("table", "slot_page", "page_slot", "age", "host", "slot_address", "clock", "slots")
    ]
    assert plan.paged == {"w": (0, 1, PAGE_SLOTS), "w__view": (0, 1, PAGE_SLOTS)}
    view = next(b for b in plan.buffers if b.name == "w__view")
    assert view.alias is None and view.resolve_shape({}) == (ROWS, COLS * 4 // 8)


def test_a_paged_buffer_with_no_slots_takes_no_paging_program():
    """Nothing can move without a slot: the plan is the kernel alone, reading the buffer through its page table,
    and takes none of the pager's tables."""
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.plan import plan_from_graph

    plan = plan_from_graph(CudaBackend().compile(_toy(slots=0)[2]))
    (kernel,) = plan.launches
    assert "w__pages" in kernel.arg_names and plan.paged == {"w": (0, 1, 0)}
    assert not [name for name in plan.inputs if name.startswith("w__")]


@requires_cuda
def test_a_paged_buffer_with_slots_is_paged_in_from_host_memory_and_evicted_least_recently_used():
    """Five steps over eight rows and three slots, the pages' home pinned host memory: every step's result is the
    eager program's, and the tables show the slots filling, a hit keeping its slot and the least recently used page
    leaving when a fourth distinct page arrives."""
    from emmy import emmy_runtime
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.device import device
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.plan import plan_from_graph

    toy, w, graph = _toy()
    plan = plan_from_graph(CudaBackend().compile(graph))
    x = torch.full((2, COLS), 2.0)
    with gpu_lock():
        host = w.pin_memory()  # the pages' home
        pager = emmy_runtime.Pager()
        pager.declare(device(), "w", (0, host.data_ptr()), (ROWS, COLS * w.element_size(), PAGE_SLOTS, 0))
        program = CompiledProgram.build_from_plan(plan, {"sel": np.array([5, 1]), "x": x.numpy()})
        pager.bind(program.executor, "w")
        slot_page_ptr, slot_page_len = pager.table("w", "slot_page")
        residency = []
        for picks in ([5, 1], [5, 7], [2, 7], [0, 3]):
            sel = torch.tensor(picks)
            program.upload_prefix({"sel": sel.numpy()})
            program.run_once()
            got = program.outputs()[plan.outputs[0]]
            np.testing.assert_array_equal(got.reshape(2, COLS), toy(w, sel, x).numpy())
            slot_page = np.frombuffer(program.executor.read("w__slot_page"), dtype=np.int64)
            residency.append(sorted(int(p) for p in slot_page[:PAGE_SLOTS] if p < ROWS))
        assert slot_page_len == (PAGE_SLOTS + 1) * 8 and program.executor.buffer("w__slot_page")[0] == slot_page_ptr
    # Slots fill (1, 5), then 7 joins; 2 evicts 1, the least recently used (5 and 7 were just used); then 0 and 3
    # evict 5 (last used two steps ago) and 2 (used a step ago, the older of the two tied at that time), keeping 7.
    assert residency == [[1, 5], [1, 5, 7], [2, 5, 7], [0, 3, 7]], residency


@requires_cuda
def test_resident_pages_stay_on_the_device_and_the_rest_page_through_the_slots():
    """Eight rows, the first three resident on the device and the rest pinned host memory, two slots: every step's
    result is the eager program's, the resident rows are read where they live and never take a slot, and the host
    rows move through the two slots, the least recently used out."""
    from emmy import emmy_runtime
    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.cuda.device import device
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.plan import plan_from_graph

    slots, resident = 2, 3
    toy, w, graph = _toy(slots)
    plan = plan_from_graph(CudaBackend().compile(graph))
    x, page_bytes = torch.full((2, COLS), 2.0), COLS * w.element_size()
    with gpu_lock():
        on_device, host = w[:resident].cuda(), w[resident:].pin_memory()
        pager = emmy_runtime.Pager()
        pager.declare(device(), "w", (on_device.data_ptr(), host.data_ptr()), (ROWS, page_bytes, slots, resident))
        program = CompiledProgram.build_from_plan(plan, {"sel": np.array([5, 1]), "x": x.numpy()})
        pager.bind(program.executor, "w")
        homes = [on_device.data_ptr() + p * page_bytes for p in range(resident)]
        homes += [host.data_ptr() + p * page_bytes for p in range(ROWS - resident)]
        residency = []
        for picks in ([5, 1], [5, 7], [2, 6], [0, 3], [4, 0]):
            sel = torch.tensor(picks)
            program.upload_prefix({"sel": sel.numpy()})
            program.run_once()
            got = program.outputs()[plan.outputs[0]]
            np.testing.assert_array_equal(got.reshape(2, COLS), toy(w, sel, x).numpy())
            table = np.frombuffer(program.executor.read("w__table"), dtype=np.int64)[:ROWS]
            assert list(table[:resident]) == homes[:resident]
            slot_page = np.frombuffer(program.executor.read("w__slot_page"), dtype=np.int64)
            residency.append(sorted(int(p) for p in slot_page[:slots] if p < ROWS))
    # 5 and 7 fill the slots; 6 evicts 5 (both last used a step ago, 5 in the lower slot); 3 evicts 7 and 4 evicts 6,
    # the least recently used. The resident 0, 1 and 2 never take a slot.
    assert residency == [[5], [5, 7], [6, 7], [3, 6], [3, 4]], residency


def test_the_symbolic_decision_kernels_keep_independent_sweeps_apart():
    """The decision program, compiled with every size symbolic, stores per-page, per-slot and per-move values from
    kernels with no grid axis. No loop sits inside a loop over another size — a per-page store inside the per-slot
    sweep repeats once per slot: each sweep is its own loop, and a rank reduces over its own size."""
    import re

    from emmy.compiler.backend.paging import _paging_plan
    from emmy.compiler.dtype import I64

    decide, _, _ = _paging_plan(I64)
    for kernel, spec in decide.kernels.items():
        stack: list[tuple[int, str]] = []
        for line in spec.source.splitlines():
            indent = len(line) - len(line.lstrip())
            stack = [(depth, extent) for depth, extent in stack if depth < indent or not line.strip().startswith("}")]
            if match := re.search(r"for \(int \w+ = \w+(?:\.x)?; \w+ < (\w+);", line):
                outer = {extent for depth, extent in stack if depth < indent} - {match.group(1)}
                assert not outer, f"{kernel}: a loop over {match.group(1)} nests in one over {outer}"
                stack.append((indent, match.group(1)))


@requires_cuda
def test_a_listed_step_moves_the_page_its_host_lists_into_the_one_slot():
    """A host that knows its launches' order lists their page itself (``listed_paging``): over four pages and one slot,
    each step brings the listed page in and evicts the last, so the page table points the listed page at the slot,
    whose bytes are that page's, and every other page at its home."""
    from emmy import emmy_runtime
    from emmy.compiler.backend.cuda.device import device
    from emmy.compiler.backend.cuda.program import CompiledProgram
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.compiler.backend.paging import listed_paging
    from emmy.compiler.dtype import F32

    pages, elements = 4, 6
    home = torch.arange(pages * elements, dtype=torch.float32).reshape(pages, elements)
    with gpu_lock():
        host = home.pin_memory()
        pager = emmy_runtime.Pager()
        pager.declare(device(), "w", (0, host.data_ptr()), (pages, elements * 4, 1, 0))
        plan = listed_paging(pages, {"w": (1, (elements,), F32)})
        program = CompiledProgram.build_from_plan(plan, {"marks": np.zeros(pages + 1, np.int32)})
        pager.bind(program.executor, "w")
        table_ptr, _ = pager.table("w", "table")
        (slot,) = np.frombuffer(program.executor.read("w__slot_address"), dtype=np.int64)
        homes = [host.data_ptr() + page * elements * 4 for page in range(pages)]
        for page in (2, 0, 3, 3):
            marks = np.zeros(pages + 1, np.int32)
            marks[page] = 1
            program.upload_prefix({"marks": marks})
            program.run_once()
            table = np.frombuffer(program.executor.read("w__table"), dtype=np.int64)[:pages]
            assert list(table) == [slot if p == page else homes[p] for p in range(pages)]
            got = np.frombuffer(program.executor.read("w__slots"), dtype=np.float32)[:elements]
            np.testing.assert_array_equal(got, home[page].numpy())
        assert program.executor.buffer("w__table")[0] == table_ptr
