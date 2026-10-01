"""The golden file: a card's measurements in the tune DB's shape, so a compile imports them by copying rows.

A file holds the DB's three tables for one card — the kernels (``Kernel``: a ``kernel`` row, plus where a target
came from), the kernel-set decisions taken on them (``RoutingRow``) and the measurements (``Row``: a ``perf`` row)
— beside the traced programs the targets were lowered from, kept for the Torch twin a bench compares against and
for re-recording. The classes declare the layout; their wire is :mod:`emmy.compiler.wire`'s, and
:meth:`GoldenFile.check` holds the rules that cross objects.
"""

from __future__ import annotations

import fcntl
import json
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, fields, replace
from pathlib import Path

from emmy.compiler import provenance
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.knob import KnobType, family_of, get, validate_family_value
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.db import RoutingRow, is_placement_knob
from emmy.compiler.wire import Wire


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def prepare_traced_graph(graph) -> None:
    """Make a traced graph the pristine program a golden stores, in place: a value a speller marked to stay
    materialized becomes an auxiliary output, and provenance is re-seeded, so a stored program's fresh lowering
    is comparable with the kernels the file stores."""
    for node in graph.nodes.values():
        if node.hints.get("trace.materialize") and node.id not in graph.outputs:
            graph.outputs.append(node.id)
    for node in graph.nodes.values():
        node.hints.remove(provenance.PROV)
    provenance.seed(graph)


@dataclass(frozen=True)
class Kernel(KernelDef):
    """One kernel of the file: its definition — the ``kernel`` row the import writes — and, for a target, where it
    came from: the traced program (``traced``, an index into ``programs``) whose ops ``origins`` it computes whole,
    specialized at ``bindings``. A piece a cut or a split minted has no program of its own; a routing row reaches
    it from its parent."""

    traced: int | None = None
    origins: tuple[str, ...] = ()
    bindings: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class Measurements(Wire):
    """A row's measurement, and the comparison the bench took beside it."""

    emmy_us: float
    reference_us: float | None = None
    reference_backend: str | None = None

    def __post_init__(self) -> None:
        if not _positive(self.emmy_us) or (self.reference_us is not None and not _positive(self.reference_us)):
            raise ValueError("emmy_us and reference_us must be positive numbers")
        if self.reference_backend is not None and not self.reference_backend:
            raise ValueError("reference_backend must be a non-empty string")


@dataclass(frozen=True)
class Latency(Wire):
    """One card's timings of a corpus case: ``emmy_us`` is the ratchet, the torch numbers say whether the case is
    ahead of or behind torch there."""

    emmy_us: float
    tcompile_us: float | None = None
    eager_us: float | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and not _positive(value):
                raise ValueError(f"{f.name} must be a positive number")


@dataclass(frozen=True, kw_only=True)
class Row(Wire):
    """One ``perf`` row, or the schedule proposed for one: the kernel (by exact identity), the sizes its symbolic
    dims were benched at, the input regime (``pins``), the schedule row (``knobs``; ``None`` for a target that has
    only been traced) and the measurement. ``name`` is a label a command selects the row by; ``latency`` is a
    corpus case's per-card timings."""

    name: str
    kernel: str
    bindings: dict[str, int] = field(default_factory=dict)
    pins: dict[str, bool | int | str] = field(default_factory=dict)
    knobs: dict[str, bool | int | str] | None = None
    measurements: Measurements | None = None
    latency: dict[str, Latency] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if not _hex(self.kernel, 64):
            raise ValueError("kernel must be a 64-character lowercase hexadecimal digest")
        if any(not name or size <= 0 for name, size in self.bindings.items()):
            raise ValueError("bindings must map non-empty names to positive integers")
        if self.measurements is not None and self.knobs is None:
            raise ValueError("measurements require knobs")

    @property
    def measured(self) -> bool:
        return self.measurements is not None


@dataclass(frozen=True, kw_only=True)
class GoldenFile(Wire):
    """A golden document: the card, the model, the traced programs, and the three tables. ``gpu_name`` comes first
    on the wire so a card-scoped reader can skip a foreign file off its head; ``note`` is free text for the reader."""

    gpu_name: str | None = None
    compute_cap: tuple[int, int]
    model: str | None = None
    model_quant_digest: str | None = None
    note: str | None = None
    programs: list[dict] = field(default_factory=list)
    kernels: list[Kernel] = field(default_factory=list)
    routing: list[RoutingRow] = field(default_factory=list)
    rows: list[Row] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_quant_digest is not None and not _hex(self.model_quant_digest, 16):
            raise ValueError("model_quant_digest must be a 16-character lowercase hexadecimal digest")

    # --- files ------------------------------------------------------------------------------

    @classmethod
    @contextmanager
    def edit(cls, path: str | Path) -> Iterator[GoldenFile]:
        """The golden at ``path``, loaded to be changed and written back — one read-modify-write, held against every
        other process on this machine (two recorders that both load before either writes would each drop the
        other's rows)."""
        destination = Path(path)
        lock = destination.with_name(destination.name + ".lock")
        lock.parent.mkdir(parents=True, exist_ok=True)
        with open(lock, "w") as handle:  # noqa: PTH123, SIM115 — flock takes a descriptor
            fcntl.flock(handle, fcntl.LOCK_EX)
            document = cls.load(destination)
            yield document
            document.dump(destination, overwrite=True)

    @classmethod
    def load(cls, path: str | Path, *, repository: bool | None = None) -> GoldenFile:
        """The golden at ``path``, checked as a repository golden when it lives in the repository (or as
        ``repository`` says) and as a working file otherwise."""
        from .repository import is_repository_golden_path  # noqa: PLC0415 — the index reads this module

        source = Path(path)
        try:
            document = cls.from_wire(json.loads(source.read_text()))
            document.check(repository=is_repository_golden_path(source) if repository is None else repository)
        except (OSError, ValueError) as exc:
            raise ValueError(f"invalid golden file {source}: {exc}") from exc
        return document

    def dump(self, path: str | Path, *, repository: bool | None = None, overwrite: bool = False) -> Path:
        """Write the golden to ``path``, atomically, checked the way :meth:`load` would read it back."""
        from .repository import is_repository_golden_path  # noqa: PLC0415 — the index reads this module

        destination = Path(path)
        self.check(repository=is_repository_golden_path(destination) if repository is None else repository)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"{destination} already exists; pass overwrite=True to replace it")
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = _document_text(self.to_wire())
        mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", dir=destination.parent, prefix=f".{destination.name}.", delete=False) as output:
                output.write(payload)
                output.flush()
                os.fsync(output.fileno())
                temporary = Path(output.name)
            temporary.chmod(mode)
            temporary.replace(destination)
        finally:
            if temporary is not None and temporary.exists():
                temporary.unlink()
        return destination

    def check(self, *, repository: bool = False) -> None:
        """The rules the declarations cannot say: every reference resolves, a kernel is stored once, a row's pins
        name known knobs and its knobs spell a schedule and never a kernel-set decision — and, for a ``repository``
        golden, that the file names its card, every row spells a schedule, and a measurement carries its
        reference."""
        if repository and not self.gpu_name:
            raise ValueError("repository golden requires gpu_name")
        kernels: dict[str, Kernel] = {}
        for index, kernel in enumerate(self.kernels):
            where = f"kernels[{index}]"
            if kernel.exact_identity in kernels:
                raise ValueError(f"{where} stores {kernel.exact_identity[:12]} a second time")
            kernels[kernel.exact_identity] = kernel
            if kernel.traced is not None and not 0 <= kernel.traced < len(self.programs):
                raise ValueError(f"{where}.traced does not resolve in this document: {kernel.traced!r}")
            if kernel.origins and kernel.traced is None:
                raise ValueError(f"{where}.origins name nodes of a program the kernel does not have")
            if kernel.traced is not None:
                node_ids = {node["id"] for node in self.programs[kernel.traced]["nodes"]}
                if missing := set(kernel.origins) - node_ids:
                    raise ValueError(f"{where}.origins reference unknown program node(s): {', '.join(sorted(missing))}")
        for index, route in enumerate(self.routing):
            for identity in (route.parent, *route.children):
                if identity not in kernels:
                    raise ValueError(f"routing[{index}] names a kernel the file does not store: {identity[:12]}")
            if not route.arm or not all(is_placement_knob(key, value) for key, value in route.arm.items()):
                raise ValueError(f"routing[{index}].arm must hold placement knobs only, got {route.arm!r}")
        for index, row in enumerate(self.rows):
            where = f"rows[{index}] ({row.name})"
            if row.kernel not in kernels:
                raise ValueError(f"{where} names a kernel the file does not store: {row.kernel[:12]}")
            for name, value in row.pins.items():
                descriptor = get(family_of(name)) if name else None
                if descriptor is None:
                    raise ValueError(f"{where}.pins names unknown knob {name!r}")
                valid = {
                    KnobType.BOOL: type(value) is bool,
                    KnobType.INT: type(value) is int,
                    KnobType.STR: isinstance(value, str),
                    KnobType.BINMASK: type(value) is int or isinstance(value, str),
                }[descriptor.type]
                if not valid:
                    raise ValueError(f"{where}.pins.{name} must be a {descriptor.type.value} value, got {value!r}")
            for name, value in (row.knobs or {}).items():
                if is_placement_knob(name, value):
                    raise ValueError(f"{where}.knobs spells a kernel-set decision ({name}={value!r}); that is a routing row")
                if repository and family_of(str(name)) in {"WORK", "TILE", "REDUCE", "STAGE", "RASTER"}:
                    try:
                        validate_family_value(name, value)
                    except ValueError as exc:
                        raise ValueError(f"{where}.knobs.{name}: {exc}") from exc
            if repository and row.knobs is None:
                raise ValueError(f"{where}: a repository row must spell a schedule (knobs)")
            measured = row.measurements
            if repository and measured is not None and (measured.reference_us is None or measured.reference_backend is None):
                raise ValueError(f"{where}.measurements must carry reference_us and reference_backend")

    # --- reading ---------------------------------------------------------------------------

    def kernel(self, identity: str) -> Kernel:
        """The kernel ``identity`` names."""
        return next(kernel for kernel in self.kernels if kernel.exact_identity == identity)

    def targets(self) -> list[Kernel]:
        """The kernels lowered from a traced program — what a trace inventory records and a bench compiles."""
        return [kernel for kernel in self.kernels if kernel.traced is not None]

    def rows_of(self, name: str) -> list[Row]:
        return [row for row in self.rows if row.name == name]

    def program(self, index: int) -> Graph:
        """Traced program ``index`` as the compiler receives it — decoded and prepared exactly as the trace inventory
        writer prepares a fresh trace, so its fresh lowering is comparable with the kernels the file stores."""
        graph = Graph.from_wire(self.programs[index])
        prepare_traced_graph(graph)
        return graph

    def reference_program(self, kernel: Kernel) -> Graph | None:
        """The PyTorch slice ``kernel`` is compared against: the traced ops it came from (``origins``), at the
        kernel's sizes, with the kernel's outputs in its order. ``None`` when the file keeps no traced ops for it,
        or when they are no exact twin — the kernel writes a value the ops do not compute, or the two have
        different boundary inputs."""
        from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
        from emmy.compiler.pipeline.dump import CompilerDump  # noqa: PLC0415
        from emmy.compiler.specialize import specialize_program  # noqa: PLC0415

        if kernel.traced is None or not kernel.origins:
            return None
        program = self.program(kernel.traced)
        body = Graph.from_wire(kernel.loop_ir)
        origins = set(kernel.origins)
        computed = {buffer for origin in origins for buffer in program.nodes[origin].buffer_names()}
        reads = set(CompilerDump.frontend_reproducer_from_origins(program, origins).inputs)
        bound = {node_id for node_id, node in body.nodes.items() if isinstance(node.op, InputOp)}
        # The kernel may read a buffer bound from a constant (a weight the lowering transposed); it may not read an
        # activation the twin computes inside or never sees.
        activations = {buffer for node in program.nodes.values() if not isinstance(node.op, ConstantOp) for buffer in node.buffer_names()}
        if not (set(body.outputs) <= computed and reads <= bound and not (bound - reads) & activations):
            return None
        graph = specialize_program(CompilerDump.frontend_reproducer_from_origins(program, origins), kernel.bindings)
        graph.outputs = list(body.outputs)
        return graph

    def executable(self, kernel: Kernel, bindings: Mapping[str, int]) -> Graph:
        """``kernel`` as the program a compile or a bench starts from: its standalone body hinted at ``bindings``, with
        every input its twin's lowering produces from a constant bound the same way — a weight the kernel reads is the
        checkpoint's, beside the twin's, not a random input."""
        from emmy.compiler import pipeline  # noqa: PLC0415
        from emmy.compiler.context import Context  # noqa: PLC0415
        from emmy.compiler.ir.base import ConstantOp, InputOp  # noqa: PLC0415
        from emmy.compiler.pipeline import Pipeline  # noqa: PLC0415

        graph = kernel.program(bindings)
        if (twin := self.reference_program(kernel)) is None:
            return graph
        ctx = Context.from_target(tuple(self.compute_cap), gpu_name=self.gpu_name)
        for node_id, node in Pipeline.build(pipeline.LOOP_PASSES).run(twin, ctx=ctx, db=None).nodes.items():
            bound = graph.nodes.get(node_id)
            if isinstance(node.op, ConstantOp) and bound is not None and isinstance(bound.op, InputOp):
                bound.op = node.op
                graph.inputs = [name for name in graph.inputs if name != node_id]
        return graph

    def path_to(self, identity: str) -> list[RoutingRow]:
        """The kernel-set decisions from a target down to the kernel ``identity`` names, in order: empty for a
        target, the minting route and its ancestors for a piece. The first route that reaches a piece is its
        path."""
        parents: dict[str, RoutingRow] = {}
        for route in self.routing:
            for child in route.children:
                parents.setdefault(child, route)
        path: list[RoutingRow] = []
        seen = {identity}
        while (route := parents.get(identity)) is not None and route.parent not in seen:
            path.insert(0, route)
            identity = route.parent
            seen.add(identity)
        return path

    def shared_regime(self, rows: Iterable[Row] | None = None) -> dict:
        """The one input regime every row (of ``rows``, or of the file) shares, or ``{}`` when they disagree — a
        compile publishes a regime only when the rows it replays agree on it."""
        regimes = {tuple(sorted(row.pins.items())) for row in (self.rows if rows is None else rows)}
        return dict(regimes.pop()) if len(regimes) == 1 else {}

    # --- writing ---------------------------------------------------------------------------

    def add_kernel(self, kernel: Kernel) -> Kernel:
        """Store ``kernel`` unless a kernel of that identity is stored already; the stored one is returned, and
        takes ``kernel``'s provenance when it had none."""
        for index, stored in enumerate(self.kernels):
            if stored.exact_identity == kernel.exact_identity:
                if stored.traced is None and kernel.traced is not None:
                    self.kernels[index] = stored = replace(stored, traced=kernel.traced, origins=kernel.origins, bindings=kernel.bindings)
                return stored
        self.kernels.append(kernel)
        return kernel

    def add_routing(self, route: RoutingRow) -> None:
        """Store one decision on one parent; a later record of the same decision replaces what it minted."""
        self.routing[:] = [stored for stored in self.routing if (stored.parent, stored.arm) != (route.parent, route.arm)]
        self.routing.append(route)

    def upsert_row(self, row: Row) -> Row:
        """Store ``row``, replacing the row of the same kernel, sizes, regime and schedule when one is stored —
        which keeps its name, so a listing that points at it keeps landing."""
        from emmy.compiler.pipeline.knob import canonical_row_key  # noqa: PLC0415

        key = (row.kernel, tuple(sorted(row.bindings.items())), tuple(sorted(row.pins.items())), canonical_row_key(row.knobs or {}))
        for index, stored in enumerate(self.rows):
            stored_key = (
                stored.kernel,
                tuple(sorted(stored.bindings.items())),
                tuple(sorted(stored.pins.items())),
                canonical_row_key(stored.knobs or {}),
            )
            if stored_key == key:
                self.rows[index] = merged = replace(
                    row, name=stored.name, latency=row.latency if row.latency is not None else stored.latency
                )
                return merged
        self.rows.append(row)
        return row


# --- the text ---------------------------------------------------------------------------------

_TABLES = ("programs", "kernels", "routing", "rows")


def _compact(value: object) -> str:
    return json.dumps(value, separators=(",", ":"))


def _lines(entries: Iterable[str], indent: str) -> str:
    """A JSON list with one entry per line, so a diff lands on the entry that changed."""
    return "[\n" + ",\n".join(indent + entry for entry in entries) + f"\n{indent[:-1]}]"


def _document_text(wire: Mapping) -> str:
    """A golden as JSON a diff reads: a header key per line, ``gpu_name`` first on the opening line (what the card
    sniff reads alone), then each table with an entry per line."""
    items = []
    for key, value in wire.items():
        text = _lines(map(_compact, value), "  ") if key in _TABLES and value else json.dumps(value)
        items.append(f"{json.dumps(key)}: {text}")
    return "{" + ",\n ".join(items) + "}\n"


def program_text(graph) -> str:
    """A traced program as the wire a golden stores in ``programs`` — ``emmy compile --ir torch -o file.json``."""
    return _compact(graph.to_wire()) + "\n"
