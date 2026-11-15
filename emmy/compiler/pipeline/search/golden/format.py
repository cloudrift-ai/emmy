"""The golden file: a card's measurements in the tune DB's shape, so a compile imports them row for row.

A file holds the DB's three tables for one card — the kernels (``Kernel``: a kernel's definition, plus where a
target came from), the kernel-set decisions taken on them (``RoutingRow``) and the measurements (``Row``: a ``perf``
row) — beside the traced programs the targets were lowered from, kept for the Torch twin a bench compares against
and for re-recording. A file holds inputs only: the Loop IR, the decisions and the measurements. A kernel's exact
identity — the key the DB files it under — and its name are computed from its Loop IR when something asks
(``KernelDef.exact_identity``, ``KernelDef.name``), so rows and decisions point at a kernel by its position in the
file's kernel list, and the file stores no name. The classes declare the layout; their wire is :mod:`emmy.compiler.wire`'s, and
:meth:`GoldenFile.check` holds the rules that cross objects.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import re
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path

from emmy.compiler import provenance
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.knob import KnobType, family_of, get, validate_family_value
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.db import RoutingRow, is_placement_knob
from emmy.compiler.wire import Wire

logger = logging.getLogger("emmy.compiler.pipeline")


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def _require_positive(timings: Wire, *names: str) -> None:
    """Refuse a timing of ``timings`` that is set and not a positive number."""
    for name in names:
        if (value := getattr(timings, name)) is not None and not _positive(value):
            raise ValueError(f"{name} must be a positive number")


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
    it from its parent. ``ref`` is how the file's rows and decisions point at the kernel while it is loaded: its
    position in the kernel list when read, a fresh number for a kernel added since, and its position again when
    written (:meth:`GoldenFile.to_wire`). It never travels."""

    wire_skip = frozenset({"ref"})

    traced: int | None = None
    origins: tuple[str, ...] = ()
    bindings: dict[str, int] = field(default_factory=dict)
    ref: int | None = field(default=None, compare=False)

    def with_ref(self, ref: int) -> Kernel:
        """This kernel under ``ref``, its identity kept where it is already known."""
        out = replace(self, ref=ref)
        return out.keyed(self.__dict__["exact_identity"]) if "exact_identity" in self.__dict__ else out


@dataclass(frozen=True)
class Measurements(Wire):
    """A row's measurement, the comparison the bench took beside it, and ``tried``: how many schedules of the kernel
    the card had measured at these sizes and in this regime when the row was recorded — the search behind the row,
    which no other field of the file can tell. It goes with the measurement when a restamp demotes the row."""

    emmy_us: float
    reference_us: float | None = None
    reference_backend: str | None = None
    tried: int | None = None

    def __post_init__(self) -> None:
        _require_positive(self, "emmy_us", "reference_us")
        if self.reference_backend is not None and not self.reference_backend:
            raise ValueError("reference_backend must be a non-empty string")
        if self.tried is not None and (type(self.tried) is not int or self.tried < 1):
            raise ValueError("tried must be a positive integer")


@dataclass(frozen=True)
class Latency(Wire):
    """One card's timings of a whole row: ``emmy_us`` is the ratchet, the torch numbers say whether the row is ahead
    of or behind torch there, and ``kernels`` is the kernel set the number timed — the target and every piece the
    recorded decisions reached when it was taken, by their ``ref`` in the file. A restamp clears the timing exactly
    when one of those kernels is re-keyed or dropped, and a bench pins the decisions that mint them. The set is
    empty only in a file written before the field, which a restamp fills once and :meth:`GoldenFile.dump` refuses."""

    emmy_us: float
    tcompile_us: float | None = None
    eager_us: float | None = None
    kernels: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        _require_positive(self, "emmy_us", "tcompile_us", "eager_us")
        if any(type(ref) is not int for ref in self.kernels):
            raise ValueError("kernels must point at kernels of the file")


@dataclass(frozen=True, kw_only=True)
class Row(Wire):
    """One ``perf`` row, or the schedule proposed for one: the kernel (by its ``ref`` in the file), the sizes its
    symbolic dims were benched at, the input regime (``pins``), the schedule row (``knobs``; ``None`` for a target that has
    only been traced) and the measurement. ``name`` is a label a command selects the row by; ``latency`` is the
    per-card timings of the whole row beside ``torch.compile`` and eager (``run --record``); ``note`` is free text
    for the reader — what a person found about the row, never a label the numbers decide."""

    name: str
    kernel: int
    bindings: dict[str, int] = field(default_factory=dict)
    pins: dict[str, bool | int | str] = field(default_factory=dict)
    knobs: dict[str, bool | int | str] | None = None
    measurements: Measurements | None = None
    latency: dict[str, Latency] | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if type(self.kernel) is not int:
            raise ValueError("kernel must point at a kernel of the file")
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
        taken = {kernel.ref for kernel in self.kernels if kernel.ref is not None}
        fresh = (ref for ref in range(len(self.kernels) + len(taken)) if ref not in taken)
        self.kernels[:] = [kernel if kernel.ref is not None else kernel.with_ref(next(fresh)) for kernel in self.kernels]

    def to_wire(self):
        """The wire, each kernel pointed at by its position in the kernel list."""
        at = {kernel.ref: position for position, kernel in enumerate(self.kernels)}
        out = super().to_wire()
        for route in out.get("routing", ()):
            route["parent"], route["children"] = at[route["parent"]], [at[child] for child in route["children"]]
        for row in out.get("rows", ()):
            row["kernel"] = at[row["kernel"]]
            for latency in (row.get("latency") or {}).values():
                latency["kernels"] = [at[ref] for ref in latency.get("kernels", ())]
        return out

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
        """Write the golden to ``path``, atomically, checked the way :meth:`load` would read it back — and with every
        latency naming its kernel set, so no file keeps the shape written before the field."""
        from .repository import is_repository_golden_path  # noqa: PLC0415 — the index reads this module

        destination = Path(path)
        self.check(repository=is_repository_golden_path(destination) if repository is None else repository)
        for row in self.rows:
            if any(not latency.kernels for latency in (row.latency or {}).values()):
                raise ValueError(f"rows ({row.name}): a latency must name the kernel set it timed; `emmy golden restamp` fills one")
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
        """The rules the declarations cannot say: every reference resolves, a kernel's ``ref`` is its own, a
        row's pins name known knobs and its knobs spell a schedule and never a kernel-set decision — and, for a ``repository``
        golden, that the file names its card, every row spells a schedule, and a measurement carries its
        reference."""
        if repository and not self.gpu_name:
            raise ValueError("repository golden requires gpu_name")
        kernels: dict[str, Kernel] = {}
        for index, kernel in enumerate(self.kernels):
            where = f"kernels[{index}]"
            if kernel.ref in kernels:
                raise ValueError(f"{where} shares its ref with another kernel")
            kernels[kernel.ref] = kernel
            if kernel.traced is not None and not 0 <= kernel.traced < len(self.programs):
                raise ValueError(f"{where}.traced does not resolve in this document: {kernel.traced!r}")
            if kernel.origins and kernel.traced is None:
                raise ValueError(f"{where}.origins name nodes of a program the kernel does not have")
            if kernel.traced is not None:
                node_ids = {node["id"] for node in self.programs[kernel.traced]["nodes"]}
                if missing := set(kernel.origins) - node_ids:
                    raise ValueError(f"{where}.origins reference unknown program node(s): {', '.join(sorted(missing))}")
        for index, route in enumerate(self.routing):
            for ref in (route.parent, *route.children):
                if ref not in kernels:
                    raise ValueError(f"routing[{index}] names a kernel the file does not store: {ref}")
            if not route.arm or not all(is_placement_knob(key, value) for key, value in route.arm.items()):
                raise ValueError(f"routing[{index}].arm must hold placement knobs only, got {route.arm!r}")
        for index, row in enumerate(self.rows):
            where = f"rows[{index}] ({row.name})"
            if row.kernel not in kernels:
                raise ValueError(f"{where} names a kernel the file does not store: {row.kernel}")
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
            for card, latency in (row.latency or {}).items():
                if unknown := [ref for ref in latency.kernels if ref not in kernels]:
                    raise ValueError(f"{where}.latency[{card!r}] names kernel(s) the file does not store: {', '.join(unknown)}")
            measured = row.measurements
            if repository and measured is not None and (measured.reference_us is None or measured.reference_backend is None):
                raise ValueError(f"{where}.measurements must carry reference_us and reference_backend")

    # --- reading ---------------------------------------------------------------------------

    def kernel(self, ref: int) -> Kernel:
        """The kernel ``ref`` points at."""
        return next(kernel for kernel in self.kernels if kernel.ref == ref)

    def identities(self) -> dict[int, str]:
        """Each kernel's exact identity by its ``ref`` — computed from the stored Loop IR (a lift per formed
        kernel, once per loaded file), which is why only an import, a restamp or a record asks. A kernel whose
        stored body the compiler no longer takes back has none and is left out: a stale kernel, whose rows describe
        nothing a compile builds (``emmy golden check`` names it)."""
        out: dict[str, str] = {}
        for kernel in self.kernels:
            try:
                out[kernel.ref] = kernel.exact_identity
            except Exception as exc:  # noqa: BLE001 — whatever the lift raises, the kernel is stale, the file is not broken
                logger.warning("golden kernel %d no longer lowers (%s: %s); its rows are no evidence", kernel.ref, type(exc).__name__, exc)
        return out

    def targets(self) -> list[Kernel]:
        """The kernels lowered from a traced program — what a trace inventory records and a bench compiles."""
        return [kernel for kernel in self.kernels if kernel.traced is not None]

    def target_rows(self) -> dict[tuple, list[Row]]:
        """The rows of each target kernel, by ``(target ref, sorted pins, sorted sizes)``: a target's own rows and its
        pieces' rows in one input regime and at one set of sizes — what one ``run --golden PATH`` realization benches,
        named by its shortest row name. A row's sizes are the bindings its own kernel's buffers name
        (:func:`~emmy.compiler.wire.symbolic_vars`). A row with none — a split's combine step, whose row records a
        reduce axis its table keeps from its parent, or a row recorded before rows named a size — counts at every size
        of its target, as it does in a compile's pick, so it joins each of the target's groups in its regime instead
        of making one of its own."""
        from emmy.compiler.wire import symbolic_vars  # noqa: PLC0415

        targets = {kernel.ref for kernel in self.targets()}
        own: dict[int, set[str]] = {}  # each kernel's sizes, read for the rows that record any
        out: dict[tuple, list[Row]] = {}
        sizeless: list[tuple[tuple, Row]] = []
        for row in self.rows:
            path = self.path_to(row.kernel)
            root = path[0].parent if path else row.kernel
            if root not in targets:
                continue
            regime = (root, tuple(sorted(row.pins.items())))
            if row.bindings and row.kernel not in own:
                own[row.kernel] = symbolic_vars(self.kernel(row.kernel).loop_ir)
            sizes = tuple(sorted((var, size) for var, size in row.bindings.items() if var in own[row.kernel]))
            if not sizes:
                sizeless.append((regime, row))
            else:
                out.setdefault((*regime, sizes), []).append(row)
        for regime, row in sizeless:
            for key in [key for key in out if key[:2] == regime] or [(*regime, ())]:
                out.setdefault(key, []).append(row)
        return out

    def rows_of(self, name: str) -> list[Row]:
        return [row for row in self.rows if row.name == name]

    def kernel_set(self, root: int) -> tuple[int, ...]:
        """``root`` and every kernel the file's decisions reach from it, each once, parents before their pieces: the
        kernels a compile of the target can deploy, whichever arms its decisions take."""
        out = [root]
        for ref in out:  # grows as decisions on the kernels reached are followed
            for route in self.routing:
                if route.parent == ref:
                    out.extend(child for child in route.children if child not in out)
        return tuple(out)

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

    def path_to(self, ref: int, preferred: Iterable[RoutingRow] = ()) -> list[RoutingRow]:
        """The kernel-set decisions from a target down to the kernel ``ref`` points at, in order: empty for a
        target, the minting route and its ancestors for a piece. The first route that reaches a piece is its
        path, the ``preferred`` routes before the file's."""
        parents: dict[int, RoutingRow] = {}
        for route in (*preferred, *self.routing):
            for child in route.children:
                parents.setdefault(child, route)
        path: list[RoutingRow] = []
        seen = {ref}
        while (route := parents.get(ref)) is not None and route.parent not in seen:
            path.insert(0, route)
            ref = route.parent
            seen.add(ref)
        return path

    def shared_regime(self, rows: Iterable[Row] | None = None) -> dict:
        """The one input regime every row (of ``rows``, or of the file) shares, or ``{}`` when they disagree — a
        compile publishes every effective precision pin, including omitted defaults, only when the rows agree."""
        from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

        regimes = {tuple(sorted((row.pins | precision_pins(row.pins)).items())) for row in (self.rows if rows is None else rows)}
        return dict(regimes.pop()) if len(regimes) == 1 else {}

    # --- writing ---------------------------------------------------------------------------

    def add_kernel(self, kernel: Kernel) -> Kernel:
        """Store ``kernel`` unless a kernel of that identity is stored already; the stored one is returned, and
        takes ``kernel``'s provenance when it had none. A kernel stored here takes a ``ref`` no other kernel of the
        file has."""
        identity = kernel.exact_identity
        for index, stored in enumerate(self.kernels):
            if stored.exact_identity == identity:
                if stored.traced is None and kernel.traced is not None:
                    stored = replace(stored, traced=kernel.traced, origins=kernel.origins, bindings=kernel.bindings).keyed(identity)
                    self.kernels[index] = stored
                return stored
        kernel = kernel.with_ref(max((stored.ref for stored in self.kernels), default=-1) + 1)
        self.kernels.append(kernel)
        return kernel

    def add_routing(self, route: RoutingRow) -> None:
        """Store one decision on one parent, the kernels named by their ``ref``; a later record of the same decision
        replaces what it minted."""
        self.routing[:] = [stored for stored in self.routing if (stored.parent, stored.arm) != (route.parent, route.arm)]
        self.routing.append(route)

    def upsert_row(self, row: Row) -> Row:
        """Store ``row``, replacing the row of the same kernel, sizes, regime and schedule when one is stored —
        including later decisions but ignoring explicit OFF defaults. Keep its name and equivalent input pins."""
        from emmy.compiler.pipeline.knob import is_off_value, values_equal  # noqa: PLC0415
        from emmy.compiler.pipeline.search.space import precision_pins  # noqa: PLC0415

        def knobs(r: Row) -> dict:
            return {key: value for key, value in (r.knobs or {}).items() if not is_off_value(family_of(key), value)}

        wanted = knobs(row)
        for index, stored in enumerate(self.rows):
            existing = knobs(stored)
            if (
                (stored.kernel, stored.bindings) == (row.kernel, row.bindings)
                and (stored.pins | precision_pins(stored.pins)) == (row.pins | precision_pins(row.pins))
                and existing.keys() == wanted.keys()
                and all(values_equal(key, value, wanted[key]) for key, value in existing.items())
            ):
                self.rows[index] = merged = replace(
                    row,
                    name=stored.name,
                    pins=stored.pins,
                    latency=row.latency if row.latency is not None else stored.latency,
                    note=row.note if row.note is not None else stored.note,
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
