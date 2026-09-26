"""The golden file: the classes that declare it — their wire is :mod:`emmy.compiler.wire`'s — the YAML layout a dump
writes, and the rules that cross objects (:meth:`GoldenFile.check`)."""

from __future__ import annotations

import os
import re
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, fields
from enum import StrEnum
from pathlib import Path

import yaml

from emmy import gpu
from emmy.compiler import provenance
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.knob import KnobType, family_of, get, validate_family_value, values_equal
from emmy.compiler.pipeline.search.pins import pins_freeze_cut
from emmy.compiler.wire import Wire

from .record import GoldenRecord

_SAFE_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


class _FlowSequence(list):
    """A YAML sequence rendered inline without changing the loaded schema."""


class _FlowMapping(dict):
    """A YAML mapping rendered inline without changing the loaded schema."""


class _GoldenDumper(yaml.SafeDumper):
    pass


_GoldenDumper.add_representer(
    _FlowSequence,
    lambda dumper, value: dumper.represent_sequence("tag:yaml.org,2002:seq", value, flow_style=True),
)
_GoldenDumper.add_representer(
    _FlowMapping,
    lambda dumper, value: dumper.represent_mapping("tag:yaml.org,2002:map", value, flow_style=True),
)


def _flow(value):
    if isinstance(value, list):
        return _FlowSequence(_flow(item) for item in value)
    if isinstance(value, Mapping):
        return _FlowMapping((key, _flow(item)) for key, item in value.items())
    return value


def _is_program(value: Mapping) -> bool:
    """A program nested inside a wire value — a constant's source graph."""
    return set(value) == {"inputs", "outputs", "nodes"}


def _short_flow(value: object) -> bool:
    if isinstance(value, Mapping):
        if _is_program(value):
            return False
        return len(repr(value)) <= 120 and all(_short_flow(item) for item in value.values())
    if isinstance(value, list):
        return len(repr(value)) <= 120 and all(_short_flow(item) for item in value)
    return value is None or isinstance(value, (str, int, float, bool))


def _style_wire_value(value):
    if isinstance(value, Mapping):
        if _is_program(value):
            return _style_program(value)
        styled = {key: _style_wire_value(item) for key, item in value.items()}
        return _flow(styled) if _short_flow(value) else styled
    if isinstance(value, list):
        return [_style_wire_value(item) for item in value]
    return value


def _style_program(program: Mapping) -> dict:
    styled_nodes = []
    for source in program["nodes"]:
        node = dict(source)
        if "attrs" in node:
            node["attrs"] = _style_wire_value(node["attrs"])
        if "inputs" in node:
            node["inputs"] = _flow(node["inputs"])
        node["outputs"] = _flow(node["outputs"])
        styled_nodes.append(node)
    styled = {
        "inputs": _flow(program["inputs"]),
        "outputs": _flow(program["outputs"]),
        "nodes": styled_nodes,
    }
    if "hints" in program:
        styled["hints"] = _flow(program["hints"])
    return styled


def _style_config(config: Mapping) -> dict:
    """A config block with its short leaves inline; a schedule (``knobs``) stays one knob per line."""
    styled = {key: _flow(value) if key == "target" else value for key, value in config.items()}
    if "realizations" in styled:
        styled["realizations"] = [
            {key: value if key == "knobs" else _flow(value) for key, value in row.items()} for row in config["realizations"]
        ]
    return styled


def _style_block(key: str, value):
    if key == "configs":
        return [_style_config(config) for config in value]
    return _flow(value) if key == "compute_cap" else value


class GoldenEntryState(StrEnum):
    INVENTORY = "inventory"
    PROPOSAL = "proposal"
    VERIFIED = "verified"


# --- the golden file, as classes --------------------------------------------------------------
#
# The file's shape is declared here and nowhere else: one generic walker turns the parsed YAML into
# these objects (``from_wire``) and back (``to_wire``), refusing unknown or missing keys with the
# path, and the leaf rules — a positive number, a hex digest — live in the constructors. Only the
# rules that cross objects stay as code, in ``GoldenFile.check``. The program and Loop IR pools stay
# wires: decoding a kernel builds a Loop op, whose construction normalizes the body, and that runs
# once, where a record's kernel graph is read, never at load.


def _hex(value: object, length: int) -> bool:
    return isinstance(value, str) and re.fullmatch(f"[0-9a-f]{{{length}}}", value) is not None


def _positive(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value > 0


def prepare_traced_graph(graph) -> None:
    """Make a traced graph the pristine program an inventory stores, in place.

    A birth-time speller may mark an internal storage value that has to remain materialized for
    a faithful target inventory (dynamic activation bits and scale are the first use); promoting
    it to an auxiliary graph output preserves the boundary without changing normal model outputs.
    And torch tracing or checkpoint spelling may hand over a graph that already crossed one
    compiler pipeline and carries implementation-piece provenance; the stable wire persists no
    provenance, so those selectors would come from one universe and replay after a fresh seed.
    Re-seeding here is exactly what the wire decoder's reader does, which is what makes a stored
    program's fresh lowering comparable with the kernels the inventory stored.
    """
    for traced_node in graph.nodes.values():
        if traced_node.hints.get("trace.materialize") and traced_node.id not in graph.outputs:
            graph.outputs.append(traced_node.id)
    for traced_node in graph.nodes.values():
        traced_node.hints.remove(provenance.PROV)
    provenance.seed(graph)


@dataclass(frozen=True)
class Target(Wire):
    """One config's stored kernel: its Loop IR in the file's ``loops`` pool, and the traced ops of its
    program it computes whole (``origins``) — the frontend slice a benchmark compares it against."""

    loop: int
    origins: tuple[str, ...] = ()


@dataclass(frozen=True)
class Measurements(Wire):
    """A model golden's measured row — one card per file, so the card is the file's."""

    emmy_us: float
    #: The comparison a bench took beside the measurement. A repository row carries one; a working row measured
    #: with none (a freeze's) carries only its own time.
    reference_us: float | None = None
    reference_backend: str | None = None

    def __post_init__(self) -> None:
        if not _positive(self.emmy_us) or (self.reference_us is not None and not _positive(self.reference_us)):
            raise ValueError("emmy_us and reference_us must be positive numbers")
        if self.reference_backend is not None and not self.reference_backend:
            raise ValueError("reference_backend must be a non-empty string")


@dataclass(frozen=True)
class Latency(Wire):
    """One card's timings of a corpus case. ``emmy_us`` is required — it is the ratchet, and a case
    without it stores nothing. ``tcompile_us`` and ``eager_us`` are the "are we ahead of or behind
    torch" half and are OPTIONAL, because some targets have no torch twin: a stored kernel holding
    part of an op benches Emmy alone, and torch.compile is dropped where it disagrees with eager (a
    random-input reproducer that produces NaN). Refusing to store the ratchet because the comparison
    is unavailable would discard the more important number of the two."""

    emmy_us: float
    tcompile_us: float | None = None
    eager_us: float | None = None

    def __post_init__(self) -> None:
        for f in fields(self):
            value = getattr(self, f.name)
            if value is not None and not _positive(value):
                raise ValueError(f"{f.name} must be a positive number")


@dataclass(kw_only=True)
class Realization(Wire):
    """One row of a target: its name, the input pins and bindings it stands under, and what it
    records — a schedule (``knobs``), a measurement, a kernel-set listing, a card's latencies."""

    name: str
    bindings: dict[str, int] = field(default_factory=dict)
    pins: dict[str, bool | int | str] = field(default_factory=dict)
    knobs: dict[str, bool | int | str] | None = None
    identity: str | None = None
    measurements: Measurements | None = None
    ranking: dict | None = None
    kernel_set: tuple[str, ...] = ()
    latency: dict[str, Latency] | None = None

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be a non-empty string")
        if any(not name or size <= 0 for name, size in self.bindings.items()):
            raise ValueError("bindings must map non-empty names to positive integers")
        if self.identity is not None and not _hex(self.identity, 64):
            raise ValueError("identity must be a 64-character lowercase hexadecimal digest")

    @property
    def state(self) -> GoldenEntryState:
        """What the row records on its own: nothing, a schedule, or a schedule with its measurement."""
        if self.knobs is None and self.measurements is None:
            return GoldenEntryState.INVENTORY
        if self.measurements is None:
            return GoldenEntryState.PROPOSAL
        if self.knobs is None:
            raise ValueError("measurements require knobs")
        return GoldenEntryState.VERIFIED

    def kernel_set_state(self, realizations: Sequence[Realization]) -> GoldenEntryState:
        """The row's state, read over the rows its ``kernel_set`` lists.

        An ordinary row decorates one kernel and answers for itself (:attr:`state`). A realization
        listing a kernel set usually carries no row of its own: a kernel SET ran, and the listed
        routing rows hold its price beside the schedule-carrying rows of the same target. So this
        answers ``VERIFIED`` only when it finds every listed row present with measurements, and finds
        measurements on every schedule-carrying row of the target too — a row whose knobs say
        something about a schedule, one key outside ``PLACE``: the child-identity receipts and,
        deliberately, a routing row whose decision is a cross-CTA ``REDUCE`` split, which the recorder
        measures like any other. One member short of that and nobody ever ran the set the listing
        describes. This is the one reading of "verified" that a repository check, a whole-file
        bench walk and the tuner's do-not-overwrite check share."""
        state = self.state
        if state is GoldenEntryState.VERIFIED or not self.kernel_set:
            return state
        by_name = {row.name: row for row in realizations}
        members = [by_name.get(name) for name in self.kernel_set]
        receipts = [
            row
            for row in realizations
            if row.identity and row is not self and row.knobs and any(family_of(str(key)) != "PLACE" for key in row.knobs)
        ]
        if any(member is None for member in members):
            return state
        if all(member.state is GoldenEntryState.VERIFIED for member in [*members, *receipts]):
            return GoldenEntryState.VERIFIED
        return state


@dataclass(kw_only=True)
class Config(Wire):
    """One stored target and its rows."""

    #: The traced program the target's ``origins`` name — the Torch twin a bench compares against — as an index
    #: into the file's pool; ``None`` for a target recorded from a measurement alone (a freeze's).
    program: int | None = None
    target: Target
    realizations: list[Realization]


@dataclass(kw_only=True)
class GoldenFile(Wire):
    """A golden document: the card, the model, the program and Loop IR pools, one config per stored
    target. ``gpu_name`` comes first on the wire so a card-scoped reader can skip a foreign file
    off its head (:func:`_file_gpu_name`)."""

    gpu_name: str | None = None
    compute_cap: tuple[int, int]
    model: str | None = None
    model_quant_digest: str | None = None
    programs: list[dict] = field(default_factory=list)
    configs: list[Config]
    loops: list[dict] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.model_quant_digest is not None and not _hex(self.model_quant_digest, 16):
            raise ValueError("model_quant_digest must be a 16-character lowercase hexadecimal digest")

    @classmethod
    def load(cls, path: str | Path, *, repository: bool | None = None) -> GoldenFile:
        """The golden at ``path``, checked as a repository golden when it lives in the repository —
        or as ``repository`` says — and as a working file otherwise."""
        from .repository import is_repository_golden_path  # noqa: PLC0415 — the index reads this module

        source = Path(path)
        try:
            document = cls.from_wire(yaml.load(source.read_text(), Loader=_SAFE_LOADER))
            document.check(repository=is_repository_golden_path(source) if repository is None else repository)
        except (OSError, yaml.YAMLError, ValueError) as exc:
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
        blocks = _pool_blocks(self)
        payload = "".join(
            blocks[key] if key in blocks else _dump_block(key, _style_block(key, value)) for key, value in self.to_wire().items()
        )
        temporary = None
        mode = destination.stat().st_mode & 0o777 if destination.exists() else 0o644
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
        """The rules the declarations cannot say: pool references resolve, pins name known knobs, a
        row's knobs agree with its pins, a kernel set names its siblings — and, for a ``repository``
        golden, that the file names its card, every row spells a schedule, and none carries working
        ranking metadata."""

        if repository and not self.gpu_name:
            raise ValueError("repository golden requires gpu_name")
        if not self.configs:
            raise ValueError("configs must be a non-empty list")
        for index, entry in enumerate(self.configs):
            where = f"configs[{index}]"
            if entry.program is not None and not 0 <= entry.program < len(self.programs):
                raise ValueError(f"{where}.program does not resolve in this document: {entry.program!r}")
            if not 0 <= entry.target.loop < len(self.loops):
                raise ValueError(f"{where}.target.loop does not resolve in this document: {entry.target.loop!r}")
            if entry.target.origins and entry.program is None:
                raise ValueError(f"{where}.target.origins name nodes of a program the config does not have")
            node_ids = {node["id"] for node in self.programs[entry.program]["nodes"]} if entry.program is not None else set()
            if missing_origins := set(entry.target.origins) - node_ids:
                raise ValueError(f"{where}.target.origins reference unknown program node(s): {', '.join(sorted(missing_origins))}")
            if not entry.realizations:
                raise ValueError(f"{where}.realizations must be a non-empty list")
            names = {row.name for row in entry.realizations}
            for row_index, row in enumerate(entry.realizations):
                row_where = f"{where}.realizations[{row_index}]"
                for name, value in row.pins.items():
                    descriptor = get(family_of(name)) if name else None
                    if descriptor is None:
                        raise ValueError(f"{row_where}.pins names unknown knob {name!r}")
                    valid = {
                        KnobType.BOOL: type(value) is bool,
                        KnobType.INT: type(value) is int,
                        KnobType.STR: isinstance(value, str),
                        KnobType.BINMASK: type(value) is int or isinstance(value, str),
                    }[descriptor.type]
                    if not valid:
                        raise ValueError(f"{row_where}.pins.{name} must be a {descriptor.type.value} value, got {value!r}")
                    if repository and family_of(name) in {"WORK", "TILE", "REDUCE", "STAGE", "RASTER"}:
                        try:
                            validate_family_value(name, value)
                        except ValueError as exc:
                            raise ValueError(f"{row_where}.pins.{name}: {exc}") from exc
                if row.knobs is not None:
                    conflicts = [
                        name for name, value in row.pins.items() if name in row.knobs and not values_equal(name, value, row.knobs[name])
                    ]
                    if conflicts:
                        raise ValueError(f"{row_where} gives conflicting input pins and measured knobs for {', '.join(sorted(conflicts))}")
                    families = {family_of(str(key)) for key in row.knobs}
                    if "PLACE" in families and families != {"PLACE"}:
                        raise ValueError(f"{row_where} mixes PLACE routing knobs with schedule knobs")
                    if families and "PLACE" not in families and pins_freeze_cut(row.pins) and row.identity is None:
                        raise ValueError(
                            f"{row_where} schedules a kernel behind pinned cut(s) without naming it; "
                            "a child-identity schedule receipt must store the child kernel's identity"
                        )
                if unknown := [name for name in row.kernel_set if name not in names]:
                    raise ValueError(f"{row_where}.kernel_set names no realization of this target: {', '.join(sorted(unknown))}")
                if repository and row.ranking is not None:
                    raise ValueError(f"{row_where} working ranking metadata cannot be promoted")
                measured = row.measurements
                if repository and measured is not None and (measured.reference_us is None or measured.reference_backend is None):
                    raise ValueError(f"{row_where}.measurements must carry reference_us and reference_backend")
                try:
                    state = row.kernel_set_state(entry.realizations)
                except ValueError as exc:
                    raise ValueError(f"{row_where} ({row.name}): {exc}") from exc
                if repository and state == GoldenEntryState.INVENTORY:
                    raise ValueError(f"{row_where} a repository row must spell a schedule (knobs)")

    def records(self) -> list[GoldenRecord]:
        """Every row of the file as the flattened record the evidence consumers read."""
        return [self.record(entry, row, config_index=index) for index, entry in enumerate(self.configs) for row in entry.realizations]

    def record(self, entry: Config, realization: Realization, *, config_index: int = 0) -> GoldenRecord:
        return GoldenRecord(
            name=realization.name,
            gpu_name=gpu.canonical_name(self.gpu_name or ""),
            compute_cap=tuple(self.compute_cap),
            model=self.model,
            program_index=entry.program,
            program_wire=self.programs[entry.program] if entry.program is not None else None,
            config_index=config_index,
            origins=tuple(entry.target.origins),
            bindings=tuple(sorted(realization.bindings.items())),
            pins=tuple(sorted(realization.pins.items())),
            loop_index=entry.target.loop,
            loop_wire=self.loops[entry.target.loop],
            knobs=dict(realization.knobs or {}),
            measurements=realization.measurements,
            ranking=dict(realization.ranking) if realization.ranking is not None else None,
            identity=realization.identity,
            kernel_set=tuple(realization.kernel_set),
            latency=dict(realization.latency) if realization.latency is not None else None,
        )

    def program(self, index: int):
        """The traced program ``index`` as the compiler receives it — decoded from the wire and prepared
        exactly as the trace inventory writer prepares a fresh trace, so its fresh lowering is
        comparable byte for byte with the targets the golden stores."""

        graph = Graph.from_wire(self.programs[index])
        prepare_traced_graph(graph)
        return graph

    def kernels(self, program: int | None = None) -> list[dict]:
        """The Loop IR kernels the targets store, each once, in target order — every target's, or
        those of one traced ``program``."""
        indexes = [entry.target.loop for entry in self.configs if program is None or entry.program == program]
        return [self.loops[index] for index in dict.fromkeys(indexes)]


_POOL_KEYS = ("programs", "loops")


def _dump_block(key: str, value: object) -> str:
    """One top-level key as YAML. Block-style keys concatenate into exactly the document
    ``yaml.dump`` writes for the whole mapping, so a block can be reused verbatim."""
    return yaml.dump({key: value}, Dumper=_GoldenDumper, sort_keys=False, width=140)


def _pool_blocks(document: GoldenFile) -> dict[str, str]:
    """The serialized program and loop pools."""
    pools = [getattr(document, key) for key in _POOL_KEYS]
    return {
        key: _dump_block(key, [_style_program(program) for program in pool]) for key, pool in zip(_POOL_KEYS, pools, strict=True) if pool
    }


def program_text(graph) -> str:
    """A traced program as the wire a golden stores in ``programs`` — ``emmy compile --ir torch -o file.yaml``."""
    return yaml.dump(_style_program(graph.to_wire()), Dumper=_GoldenDumper, sort_keys=False, width=140)


def kernel_pool_text(kernels: Iterable[Mapping]) -> str:
    """Loop IR kernels as the pool a golden stores them in, sorted by output set, so two pools diff
    line by line whatever order their kernels came in: what ``emmy golden kernels`` prints for a
    golden and ``emmy compile --golden PATH --program N --ir loop -o fresh.yaml`` writes for its fresh
    lowering."""
    ordered = sorted(kernels, key=lambda kernel: sorted(kernel["outputs"]))
    return yaml.dump([_style_program(kernel) for kernel in ordered], Dumper=_GoldenDumper, sort_keys=False, width=140)
