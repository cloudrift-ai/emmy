"""One row of a golden as the evidence consumers read it — the flattened record — with the pin helpers that
read a row's regime, and the record's compiler-facing derivations: its stored kernel through the current loop
passes, its lift to Tile IR, its structural features."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from emmy import config
from emmy.compiler.graph import Graph
from emmy.compiler.pipeline.knob import family_of
from emmy.compiler.pipeline.search.data.shape import ShapeKey
from emmy.compiler.structural import digest

if TYPE_CHECKING:
    from .format import Latency, Measurements


def fast_math_knobs(knobs: Mapping) -> bool:
    """Whether recorded knobs select a precision-trading realization."""
    from emmy.compiler.ir.schedule import Tile, Work  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import FAST_EXP  # noqa: PLC0415

    for key, value in knobs.items():
        spelling = str(value)
        if family_of(str(key)) == "TILE" and spelling:
            try:
                plan = Tile.parse(spelling, Work(kind="warp", units=(1, 1)))
            except ValueError:
                plan = None
            if plan is not None and plan.is_warp and plan.atom.operand_dtype("c").nbytes == 2:
                return True
        if key == FAST_EXP.name and spelling.casefold() in {"true", "1", "yes", "on"}:
            return True
    return False


def precision_trading_pins(pins: Mapping) -> bool:
    """Whether recorded pins enable a precision-trading compiler or NVCC policy."""
    umbrella = bool(pins.get("FAST_MATH", False))
    return umbrella or any(bool(pins.get(name, False)) for name in ("FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA"))


def pins_freeze_cut(pins: Mapping) -> bool:
    """Whether the input pins freeze any placement cut (a ``PLACE…=cut`` pin) — the ONE spelling
    of the predicate behind both the loader's receipt validation and :attr:`GoldenRecord.is_receipt`."""

    return any(family_of(str(name)) == "PLACE" and str(value) == "cut" for name, value in pins.items())


@dataclass(frozen=True)
class GoldenRecord:
    name: str
    gpu_name: str
    compute_cap: tuple[int, int]
    model: str | None
    #: The traced program the target's ``origins`` name — the Torch twin a bench compares against — and its index
    #: in the document's pool; ``None`` for a target recorded from a measurement alone.
    program_index: int | None
    program_wire: dict | None
    origins: tuple[str, ...]
    bindings: tuple[tuple[str, int], ...]
    pins: tuple[tuple[str, object], ...]
    knobs: dict
    measurements: Measurements | None
    ranking: dict | None
    loop_index: int | None = None
    loop_wire: dict | None = None
    #: Which config entry of its document the record came from. A config is one kernel set of one target: several
    #: configs can hold one loop (a target's fused rows, and each route's receipts), and each is its own set.
    config_index: int = 0
    #: The record's stored deploy identity (``identity_key(with_io=True)``, see :func:`kernel_identity`), when the
    #: file keeps one. Model inventories mostly do not; the realization corpus does, because a new
    #: fingerprint fact must show up as a diff there rather than silently re-key a checked-in
    #: reproducer. A stored identity is the strict decode's kernel selector, and it is how a
    #: **child-identity schedule receipt** names its kernel: a record whose pins freeze a cut lowers
    #: to several kernels, and only the stored identity says which child this row's schedule
    #: decorates (and so which kernel's ``S_*`` signature its row is evidence under).
    identity: str | None = None
    #: The routing rows this realization's kernel set holds, by name — what ``record_greedy_pick``
    #: listed for it, in the order the compile took the decisions. A realization listing them
    #: usually carries no measurement of its own: those rows and the target's schedule-carrying
    #: rows hold the measurements, so they decide whether it verifies (:meth:`Realization.kernel_set_state`),
    #: what its replay spells (:func:`_replay`) and what a bench of it pins
    #: (:func:`kernel_set_pins`). Empty where the compile took no kernel-set decision, leaving a
    #: realization that carries its own measured row and needs no listing.
    kernel_set: tuple[str, ...] = ()
    #: Measured microseconds per ``Context.hardware_id``: ``{card: Latency}``. A model golden is one
    #: file per card and uses the flat ``measurements`` block instead; a corpus case is one file
    #: across many cards, which a flat block cannot hold.
    latency: dict[str, Latency] | None = None

    @property
    def is_routing(self) -> bool:
        """Whether this row records a kernel-set decision — a placement cut, or a cross-CTA split's
        ``g<n>`` arm, which mints its pieces the same way — rather than a kernel schedule."""
        from emmy.compiler.pipeline.search.pins import stampable_reduce  # noqa: PLC0415

        def arm(key: str, value) -> bool:
            family = family_of(str(key))
            return family == "PLACE" or (family == "REDUCE" and stampable_reduce(str(value)) == "")

        return bool(self.knobs) and all(arm(key, value) for key, value in self.knobs.items())

    @property
    def is_receipt(self) -> bool:
        """Whether this row is a child-identity schedule receipt: a schedule row recorded behind
        pinned cut(s), whose stored ``identity`` names the child kernel the row decorates."""
        return self.identity is not None and not self.is_routing and pins_freeze_cut(dict(self.pins))

    @property
    def route(self) -> dict[str, str]:
        """The placement this record carries — every ``PLACE`` key of its pins and knobs, spelled
        as recorded. A routing row keeps it in ``knobs``; a receipt, a corpus case or an ``--ab``
        row freezes it in ``pins``. Empty for a plain schedule row, which says the kernel it
        decorates ran fused."""
        route = {str(key): str(value) for key, value in self.pins if family_of(str(key)) == "PLACE"}
        route.update((str(key), str(value)) for key, value in self.knobs.items() if family_of(str(key)) == "PLACE")
        return route

    @property
    def schedule_row(self) -> dict[str, str]:
        """The schedule half of the record — its decided tuning knobs minus the route, as the evidence
        index carries them (an OFF ``''`` is a decided value and stays)."""
        from emmy.compiler.pipeline.knob import tuning_knob_items  # noqa: PLC0415

        return {key: value for key, value in tuning_knob_items(self.knobs) if family_of(key) != "PLACE"}

    @property
    def pool_group(self) -> tuple:
        """Which candidate pool this record belongs to — the ONE place that question is answered, so every
        consumer that groups goldens groups them the same way. (A grouping key over RECORDS —
        distinct from the scheduler's per-compile ``pool_id`` stamp.)

        Composed from the target kernels' identity keys — the one identity function — around the
        card and the record's pin regime: per fused kernel, the structural variant key
        (``identity_key(with_io=True, with_knobs=True)`` — cluster siblings share a schedule
        space, so they rightly share a pool) folded with the symbolic-dim hints the enumeration
        sizes against. Node-id spelling never enters, so two recordings of one program made in
        different sessions FUSE — the wire-digest key this replaces split them — and any fact
        that changes the kernels shows up in their keys, so the key stays sufficient. It keys on
        what the enumeration READS, never on what it produced, so it does not go stale when the
        scheduler changes; bindings stay out (they bind replay values, not the space).

        Best-effort like every record-side derivation: a target the current compiler no longer
        lowers falls back to the persisted wire's digest, so a stale record still groups
        deterministically (alone) instead of breaking a fit."""
        from emmy.compiler.dim import DEFAULT_SEQ_HINT  # noqa: PLC0415

        try:
            _lowered, nodes = _target_kernel_nodes(self)
            kernels = tuple(
                sorted(
                    digest(
                        op.identity_key(with_io=True, with_knobs=True) or "",
                        tuple(
                            d.hint or DEFAULT_SEQ_HINT
                            for t in (*op.inputs.values(), *op.outputs.values())
                            for d in t.shape
                            if not d.is_static
                        ),
                    )
                    for op in (node.op for node in nodes)
                )
            )
        except Exception:  # noqa: BLE001 — a stale record must never break the fit's dataset build
            kernels = (hashlib.blake2b(json.dumps(self.loop_wire, sort_keys=True).encode(), digest_size=16).digest(),)
        return (self.gpu_name, tuple(self.compute_cap), kernels, self.pin_key)

    @property
    def pin_key(self) -> tuple:
        """This record's pins as a hashable tuple — already sorted, as the loader stores them."""
        return tuple((k, str(v)) for k, v in self.pins)

    @property
    def program(self):
        """The stable Torch IR payload, decoded."""
        if self.program_wire is None:
            raise ValueError(f"{self.name}: the target was recorded from a measurement alone and has no traced program")
        return Graph.from_wire(self.program_wire)

    @property
    def kernel_graph(self):
        """The stored kernel's Loop IR, unspecialized."""
        return Graph.from_wire(self.loop_wire)

    @property
    def target_program(self):
        """The stored kernel as a standalone program, specialized to this record's bindings."""
        from emmy.compiler.specialize import specialize_program  # noqa: PLC0415

        return specialize_program(self.kernel_graph, dict(self.bindings))

    @property
    def reference_program(self):
        """The PyTorch slice the stored kernel is compared against: the traced ops it came from
        (``origins``) with the kernel's outputs in its order. ``None`` when the golden keeps no
        traced ops for it, or when they are no exact twin — the kernel writes a value the ops do not
        compute, or the slice and kernel have different boundary inputs. Comparison only: the stored
        kernel stays the identity."""
        from emmy.compiler.ir.base import InputOp  # noqa: PLC0415
        from emmy.compiler.pipeline import CompilerDump  # noqa: PLC0415

        if not self.origins:
            return None
        kernel = self.kernel_graph
        computed = {buffer for origin in self.origins for buffer in self.program.nodes[origin].buffer_names()}
        reads = CompilerDump.frontend_reproducer_from_origins(self.program, set(self.origins)).inputs
        bound = {node_id for node_id, node in kernel.nodes.items() if isinstance(node.op, InputOp)}
        if not (set(kernel.outputs) <= computed and set(reads) == bound):
            return None
        graph = self._frontend_slice(self.origins)
        graph.outputs = list(kernel.outputs)
        return graph

    def _frontend_slice(self, origins):
        from emmy.compiler.pipeline import CompilerDump  # noqa: PLC0415
        from emmy.compiler.specialize import specialize_program  # noqa: PLC0415

        return specialize_program(CompilerDump.frontend_reproducer_from_origins(self.program, set(origins)), dict(self.bindings))

    @property
    def target_key(self) -> tuple:
        """Document-local identity shared by candidate rows for one target."""
        return ("loop", self.loop_index)

    @property
    def binding_map(self) -> dict[str, int]:
        return dict(self.bindings)

    @property
    def pin_map(self) -> dict[str, object]:
        return dict(self.pins)

    @property
    def shape_key(self) -> ShapeKey:
        """The arithmetic-identity descriptor for eval / diagnostics grouping, derived from the
        lowered target's stamped histogram. NOT the deploy join key — that is
        :func:`kernel_identity` (strict structural identity); this key only groups eval rows."""
        return ShapeKey.from_s_features(self.structural_features)

    @property
    def structural_features(self) -> dict[str, float]:
        """Current compiler features, derived lazily through target provenance."""
        return dict(_derive_structural_features(self))

    @property
    def origin_ops(self) -> tuple[str, ...]:
        if not self.origins or self.program_wire is None:
            return ()
        by_id = {node["id"]: node["op"] for node in self.program_wire["nodes"]}
        return tuple(by_id[origin] for origin in self.origins)

    @property
    def dtype(self) -> str:
        """Public dtype spelling of the stored kernel's first output."""
        graph = self.target_program
        tensor = graph.buffer(graph.outputs[0])
        if tensor is None:
            raise ValueError(f"{self.name}: Loop IR target has no output tensor")
        output_dtype = tensor.dtype.name
        return {"f16": "fp16", "f32": "fp32"}.get(output_dtype, output_dtype)

    @property
    def is_matmul(self) -> bool:
        """Whether this target is a plain frontend contraction — read off the STORED origin
        operations alone (a fused norm→linear names its norm origin too, so the subset test
        separates them), never by lowering the target: an eval listing must classify a record
        the current compiler can no longer lower."""
        return bool(self.origin_ops) and set(self.origin_ops) <= {"torch.matmul", "torch.linear"}

    @property
    def emmy_us(self) -> float:
        return float(self.measurements.emmy_us) if self.measurements is not None else 0.0

    @property
    def reference_us(self) -> float:
        return float(self.measurements.reference_us or 0.0) if self.measurements is not None else 0.0

    @property
    def reference_backend(self) -> str | None:
        return self.measurements.reference_backend if self.measurements is not None else None

    @property
    def dynamic(self) -> bool:
        return self.shape_key.is_dyn

    @property
    def sm_count(self) -> int | None:
        from emmy import gpu  # noqa: PLC0415

        spec = gpu.by_name(self.gpu_name)
        return spec.sm_count if spec else None


def regime_pins(record: GoldenRecord) -> dict:
    """The record's INPUT pin regime — its pins minus the route: the precision knobs (``FAST_MATH``
    and friends) a replay publishes to the environment so the record reads as live evidence
    (:func:`regime_live`). The schedule row and the route never travel this way; they are
    measured rows the evidence pick joins to the kernel they were recorded for."""
    return {str(key): value for key, value in record.pins if family_of(str(key)) != "PLACE"}


def kernel_set_pins(record: GoldenRecord, records: Sequence[GoldenRecord]) -> dict:
    """The arms of the routing rows ``record``'s ``kernel_set`` lists, as one hand pin — what a
    bench of that realization publishes so the compile reaches the kernel set the recording
    measured.

    A routing row's knobs ARE its arm — a placement cut's ``PLACE@seam: cut`` or a cross-CTA
    split's ``REDUCE`` value — so both kinds travel. A cascade's later decisions are taken on the
    pieces the earlier ones mint, and a scoped ``PLACE`` pin that resolves on no kernel addresses
    another kernel of the graph (``030_cut._placement_restriction``), so publishing every routing
    row's keys at once reproduces the whole cascade rather than only its first step. Empty for a
    record that names no route, which is the ordinary row whose own knobs are its pin.

    Both precision lanes record their rows under one name, so a listed name resolves inside the
    record's own regime first: the standard lane's split is not the fast-math lane's.

    A row naming a piece a cut minted publishes only its ``PLACE`` keys. Its other knobs address
    that piece by identity, not by seam: published as a hand pin they would reach every kernel of
    the graph (two pieces' splits collapsing onto the last value, a piece that cannot split
    refusing). Its row decides that piece by identity instead, as evidence."""
    regime = regime_pins(record)
    by_name: dict[str, GoldenRecord] = {}
    for other in records:
        if other.name not in by_name or regime_pins(other) == regime:
            by_name[other.name] = other
    pins: dict[str, str] = {}
    for name in record.kernel_set:
        referenced = by_name.get(name)
        if referenced is None:
            continue
        own_kernel = referenced.identity in (None, record.identity)
        pins.update({str(key): str(value) for key, value in referenced.knobs.items() if own_kernel or family_of(str(key)) == "PLACE"})
    return pins


def shared_regime_pins(records: Sequence[GoldenRecord]) -> dict:
    """The one input regime every record shares, or ``{}`` when they disagree — a compile publishes
    a regime only when the records it replays agree on it, because choosing one would silently
    change which realization was requested."""
    regimes = {tuple(sorted(regime_pins(record).items())) for record in records}
    return dict(regimes.pop()) if len(regimes) == 1 else {}


def _record_cache_key(record: GoldenRecord) -> tuple:
    return (id(record.loop_wire), record.target_key, record.compute_cap, record.bindings)


def _target_kernel_nodes(record: GoldenRecord):
    """The record's stored kernel through the CURRENT loop passes: ``(lowered graph, nodes)``, one
    node per kernel the stored Loop IR lowers to. Raises when it lowers to none — the strict
    tripwire's loud case."""
    from emmy.compiler.context import Context  # noqa: PLC0415
    from emmy.compiler.ir.loop import LoopOp  # noqa: PLC0415
    from emmy.compiler.pipeline import LOOP_PASSES, Pipeline  # noqa: PLC0415

    ctx = Context.from_target(record.compute_cap, gpu_name=record.gpu_name or None)
    lowered = Pipeline.build(LOOP_PASSES).run(record.target_program.copy(), ctx=ctx)
    # One kernel per PRODUCER, not per output: a multi-output kernel (an NVFP4 re-encode emits
    # packed codes beside their block scales) produces several of the graph's outputs, and
    # counting it once per output made a single-kernel target read as "lowers to N kernels".
    producers = (lowered.producer(output) for output in lowered.outputs)
    nodes = list({node.id: node for node in producers if node is not None and isinstance(node.op, LoopOp)}.values())
    if not nodes:
        raise ValueError(f"{record.name}: the persisted target selects no kernel after lowering")
    return lowered, nodes


def _lifted_target(record: GoldenRecord):
    """Lift the record's single selected kernel to Tile IR — the tree the cut pass schedules: the
    lift, then the twist rewrite, exactly as ``tile/lift`` runs them. A placement key is
    spelled on that tree, so decoding it against the lift alone would name sites the fused
    single-pass carrier no longer has."""
    from emmy.compiler.pipeline.passes.tile._fromloop import lift_loop_op, lift_serial  # noqa: PLC0415
    from emmy.compiler.pipeline.passes.tile._twist import rewrite_twisted  # noqa: PLC0415

    lowered, nodes = _target_kernel_nodes(record)
    if len(nodes) != 1:
        raise ValueError(f"{record.name}: target lowers to {len(nodes)} kernels — a row decorates exactly one")
    node = nodes[0]
    node.op = node.op.with_io(lowered, node)
    # A serial kernel lifts its carried states as state buffers, as ``tile/lift`` does.
    tile = lift_serial(node.op, name=node.id, prefix=node.id)[0] if node.op.body.carries else lift_loop_op(node.op, name=node.id)
    tile = replace(tile, op=rewrite_twisted(tile.op, tile.axes))
    # A fork's root op is always matcher-refreshed (``_match_at`` runs ``with_io`` on every matched
    # node before the rule that offers the fork), so the record side mirrors the io through that
    # same call rather than a hand-rolled map: a multi-output kernel — an NVFP4 re-encode emits
    # packed codes beside their block scales — is bound to every one of the node's output buffers,
    # and the dtype half of the deploy identity (``identity_key(with_io=True)``) reads the same
    # output fingerprint on both sides.
    return tile.with_io(lowered, node)


def _derive_structural_features(record: GoldenRecord) -> tuple[tuple[str, float], ...]:
    """Lower the exact replay target and recover its unique ``S_*`` row."""
    from emmy.compiler.pipeline.knob import STRUCT_PREFIX  # noqa: PLC0415

    _lowered, nodes = _target_kernel_nodes(record)
    signatures = {
        tuple(
            sorted((name, float(value)) for name, value in (getattr(node.op, "knobs", {}) or {}).items() if name.startswith(STRUCT_PREFIX))
        )
        for node in nodes
    }
    signatures.discard(())
    if len(signatures) != 1:
        raise ValueError(f"{record.name}: target resolves to {len(signatures)} structural targets")
    return next(iter(signatures))


#: The precision-trading pin universe the regime check covers in BOTH directions — a record
#: that omits one of these was measured with it OFF, and must not deploy when it is live-ON.
_PRECISION_PINS = ("FAST_MATH", "FAST_EXP", "F16_MMA_F32_ACC", "FP8_MMA")


def regime_live(record: GoldenRecord) -> bool:
    """Whether the record's input-pin regime IS the live one — exact per pin: a BOOL pin compares
    against its effective precision policy (other BOOLs default off), anything else against the raw env
    string. Strict BOTH ways: a record measured under FAST_MATH is no evidence for a standard
    deploy, and a standard record none under a live precision-trading pin — the precision universe
    (:data:`_PRECISION_PINS`, umbrella semantics per ``space.precision_pin``) is compared even for
    pins the record omits (omitted = measured OFF). ``PLACE`` pins are the record's route, not a
    regime."""
    from emmy.compiler.pipeline.knob import KnobType, registry  # noqa: PLC0415
    from emmy.compiler.pipeline.search.space import precision_pin  # noqa: PLC0415

    knobs = registry()
    pins = record.pin_map
    for name, value in pins.items():
        if family_of(str(name)) == "PLACE":
            continue
        kn = knobs.get(str(name))
        raw = kn.raw() if kn is not None else config.knob_raw(str(name))
        if kn is not None and kn.type is KnobType.BOOL:
            live = precision_pin(kn) if name in _PRECISION_PINS else kn.parse(raw) if raw is not None else False
            if bool(value) != live:
                return False
        elif (raw or "") != str(value):
            return False
    umbrella = bool(pins.get("FAST_MATH", False))
    for name in _PRECISION_PINS:
        recorded = bool(pins.get(name, umbrella))
        kn = knobs.get(name)
        live = bool(precision_pin(kn)) if kn is not None else False
        if recorded != live:
            return False
    return True
