"""The golden pools: one kernel's schedule space on one card, in one precision regime, at one set of sizes, and the
golden rows measured on it — the supervision a training group is built over.

The regime vocabulary lives here too: :data:`REGIME_PINS` maps arithmetic and cache regimes to golden input pins.
:func:`regime_of` reads a row's compiler flags and cache mode back to that key. The freeze and export preserve it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

from emmy.compiler.context import FAST_MATH_FLAG
from emmy.compiler.pipeline.search.dataset.kernel import KernelDef
from emmy.compiler.pipeline.search.space import PRECISION_KNOBS, precision_pins
from emmy.compiler.wire import Wire


def regime_of(flags: str, cold_cache: bool = False, precision: dict | None = None) -> str:
    """The recorded precision and cache regime, with independent overrides preserved and no ambient pins."""
    pins = precision_pins(precision or {"FAST_MATH": FAST_MATH_FLAG in flags.split()})
    umbrella = pins["FAST_MATH"]
    parts = [FAST_MATH_FLAG] if umbrella else []
    parts.extend(f"{name}={int(value)}" for name, value in pins.items() if name != "FAST_MATH" and value != umbrella)
    return " ".join([*parts, *(["cold-cache"] if cold_cache else [])])


# Keys spell only overrides; publication pins every recorded choice, including the hot-cache default.
REGIME_PINS = {}
for _values in product((False, True), repeat=len(PRECISION_KNOBS)):
    _pins = {knob.name: value for knob, value in zip(PRECISION_KNOBS, _values, strict=True)}
    for _cold in (False, True):
        REGIME_PINS[regime_of("", _cold, _pins)] = {**_pins, "COLD_CACHE": _cold}


@dataclass(frozen=True)
class GoldenRow(Wire):
    """One verified row: the schedule row a golden file recorded on the pool's kernel, the microseconds it
    measured, and the source it was filed under (``golden:<digest>``)."""

    knobs: dict[str, str]
    us: float
    source: str


@dataclass(frozen=True)
class GoldenPool(Wire):
    """One candidate pool the golden files record verified rows in: one kernel on one card, in one precision regime
    (``regime``: a key of :data:`REGIME_PINS`), at one set of sizes, and the golden rows measured on it. The pool is
    enumerated from the kernel's own definition (``kernel.loop_ir``); a kernel formed from no loop op
    (``kernel.formed`` false: a piece carved from a twisted tree, which only its parent's program reaches) has none,
    and its pool is skipped by name. A placement pool also holds the constant inputs its golden program binds
    (``placement_constants``, each constant's wire by input name), so the placement walk offers the forks a deploy
    sees — the layout fork of a transposed weight among them; the kernel's standalone definition reads every input
    as an activation."""

    gpu: str
    cap: tuple[int, int]
    regime: str
    kernel: KernelDef
    bindings: dict[str, int]
    rows: tuple[GoldenRow, ...]
    placement_constants: dict[str, dict] = field(default_factory=dict)

    @property
    def name(self) -> str:
        """The pool's label in a report: the kernel's C name and the head of its exact identity — the name a freeze
        gives its realizations — with the sizes when the kernel is symbolic."""
        sizes = " ".join(f"{var}={size}" for var, size in sorted(self.bindings.items()))
        return f"{self.kernel.name}.{self.kernel.exact_identity[:12]}" + (f" {sizes}" if sizes else "")

    @property
    def pins(self) -> dict:
        """The input pins the pool's rows were measured under."""
        return REGIME_PINS[self.regime]

    def schedule_rows(self) -> list[dict[str, str]]:
        """Each golden row's schedule row."""
        return [row.knobs for row in self.rows]

    @property
    def emmy_us(self) -> float:
        """The fastest golden time recorded in the pool."""
        return min(row.us for row in self.rows)
