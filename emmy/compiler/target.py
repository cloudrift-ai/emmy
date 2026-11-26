"""Compile-time hardware target selection.

The tile-IR passes that gate on compute capability (``050_use_tma`` for
TMA, ``060_use_async_copy`` for cp.async) read the target via
:func:`compute_capability`. By default that probes the live CUDA device
through the runtime. Callers can override the target via :func:`set_target` so
the compiler emits code for a different architecture than the host —
useful for ``emmy compile`` on a CPU box that wants to see the
sm_120 codegen path, or for cross-compiling for a benchmark target.

CLI commands attach their compute-capability option with :func:`add_target_arg`
and resolve it via :func:`apply_target_arg`. The options accept the
canonical NVIDIA spelling (``sm_80``, ``sm_90``, ``sm_120``); the
optional architecture suffix on Hopper (``sm_90a``) is stripped.
"""

from __future__ import annotations

import functools
import logging
import re

_logger = logging.getLogger(__name__)

_OVERRIDE: tuple[int, int] | None = None

#: The target spelling that names the host CPU: ``run`` compiles and measures on the CPU backend.
CPU_TARGET = "cpu"


#: The element counts one vector access of a buffer of each dtype may hold: the 4-, 8- and 16-byte load and store
#: widths over the element's size. A fact of the card, read by the vector passes and by whoever spells the access.
_VECTOR_WIDTHS: dict[str, tuple[int, ...]] = {
    "f32": (2, 4),
    "f16": (2, 4, 8),
    "bf16": (2, 4, 8),
    "f8e4m3": (2, 4, 8, 16),
    "f8e5m2": (2, 4, 8, 16),
    "f4e2m1x2": (2, 4, 8, 16),
}


def vector_widths(dtype: str) -> tuple[int, ...]:
    """The element counts a vector access of ``dtype`` elements may hold, narrowest first; empty when none."""
    return _VECTOR_WIDTHS.get(dtype, ())


#: The operations a half-precision value computes natively, by the abstract names the elementwise table gives them:
#: the card rounds each once at f16. Any other operation over f16 computes in f32 and converts its result once. Not
#: ``sigmoid``: its half spelling ``1/(1+hexp(-x))`` rounds three times, which lands up to two half-ulps from torch's
#: compute-in-f32-round-once value and fails the strict ``rtol=1e-3`` gate.
_NATIVE_F16_OPS: frozenset[str] = frozenset(
    {
        "add",
        "subtract",
        "multiply",
        "divide",
        "maximum",
        "minimum",
        "exp",
        "log",
        "sqrt",
        "rsqrt",
        "tanh",
        "fabs",
        "abs",
        "negative",
        "copy",
        "reciprocal",
        "relu",
    }
)


def native_f16_ops() -> frozenset[str]:
    """The operations the target computes at f16 without leaving it; the type pass promotes every other one to f32."""
    return _NATIVE_F16_OPS


#: How many copies of a loop's body NVIDIA's compilers wrote (CUDA 13.0, ``sm_120``), read off cicc's PTX and ptxas's
#: machine code where they unrolled loops on their own. The unroll passes write the same copies into the Kernel IR.
#: ``117_unroll_loops``, a loop a lowering site flagged: written out in full up to ``FLAGGED_WHOLE`` trips, else
#: stepping ``FLAGGED_PER_TRIP`` copies a trip at most.
FLAGGED_WHOLE, FLAGGED_PER_TRIP = 64, 32
#: ``118_unroll_by_size``, any other loop, by the work one trip does. A static loop: the overhead of a trip, the most
#: work it is written out in full under, the most work one trip of it holds when it steps several.
STATIC_TRIP, STATIC_WHOLE, STATIC_PARTIAL = 2, 480, 199
#: The copies ptxas made of each trip of a static loop cicc copied twice.
PTXAS_AGAIN = 4
#: A stepping static loop ptxas wrote out in full: at most this many trips, each of at most this much work.
PTXAS_WHOLE_TRIPS, PTXAS_WHOLE_WORK = 4, 128
#: A loop whose trips only the launch knows: the overhead of a trip, the most work of a trip of its stepping loop, its
#: most copies, and the most work of one copy it is unrolled at.
RUNTIME_TRIP, RUNTIME_WORK, RUNTIME_COPIES, RUNTIME_LARGEST = 7, 228, 16, 89
#: A strided loop from a start that differs between threads: the fewest trips it is copied from.
THREAD_START_LEAST = 24
#: The instructions cicc writes for one call of a math function, by elementwise operation; any other operation is one.
MATH_INSTRUCTIONS: dict[str, int] = {
    "exp": 14,
    "exp_fast": 2,
    "expm1": 27,
    "log": 28,
    "log_fast": 2,
    "log2": 30,
    "log10": 29,
    "log1p": 32,
    "sin": 88,
    "cos": 89,
    "tan": 87,
    "tanh": 17,
    "sinh": 30,
    "cosh": 21,
    "arctan": 20,
    "arcsin": 24,
    "arccos": 27,
    "erf": 26,
    "pow": 113,
    "power": 113,
    "pow_fast": 3,
    "sigmoid": 15,
    "sigmoid_fast": 4,
}


def parse_sm(spec: str) -> tuple[int, int]:
    """Parse ``sm_NN`` / ``sm_NNN`` (with optional arch suffix) into ``(major, minor)``.

    Examples: ``sm_80`` → (8, 0), ``sm_86`` → (8, 6), ``sm_90`` → (9, 0),
    ``sm_90a`` → (9, 0), ``sm_120`` → (12, 0).
    """
    m = re.fullmatch(r"sm_(\d+)[a-z]?", spec.strip().lower())
    if not m:
        raise ValueError(f"invalid SM target {spec!r} — expected e.g. sm_80, sm_90, sm_120")
    digits = m.group(1)
    return (int(digits[:-1]), int(digits[-1]))


def set_target(cap: tuple[int, int] | None) -> None:
    """Set (or clear) the compile-time compute-capability override.

    Pass ``None`` to revert to live device probing. Clears the
    :func:`compute_capability` cache so the next caller sees the change.
    """
    global _OVERRIDE
    _OVERRIDE = cap
    compute_capability.cache_clear()


def add_target_arg(parser, *, dest: str = "target", option: str = "--target") -> None:
    """Add a compute-capability argument to ``parser``.

    The parsed value is a string; pass it to :func:`apply_target_arg` after parsing
    to install the override. ``compile`` / ``tune`` / ``trace`` spell it ``--target``;
    ``run`` spells the same override ``--gpu-arch``.
    """
    parser.add_argument(
        option,
        dest=dest,
        default=None,
        metavar="sm_NN",
        help=(
            "Compile-time target compute capability (e.g. sm_80, sm_90, sm_120). "
            "Overrides the live device's capability, so passes that gate on "
            "hardware features (TMA, cp.async) take the same path they would on "
            "the target GPU. Default: probe the active CUDA device. "
            f"``run`` also takes ``{CPU_TARGET}``: compile, check and measure on the host CPU's backend."
        ),
    )


def apply_target_arg(args, *, dest: str = "target") -> None:
    """Install the target from a parsed-args namespace, if set."""
    spec = getattr(args, dest, None)
    if spec is None:
        return
    cap = parse_sm(spec)
    set_target(cap)
    _logger.info("compile target set to sm_%d%d (override)", cap[0], cap[1])


@functools.cache
def compute_capability() -> tuple[int, int]:
    """Active compute capability as ``(major, minor)``.

    Returns the override set via :func:`set_target` if any; otherwise
    delegates to :func:`live_compute_capability`. Cached so repeated
    rule firings don't re-query the driver; :func:`set_target` clears
    the cache.
    """
    if _OVERRIDE is not None:
        return _OVERRIDE
    return live_compute_capability()


@functools.cache
def live_device_features() -> dict[str, float]:
    """Physical properties of the live CUDA device — SM count, shared memory per
    SM / per block, register file, warp size — for the online prior's
    hardware-regime features (see :meth:`Context.features`). These are SKU facts
    CUDA reports but compute-capability alone doesn't fix (an sm_120 laptop and an
    sm_120 RTX 5090 differ in SM count). Delegates to
    :func:`emmy.gpu.probe_live_features`, which probes the live device via
    the runtime and, when none is visible (GPU-less CI / offline eval), falls back to the
    **memorized** specs of :data:`emmy.gpu.DEFAULT_GPU` — so offline hosts get
    faithful per-SKU features instead of none. Cached — physical and
    target-independent, so :func:`set_target` need not clear it."""
    from emmy import gpu  # noqa: PLC0415

    return gpu.probe_live_features()


@functools.cache
def live_compute_capability() -> tuple[int, int]:
    """The live CUDA device's compute capability, ignoring any
    :func:`set_target` override.

    Returns ``(0, 0)`` when no device answers. Used by ``Context.probe``
    to size ``max_dynamic_smem`` to what the actual hardware can honor
    even when the target-derived gate cap is higher.
    """
    from emmy.compiler.backend.cuda.device import compute_capability  # noqa: PLC0415

    return compute_capability() or (0, 0)
