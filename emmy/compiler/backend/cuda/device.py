"""The live CUDA device, reached through the runtime's context.

One context per process, created on first use. ``CUDA_VISIBLE_DEVICES`` is how a pinned
worker selects its card, so ordinal 0 is always the right device here.
"""

from __future__ import annotations

_DEVICE = None


def device():
    """The process's CUDA context, created on first use."""
    global _DEVICE
    if _DEVICE is None:
        import emmy_runtime  # noqa: PLC0415 — the extension loads the driver lazily

        _DEVICE = emmy_runtime.Device(0)
    return _DEVICE


def compute_capability() -> tuple[int, int] | None:
    """The live device's ``(major, minor)``, or ``None`` when no device answers."""
    try:
        return tuple(device().compute_capability())
    except Exception:  # noqa: BLE001 — no driver, no device, or a failed context
        return None


def properties() -> dict[str, float] | None:
    """Per-device limits (SM count, shared memory, registers, warp size, memory), or ``None``."""
    try:
        return device().properties()
    except Exception:  # noqa: BLE001 — no device to probe
        return None


def name() -> str | None:
    """The device's product name, or ``None`` when no device is visible."""
    try:
        return device().name()
    except Exception:  # noqa: BLE001 — no device to probe
        return None


def context_poisoned() -> bool:
    """Whether this process's context is in a sticky-error state.

    ``False`` when no context was ever created here. A synchronize surfaces the sticky
    status an earlier illegal access left behind; it would block on a hung kernel, so a
    caller that knows a kernel hung must not probe."""
    if _DEVICE is None:
        return False
    try:
        _DEVICE.synchronize()
    except Exception:  # noqa: BLE001 — any driver error here means the context is unusable
        return True
    return False
