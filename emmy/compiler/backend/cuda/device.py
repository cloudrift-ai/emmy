"""The live CUDA device, reached through the runtime's context.

Contexts follow torch's current logical device. A worker restricted by ``CUDA_VISIBLE_DEVICES``
still selects logical zero; without CUDA-capable torch, the runtime uses that visible device.
"""

from __future__ import annotations

_DEVICES = {}


def torch_module():
    """torch, or ``None`` when it is missing or sees no device — the runtime then allocates."""
    try:
        import torch  # noqa: PLC0415

        return torch if torch.cuda.is_available() else None
    except ImportError:
        return None


def _ordinal():
    torch = torch_module()
    return torch.cuda.current_device() if torch is not None else 0


def device():
    """The host's current logical CUDA device, cached independently of other devices."""
    ordinal = _ordinal()
    if ordinal not in _DEVICES:
        from emmy import emmy_runtime  # noqa: PLC0415 — the extension loads the driver lazily

        _DEVICES[ordinal] = emmy_runtime.Device(ordinal)
    return _DEVICES[ordinal]


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
    """Whether the selected device's existing context is in a sticky-error state.

    ``False`` when no context was ever created here. A synchronize surfaces the sticky
    status an earlier illegal access left behind; it would block on a hung kernel, so a
    caller that knows a kernel hung must not probe."""
    if not _DEVICES:
        return False
    try:
        current = _DEVICES.get(_ordinal())
        if current is not None:
            current.synchronize()
    except Exception:  # noqa: BLE001 — any driver error here means the context is unusable
        return True
    return False
