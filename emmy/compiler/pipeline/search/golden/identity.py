"""A record's kernel identity under the current compiler."""

from __future__ import annotations

from .record import GoldenRecord, _lifted_target


def kernel_identity(record: GoldenRecord) -> str | None:
    """The record's kernel identity under the CURRENT compiler — the strict decode's and the drift
    key (``identity_key(with_io=True)``). A STORED identity is returned as-is: it is how a
    child-identity receipt names the one split child its schedule decorates (the target's own lift
    stops at the pre-cut kernel and cannot say), and a stale stored identity selects nothing — the
    strict decode is where that fails loudly. Without one, the identity is derived as the lift of the
    record's ONE target kernel, through the exact total lift the live compile uses
    (``_fromloop.lift_loop_op``). ``None`` when the record cannot carry a deploy identity: the target
    lowers to several kernels (a schedule row decorates exactly one), or selection/lifting fails —
    best-effort here (a corpus row must never break a compile); nightly strict decoding is where
    failure is loud. Deploy never joins on this key: a record deploys as measured rows, matched by
    ``S_*`` features plus the exact ``I_kernel`` stamp, off the rows the golden import files
    (``golden.evidence``)."""
    if record.identity is not None:
        return record.identity
    try:
        return _lifted_target(record).identity_key(with_io=True)
    except Exception:  # noqa: BLE001 — see the docstring; the decode tripwire re-derives loudly
        return None
