"""The feature row of a measured ``perf`` row — the one reading of a measurement the measured pools are built from."""

from __future__ import annotations

import functools

from emmy.compiler.pipeline.search.features import Featurizer


@functools.cache
def _card_context(gpu_name: str, cc: int):
    """The context of ``gpu_name`` at compute capability ``cc`` (``H_cc`` encoding), from the registry's memorized
    specs — a row's own card's, never the live device's."""
    from emmy.compiler.context import Context  # noqa: PLC0415

    return Context.from_target(divmod(cc, 10), gpu_name=gpu_name)


def measured_features(row, kernel) -> dict[str, float]:
    """The feature row of a measured ``perf`` row of ``kernel`` (its op): its schedule row, under its card's
    ``H_*`` and the opt level it was measured under. A row stores no feature of its own — the ``H_*`` are a function
    of the card and the ``S_*`` of the kernel, both computed here by live code, so a stored row outlives a change
    to either. Only for rows :func:`~.freeze.freeze_reason` admits (a registry card)."""
    ctx = _card_context(row.gpu, row.cc)
    return Featurizer({**ctx.features(), "H_opt": float(row.opt)}, ctx).features(kernel, row.knobs)
