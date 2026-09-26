"""Golden YAML: the evidence store — its file format, the record consumers read, its strict decode, the seam that
files its rows into the tune DB a compile reads (``evidence``), and the repository index. One module per job; this
package is the public surface."""

from .decode import (
    decode_record,
    lead_of,
    piece_row,
    siblings_of,
    unmatched_reason,
)
from .format import (
    Config,
    GoldenEntryState,
    GoldenFile,
    Latency,
    Measurements,
    Realization,
    Target,
    kernel_pool_text,
    program_text,
)
from .identity import kernel_identity
from .record import (
    GoldenRecord,
    fast_math_knobs,
    kernel_set_pins,
    pins_freeze_cut,
    precision_trading_pins,
    regime_live,
    regime_pins,
    shared_regime_pins,
)
from .repository import (
    GOLDEN_RECORDS,
    goldens_by_name,
    goldens_for_live_gpu,
    is_repository_golden_path,
    live_recorded_goldens,
    records_for_card,
    records_override,
    scope_digest,
    scope_explicit,
    sole_evidence,
)

__all__ = [
    "GoldenRecord",
    "fast_math_knobs",
    "kernel_set_pins",
    "pins_freeze_cut",
    "precision_trading_pins",
    "regime_live",
    "regime_pins",
    "shared_regime_pins",
    "Config",
    "GoldenEntryState",
    "GoldenFile",
    "Latency",
    "Measurements",
    "Realization",
    "Target",
    "kernel_pool_text",
    "program_text",
    "kernel_identity",
    "decode_record",
    "lead_of",
    "piece_row",
    "siblings_of",
    "unmatched_reason",
    "GOLDEN_RECORDS",
    "goldens_by_name",
    "goldens_for_live_gpu",
    "is_repository_golden_path",
    "live_recorded_goldens",
    "records_for_card",
    "records_override",
    "scope_digest",
    "scope_explicit",
    "sole_evidence",
]
