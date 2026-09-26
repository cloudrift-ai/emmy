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
    prepare_traced_graph,
    program_text,
)
from .record import (
    GoldenRecord,
    kernel_set_pins,
    regime_pins,
    shared_regime_pins,
)
from .repository import (
    golden_records,
    goldens_for_live_gpu,
    is_repository_golden_path,
    records_for_card,
    records_override,
    scope_digest,
    scope_explicit,
    sole_evidence,
)

__all__ = [
    "GoldenRecord",
    "kernel_set_pins",
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
    "prepare_traced_graph",
    "program_text",
    "decode_record",
    "lead_of",
    "piece_row",
    "siblings_of",
    "unmatched_reason",
    "golden_records",
    "goldens_for_live_gpu",
    "is_repository_golden_path",
    "records_for_card",
    "records_override",
    "scope_digest",
    "scope_explicit",
    "sole_evidence",
]
