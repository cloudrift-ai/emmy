"""Golden file: the evidence store — its file format, the record consumers read and the set they read together, its
strict decode, the seam that files its rows into the tune DB a compile reads (``evidence``), the repository index,
and the working golden's writers (``working``). The check and restamp of a golden against the fresh lowering of its
programs is ``restamp``, imported by module: its function goes by the module's name. One module per job; this
package is the public surface."""

from .decode import (
    decode_record,
    piece_row,
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
    GoldenRecords,
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
from .working import (
    TraceInventoryResult,
    append_trace_inventory,
    greedy_pick_rows,
    kernel_programs,
    kernel_set_prices,
    lowered_kernels,
    preflight_trace_inventory,
    record_greedy_pick,
    record_latency,
    validate_working_gpu,
    write_trace_inventories,
    write_trace_inventory,
)

__all__ = [
    "GoldenRecord",
    "GoldenRecords",
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
    "piece_row",
    "unmatched_reason",
    "golden_records",
    "goldens_for_live_gpu",
    "is_repository_golden_path",
    "records_for_card",
    "records_override",
    "scope_digest",
    "scope_explicit",
    "sole_evidence",
    "TraceInventoryResult",
    "append_trace_inventory",
    "greedy_pick_rows",
    "kernel_programs",
    "kernel_set_prices",
    "lowered_kernels",
    "preflight_trace_inventory",
    "record_greedy_pick",
    "record_latency",
    "validate_working_gpu",
    "write_trace_inventories",
    "write_trace_inventory",
]
