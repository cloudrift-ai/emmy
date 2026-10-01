"""Golden file: a card's measurements in the tune DB's shape — the file format (``format``), the copy into the DB a
compile reads (``evidence``), the repository index and the evidence scope (``repository``), the rewrite onto a fresh
lowering (``restamp``) and the working golden's writers (``working``). One module per job; this package is the
public surface."""

from .evidence import evidence_db, file_source, import_file, import_rows, regime_context, regime_live
from .format import GoldenFile, Kernel, Latency, Measurements, Row, prepare_traced_graph, program_text
from .repository import (
    document_of,
    documents_for_card,
    evidence_scope,
    is_repository_golden_path,
    live_gpu_key,
    repository_documents,
    repository_golden_paths,
    scope_digest,
    scope_explicit,
    sole_evidence,
)
from .restamp import Report, definition, lift_targets, mint, restamp
from .working import (
    TraceInventoryResult,
    append_trace_inventory,
    inventory,
    preflight_trace_inventory,
    record_greedy_pick,
    record_latency,
    validate_working_gpu,
    whole_origins,
    write_trace_inventories,
    write_trace_inventory,
)

__all__ = [
    "GoldenFile",
    "Kernel",
    "Latency",
    "Measurements",
    "Report",
    "Row",
    "TraceInventoryResult",
    "append_trace_inventory",
    "definition",
    "document_of",
    "documents_for_card",
    "evidence_db",
    "evidence_scope",
    "file_source",
    "import_file",
    "import_rows",
    "inventory",
    "is_repository_golden_path",
    "lift_targets",
    "live_gpu_key",
    "mint",
    "preflight_trace_inventory",
    "prepare_traced_graph",
    "program_text",
    "record_greedy_pick",
    "record_latency",
    "regime_context",
    "regime_live",
    "repository_documents",
    "repository_golden_paths",
    "restamp",
    "scope_digest",
    "scope_explicit",
    "sole_evidence",
    "validate_working_gpu",
    "whole_origins",
    "write_trace_inventories",
    "write_trace_inventory",
]
