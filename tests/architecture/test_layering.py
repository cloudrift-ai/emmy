"""Layering guards — static import checks that lock in dialect boundaries.

These tests don't exercise behavior. They scan source files for forbidden
import patterns that would silently undo earlier refactors. Fast, no
GPU, no test fixtures — they just open files and grep.

Update / extend when adding a new layered subsystem; add a new test in
this file rather than scattering one-off greps elsewhere.
"""

from __future__ import annotations

import ast
import functools
import pathlib
import re

# Repo root: tests/architecture/ → tests/ → repo
_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

_POST_DECOMPOSITION_ROOTS = (
    "emmy/config.py",
    "emmy/compiler/backend",
    "emmy/compiler/dtype.py",
    "emmy/compiler/ir/kernel",
    "emmy/compiler/ir/loop",
    "emmy/compiler/ir/stmt",
    "emmy/compiler/ir/tile",
    "emmy/compiler/pipeline/passes/loop",
    "emmy/compiler/pipeline/passes/lowering",
)


def test_checkpoint_format_names_stop_at_frontend_decomposition() -> None:
    """Checkpoint-format implementation details may not enter downstream dialects or passes."""
    forbidden = re.compile(r"\b(?:trellis|exl3)\b", re.IGNORECASE)
    offenders: list[str] = []
    for relative in _POST_DECOMPOSITION_ROOTS:
        root = _REPO_ROOT / relative
        paths = sorted(root.rglob("*.py")) if root.is_dir() else [root]
        for path in paths:
            for lineno, line in enumerate(path.read_text().splitlines(), start=1):
                if forbidden.search(line):
                    offenders.append(f"{path.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "checkpoint-format names must stop at frontend decomposition; downstream code and comments "
        "may contain only generic IR:\n" + "\n".join(offenders)
    )


def test_lowering_tile_does_not_import_kernel_ir() -> None:
    """``tile/**/*.py`` may not import from ``ir.kernel.ir``.

    The Tile-IR / Kernel-IR boundary is: Tile passes encode scheduling
    *decisions* (``StagePolicy``, ``AsyncWait``, ``WarpSpecialize``);
    Kernel-IR carries hardware *primitives* (``Sync``, ``Smem``,
    ``Mbarrier*``, ``SetMaxNReg``, …) that the materializer
    (``100_materialize_tile``) emits. A Tile pass that imports
    Kernel-IR types is fabricating Kernel stmts inside a Tile body —
    exactly what PR #166 did to ``085_warp_specialize`` and what the
    WarpSpecialize Statement refactor undid.

    If this test fires, either:
    1. you genuinely need a new Tile-level abstraction (add a Statement to
       ``ir/tile/ir.py``, lower it in ``100_materialize_tile``), or
    2. you're in the wrong directory — Kernel-IR-emitting passes live
       under ``lowering/kernel/``.
    """
    tile_dir = _REPO_ROOT / "emmy" / "compiler" / "pipeline" / "passes" / "tile"
    assert tile_dir.is_dir(), f"tile/ not found at {tile_dir}"
    forbidden = "from emmy.compiler.ir.kernel"
    offenders: list[str] = []
    for py in sorted(tile_dir.rglob("*.py")):
        text = py.read_text()
        for lineno, line in enumerate(text.splitlines(), start=1):
            if forbidden in line:
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "tile/**/*.py must not import from ir.kernel — layering violation.\n" + "\n".join(offenders)


def test_lowering_tile_does_not_import_kernel_passes() -> None:
    """``tile/**/*.py`` may not import from the ``lowering/kernel`` pass layer.

    The tile layer (``enumeration`` + ``assembly`` + ``split``) runs ABOVE the
    kernel pass layer; a tile pass importing ``lowering.kernel`` is a back-edge in
    the pass DAG — a tile pass depending on a downstream kernel pass's internals.
    What the two layers genuinely share lives in the ``lowering/`` root modules —
    ``_addr`` (addressing algebra: ``gmem_axis_step`` / ``gmem_row_stride``, ``BYTE_SLAB_PAD``) —
    layout facts imported by both layers.

    If this fires: move the shared helper into a ``lowering/`` root module and import
    it there from both layers, rather than reaching down into ``lowering/kernel``.
    """
    tile_dir = _REPO_ROOT / "emmy" / "compiler" / "pipeline" / "passes" / "tile"
    assert tile_dir.is_dir(), f"tile/ not found at {tile_dir}"
    forbidden = "emmy.compiler.pipeline.passes.lowering.kernel"
    offenders: list[str] = []
    for py in sorted(tile_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden in line and "import" in line:
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "tile/**/*.py must not import from lowering/kernel — back-edge in the pass DAG.\n"
        "Shared structural predicates live in lowering/_predicates.\n" + "\n".join(offenders)
    )


def test_ir_does_not_import_pipeline() -> None:
    """``ir/`` defines the dialects, body normalization and the schedule model; the pipeline consumes them.

    Knobs, pins and evidence live in the pipeline, so nothing an IR module computes — a canonical
    form, an identity, a schedule domain — can depend on a tuning choice.
    """
    ir_dir = _REPO_ROOT / "emmy" / "compiler" / "ir"
    forbidden = re.compile(r"^\s*(?:from|import)\s+emmy\.compiler\.pipeline\b")
    offenders = []
    for py in sorted(ir_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden.search(line):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "ir/ must not import the pipeline layer:\n" + "\n".join(offenders)


def test_normalization_has_one_entry_point() -> None:
    """Body normalization is the canonical form every Loop IR body and every identity digest takes; it answers to no
    knob, pin or evidence. It lives in the ``emmy-ir`` crate behind one entry point, ``normalize_body``; the package
    exports beside it only the three steps a pass may run on its own, and holds no transform of its own."""
    import emmy.compiler.ir.stmt as stmt

    steps = {name for name in stmt.__all__ if name in ("normalize_body", "fold_affine", "simplify_body", "prepare_body")}
    assert steps == {"normalize_body", "fold_affine", "simplify_body", "prepare_body"}
    package = _REPO_ROOT / "emmy" / "compiler" / "ir" / "stmt"
    assert sorted(path.name for path in package.glob("*.py")) == ["__init__.py", "glue.py"], "the statement package is its exports and the glue"
    crate = (_REPO_ROOT / "crates" / "emmy-ir" / "src" / "normalize" / "mod.rs").read_text()
    assert "pub fn normalize_body(" in crate


_STMT_PACKAGE = "emmy.compiler.ir.stmt"


def _imports(path: pathlib.Path):
    """Every ``(module, names)`` a file imports, function-local imports included."""
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.ImportFrom) and node.module:
            yield node.module, [alias.name for alias in node.names]
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name, []


def test_the_statement_module_is_entered_through_its_package() -> None:
    """Outside ``ir/stmt``, the compiler and its tests import the statement package's exported names, nothing else.

    A submodule path or a private name is a dependency on how the module is built. The package's export list is the
    whole interface, so the module can be rearranged, or replaced, behind it. The tests of how it is built are the
    one exception, and they sit in one directory that leaves with the implementation.
    """
    import emmy.compiler.ir.stmt as stmt

    inside = (_REPO_ROOT / "emmy" / "compiler" / "ir" / "stmt", _REPO_ROOT / "tests" / "compiler" / "ir" / "stmt" / "implementation")
    offenders = []
    for py in sorted([*(_REPO_ROOT / "emmy").rglob("*.py"), *(_REPO_ROOT / "tests").rglob("*.py")]):
        if any(root in py.parents for root in inside):
            continue
        for module, names in _imports(py):
            if module.startswith(_STMT_PACKAGE + ".") or (module == _STMT_PACKAGE and not set(names) <= set(stmt.__all__)):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}: from {module} import {', '.join(names)}")
    assert not offenders, "import these from the statement package's exported names:\n" + "\n".join(offenders)


def test_the_statement_module_exports_a_pinned_list() -> None:
    """The statement module's interface is this list. A name added here is a decision, not a convenience."""
    import emmy.compiler.ir.stmt as stmt

    kinds = ["Statement", "Load", "Assign", "Affine", "Accum", "Carry", "Pre", "Init", "Let", "Write", "Select", "SelectBranch"]
    kinds += ["ZeroPrologue", "Loop", "StridedLoop", "Cond"]
    containers = ["Body", "Identity", "OutputSpec"]
    functions = ["normalize_body", "canonicalize_identity", "free_names", "exposed_defines", "rename_free", "binary"]
    functions += ["mask_select_predicate", "pretty_body", "parse_body", "ordering_constraints", "dtype_promote"]
    # Normalization steps a pass calls on its own, and the two kinds fusion assembles a region with.
    steps = ["fold_affine", "simplify_body", "prepare_body"]
    assembly = ["Call", "Subroutine"]
    # The names of a kind's fields, for the one pass that generalizes whole statements.
    declaring = ["fields_of"]
    assert sorted(stmt.__all__) == sorted(kinds + containers + functions + steps + assembly + declaring)


def test_the_statement_module_depends_on_expressions_and_values_only() -> None:
    """``ir/stmt`` is the vocabulary every IR shares, so it sits below all of them: it imports the extension that
    implements it, expressions, axes, dtypes and the structural and wire forms. No op, tensor, graph, backend,
    renderer or pipeline."""
    allowed = (
        _STMT_PACKAGE,
        "emmy.emmy_ir",
        "emmy.compiler.ir.expression",
        "emmy.compiler.ir.axis",
        "emmy.compiler.ir.sigma",
        "emmy.compiler.ir.elementwise",
        "emmy.compiler.dtype",
        "emmy.compiler.dim",
        "emmy.compiler.structural",
        "emmy.compiler.wire",
        "emmy.utils",
    )
    offenders = []
    for py in sorted((_REPO_ROOT / "emmy" / "compiler" / "ir" / "stmt").glob("*.py")):
        for module, _names in _imports(py):
            if module.startswith("emmy") and not any(module == root or module.startswith(root + ".") for root in allowed):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}: {module}")
    assert not offenders, "the statement module must not depend on these:\n" + "\n".join(offenders)


def test_a_statement_is_copied_and_read_through_its_own_methods() -> None:
    """Outside ``ir/stmt``, nothing edits a statement with ``dataclasses.replace`` (``Statement.replace`` is the one way)
    and nothing lists a statement's dataclass fields: a caller reads a statement through the statement protocol, so
    a kind need not be a dataclass.

    The class such a call receives is not visible in the source, so this lowers the first corpus case of each
    family and watches the calls. The wire form and the structural form may list fields: they are where a
    statement's class states them, and they read a dataclass kind on its behalf.
    """
    import dataclasses
    import sys

    from emmy.compiler.ir.stmt import SelectBranch, Statement
    from tests.compiler.realization import helpers

    kinds = (Statement, SelectBranch)
    inside = str(_REPO_ROOT / "emmy" / "compiler" / "ir" / "stmt")
    forms = {str(_REPO_ROOT / "emmy" / "compiler" / name) for name in ("wire.py", "structural.py")} | {dataclasses.__file__}
    listers = {dataclasses.fields.__code__: "class_or_instance", dataclasses.asdict.__code__: "obj", dataclasses.astuple.__code__: "obj"}
    copies: set[str] = set()
    listings: set[str] = set()

    def watch(frame, event, _arg):  # noqa: ANN001, ANN202 — the ``sys.setprofile`` signature
        if event != "call" or not (frame.f_code is dataclasses.replace.__code__ or frame.f_code in listers):
            return
        subject = frame.f_locals.get(listers.get(frame.f_code, "obj"))
        caller = frame.f_back.f_code.co_filename
        if not (isinstance(subject, kinds) or (isinstance(subject, type) and issubclass(subject, kinds))) or caller.startswith(inside):
            return
        if frame.f_code not in listers:
            copies.add(f"{pathlib.Path(caller).relative_to(_REPO_ROOT)}:{frame.f_back.f_lineno}")
        elif caller not in forms:
            listings.add(f"{pathlib.Path(caller).relative_to(_REPO_ROOT)}:{frame.f_back.f_lineno}")

    cases = [helpers.load_case(sorted(family.glob("*.json"))[0]) for family in sorted(helpers.CASES_DIR.iterdir()) if family.is_dir()]
    previous = sys.getprofile()
    sys.setprofile(watch)
    try:
        for case in cases:
            helpers.lowered(case, case.context())
    finally:
        sys.setprofile(previous)
    assert not copies, "copy these statements with ``stmt.replace(...)``:\n" + "\n".join(sorted(copies))
    assert not listings, "read these statements through the statement protocol:\n" + "\n".join(sorted(listings))


def test_a_class_is_on_the_wire_by_what_it_registers() -> None:
    """Nothing under ``emmy/`` asks whether a value inherits ``Wire``, and only the mixin's own defaults read a
    dataclass field for the wire: a class joins through ``wire.register`` with its tag, its two wire functions and
    its parts, so one that cannot be a dataclass or inherit a Python base travels like any other."""
    by_base = re.compile(r"\b(?:isinstance|issubclass)\([^()]*\bWire\b")
    offenders = [
        f"{py.relative_to(_REPO_ROOT)}:{number}"
        for py in sorted((_REPO_ROOT / "emmy").rglob("*.py"))
        for number, line in enumerate(py.read_text().splitlines(), 1)
        if by_base.search(line)
    ]
    assert not offenders, "ask the wire registry, not the base class:\n" + "\n".join(offenders)
    wire = ast.parse((_REPO_ROOT / "emmy" / "compiler" / "wire.py").read_text())
    reflecting = {
        function.name
        for function in ast.walk(wire)
        if isinstance(function, ast.FunctionDef)
        for node in ast.walk(function)
        if isinstance(node, ast.Name) and node.id in ("fields", "is_dataclass")
    }
    assert reflecting == {"wire_parts", "_wire_fields"}


# ---------------------------------------------------------------------------
# Fix 1 firewall — keep the
# assemble→TileOp boundary clean: every benched scheduling choice lives in the
# enumeration ``Schedule`` (above assemble); every kernel pass is *mechanical*
# (it lowers the stamped ``TileOp``, it never re-derives a scheduling decision).
# These two guards are the plan's "structural firewall": without them Fix 2's
# "mechanical kernel passes" rule is unenforceable and a new kernel pass can
# silently re-accrete the schedule-lives-in-the-tree-shape brittleness the
# block-DAG refactor removed upstream.
# ---------------------------------------------------------------------------

_KERNEL_DIR = _REPO_ROOT / "emmy" / "compiler" / "pipeline" / "passes" / "lowering" / "kernel"

# The enumeration / split move-composer layer: where the *forks* live — offer
# enumeration, knob schema, and the algebra classifiers. A
# kernel pass importing any of these is reaching above assemble to (re-)make a
# scheduling decision instead of reading a stamped fact off the ``TileOp``.
_FORBIDDEN_IMPORTS = (
    "passes.tile.enumeration",
    "passes.tile.split",
)

# Schedule-decision functions (offer enumeration / algebra classifiers). A
# kernel pass must not *call* one — not even a re-rolled
# local copy. ``classify_fragment_epilogue`` (``lowering/_predicates.py``) is the
# one allowed ``classify*`` name: it is a mechanical fragment-epilogue lowering
# analysis shared by both layers, not a ranked scheduling choice.
_FORBIDDEN_CALL_RE = re.compile(
    r"\b("
    r"\w*_offers"  # thread_offers / map_reg_offers / warp_bk_offers / coop_reduce_offers / …
    r"|stage_candidates|legal_decomps|eligible_atoms"
    r"|classify_algebra|classify_matmul_operands"
    r")\b"
)
_ALLOWED_CALL_NAMES = frozenset({"classify_fragment_epilogue"})


def _kernel_pass_files() -> list[pathlib.Path]:
    assert _KERNEL_DIR.is_dir(), f"lowering/kernel/ not found at {_KERNEL_DIR}"
    return sorted(_KERNEL_DIR.glob("*.py"))


def test_lowering_kernel_does_not_import_enumeration() -> None:
    """``lowering/kernel/*.py`` may not import from the enumeration / split layer.

    Fix 1's import half of the firewall. The enumeration passes
    (``tile/enumeration/`` + ``tile/split/``) own the *forks* — the offer/knob/
    move/classifier modules the search benches over. assemble renders the chosen
    ``Schedule`` into a ``TileOp``; from there every kernel pass is mechanical.
    A kernel pass that imports an offer / knob / cut / classifier module is
    making (or re-deriving) a scheduling decision below assemble — the exact leak
    this fix closes.

    If this fires: the fact you need is a *scheduling decision* — stamp it on the
    ``TileOp`` node in ``assembly/`` and read the attribute here, rather than
    importing the enumeration module that decides it.
    """
    offenders: list[str] = []
    for py in _kernel_pass_files():
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if any(mod in line for mod in _FORBIDDEN_IMPORTS):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "lowering/kernel/*.py must not import from the enumeration/split fork layer — "
        "a kernel pass reads stamped TileOp attributes, it does not re-decide the schedule.\n" + "\n".join(offenders)
    )


def test_lowering_kernel_calls_no_schedule_classifier() -> None:
    """No ``lowering/kernel/*.py`` calls a schedule classifier / offer fn.

    Fix 1's call half of the firewall — defense in depth over the import guard,
    and it also catches a re-rolled *local* copy of a classifier (no import to
    flag). A kernel pass may read a stamped ``TileOp`` attribute (``StageBundle.
    policy``, ``SerialTile.kind``, ``WarpSpecialize.ring_depth``, …) but must not
    invoke the offer enumeration or algebra classification
    that produced those facts — those are mechanical-pass discipline violations.
    """
    offenders: list[str] = []
    for py in _kernel_pass_files():
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            for m in _FORBIDDEN_CALL_RE.finditer(line):
                if m.group(1) not in _ALLOWED_CALL_NAMES:
                    offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
                    break
    assert not offenders, (
        "lowering/kernel/*.py must not call a schedule classifier / offer fn — "
        "read the stamped TileOp attribute instead of re-deriving the decision.\n" + "\n".join(offenders)
    )


def test_search_dataset_does_not_import_the_prior() -> None:
    """``search/dataset/*.py`` may not import from ``search.prior``.

    The data layer describes candidates and their labels — a :class:`Group` is a packed pool plus what
    is known about its rows, and that is the same object whether a trainer, a fold harness or an
    evaluation report is asking. Deciding what a SCORE means is the layer above: a dataset that imported
    the model would be making that decision for it.

    If this fires, the fact you reached for is either feature SPELLING (``search/features.py``, imported
    by both layers) or a model DECISION that belongs to its caller — pass it in.
    """
    data_dir = _REPO_ROOT / "emmy" / "compiler" / "pipeline" / "search" / "dataset"
    assert data_dir.is_dir(), f"search/dataset/ not found at {data_dir}"
    # Both spellings, because the repo uses both: the dotted path (``from ….search.prior.offline import X``)
    # and the package-import idiom (``from ….search import prior``) that ``commands/fit.py`` already writes for
    # ``features``. A guard that caught only the first would wave through the form a future violation is most
    # likely to be written in.
    forbidden = re.compile(r"^\s*(?:from|import)\s+emmy\.compiler\.pipeline\.search(?:\.prior\b|\s+import\s+\(?\s*prior\b)")
    offenders: list[str] = []
    for py in sorted(data_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden.match(line):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, (
        "search/dataset/*.py must not import search.prior — the data layer carries columns and labels; "
        "each model class decides for itself which columns it wants.\n" + "\n".join(offenders)
    )


_IDENTITY_HOME = "emmy/compiler/ir/base.py"

#: The identity interface methods and the modules allowed to define/override each. ``Op`` (the
#: home) declares ONE public function — ``identity_key`` (``CudaOp`` overrides it whole with its
#: rendered-source digest); ``_body_identity`` is the ONE extension point (``BodyOp`` answers
#: with its stored body, ``TileOp`` with its derived ``loop_body``), and ``hint_extent`` is the
#: ``Axis`` property the scheduler sizes against. ``deploy_identity`` / ``cache_key`` /
#: ``pool_key`` are retired lattice points, spelled at call sites via ``identity_key`` flags (or
#: minted at the scheduler's one site) — a method of any of those names reappearing anywhere is a
#: second spelling.
_IDENTITY_DEFINERS = {
    "identity_key": {_IDENTITY_HOME, "emmy/compiler/ir/cuda/ir.py"},
    "body_identity": {_IDENTITY_HOME, "emmy/compiler/ir/body_op.py", "emmy/compiler/ir/tile/ir.py"},
    "io_fingerprint": {_IDENTITY_HOME},
    "deploy_identity": set(),
    "cache_key": set(),
    "pool_key": set(),
    "hint_extent": {"emmy/compiler/ir/axis.py"},
}


def test_kernel_identity_is_not_redefined_outside_its_home() -> None:
    """No module outside the home may DEFINE one of the identity readings.

    Identity has several legitimate meanings — the algebra digest, the deploy join key, the
    schedule-space key, the ``S_*`` row — and each is answered once, on the ``Op`` interface,
    computed from the op's complete Loop-IR body and its io buffers. What this forbids is a
    second copy of one of THESE readings: a key re-derived beside its caller drifts from the
    thing it identifies, silently, and the failure is two kernels sharing a cache entry or a
    golden record joining the wrong shape.

    ``pool_key`` shipped omitting per-axis extents for exactly that reason — it lived next to the
    enumeration it keyed, read as a local memo detail, and nobody treated it as a contract with
    three external consumers.

    Deliberately narrow: it matches the interface's own names, not every function whose name
    contains "identity" or "sig". A guard that flagged those would be exempted rather than
    obeyed, and the exemption list would become the real rule.
    """
    forbidden = re.compile(r"^\s*def\s+_?(" + "|".join(_IDENTITY_DEFINERS) + r")\s*\(")
    offenders: list[str] = []
    for py in sorted((_REPO_ROOT / "emmy").rglob("*.py")):
        relative = py.relative_to(_REPO_ROOT).as_posix()
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            m = forbidden.match(line)
            if m and relative not in _IDENTITY_DEFINERS[m.group(1)]:
                offenders.append(f"{relative}:{lineno}: {line.strip()}")
    assert not offenders, (
        f"these readings live on the Op interface ({_IDENTITY_HOME}); call the method instead of "
        "re-deriving. A fact a schedule reads that the key misses is a modeling gap to fix in the "
        "op's loop body or io, never a side-channel fingerprint.\n" + "\n".join(offenders)
    )


def test_a_golden_file_stores_inputs_only() -> None:
    """A golden file and a corpus case hold inputs — the Loop IR, the decisions, the measurements — and nothing
    computed from them: no kernel identity, no ``S_*`` stamp. A stored copy of a computed value has to be kept in
    agreement with the computation, and every time the two drifted a row stopped pricing its own fork with nothing
    reporting it. So the field lists are pinned here: a new field is an input, or it does not go in the file.

    The other half of the invariant — an op's knobs hold decisions only — runs the compiler, so it lives in
    ``tests/compiler/pipeline/test_strategies.py``."""
    from dataclasses import fields

    from emmy.compiler.pipeline.search.db import RoutingRow
    from emmy.compiler.pipeline.search.golden import GoldenFile, Kernel, Latency, Measurements, Row

    assert [f.name for f in fields(Kernel)] == ["loop_ir", "name", "formed", "key", "traced", "origins", "bindings"]
    assert [f.name for f in fields(Row)] == ["name", "kernel", "bindings", "pins", "knobs", "measurements", "latency", "note"]
    assert [f.name for f in fields(RoutingRow)] == ["parent", "arm", "children"]
    assert [f.name for f in fields(Measurements)] == ["emmy_us", "reference_us", "reference_backend", "tried"]
    assert [f.name for f in fields(Latency)] == ["emmy_us", "tcompile_us", "eager_us", "kernels"]
    assert [f.name for f in fields(GoldenFile)] == [
        "gpu_name",
        "compute_cap",
        "model",
        "model_quant_digest",
        "note",
        "programs",
        "kernels",
        "routing",
        "rows",
    ]
    cases = sorted((_REPO_ROOT / "tests/compiler/realization/cases").rglob("*.json"))
    assert cases, "the realization corpus is empty"
    computed = re.compile(r'"(?:exact_identity|structural_identity|stamps)"\s*:')
    assert not [path.relative_to(_REPO_ROOT).as_posix() for path in cases if computed.search(path.read_text())]


def test_nothing_reaches_into_the_scheduler_for_identity() -> None:
    """Identity readers import ``ir/tile/identity``, never a lowering pass's private module.

    ``search/golden.py`` and ``search/policy/greedy.py`` used to import ``deploy_identity`` and
    ``pool_key`` from the private tile scheduler — inside functions, with ``noqa: PLC0415``,
    because the module-level direction is a cycle (the scheduler imports ``search.space``). A
    deferred import with a lint suppression is what a layering inversion looks like when the code
    has not moved yet.
    """
    forbidden = re.compile(r"lowering\.tile\._classic\s+import\s+.*\b(?:deploy_identity|pool_key)\b")
    offenders: list[str] = []
    for py in sorted((_REPO_ROOT / "emmy").rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden.search(line):
                offenders.append(f"{py.relative_to(_REPO_ROOT).as_posix()}:{lineno}: {line.strip()}")
    assert not offenders, "read kernel identity off the Op interface (Op.deploy_identity)\n" + "\n".join(offenders)


def test_every_buffer_bearing_stmt_can_rename_its_buffers() -> None:
    """``Statement.rename_buffers`` is the setter counterpart of ``external_reads`` /
    ``external_writes``: a stmt kind that DECLARES external buffers without knowing how to
    rename them silently breaks every buffer rebind (structural identity, the session kernel
    cache) the day it first appears inside a body those walk. Wrapper stmts whose
    declarations aggregate a nested body are exempt — ``Body.rename_buffers`` reaches their
    leaves through the recursive map."""
    import importlib
    import pkgutil

    import emmy.compiler.ir as ir_pkg
    from emmy.compiler.ir.stmt import Statement

    for mod in pkgutil.walk_packages(ir_pkg.__path__, prefix="emmy.compiler.ir."):
        importlib.import_module(mod.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    offenders = [
        cls.__name__
        for cls in subclasses(Statement)
        if ("external_reads" in cls.__dict__ or "external_writes" in cls.__dict__)
        and "rename_buffers" not in cls.__dict__
        and not cls.__dict__.get("nested", None)  # wrapper aggregators rename through their bodies
    ]
    assert not offenders, (
        "these stmt kinds declare external buffers but cannot rename them — add a "
        "rename_buffers override beside the declaration:\n" + "\n".join(sorted(offenders))
    )


def _kernel_kinds() -> list[type]:
    import emmy.compiler.ir.kernel.ir as kernel_ir
    from emmy.compiler.ir.stmt import Statement

    return [
        cls
        for cls in vars(kernel_ir).values()
        if isinstance(cls, type) and issubclass(cls, Statement) and cls.__module__ == kernel_ir.__name__
    ]


def test_a_kernel_ir_kind_is_declared_not_programmed() -> None:
    """A Kernel IR statement kind is data: fields with their roles, how it prints, and at most a constructor check,
    declared in the emmy-ir crate with the shared kinds. The statement protocol follows from the roles; what rendering
    or a lowering asks of a statement beyond that is a function over it, beside the render functions, never a
    Python member attached to the class."""
    import types

    kinds = _kernel_kinds()
    offenders = [
        f"{cls.__name__}.{name}"
        for cls in kinds
        for name, member in vars(cls).items()
        if isinstance(member, (types.FunctionType, property, functools.cached_property, staticmethod, classmethod))
    ]
    assert len(kinds) == 30 and all(cls.__module__ == "emmy.compiler.ir.kernel.ir" and "_fields" in vars(cls) for cls in kinds)
    assert not offenders, "state these as field roles, or move them to functions over the statement:\n" + "\n".join(sorted(offenders))


def test_every_op_is_a_frozen_dataclass() -> None:
    """``Op`` immutability is the type, not a convention: every ``Op`` dataclass is
    ``frozen=True``, its maps land as ``frozendict`` (``Op.__post_init__``), and mutation-shaped
    bugs (a stamp editing a shared instance, a cache outdated by a silent write) are
    unrepresentable. A rewrite is always a ``replace`` + node rebind."""
    import importlib
    import pkgutil

    import emmy.compiler.ir as ir_pkg
    from emmy.compiler.ir.base import Op

    for mod in pkgutil.walk_packages(ir_pkg.__path__, prefix="emmy.compiler.ir."):
        importlib.import_module(mod.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    offenders = [
        cls.__name__ for cls in {Op, *subclasses(Op)} if hasattr(cls, "__dataclass_params__") and not cls.__dataclass_params__.frozen
    ]
    assert not offenders, "every Op dataclass must be frozen=True:\n" + "\n".join(sorted(offenders))


def test_the_op_identity_surface_is_exactly_identity_key() -> None:
    """One identity function. Any public method on any ``Op`` whose name says identity or key —
    beyond ``identity_key`` itself — is a second spelling waiting to drift; the retired names
    (``deploy_identity`` / ``cache_key`` / ``pool_key`` / ``structural_key``-on-ops) must not
    return under new spellings either."""
    import importlib
    import pkgutil
    import re as _re

    import emmy.compiler.ir as ir_pkg
    from emmy.compiler.ir.base import Op

    for mod in pkgutil.walk_packages(ir_pkg.__path__, prefix="emmy.compiler.ir."):
        importlib.import_module(mod.name)

    def subclasses(cls):
        for sub in cls.__subclasses__():
            yield sub
            yield from subclasses(sub)

    pattern = _re.compile(r"identity|(?:^|_)key")
    offenders = {
        f"{cls.__name__}.{name}"
        for cls in {Op, *subclasses(Op)}
        for name, value in vars(cls).items()
        if callable(value)
        and not name.startswith(("_", "is_"))  # ``IndexMapOp.is_identity`` is the identity MAP predicate, not a key
        and pattern.search(name)
        and name != "identity_key"
    }
    assert not offenders, "the public Op identity surface is identity_key alone:\n" + "\n".join(sorted(offenders))


def test_search_dataset_does_not_import_search_db() -> None:
    """``search/dataset/*.py`` may not import ``search.db``: the dataset describes candidates and their labels and
    travels as a directory; reading a DB is the export's job (``search/db/export.py``), the one place the two
    packages meet, and the one direction they do."""
    dataset_dir = _REPO_ROOT / "emmy" / "compiler" / "pipeline" / "search" / "dataset"
    forbidden = re.compile(r"^\s*(?:from|import)\s+emmy\.compiler\.pipeline\.search\.db\b")
    offenders = []
    for py in sorted(dataset_dir.rglob("*.py")):
        for lineno, line in enumerate(py.read_text().splitlines(), start=1):
            if forbidden.search(line):
                offenders.append(f"{py.relative_to(_REPO_ROOT)}:{lineno}: {line.strip()}")
    assert not offenders, "search/dataset/*.py must not import search.db — the export reads the DB, the dataset never does:\n" + "\n".join(
        offenders
    )
