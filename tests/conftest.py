"""Shared pytest fixtures for all test modules."""

import functools
import json
import os
import random
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

# Cross-process GPU lock for CUDA tests. Set on conftest import so every
# xdist worker (and any subprocess it spawns) coordinates on the same
# path. With this set, ``CudaBackend.run`` (via
# :func:`emmy.compiler.backend.cuda.program.run_program`) holds the
# lock end-to-end across compile + allocate + ``pre_run`` callback +
# kernel launches + ``.get()``. Tests that compare emmy against
# torch eager pass ``pre_run=<eager closure>`` so the eager forward
# and the emmy launches share one uninterrupted GPU window —
# without this serialization, peer-worker CUDA activity interleaves
# with our kernels and the per-position fp32 rounding drift breaks
# the accuracy comparison.
# Per-uid path: on a multi-user runner the first user's lock file (mode 0644, sticky /tmp)
# is unopenable by everyone else, so a shared path fails the run with PermissionError
# instead of serializing (CI run 32339655489). Cross-user serialization was never real.
os.environ.setdefault("EMMY_GPU_LOCK", f"/tmp/emmy-gpu-{os.getuid()}.lock")

# The CPU lane sees no CUDA device at all. A host with a card but CPU-only torch (the CI runners)
# cannot run a kernel, yet the runtime still reaches the card through the driver and the prior
# features its SM count and memory, so a pick, and the kernel count a test asserts, followed
# whichever card the runner happened to hold. Hidden, every CPU lane features the default card.
if not torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""

# Emmy's CPU-parallel work (a cold-pool draw, a dataset export) runs a process per core by default. The suite
# already runs a worker per core, so a test runs it in its own process instead; the results are the same either way.
os.environ.setdefault("EMMY_WORKERS", "1")
# It also draws a smaller cold pool than a deploy (8192): the suite asserts picks, not their quality, and the prior
# reproduction gate draws the same size. The gate's cost is near-linear in it; 2000 made the gate 62 percent of the
# CI job.
os.environ.setdefault("EMMY_POOL_DRAW", "512")

# The session's own tune DB, fresh and removed at exit, unless the caller points ``EMMY_TUNE_DB`` at
# one. The default ``~/.cache/emmy/autotune.db`` is machine-local, mutable evidence: a CLI compile
# picks from its rows, so the same test would decide differently on a box that once measured.
# Set on import, like the lock above, so the xdist workers and every subprocess share the one file.
if "EMMY_TUNE_DB" not in os.environ:
    _TUNE_DB_DIR = tempfile.TemporaryDirectory(prefix="emmy_test_tune_db_")
    os.environ["EMMY_TUNE_DB"] = os.path.join(_TUNE_DB_DIR.name, "autotune.db")


# ── CUDA context poisoning containment ──────────────────────────────
# An illegal / misaligned access leaves the CUDA context in a STICKY error
# state: every later CUDA call in the process returns that same status until
# the context is torn down, which no in-process caller can do. So one faulting
# test silently takes down every later CUDA test in its xdist worker — the run
# that motivated this reported 1 failure and 51 errors, and all 51 named
# innocent tests, because ``--dist=loadgroup`` keeps the whole ``cuda`` group on
# one worker. Worse, the fault does not have to fail anything: an
# ``xfail(strict=False)`` swallows it as an expected failure and the context
# stays poisoned regardless.
#
# The probe is the one the bench worker already trusts
# (``_bench_worker._context_dirty``): a ``deviceSynchronize`` surfaces the
# sticky status. Running it after every test buys the attribution — the process
# stops at the test that poisoned the context, so the culprit is named instead
# of the first innocent bystander. Under xdist the controller then restarts the
# worker on a fresh context and reschedules the rest of its queue, so the run
# still finishes and reports exactly one failure.
def _cuda_context_poisoned() -> bool:
    """Whether the live CUDA context is in a sticky-error state.

    ``False`` when no CUDA context can exist: ``torch.cuda.is_available()`` is
    the CPU-lane gate, and the runtime reports ``False`` itself when nothing in
    this worker ever created a context.
    """
    from emmy.compiler.backend.cuda.device import context_poisoned

    return torch.cuda.is_available() and context_poisoned()


#: Node id of the test that poisoned the context, set the moment it is detected and
#: consumed by the next test's setup — see :func:`pytest_runtest_setup`.
_POISONED_BY: str | None = None


@pytest.hookimpl(trylast=True)
def pytest_runtest_teardown(item):
    global _POISONED_BY
    if _POISONED_BY is not None or not _cuda_context_poisoned():
        return
    _POISONED_BY = item.nodeid
    print(
        f"\nCUDA CONTEXT POISONED by {item.nodeid}\n"
        "The context is in a sticky-error state, so every later CUDA call in this process "
        "returns the same error. This process will stop before the next test rather than "
        "cascade the fault across the rest of its queue.\n",
        file=sys.stderr,
        flush=True,
    )


def pytest_runtest_setup(item):
    """Stop the process once the context is poisoned, before running anything else on it.

    Deliberately here and not at the moment of detection: the poisoning test's own reports
    must reach the xdist controller first, or the controller sees a worker that died mid-test,
    re-queues that test on a fresh worker, and it poisons that one too — a crash loop rather
    than containment. Dying in the NEXT test's setup leaves only that (innocent) test to
    reschedule, which then passes on a clean context.
    """
    if _POISONED_BY is None:
        return
    message = f"stopping: the CUDA context was poisoned by {_POISONED_BY}; {item.nodeid} cannot run on it"
    if hasattr(item.config, "workerinput"):
        # An xdist worker: die so the controller respawns it with a clean context and
        # reschedules the tests this worker had not reached.
        print(f"\n{message}\n", file=sys.stderr, flush=True)
        os._exit(1)
    pytest.exit(message, returncode=1)


@pytest.fixture(autouse=True)
def _isolate_offline_file(monkeypatch):
    """Drop any dev-machine ``EMMY_OFFLINE_FILE`` override so tests always score
    through the repo-checked ``weights/schedule.json``. Unlike the prior file, the
    default here must NOT be a tmp path — a missing offline artifact is a hard
    error by design (no silent fallback), and the shipped one is what tests
    exercise."""
    monkeypatch.delenv("EMMY_OFFLINE_FILE", raising=False)


@pytest.fixture(autouse=True)
def _seed_rng():
    """Pin RNGs for every test so numerical-tolerance assertions
    (e.g. ``test_torch_ops.test_unary``) don't flake on inputs that
    happen to land in tight regions. Determinism > tolerance — a real
    precision regression should still trip these tests.

    Also reseeds module-level ``rng = np.random.default_rng(...)``
    Generators in test modules. They're instantiated once at import,
    so successive ``rng.uniform`` calls inside parametrized tests
    drift across the session and produce order-dependent flakes
    (sigmoid/tanh/rsqrt at near-zero inputs etc.). Re-binding ``rng``
    to a fresh ``default_rng`` with the original seed restores
    intra-test determinism without changing any test's input
    distribution."""
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    for mod in list(sys.modules.values()):
        if mod is None or not getattr(mod, "__name__", "").startswith("tests."):
            continue
        rng = getattr(mod, "rng", None)
        if isinstance(rng, np.random.Generator):
            seed = getattr(mod, "_RNG_SEED", 0)
            mod.rng = np.random.default_rng(seed)


PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
RECIPES_DIR = os.path.join(PROJECT_ROOT, "recipes")

# ── LPT static bucketing for pytest-xdist ───────────────────────────
# Record per-test call durations in the pytest cache; next run partitions
# items across N worker buckets via LPT (longest-processing-time-first)
# greedy — each item goes to the currently-lightest bucket. Buckets are
# tagged via @pytest.mark.xdist_group so `--dist=loadgroup` routes every
# item in a bucket to the same worker. Theoretical makespan is the load
# of the heaviest bucket (lower bound = longest single test).
#
# The pytest cache only helps a box that has already run the suite once —
# CI starts every job with an empty one, so the balancing never fired there
# and the long poles landed wherever chance put them. The checked-in CPU and GPU
# duration files make the FIRST run balanced: nodeid → seconds maps. The nightly
# ``make test-durations`` run refreshes the CPU file without touching GPU rows.
# It holds only the tests that set the makespan (see ``_MIN_RECORDED``);
# anything absent is assumed cheap (``_UNKNOWN_COST``). A stale or partial
# baseline costs balance, never correctness — the cache overlays it, so a local
# run's own measurements always win over the committed numbers.

_DURATIONS_KEY = "test_durations/call"
_DURATIONS_CPU_FILE = os.path.join(os.path.dirname(__file__), "durations_cpu.json")
_DURATIONS_GPU_FILE = os.path.join(os.path.dirname(__file__), "durations_gpu.json")
_CALL_DURATIONS: dict[str, float] = {}

#: Below this a test is noise to the bucketing, not a row: the few hundred tests
#: at or above the floor set the makespan, and the file does not churn on the rest.
_MIN_RECORDED = 5.0
#: What an unlisted test is assumed to cost when bucketing (see ``_MIN_RECORDED``).
_UNKNOWN_COST = 0.05
#: Markers whose tests are deselected from the default suite. They cannot distort ITS bucketing,
#: so they are excluded from the written baseline.
_OFF_LANE_MARKERS = ("perf",)
#: Node ids seen carrying an off-lane marker this session (filled during collection).
_OFF_LANE_ITEMS: set[str] = set()


def pytest_addoption(parser):
    parser.addoption(
        "--write-durations",
        action="store_true",
        help="Rewrite tests/durations_cpu.json from this run's CPU timings.",
    )


def pytest_runtest_logreport(report):
    if report.when == "call":
        _CALL_DURATIONS[report.nodeid] = report.duration


def _load_durations(*paths: str) -> dict[str, float]:
    durations = {}
    for path in paths:
        try:
            with open(path) as fh:
                durations.update(json.load(fh))
        except (OSError, ValueError):
            pass
    return durations


def _refresh_durations(previous: dict[str, float], measured: dict[str, float]) -> dict[str, float]:
    """Merge one run's timings into the recorded rows.

    A recorded row holds through any measurement within half its value, even one under the floor, so a
    test near the floor does not flip in and out of the file night after night. Outside that band the
    measurement replaces it, and a row enters only at the floor. A test absent from the run drops out.
    """
    fresh = {}
    for key, value in measured.items():
        old = previous.get(key)
        if old is not None and abs(value - old) < old / 2:
            fresh[key] = old
        elif value >= _MIN_RECORDED:
            fresh[key] = round(value, 2)
    return fresh


def pytest_sessionfinish(session):
    # ``workerinput`` marks an xdist WORKER — only the controller sees every
    # test's report, and letting each worker write would race on the file.
    is_controller = not hasattr(session.config, "workerinput")
    if session.config.getoption("--write-durations") and _CALL_DURATIONS and is_controller:
        # Replace CPU rows so renamed tests drop out. The nightly runner has no GPU,
        # so its skipped CUDA tests must not remove the separate GPU baseline.
        # Point --write-durations at the whole suite, never a subset.
        measured = {k: v for k, v in _CALL_DURATIONS.items() if k not in _OFF_LANE_ITEMS and not k.endswith(("@cuda", "@cuda-cli"))}
        fresh = _refresh_durations(_load_durations(_DURATIONS_CPU_FILE), measured)
        with open(_DURATIONS_CPU_FILE, "w") as fh:
            json.dump(dict(sorted(fresh.items())), fh, indent=1)
            fh.write("\n")

    cache = getattr(session.config, "cache", None)
    if cache is None or not _CALL_DURATIONS:
        return
    existing = cache.get(_DURATIONS_KEY, {}) or {}
    existing.update(_CALL_DURATIONS)
    cache.set(_DURATIONS_KEY, existing)


def _num_workers(config) -> int | None:
    """Mirror xdist's -n resolution: int, 'auto', 'logical', or None."""
    try:
        n = config.getoption("numprocesses", None)
    except ValueError:
        return None
    if n in (None, 0):
        return None
    if isinstance(n, int):
        return n if n >= 1 else None
    if n in ("auto", "logical"):
        return os.cpu_count() or 1
    try:
        return int(n)
    except (TypeError, ValueError):
        return None


def _is_cuda_item(item) -> bool:
    """True iff this test item issues CUDA work.

    Detected via (a) a ``skipif`` marker whose reason starts with
    ``"CUDA not available"`` (the ``requires_cuda`` decorator used across
    ``tests/compiler/``), (b) a ``[cuda...]`` callspec id (the
    ``run_graph`` fixture's third variant + every ``test_e2e_accuracy``
    parametrization), or (c) an explicit ``xdist_group("cuda")`` marker
    (the ``tests/serving/**/*_gpu.py`` pytestmark convention). The explicit
    marker MUST be honored here: otherwise the LPT bucketing below adds a
    function-level ``w<N>`` group that shadows the module-level ``cuda``
    mark (``get_closest_marker`` prefers function-level), scattering the
    test off the serialized CUDA worker. One of those signals is true for
    every test that actually touches the device today; new CUDA-using
    tests inherit routing for free as long as they reuse the conventions."""
    for mark in item.iter_markers(name="skipif"):
        reason = mark.kwargs.get("reason", "")
        if isinstance(reason, str) and reason.startswith("CUDA not available"):
            return True
    for mark in item.iter_markers(name="xdist_group"):
        if mark.args and mark.args[0] == _CUDA_GROUP:
            return True
    nid = item.nodeid
    return "[cuda" in nid or "-cuda-" in nid or nid.endswith("-cuda]")


_NO_TOOLCHAIN = "CUDA not available (need the emmy.emmy_runtime extension + GPU + nvcc)"


@functools.cache
def _cuda_unavailable_reason() -> str | None:
    """Why Emmy CUDA tests cannot run here, or ``None`` when they can.

    A visible device is not sufficient: Emmy compiles its kernels with the CUDA
    toolkit's ``nvcc`` binary, and launches them through the ``emmy.emmy_runtime``
    extension.  Cached — the answer is a property of the host, and the probe
    touches the driver.
    """
    try:
        from emmy.compiler.backend.cuda.device import compute_capability
        from emmy.compiler.backend.cuda.nvcc import nvcc_path

        if compute_capability() is None or nvcc_path() is None:
            return _NO_TOOLCHAIN
    except Exception:  # noqa: BLE001 -- an unusable CUDA runtime means skip
        return _NO_TOOLCHAIN
    return None


def pytest_report_header() -> list[str]:
    """Name an unusable CUDA toolchain once, at the top of the run, instead of
    letting it reappear as every CUDA test's individual failure."""
    reason = _cuda_unavailable_reason()
    return reason.splitlines() if reason else []


# xdist_group for every IN-PROCESS CUDA-touching test. The host only has
# one GPU; running CUDA tests across multiple xdist workers concurrently
# would mean two processes pushing kernels onto the same device. Even
# with ``EMMY_GPU_LOCK`` serializing the kernel-launch window and
# ``backend.run(pre_run=...)`` pulling the torch eager forward into the
# same lock, multi-kernel attention schedules still occasionally drift
# enough across worker contexts (per-context SM scheduling differs, and
# fp32 atomic-add commit order with it) to break the
# ``test_attention_chains`` / ``test_block_accuracy`` 1e-4 thresholds.
# Pinning all in-process CUDA tests to one group makes them run
# sequentially on one worker; non-CUDA tests still parallelize via the
# LPT buckets below.
_CUDA_GROUP = "cuda"

# Separate group for CUDA tests that drive the CLI through the ``run_cli``
# fixture: each spawns a FRESH subprocess with its own CUDA context, so
# they don't share the in-process worker's context and don't need to ride
# the (long) ``cuda`` chain. They still need bounded concurrency — left
# ungrouped, ~30 workers can each hold a live CUDA subprocess (~1 GB a
# piece) and OOM the card — so they serialize among themselves on a
# second worker, in parallel with the in-process chain. (Sharding this
# chain 3-way was tried and bought only ~5 s — the in-process ``cuda``
# chain is the critical path — so one shard keeps it simple.)
_CUDA_CLI_GROUP = "cuda-cli"


# ``tryfirst``: xdist's worker-side ``WorkerInteractor.pytest_collection_modifyitems``
# bakes each item's ``xdist_group`` into the nodeid it reports to the
# controller's loadgroup scheduler — and pluggy calls it BEFORE a plain
# conftest hook (the interactor registers after conftests, so LIFO order
# puts it first). Without ``tryfirst`` every marker added here lands too
# late: the routing silently degrades to plain ``load`` and CUDA tests
# scatter across workers (concurrent CUDA contexts → flaky GPU OOM in
# the ``run_cli`` subprocess tests, accuracy drift in attention chains).
@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    import heapq

    # Deselect every off-lane marker unless explicitly requested. Lives here — not in
    # ``tests/perf/conftest.py`` — so the gate holds for ANY ``tests/``
    # collection (e.g. ``pytest tests/serving/``), not only runs that happen
    # to collect ``tests/perf/`` and load its conftest.
    selected = config.getoption("-m") or ""
    _OFF_LANE_ITEMS.update(i.nodeid for i in items if any(m in i.keywords for m in _OFF_LANE_MARKERS))
    for marker in _OFF_LANE_MARKERS:
        if marker in selected:
            continue
        skip = pytest.mark.skip(reason=f"{marker} marker not selected; run with `pytest -m {marker}`")
        for item in items:
            if marker in item.keywords:
                item.add_marker(skip)

    # Step 1: pin every CUDA-touching item to an xdist_group so each
    # chain lands on one worker and runs sequentially — ``cuda`` for
    # in-process device work, ``cuda-cli`` for ``run_cli`` subprocess
    # tests (own CUDA context per subprocess; see the group comments
    # above). Skip the LPT bucketing for those items entirely — they're
    # already grouped.
    cuda_items: list = []
    other_items: list = []
    cuda_reason = _cuda_unavailable_reason()
    # Only the first line on each item — the full remedy is in the report header.
    skip_cuda = pytest.mark.skip(reason=cuda_reason.splitlines()[0] if cuda_reason else "")
    for it in items:
        if _is_cuda_item(it):
            if cuda_reason:
                it.add_marker(skip_cuda)
            group = _CUDA_CLI_GROUP if "run_cli" in getattr(it, "fixturenames", ()) else _CUDA_GROUP
            it.add_marker(pytest.mark.xdist_group(group))
            cuda_items.append(it)
        else:
            other_items.append(it)

    # The committed baseline first, this box's own cache over it — a local run's
    # measurements beat the checked-in numbers on the machine that took them,
    # while CI (empty cache) still gets a balanced first run off the baseline.
    durations = _load_durations(_DURATIONS_CPU_FILE, _DURATIONS_GPU_FILE)
    cache = getattr(config, "cache", None)
    if cache is not None:
        durations.update(cache.get(_DURATIONS_KEY, {}) or {})
    nworkers = _num_workers(config)
    if not durations or nworkers is None or nworkers < 2:
        items[:] = cuda_items + other_items
        return

    def dur(item) -> float:
        return durations.get(item.nodeid, _UNKNOWN_COST)

    sorted_others = sorted(other_items, key=dur, reverse=True)

    # Reserve one worker per CUDA group (``cuda`` + ``cuda-cli``);
    # LPT-bucket the rest across the remaining workers. With small
    # nworkers we fall back to a single bucket (no-op grouping). Sum
    # CUDA-item durations into one CUDA load so it competes for ordering
    # with the other heavy buckets.
    #
    # Off-GPU there is nothing to reserve FOR: every CUDA item skips in
    # microseconds, so the two chains cost nothing and holding workers back
    # for them just shrinks the pool. That was the CI shape — a 4-core runner
    # squeezed the whole suite onto 2 workers to reserve 2 for chains of
    # pure skips.
    cuda_load = sum(dur(it) for it in cuda_items)
    other_workers = max(1, nworkers - (2 if torch.cuda.is_available() else 0))

    # LPT: pop the lightest bucket, add this item, push back.
    buckets: list[tuple[float, int, list]] = [(0.0, w, []) for w in range(other_workers)]
    heapq.heapify(buckets)
    for it in sorted_others:
        load, wid, bucket = heapq.heappop(buckets)
        bucket.append(it)
        heapq.heappush(buckets, (load + dur(it), wid, bucket))

    # Tag non-CUDA items with their bucket's xdist_group so loadgroup
    # routes them together. CUDA items keep their pre-applied ``cuda`` /
    # ``cuda-cli`` group from step 1.
    buckets_sorted = sorted(buckets, key=lambda b: -b[0])
    reordered: list = []
    for _load, wid, bucket in buckets_sorted:
        group = f"w{wid}"
        for it in bucket:
            it.add_marker(pytest.mark.xdist_group(group))
            reordered.append(it)
    # Put CUDA bucket first when it dominates load, otherwise interleave
    # with the largest non-CUDA bucket. Heaviest-first dispatch lets xdist
    # start the longest serial chain immediately.
    if cuda_load >= buckets_sorted[0][0]:
        items[:] = cuda_items + reordered
    else:
        items[:] = reordered + cuda_items


@pytest.fixture(scope="session")
def project_root():
    """Absolute path to the project root directory."""
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def recipes_dir():
    """Absolute path to the recipes/ directory."""
    return RECIPES_DIR


@pytest.fixture(scope="session")
def run_cli(project_root):
    """Return a callable that invokes the emmy CLI as a subprocess."""

    def _run(*args):
        result = subprocess.run(
            [sys.executable, "-m", "emmy.emmy", *args],
            capture_output=True,
            text=True,
            cwd=project_root,
        )
        return result.returncode, result.stdout, result.stderr

    return _run


@pytest.fixture
def make_bench_config(recipes_dir):
    """Return a factory that writes a temporary bench config.yaml."""

    def _make(tmp_dir):
        config = {
            "benchmark": {
                "model_dir": "/hf_models",
            },
        }
        config_path = os.path.join(str(tmp_dir), "config.yaml")
        with open(config_path, "w") as f:
            yaml.dump(config, f)
        return config_path

    return _make


# ── Compiler dump fixture ──────────────────────────────────────────


@pytest.fixture
def dump_dir(request):
    """Dump compilation artifacts to _test_data/<test_name>/ for manual inspection."""
    safe_name = request.node.name.replace("[", "_").replace("]", "_").replace("/", "_")
    dump_path = Path(PROJECT_ROOT) / "_test_data" / safe_name

    from emmy.compiler.pipeline.dump import CompilerDump

    return CompilerDump(dir=dump_path)


# ── Unit-test fixtures ──────────────────────────────────────────────


@pytest.fixture
def tmp_recipe_dir(tmp_path):
    """Create a temp directory with a sample recipe.yaml using matrices format."""
    recipe = {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
        "matrices": [
            {
                "deploy.gpu": "NVIDIA GeForce RTX 5090",
                "deploy.gpu_count": 1,
            },
            {
                "deploy.gpu": "NVIDIA H200 141GB",
                "deploy.gpu_count": 8,
                "engine.llm.tensor_parallel_size": 8,
                "engine.llm.context_length": 16384,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
                "benchmark.random_input_len": 8000,
                "benchmark.random_output_len": 8000,
            },
            {
                "deploy.gpu": "NVIDIA H100 80GB",
                "deploy.gpu_count": 4,
                "engine.llm.tensor_parallel_size": 4,
                "engine.llm.vllm.extra_args": "--kv-cache-dtype fp8",
            },
        ],
    }

    recipe_path = tmp_path / "recipe.yaml"
    with open(recipe_path, "w") as f:
        yaml.dump(recipe, f)

    return str(tmp_path)


@pytest.fixture
def sample_config():
    """Return a resolved config dict for testing compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_sglang():
    """Return a resolved config dict for SGLang compose generation."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 8192,
                "sglang": {
                    "image": "lmsysorg/sglang:v0.5.9",
                },
            }
        },
        "benchmark": {
            "max_concurrency": 128,
            "num_prompts": 256,
            "random_input_len": 4000,
            "random_output_len": 4000,
        },
    }


@pytest.fixture
def sample_config_multi():
    """Return a resolved config dict for multi-instance testing."""
    return {
        "model": {"huggingface": "test-org/test-model"},
        "engine": {
            "llm": {
                "tensor_parallel_size": 4,
                "pipeline_parallel_size": 1,
                "gpu_memory_utilization": 0.9,
                "context_length": 16384,
                "vllm": {
                    "image": "vllm/vllm-openai:v0.17.0",
                },
            }
        },
        "deploy": {"gpu": "NVIDIA H100 80GB", "gpu_count": 8},
    }
