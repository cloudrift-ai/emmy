# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Communication

Write to the user the way you would speak to them.

- Short sentences. One idea per sentence.
- Say the few things that matter most. Leave out the rest unless the user asks.
- No file paths, symbol names, or code unless the user asks for them.
- Simple technical English. Common words over rare ones.
- Only established vocabulary: [`GLOSSARY.md`](GLOSSARY.md) terms, other standard repo or field terms, or plain
  language. Never invent a label.
- Answer first. Add background only when it changes what the user does next.
- Disagree when you have reason to. Say what is wrong and what you would do instead. The goal is the right
  answer, not agreement.
- Raise a concern or change the direction of the conversation only when the stake is real: a wrong approach,
  a risk, a much better option. Small things are not worth the interruption. Judge it the way one developer
  would judge it when talking to another.

## Project Overview

Emmy is a Python tool for deploying and benchmarking LLM inference on GPU servers. It supports vLLM and SGLang engines, provides a CLI for local and remote (SSH) deployment of models via Docker Compose, plus automated benchmarking across multiple servers.

`README.md` is the canonical project overview and architecture index. Read it first, then use its links to locate the
relevant subsystem documentation. Do not duplicate the architecture index in this file.

When the user asks about a CLI flag, recipe field, or matrix combinator, use the README index to find and read the
relevant `ARCHITECTURE.md` before answering.

## Prerequisites

- Python 3.12+ with `venv`
- A Rust toolchain (`cargo`): compiled programs execute through the Rust runtime, which `make setup` builds into the
  package as the `emmy.emmy_runtime` extension; use the README architecture index for the runtime's design.
- `make setup` to create the virtual environment and install dependencies
- Docker and Docker Compose for local deployments
- `HF_TOKEN` environment variable for HuggingFace model downloads
- `EMMY_DUMP_DIR` environment variable (optional) — when set, compiler stages dump intermediate debug artifacts
  (graphs, CUDA kernels, execution plans) to this directory. Frontend provenance slices used by `run --bench` stay
  in memory; stable Torch IR is persisted only inside golden files. Kernels are named after the operations they realize
  (`k_rms_norm`, `k_sdpa_reduce`).
- `EMMY_TUNE_DB` environment variable (optional) — overrides the default tuning SQLite cache path
  (`~/.cache/emmy/autotune.db`), the store of measured kernel rows. `emmy run --bench` writes to it, and a greedy
  `compile` / `run` / `serve` creates it on first use: the golden rows in scope (the live card's repository goldens,
  or the file `--golden PATH` names) are imported into it before the compile picks, once per golden digest. NOTE:
  those commands resolve forks through ONE measured-evidence pick — this DB's `perf` rows and the golden rows among
  them, ranked fastest-first; a kernel-set decision (a routing row, a golden's cut or split) is priced as the sum of
  its pieces' rows; and the priors (fit by `emmy fit`: the schedule prior `weights/schedule.json` for a kernel's
  schedule rows, the placement prior `weights/placement.json` for a kernel-set fork's arms) decide only where nothing
  was measured; `--strict-evidence` turns that fall-through into an error. `run --golden
  PATH --bench` writes what it measures back into this DB (`--record` / `--record-greedy` write the golden file too),
  which is how a golden row becomes what the next compile picks. Use the README architecture index for the prior
  and the evidence design.

All `EMMY_*` config env vars are read and written through one module — `emmy/config.py`, the sole owner of
`os.environ` for these vars (the `EMMY_<KNOB>` namespace is the one exception, owned by
`compiler/pipeline/knob.py`; provider/secret vars stay with `emmy/redact.py`). CLI `--flag` overrides (e.g.
`--nvcc-flags`) resolve through `config.py` inside the library, not the command layer, so programmatic callers and tests
get the same precedence. `config.py` is the source of truth for the full var list — do not maintain a copy here.

## Compiler Invariant: Loop Fusion Is Maximal

Loop fusion merges every structurally legal region, to fixpoint, and nothing else. Kernel boundaries come later, from
cuts (`030_cut`), picked by evidence. No one can know before measuring that a merge is a bad decision, and the maximal
region is what keeps every kernel variant open to try.

- **Never add a fusion gate.** No size, cost, recompute, schedulability, recognizer or speed bound on a merge; no
  retained early boundary; no "fusion should not merge X" rule. This holds for code, plans, reviews and proposals to
  the user.
- **A slow or unschedulable fused kernel is a cut or lowering problem.** Fix it with a cut that removes the duplicated
  work, a schedule, or lowering coverage.
- **A region the splicer cannot build is a compiler bug to raise,** never a smaller region.

The passes `ARCHITECTURE.md` owns the design; `tests/compiler/passes/test_maximal_fusion.py` guards it.

## Compiler Invariant: Normalization Carries No Knob

Body normalization (`ir/stmt/normalize.py`, the `normalize_body` driver) is the canonical form every Loop IR body
takes at construction and inside every identity digest. It runs the same on every body, so it answers to no knob, pin
or evidence, and nothing under `ir/` imports the pipeline that holds them.

- **A transform that something decides is a pipeline pass.** A knob the search sets, a pin, a pass stage: the
  transform lives in the pass file that owns the decision under `pipeline/passes/`, even when it is a pure body → body
  function another pass could reuse. Every function in `normalize.py` runs inside `normalize_body`; one that does not
  is a pass in the wrong place.
- **A pass may call a normalization step; normalization never calls a pass.**
- **Nothing is added to normalization to make a kernel faster.** A canonical form moves every kernel identity and
  every golden; whether a transform pays is the knob's evidence question.

The IR `ARCHITECTURE.md` owns the design; `tests/architecture/test_layering.py` guards the module's single entry
point and the `ir/` → pipeline import boundary.

## Compiler Invariant: Kernel Facts Are Computed, Never Stored

A source of truth — an op, a golden file, a corpus case — holds inputs only: the body, the io, the decisions, the
measurements. Anything computed from them — a kernel's exact identity, its `S_*` structural features, whether its
schedule space holds a warp plan — is computed where it is read. A stored copy has to be kept in agreement with the
computation, and one that drifts fails silently: its row stops deciding its fork. A cache — the tune DB's kernel key, a
dataset, the weights, a memo — may hold computed values under a version, and is re-created when stale.

- **Never write a fact onto `op.knobs`.** An op's knobs are the decisions taken on it, every one a registered knob. A
  fact about the body or the card is a function to call, never a value to carry.
- **Never add a computed field to the golden format.** A stored kernel is its Loop IR, its name and where it came
  from; rows and decisions name it by its `ref` in the file. A new field is an input, or it does not go in the file.
- **Evidence joins on the kernel's exact identity and the context, nothing else.** No feature takes part in the join.
- **A new computed value in a cache needs a version.** A change to the computation bumps it and the cache is
  re-created; nothing migrates.

The pipeline `ARCHITECTURE.md` owns the design; `tests/architecture/test_layering.py` pins the golden field lists,
`tests/compiler/pipeline/test_strategies.py` checks that every op's knobs are decisions, and
`tests/compiler/pipeline/search/db/test_db.py` pins the identities the tune DB's version was cut at.

## Running Tests

`make test` runs the whole suite. It takes many minutes, so **do not run it while developing** — run only the tests
that cover the change, under a two-minute budget, and leave the full suite for the finalization stage (see
Contribution Instructions).

```bash
./venv/bin/pytest tests/test_storage.py -v                        # one file
./venv/bin/pytest tests/compiler/passes/test_fusion_rules.py -k pointwise   # a few tests
make test                                                        # the whole suite — finalization only
```

`make test` compiles CUDA kernels at **`-Xcicc -O1`** — the **correctness lane**: `-O1` changes runtime perf, not
numerics, and the deployable perf tests (`tests/perf`, `-m perf`) are skipped here, running at `-O3` via
`make bench-kernels`. It also sets `EMMY_GOLDEN_FILE=` (set, empty): no repository golden is evidence in this lane,
because the lane never asks how fast a pick is and importing a card's goldens is work every worker process would
repeat; a test that needs golden evidence scopes it itself (`--golden PATH`, `golden.records_override`). To re-run the suite at deployable `-O3`, prefix `EMMY_NVCC_FLAGS=` (empty) or run `pytest`
directly. Every pytest session, `make test` or direct, also runs on a fresh tune DB (the root `tests/conftest.py`
points `EMMY_TUNE_DB` at one unless the caller names one), so no test picks from a machine's stored measurements; a
test that needs a DB sets its own.

The lane saves far less than this file used to claim. Measured on an RTX 5090 (CUDA 13.0, 16 cores, one repo, only the
opt level varying): cold cubin cache **923 s at `-O1` vs 1031 s at `-O3`** (1.12×); warm **718 s vs 760 s** (1.06×);
identical results every run. The retired "~3× faster" was never re-measured after the WMMA→`mma.sync` migration
removed the cicc unroll blowup it rested on. The cold/warm gap also puts kernel compilation at roughly a fifth of the
suite's wall time, so it is not the dominant cost either. Keeping `-O1` here buys ~12% cold; dropping it would leave
one compile regime everywhere in the repo.

The default suite holds every repository golden — the hardware goldens and each recipe's model golden —
to the fresh lowering of its own traced programs: a restamp (`emmy golden restamp`) must change nothing, one test node
per traced program so the work scatters over the xdist workers and a failure names the kernels, decisions and rows the
compiler now disagrees with. Lowering is GPU-free, so a stale golden is detectable on any machine. There is no list of
expected failures. The fix is `emmy golden restamp PATH`: every kernel takes the body a fresh lowering gives it, every
decision is taken again on the fresh parent, a row whose kernel was re-keyed — the stored body and the fresh one are
two kernels — keeps its schedule and loses its microseconds (a proposal, no evidence until a record run on the card
measures it again), a kernel no fresh kernel writes is dropped with its rows. A golden stores no identity, so a change
to how identity is computed re-keys nothing. The `refresh-golden` skill owns the whole flow, including the
record run and the delete-or-re-record decision. Never re-record a row to make a red node green. The nightly
`onboard-model` workflow still owns a model golden's exact-GPU replay.

When running a large subset (e.g. `tests/compiler/`), pass the same `-n auto --dist=loadgroup` flags `make test` uses to
parallelize (add `-p no:randomly` for a stable order):

```bash
./venv/bin/pytest tests/compiler/ -p no:randomly -n auto --dist=loadgroup
```

`-n auto` spawns one worker per core; `--dist=loadgroup` keeps tests sharing an `xdist_group` (e.g. CUDA context) on the
same worker.

### The realization corpus

`tests/compiler/realization/` replays pinned schedules from checked-in case files. A case's expectation is its
filename: no suffix means every stage must pass, `_xfail_<stage>` means it is a known gap expected to fail at
`realized`, `built` or `correct`.

- A case **without** a suffix that fails is a regression. Fix the compiler. **Never add an `_xfail_` suffix to make a
  red test green** — that converts a regression into a recorded gap and the ratchet stops meaning anything.
- A case **with** a suffix that passes means the gap closed. `git mv` the file to drop the suffix; do not delete the
  case.
- A **stale case** failure means the lowering or a schedule codec changed and the stored kernels are no longer what
  the program lowers to. `make test` detects this on its own, on any machine; `make test-corpus-regen` is the fix —
  the same restamp every golden gets. It refuses to write when a case's verdict also changed; that refusal is the
  signal, not an obstacle to work around.
- **The corpus never asks for something this machine cannot do.** With no GPU, the only obligation is the stale case
  above, and it is always fixable where you are: `realized` runs at the case's declared capability, while `built` and
  `correct` run only on a card whose capability equals it.
- **Latency is measured in `tests/perf/`, whose case list IS the corpus.** `make test` compiles at `-O1` and never
  measures; `make bench-kernels` benches every closed case the card can run, prints the comparison against eager and
  `torch.compile`, and reports a case slower than its stored number. A regression there is a finding, not a failure.
- **Never write a benchmark script.** `emmy run --golden FILE --bench --record` benches a golden and writes its
  timings back. If it cannot express what you need, that is a missing flag to add, not a script to write.

Before adding a case, read `tests/compiler/realization/ARCHITECTURE.md` — it owns what earns a case, the knob spelling
rules, and what deliberately stays in Python.

### macOS: the suite exits 139 (SIGSEGV)

The loop backend JIT-compiles kernels in-process through cppyy / Cling (`emmy/compiler/ir/loop/runner.py`), and
cppyy-cling 6.32.8 bundles LLVM 16. That compiler cannot parse the libc++ headers in the Xcode 26 SDK, whose
`is_convertible.h` uses the `__is_nothrow_convertible` builtin unconditionally. Cling faults while building its
precompiled header, so every test that imports cppyy dies on a native crash: `pytest tests/compiler/ir/ --collect-only`
exits 139 during *collection*, and `make test` loses its xdist workers to "node down: Not properly terminated".

Rebuild the precompiled header once against an SDK whose libc++ Cling can still parse — any installed 15.x will do:

```bash
SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX15.4.sdk CLING_REBUILD_PCH=1 \
  ./venv/bin/python -c 'import cppyy; cppyy.cppdef("int probe() { return 42; }"); assert cppyy.gbl.probe() == 42'
```

This writes `venv/lib/python3.12/site-packages/cppyy_backend/etc/allDict.cxx.pch.20.6.32.8`, after which cppyy imports
cleanly with no `SDKROOT` set. Repeat it whenever `venv/` is recreated or cppyy is reinstalled. Once cppyy releases a
Cling built on a newer LLVM, upgrading it replaces this workaround.

## CLI Commands

The full CLI reference is linked from the README architecture index. Do **not** duplicate that reference here; read
it before answering any CLI-flag question. Quickstart for the common paths:

| Command | Purpose |
| --- | --- |
| `emmy deploy {local,ssh,cloud} <model> …` | deploy via docker compose locally, over SSH, or on a freshly provisioned cloud VM |
| `emmy bench recipes/* [--filter KEY=PATTERN] [--no-teardown]` | deploy + benchmark + teardown across cloud VMs; `teardown <run_dir>` cleans up afterwards |
| `emmy vm create gpu --gpu NAME --gpu-count N` | provision a GPU VM by name (also `vm create/delete {gcp,cloudrift}`; `vm delete cloudrift --tag` by rental tags; `vm available NAME…` says which CloudRift can rent now) |
| `emmy serve <model> [--runner generate] [--bench] [vllm flags…]` | serve via vLLM, or opt into native text serving with `--runner generate --native` |
| `emmy compile <model_or_ir> [--layer N] [--ir STAGE] [--dynamic …] [--target sm_NN]`, `emmy compile --golden PATH --program N` | trace + run the compiler; print or save any IR stage; compile a golden's stored traced program |
| `emmy run <model_or_ir_or_--code> [--bench]` | compile + execute on the CUDA backend, check accuracy, optionally bench vs eager / `torch.compile` |
| `emmy eval {prior,golden} …` | `eval prior DATASET [--pools {golden,measured}]` scores a dataset's pools with the prior of the dataset's space and re-decides each pool with no measurement in scope; `eval golden --golden PATH --serving-config PATH` audits a golden against its serving matrix |
| `emmy golden {list,check,restamp} [PATH…]` | list measured rows beside `torch.compile`, slowest relative to it first (the `compiler-gaps` skill reads it); say what a restamp onto the fresh lowering of a golden's own programs would change; write that rewrite (every repository golden by default) |
| `emmy fit DATASET WEIGHTS [--folds N]` | fit the prior of a dataset's space from its golden groups and cross-validate it; the whole refit is README's "Fit the priors" |
| `emmy db {import,export,freeze,check} --db PATH …` | fill a DB instance from the freeze directories, golden files and tune DBs named on the command line, or every repository golden (`--repository`; nothing by default, and never the tune DB), each file's rows filed under the identity computed from its stored kernel; export its rows as the dataset of one space (`--space {schedule,placement}`) the fit and `eval prior` read; snapshot it into a freeze; check its tables agree with themselves |
| `emmy {pull,trace,generate,inspect,compare} …` | model download, IR tracing, the naive generation oracle, IR inspection, dump diffing |
 Refitting the priors is the commands under README's "Fit the priors". Every path is explicit — the
DB instance `emmy db import --db PATH` fills (the tune DB's tables in a file of their own, never read by a compile; a
measurement freeze directory under `emmy/compiler/pipeline/search/freezes/`, tracked in **git LFS**, none checked in
at the moment, or a tune DB joins the goldens the same way), and the dataset `emmy db export` writes from it (a
`manifest.json` beside one matrix file per pool, which `emmy fit` and `emmy eval prior` read; the readers never open
the DB) — and nothing has a default, so a refit never touches the tune DB. The examples keep both under `_data/`,
which git ignores. `emmy fit DATASET WEIGHTS` rewrites the checked-in weights of the dataset's space. Nightly refresh
owns routine prior refits, including after repository goldens change. Unless explicitly requested, do not refit or
commit weights as part of PR finalization. A stale dataset or artifact is refused after a featurizer version bump.

Quick test models / scripts (for local iteration):

- Ungated generative smoke model: `Qwen/Qwen3-0.6B` (Qwen3 arch — same family as the embedding smoke model;
  serving-validated on a 4080, tuned TPOT 1.28x stock; defaults to thinking mode — pass `enable_thinking: false`
  in chat probes for terse outputs). `TinyLlama/TinyLlama-1.1B-Chat-v1.0` stays as the ungated **Llama-arch**
  smoke model. GPU embedding model (0.6B): `Qwen/Qwen3-Embedding-0.6B`
- Benchmark/profiling helpers live under `scripts/` (`bench_block.py`, `profile_gen_decode.py`,
  `new_models.py`, `digest_kernels.py` — the kernel-source byte-identity
  gate for tile-IR storage migrations, each case also asserting its pins reached a kernel) — run with `--help` for
  usage;
  the skills that drive them document the flows.
- **Never write a benchmark script.** `emmy run --bench --json PATH` is the machine-readable record every consumer
  reads; `--golden FILE` benches every realization in a golden and `--realization NAME` selects one. If the CLI
  cannot express what you need, that is a missing flag to add, not a script to write — the two scripts that
  re-implemented it (one parsing stdout with a regex, one diffing perf snapshots) had silently stopped working before
  anyone noticed.

## Key Make Targets

- `make setup` — create venv and install dependencies (includes ruff)
- `make test` — run `pytest` using the venv (skips the off-lane `perf` / `goldens` tests). Compiles
  kernels at `-Xcicc -O1` (correctness lane, ~12% faster than `-O3` on a cold cache; perf tests use `-O3` via
  `make bench-kernels`)
- `make test-corpus-regen` — restamp the realization corpus's derived half after a lowering or schedule-codec
  change (`make test` detects the staleness on any machine; this applies the fix)
- `make test-durations` — re-measure `tests/durations_cpu.json`, the checked-in CPU test timings the suite balances its
  xdist workers on; the **Nightly refresh** workflow commits updates directly to `main`
- `make lint` — run `ruff check`, `ruff format --check`, and check test-duration formatting
- `make format` — auto-format code, fix lint violations, and sort test durations without re-measuring
- `make bench` — run benchmarks (`emmy bench recipes/*`)
- `make bench-kernels` — run per-kernel perf comparison vs PyTorch (`tests/perf/`, requires CUDA)
- `make wheel` — build the wheel into `dist/` (stages the bundled recipes first; see the Release section of
  `README.md`)
- `make clean` — remove venv and generated files

## Documentation Conventions

These are invariants — they hold for every doc change, no exceptions:

- **Plans are ephemeral. Never reference `plans/*.md` from durable docs (AGENTS.md, README.md, any `ARCHITECTURE.md`) or
  from code (comments/docstrings).** A plan is a transient working note; anything worth keeping gets written into the
  durable doc itself, and the plan pointer is dropped. (`grep -rn "plans/" --include='*.py' emmy/` and over the
  durable docs must stay empty.) Plan *lifecycle* is governed by the Contribution Instructions below.
- **`ARCHITECTURE.md` files describe concepts, invariants, and the few key entry-point modules — not every file.** Do
  NOT add exhaustive per-file "module tree" tables or `file.py:123` line-number citations; they churn on every refactor
  and rot immediately. Name a module/symbol only when it is a genuine entry point, and refer to it by name, not line.
- **README.md routes; AGENTS.md does not duplicate.** README is the canonical architecture index. Each subsystem's
  detail lives in its nearest `ARCHITECTURE.md`; do not repeat links, CLI details, environment variables, or reference
  lists here.
- **Only use established terminology — [`GLOSSARY.md`](GLOSSARY.md) is the stable vocabulary.** In code comments,
  documentation, reports, commit messages, PR bodies, and when communicating with the user, use glossary terms, other
  established repo/field terms, or plain language. Never coin new labels; replace any invented term with the correct
  established term or a plain-word explanation.

**Wrap every `.md` file in the repo to ~120 characters.** This includes `README.md`, every `ARCHITECTURE.md`, every file
under `docs/`, and any other markdown anywhere in the tree. Do NOT wrap at 70–80 characters — that is the default
markdown habit, and it is wrong for this repo. Aim for lines in the 90–120 range.

Table rows, ASCII diagrams, and long URLs may overflow past 120 if wrapping would hurt readability — that's the only
acceptable reason to go wider. Python code stays under 140 chars (Ruff-enforced).

A pull-request or issue body is not a file in the repo: write it in unwrapped paragraphs and let GitHub wrap them.
Manual line breaks there only make it harder to edit.

## Contribution Instructions

Two speeds. Development is fast: small test subsets, quick commits, no sweeps. Finalization is strict and runs
once, at the end of the PR. Every step below is required for every code change, but each belongs to its own
stage — do not pull finalization work into the edit loop, and do not skip it at the end.

### Writing code

1. Create a feature branch from `main` (e.g. `feature/my-new-feature`) — NEVER commit directly to `main`
2. Write code following guidelines in `STYLE.md`, `README.md` and `ARCHITECTURE.md` files in respective folders
3. Add tests if reasonable (in `tests/` following `tests/ARCHITECTURE.md` guidelines)

**Keep PRs minimal.** Retain only durable implementation, tests, documentation, recipes, and publication evidence.
Delete exploratory scripts, intermediate experiments, run snapshots, and executed plans once their conclusions are
encoded in a durable artifact.

**Do not script open-ended reasoning.** Code should implement stable, reusable mechanics with a clear contract.
Model- or experiment-dependent judgment—such as interpreting heterogeneous benchmark evidence or deciding how to
assemble every possible serving report—belongs in skill instructions and agent reasoning, not in the benchmark
harness, result validators, or a growing family of one-off scripts. Simple, readable mechanical post-processing may
be embedded directly in a recipe; if the logic needs a large decision tree or model-specific policy, keep it out of
code. It is fine to write code that processes structural data by selecting fields, reshaping rows, sorting, joining,
or producing a CSV, TSV, or JSON table. Do not write scripts that interpret results or assemble human-readable
reports; agents perform that reasoning and write the report.

### While developing — go fast

4. **Run only the tests that cover what you changed.** Name the test files or test ids. A directory sweep is not a
   subset. The full suite takes many minutes; it belongs to finalization, not to the edit loop.
5. **Hold a two-minute budget** for every test run or exploratory script you start yourself. If it does not finish in
   two minutes, cut the scope: fewer tests, a smaller model, a smaller shape. A longer run needs a reason and the
   user's agreement.
6. **Commit as soon as the chosen subset is green.** Do not run the full suite, the linter, or the documentation pass
   before a commit. Say in the commit message what you did not verify.
7. **Open a draft PR with the first push.** Use `.github/PULL_REQUEST_TEMPLATE.md` as a guide, but do not edit it; a
   title and a rough abstract are enough at this point. Everything after that lands as more commits on the same PR.

### Finalization — once per PR (MANDATORY — do NOT skip these)

Run this stage once, at the end, over the complete diff. It is the only place the full suite, the linter, and the
documentation sweep belong. Do not spread these steps across the development commits.

Audit the diff:

8. **Remove unnecessary functionality**: Which new functionality can be removed?
9. **Reuse existing mechanisms**: Which existing CLI, library, recipe, or skill can be reused instead?
10. **Rethink touched functionality**: Can existing functionality be rearchitected around the PR's needs so one
    simpler shared design replaces parallel or specialized paths?
11. **Remove obsolete code**: Delete existing code that the PR makes unnecessary. Apply the boy-scout rule within the
    PR's scope and leave touched code cleaner, without expanding into an unrelated refactor.
12. **Keep reasoning and reports out of code**: Which logic should become concise agent instructions? Code may
    transform structural data, but scripts must not interpret results or assemble human-readable reports.
13. **Minimize the diff**: Can the same outcome be achieved with fewer changed lines, files, flags, and abstractions?
14. **Check the core line balance**: run `git diff --stat main -- emmy/`. A PR that introduces no new functionality
    (a refactor, a cleanup, a fix) must NOT increase the line count under `emmy/` — net growth there without new
    capability is the typical sign of architectural creep: another special case, helper, or early return layered onto
    a design that no longer fits. If the balance is positive, do not shave lines cosmetically to pass the check —
    restructure so the layering disappears, or say explicitly in the PR body why the growth is justified.
15. **Apply the audit findings**: perform the removals and consolidation now, not after review. Tests must protect the
    smaller contract, not preserve unnecessary machinery.

Then update the documentation:

16. **Update `STYLE.md`** if any style changes were introduced — READ the current `STYLE.md` and compare
17. **Update `README.md`** if project setup, structure, or usage patterns changed — READ the current `README.md`
    and compare
18. **Update `AGENTS.md`** if general instructions are no longer accurate — READ this file and compare
19. **Update `ARCHITECTURE.md`** files in every directory that was modified — READ each relevant
    `ARCHITECTURE.md` and compare
20. **Prune `plans/`**: if the change executed/landed a plan, **delete that plan file**. Then enforce the cap — if
    `plans/` holds more than 10 files, remove the executed/obsolete ones; if all remaining plans are still incomplete,
    remove the oldest. Never add a `plans/*.md` reference to durable docs or code (see Documentation Conventions).
21. **Check terminology**: review every text this change adds or edits — code comments, docstrings, docs, report
    text, the commit message and PR body — against [`GLOSSARY.md`](GLOSSARY.md). Remove any invented terms and
    replace them with the correct established term or a plain-word explanation (see Documentation Conventions).

Then run the gates, in this order, after every edit above is in:

22. **Leave prior refits to nightly refresh.** Golden changes do not require a refit or a weights commit in the PR.
    If the reproduction gate fails, name the failing nodes in the PR body; do not refit just to make them pass.
23. **Run the full suite**: `make test` — fix any failures. If a realization case comes back stale, `make
    test-corpus-regen` applies the fix; if a repository golden stops being the fresh lowering, `emmy golden restamp`
    applies that one (the `refresh-golden` skill). If golden rows go red, name the change that did it in the PR body —
    do **not** re-record them to make it green, which enshrines the regression as the new reference.
24. **Let the nightly workflow refresh CPU test durations.** Missing duration rows do not fail `make test`.
    The nightly run re-measures the whole CPU suite and commits `tests/durations_cpu.json` directly to `main` when it
    changes. It records tests of 5 s or more; a recorded entry holds while new measurements stay within 50% of it.
    GPU timings stay in `tests/durations_gpu.json` and are not rewritten on the CPU runner.
25. **Run the linter**: `make lint` — if it fails, run `make format` and re-check
26. **Write the PR body** in an untracked temporary file outside the repository, using
    `.github/PULL_REQUEST_TEMPLATE.md` as a guide. Never replace the tracked template with a PR's content. The title
    is a functional description readable with no context. The abstract is one short plain-English paragraph — no
    bullets, no code references. One optional artifact may follow it — a small table, a diagram, a few lines of
    output — when it carries the claim better than the paragraph. Then a horizontal rule, then everything else —
    decisions, measurements, what broke, what got slower, what was removed — under headings that fit the story.
    `Abstract` is the only fixed heading.
27. **Revise the PR body at least twice before posting.** Write it, then reread it as a reviewer with no
    context, check it against the template and against the design philosophy below, cut, and repeat. A first
    draft is always too long: it lists what was done instead of saying what the change is, and it keeps
    sentences that no reviewer would miss. Stop when nothing else can come out without losing the point.
28. **Mark the PR ready for review.** This is the last step. A draft PR that has not been through finalization
    is not ready, whatever else is green.

# Behavioral Guidelines:

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Apply the boy-scout rule within the PR's scope: simplify adjacent functionality exposed by the change when that
  cleanup reduces the total design and remains covered by tests.
- Do not turn scoped cleanup into an unrelated refactor.
- Match existing style, even if you'd do it differently.
- Remove existing dead code or duplication in the touched path when it is confidently obsolete; otherwise mention it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Remove pre-existing code that the new design makes unnecessary.

The test: Every changed line should trace directly to the user's request.

**Exception — a major win.** If the change exposes a restructure that would make the design clearly simpler, say so.
Do not do it silently and do not bury it: describe the win, say what it costs, and let the user decide. Staying
surgical is the default, not a reason to keep quiet about a better shape.

## 4. Design Quality Over Compatibility

**A simpler design beats a compatible one.** Every artifact in Emmy can be regenerated — goldens, corpus cases, tune
databases, priors, dumps. None of them is a reason to keep a worse design alive. Breaking a format, a name, or a
stored file is cheap; carrying a second path forever is not.

The measure of quality is the design itself: how few concepts it takes to explain, and how few lines the core under
`emmy/` needs to carry it. Growth in the core without new capability is the warning sign — see the line-balance step
in the Contribution Instructions.

Dropping a feature is allowed too, when the feature is what forces the complexity. That call belongs to the user, not
to you. Your job is to offer it: name the feature, name the simplification it buys, and wait.

## 5. No Noise

**Surface what matters, when it matters.** Which concerns are worth raising depends on the stage.

- While prototyping or refactoring: broken paths, slower kernels, and a smaller feature set are fine. Keep working.
  Do not stop to report them.
- At finalization: every one of them has to be surfaced and written into the PR body — what broke, what got slower,
  what was removed, and why.

The same fact is noise in the middle of the work and required at the end. Judge by the stage.

## 6. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.
