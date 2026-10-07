# Pipeline Architecture

The pipeline is the part of the compiler that turns a traced graph into finished CUDA kernels, one rewrite at a time.
This document explains it end to end for someone new to the code. It assumes you know the shared vocabulary in
[`GLOSSARY.md`](../../../GLOSSARY.md) — fork, knob, candidate, prior, evidence, golden configuration — but nothing
about the internals of this package. Words that carry a special meaning inside the pipeline are explained in plain
language where they first appear; the few that also turn up in neighboring documents are in the glossary.

Four companion documents cover what this one doesn't:

- The rewrite rules themselves and their authoring invariants → [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).
- What each IR dialect looks like → `ir/ARCHITECTURE.md`.
- A gentler, worked-example introduction to the same material, written for someone who has not read the code → the
  Tutorials section of the course site (`docs/docs/tutorials/`). It covers forks, the deploy evidence hierarchy, the
  prior and the goldens in eight short pages; this file remains the reference the tutorials defer to.

**How to read this.** "The big picture" below is the mental model everything else refines — read it first, in order.
After that the Parts are largely independent:

| Part | Covers | Read it when |
|------|--------|--------------|
| 1 | The rewrite engine: patterns, rule contract, splicing | you are writing or debugging a rule |
| 2 | Forks: how a rule's alternatives are represented | a fork's options look wrong or incomplete |
| 3 | The prior and **the deploy evidence hierarchy** | you need to know why a compile picked what it picked |
| 4 | The driver: the greedy compile through `Run.resolve` | you are tracing control flow through a compile |
| 5 | Recording evidence: how a measurement reaches the tune DB | you want a bench to decide the next compile |
| 6 | Persistence: the two identities, the tables, the freeze | you are adding a cache, a table, or a column |
| 7 | Golden configs and the A/B integrity gates | you are recording or replaying goldens |
| 8 | Evaluating the prior and the goldens (`emmy eval`) | the prior is picking badly and you want to know where |
| 9 | Tile lowering, at the pipeline level | you want the pipeline-side view of `tile/` |

The reference sections after Part 9 — "Tunable knobs", the pass table, the dump hooks — are lookup material, not
reading material.

## The big picture

The compiler lowers a graph by repeatedly applying small rewrite rules. Most rewrites are deterministic: there is one
right answer and the rule just returns it. Some choices are not. Tile sizes, staging depth and split-K all have many
valid answers, and which one is fastest depends on the GPU and the shapes. A rule facing such a choice returns *all*
its options. That return is a **fork**, and the machinery around the rules — never the rule itself — decides which
option wins.

Almost everything in this package exists to answer one question well: *at each fork, which option do we take?*
Measuring and picking are two different jobs, done at different times.

**`emmy run --bench` has a GPU and time to spend.** It builds kernels, times them, and records every clean
measurement as a `perf` row in the tune DB, a SQLite database (Part 5). A golden row is such a measurement, reviewed
and checked in (Part 7).

**`emmy compile` / `emmy run` / `emmy serve` (a "greedy" compile) benchmarks nothing.** Every fork is decided on the
spot from knowledge recorded earlier, in a fixed order — measured first:

1. **Measured evidence** — every measurement the compile can see, in one index: the tune database's `perf` rows for
   this compile's context, and the **golden rows** in scope — the repository's per-card golden files, or the file
   `--golden PATH` names (Part 3). A golden row is a measurement like any other: it joins a candidate by the kernel's
   exact identity and value-of-position agreement. It competes with local rows on µs alone.
2. **The prior** — the offline model, fit ahead of time and shipped with the repo (Part 3).
3. **Option-0** — the first option in the order the rule emitted them. This is only the no-evidence fallback;
   enumeration order carries no performance meaning. Under **strict evidence** (`--strict-evidence`,
   `EMMY_STRICT_EVIDENCE`) a compile refuses to reach steps 2 and 3 at all: a fork no measurement decides raises
   `EvidenceError` naming the kernel.

That order has a name — the **deploy evidence hierarchy**. The list above is only a summary. **Part 3's "The deploy
evidence hierarchy" is the authoritative statement** of the exact order, of what the evidence index holds, and of the
rule that measured evidence applies only to a compile at deployable `-O3` flags.

Kernel-set forks — a cut, a split, a weight layout: the ones that change a kernel's identity or the kernel set —
follow the same rule, and are decided before any schedule. A cut or split decision the tune DB stores on the kernel
(a **routing row**, imported from a golden) is priced as the sum of its pieces' fastest rows; a layout arm by its
resulting kernel's measured row. With no measured arm, the fork goes to the placement prior, which ranks every arm
from the kernels it leaves; with no placement prior, the first arm keeps the kernel as it is. No arm is scheduled to
decide the fork: each piece an arm mints gets its schedule at its own fork (Part 4).

### The four stores

Four stores hold everything a compile can know. They have different writers, different readers and different
lifetimes, and telling them apart is the single most useful thing to learn early.

| Store | Where it lives | Written by | Consulted by |
|-------|----------------|------------|--------------|
| **Golden files** | model goldens under `recipes/<model>/golden/`; hardware goldens under `search/golden/records/` — the tune DB's tables for one card, beside the traced programs (Part 7) | `run --bench --record-greedy` / `--record` into a working golden, reviewed and promoted (Part 7) | greedy compile — measured rows in the one evidence index (the per-card files, or `--golden PATH`); `run --golden PATH --bench` measures them; `emmy db import` loads them into the dataset DB, whose export (`emmy db export`) is the dataset `emmy fit` and `emmy eval prior` read |
| **`perf` table** | the tune DB (`~/.cache/emmy/autotune.db`), beside the `kernel` and `routing` rows its rows are of | `run --bench` — every clean pinned row (golden / `--ab` / `--tune`) and the greedy re-bench, per kernel (`search/bench_record.py`, Part 5); the golden import — the golden rows in scope, once per golden digest | greedy compile (measured evidence); the per-variant replay cache |
| **Dataset DB** | the file `emmy db … --db PATH` names (`_data/dataset.db` in the examples, under the ignored `_data/`; never the tune DB) — the same tables in a file of their own | `emmy db import`, from the freeze directories, golden files and tune DB files named on its command line (the hardware goldens `search/golden/records/*.json` for the offline prior; nothing by default) — each file's tables, its rows filed under the identity computed from each stored kernel | `emmy db export` and nothing else — **never** a deploy |
| **Dataset** | the directory `emmy db export` is given (`_data/dataset` in the examples) — a `manifest.json` beside one matrix file per pool (`search/dataset/document.py`) | `emmy db export`: every golden pool enumerated from its kernel's definition and featurized, every measured pool, the provenance | `emmy eval prior` (both kinds of pool) and `emmy fit` — **never** a deploy |

Of the four, only the goldens travel with a clone: they are the only *measured* data a fresh machine has. The tune
DB is a machine-local cache written by local benches, so a freshly rented box starts with the goldens plus the
shipped offline prior artifact and nothing else.

```
WRITERS                                STORES                                READERS

run --bench pinned/golden/--ab rows ──▶ perf table   (autotune.db) ─┐
golden import (once per digest) ─────▶ perf table   (autotune.db) ─┴──▶ greedy compile: ONE measured-evidence index
                                                                         (perf + golden rows on µs) — schedule AND
                                                                         kernel-set forks
emmy db import ◀─ freezes, goldens, tune DBs ─▶ dataset DB (_data/dataset.db) ─▶ emmy db export ─▶ dataset (_data/dataset)
dataset (manifest.json + one matrix per pool) ─┬▶ emmy eval prior (never a deploy)
  one per space: _data/schedule, _data/placement └▶ emmy fit ─▶ weights/schedule.json, weights/placement.json (repo)
recorded from those rows ────────────▶ recipe-local / hardware golden file ──▶ greedy compile (golden rows: the
                                                                              card's files, or --golden PATH)
                                       weights/schedule.json, placement.json ▶ greedy compile, the priors
```

Everything above is measured in ONE regime: the deployable one a compile runs in. A bench runs at the flags a deploy
compiles with, so a measured latency is the deployed latency and no store needs a per-regime lane (Part 3).

### How one fork gets decided, end to end

A worked example, to fix the vocabulary. Take `emmy compile` on a machine whose tune DB holds rows. A tile-lowering
rule matches a `LoopOp` and returns several tile options.

1. The engine turns the option list into a lazy fork tree and hands the fork point to `greedy_decide` (Parts 2, 4).
2. `greedy_decide` first tries to descend directly to a measured complete row. Without direct evidence,
   it ranks the complete offered rows — streamed off the lazy walk in bounded chunks, so memory stays O(chunk)
   however large the pool; no kernel is built until the choice is made (Part 4).
3. Each compared row contains the compile context's `H_*` features (which GPU, which nvcc flags), the kernel's `S_*`
   features (a summary of its body and loop extents, computed from the kernel), and complete knob values (Part 6).
4. **The evidence index.** The fastest measured row of the same kernel — the same exact identity — that agrees with
   an offered leaf: a tune DB `perf` row under this compile's own context key, or a golden row in scope; the two
   compete on µs alone. Agreement means every knob the leaf has decided has the same value in the row.
5. **The prior.** Otherwise: the `mean_scores` argmin over complete offered rows. A pool whose minted size bound
   exceeds the cold-pool budget is ranked over a deterministic drawn subset (seeded uniform descents through the
   lazy tree — legal complete rows, every level covered) instead of walked at full length: the cold pick needs a
   reasonable kernel, and the optimal one comes from measured evidence, which descends directly whatever the pool
   size. The seed is the pool's schedule-space stamp, which spells the precision gates by effect (Part 6's pool
   identity), so equivalent effective precision gates draw the same subset, while precise and fast defaults can
   draw different subsets. Drawing has a hard option-check budget while one descent fits
   inside it. If one complete descent's declared bound is already larger, exactly one descent attempt is the
   soft-cap exception; an empty sample fails rather than walking the full pool or substituting a partial
   branch. Under strict evidence this step is never reached: the
   compile raises `EvidenceError` for the kernel instead.
6. Ties at every step break by `knob.canonical_row_key`, never by the order the rule emitted its options in.
7. The winning leaf is built for real. The µs of whichever row decided it is written onto the fork's
   `Decision.score`, and the resolve moves to the next fork.

With no evidence and no prior at all, every fork falls to the first emitted leaf — not a chosen default, just what
is left when there is nothing to rank with (env pins still apply — a pinned family never reaches a decide).

### Terms used throughout

Everything in this table recurs on nearly every page below. The rest of the document uses these words freely.

| Term | Meaning |
|------|---------|
| **rule** | One pattern + rewrite function in a `NNN_<name>.py` file under a pass directory. |
| **pass** | An ordered directory of rules; the pass layout is frozen in a `Pipeline`. |
| **candidate** | One in-flight compilation state (a graph snapshot part-way through the pipeline). |
| **fork** | A rule returning multiple alternatives; the engine turns each option into a child candidate. |
| **knob** | A named tuning dimension (e.g. `TILE`, `STAGE`). Every fork option is identified by the knob values it fixes. |
| **to pin a knob** | To force a knob's value by hand instead of letting the compiler choose — from the environment (`EMMY_STAGE=d2/smem-async`), or by reproducing a golden entry's recorded values. A *pinned row* is a benchmark of such a forced configuration. |
| **to stamp a value** | To write a value onto an op as metadata, where later passes can read it: its kernel name, the knob values a fork decided. Nothing computed from the op is stamped onto it: a kernel's `S_*` *stamps* — its shape/body features — keep the name but are computed from the kernel where they are read (`features.stamps`). |
| **to realize** | A recorded configuration *realizes* at a fork when the options the compiler actually offers there include one that matches it. A recording that realizes nowhere cannot be deployed, no matter how good its recorded µs. |
| **regime** | The compile settings a measurement was taken under, or that a compile is running under: mainly the nvcc optimization level (`H_opt`) — `-O3` is the **deployable** one, and the only one anything is measured in — plus whether fast math is on. |
| **prior** | The ranking model — the **offline prior**, fit ahead of time by `emmy fit` and shipped with the repo. It answers only where nothing measured decides. |
| **terminal** | A fully-lowered candidate (every fork on its path resolved) that can be benchmarked. |
| **golden file** | A card's measurements in the tune DB's shape — kernels, kernel-set decisions, measured rows — beside the traced programs they came from. It stores inputs only; a compile imports its rows under the identity it computes from each stored kernel. |
| **the variant key** | The variant key measurements are stored under — `identity_key(with_io=True, with_knobs=True)`: the canonical Loop-IR body (a `TileOp` derives it schedule-free from its term) with its buffer roles typed + the knob row. Dialect-free: every stage of one rewrite chain keys off the same content. |

## Module map

| Module | What lives there |
|--------|------------------|
| `pipeline.py` | Engine core: `Pattern` / `Match` / `Rule` / `Pass` / `Pipeline` (the frozen pass layout) plus `Run` — the per-run state and engine loop. |
| `fork.py` | The `Fork` interface, `DeferredFork` (a leaf whose rewrite is built on expansion) and the lazy schedule tree (`_ScheduleTree` / `_ScheduleFork`) that `schedule.py` builds; `iter_leaves` / `leaf_for`, the walks `ForkPoint` wraps. |
| `schedule.py` | The generic adapter from a semantic `ScheduleContext` and codec to lazy schedule Forks, including pool sampling. |
| `knob.py` | The `Knob` descriptor system and the `EMMY_<KNOB>` env namespace (borrowing `config.knob_var` / `config.knob_raw`; `tuning_knob_items` is the tuning-knob view of a row). Holds NO concrete knob declarations. |
| `search/space.py` | **The single home of concrete `Knob` declarations.** It declares schedule codec knobs and kernel-lowering policy knobs; the classic typed move catalogs live with the classic model under `ir/schedule`. Registration is construction (`Knob.__post_init__`), and `knob.registry()` imports `space.py` before answering. |
| `search/features.py` | The featurizers (`Featurizer`, the one feature row every prior reads; `stamps`, a kernel's `S_*` features computed from its body; `knob_features`, `tile_signature`, the `D_*` / `MMA_*` encodings) — kept beside `space.py` so the whole space (dimensions × values × encoding) is analyzable in one package. |
| `search/db.py` | `SearchDB`, the persistent SQLite store (Part 6). |
| `search/policy/greedy.py` | `greedy_decide` — the fork resolver `compile` / `run` / `serve` use: Part 3's hierarchy, one pick per fork. |
| `search/strategy/` | The search shape above the loop: `base.SearchStrategy` and its one realization, `greedy.GreedyStrategy` — the greedy compile's retry orchestration (Part 4). |
| `search/inventory.py` | `KernelInventory`, the splice watcher that reports each kernel-set decision a lowering takes, and `record_routing`, the routing-row writer. The golden restamp and `run --record-greedy` compose one into their pipeline (Part 6). |
| `search/autotune.py` | `run --tune`'s proposal side: `schedule_space` enumerates the one scheduled kernel under the greedy kernel set, `prior_scores` ranks its rows, and `Autotuner` proposes the prior's best rows, then Bayesian optimization batches over the knob values split into their parts. It never touches the GPU; `run` measures. |
| `search/bench_record.py` | The perf-row writers: `persist_kernel_perf` (a `kernel` + `perf` row per benched kernel), `persist_bench_failure` (the `bench_fail` row of the kernel a failure names), `kernel_row` and `point_stats`. `run --bench` records through them (Part 5). |
| `search/prior/` | The ONE ranking path: the `Prior` ABC (`base.py`) and its one implementation, `OfflinePrior` (`offline.py`), which `load_prior` builds from the weights artifact. `catboost_model.py` holds `CatBoostModel`, the offline prior's scoring function as a value object — the one definition the fitter trains and the deploy path ranks by. `fit/` is the offline fitter, split by responsibility — `catboost.py` trainer, `cv.py` fold harness, `tables.py` the rank-table rendering, `run.py` the pure `emmy fit` run harness. The candidate pool it all trains over is `search/dataset/group.Group`, one layer down: a pool is data, not a fitter detail. |
| `search/metrics.py` | What a scored candidate pool is worth, as pure functions over numbers: golden ranks and their tie conventions, `topk_pick` / `topk_regret` against measured latencies, and Spearman ρ. No model, no I/O, no strings, so the callers cannot each hold a slightly different definition — every rank and regret metric resolves here. Rendering lives with the caller (`prior/fit/tables.py` for the fit's rank tables; the other top-k summaries have not been unified yet). |
| `search/dataset/` | The training data as values and as a document. `Group` (`group.py`) — one candidate pool packed as a matrix plus one label per row; the base says nothing about what the labels mean, which is all a ranking metric needs. `GoldenGroup` is the subclass whose labels MARK rows (`golden_ids`) rather than measure them, and it carries the `GoldenPool`s it was built from (`pool.py`: card, regime, sizes, the verified `GoldenRow`s, and the `KernelDef` the pool is enumerated from — `kernel.py`); `MeasuredGroup` is the one whose labels ARE the microseconds. `Dataset` (`document.py`) is the groups as a directory — `manifest.json` beside one `.npy` per group — written by `emmy db export`, read by `emmy fit` and `eval prior`; the leaf values are wire classes. `measured_features` (`sample.py`) is the feature row of a measured `perf` row, computed from the row's kernel, and `ShapeKey` the compact shape key read off a kernel's stamps. Nothing here reads a DB — `db/export.py` builds the groups, the one place the two packages meet — and nothing imports `search/prior/`: a group carries every column it was given, and the model narrows to the ones it wants when it asks for the matrix. |
| `search/db/` | The SQLite store (`SearchDB`: the kernels, the decisions that minted them, their measurements) and what fills and drains it: `freeze.py`, a DB's admitted rows as a golden file per card (`freeze_reason` is the one admission rule every measured-pool reader applies), and `export.py`, its rows as the dataset (`golden_pools`, `measured_groups`, `export_dataset`) — where `db` rows become `dataset` values, in that one direction. |
| `search/golden/` | The golden package, one module per job: the file format in the DB's shape (`format`: `GoldenFile`, `Kernel`, `Row`), the import into the DB a compile reads (`evidence`), the repository index and the evidence scope (`repository`), the rewrite onto a fresh lowering (`restamp`: `restamp`, `mint`, `lift_targets`), the working golden's writers (`working`: trace inventories, `record_greedy_pick`, `record_latency`). |
| `slice.py` | Isolates one finalized kernel into a standalone graph (used by the working golden's per-kernel slices). |
| `dump.py`, `rule_diff.py` | The dump and `-vv` presentation layers (see the end of this file). |
| `passes/{frontend,loop,lowering}/` | The rules themselves — documented in [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); a per-pass overview table is near the end of this file. |

## Part 1: The rewrite engine

This Part covers the mechanics every rule author touches: how a pattern matches, what a rewrite may return, and how
the engine splices the result back. Nothing here involves tuning.

### Patterns and matching

A `Pattern(name, op_type, constraints={})` matches one node by op type plus optional `node.op` field equality. A *list*
of patterns matches a chain: the seed node matches `pattern[0]`, its sole consumer matches `pattern[1]`, and so on.
Multi-node patterns only fire when each intermediate node has exactly one consumer.

`match_pattern(graph, pattern) → list[Match]` walks every topo-ordered seed. Overlapping matches are allowed — the
rewriter exits after the first successful rewrite per iteration, so overlap is just candidate enumeration.
`Match.nodes` maps each pattern entry's name to the matched `Node`. `Match.consumed` and `Match.output` are
overridable by the rewrite function, to control which nodes the splicer removes and which buffer edges get rewired.
Matches retain the watched node objects themselves, so `Match.is_alive()` rejects removal followed by a different
node at the same graph id even when the Python allocator would otherwise recycle an integer object address.

### Writing a rule

Every file named `NNN_<name>.py` under a pass directory is a rule:

```python
PATTERN = [Pattern("root", SomeOp), ...]  # required


def rewrite(ctx: Context, graph: Graph, match: Match) -> Graph | Op | list[Graph | Op]: ...
```

- The dispatcher binds `rewrite`'s parameters **by name**. Reserved names: `graph`, `match`, `root`, `out`, `ctx`.
  Pattern names from `PATTERN` bind to their matched `Node` objects. Anything else binds positionally to
  `root.inputs[i]`. Take only what you need — `ctx` is optional.
- Files starting with `_` (e.g. `_broadcast.py`) are **not** loaded as rules — they're shared helpers.
- Raise `RuleSkipped(reason)` to decline a match; the engine logs the reason at DEBUG and moves on. Add
  `reject=True` when the decline is a **lowering** refusing the offered row rather than a pass legitimately passing
  the node on: that records into the `rejections` sink below, so the report a stranded node raises can name the pass
  and the reason it declined. Without it the node still stops the compile — what counts as stranded is read off the
  terminal graph, not off the sink — but the report can only name the op the node is stuck on.
- A rule module may declare `FIXPOINT = True`. After every successful rewrite the cursor stays on that rule; only a
  quiescent match batch advances. `030_cut` uses this so every fresh kernel finishes structural cuts before scheduling.

### Strategies — engine events for cross-cutting concerns

The engine is IR-dialect-agnostic: it emits a small fixed set of EVENTS — frozen records of an engine moment
(handlers act on the compilation state an event references, never on the event) — and never branches on pass
names, dialects, or per-concern flags. Two prefixed protocols share the "strategy" vocabulary, each with its own
ABC, told apart by what the loop is doing when they act:

- **`PipelineStrategy`** (`pipeline/strategy.py`) — reacts to what the loop DID (events below): the cross-cutting
  concerns (provenance, the kernel inventory). Never steers the resolution; the loop's trajectory is
  identical without them.
- **`SearchStrategy`** (`search/strategy/base.py`) — the search SHAPE above the loop: which pass lists a resolve
  covers, what decides its forks (`greedy_decide`'s decide callback), and what happens when a pick fails to lower
  (`GreedyStrategy`, which implements `run(graph, ctx)`).

The composition chain for a compile: a `SearchStrategy` constructs a run with a decide callback inside; the loop emits
events that `PipelineStrategy` instances act on. Each layer only knows the one below it. Every cross-cutting
concern is a `PipelineStrategy` implementing the event methods it cares about; extension is a new strategy over
the existing events (or a new event field), never a new engine parameter. The events, each a payload object:
`RunStartEvent` (a loop starts driving a graph), `SpliceEvent` (before a `Graph` fragment splices in — op identities
stable; it carries the selected fork's knob delta because a fragment does not inherit the consumed op's knobs;
strategies may mutate fragment OPS, never the graph or cursor), `SplicedEvent` (after the splice, carrying its
`SpliceReceipt` — `Graph.splice` is pure surgery and hands back what it did), and `PassEndEvent` (a named pass
completed a quiescent scan).

Two binding scopes share the protocol:

- **Discovered** (build-scoped): strategy modules are plain `.py` files at the top level of `passes/`
  (`passes/provenance.py`); `Pipeline.build` collects every `PipelineStrategy` subclass they
  define into shared instances (`strategy.discovered_strategies`), class-name-sorted. Dispatch order MUST NOT be
  load-bearing — no strategy may depend on another having handled an event first. Build-scoped instances are shared
  across runs and candidates: immutable config plus content-keyed caches only, never trajectory state.
- **Composed per run** (`Pipeline.with_strategies`): instances with per-run state — e.g. a record run's
  `KernelInventory` — composed into the run's own pipeline instance after the discovered set. A pipeline
  composed with stateful strategies serves one run; sharing across runs is only safe when every strategy is
  stateless.

The one discovered strategy: **`ProvenanceStrategy`** owns op provenance end to end (`seed` at run start,
`propagate` from the splice receipt, mint for `frontend/decomposition`'s fragments, aggregate for everything else)
and keeps the replaced result's ultimate `Op.source` object on its rewrite fragments. The pattern root may be an
upstream producer while `Match.output` names the consumer result that the fragment replaces. A fragment may consume
inputs from other origins without losing that result identity; those producer edges retain their own sources. The
source identity lets semantic rewrites distinguish a frontend operation's private decomposition edges from tensor
boundaries between operations. A pipeline built without the strategy has no provenance anywhere, and `graph.py`
imports none of it.
No strategy stamps a kernel's identity or its `S_*` features: both are computed from the kernel where they are read
(Part 6), so no rule can observe a kernel before its facts exist. The search shapes (`SearchStrategy` subclasses)
are the same idea one level up — they own
loop composition and terminal aggregation — but are constructed by their entry points, not discovered.

A rule always sees **graph-true operand Tensors**: op `inputs` / `outputs` are refreshed from the graph at match build
AND again at apply time (`Candidate.try_rewrite`). This matters because an earlier apply in the same batch may have
swapped a consumed node's op for a rebuilt instance still carrying its `(f32, ())` seeding placeholders — a change
`Match.is_alive`'s node-identity check cannot see. (This was the gemma o_proj misdeploy: a scalar tile shipped at 16x
the kernel's measured mma rows because the warp atom gate read placeholder dtypes off an all-f16 graph.)

### The three kinds of rewrite result

The return type discriminates the rewrite flavor:

- **Functional** — returns a `Graph` fragment, spliced in place of `match.output` (defaults to `match.root_node_id`).
  A dictionary maps several old buffers, including secondary outputs, to fragment output buffers in one splice.
  Fragment `InputOp` nodes reference existing graph buffers by id; non-Input nodes get fresh ids.
- **In-place** — returns an `Op`. The engine assigns it to `root.op` directly, preserving the node id, inputs list,
  output Tensor and hints. The lowering rules use this because `KernelOp.arg_order` / `CudaOp.arg_order` embed the
  original node id as the output buffer name — a fresh id would break the generated kernel's buffer binding.
- **List = fork.** A rule unsure which parameter to use returns the alternatives as a list, in any order. The engine
  offers them ALL to the resolve's decide callback as one `ForkPoint`, and the greedy pick chooses by measured
  evidence, then by a `Prior` (Part 3). A single-option return (or a bare `Graph` / `Op`) is the deterministic
  case — no fork.

### Rules must be idempotent

Every rule MUST be idempotent on its own output. The engine re-runs the entire pipeline on each popped candidate from
pass 0, so a rule whose output is already in the graph must `RuleSkipped` or have a pattern that no longer matches.
Most rules satisfy this implicitly via op-type changes (`LoopOp` → `TileOp`); the rest carry explicit
`raise RuleSkipped("already X")` guards.

### How fragments are spliced in (`engine._apply_replacement`)

1. Walk the fragment in topo order. `InputOp` nodes forward their id to the existing graph buffer (external
   reference); non-Input nodes are added with fresh ids.
2. Rewire each requested old buffer's consumers and `graph.outputs` slots to its fragment output buffer.
3. Merge redirected owners' hints onto their new producers; when all ports belong to one multi-output node, merge
   the dissolved internal nodes there too.
4. Remove consumed nodes, restore each redirected primary or secondary buffer's old identity, and drop orphans.

## Part 2: Forks — how choices are represented

A fork is how a rule says "these N options are all correct; you pick". This Part covers how the options are
represented, what identifies one, and what happens to an option that turns out to be invalid. Part 3 covers how one
gets chosen.

### Lazy hierarchical forks

A fork with many options would be expensive to build eagerly, and most options are never visited. So forks are lazy
trees. `Fork` (`fork.py`) is an interface with four members:

- `knobs` — the knob values this fork level fixes. Those values are the variant's identity: both the perf DB and the
  prior are keyed on them, and they can be read **without expanding** the fork.
- `is_leaf` — whether this is a concrete option or an inner branch.
- `expand()` — builds the next level of options.
- `sample_child(rng)` — one option of the next level, drawn uniformly, or `None` where the branch has none. The
  default expands and draws; the schedule fork answers without expanding, by asking its context for one extension
  (`ScheduleContext.random_step`). This is the step of a random descent (`fork.descent_sample`): the cold-pool
  draw of a greedy compile and the pool draw of a dataset export (`PoolSample.draw`) both walk it, so a draw costs
  the extensions it tries, never the frontiers it passes. The cold-pool draw runs in 64 pieces, each seeded on
  the pool identity and its index, on `EMMY_WORKERS` forked processes (`fork.parallel_descent_rows`; one per
  core by default, `1` in this process, which the test suite sets). A worker returns knob rows, since the lazy tree
  cannot be pickled, and the greedy builds only the leaf it picks; the rows are the same at any worker count.

A pick calls `expand()` only on the branches it descends into, so only the subtrees a resolve actually walks ever get
built. `DeferredFork` is a leaf whose selected rewrite is materialized only when expanded — what the cut and split
passes offer, and what the engine lifts a concrete `Op` or `Graph` option into. Structural leaves mark themselves
directly, so graph-building can remain lazy without policy inspecting their implementation.

The one hierarchical fork is the lazy schedule tree (`schedule.py`, over a semantic `ScheduleContext`): each
`_ScheduleFork` prefix expands to the next decided site, and its leaves are `ScheduleLeaf`s. Complete leaf walks are
iterative and depth-first, so a maximal fused kernel with thousands of schedule levels does not consume the Python
call stack.

### Every finished option carries a value for every knob

**Every emitted variant carries an explicit value for every declared knob** — no complete option leaves a knob absent.
This rule is known in the code as the knob-stamp invariant.

Each `Knob` declares an `off` value, meaning "unused here / this pass declined it". At the end of each pass
(`Cursor.advance` → `_off_fill_pass`, via `knob.apply_off_defaults`), the pipeline fills in that `off` value for any
of *that pass's* knobs the variant left unspecified. It covers a pass that acted, one that declined, one that was
skipped and one that returned no variants, all the same way. Filling only the just-finished pass's knobs is
deliberate: writing a later pass's knob early would trip that pass's idempotency guard.

Why it matters: the featurizer fills any missing feature column with NaN. Because a finished option always writes an
explicit "off", NaN can mean exactly one thing — *not yet decided*, i.e. a partly-decided option seen part-way down
the fork tree — and never "decided: unused", which is the explicit off value on a complete option. A knob with no
`off` value (the `_UNSET` default — a knob its owning pass always writes itself) is never auto-filled. Which code path
a variant belongs to is always read off knob *values* (`knob.is_warp` / `knob.mma_atom`), never off a knob's presence.
Verified by `tests/compiler/passes/test_knob_stamp_invariant.py`.

### Invalid options: rewrites that get filtered, and rewrites that raise

A rewrite that *returns* an op failing `Op.validate(ctx)` — e.g. a `KernelOp` whose smem exceeds
`ctx.max_dynamic_smem` — is dropped by `Candidate.try_rewrite`. In a single-path greedy compile that is fatal: it
leaves the node un-lowered. The
same holds for a rewrite that *declines* the row (`RuleSkipped`) and for a node no rule matched: the node reaches the
terminal with its pre-final op either way. So:

- **The settled terminal is the evidence, not the sink.** A pipeline whose last pass is `lowering/cuda`
  (`Pipeline.lowers_to_cuda`) promised a graph of `CudaOp`, so any node still holding a `LoopOp` / `TileOp` once the
  resolution settles is stranded — whether or not a rule recorded anything. That set is what drives the greedy
  retries and what `_raise_on_unlowered` raises the loud `LoweringError` on, instead of leaking a cryptic
  `non-CudaOp` `TypeError` to the backend. Reading the sink alone missed every strand nothing records — a
  materializer declining a row with an ordinary `RuleSkipped`, or no rule matching the node at all — and those
  compiles returned a half-lowered graph and reported success. A truncated pipeline (`TILE_PASSES`, `LOOP_PASSES`)
  terminates in an earlier dialect by design, so there only a node with a recorded rejection counts.
- `Pipeline.run` installs a `rejections` sink on the `Run`, recording each drop as `(node, pass, reason)`. It does not
  decide what is stranded; it supplies the pass and reason the error names.

A rewrite that *raises* mid-lowering — a deterministic pass hitting an un-representable shape — is the same dead end
expressed as an exception, and `resolve` lets it propagate. The raise lands wherever the rule's code runs: in the rule
batch itself, or later, when a deferred `Fork`'s thunk fires at expand/resolve time — both sit inside the sink.

## Part 3: The prior — how choices are ranked

This Part answers "how does a fork get decided when nothing may be benchmarked?". Its core is **the deploy
evidence hierarchy**: the fixed order a greedy compile walks, measured evidence first, then the prior. The sections
before it explain the machinery that order leans on; the ones after it are the guards that keep the machinery honest.

### One ranking path

Ranking always happens in one place: the greedy pick asks a single `Prior`. Forks carry no score of their own, and
nothing builds or scores a `TileOp` merely to rank it — the one featurizer (`features.Featurizer`) turns the kernel
and the row's knob values straight into features. Several older per-variant scoring mechanisms were removed in favor
of this single path and the design it retired.

That one path has one model: the `OfflinePrior` that ships with the repo, which `load_prior` builds from its weights
artifact. It is the *cold* ranker — what answers on a machine that has no measurement of the kernel, a freshly rented
box, say — and it answers nowhere else, because measured evidence outranks it at every fork (the hierarchy below).

### The offline prior

`OfflinePrior` scores a candidate with a CatBoost ranker — a sum of small decision trees — over the `D_*` features,
hand-designed descriptions of a tile's geometry and its occupancy, fitted ahead of time. It never falls back on the
order the rule emitted its options in. The complete scoring function lives in the repo-checked artifact
`search/prior/weights/schedule.json`: the column order, the scalar `scale`, a `feat_ver` version, a `provenance`
block, and the trees themselves in CatBoost's own JSON model format (`model`), which CatBoost loads back with
predictions identical to its binary form. The offline fitter writes it (`search/prior/fit/`, driven by `emmy fit`).
The training pools are the golden groups of the dataset `emmy db export` writes (`db/export.py` over
`search/ranking.build_golden_groups`, Part 8): one per kernel, card, regime and sizes a golden file recorded a row on,
enumerated from the kernel's own definition — the fit reads the directory and enumerates nothing.

**The placement prior** is the same model class over another space. `weights/placement.json` ranks the arms of a
kernel-set fork (`pins.KERNEL_SET_DOMAINS`) — keep the kernel whole, cut one offered seam, split it across CTAs at one
width, store a constant in its source layout — each featurized as `P_*` columns from the `S_*` stamps of the kernels
the arm leaves (`features.piece_features`: the piece count, each stamp summed and maxed over the pieces), plus the
number of kernel roots that fold a whole contraction. That fact separates cuts with equal Loop histograms but
different projection placement. Its dataset is `emmy db export --space placement`: one pool per kernel-set fork of
every golden kernel, walked through the lift and the cut pass only (`ranking.walk_placement`), the arm the golden
took marked — the cut, or the split width; the first arm, which keeps the kernel as it is, where it took none. A
fork's group carries the report tier of its domain — `place`, `split` or `layout` — or `dyn` where the kernel has a
symbolic axis, as every golden group of a symbolic kernel does. The tier comes from the root kernel's derived shape
and must agree with the dynamic flag in every arm's features. The greedy asks it at every kernel-set fork no
measured arm decides (`policy/greedy._kernel_set_pick`), with the same featurizer, so the dataset's rank and the
deploy's pick are one computation. Both artifacts name their `space`, and a reader refuses the other's. The
dataset holds no layout fork today: a kernel's own definition reads no constant, so no walk reaches one, and the
prior's pick at a layout fork is an extrapolation until evidence decides it.

The placement view also retains `H_cc`, `H_total_mem` and `H_fast_math`. They are constant inside a fork, but a tree
can combine them with arm features to learn a different ranking per card, including same-die SKUs with different VRAM,
and per precision regime: a golden can split a kernel under fast math and keep it whole in the precise regime.
The export prices nothing: the label is what the golden did. The import marks every decision on the way down to a
golden row as taken under that row's card, precision regime and sizes (the `taken` table), whether or not the row
holds a time, so a golden that cut a kernel marks the cut and one that kept it whole marks keep-fused. A routing row
names no card, so it cannot mark a cut on a card whose golden did not take it. Shared cut parents receive one pool
per context a golden cut them under, and the walk keeps decisions within it.

The proxy stays uncalibrated, and nothing in the deploy path corrects it by hand. Where a prior ends up deciding a
production election, the defect is the missing evidence — no recorded golden or measured row for that kernel — and
the fix is to record it, not to bound the estimate. (`D_serial_cell_work`, the kernel's per-thread serial work
log-scaled, rides the featurization as an ordinary fit signal.)

What a newcomer needs to know about the fit:

- **The objective is the deployed ranking.** `QuerySoftMax` over one group per candidate pool: every row a golden
  verified is a positive, and a uniform draw of the other rows (`--negatives` per pool) are the negatives. The rank
  every report quotes is still taken over the FULL pool, through the same `CatBoostModel.score_rows` the deploy path
  ranks by, so the sampling never reaches a metric.
- **The trainer is an object, and fitting is pure.** `CatBoostTrainer` carries the hyperparameters — feature names,
  trees, depth, learning rate, negatives, mining rounds, seed — and `fit(groups)` returns a `CatBoostFit` without
  touching the trainer, so one instance serves every cross-validation fold. A fit is not byte-reproducible (CatBoost's
  histogram build is threaded); two fits are compared by their metrics files.
- **A group is a candidate pool, and it may have more than one right answer.** `GoldenGroup.golden_ids` is the
  set of rows in that pool a golden verified: usually one, several when the builder matched
  several goldens onto one pool (the same shape recorded under two names, or one name recorded twice). Which
  goldens share a pool is settled before any group is built, so a group's labels are final at construction.
  The per-group rank is then the BEST rank over that set (`search/metrics.best_rank`), because deploy ships one
  config: any acceptable one ranked first is the win. Sibling positives are never drawn as negatives.
- **A pool may be a SAMPLE of itself.** `emmy db export --pool-sample N` draws its candidates during enumeration
  (`search/pool.py`), so `Group` carries both the drawn rows and `total`, the true pool size a report prints beside
  the raw sample rank.
- **Absent is not zero.** A feature a row never stamped is `NaN`, CatBoost's own missing bucket, so "this knob is not
  decided" stays a different fact from a knob legitimately at 0. The dataset stores `NaN` and the model reads it.
- **Loading is strict.** A missing artifact, one whose `feat_ver` does not match, or one lacking `cols`, `params` or
  `model` is a hard error — refit it, never a silent fallback. The error surfaces in `eval` / `fit`, which load the
  prior directly. A greedy compile wraps `load_prior` best-effort, so there a bad artifact does not abort the compile:
  it produces the no-prior resolve described under the hierarchy below (first leaf, with the evidence index unread
  along with the prior object). `EMMY_OFFLINE_FILE` (or `emmy eval … --offline-file`) swaps in a candidate fit for
  an A/B.
- **Symbolic-axis kernels are one more split.** A kernel whose tiles are masked because an axis is symbolic carries
  the stamp `S_ext_n_symbolic_axis`; every feature view keeps it, and the trees split on it to price both regimes in
  one model.
- **The quality score is turned into a positive stand-in for latency by an exponential** (`exp(-scale·quality)`),
  so a greedy argmin reads it like a latency.

**Known gap: the fit never sees the rows a deploy ranks.** The fit trains on a 2000-row draw of each pool and 500
sampled negatives, and the reproduction gate scores a 512-row draw. A cold deploy ranks 8192 rows (`EMMY_POOL_DRAW`)
drawn from the WHOLE pool (`policy/greedy._descent_sample`), so it reaches candidates no fit or gate ever scored, and
the model can rate some of them far above the golden. On the V100 Qwen3.8-27B-FP8 golden the deployed lm_head ran
29× slower than its golden row and a fused matmul-reduce exceeded the 60 s bench limit, while every gate slice
reproduced. The fix is a fit and a gate that draw the way the deploy draws; until then a cold V100 compile needs
recorded evidence.

**A subtlety about features.** The `H_*` features (which GPU, which nvcc level) have the same value for every
candidate competing at one fork, so on their own they cannot change a ranking within that set. A tree can still
combine one with a per-candidate feature — split on the card, then on the accumulator width — which is how a single
model can prefer f16 accumulation on one architecture and f32 on another; the default view carries `H_cc` for that
reason. The `D_w_grid_*` features separate candidates with the same tile but a different warp grid, which used to
produce byte-identical feature vectors. A feature that is a monotone transform of another, or a threshold on one,
does not exist: a tree forms it with a split.

### What a `Prior` offers its callers

The names below recur throughout this document; together they are the whole public API of a `Prior`:

| Member | Caller | What it is |
|--------|--------|------------|
| `mean_score` / `mean_scores` | deploy + eval ranking | The model's latency prediction for one row / for a batch of candidates. |
| `pick(rows)` | deploy + eval | The `mean_scores` argmin with the canonical tie-break. Returns `(index, predicted µs)`. `greedy_decide` consults the evidence index before it asks, so the `Prior` never owns the whole hierarchy. |
| `mean_score_features` / `mean_scores_features` | the model class's own seam | Scoring a row that is ALREADY in feature form. `mean_score` / `mean_scores` featurize and delegate here, so a model class implements the featurized half only. Not a pool-scoring surface — that is `score_rows`, which projects a packed matrix and is what the fitter and the evaluation report use. |

### The deploy evidence hierarchy

`greedy_decide` (`compile` / `run` / `serve`, via `Run.resolve`) never explores: at each fork it picks once, working
down the list below from the top.
**This list is the authoritative order** — the summaries elsewhere in this file defer to it.

**Measured evidence, then the prior — that is the whole ranking mechanism.** Every measured row is a recording of
something that ran, and the prior is a model fitted to such recordings; there is no hand-written step anywhere below
them. The passes that produced the candidates ordered nothing, defaulted to nothing and withheld nothing (see
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md)), so when nothing measured and no prior speaks the pick falls out of
the enumeration's emission order, which carries no meaning. Such a pick can be far off the best kernel in the space,
and that is an accepted outcome of the design — the fix is a measurement (a benched golden row, a `run --bench`) or a
better-fitted prior, never a preference written into a pass or into this policy. A compile that would rather fail than
guess says so: under **strict evidence** (`config.strict_evidence`, `EMMY_STRICT_EVIDENCE`, set by
`--strict-evidence` on `run` / `compile` / `serve`) a fork with more than one option that no measured row decides
raises `EvidenceError` (`greedy._require_evidence`) naming the kernel and the fork, and the `prior=None` emission-order
fallback raises the same way.

At a **schedule fork** (one kernel's row):

1. the **evidence index** (`greedy._db_measured_index`): the fastest row of the same kernel — rows are indexed by the
   exact identity of the kernel they measured, and the fork asks for its own (`wire.kernel_identity`) — that agrees
   with an offered leaf, whether it came from the tune DB or from a golden file. `_direct_measured_pick` descends the
   lazy tree straight to it, asking each branch whether it admits the row (`Fork.admits`: a schedule branch spells a
   prefix of what its leaves will spell, a level branch a projection of it, and a knob the row leaves undecided is
   free), so a partial row reaches its leaf whatever the pool size. The prefix reading excludes the empty spelling: a
   site the branch itself DECIDED off admits only off, since every string extends the empty one and the descent would
   otherwise fail no earlier than leaf matching. An off merely inherited down the branch is a pin rather than a
   decision and still admits. The index is built once per compile and memoized per process on
   the DB path and mtime, the context key and the card. It holds the DB's CUDA `perf` rows for this
   compile's context key (one lane, because a sweep measures in the regime a deploy compiles in; rows from a
   deliberately non-deployable `--nvcc-flags` run key elsewhere and are simply never consulted) — the box's own
   `run --bench` rows and the **golden rows** in scope, which the compile imports before it picks
   (`golden.evidence.evidence_db`): the live card's repository files, or the file `--golden PATH` names, once per
   golden digest into the tune DB — created on first use, and imported under the lock beside the file so the workers
   of a parallel boot sharing one DB import it once — or into an in-memory instance when the compile has none; a
   re-recorded file changes the digest, and its earlier rows on this card and regime are let go first. A golden file is
   the DB's shape, so the import goes row for row: every kernel a `kernel` row, every decision a `routing` row (and a
   `taken` row where a row of the file sits below it), every measured row in the live input regime
   (`evidence.regime_live`) a `perf` row of its kernel. The one thing the import computes is the key: the file names
   a kernel by a local `ref`, the DB by the exact identity of its Loop IR (`GoldenFile.identities`). An unmeasured
   row (a proposal) is not evidence: `run --golden PATH --bench` measures it under a hand pin and writes the
   measurement as `perf` rows, after which it deploys like any other;
2. the prior's `mean_scores` argmin — only when no candidate has any evidence at all. Score ties break by
   `knob.canonical_row_key`, never by the order options were emitted in.

At a **kernel-set fork** (the cut pass's placement fork and its cross-CTA split fork), the same rule holds — measured
first — over one more kind of evidence. A kernel-set decision is a `routing` row on the exact kernel — a golden's cut
or split, imported as one — and its price on this card is the sum of its pieces' fastest rows, each piece at its own
projection of the fork's bindings, all-or-nothing (`SearchDB.priced_arms`); a decision no piece's row prices is off
the measured ballot, which is what a golden's cross-CTA split timed as a whole is until its pieces are benched. An
offered split or cut that no routing row names is priced the same way, from its own pieces' rows, so a bench that
measured a split's partial and finalize (and wrote no routing row) still puts that split on the ballot.
`greedy._route_candidates` turns EVERY measured row of the kernel, and every priced decision on it, into a candidate,
each one of the pass's OWN offered arms: the arm the row spells (`pins.spelled_arm` — a schedule row the fused /
unsplit arm, since the kernel it decorates ran that way; a routing arm the composed arm that cuts exactly the several
offered seams it marks `cut` — the one decision a pinned compile consumed them as, which the cut pass offers beside
its single seams wherever a stored decision of the kernel names it (`pins.composed_routes`, registered by
`GreedyStrategy.run` under the parent's exact identity) — else the first offered seam it marks, or the offered plan
whose `g<n>` half its `REDUCE` value carries; an arm whose cut seams are not on this ballot decides nothing). Among
measured arms the fastest wins; strict evidence refuses a kernel-set fork no measured arm decides — a fork with more
than one arm left, that is: a hand pin that leaves one arm decides it, which is how a kernel set gets recorded under
strict evidence before its routing row exists, and the strict check then falls on the pieces. With no measured arm,
the fork goes to the placement prior (`_kernel_set_pick`: its argmin over the arms' `P_*` rows, the arm that keeps the
kernel whole included); without the shipped placement weights, or on a resolve with no schedule prior, the first arm
wins — the kernel stays fused, unsplit and folded. No arm is scheduled to decide the fork (Part 4). A measurement can
also disqualify: an arm that leaves a kernel whose every measured variant failed (`_Measured.failed`, the watchdog's
`bench_fail` rows) is off the ballot while another arm remains. Nothing is installed on the kernel: a piece a cut or
split mints is a brand-new kernel (`knob.consume_kernel_row` strips every decision family), its own forks consult the
rows of its own identity, and a piece that fails to lower re-ranks at its own forks and, once no row of it binds,
retires the one cut that minted it (`Pipeline.run`'s retry).

The cut pass's layout fork also changes the kernel identity. It offers a transposed constant's folded storage and its
source storage as separate kernels; weights with equal reads may choose source storage together. Search compares each
arm's measured kernel row or a measured later route from that kernel. A layout routing row records the choice, but it
does not price the folded parent from the source child's row. With no measured arm, strict evidence refuses the fork;
otherwise the placement prior ranks its arms like any kernel-set fork's.

Env pins sit ABOVE the whole list: a hand pin (`--ab`, `EMMY_KNOBS`, `EMMY_<KNOB>`) settles the pinned families before
any fork reaches a decide. That is how a row is MEASURED — `run --golden PATH --bench` pins each golden row and each
`--ab` row for its own compile — not how a golden deploys.

**Auditing the golden rows.** Whether the recorded goldens still decide a deploy is the strict-evidence question
asked of a whole serving matrix. `eval golden --golden … --serving-config` compiles each precision lane's serving
twins inside `golden.sole_evidence([file])` — the lane's rows are the golden scope, no tune DB is passed, and strict
evidence is on — so a fork no golden row decides is an `EvidenceError` naming the kernel rather than a prediction, and
the answer is the same on every machine that holds the same file.

### Golden rows: what a golden file is at deploy

**The per-GPU golden files are the only *measured* data that ships with a clone.** A golden file holds the tune DB's
three tables for one card — the kernels it measured, the kernel-set decisions taken on them and the measured rows —
beside the traced programs the targets were lowered from. Its uses are measured evidence for the greedy compile
(above), pinned measurement (`run --golden PATH --bench`, `--ab`), training data for the offline prior (`emmy fit`,
through `emmy db import` and `emmy db export`), the `emmy eval` datasets, and regression reference points.

At deploy a golden is tune DB rows, nothing more (`golden.evidence.import_rows`): each kernel entry is a `kernel`
row, each routing entry the `routing` row of the decision it records, each measured row the `perf` row of its
kernel — captured, with the golden's digest as its source. A file stores no identity, so the import computes the key:
each stored kernel's exact identity, from its Loop IR (`GoldenFile.identities` — a lift per formed kernel, once per
loaded file), and the rows are filed under it. A stored kernel whose body no longer lowers is skipped with a warning,
and its rows are no evidence. Whether a file still describes what the compiler builds is the restamp's question
(Part 7): a stale file is one `emmy golden restamp` would change.

**Goldens are the prior's training data.** `emmy fit` enumerates, for each kernel a golden row was measured on, the
candidates that kernel offers, and trains the weights to rank the recorded row well inside that set (Part 8).

### Featurizer versioning

`features.FEATURIZER_VERSION` is written onto every stored training artifact:

- **The offline weights artifact** carries it as `feat_ver`, and a mismatch refuses the load (above). The dataset
  `emmy db export` writes carries it too, and `emmy fit` / `eval prior` refuse a stale one: rows featurized under a
  retired version's names produce meaningless feature vectors, and a fit on them collapses to constant predictions.
- **The DB's rows and a golden file's** carry no featurizer version, because they store no feature: a kernel's `S_*`
  stamps are computed from its stored Loop IR where a row is featurized, so a stored row outlives a featurizer
  change. What the DB does version is its kernel key (Part 6).

Bump the constant on any incompatible change to knob naming or feature encoding; artifacts from the old version are
then refused instead of poisoning the model.

## Part 4: The driver — the greedy compile

`Pipeline.build(passes)` wraps a pass list; the result exposes the compile entry point, `Pipeline.run`, which drives
the one `Run` engine loop, `Run.resolve`: a deterministic resolution whose forks a decide callback answers.

### `Run` — the per-run state

`Run` bundles everything scoped to one compilation: the pipeline, the `ctx`, the tune DB, the dump and the
`rejections` sink. `Pipeline` stays a frozen, shareable pass layout, while everything that collects output for one run
lives on the `Run` and is reached through the candidate (`cand.run.dump`, `cand.ctx`).

### `Run.resolve` — deterministic resolution

`Run.resolve(graph, decide) -> (Graph, list[Decision])` walks the graph once, one rule batch (`Run._step`) at a time:
ONE live graph is mutated in place, with no sibling snapshots and no per-fork copies, so the terminal IS the graph it
started from. At each
undecided fork a `decide` callback gets a `ForkPoint` (the `Match`, the raw options as the rule emitted them, the op
as it was before the decision, `ctx`) and returns the option to apply. The fork point owns the walk over its offer:
`leaves()` streams the complete leaves depth-first in emission order (the order a score tie falls back to, option-0
first), `flat()` lists them, and `find(row)` is the one row-directed descent the evidence pick, the decision memo's
replay and the golden replay share — so every consumer reads an offer the same way. The `fork` module's `iter_leaves`
and `leaf_for` remain for option sequences that are no fork point: a rule's forks before they are offered, a
filtered sibling list.

The returned trace — one `Decision(rule_name, node_id, chosen_kind, knob_delta, score, n_options)` per decided fork —
is the resolution's process-state output. Questions like "did this compile take a structural pick" or "what did the
partition fork predict for this kernel" are trace queries, never accumulated policy attributes. The greedy compile
copies only the final trace's placement receipts into graph attribution so a later A/B integrity check can verify
structural pins after the splice has consumed them.

### `Pipeline.run` — the greedy compile

`Pipeline.run(graph, *, backend=None, db=None) -> Graph` is a single-shot greedy compile: a deterministic resolution
(`Run.resolve`) with the greedy pick (`greedy_decide`) — NOT a search. No frontier, no tree, no benching. The graph is
copied once per attempt and resolved in place — no per-fork copies.

`emmy run --golden PATH --strict` consumes a working golden. It visits every distinct realization name sequentially
in the current process, or one selected with `--realization NAME`. A named realization benches as a pinned row
whatever its measurement state; the whole-file walk benches a name's verified rows and skips its proposals (the
unmeasured rows). The ordinary strict run accepts only captured whole-forward
timing with direct eager correctness at `rtol=atol=1e-3`. Process isolation and repeated observations come from
independent command invocations, not a second orchestration layer inside `run`.

**Greedy compares complete rows.** A branch carries only a partial schedule and is not a valid prior input. Measured
rows descend directly to their exact offered row; otherwise the prior ranks the complete offered rows
and selects the global argmin. This preserves the fitted model's semantics, at the cost of traversing the row space
until schedule composition and scoring are factorized end to end. The `OfflinePrior` ranks (including a positive
`MMA_tier` warp preference — a fitted weight, not a hand-written rule); if `load_prior` returns nothing every fork
falls to the first leaf in emission order, which is meaningless and may be slow. Greedy benches nothing, so it can
only *use* a prior.

**And it scores each decision once.** A decision is a conclusion over evidence, so it is memoized GREEDY-SIDE (one
factory call — one compile attempt; never shared ambient state): the memo
keys on the minted pool identity (`Fork.pool_id` — the deploy identity plus the knob / hint / pin
discriminators it excludes) plus the node's blocklist content, so N same-shape kernels score once and the rest replay
by descending the lazy tree to the one matching leaf (`_find_decided_leaf`, through the same `Fork.admits` — an
O(path) descent),
while a validate-retry with a blocked tile is a different key and re-decides.

**Every deploy pick breaks ties by candidate content, never enumeration order.** The model can score many
same-featurized siblings identically (the offline `D_*` geometry doesn't separate an `f2x4` from an `f4x2` fragment or
the `bk` variants — 8 exact ties at the gemma-4 m16 mlp_down/o_proj forks), and one measured row / one golden prefix
can match several offered candidates. Every tier therefore resolves its ties through `knob.canonical_row_key` (the
sorted tuning-knob rendering): the model argmin (`Prior.pick` and the greedy fallback), the measured-evidence
argmin, and the golden realization pick. An order-broken tie is a per-boot coin flip — leaf order
can shift across processes — and shipped the 2026-07 RTX 5090 gemma-4 image with a bimodal boot-time cubin set
across boots.
Rendered bytes are pinned across fresh interpreters by `test_source_determinism.py`.

**The kernel set is decided first, then each kernel's schedule.** A kernel-set fork — a cut, a split, a layout —
is decided from what its arms are: a measured arm, else the placement prior over the kernels each arm leaves, else
the first arm (Part 3). No arm is scheduled to price it, so the two kinds of decision never meet: the pieces an arm
mints are brand-new kernels, and each gets its schedule at its own schedule fork, from its own pool, like any other
kernel. A pipeline that ends between `tile/cut` and `tile/schedule` (`compile --passes dolfstp`) therefore decides
its cuts exactly as a full compile does, and schedules nothing. **No arm is withheld to keep a kernel set
unchanged.** A retired cut withdraws ONE splice — the blocklisted decision identity at that node — and the fork is
decided again over what remains.

**Evidence joins on the kernel's exact identity and the context, and nothing else.** A measured row describes a
candidate when it was measured on the candidate's kernel: the index (`greedy._Measured`) is keyed by exact identity
for the evidence pick (tune DB rows and golden rows alike) and for the disqualification tier, and a fork asks for
the rows of its own kernel (`wire.kernel_identity`). No feature takes part in the join, so a featurizer change
cannot disable a measured read, distinct bodies with equal histograms never exchange schedules, and a piece a cut
mints never reads its parent's rows. The index spans the deploy's own context key — one regime, one key, one lane —
and the pick is the plain argmin over the matching measured rows.

**Retries are decide-wrappers over a deterministic re-resolve** — every other choice replays identically (cheap
non-chronological backtracking, no snapshots). A fragment kernel's refused row blocklists at that piece's own schedule
fork, so the composed route replays while the piece re-ranks, across as many retries as the piece has rows. Only once
no row of it binds is a structural pick retired, and only one: the cut that minted the piece (the trace's `Decision`
records the ids a splice minted), blocklisted by its decision identity at its own fork, where the decide withdraws
that splice and decides the fork again over the remaining arms with the same evidence — so a disqualified fused side
keeps losing, and the fused root returns only when every cut above the piece has been retired in turn. The
retirement is logged at WARNING with the rejection reason.

**Greedy validity fallback.** The whole greedy retry orchestration is search policy, owned by
`strategy/greedy.GreedyStrategy` — `Pipeline.run` is a thin entry point delegating to it. The prior ranks by
predicted latency, which can rank a tile that fails `validate(ctx)` (smem / thread budget) first, and greedy
benches nothing, so it cannot find out by trying. So when a deterministic compile leaves a node un-lowered, the
strategy blocklists the `tile_identity` of the pick the resolve made at that node — read off the trace, never off
the terminal node's own knob row, which a kernel-stage pass can stamp with a policy knob (`LOOPIFY`) no schedule
leaf spells — and re-resolves: `greedy_decide(blocked=…)` drops the matching leaf and picks the next-best. This is
bounded by `_MAX_GREEDY_RETRIES`.
When the retry budget exhausts with the node still un-lowered (a prior can rank many over-budget tiles above the
first in-budget one), the strategy takes one last **emission-order resolve**
(`greedy_decide(blocked=…, prior=None)`): its point is that it ignores the prior whose extrapolation caused the
overflow, and the blocklist rides along so this last resolve can never re-pick a tile that already
failed `validate(ctx)`; the measured arms still decide the kernel-set forks they spell. It is a validity fallback,
not a quality one — it makes no claim about the speed of what it lands on, and the enumeration promises it no
particular leaf. When that leaf leaves the node un-lowered too, `_raise_on_unlowered` fires the loud `LoweringError`.

What counts as un-lowered depends on how far the pipeline runs (`Pipeline.lowers_to_cuda`). A pipeline that reaches
the final lowering pass promises a graph of `CudaOp`, so **every** surviving `TileOp` / `LoopOp` is stranded, whether
or not a rule recorded a rejection for it — a materializer that declines a row with a plain `RuleSkipped`, or a rule
that never matched, strands the node while recording nothing, and used to escape all three fallbacks above and leave
a half-lowered graph the compile reported as a success. A truncated pipeline terminates in an earlier dialect by
design, so there only a node with a recorded rejection counts.

## Part 5: Recording evidence

Nothing in this package benchmarks on its own initiative. A measurement is taken by `run --bench`, and it reaches
the next compile as a `perf` row of the tune DB. Every writer records in the deployable regime — the flags `compile` /
`run` / `serve` use — so a recorded latency is the deployed latency and no store carries a per-regime lane (Part 3).

- **`run --bench`** records every clean pinned row (a golden row, an `--ab` row) and the greedy pick's own isolated
  re-bench, one `kernel` + `perf` row per benched kernel (`bench_record.persist_kernel_perf`), so a replayed golden
  and a hand pin are indistinguishable to the evidence pick. A failed bench records the `bench_fail` row of the
  kernel the failure names (`bench_record.persist_bench_failure`) and nothing for the innocent kernels; a
  compile-budget overrun records nothing (Part 6 has the blame rule). `--no-record-evidence` turns the recording
  off, and a row that failed one of Part 7's integrity gates is never recorded.
- **`run --golden PATH --bench --record`** writes the measurement into the working golden as well, onto the row it
  measured by exact name, pins and knobs. **`--record-greedy`** writes the kernel set the greedy pick took — the
  kernels, one routing row per kernel-set decision, one measured row per kernel (Part 7). Either row is evidence
  wherever the file is in scope.
- **The golden import** (`golden.evidence.evidence_db`) files the golden rows in scope into the tune DB, once per
  golden digest, before a compile picks (Part 3). That is how a row recorded on one card reaches every compile on that
  card.

All bench timings are **CUDA-graph-captured** by default (pure GPU time); each `perf` row records its mode in the
`captured` column, and on write a captured measurement supersedes a wall-semantics one for the same key (never the
reverse), so old rows upgrade in place.

**One measurement regime.** The single-regime rule was measured, not assumed. An earlier sweep ranked candidates at
`-Xcicc -O1` and re-benched near-best configs at `-O3`, which put a proxy in charge of the search. The proxy's error
was *biased along tile area* — the axis being tuned — so it systematically priced wide tiles as slow (paired over
1,818 configs: p90 regret 1.68×, and the `-O1` argmin was the `-O3` argmin on only 44.5% of pools), and the compile
time it was buying no longer existed: over 4,888 nvcc compiles, `-O3` compiled at a median 0.96× of `-O1`.

Searching a kernel's schedule space on the card is future work: a ranked sweep that scores the kernel's complete rows
with the offline prior, benches the top few, searches locally from the best measured one, and writes what it measures
as rows of this same DB.

## Part 6: Persistence and keys

This Part is about what survives a process: which identity a row is keyed by, which table it lands in, and how a live
store becomes a reproducible snapshot. Read the keying map before adding any cache or column.

### The keying map: two identities

Everything the `search` package stores or replays is keyed by one of TWO identities — when adding a cache or table,
pick one; don't invent a third:

- **Variant identity = `(context, kernel, knobs)`** — anything *predictive or replayable*. A prior is a pure function
  of it: `Featurizer.features(kernel, knobs)` joins the context's `H_*` features, the kernel's `S_*` stamps
  (`features.stamps`: a stmt/op histogram, loop extents and operand dtypes) and the knob row (tuning knobs encode by
  type, `MMA` expands to atom props). The knob row holds decisions only; every fact about the kernel is computed
  from the kernel.
- **Measurement identity = the kernel's exact identity, the sizes it was benched at and its knobs**, under the card
  and the regime — ground truth about *materialized kernels*: `perf` rows (the per-variant replay cache) and the
  kernels a routing row names. The exact identity (`identity_key(structural=False, with_io=True)`)
  keys the `kernel` table too: the clustered deploy identity merges kernels that differ only in their pointwise op,
  so it cannot key a definition.

### Kernel facts are computed, never stored

A source of truth holds inputs only, and anything computed from them is computed where it is read. The sources of
truth are an op (its body, its io, and the decisions on `op.knobs`), a golden file and a corpus case (the Loop IR,
the decisions, the measurements). What is computed:

- **The kernel of an op** is the tile its schedule fork was offered: the nearest unscheduled `TileOp` on the op's
  `source` chain (`wire.kernel_tile`). A schedule that realizes the kernel through another term (a carried state's
  serial form) does not change which tile that is, so a scheduled tile and the `KernelOp` and `CudaOp` lowered from
  it name the same kernel.
- **Its exact identity** is that tile's (`wire.kernel_identity`) — the one key measured evidence joins on: a `perf`
  row, a routing row, a fork's offer and a golden's stored kernel name one kernel alike.
- **Its `S_*` stamps** are `features.stamps(op)`: `structure_features` of the body the identity digests — the tile's
  derived `loop_body`, or a loop op's own body where no tile stands behind it — with the operand dtypes read off the
  kernel's own io, memoized on the kernel.
- **Warp eligibility** is the classic scheduling problem's answer on that kernel (`Featurizer.warp_eligible`), asked
  when a schedule row is featurized and emitted there as the column `S_warp_eligible`. A placement arm gets none.

A cache may hold a computed value under a version, and is re-created when stale: the tune DB's `kernel` key (below),
a dataset (its format version and `feat_ver`), the weights (`feat_ver`), a memo on an immutable kernel. Nothing
writes a computed value into a source of truth: `tests/architecture/test_layering.py` pins the golden field lists,
and `tests/compiler/pipeline/test_strategies.py` checks that every knob on every op of a lowered program is a
registered decision knob.

### Search persistence: the tables on disk
 **`SearchDB`** (`db.py`) is a SQLite store — one schema in several instances. The tune DB (`EMMY_TUNE_DB`) is what
a compile reads and `run --bench` writes — a compile creates it on first use and imports the golden rows in scope
into it before
it picks (`golden/evidence.py`); a dataset DB (the file `emmy db … --db PATH` names) holds the same tables filled by
`emmy db import`, and is what the measurement-data readers read, so an import there can never change a deploy. The
tables hold compilable kernels, the decisions that minted them, and measurements of them — nothing else.

- **`kernel`** — one row per kernel, keyed by its exact identity: its Loop IR wire (`wire.kernel_wire` — the one-node
  program of the loop body the kernel was formed from, bound to its own buffers), its C name and `formed`. The key
  is the one computed value the DB holds: a cache key, computed from the wire by whoever writes the row (a bench off
  its live kernel, an import off a golden's stored one) and valid under the DB's version (below). Nothing else
  computed from a kernel is stored — its `S_*` stamps are computed from the wire where a row is featurized. The
  lowering passes take a formed kernel's wire back to the kernel — the lift and the twist give the same tile, so the
  same exact identity and stamps are computed from it (`tests/compiler/ir/test_kernel_wire.py` holds every kind of
  kernel the corpus mints to it) — which is what a freeze re-lowers. A piece carved from a twisted tree (an attention
  cut or split piece) is formed from no loop op: the lift does not take its derived body back, so its row keeps that
  body, `formed` false — a loop op over it has the kernel's identity and stamps as it stands — and only its parent's
  program reaches it. The fused kernel of a slice and a piece a cut or a split minted are rows alike, so the same
  kernel reached from two parents has one definition — what a candidate pool enumerates from. A kernel wire enters
  the LOWERING passes, never the Loop passes, which normalize a size-one axis away and mint another kernel.
- **`context`** — one row per backend, card and regime: the card (`Context.hardware_id`, the PCIe product name — two
  SKUs off one die, H100 and H200, RTX 5090 and RTX PRO 6000, share a compute capability, and without it their rows
  would meet under the keep-best upsert), the target as the backend spells it (`sm_120`), the cicc opt level and the
  residual compiler flags (`""` in the plain regime, so `""` and `-Xcicc -O3` are one regime).
- **`schedule`** / **`schedule_knob`** — one row per distinct schedule row, the in-kernel choices a leaf kernel was
  measured with, keyed by the digest of its knobs as strings (a knob's value is its spelling). Never a placement
  knob: a `PLACE` key or a cross-CTA `REDUCE` half is refused as a measurement.
- **`placement`** / **`placement_knob`** — one row per distinct kernel-set decision a cut arm spells, with one key
  per seam actually cut (the fork's other spellings of a seam resolved through the splice event's aliases).
- **`routing`** — one row per PIECE of one decision on one parent, in the fragment's order: the parent's exact
  identity, the placement, the position and the piece's exact identity, resolved after the splice, when a piece's
  buffers are bound and its identity is the one the assembled route runs (`inventory.record_routing`, from the
  `KernelInventory` splice watcher a record run composes in). A piece with forks of its own is the parent of
  further rows. A decision has no
  measurement of its own: its price on a context is the sum of its pieces' fastest rows there, all-or-nothing
  (`SearchDB.priced_arms`), which is what the deploy pick reads.
- **`perf`** — one measurement per compilable kernel variant per context, keyed `(context, kernel, bindings,
  schedule)`: the sizes a dynamic kernel's symbolic dims were benched at (`bindings`, `{}` for a static kernel — the
  identity ignores the hint, so without them one kernel at two sizes would be one row), then the stats, `status`,
  `captured`, a `bench_fail` row's `error` and `source` (`measured`; `golden:<digest>` or `freeze:<digest>` for an
  imported row — the digest of the golden scope or the freeze it came from, which is how a re-recorded golden's rows
  are told from the current file's and let go).
  Failed rows ARE recorded — the next compile disqualifies that arm instead of failing the same way again — and an
  `ok` row is never downgraded by
  a later failure; a config whose **compile** ran past its budget is not recorded at all, since a stored row would
  make it a permanent cache hit that is never re-benched (see the two bench budgets in
  `backend/cuda/ARCHITECTURE.md`). No route rows, no whole-slice totals, no kernel-set verdicts.

Readers see a FLAT `PerfRow` — the context's columns, the kernel named by its exact identity (`PerfRow.kernel`),
the sizes, and `knobs`, the schedule row alone. A reader that needs the kernel's features lifts its stored definition
(`freeze.kernel_ops`: `KernelDef.op` at the row's sizes) and computes them; the joins live in `db.py` and nowhere
else.

**Nothing migrates.** A file whose tables have other columns than the DDL was written by another emmy: a writer open
re-creates EVERY table empty (dropping one would orphan the rows that reference it) — the rows are regenerable
(re-bench, or `emmy db import --fresh`) — and a read-only open refuses the file. Foreign keys are enforced on
every connection. The same holds for the `kernel` key: `PRAGMA user_version` (`db._VERSION`) names the wire and the
identity computation the rows were written under, and a file under another version is re-created the same way. Bump
it when either changes — `tests/compiler/pipeline/search/db/test_db.py` pins a few identities and goes red when the
computation moves.

**Drift checks** (`emmy db check`, `SearchDB.drift`). A tune DB is a cache: a row the current code disagrees with
is re-benched or re-imported, never patched, so the checks are the cheap ones over the tables themselves — a schedule or
placement digest matches its knob rows, every row names the rows it references, every context names a registry card,
schedule knobs and placement knobs stay apart. Nothing decodes a stored wire here. The artifact that has to survive
a code change is the freeze, which is re-lowered on import.

**Measurement freeze** (`db/freeze.py`, written by `emmy db freeze`). The tune DB is a live store, so a model
fit or evaluated straight from it is not reproducible. A *freeze* is a snapshot written as a golden file per card
(Part 7's format), the DB's tables with each kernel named by a ref of the file: every measured kernel's definition
with the routing rows that reach it and the parents they reach it from, and a row per measurement — its schedule
row, its regime as the input pin `FAST_MATH`, its median. A freeze has no traced programs, so it is evidence and
training data but nothing a restamp can lower again; a compiler change that re-keys a kernel leaves its frozen rows
behind.

- **What freezes** is `freeze_reason(row, shape)`'s call, `shape` being the stamps of the row's kernel, computed
  from its stored definition (`freeze.kernel_ops`): an `ok` row on a card the GPU registry knows, at the deployable
  opt level, whose kernel still lowers (one that does not is dropped as a stale kernel), that passes the two
  physical-plausibility checks (`implausible_value_reason`, which reads the row's
  `bindings` as the size a symbolic axis ran at, and `impossible_kernel_reason`). The fast-math flag alone decides
  which of the two precision regimes a row is in (`regime_of`), and no other compiler flag is stored or gated on. A
  failed bench is not a measurement (the tune DB keeps it), and a row a compile imported from a golden file is the
  file's. The opt-level gate is what keeps a freeze a fair yardstick: a freeze is the corpus a reported prior number
  is computed over, so rows from a regime nothing deploys in would put half a card's pools in a lane no one runs.
  `measured_groups` inherits the same filter, and keys its pools by regime, so an analysis over a live DB agrees with
  one over a freeze.
- **Freezing the same DB twice yields the same bytes**: rows sort by content, and the golden dump is deterministic.
  A file's identity is its bytes: `emmy db import` sources its rows by the file's kind and digest —
  `freeze:<sha256[:12]>` for a freeze directory's files, `golden:<sha256[:12]>` for a golden file
  (`golden.evidence.file_source`) — so a report over an instance names the exact files it was computed over, and
  naming a file again is a no-op. A held file is a fact of its own (the `source` table, written by the import whatever
  became of the file's rows), so a golden none of whose rows is a measurement — a restamped one keeps its schedules
  and loses its microseconds — is held and simply contributes no row.
- **Importing goes row for row, and computes the key.** `emmy db import` reads freeze directories, golden files and
  tune DBs (frozen first, so one path serves all) named on its command line, or every repository golden
  (`--repository`: the hardware goldens and each maintained recipe's, the priors' training set — README, "Fit the priors");
  nothing by default — and hands each file's tables to the golden importer (`golden.evidence.import_file`) once per
  regime the file holds: `record_kernel`, `record_routing`, `record_perf`, each under the exact identity computed
  from the stored kernel's Loop IR. A stored kernel whose body no longer lowers is skipped with a warning, with its
  rows; a stale file is a restamp, not an import option. A golden file's rows become, through
  `emmy db export`, the golden pools `emmy fit` and `eval prior` read (Part 8); a compile never reads this instance,
  and the tune DB imports the goldens on its own.
- No freeze is checked in at the moment. The RTX 5090 freeze predates the `kernel` table and was dropped rather than
  converted; the card is re-collected through the `perf` writer, after which `emmy db freeze` writes the next
  one into `search/freezes/`.

**Recording benches** is Part 5's subject: a `run --bench` that benched rows with hand-forced knob values (golden or
`--ab` rows) records each clean measurement — plus the greedy pick, through its comparable `greedy (isolated)`
re-bench — per kernel through `bench_record.persist_kernel_perf`, so a replayed golden and a hand-pinned row are
indistinguishable to the evidence pick. Rows that were flagged (a pin that did not match, a wrong answer, an
implausible arithmetic intensity) and anything from the `--ir` path are never recorded.

**A failure blames one kernel** (`bench_record.persist_bench_failure`, the one writer for a failed bench, which `run
--bench`'s greedy row comes through). A kernel whose row is a `bench_fail` fails every graph it is in — its identity
is its rendered source and launch geometry, the same bytes wherever it appears — so a failure is recorded only
against the kernel the failure names — the watchdog's hang, or nvcc's refusal of a kernel's source — and the innocent
kernels stay rowless; the name is read off the message text, since the exception class does not cross the worker
pipe, and with the quote `repr` escapes when the message also holds a `"`. A failure that names no kernel in a
multi-kernel graph — a wall kill — blames none of them and records nothing: the DB holds measurements of kernels and
nothing else.

## Part 7: Golden files and the A/B integrity gates

Hardware goldens under `search/golden/records/` hold standalone operations and programs derived from models, one
file per exact GPU. Extend that file with missing cases, storing each kernel once and preserving existing
measurements. Retain the selected routing graph and measured descendants, with parent decisions before children.
Experiment records stay in place; only accepted routes join the hardware corpus.

A golden file is a card's measurements in the tune DB's shape. It serves four purposes: measured evidence for the
greedy compile (Part 3), pinned measurement (`run --golden PATH --realization NAME --bench`, `--ab`), training data for
the offline prior, and a regression reference. This Part covers the file, what a record run writes into it, the
restamp that keeps it current, and the checks that keep the A/B honest.

**The file is the DB's shape.** `golden/format.py` declares it as `GoldenFile`: the card and model; `programs`, the
traced Torch IR programs the targets were lowered from (provenance: the Torch twin a bench compares against, and what
a record run re-compiles); `kernels`, one `Kernel` per kernel — its definition (`dataset.KernelDef`: the standalone
Loop IR body, the C name and `formed`) and an optional `key`, plus, for a target, the traced program it came from
(`traced`), the ops it computes whole (`origins`) and the sizes that specialized it (`bindings`); `routing`, the
`RoutingRow` of every kernel-set decision — the parent, the arm, the pieces it minted; and `rows`, one `Row` per
`perf` row — the kernel, the sizes its symbolic dims were benched at, the input regime (`pins`), the
schedule row (`knobs`), the measurement, and a `name` a command selects it by. A piece a decision minted has no
program of its own: a routing row reaches it from its parent (`GoldenFile.path_to`). A row with no measurement is a
proposal, not evidence; a row with no schedule is a target that has only been traced. Every object writes its own
wire (`emmy/compiler/wire.py`: one mixin, one walker, a field at its default omitted), `GoldenFile.check` holds the
cross-object rules (every reference resolves, a `ref` names one kernel, a row spells a schedule and never a kernel-set
decision, a repository golden names its card and measures with a reference), `load` / `dump` read and write a file
through it, and `edit` is the one locked read-modify-write every record run goes through. The dump writes a header key
per line with `gpu_name` alone on the first (a card-scoped reader skips a foreign file off that line) and an entry per
line in each table, so a diff lands on the entry that changed.

**The file stores inputs only.** A stored kernel has no identity and no stamps beside its Loop IR. Rows and routing
rows name a kernel by its `ref` in the file (`Kernel.ref`): its `key`, or its C name where no key is set — a second
kernel sharing a C name takes a key like `k_name#2`. The exact identity is computed from the wire when something
asks (`KernelDef.exact_identity`, through `KernelDef.op`: a formed kernel's body goes through `tile/lift`; a kernel
formed from no loop op stores its derived body, and a loop op over it has the kernel's identity as it stands), and
`GoldenFile.identities` gives every kernel's by `ref` — which only an import, a restamp or a record asks. A stored
kernel whose body the compiler no longer takes back has no identity: a stale kernel, whose rows are no evidence.
`tests/architecture/test_layering.py` pins the field lists, so a new field is an input or it does not go in the file.

**Repository goldens are the entire compatibility boundary.** The file carries no format version. When its layout or
the IR's wire changes, regenerate every repository golden in the same change; the checked-in corpus staying loadable is
the compatibility gate. `emmy trace` writes a working golden: every kernel a program lowers to at the card's context,
and one unmeasured row per kernel and template (`golden.working.inventory`); a template with sizes specializes the
program first, since a kernel at a size is a kernel of its own.

**A record run writes what the DB holds.** `run --golden PATH --realization NAME --bench --record-greedy`
(`golden.record_greedy_pick`) writes the kernel set the greedy compile picked: the kernels it minted, one routing row
per kernel-set decision the splice watcher reported (`search/inventory.py`), and one measured row per CUDA kernel —
the tile kernel it lowered from, its realized schedule and later storage choices, its own isolated launch timing —
under the seed row's input regime with the compile's own precision gates laid over it (`pins.measured_precision_pins`)
and the greedy comparison row as `same-input-greedy` reference. A row of the same kernel, sizes, regime and schedule
takes the new timings. Recorded this way, a strict-evidence compile picks the same kernel set again from the file's
rows alone (no tune DB, no prior). A routing row the unpinned cut pass does not take again is refused before anything
is written, since the next restamp would drop it: a composed cut closes its pieces, so a cut pinned on one of them
(`PLACE@place_<token>/…`) is recorded by pinning that seam on the parent instead. Each measured row carries `tried`:
the schedules of its kernel the tune DB held at its sizes and regime on the card (`bench_record.measured_schedules`) —
the search behind the row, which nothing else in the file can tell, and which goes with the measurement when a restamp
demotes the row. `--record` writes a row's per-card latencies (`golden.record_latency`), the corpus's ratchet. A record
run always times `torch.compile`, and `--record-greedy` writes the same block onto the seed row: the whole pick's time
beside `torch.compile` and eager, the number `emmy golden list` reads a gap against `torch.compile` from. A row's `note`
is free text a person writes about it, never a label computed from the numbers. Both writers refuse a canonical path:
a re-record works on a copy.

**One builder writes every kernel entry.** `golden.definition` builds every entry — a trace inventory's, a record
run's, a restamp's — from the tile kernel through `bench_record.kernel_row` and spells the body under the entry's own
name, so a kernel recorded off a compiled program and the same kernel re-derived from its program are one entry.

**The restamp keeps a file current, and a current file is one the restamp leaves unchanged.** `golden.restamp`
lowers every traced program afresh (`lift_targets`: the loop passes and the lift, at the kernel's sizes), matches each
target kernel to a fresh one by the buffers it writes, and takes every kernel-set decision again on the fresh parent
(`mint`: the parent's body through the lift and the cut pass, each fork on a kernel the stored path decides taking
the arm its routing row spells through the same `pins.spelled_arm` the deploy reads a row with, the pieces read off the
splice watcher). What it keeps is decided per entry, never guessed: a kernel that kept its identity keeps its entry —
body and name (one identity can be minted by several parents, each spelling the body's buffers its own way); one the
fresh lowering re-keys — the stored body and the fresh one are two kernels, their exact identities, both computed
now, differ — takes the fresh body, spelled under the stored name and known by the stored `ref`; a target no fresh
kernel writes is dropped with its decisions and rows; a decision the fresh parent takes with another arm, or that
mints another number of pieces, is dropped with its pieces' rows; a row whose kernel was re-keyed keeps its schedule
and loses its measurement — a proposal, no evidence until a record run on the card measures it again. The file holds
no identity or stamp to take, so a change to how identity is computed re-keys nothing and costs no measurement.
`emmy golden check` reports what a restamp would change, `emmy golden restamp` writes it, the suite holds every
repository golden to "nothing" per traced program (`tests/compiler/pipeline/search/test_golden.py`), and the
realization corpus's staleness test is the same restamp (`tests/compiler/realization/ARCHITECTURE.md`). Both are
GPU-free; the `refresh-golden` skill is the flow around them, including the record run a demoted row needs.

When a row is dropped, restamp rechecks the surviving rows against their remaining siblings until no further row is
lost. A kernel set missing a member loses its verified state even if it has a stored measurement. Kept and demoted
counts include only surviving rows; an empty result is refused instead of replacing the file with incomplete evidence.

The preferred reference is the runnable Torch slice (`torch-eager`) or the applicable library kernel (`cublas`). A
target's slice is its stored origins cut from the traced program (`GoldenFile.reference_program`), taken when the
kernel writes only values those ops compute and reads no activation the slice does not; nothing is lowered to find
it. The program a compile or a bench of the target starts from is its stored body with every input the slice's
lowering produces from a constant bound the same way (`GoldenFile.executable`): a weight the kernel reads is the
checkpoint's beside the twin's, through the same transpose, never a random input. A kernel with no stored origins has
no frontend callable; such a target may use a separately
compiled, repeated O3 `same-input-greedy` row as its reference only when the candidate and reference execute on
identical deterministic inputs, their outputs pass the normal accuracy policy, and the model report discloses that
this checks compiler-configuration parity rather than independent framework correctness.

**A matmul golden's layout is part of what it measures.** The embedded Torch IR spells the serving Linear layout — B
given `(N, K)`, contracted as `x @ w.T`; the traced contraction carries `b_trans`. The warp tier stages it like any
canonical matmul, so the same STAGE spellings realize on both layouts — but the measured µs still differ per layout
(different slab geometry and gmem walk), which is why a record meant for a served model's linear fork must be MEASURED
on the `F.linear` snippet, and why a canonical entry (the harness/eval truth) and a `trans_b` entry (the serving truth)
both stay current.

**Provenance validation.** `emmy eval golden --golden GOLDEN_FILE --serving-config PATH` derives model, revision,
GPU, canonical file, precision regimes, and reachable static/symbolic widths from one pinned env, requires that exact
file and live GPU, validates that every twin's target kernels carry a row at every size and regime the config reaches
the twin at, and compiles the serving twins with the file's rows as the only evidence under strict evidence (Part 3): a
twin with a fork no row decides fails the gate naming the kernel. A recorded row's health beyond that is its pinned
measurement (`--ab` / `run --golden PATH --realization NAME --bench`, gated by the A/B integrity checks below).

**Live-GPU scoping.** `run` / `compile --realization NAME` (no `--golden PATH`) search the **live** card's repository
goldens (`repository_documents`) — names repeat across per-GPU golden files with diverging shapes/dtypes, so a flat
union can select another card's spelling — and every file on an uncovered card or off-GPU. `--golden PATH` instead
names an explicit file, whose GPU header is checked against the live device.

**The A/B carries three integrity gates:**

1. **Realized-vs-pinned knob check — a miss FAILS the row before it benches.** A structurally invalid pin silently
   falls back to the planner's own pick, so benching it would compare greedy against itself and report a fake 1.00×
   under the pin's name. The check runs right after the
   pinned compile. A pin that matches none of the knob values the compile actually produced marks the row
   `pin_unmatched` / `unreproducible pin … NOT benched` (a loud error log; the row is kept in the table and in
   `--json`, and no GPU time is spent), and the remaining rows still run. Matching is aware of knob families — a
   golden key must equal the exact site the compile produced — while values are compared through the registered
   knob's canonical `Knob.parse`, so alternative spellings of the same value, like `FAST_EXP=1`, do not raise a false
   alarm. A pin satisfied by ANY kernel counts as honored, which is what makes split main+finalize pairs
   work, but it does mean that a pin dropped on its intended kernel goes undetected if a sibling kernel happens to
   match it. `PLACE` is consumed before CUDA emission, so the final greedy resolution's placement receipts ride the
   compiled graph as attribution and supply its realized side. Bare `PLACE=fuse` accepts an empty placement trace;
   a site-scoped pin still requires its site. The `g<n>` cross-CTA stage of a `REDUCE` value is
   structural and cannot be read off a knob stamp, so the check skips it. A split replaces the kernel it splits, and
   `knob.consume_kernel_row` strips the schedule row from the pieces it mints — no piece may carry the `g<n>` it came
   from — so the receipt is the piece's sliced reduce axis, not a stamp. Only that stage is exempt: the rest of the
   value (`coop` / `r<n>`) is decided by the piece on its own body and stays gated. The cost of the exemption is that a
   `g<n>` pin which genuinely never split cannot be told apart from one that did.
2. **Arithmetic-intensity check.** A row whose FLOP/s, implied by its shape, exceeds the peak recorded in the live
   GPU's `GpuSpec` is flagged as a bad measurement rather than a fast kernel.
3. **Wrong-answer check.** Each pinned config is executed once on the greedy run's inputs and its outputs are
   compared, which catches kernels that are silently wrong (a skipped finalize produces plausible-looking garbage).

**Every `run --bench` row is measured in a bench worker process that can be SIGKILLed — the parent never launches a
kernel.** The greedy comparison (eager / torch.compile / emmy, with the torch side rebuilt inside the child) and
every pinned golden / `--ab` row run as jobs on ONE persistent worker per run
session. That makes the A/B survive any failed row by construction: a hung kernel dies with the SIGKILLed child, the
parent's CUDA context stays clean, the row is reported `bench_fail` with its reason, and the next row's job starts a
fresh child — no escalation modes, no `os._exit`.

- NOTE: the *process* is the same for every row, but the measurement *environment* is not. The greedy row is benched
  interleaved with the live torch closures, so torch's allocator state and cuBLAS's L2 carve-outs are resident, while
  a pinned row is benched emmy-only in a job that never touches torch. A greedy-row µs and a pinned-row µs for the
  same config are therefore NOT directly comparable (the gap observed in practice is ~7% on split-K pairs).
- One number cannot be both comparable to torch and comparable to the pinned rows. So whenever pinned rows are
  benched, the greedy graph is ALSO re-benched emmy-only through the same pinned path (one extra worker job, no
  recompile). That produces the `greedy (isolated)` row printed beneath each greedy kernel in the table, and the
  `greedy.isolated` block in `--json`. Those are the baseline the pinned rows' speedups are measured against.
  **Record goldens from `--ab` / golden rows only, never from the greedy row's number.**
- The greedy pick hanging, or blowing the bench budget, is itself a *finding* — precisely the hazard a golden exists
  to prevent — so the pinned rows are still benched afterwards. Pinned rows that fail to compile or to bench are kept
  as `bench_fail` rows, never dropped, and the run exits non-zero if any row failed.
- The greedy job also carries the accuracy check: the emmy program runs on the rebuilt module's real inputs in-child,
  and a numeric failure aborts the run, because a latency table for a miscompiling program is meaningless. That run's
  `(inputs, outputs)` become the pinned rows' wrong-answer reference.
- Only the no-`--bench` accuracy probe still runs in-process (it hosts the `--debug` per-launch dumps and the ncu
  child's profiled launches), so with `--bench` those two want a separate plain `run`.

Plus `--json PATH` — a machine-readable record of the whole comparison (backends / greedy kernels / pinned rows with
their flags and a `status` field: `ok` / `pin_unmatched` / `bench_fail` / `compile_timeout` (the config's compile ran
past its budget, so nothing about it was measured and the row is reported but never recorded); a failed greedy block carries
`status: bench_fail` and an `error`, with null timings), so a sweep's judgments can be traced to flagged fields
instead of to parsed terminal text. Each kernel row also carries **`record_knobs`**: the tuning knobs the compile
actually produced, validated as one complete exact classic row by `knob.complete_kernel_row`. That is the map to
copy verbatim into a golden file `knobs:` entry; no recording helper fills absent choices or drops scopes. Golden
rows attach to the run's SHAPE rather than to a kernel node, so a pinned row
whose shape matches no greedy kernel — because greedy deployed a split partial+finalize pair — still prints and still
lands in the record.

## Part 8: Evaluating the prior and the goldens (`emmy eval`)

`emmy eval prior` is how you find out whether a prior is any good and, when it isn't, where it goes wrong. It runs
over a dataset `emmy db export` wrote — its golden pools or its measured pools — and scores them with the shipped
prior of the dataset's space, or with the artifact `--offline-file` names. Beside the ranks it re-decides every pool
with no measurement in scope (`prior/reproduce.py`: the greedy tile lowering against the closest golden row; the cut
pass with the placement prior deciding against the golden's arm) — the deploy-faithful check the reproduction gate in
`make test` asserts a rate on, per slice of a repository golden's pools (README, "Fit the priors").

**Two datasets, two questions, one report.** `search/prior/report.py` assembles both into one serialisable schema
(`--json`). With `--compare-to`, `eval prior` scores two weights on the same golden pools and compares their median
ranks group by group; changed coverage or a higher median in any group prevents a candidate from qualifying. This mode
skips the separate reproduction walk. Without it, `--json` also carries one entry per pool of that walk: the prior's
pick, the closest golden row and, when a golden row of the pool is exactly the pick, its time over the pool's best as
`regret` (`unmeasured` in the table otherwise — the cost of a pick nobody measured is unknown until that schedule is
recorded as an ordinary row). The nightly refresh posts those counts per space.
`emmy fit` writes the same summaries into its `metrics.json`, through the same
`report.rank_metrics`. The report does not define the metrics:
`search/metrics.py` owns every metric's definition, and `Prior.score_rows(group)` — the pool-shaped scoring surface,
projecting the packed matrix onto the model's own columns with its own absent-value fill — is where a score comes
from.
 - A MEASURED pool (`--pools measured`: the dataset's measured pools — the DB's `perf` rows, every candidate benched,
grouped by `(gpu, kernel signature, H_opt)`) can answer what a wrong pick COST — Spearman over the pool, and regret at
k=1 (the deploy question: the pick ships, so its latency IS the cost) and k=10 (the tuning question: bench the top
ten, keep the measured best). This is the half that tracks deployed speed.
- A GOLDEN pool (`--pools golden`: an enumeration with the verified-optimum row marked) can only answer WHERE the
  known-good row landed, and is reported as a SCREEN. A rank is blind to the latency gap behind it, and the corpus
  aggregate is dominated by pools small enough to rank by accident, so golden summaries are stratified by pool size.

**Every summary publishes what it covered.** Summaries carry the axes they were keyed on as a dict — measured: `gpu` ×
`H_opt`; golden: `gpu` × `tier` × pool-size bucket — along with how many pools keyed into them, how many the model
could not score at all, and — where a metric has a size minimum and so covers fewer pools than the summary holds —
that metric's own count. The minimums differ (regret needs two rows, Spearman five, regret@10 eleven), so on the RTX
5090 freeze's 176 pools those counts are 149, 107 and 45. An aggregate that averaged the excluded pools in would be
reporting mostly arithmetic.

**A measured pool is keyed on the KERNEL, not on the site that offered it.** The key digests the `S_*` stamps of
the kernel the row measured (`export.kernel_sig`, over `features.stamps` of the kernel's stored definition). Two
kernels of one feature structure on one card share a training pool whatever produced them. Deploy evidence is
stricter: it joins on the exact identity, so distinct bodies with equal histograms never exchange measured schedules.
The stamps are a function of the kernel alone — the tile its schedule fork was offered — so nothing a schedule fork
decides can move an `S_*` value, and sibling schedules cannot be split apart.

Keying on the site that offered the kernel gets it wrong in both directions, and the RTX 5090 freeze shows both. It
**over-merges**, because that key digests the *pre-descent offer op*: nine pools paired a fused `rms_norm`→linear
megakernel with a row for just one kernel of the same op's unfused realization — a 5.9 µs norm kernel filed as a
rival of a 131 ms whole-op row, where the unfused pair actually costs 24–191 µs. And it **fragments**: 73 structures
were searched in two separate pools, the losing pool's best landing a median 1.46× behind the winning pool's (p90
3.89×, worst 14×) — the same kernel appeared in two structural contexts. Against the offer-site key the
kernel key gives 336 pools rather than 401, but more rows sitting beside a rival (3778 of 3817 against 3760) and a
median pool of 7 rather than 5; the pool count falls because merging is the point.

**Both kinds of pool come from the dataset** (`Dataset.load`): the positional argument names the directory — the
export of the dataset DB `emmy db import --db PATH` filled. Another instance's data — a tune DB, for one machine's
measurements — reaches the report through `emmy db export --db PATH OUT`, never through the report opening a DB
itself. A dataset is a build product, read back under the format version and the featurizer version it was written
at: the manifest interns each kernel's definition once and carries its exact identity beside it (`identities`), so a
reader names a pool without lifting its kernel again.

**A golden's rank counts ties against it** (via `search/metrics.dual_rank`). The golden's rank counts every candidate
scoring strictly better PLUS every candidate that ties with it and was emitted earlier. A tie is counted as a loss
because greedy's argmin, faced with equal scores, takes whichever came first. Counting only strictly-better candidates
would report rank 0 for every row inside a plateau of equal scores, which once let a saturated prior score "top-1" on
goldens that real cold deploys missed by 12–29×. Both counts come from ONE computation (`search/metrics.dual_rank`):
the pessimistic rank is the one that gates, and the strictly-better **optimistic** rank is reported beside it in `emmy
fit`'s metrics file. The gap between them is the width of the tie plateau at the golden's score, and thus an early
warning that the scores are saturating. **A golden pool is one kernel's schedule space, read from the dataset DB by
the export.** `db/export.golden_pools` groups the instance's `golden:` rows the freeze admits (`freeze_reason`, the
one admission rule) by card, regime, kernel (the exact one the rows were measured on — the pool has to be enumerated
from a definition, which is why it keys on the kernel where the measured pools key on the stamp signature) and sizes,
beside a count of the rows it dropped, as `measured_groups` does; `ranking.build_golden_groups` enumerates each pool
from the kernel's own definition (`KernelDef.program`: the stored body at the rows' sizes, through the tile lowering
alone, under the regime's pins) and finds each golden row in it by `features.tile_signature`. The base features are
the pool's context and the kernel's stamps, computed from that definition — a golden's program is never
read. Two pools that featurize byte-identically fold into one group after packing, so pointwise siblings of one shape
still train as one pool. `emmy db export` runs this ONE builder and writes its groups as the dataset `emmy fit` and
`eval prior` read, so the eval and the fit see the same pools, the same sampling draw and the same rows. The builder
enumerates pools `jobs` at a time, one pool per worker process: a pool's draw is a pure function of its tree and the
seed, and the results are folded in the pools' order, so the groups are the same at any count; the library default is
one process (the suite runs its own workers) and the CLI asks for every core. A pool's context is
`Context.from_target(cap, gpu_name=…, compile_flags=regime)` — the card the rows were measured on with its known SM
count and smem specs, and the regime's flags — never the host's. Building them for the host's context makes golden
ranks machine-dependent, because the occupancy features then describe tiles for a GPU that is not the one the row came
from. A golden that lowers to several kernels is one pool per piece, each holding the rows measured on it.

The export packs the pools over the FULL featurization; a fit projects them onto its feature view, and the model
records the columns it reads. Scoring a pool packed under a narrower view would ask the model about a kernel with
no shape, which is why every pool is packed whole.

**The per-fork view is retired.** Until 2026-08 this part also documented three node-tree diagnostics: fork-sibling
regret (what following the prior's pick at each fork cost, bucketed by knob family), a golden-anchored descent (how
far a golden's path was covered by the explored tree), and per-feature blame / ablation Δ. They are gone, with
`Dataset.from_node_rows` and the `Prior.masking_exact` chain that existed only to caveat the ablation numbers.

Two reasons, and the second is why nothing replaced them in kind. They answered questions about a search tree, and
the store holds none: every `perf` row is a benched kernel variant with no parent, so the fork metrics had nothing to
group by and degraded to leaf-level numbers that `eval prior`'s summaries now compute directly. And the ablation half
rested on hiding one feature at a time, which attributes an effect among correlated features with no unique answer —
hiding any one of a redundant block of geometry features costs the same Δ.

`--offline-file` (env `EMMY_OFFLINE_FILE`) swaps in a candidate weights artifact — comparing two fits is running the
same eval against two files and diffing the reports.

**`emmy eval golden`** is the other evaluation, of a golden file rather than of the prior. `emmy eval golden --golden
GOLDEN_FILE --serving-config PATH` asks whether the file's rows still decide every fork of the serving matrix under
strict evidence, on the card the file names; Part 7's provenance validation describes the gate.

## Part 9: Tile lowering at the pipeline level

`tile/lift/010_lift` converts each maximally fused `LoopOp` to one unmapped `TileOp`. It peels the outer parallel
axes and mechanically lifts every inner reduction as a nested `Fold`; each term orients a bilinear lift A-first at
formation, and `TileOp` construction canonicalizes the complete tree — an identity projection dissolves into its
operand, same-value cones become one shared object. No Tile IR classifier runs. An output loop's per-cell
projection is a zero-axis term evaluated over its sweep axis, while its writes live as `OutputSpec`s at the `TileOp`
boundary.

`020_twisted` rewrites the exp-family composition over that canonical tree. The single `030_cut` pass reaches a
fixpoint over three ordered domains: it offers the maximal tree and every semantically closed stored child-Fold seam
through `PLACE`, then folded and source storage for eligible transposed constants through `LAYOUT`, then the unsplit
tree beside every cross-CTA reduce split the head Fold admits. A selected cut writes the complete child state to
workspaces; a selected split slices the same Fold and folds partial state tuples with its stored combine. Those pieces
and a selected source-layout variant return fresh unmapped TileOps. `040_schedule` then enumerates schedules over each
stored Fold tree only. Independent roots stay fused and combine only schedules with matching
physical output-axis tile widths and unit counts.

The complete structural invariant is documented in
[`ir/tile/ARCHITECTURE.md`](../ir/tile/ARCHITECTURE.md), and pass behavior in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md).

## Tunable knobs

A **`Knob`** (`knob.py`) is the canonical schema for one tuning dimension: name, type (`INT` / `BOOL` / `BINMASK` /
`STR`), candidate `hints` (advisory — the rule still validates structural fit), and a help string. Rules stamp values
into `TileOp.knobs` dicts; the search reads those back as the row a measurement is filed under. Every
knob is declared **in `search/space.py`** — the single home for the whole tunable surface — and imported by the rule
that resolves it (for the schedule codecs, the tile scheduler's row enumerator). Declaring a `Knob` IS
registering it (`Knob.__post_init__`); `knob.registry()` imports `space.py` before answering, so the set is complete
in any process — no module scanning, no manual registration. `knob.py`
also owns the `EMMY_<KNOB>` env namespace (decode per `Knob` type; `config.py` remains the sole owner of
`os.environ`).

### Pinning knobs from the environment

Two equivalent forms:

- **Per-knob:** `EMMY_<NAME>=<value>` (e.g. `EMMY_STAGE=d2/smem-async`). Read by the rule that owns the knob via
  `Knob.narrow`. The env-var key is built by `config.knob_var` and read via `config.knob_raw` / `config.int_env`.
- **Aggregate:** `EMMY_KNOBS="K1=V1,K2=V2,..."` (e.g.
  `EMMY_KNOBS="WORK=w2x2,TILE=mma_m16n8k16_f16_f32/f2x2/k2,STAGE=d2/smem-async"` — the worker widths ride `WORK`, so a
  `TILE` / `REDUCE` pin that embeds its own raises). Parsed once at `knob.py` import via
  `apply_knobs_env()`, which splats each entry into the corresponding `EMMY_<K>` var
  (`config.set_knob(..., overwrite=False)`). An explicit per-knob var wins over the aggregate.

For structural and kernel-lowering knobs, a pin replaces the compiler's choice through `Knob.narrow`; a value
outside the knob's hint tuple can therefore remain authoritative while downstream structural gates still apply.
Classic schedule parameters are deliberately stricter: `WORK`, `TILE`, `REDUCE`, `STAGE`, and `RASTER` are the values
the sites of Algorithm 1 offer, checked with the catalog's own rules, and never add a member. A value the applicable
site cannot take yields no schedule row. The replay paths
(`run --bench --golden` / `--ab`) verify realized-vs-pinned knobs on every pinned row right after the pinned compile
and fail a mismatch (`unreproducible pin … NOT benched`) instead of benching a fallback (see Part 7).

Classic schedule parameters use exact canonical spelling. Invalid widths, unavailable atoms or transports, K-step
mismatches, and over-budget scalar tiles are absent from their static factors, so those values match no row. Persistent
rows still pass through `ClassicScheduleCodec`, which rejects aliases, malformed values, and incompatible complete
schedules. Outside the classic schedule families, each owning `Knob` retains its parser; for example, a `BOOL` knob
rejects an unrecognized value instead of coercing a typo (`ture`) to `False`.

### Registered knobs

All declared in `search/space.py`; see [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the per-rule mechanics.
The "owning rule" for the schedule codecs is the tile scheduler (the `040_schedule` rule), whose recursive row
enumerator spells each family exactly once, site-local, where a row becomes stored state.

**`PLACE`** (STR structural fork, `fuse` or `cut`) — a stored Fold edge's kernel placement, addressed by the
structural tree-path codec before classic sites exist. The maximal fused kernel and every semantically closed cut are
siblings — closure counting the offer-time provider closure and dependent-seam composition `passes/ARCHITECTURE.md`
describes. A cut is
consumed by the graph splice and therefore is not stamped on either fresh kernel; exact routing replay reads it from
the structural decision trace. A scoped cut consumes its authoritative placement decision on both fresh pieces, which
proceed to scheduling. Bare `PLACE=cut` selects the primary seam and consumes placement on both pieces. Only unpinned
cuts leave their pieces able to re-enter placement and expose smaller seams.

**`WORK`** (STR codec) — the kernel-global **worker inventory**, spelled exactly once per row (step 7):
`w<M>x<N>[+p<np>]` (warps — the mma tier; `+p<np>` the dedicated producer band the retired per-row `WSPEC` key
spelled) / `t<N>x<M>` (the scalar thread tile, native n-then-m) / `t<N>` (the 1-D cooperative width). Empty = a
1-thread register strip whose launch geometry stays derived. The tier discriminator IS the worker kind — never a
per-`TILE` spelling. Option assembly derives the inventory from site choices, the complete typed schedule stores it
once, and acceptance fails loudly on cross-site disagreement (one kernel, one inventory).

**`TILE`** (STR codec, the tile schedule) — the **output-fragment** codec, site-local since step 7. A
contraction's output tile is *either* the **scalar** register sub-tile `f<fn>[x<fm>]` *or* the **warp** tensor-core
mma tile `<atom>/f<FM>x<FN>[/k<bk>]` (atom + register sub-tile + K-chunk) — no worker tokens; the worker halves live
in `WORK`. Empty means only per-cell; a parallel unit-register thread tile spells `f1`, so the exact row is
injective without inferring one node's choice from another node's `WORK` claim. The retired embedded-worker
spellings (`n<N>[x<M>]/f…`, `a:<atom>/w<WM>x<WN>/f…/k<bk>`) RAISE —
the worker widths have exactly one home, so a value carrying its own cannot decode into a second, self-contained
reading. There is no alias vocabulary: old `a:scalar` / `a:none` scalar tags, alternative atom names, reordered
tokens, leading-zero widths, and surrounding whitespace all raise rather than naming the same schedule twice.

**`REDUCE`** (STR codec, the tile schedule) — the reduce-axis partition codec, site-local since step 7:
`[g<n>[a|k]][/coop[-t]][/v<n>][/r<n>]` — `g` cross-CTA split-K (+ finalize letter), `coop` the cooperative-thread
fold (its WIDTH lives in `WORK`; `-t` the transposed lane map), `v` the lane's contiguous run (adjacent reduce
elements under `coop`, which one vector load reads; adjacent output columns under `coop-t`), `r` ILP register fold.
Empty = serial (the per-thread remainder is derived, never spelled); the retired `b<n>` coop-width spelling raises.
The cross-CTA split is the `g<n>` field (GRID stage), and the
**finalize** is that field's trailing letter — `g<n>a` = in-place `atomicAdd` (one kernel, additive single-fold
carriers only, with no f16/bf16 destination; both tiers — an mma partial's C fragment rides `RegStore.atomic`),
`g<n>k` = deferred f32 `__partial` workspace + a sibling combine kernel (any carrier; the only legal arm for a
low-precision output, a multi-component twisted carrier, and a multi-channel ⊗-combine). A direct atomic low-precision
destination would round once per partition and can cross the strict correctness boundary; the deferred arm combines
carrier state in f32 and rounds once. Pin
via `EMMY_REDUCE=g2k` (one flat knob — no per-axis `EMMY_REDUCE_<axis>`, no `EMMY_FINALIZE`). The split is realized by
`tile/cut/030_cut` as a graph rewrite whose pieces are **brand-new kernels** — unmapped, knob-free,
each scheduled at its own fork; whether to split is a kernel-set decision taken before
any of them is scheduled, and the split is
CONSUMED by the kernel that realizes it (the sliced axis is a `Window` of its parent, so nothing partitions it
twice). See [`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md) for the invariant. The
letter round-trips through `Reduce.parse`/`spell` and reads back as `Reduce.finalize`. The atomic finalize
applies the kernel's projection epilogue **per partition** before the `atomicAdd`, so it is only correct when that
projection *distributes* over the add (`Σ φ(xₛ) = φ(Σ xₛ)`): a constant scale like `mean`'s `×1/N` distributes and
rides the atomic; a non-distributive epilogue (`l2`'s `sqrt`, a fused bias/activation) is refused
(`ValueError` → pin `g<n>k`, which projects once after the combine). The check is
`ir/stmt/passes.projection_distributes`, applied by `tile/_split.atomic_finalize`.

Two deploy-only dominance/default rules live beside the generic schedule enumeration. A coalesced wide-K `F.linear`
MATVEC (the M=1 contraction-demotion tier) always uses a `b32` single-warp fold unless `REDUCE` is explicitly pinned;
the serial sibling walks all of K in one thread and measured 4–16× slower on DiT conditioning projections. That
dominance rule is generic (any card, any coalesced wide-K matvec), which is why it stays here.

The SKU-exact `facebook/DiT-XL-2-256` deploy overrides that used to sit beside it are GONE — a hardcoded contraction
table, a flash-winner matcher and a `b128` LayerNorm `REDUCE` narrowing, all string-matched on `NVIDIA GeForce RTX
4080`. A recorded winner belongs in the golden corpus as a versioned, exactly-replayable pinned row — never in code.

What could NOT be expressed was deleted rather than mis-filed, for one reason: **no recorded program in the corpus
describes a LayerNorm**. The DiT prologue is AdaLayerNorm-Zero, while every fused recorded program (`norm_linear` /
`mlp_geglu`) is RMSNorm by construction, and the reduce entries are `torch.sum` / `torch.nn.RMSNorm`. An entry filed
under those would hand every re-tune consumer the wrong kernel to rebuild. So the two LayerNorm→linear
contractions and the LayerNorm statistic reduce now deploy off the prior; adding a LayerNorm-cone kind is what would
let them be recorded.

**`STAGE`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the operand-staging codec
`d<depth>/smem|smem-async|smem-tma[/p<reg_depth>]` on the typed `Stage` schedule struct (composes with both
fragments of the `TILE` knob): `d<depth>` the gmem→smem ring depth, `sync`/`cp.async`/TMA transport, `p<reg_depth>` the
smem→register double-buffer (on a `wgmma` drain, which loads no fragments, the MMA groups left in flight).
`stage=None` (unset / unparseable) = gmem-direct. A `STAGE` value names only what
the schedule CHOOSES — rotation and refill discipline derive at materialization from the depth alone (which is why the
retired `ring` flag compiled byte-identically with and without it), and `smem` / `bk_elems` are resolver outputs,
never spelled. See `lowering/kernel/ARCHITECTURE.md`.

`d1/reg` selects persistent register storage for a matrix recurrence whose rows are independent across warps.
The chunk loop runs inside each CTA, with one FP32 carry slot and reused matrix fragments. Its promotion interval
is the `TILE` K chunk. The prior receives a register-storage indicator rather than shared-memory pipeline features;
existing fitted artifacts have no coefficient for the new indicator until refitted.

**`WSPEC`** (STR codec, RETIRED) — the warp-specialization producer band `p<np>` is INVENTORY: realized rows spell
it as `WORK`'s `+p<np>` suffix, `SCHEDULE_FAMILIES` no longer lists it, no shipped golden carries the key, and the
enumeration neither reads the `EMMY_WSPEC` pin nor offers a `WSPEC` level — pin `EMMY_WORK=w4x2+p2` instead. A stray
`WSPEC` key on a stored row is no longer stripped before matching; it simply names a family no row decides, which
the "family not decided at this fork" rule already reads as free. The `Knob` declaration is gone, and so is the
codec that served it — what survives is one integer, `WarpSpec.producer_warps`, which the materializer reads off
`TileOp.workers`. A band is
legal on a warp `TILE` over a resolved **TMA** `STAGE` within the thread budget (`block_threads + 32·p ≤ 1024`,
`32·p ≤ block_threads`) with no cross-CTA split; an inventory whose band nothing can drive enumerates no row at
all, rather than silently degrading to uniform. Empty = uniform SIMT. Materialized as the staged K-loop's
producer/compute band split (`_stage._producer_band_kloop`).

**`RASTER`** (STR codec, the tile schedule → `lowering/kernel/010_materialize`) — the CTA launch-order
codec (bare/root-global; the fifth schedule-fork level): `gm<G>` iterates `G` M block-tiles fastest per
launch stripe so consecutive CTAs share the streamed B slab (L2 reuse — the flat order streams B from DRAM once per
M-row: `A + C + B×2` measured on the 4090's `mlp_gate_up`, 503.6 vs cuBLAS's 365.8 MB); `gn<G>` is the transpose
(A streamed); empty = the flat N-fastest row-major order. Changes
no per-CTA work, layout, or schedule — only the block-id decode (`ir/kernel` `Tile.render`, `Tile.raster_axes` the
`grid_tile` eligibility). The fixed 2-D contraction domain is `('', 'gm8', 'gn4', 'gn8')`; the schedule restriction
keeps `gn4` and `gn8` out unless an exact `RASTER` parameter selects one. Wall-time effect is small and shape-dependent
(±2–4% measured), so golden evidence arbitrates per shape.

**`SHARED_CARRY`** (INT, `lowering/kernel/035_shared_carry`) — late carried-state storage: `0` keeps global storage
and one launch per ordered step, `1` keeps the state in two shared buffers, each row padded by one column. The pass
offers the shared layout only when state reads prove CTA ownership and both buffers plus existing scratch fit. The
choice is part of the measured kernel row, so a recorded row replays its storage without a manual pin. Shared
storage is not a rule: with few independent batches and a large per-step grid it serializes the card (2 CTAs doing
every row: 163 ms against 2.8 ms for ordered launches on a V100), so evidence decides.

**`S_*` and `H_*` are not knobs.** They are feature columns (`features.STRUCT_PREFIX` / `CTX_PREFIX`): `S_*` a
kernel's structural features (statement/op histogram + loop extents + operand dtypes), computed from the kernel
(`features.stamps`); `H_*` the card and regime (`Context.features`). Neither is written onto an op: an op's knobs
hold decisions only, every key a registered knob (Part 6).

**`FAST_MATH` / `F16_MMA_F32_ACC` / `FP8_MMA` / `FAST_EXP`** (BOOL, pin-only precision restrictions /
`lowering/kernel/085_fast_exp`) — the **precision-trading family**. Precedence per knob: its own pin > the
`FAST_MATH` umbrella > true (`space.precision_pin`). `FAST_EXP` swaps libm `expf` for `__expf`;
`F16_MMA_F32_ACC` admits the fixed domain's f16-accumulate atom choices (`mma_m16n8k16_f16_f16` — chunked f32
register promote), while `FP8_MMA` admits its native fp8 atoms. Without the effective gate, Algorithm 1's immutable
context excludes those choices while composing its lazy frontier.
`FAST_MATH` also controls NVCC `--use_fast_math` and the typed invariant-divide rewrite. It remains `unfeatured`:
schedule choices keep their concrete knob identity, and effective compiler flags separate measurement contexts.
New golden inventories and measurements record the effective umbrella explicitly, so replay preserves that regime.

### Classic schedule keys

Structural choices finish before the `TileOp` indexes its Fold root. Each shared Fold object gets
one preorder integer id; every consumer operand position gets a distinct `(consumer id, operand position)` tuple, even
when two edges reach the same producer. The strict codec spells kernel choices as bare `WORK` / `RASTER`. `TILE`,
`REDUCE`, and `STAGE` are also
bare when their family has one applicable consumer node; only an ambiguous family carries the site's route
(`TILE@map.1/twist.1/inner`, the placement grammar). `STAGE` is one
transport decision shared by the applicable operand edges at that consumer. Empty direct values remain explicit, so
every leaf has the same key vocabulary.

`PLACE` alone retains the Fold tree-path grammar because it changes kernel boundaries before classic sites exist.
Repository goldens retain the shortest unambiguous grammar; mutable tune DB rows are discarded after a re-key,
never migrated. `tuning_knob_items` renders keys as stored and all decode paths use `ClassicScheduleCodec`.

### Odds and ends

- `BINMASK` parsing accepts a binary string (`"101"` = bits 0 and 2), the keywords `"all"` / `"none"`, or a decimal /
  `0x`-hex int clamped to the candidate width.
- `tuning_knob_items` leaves `BOOL` knobs out of the tuning-knob view — they are treated as markers saying that a
  pass ran.
- `HOIST_COMPUTE` and `PAD_SMEM` are BOOL autotune forks emitted in a fixed order, with the greedy default first
  (inline-fuse and pad-on respectively); both honor their `EMMY_*` pin.
- The alignment padding for a masked-K MMA block is **not** a fork. It is written onto the `Source` at staging as an
  intrinsic property, because it is almost always a win, and a greedy compile deploys it without needing a measurement.

## Pass directories

Pass files are numerically prefixed so `sorted()` picks them up deterministically. Pick a fresh prefix when adding a
rule; the loader ignores the prefix itself — it only makes the ordering readable. Per-pass authoring invariants are in
[`passes/ARCHITECTURE.md`](passes/ARCHITECTURE.md); the tile passes (`010_lift` → `040_schedule`) and the set
of algebraic rewrites they may apply are documented there too.

| Pass                      | What rules do                                                                                |
|---------------------------|----------------------------------------------------------------------------------------------|
| `frontend/decomposition/` | Rewrite frontend ops (`LinearOp`, `MatmulOp`, `SdpaOp`, layout ops, fused `rms_norm` / `layer_norm` / `softmax`) into tensor-IR primitives + layout-only `IndexMapOp`s, broadcast-explicit via `_broadcast.broadcast_to`. |
| `frontend/optimization/`  | `compose_indexmaps`: collapse chains of single-source / single-consumer `IndexMapOp` into one coord_map, so trivial layout kernels don't block fusion. |
| `loop/lifting/`           | `lift_*` rules wrap each surviving tensor primitive in a trivial one-op `LoopOp`; an additive scan writes its accumulator after every ordered scan-axis update. |
| `loop/fusion/`            | `roll_recurrence` first rolls an unrolled recurrence into one kernel that carries its state (`passes/ARCHITECTURE.md`). `merge_loop_ops` then maximally splices each downstream Loop region without consulting Tile IR or schedule support. Non-reconvergent consumers become ports of one multi-output `LoopOp`; one shared splicer worklist deduplicates their common producers. Only semantic splice legality stops a merge. |
| `loop/stamp/`             | `stamp_loop_names` (`provenance.name_for`, e.g. `k_rms_norm_3f2a1b`) — the name is the one thing stamped. Runs last in the loop dialect, after maximal fusion. |
| `tile/{lift,cut,schedule}/` | `010_lift` forms each kernel through `lift_kernel`, the formation cut and split pieces share: it re-fuses free axes a fused reshape split (`p → f/Q, q → f%Q`, kept only when every access folds clean), so split and unsplit spellings of one contraction converge to one kernel identity, then converts the complete inner loop nest to a canonically factored Fold tree; `020_twisted` rewrites the exp family; `030_cut` reaches a fixpoint over stored-edge cuts, constant layouts, then cross-CTA splits; `040_schedule` schedules each stored tree. |
| `lowering/kernel/`        | `010_materialize` lowers the selected schedule through `_factor.factorize`, followed by the Kernel IR peepholes. See [`passes/lowering/kernel/ARCHITECTURE.md`](passes/lowering/kernel/ARCHITECTURE.md). |
| `lowering/cuda/`          | `delegate_zero_init` (first) moves an atomic accumulator's per-launch zero-init off the runtime memset and into a dataflow-predecessor kernel as a `ZeroPrologue` stmt (every thread of the grid writes a stride of zero words ahead of the kernel's own work; stream order guarantees happen-before) — one CUDA-graph MEMSET node saved per site; the capture's first launch and symbolic-shaped accumulators keep their memset, and the slab planner starts the buffer's live interval at the delegating launch (`CudaOp.zero_prologues`). `lower_kernelop` then renders the `KernelOp` body to a `__global__` source string (`ir/kernel/render.py::render_kernelop`) and mutates the node's op to `CudaOp` in place. |

SiLU decomposition follows PyTorch opmath precision: f16 and bf16 inputs widen once to f32, the primitive
negative/exp/denominator/reciprocal chain and final product compute in f32, and the result converts once to the
declared output dtype. F32 and f64 inputs retain their dtype, so decomposition never demotes a wider input.

## Dump hooks (`dump.py`)

`CompilerDump.on_pass(idx, pass_name, graph)` dumps the post-pass graph uniformly for every pass:
`NN_<pass_name>.{json,txt,dot}` (+ `NN_<pass_name>.kernels.txt` if any node has a non-empty `pretty_body()`). Slashes
in the pass name flatten to underscores. The pre-pipeline input graph is dumped separately as `00_input.*` via
`dump.dump_input_graph(graph)`. The uniform strategy means adding a pass automatically gets dumped — no registration.

Per compute kernel, `_dump_per_kernel` writes `<prefix>.kernels/<kname>.json` — a standalone lowered sub-graph
(kernel + its `InputOp` / `ConstantOp` producers) loadable via `emmy run --ir`. Original frontend slices selected by
provenance stay in memory for `run --bench`'s per-kernel benchmarking and are never written as trace artifacts.

## Per-rule diff output (`rule_diff.py`)

At `compile -vv` (DEBUG) the engine emits one block per rule application: a unified diff between the matched subgraph
and the rewritten fragment, bracketed by `>>> <pass>:NNN_rulename` and `<<< <pass>:NNN_rulename` markers. The `<pass>`
prefix is the single-letter shorthand from `PASS_SHORTHAND` (`d` / `o` / `l` / `f` / `n` / `s` / `t` / `p` / `h` / `k`
/ `c`) — the same letters the CLI accepts in `--passes dolfstph` (`commands/compile.py` imports `PASS_SHORTHAND` so
the flag and the marker prefix can't drift). The three tile passes have a letter each, so `--passes dolfstp` ends
after the cut pass: the greedy decides every kernel-set fork as a full compile does (no arm is scheduled to decide
one) and the tile IR shows the chosen kernel set unscheduled. Skipped rules collapse to a one-liner. The bracketing
makes per-rule / per-pass slicing trivial via `awk`; ANSI color is applied only inside the diff body so the markers
stay plain ASCII. Color follows `compile --color`. Body-carrying ops render through their own `pretty_body` (the
in-flight `TileGraphOp` pretty-prints its block-DAG), so a tile-pass diff reads as a readable block-DAG delta. The
structured `.rules.json` dump is unaffected — the diff is purely presentation.

## Invariants

- A rule module must not reach into the engine's internals; its interface is `PATTERN` + `rewrite(graph, match)`.
- `pipeline/` imports from `ir/` but never from `backend/`. Lowering rules produce IR; executing that IR is the
  backend's job.
