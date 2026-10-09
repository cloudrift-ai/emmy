# The realization corpus

A data-driven regression lane for pinned schedules. Most cases are minimized reproducers of one failure class: **a
schedule that should be realizable is not**. A small capability baseline also keeps the live GPU stages exercised where
the corpus would otherwise have no exact-capability case. Every case has one traced program, one target kernel and one
authored kernel set, and the lane replays it against the compiler in front of you as the compile's only evidence,
strict, to ask whether the compiler realizes, builds and runs the set the way a deploy would.

Symbolic cases keep runtime dimensions in the CUDA compile. Their NumPy frontend twins bind those dimensions to
the generated input shapes, since the reference evaluator requires concrete reshape and index-map extents.

This directory is kind-organized in the sense `tests/ARCHITECTURE.md` sanctions: its cases span lowering, the CUDA
backend, the pin machinery and the golden loader, and they share one workflow.

## Layout

```
helpers.py            # load, regenerate, complete, and the three oracles
regen.py              # `make test-corpus-regen` — applies the fix the staleness test detects
test_realization.py   # one parametrized walker over cases/
cases/<family>/<name>.json
```

## What earns a case

A case earns a place in the corpus in either of two ways:

- A realization gap: a schedule family that is never offered, a pin that refuses or fails to lower, or a pin that
  runs wrong.
- Capability coverage: a small representative set from a model-agnostic hardware golden when that capability would
  otherwise run no `built` or `correct` nodes. Keep distinct kernel kinds and schedules, omit measurements and GPU
  names, and stop once the main paths have live coverage. This proves those rows still build and run; it does not
  qualify the full golden.

One case per regime. A second size of a row the corpus already realizes (a bigger grid on the same tile, a batch that
only multiplies the launch) or a second spelling of the same identity and row (an output dtype the identity does not
distinguish, the same program at another tile width) is not a case: it walks the same code three more times and
catches nothing the first did not. A sweep that exists to be measured belongs to the perf lane, which reads the corpus,
and `cases/qwen3emb/` is that sweep; a family that is not one keeps one size per regime.

Two neighbouring failure classes deliberately do **not** earn a case, because admitting them would make the ratchet
meaningless:

- **search shortfall** — the schedule realizes and the prior simply does not pick it when nothing measured is in
  scope. Fix it by measuring, or report it as a prior finding. (A row that *is* in scope and still is not picked is
  not a shortfall: it is the deploy contract failing, and `realized` reports it.)
- **code generation quality** — the right tier is present and still loses. Report it; do not record it.

A schedule the compiler *correctly refuses* is not a gap either. A slab that does not fit, a byte transport with no
sibling on a computed operand, a masked axis with no register block to stage — these are right answers, and their
tests assert the refusal message, which a row cannot express. Keep them in Python.

That rule is about what earns a case, not about what a case then measures. Once a case exists it is an ordinary
reproducer and carries whatever the lane can learn from it.

## The case file

A golden file (`emmy/compiler/pipeline/search/golden/ARCHITECTURE.md` is in the pipeline's) with one traced program,
one target kernel, the kernels its authored decisions mint, those decisions as routing rows, and one authored row per
kernel of the set — and nothing else:

```json
{"compute_cap": [12, 0],
 "programs": [
  {"inputs":["a","b"],"outputs":["c"],"nodes":[…]}
 ],
 "kernels": [
  {"loop_ir":{…},"name":"k_matmul_5b7645","formed":true,"traced":0,"origins":["c"]}
 ],
 "rows": [
  {"name":"k_matmul_5b7645","kernel":"k_matmul_5b7645","pins":{"FAST_MATH":true},"knobs":{"WORK":"w2x2","TILE":"mma_m16n8k16_f16_f16/f4x8/k2","REDUCE":"g2k","STAGE":""}}
 ]}
```

Why each part, and why nothing else:

- `programs` / `kernels` / `compute_cap` — the reproducer: the kernel's own Loop IR, which every stage starts from, and
  the stable Torch IR its `origins` came from, which `correct` compares against. Not a code snippet, so a frontend
  change cannot silently alter what the corpus tests. The kernels are DERIVED: each body is what the compiler in front
  of you makes of the program, and the staleness test below holds it to that. A case stores no identity and no
  stamps; both are computed from the body where they are read.
- `routing` — the kernel-set decisions the case authors: each `PLACE@seam: cut` or cross-CTA `REDUCE` arm on the
  kernel it is offered on, and the pieces it mints. A case that cuts its target carries no row on the target: the
  target never runs, its pieces do.
- `rows` — the authored schedules, one per kernel of the set, each naming its kernel by its `ref` in the file (the
  kernel's `key`, or its C name where no key is set; a second kernel sharing a C name carries a key like
  `k_matmul_5b7645#2`). `name` is a label, written once and never re-derived: the kernel's provenance name for the
  target's row (`k_matmul_5b7645` — the ops it realizes, as the backend and the profiler show it), that name plus the
  piece's identity prefix for a further row. `--realization` selects a row by it, so it has to stay put whatever the
  compiler does to keys and features. `pins` are the input regime; `knobs` are the row the kernel realizes, spelled
  on that kernel's own tree. Regeneration structurally cannot produce these, which is what makes the staleness
  mechanism safe.
- The optional per-card `latency` block on a row is the only addition the corpus makes to the golden schema.

Three spelling rules decide what a case actually asserts:

- **On a kernel with several sites for one family, spell the family by route.** A bare `TILE` there asks for one of
  the sites — one carries the value, the rest are OFF — so a case that means "this tile at BOTH contraction roots"
  and spells it bare asserts something weaker than it reads, and passes on a schedule it was written to refuse.
- **A row is a complete schedule.** It spells every site its kernel decides, OFF as `''`, exactly as a golden row
  does, because the evidence pick matches it against one enumerated schedule. A partial row — a key left out for the
  fork to choose, a bare family on a kernel whose sites are routed — equals no schedule, and strict evidence names the
  kernel. `helpers.complete` writes the rows a compile realizes for the kernels no row names, and drops a row naming
  a kernel the compiler no longer mints.
- **Binding a symbolic dimension specializes the program.** A case with no sizes keeps its symbolic axis and runs at
  the dimension's own `Dim` hint — the size `emmy run` already resolves a symbolic reproducer to. The corpus has no
  spelling for "compile at the hint, run at some other size", so a sweep of one symbolic kernel across many runtime
  sizes stays in Python.

## Expectations live in the filename

There is no manifest. Extending the corpus is writing one file:

```
<family>/<name>.json                  # closed — every applicable stage must pass
<family>/<name>_xfail_realized.json   # open — strict xfail at that stage
<family>/<name>_xfail_built.json
<family>/<name>_xfail_correct.json
```

The open-gap inventory is `ls cases/**/*_xfail_*.json`, and the completion gate is "no file matches that glob". Closing
a gap is a `git mv`, so the diff shows the closure as a rename. And two concurrent runs on different models can each
add a case without touching a shared file.

The cost is that the filename is semantic, so an `_xfail`-shaped token naming something other than the three stages is
a hard error rather than a silently-closed case.

An `_xfail_*` file's `note` must carry an `evidence:` paragraph naming why the schedule *should* be realizable — a
sibling card's golden carrying that family for the same structural identity, the same family already winning at a
neighbouring binding, or an explicit roofline argument. Without it the corpus fills with speculation. The rest of the
note is prose about where the gap came from; regeneration keeps it.

## The three stages

| Stage | Assertion | GPU |
| --- | --- | --- |
| `realized` | with the case as the compile's only evidence, the graph lowers through `CUDA_PASSES` at the declared capability, `unreproducible_pin_flag` is `None`, every authored family is stamped, and every kernel-set decision the case records was taken | no |
| `built` | lower the same way on the live card, then build a `CompiledProgram` — nvcc accepts it | yes, exact capability |
| `correct` | run against the reference within tolerance | yes, exact capability |

All three run under `helpers.evidence_scope`: the case's file is the whole golden scope, strictly (`golden.sole_evidence`,
the scope the release gate compiles under too; each row standing in as a measured row — a case authors schedules
rather than measuring them, and a proposal is no evidence), so a fork no row decides is an `EvidenceError` naming the
kernel, never a prior's guess; the tune DB is not consulted, and the environment carries the case's input pins alone —
the regime it was authored under, never a route or a schedule row. The route and the rows reach the compile as the
DB rows of the kernels they decide — imported into the compile's DB, as every compile imports its golden scope
(`golden/evidence.py`), and read through the same evidence pick every `compile` / `run` / `serve` uses
(`greedy._route_candidates`) — or they do not reach it at all. That is the deploy contract, asked of every case on
every commit: a row the compiler can honour under a pin but does not select when it is the evidence — a schedule that
equals no leaf of the kernel that deploys, a decision whose arm the fresh parent does not take — fails `realized`, and
the failure names what was lost: strict evidence naming a kernel whose row no leaf equals is the lockout a case exists
to catch. A kernel-set decision is checked through the engine's own splice events (`PipelineStrategy.on_splice`),
because no stamp on the resulting kernels can show a placement cut or a cross-CTA split that was not taken.

Each is its own test node, so an `_xfail_<stage>` suffix lands on exactly the stage it names; the stages past a
declared gap are skipped, because a schedule that never realizes has nothing to run.

`realized` always runs at the **declared** capability, so an sm_70 lockout is exercised on any box, GPU or not. `built`
and `correct` run only when the live capability **equals** the declared one — a pinned schedule is a claim about one
capability, never about a merely newer card.

The reference for `correct` is the kernel's traced ops (the target's `origins`) run on the numpy backend, the slice
`emmy run` benchmarks against (`GoldenFile.reference_program`); a kernel with no exact frontend twin compares against
the same-input greedy execution of the same program. Both sides share random weights by source path and bind them
through their own load transformations, so a lowered transpose still reads the reference's weight.

## Latency

A case may also carry measured microseconds. **The measuring lives in `tests/perf/`**, not here: `make test` compiles
at `-Xcicc -O1`, which is not a measurement lane, so a latency assertion in this directory would measure the wrong
regime entirely. That lane benches every closed case its card can run, joins the result to its comparison table, and
compares the same measurement against the stored number — one bench, both answers. See `tests/perf/ARCHITECTURE.md`.

A slower case **reports** rather than fails: enforcement belongs in a human reviewing the timing-refresh pull request,
not in a red test a legitimate correctness fix could pin red forever. Nothing auto-updates a stored number — an
automatic ratchet ends up pinned to the luckiest noise excursion ever observed.
`emmy run --golden FILE --realization NAME --bench --record` is the only writer.

The band is 5%, measured rather than guessed: ten cases spanning 1.5 us to 579 us, four estimates each on an idle RTX
5090, put the best-of-three estimator's own spread at a median of 0.17% and a maximum of 0.74%.

Latency lives in an optional per-card block on the case's first row, keyed by `Context.hardware_id` — the identity that
already separates same-die SKUs like H100 from H200, which a free-text card name does not. Both numbers are stored,
because the block answers two questions and only one of them is a ratchet: `emmy_us` against its own stored value says
*did we regress*, and `tcompile_us` beside it says *are we ahead of or behind torch*, per case, per card. That ratio,
sorted, is the optimization worklist.

A closed case with no timing for the live card is reported once at session end, on a card that can answer it and
nowhere else. That asymmetry is deliberate: the derived-half check is GPU-free so it fires everywhere and its fix works
everywhere, while a timing can only be produced on the machine holding the card.

The perf command names the case's first row (`run --golden <case> --realization <name>`), which benches it as a pinned
row whatever its measurement state: that row's input `pins`, the decisions that mint its kernel, and its schedule
`knobs` are published as a hand pin for that one compile, and the case's file is the compile's golden evidence, so the
set the case authors is the one measured, never the planner's own pick under its name.

## Staleness: regeneration, not stamps

The kernels a program lowers to and the schedule codec's spellings change often, so a stored case rots. The failure
mode that matters is silent: a retired knob spelling canonicalizes to itself, matches no candidate, and reports as a
lockout — a phantom compiler gap. For an open case the mirror applies: the xfail keeps passing and the ratchet stops
ratcheting.

**Detection is a test, not a command.** `test_case_derived_half_is_current` restamps each case the way every
repository golden is restamped (`golden.restamp`: the kernels re-derived from a fresh lowering of the program, every
decision taken again, the rows' knobs re-canonicalized) and asserts the result equals what is stored. The check is
GPU-free at a fraction of a second per case, so codec and lowering drift is caught on the pull request that
causes it, by the commit that causes it. A stored kernel is stale when its body and the fresh lowering compute
different exact identities; a change to how identity is computed alone leaves every case current.

`make test-corpus-regen` only *applies* the fix. That split is the shape the repository already uses twice:
`ruff format --check` detects while `make format` fixes.

Four rules make it load-bearing:

1. **Regenerate through the library, not a CLI.** The restamp runs under an explicit `Context.from_target(compute_cap)`
   and stamps no card, which is what makes the check machine-independent, so it fires and its fix works on any box.
2. **Validate authored knobs strictly.** `validate_family_value` requires every classic value to use its sole wire
   spelling. Loading fails loudly on `STAGE=d2/ring`, `WORK=zzz9x9`, `TILE=mma_m64n64k64_…` or `REDUCE=g2z`, while a
   canonical but unreachable pin (`WORK=w7x13`, `TILE=…/f99x99/k8`) parses cleanly and falls through to `realized`,
   where a genuine lockout belongs.
3. **Refuse to write when a verdict changed.** If one commit re-keys a kernel and breaks realization, regeneration
   fixes the first and must not let the second ride along. It names the affected cases and exits non-zero; resolving
   them is a review conversation, not a mechanical step.
4. **Keep the note.** A case's evidence citation is the file's `note` field, not a comment, so a regeneration and
   every dump carry it.

A row whose kernel the restamp re-keys loses its latency with the re-key: the timing was of another kernel. A row
naming a kernel the fresh lowering no longer mints is dropped by the restamp and named in its report, and `COMPLETE=1`
(`regen.py --complete`, `helpers.complete`) authors a fresh row for the kernel standing in its place.

**The authored half rots differently.** Those rules are about the DERIVED half, and they all assume the case still
loads. When an IR dataclass loses a field, every case whose stored program serialized it stops parsing —
`GoldenFile.load` refuses the whole document on an unknown field, so `test_case_derived_half_is_current` reports a
load error instead of a mismatch, and regeneration cannot help because it has nothing to read. The fix belongs with the
commit that retires the field: drop the retired key from the stored programs. That is lossless exactly when the field
sits at its default in every case, which is the ordinary situation for one only a now-deleted construct ever set; a
case that used the construct for real has lost its program and needs re-authoring, not editing. So retiring an IR field
means grepping the corpus for its wire name, not only re-running regeneration.

## Adding a case

```bash
case=tests/compiler/realization/cases/<family>/<name>_xfail_<stage>.json
emmy trace -c "<snippet>" --target sm_<cc> -o "$case"
# then author the row's knobs (copied from `run --json`'s record_knobs) onto the traced row
make test-corpus-regen COMPLETE=1   # restamps, adds a row per undescribed kernel and a routing row per decision taken
```

`COMPLETE=1` compiles each target the way the deploy reads it and appends a row for every kernel of the set no row
names — that kernel's `ref`, the input regime and the schedule row the compile realized on it — and a routing row
for every kernel-set decision the compile took, so strict evidence has a row at every fork. That is authoring: the
added rows are enumerable schedules the case pins from then on, and a kernel the author cares about should get its row
by hand before the completion fills in the rest.

Then prove the case reproduces the gap: it must fail without the suffix and pass with it.

## What stays in Python

The boundary is what a row can express.

- **Corpus, as data:** whether a pinned schedule is realized, built and correct at a capability.
- **Python, as code:** *how* — emitted-source substrings, bit-identity between two configs, kernel counts, refusal
  messages, compile-budget claims, and a sweep of one symbolic kernel across several runtime sizes.

The corpus is therefore overwhelmingly additive. It subsumes the accuracy-only tests whose whole assertion was "this
program under this pinned schedule computes the right answer"; a test that also asserts structure keeps its structural
half.
