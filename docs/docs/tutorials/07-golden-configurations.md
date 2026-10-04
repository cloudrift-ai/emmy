---
sidebar_position: 7
title: "7. Golden Files"
description: The reviewed, per-GPU measurements that ship with the repository — a card's tuning database as a file, how one is recorded, and how it is kept current.
keywords: [Emmy, golden file, evidence, benchmark, pin, restamp]
---

# 7. Golden Files

A **golden file** is a card's measurements in the shape of the tuning database: the kernels the card measured, the
kernel-set decisions taken on them, and the measured rows — beside the traced programs the kernels came from. They
matter more than their modest description suggests, because as [the stores page](./04-measuring-and-recalling.md)
noted, **they are the only measured data that travels with the repository**. On a freshly rented machine they are the
difference between deploying on evidence and deploying on a guess.

They do three jobs at once:

1. **Measured evidence for the deploy.** A compile imports the file's rows into the tuning database before it picks, so
   every measured row is one more row in the measured-evidence index the greedy pick reads — beside the rows a bench
   measured on this machine; see [the hierarchy page](./06-deploy-evidence-hierarchy.md). A golden is a preference
   among measured rows, never a forced pin. The same row also replays under a hand pin for a measurement
   (`run --golden PATH --realization NAME --bench`).
2. **The training data** for the offline prior, which is fitted on them.
3. **A regression reference** — if today's compiler produces something slower than the recording, that is a defect
   with a number attached.

## What one looks like

Model goldens live under `recipes/<model>/golden/`, in one file per exact GPU model and compute capability. The
maintained model-agnostic golden records live under `emmy/compiler/pipeline/search/golden/records/`. A file is JSON
with four tables, one entry per line:

```json
{"gpu_name": "NVIDIA GeForce RTX 5090",
 "compute_cap": [12, 0],
 "model": "google/gemma-4-12B-it",
 "programs": [
  {"inputs":[...],"outputs":[...],"nodes":[...]}
 ],
 "kernels": [
  {"loop_ir":{...},"name":"k_linear_7a1c2e","formed":true,"traced":0,"origins":["linear_7"],"bindings":{"num_tokens":32}}
 ],
 "routing": [
  {"parent":"k_linear_7a1c2e","arm":{"PLACE@inner.1/map":"cut"},"children":["k_linear_7a1c2e__place_d2802d6545","k_linear_7a1c2e#2"]}
 ],
 "rows": [
  {"name":"gemma4_12b.norm_q_proj.m32","kernel":"k_linear_7a1c2e","pins":{"FAST_MATH":false},"knobs":{"WORK":"w1x16","TILE":"mma_m16n8k16_f16_f32/f2x2/k2","REDUCE":"g8k","RASTER":"","STAGE":"d2/smem"},"measurements":{"emmy_us":26.7,"reference_us":19.8,"reference_backend":"cublas"}}
 ]}
```

- `programs` are the traced Torch IR programs — provenance: the twin a benchmark compares a kernel against, and what
  a record run re-compiles.
- `kernels` are the kernels' definitions: the standalone Loop IR body, the C name, and, for a kernel lowered from a
  program, which program (`traced`), which of its ops it computes whole (`origins`) and the sizes that specialized it
  (`bindings`). A piece a cut or a split minted has no program of its own; a routing row reaches it from its parent.
  Nothing computed from a kernel is stored beside it. The identity the tuning database keys a kernel by, and the
  structural features the prior reads, are computed from the Loop IR when they are needed. Inside the file, rows and
  routing rows name a kernel by its C name, or by a `key` such as `k_linear_7a1c2e#2` where two kernels share one.
- `routing` are the kernel-set decisions: the kernel a decision was taken on, the arm (a `PLACE@seam: cut`, or the
  cross-CTA half of a `REDUCE` value) and the pieces it minted, in order. A decision has no time of its own: at the
  parent's fork it is priced as the sum of its pieces' fastest rows.
- `rows` are the `perf` rows: the kernel, the sizes its symbolic dims were benched at, the input regime (`pins`, where
  `FAST_MATH` lives), the schedule row (`knobs`, spelled as [the forks page](./03-forks-and-knobs.md) spells them,
  every family written out) and the measurement beside a named reference. `name` is a label a command selects a row
  by. A row with no measurement is a proposal, not evidence.

Names repeat across files — every GPU has its own `matmul.square.512` — with different shapes, different data types
and different measured times. So `--realization NAME` without `--golden PATH` resolves inside the live card's files.
Pooling them would mean replaying one card's configuration on another.

## Recording one

A golden is recorded from a side-by-side comparison run:

```bash
emmy run --realization matmul.square.512 --bench --ab "WORK=w2x2,TILE=f2x8,STAGE=d2/smem-async"
```

To verify a row still living in a working golden file, select both the file and the row. The same two flags spell it
on every command (`run`, `compile`, `serve`):

```bash
emmy compile --golden _tune/model/working.json --realization target.name --ir cuda
emmy run --golden _tune/model/working.json --realization target.name --bench
```

The row's kernel supplies the program regardless of state — its stored body, with a weight it reads bound from the
same checkpoint tensor as its twin's. A row named explicitly is always benched as a pinned row, measurement state
notwithstanding; `run --golden PATH` alone walks every target kernel and benches only its measured
rows, skipping proposals. `--ab` is a hand pin for one extra bench row — a way to try a row, not a way to replay a
golden.

That compiles the kernel the way the compiler would on its own, then compiles it again with the given knob values
pinned, and prints both. Whatever it measured cleanly is written into the tuning database by default — per-kernel rows
the next compile deploys from — which is how a replayed golden or a hand-pinned row becomes what the compiler chooses.
`--record-greedy` writes the greedy pick's whole kernel set back into the working golden exactly as the database holds
it: the kernels, a routing row per decision, a measured row per kernel. Two rules about which number to copy:

- **Record from a pinned row, never from the ordinary comparison row.** The ordinary row is measured interleaved with
  the PyTorch baselines, so the allocator state and cache contents of another framework are resident while it runs. It
  is the right number to compare against PyTorch and the wrong number to compare against a pinned row — the gap
  observed in practice is around 7 percent. Whenever pinned rows are measured, the ordinary configuration is
  additionally re-measured on its own, and *that* is the baseline the pinned rows are compared against.
- **Copy the whole knob map, including the families that are off.** Each row prints the knob values the compile
  actually produced, with every schedule family written out explicitly. An entry that omits a family leaves that
  family to whatever the compiler fills in when the entry is replayed, which shifts as the compiler evolves — a
  recurring source of regressions that look real and are not.

Three checks guard the measurement before it is believed:

1. **The pin must have been honored.** The knob values the compile produced are compared against the pinned ones, and
   a mismatched row fails without being measured at all. A structurally invalid pin silently falls back to the
   compiler's own choice, so measuring it would compare the compiler against itself and report a flattering result
   under the pin's name.
2. **The arithmetic must be plausible.** A row implying more arithmetic per second than the card can physically
   perform is flagged as a bad measurement rather than celebrated as a fast kernel.
3. **The answer must be right.** Each pinned configuration is executed once on the same inputs as the ordinary run and
   its outputs compared. A kernel that skips a step it should not have produces plausible-looking garbage very
   quickly.

Every row is measured in a separate worker process that can be killed, so one configuration that hangs takes down its
own process, is reported as a failure, and the remaining rows continue.

## Recording rules worth knowing

- **A decision and a schedule are two different tables.** A cut is decided before schedules are chosen, so it is a
  routing row on the kernel it was taken on, and each piece it mints has a schedule row of its own. As evidence the
  routing row is the measured price of that kernel set: at the placement fork it is the sum of its pieces' rows and
  outranks any arm the prior would have to price. A kernel that ran whole has a schedule row and no routing row; the
  schedule row itself says the kernel ran whole.

- **A kernel is its identity.** In the tuning database, rows are keyed by the kernel's exact identity — the digest of
  its normalized body and its buffers' types and shapes — and that identity is what a compile joins a candidate on.
  The file does not store it: the import computes it from each stored body. A row either names a kernel the compile
  builds, or it is not consulted.

## Validating a file

One command checks a corpus against its pinned serving envelope:

```bash
emmy eval golden --golden <canonical-golden.json> --serving-config <models/slug.env>
```

The serving config names that exact file and supplies the model, revision, GPU, and the sizes and regimes each
serving twin reaches. The command must run on that GPU. It validates the provenance, proves every twin's kernels
carry a row at every size and regime the config reaches them at, then compiles the freshly traced serving twins of
every precision lane with the file's rows as the only evidence, strictly: a twin with a fork no row decides fails the
release naming the kernel. Beyond that, a recorded row's health is its pinned replay — `run --golden PATH
--realization NAME --bench` reproduces it under the A/B integrity gates above.

### Against a fresh lowering

A row is evidence for the kernel it names, and a deploy keys rows by the kernels it lowers fresh from the model. When a
compiler change moves that lowering the file goes stale: serving builds kernels none of its rows describe. Two
commands cover it, and neither needs a card:

```bash
emmy golden check [PATH…]      # what a restamp onto the fresh lowering of the golden's own programs would change
emmy golden restamp [PATH…]    # write that rewrite
```

A restamp lowers every traced program again, gives each kernel the body the fresh lowering gives it, and takes every
kernel-set decision again on the fresh parent. A kernel that kept its identity keeps its rows and their measurements.
A re-keyed kernel — the stored body and the fresh one are two different kernels — keeps its rows as proposals: the
schedule stays, the microseconds go, and a record run on the card measures them again. Both identities are computed
at restamp time, so a change to how identity is computed re-keys nothing. A kernel no fresh kernel writes, and a
decision the fresh parent no longer takes the same way, are dropped with their rows and named. A file the restamp
would leave unchanged is current, which is what `check` and the test suite ask; a file nothing survives in is left
untouched. Both default to every repository golden. The `refresh-golden` skill is the whole flow, including what
needs a card.

## Two smaller rules

**Fast math never loses.** Entries recorded with faster, less precise arithmetic are only kept when they are faster
than the best ordinary sibling — a slower one documents a configuration nobody should replay, so such rows are
dropped.

**The goldens are the prior's training data.** The offline prior is fitted on them, and the next page explains
how.

## See it yourself

Read a real file — a row is one line, and its name says which target and shape it records:

```bash
find recipes -path '*/golden/*.json' -print
grep -n '"name": "post-sym.k_linear_mean_reduce' recipes/gemma-4-12B-it/golden/rtx5090_sm120.json | cut -c1-300
```

Then run the file-scoped validation on the GPU named by the serving config. It needs model configuration and
allocation metadata, but no weight payload:

```bash
emmy eval golden --golden recipes/gemma-4-12B-it/golden/rtx5090_sm120.json \
  --serving-config docker/vllm-emmy-serve/models/gemma-4-12b-it.env
```

Next: [8. Inside the prior](./08-inside-the-prior.md).
