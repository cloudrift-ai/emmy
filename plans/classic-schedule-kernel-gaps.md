# Classic schedule: experiments to run and gaps to fill after the kernel comparison

Written 2026-11-14 against main `72fd8cb52`. It follows the kernel comparison in the classic schedule's architecture
doc (#1416) and a read of NInfer's source (`Neroued/ninfer` at `d9ad9dea4`). The target is the one the Qwen3.8-27B
NVFP4 parity plan sets: native serving on one RTX 5090 against stock vLLM, one request at a time. That plan owns the
cells, the bars and the record flow; this one adds what the comparison found and orders it.

Every number below is from this session or the parity plan and is a starting point, not evidence. Measure again on
the card and the build in use before building on one.

## What the comparison says matters

| Where the time is | Today | Best known | Kind |
|---|---|---|---|
| Gated DeltaNet recurrence, 256 rows, one layer | about 1.06 ms recorded | FlashInfer 38 µs alone, 80 µs as vLLM calls it | gap |
| Prefill chunk width | 64 rows | NInfer 1,024 | experiment, then records |
| FP8 projections at wide chunks | FP8 weights, BF16 activations | encoded activations on tensor cores | gap |
| Causal attention, head size 128 / 256 | 1.18x / 1.37x PyTorch | FlashAttention-2 | gap |
| Output head in a verify step | three one-row launches, 1.3 ms | one launch | experiment |
| PTX build | untimed on this model | nvcc's build | experiment |

NInfer serves this model at 6.3 ms TPOT and 0.62 s first token at 7,680 tokens (2026-10-29). It publishes no
comparison. Its lead comes from a kernel per row count and a two-launch recurrence, not from fewer launches: about 570
launches per forward, counted from source.

## Experiments

Each is small, needs no compiler change unless it pays, and has a result that decides the next step.

- **E1. Time the PTX build on this model.** The parity plan's P0. Everything below reads its numbers.
- **E2. FP8 through the block-scaled instruction with a unit scale.** NInfer's source says the plain FP8 instruction
  accumulates at half rate on the 5090. Take one recorded FP8 product kernel's PTX, swap the instruction by hand in
  scratch, and time both in one run. Check the answer matches bit for bit first.
  - Faster by 1.2x or more: register the atom and re-record the FP8 rows.
  - Level: write the result into the comparison's FP8 section and drop it.
- **E3. FlashInfer's recurrence at 512 and 1,024 rows.** The scratch probe has 64 and 256. This gives the floor per
  layer at each chunk width A2 of the parity plan measures, so each width's gap is known before any record.
- **E4. Chunk size 16 inside the recurrence.** NInfer solves 16 x 16 on plain cores; ours is 64. The trace cannot go
  below 64 today (the pad of non-zero width). Time the triangular solve alone at 16 and 64 from a hand-written body
  before fixing the trace. Only a clear win at 16 earns the trace work.
- **E5. A kernel per row count.** NInfer uses a different product kernel at 1, 2-5, 8-64 and 1,024+ rows. For the
  gate/up and head weights, pin the warp-split-over-K shape at 8, 32 and 64 rows against the recorded row, in one
  run each. A win is a row to record; a pin that does not realize is a gap to name in the comparison.
- **E6. The three-row head.** The parity plan's B3: trace the W4A4 head at three rows and measure. Worth about 0.8 ms
  a verify step.
- **E7. A shortlist head for the drafter.** NInfer ranks the draft over 131,072 frequent tokens; the target keeps the
  full head. Measure acceptance and the drafter's step with a shortlist built from the tokenizer's frequencies. This
  goes past stock vLLM, so it runs only after the bars are met or if B of the parity plan stalls.
- **E8. FlashAttention-2's own tile at head size 256.** Pinned, it realizes at 37 µs with 255 registers and a spill;
  the recorded row is 40.2 and PyTorch 29.4. Read the PTX for the spill. If the register count is the PTX compiler's,
  that is its speed work, with this kernel as the case.

## Gaps to fill

Ordered by what they return on this model. Each is a cut, a schedule or lowering coverage; none is a fusion bound.

1. **The recurrence, expressed as FlashInfer's algorithm.** Its own section below. This is most of the prefill gap
   and most of the verify step (14.3 of 21.6 ms).
2. **An FP8 tile that takes an encoded activation.** Declaring the layers W8A8 today picks scalar schedules (6.6 ms
   against 0.15). It pays only at wide chunks, so it follows A2's width decision. E2 decides which instruction it
   uses.
3. **Causal attention.** Skipping masked chunks cannot pay on a one-wave launch. What is left to try: the causal
   kernel's tile at long windows, and E8's register count. Gate: at or under PyTorch at head sizes 128 and 256.
4. **A row held in shared memory for a cooperative reduce.** A `STAGE` pin on one does not realize, so softmax and
   RMSNorm read the row from global memory once per statistic (8.1 against 7.3 µs, 8.5 against 7.6). Small per
   launch, but these run in every layer.

Tooling gaps that cost time in this session:

- **FlashInfer is not a baseline of `emmy run --bench`.** The bench compares with eager PyTorch and `torch.compile`
  only, so every FlashInfer time in the comparison came from a scratch probe. Add it as a third baseline for the
  programs it has a kernel for (the recurrence, norms, sampling, paged attention), stored with the program like the
  other two. E3 then needs no probe, and R4's gate reads from the golden.
- **A pin that does not realize fails silently.** The run reports a time with empty knob columns. Make it an error.
- **`emmy golden list` prints a program's first row,** which can be slower than the row a compile deploys. Print the
  fastest, or both.
- **Stale whole-program latencies in the 5090 hardware golden.** They go with the next re-record of that file, not
  before.

Gaps the comparison names that this model does not need. Leave them until a model does:

- top-k, sort and sampling; scatter-add; a scan across CTAs;
- Stream-K; a persistent launch and ping-pong on Hopper; any `tcgen05` atom;
- 4-bit integer weights; a resident mixture-of-experts kernel; async copies of a paged operand;
- Winograd and FFT convolution, which are other algorithms and not schedules.

## Expressing FlashInfer's recurrence

The goal is that a row of knobs spells the reference kernel, so the search can find it and other recurrent models get
it too. Nothing here calls FlashInfer's code.

What the reference does that Emmy cannot say today, and where each part belongs:

| Part of the algorithm | Today | Where it is fixed |
|---|---|---|
| Inverse of the chunk's unit lower-triangular matrix by merging blocks, each merge two small products | traced as a 63-step forward substitution in f32 | a rewrite pass with a knob, so evidence decides it; never normalization |
| The chunk's work in one launch per head | nine kernels | unknown until diagnosed: a cut picked without evidence, or a region the splicer cannot build |
| The 128 x 128 state kept in the CTA over every chunk | registers only, and only for a loop of maps and additive products; else one launch per step | the schedule: a carried state held in shared memory, and the chunk loop as a serial axis of the launch |
| Warp groups with different work in one CTA, under barriers | only the load warp | the schedule: a new `WORK` spelling |
| The program at 64 rows | does not compile; one body's normal form does not settle | a compiler bug to fix first |

The inverse does not read the state. So the per-chunk products and the inverse can run for every chunk at once across
the grid, and only the state update is serial. NInfer uses this: two launches a layer, no warp groups with roles. That
is the shape to build first, because it needs one new schedule concept instead of two.

Steps:

- **R0. Fix the 64-row compile.** Reproduce the body whose normal form does not settle and fix it. Gate: the 64-row
  recurrence compiles and matches the reference.
- **R1. Say why the chunk is nine kernels.** Compile the 256-row recurrence with every seam pinned to fuse. If it
  builds, the nine are picks, and the fix is evidence: record the fused arm. If it does not, name each refusal; by
  the fusion invariant each is a bug to raise. Gate: a list of the nine boundaries, each with its cause.
- **R2. The blocked inverse as a rewrite.** A pass that recognizes the forward substitution of a unit
  lower-triangular system and offers the block-merge form, whose merges are products the schedule already tiles.
  E4 decides the base block size (8, as the reference, or NInfer's direct 16 x 16 on plain cores). Gate: the inverse
  alone at 64 x 64 within 2x of the reference's share; bit pattern checked against the substitution in f32 within the
  usual tolerance.
- **R3. A carried state in shared memory.** Extend the schedule that keeps a state in registers across a static loop:
  the state may live in shared memory, and the loop body may hold the products the update reads (`k S`, `q S`,
  `k^T u`). The chunk loop is a serial axis of one launch per head. This is the new concept; write its design into
  the schedule architecture doc and get it reviewed before building. Gate: the state update over 256 rows as one
  launch, state never written to global memory between chunks.
- **R4. Two launches, measured.** R2's kernel over all chunks, then R3's. Gate: one layer at 256 rows within 3x of
  FlashInfer as vLLM calls it (240 µs); tokens unchanged on the probes; the rows recorded in the recipe golden.
- **R5. One launch, only if R4 leaves a gap worth it.** Warp groups with roles and barriers, and the two kernels
  fused. Measure R4 against the FlashInfer baseline first; build this only for a gap that the baseline shows is in
  the launch boundary or the idle warps.
- **R6. The decode step.** The reference streams the state in tiles through a two-stage pipeline. Check whether the
  width-3 step needs more than R3 plus an existing `STAGE` row before adding anything.

## Order

- **No card:** the FlashInfer baseline in the bench; E4's hand-written body; E5's and E6's traces and lowering
  checks; the silent-pin error; R3's design written for review.
- **Dev box:** E2, E3, E5, E8. Each fits the two-minute budget, in capped units.
- **Rental, with the parity plan's first:** E1, then E6's record.
- **Then:** R0 and R1 (no card), R2 to R4, then A2's width, then gap 2 if the width needs it, then gaps 3 and 4.

## Risks

- **The recurrence is the large one and its size is unknown until R1.** R3 is a new schedule concept; its design
  goes to the owner before any code.
- **NInfer's half-rate claim is a source comment,** not a measurement of ours. E2 may show nothing.
- **A pinned win may not survive as a deployed row.** Compare inside one run, and record only what the same-run
  comparison supports. This session's false softmax win came from comparing a pin with a whole-program latency.
