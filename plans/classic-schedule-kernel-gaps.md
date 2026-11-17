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

Run 2026-11-14 on the dev-box RTX 5090 (CUDA 13.0), each comparison inside one run or one session. Results that
last are written into the kernel comparison in the classic schedule's architecture doc.

| | Experiment | Result | What follows |
|---|---|---|---|
| E1 | The PTX build on this model | Six programs of the recipe golden, kernels only: five within 4% of their stored time (59/60, 51/52, 449/433, 13/13, 300/296 µs), one faster (239 against 290). A seventh timed out in compile. No serving cell was measured | The parity plan's P0 still owns the four cells |
| E2 | FP8 through the block-scaled instruction, unit scale | Twice the rate, bit-equal, for `e4m3` and `e5m2`. Real tiles 1.2 to 1.5x | **Done in this PR,** in the PTX compiler; no atom. The golden's FP8 rows keep old times |
| E3 | FlashInfer's recurrence at wide chunks | 63 µs at 64 rows, 87 at 256, 145 at 512, 157 at 1,024 | The floor for R4 and for each width of A2 |
| E4 | Block size 16 inside the recurrence | Substitution over 256 rows: 31 µs in blocks of 16, 1,837 in blocks of 64 | R2 below changes: a chunk of 16 needs no blocked inverse |
| E5 | A kernel per row count | Level with cuBLAS at 32 and 64 rows; at 8 rows a pin beats the pick 28 to 19. Warps cannot share one K range | A new gap (5 below). The 8-row pick is missing evidence |
| E6 | The three-row head | Not run: it needs the model's trace | The parity plan's rental |
| E7 | A shortlist head for the drafter | Not run: by its own condition it waits for the bars | — |
| E8 | FlashAttention-2's tile at head size 256, causal | 46 µs against the deployed 39, PyTorch 33. Both at 255 registers with a spill | The tile is not the gap. One instruction row is 128 accumulator registers; see gap 3 |

Two corrections to the first draft of this plan:

- E8 named the wrong program. The 37 µs pin with a spill was head size 128, where the deployed row is 13.
- `emmy golden list` does print every measured row. Only the ratio to `torch.compile` reads the whole-program time,
  which is stale on some programs. That is data to re-record, not a tool to fix.

## Gaps to fill

Ordered by what they return on this model. Each is a cut, a schedule or lowering coverage; none is a fusion bound.

1. **The recurrence, expressed as FlashInfer's algorithm.** Its own section below. This is most of the prefill gap
   and most of the verify step (14.3 of 21.6 ms).
2. **An FP8 tile that takes an encoded activation.** Declaring the layers W8A8 today picks scalar schedules (6.6 ms
   against 0.15). It pays only at wide chunks, so it follows A2's width decision. E2 settled the instruction: the
   block-scaled form.
3. **Causal attention.** Skipping masked chunks cannot pay on a one-wave launch, and E8 rules the tile out. What is
   left is the code of one key block and, at head size 256, the register ceiling: an accumulator of 16 by 256 values
   takes 128 registers a thread. An accumulator kept in shared memory, or a product split over the head, would lift
   it. Gate: at or under PyTorch at head sizes 128 and 256.
4. **A row held in shared memory for a cooperative reduce.** A `STAGE` pin on one does not realize, so softmax and
   RMSNorm read the row from global memory once per statistic (8.1 against 7.3 µs, 8.5 against 7.6). Small per
   launch, but these run in every layer.

5. **Warps that share one K range.** For 8 to 64 rows a hand-written kernel gives each warp a slice of K and adds
   the accumulators at the end. `REDUCE=coop` takes threads only. Gate: a row that spells it, and a time under
   cuBLAS at 8 rows.

Tooling gaps that cost time in this session:

- **FlashInfer is not a baseline of `emmy run --bench`.** The bench compares with eager PyTorch and `torch.compile`
  only, so every FlashInfer time in the comparison came from a scratch probe. Add it as a third baseline for the
  programs it has a kernel for (the recurrence, norms, sampling, paged attention), stored with the program like the
  other two. E3 then needs no probe, and R4's gate reads from the golden.
- **A pin that does not realize: fixed in this PR.** The check existed. It missed a `WORK` pin on a kernel that took
  no schedule, and two pins written with a space instead of a comma were read as one value. Both are errors now.

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
  E4 says to try a chunk of 16 first: its plain substitution costs 31 µs for 256 rows, under the reference's whole
  kernel, so R2 may reduce to tracing the rule at 16 (the trace cannot go below 64 today). Gate: the inverse
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

### What the first steps found (2026-11-15)

Dev-box RTX 5090, 48 heads x 128, 256 rows, one session. FlashInfer's whole kernel is 87 µs on this shape.

- **R0 done.** The 64-row program compiles. The fault was in the value numbering: it told `a - b` from `-b + a`,
  which the term order picks between. A sum is now one node over its terms and offset.
- **R1 answered: the boundaries are picks.** Every seam fused gives 9 kernels, every seam cut 17. No refusal.
- **The state update is already fast.** The chunk loop is one serial launch on the register schedule: 25 µs for
  four chunks of 64, 35 µs for sixteen chunks of 16. R3 as written (a carried state in shared memory) is not needed
  for the 3x gate.
- **The time is in four products around it** (1,388, 605, 142 and 140 µs), each a fused region whose cut is never
  picked. A cut pinned on a slow kernel by its name is correct alone, and its pieces take bf16 tensor-core tiles.
- **A respelled rule plus cuts ran 268 µs, and 195 after #1411's scan fix**, correct against eager. The respelling
  moves every pointwise scale onto the small 64 x 64 matrices, so q, k and v stay plain bf16 operands; it matches
  Transformers' rule in PyTorch (state within 9e-8). The substitution is 53 of the 195 (`t256`, `coop-t`, shared
  carry). It is now the traced rule.
- **Chunk 16 did not pay as traced.** The stacked start states become a 16-way select copy (169 µs), and reading
  the output inside the chunk loop leaves fifteen separate products in one kernel.

Faults found on the way, each a compiler bug to raise:

- An inverse written as repeated squaring is rolled into a serial kernel; at a chunk of 64 its compile takes 25 GB.
- A bare `PLACE=cut` on the respelled program gives wrong results. Each cut pinned alone is correct.
- A cut pinned on a serial kernel does nothing and is not reported.

What follows, in order:

1. Done: the respelling is the traced rule. The trace takes it only above one chunk of 64 rows, so it re-keys the
   256-row programs alone; today one golden holds them (the NVIDIA 27B checkpoint's, RTX 5090).
2. Done: the 256-row roots are recorded on the RTX 5090. Six roots sum to 474 µs against 524 before. The root
   that applies the inverse got slower (279 to 313): its scaled products cut into pointwise kernels and scalar
   products that refuse a tensor-core tile. That root is the next thing to fix.
3. The substitution, 53 µs: a tuned row first (it ran 31 µs in blocks of 16 alone), then R2's blocked inverse.
4. R5 only for what is left against 87 µs.

## Order

- **No card:** the FlashInfer baseline in the bench; R0 and R1; the trace of the recurrence at a chunk of 16; R3's
  design written for review.
- **Dev box:** evidence for the 8-row product pick; R2 at 16.
- **Rental, with the parity plan's first:** the four cells on the PTX build with the new FP8 instruction; the FP8
  rows of the hardware and recipe goldens measured again; E6.
- **Then:** R3 and R4, then A2's width, then gap 2 if the width needs it, then gaps 3 to 5.

## Risks

- **The recurrence is the large one and its size is unknown until R1.** R3 is a new schedule concept; its design
  goes to the owner before any code.
- **A pinned win may not survive as a deployed row.** Compare inside one run, and record only what the same-run
  comparison supports. This session's false softmax win came from comparing a pin with a whole-program latency.
