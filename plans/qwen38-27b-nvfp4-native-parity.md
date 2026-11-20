# Qwen3.8-27B-NVFP4 on one RTX 5090: what is left for native serving to match stock vLLM, one request at a time

Written 2026-11-12, updated 2026-11-14 against main `94bb6d8c5`, from the recipe DISCOVERY's 2026-11-12 section (one
card, native and stock in one session, #1392), the 256-row records of #1397, and the move to Emmy's own PTX compiler
(#1399). This plan covers serving one request at a time. Concurrency stays in the native serving parity plan (its
milestones 19-25); its design is written in the native serving architecture doc and waits for the owner's approval.
That plan also keeps Qwen3.6, the follow-ups ledger and the switch.

The recipe is `Qwen3.8-27B-NVFP4-NVIDIA`: NVIDIA's checkpoint, the MTP drafter at two tokens, an FP8 KV cache, the
262,144-token window.

## Where it stands

One rented RTX 5090 (CloudRift, CUDA 13.3), one session, native and stock vLLM 0.32.0 both, the pack exported under
strict evidence from the golden alone. The bar is the older plan's, read against this session's stock: TPOT within
1.2x stock at the short prompts and 1.5x at the longest prompt both serve, first token within 2x.

| prompt | native TPOT | stock | bar | native first token | stock | bar |
|---|---|---|---|---|---|---|
| 256 tokens | 9.84 ms | 7.87 ms | 9.44, **missed by 4%** | 0.143 s | 0.159 s | 0.32, met |
| 7,680 tokens | 8.72 ms | 7.73 ms | 9.28, met | 4.37 s | 0.753 s | 1.51, **missed, 5.8x stock** |
| 197,632 tokens | 11.4 ms | 9.54 ms | 14.3, met | 172 s | 67.9 s | 136, **missed, 2.5x stock** |
| 261,632 tokens | 11.4 ms | does not fit | serves | 251 s | does not fit | serves |

Standing conditions, all held: GSM8K 0.725 / 0.755 strict / flexible match, device memory at rest 27.9 GiB (stock
30.4), the full window serves, the pack exports under strict evidence.

**Every time in this plan was measured on nvcc's build.** Main now builds every kernel with Emmy's PTX compiler, which
has had no speed work yet. On the corpus one NVFP4 kernel has 574 machine instructions against nvcc's 216 (34 branches
against 2: a loop left rolled), and one matmul uses 255 registers against 174. A new code generator re-keys no row, so
the golden's times still read as evidence. Nothing of this model has been timed on the PTX build (P0).

Where the time goes, from `nsys --cuda-graph-trace=node`, each launch priced at its kernel's median:

| launches | 64-row chunk | verify step, window up to 8,192 | verify step, 197,632 |
|---|---|---|---|
| Gated DeltaNet layers (48) | 27.7 ms | 14.3 ms | 14.4 ms |
| attention layers' `pre` and `post` (16 each) | 5.8 ms | 3.9 ms | 4.0 ms |
| attention | 1.2 ms | 0.35 ms | 4.5 ms |
| output head | 0.45 ms, one row | 1.3 ms, three one-row launches | 1.3 ms |
| the drafter's layer and prologue | 0.44 ms | 1.4 ms | 1.9 ms |
| state carry, rotary store, unfold, the rest | 0.3 ms | 0.3 ms | 0.3 ms |
| sum | 35.9 ms | 21.6 ms | 26.4 ms |

- The 7,680-token first token is 120 chunks at 36.4 ms each, host time included.
- The 197,632-token first token is 171 s of kernels: 86 s in the Gated DeltaNet layers, 62 s in attention, 18 s in
  the attention layers' halves.
- A verify step keeps 2.26 tokens in this session (2.43 on 2026-11-07); acceptance read 63-77% across the prompts.

## Done since 2026-11-12

- **One cache read per verify step (#1392).** The step's rows fold beside the query group. Gate met: the verify step
  at 197,632 positions is 4.8 ms above the step at a window of up to 8,192 (bound 8 ms; it was 14.4). TPOT at 197,632
  tokens fell from 14.7 to 11.4 ms.
- **The golden measured again (#1392):** 375 measured rows of 412, after #1365 had left 225 of 415.
- **The 256-row Gated DeltaNet program traces and records (#1398, #1397).** The scan now carries the state and the new
  values only and the outputs are read after it, so the program no longer asks for a 73 GiB workspace. 25 of the 28
  256-row targets are recorded. A Gated DeltaNet layer takes about 1.06 ms at 256 rows: about 4 µs a row against 9 µs
  at 64 rows. That replaces the two-point estimate (15.6 ms a chunk plus 0.32 ms a row for the 48 layers), which put
  them at 98 ms a 256-row chunk; the records sum to 51 ms.
- **A pin addresses a kernel by its name alone (#1400),** so a nested piece takes its own schedule. The tie between a
  parent's pin and its child's is gone.
- **The runtime takes a wider chunk (#1392):** `--prefill-size N` at export; the pack keeps the plugin's chunk and the
  decode width for the tail, and the recipe names the width (`SERVE_NATIVE_PREFILL`).

## The gaps, largest first

1. **First token at 7,680 tokens: 5.8x stock, the bar 2x.** Prefill cost per token (A).
2. **First token at 197,632 tokens: 2.5x stock.** The same per-token cost, beside 62 s of attention (A).
3. **TPOT at 256 tokens: 4% over the bar.** The verify step is 21.6 ms; stock's drafted step was 18.1 ms on
   2026-11-01, another card (B).

## P0. Time the PTX build first

Measure the four cells again on main and capture the 64-row chunk and the verify step as above. Compare each launch
class with its 2026-11-12 time.

- If nothing moved past the session's noise, every number here stands.
- A launch class that got slower is the PTX compiler's speed work, named with its kernels, not this plan's.
- If times moved, measure the golden's rows again on the PTX build before any record of A or B, so old and new rows
  come from one code generator.

Gate: the cell table and both captures on the PTX build, each launch class beside its nvcc time.

## A. Prefill

The 7,680-token cell allows 1.51 s: 0.196 ms a token for everything. Stock spends 0.098 ms a token.

What the 256-row records say. They are rows measured one kernel at a time; no 256-row chunk has been captured.

- **A 256-row chunk cannot pass the 7,680-token cell alone.** Its Gated DeltaNet layers are 48 × 1.06 ms = 51 ms a
  chunk, 0.199 ms a token: 1.53 s at 7,680 tokens before anything else runs.
- **512 rows is at the edge.** A line through the 64- and 256-row layers (577 µs and 1.06 ms) puts the layers near
  1.2 s at 7,680 tokens with a 512-row chunk. That leaves about 0.3 s for everything else, which took 0.98 s at 64
  rows and is mostly weight reads. The two points are different programs (four roots and twelve), so this is an
  estimate only.
- **The 197,632-token cell closes with a 256-row chunk.** The layers fall to about 39 s (86 s today). If attention
  costs per token what it does at 64 rows (62 s) and the halves no more than today's 18 s, the cell is at most about
  125 s against 136 s.

Steps:

- **A1. Finish the 256-row chunk.** What #1397 left:
  - **The combine row, fixed in code first, off the card.** A record stores a split attention's combine row at the
    stored program's hint; the export looks it up at each window range's own size. The split arm then goes unpriced,
    the unsplit kernel wins, and nothing fails: on 2026-11-12 a pack ran its decode attention unsplit (389 µs a layer
    against 11). #1392 patched the rows by hand. Gate: a test where a split recorded at a window prices at that
    window's range.
  - **The two 256-row attention targets.** As picked they take 7.2 ms a layer. Record their splits at each window
    range, as the 64-row twins have them.
  - **One `post` kernel whose pick hangs.** Under the 64-row twin's route its record failed earlier with an unusable
    wrong-answer reference; start there.
  - **Then the pack:** strict export at 256 rows, the strict serving audit, one 256-row chunk under nsys written as the
    table above with a 256-row column, and the first token at 7,680 and 197,632 tokens.

  Gate: tokens unchanged on the probes against the 64-row pack; device memory at rest not above stock's 30.4 GiB; the
  full window serves; the captured chunk's kernels within 10% of the measured chunk; the 197,632-token first token at
  most 136 s, read as its terms.
- **A2. Measure 512 rows; do not extrapolate.** Also 1,024 if the activations fit. Off the card first: the program
  lowers at sm_120, asks for no oversized workspace, and keeps the recurrence across its 64-token pieces as one roll.
  Record as below. Gate: the per-kernel table at each width and the 7,680-token first token each width implies.
- **A3. The per-row launches at the kept width.** Scope comes from A1's and A2's tables. Known candidates:
  - **The Gated DeltaNet layer's kernels:** sweep each one's neighbouring schedules at the wide width
    (`emmy run --golden --kernel --bench`). By the fusion invariant a launch no offered schedule brings down is a cut,
    a schedule or lowering coverage.
  - **The FP8 layers.** The checkpoint is modelopt MIXED_PRECISION: the Gated DeltaNet in and out projections and
    attention q/k/v/o are FP8 W8A8 with static scales. Emmy runs them as FP8 weights with BF16 activations, because the
    loader recognizes static FP8 only under `quant_method: fp8`. Declaring them made the compile pick scalar schedules
    (o_proj 6.6 ms against 0.15 ms): no e4m3 tensor-core tile takes an encoded activation. That tile is compiler work
    first. It pays only at wide chunks, where these layers stop being bound by their weight read.

  Gate: the kept width's captured chunk implies at most 1.51 s at 7,680 tokens; its rows are in the recipe golden.
- **A4. Gate.** First token at 7,680 tokens at most 1.51 s; at 197,632 at most 136 s; the 256-token cells unchanged.

How a wide twin records (#1397):

- Record it inside a working copy of the recipe golden: its record compiles the narrower twins first, under strict
  evidence.
- Read a fresh root's pieces before recording its greedy pick. The prior's picks on the new Gated DeltaNet roots were
  far off: the triangular solve picked at 21.7 ms, recorded at 0.14 ms.
- A 60-schedule tune of the kernel fixed every slow pick; none needed a hand cut. Commit the tune's freeze beside the
  card's.
- An unmeasured cut loses even when it is right: its pieces need their own rows before the totals compare.

The decision A2 and A3 can force: if 512 rows and the sweeps leave the 7,680-token cell above 1.51 s, what is left is
the FP8 tile or other compiler work of unknown size. The owner then chooses between sizing that work and accepting the
cell above 2x with its floor written down. The 197,632-token cell does not wait on that choice.

## B. The short-prompt step

The 256-token TPOT passes with a verify step of at most 20 ms at the 4,096 window: 20.7 ms holds the bar at this
session's 2.26 tokens a step. Equal to stock at that acceptance is about 17 ms.

- **B1. Split the 14.3 ms again, kernel by kernel,** from P0's capture. The last split by piece is #1282's (a 6.9 ms
  floor) and predates #1289, #1295, #1318, #1322 and #1329.
- **B2. The width-3 Gated DeltaNet step toward its floor:** the width-3 weight readers, the recurrence's seed read,
  and the in-place state read only if those miss. Cold-checked rows only.
- **B3. The output head once a step.** The coded head runs at one row, so a verify step launches it three times: 1.3
  ms against 0.45 ms for one row. A three-row program of the old weight-only head was offered only bf16 tiles (1.68 ms
  at best). The W4A4 spelling now offers the e2m1 tile, which holds three rows in one tile. Trace and measure a
  three-row W4A4 head twin; near the one-row time it gives back about 0.8 ms a step. Gate: tokens unchanged on the
  probes.

Gate: the verify step at most 20 ms at the 4,096 window; TPOT at 256 tokens at most 9.44 ms; the long cells unchanged.

## C. Closing

- **C1. One protocol session, all three lanes.** Stock, the plugin and native, every cell, one card. #1392's session
  measured stock and native, not the plugin. Gate: every bar cell green, or each miss named with its measured floor for
  the owner to decide.
- **C2. Promotion.** The rows A and B record go into the RTX 5090 hardware golden with the five kernel trees the older
  plan's milestone 17 still holds back; both priors refit once; the `prior-pending` tag dropped. The three compiler
  correctness items of the older plan's ledger land first.
- **C3. The native image** for one-request serving (the older plan's milestone 18). The switch, its milestone 26, needs
  concurrency and stays there.

## Order and card time

- **No card, now:** the combine-row fix in code; A2's lowering check at 512 and 1,024 rows; B3's three-row head twin
  traced and lowered.
- **Rental 1:** P0; A1's records, pack, capture and first token; B1's split from P0's capture.
- **Rental 2:** A2's records and capture; A3's sweeps; B2's and B3's records; the A4 and B gates.
- **Rental 3:** C1.

One golden-writing change at a time. If P0 says the golden must be measured again, that lands before any A or B record.

Rented 23.5 GiB hosts and CPU-capped containers have the traps the `onboard-model` skill lists; record runs have the
ones the `refresh-golden` skill lists.

## Risks

- **The PTX build may be slower on this model.** Then the numbers here move before any work in this plan does, and
  part of the gap belongs to the PTX compiler's speed work.
- **512 rows may land at the bar with no slack,** and the 7,680-token cell may need the FP8 tile.
- **A wide program is new ground for the prior.** Its picks on new roots were up to 150x slow; every new root's pieces
  are read before a greedy pick is recorded.
- **A wide chunk's activations are unmeasured above 256 rows.** Device memory at rest must stay under stock's 30.4 GiB.
- **Acceptance moves the short cell:** 2.26 tokens a step in this session, 2.43 on 2026-11-07. The 20 ms step gate
  holds at the lower one.
