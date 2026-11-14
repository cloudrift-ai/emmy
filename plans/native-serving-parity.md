# Native serving to parity with vLLM, and the switch

Goal, from the owner: "Have native serving working comparable to vllm so we can switch to it completely." The native
server replaces the vLLM plugin lane for the two maintained RTX 5090 recipes — `Qwen3.8-27B-NVFP4-NVIDIA` and
`Qwen3.6-35B-A3B-FP8` — once it meets the bar below, serves concurrent traffic, and ships through the recipes' images.
The 27B gates first: its gaps are kernel-level and the closer race; Qwen3.6's largest gap is structural (half its
experts live off the card). Continuous batching lives in this plan: its design and mechanics run beside
single-request parity, and its closing promotion follows parity's.

Written 2026-10-29 from the measured comparison tables in both recipes' DISCOVERYs (the stock-and-plugin session and
the NInfer session, merged through #1304), `emmy/serving/native/ARCHITECTURE.md`, and the open PRs. Revised the same
day, after #1289, #1295, #1303 and #1304 merged and the RTX 5090 tuning plan this plan grew out of was deleted: its
still-open items are the follow-ups list below, and its record-run traps are in the `refresh-golden` skill. Each
milestone here is one PR, stacked on the one before where they touch the same files; a measurement milestone is a
DISCOVERY PR.

## The bar

"Comparable to vLLM" means, on the DISCOVERY protocol (one rented x16 RTX 5090, one session per model, one request at
a time, FP8 KV cache, the MTP drafter at two tokens, the recipe's 262,144-token window, the same prompts and client,
GSM8K 200 questions 5-shot), against stock vLLM 0.32.0 measured in the same session:

- **TPOT** median within **1.2x stock** at the 256- and 7,680-token prompts; within **1.5x stock** at the longest
  prompt both serve (197,632 for the 27B; the window for Qwen3.6).
- **Time to first token** median within **2x stock** at every prompt length both serve.
- **Floor for retiring the plugin lane:** native matches or beats the plugin lane in every cell, the concurrency
  cells milestone 1 measures included. This is the hard condition for the switch; the stock margins are the goal.
- **Standing conditions, every gate:** at concurrency one, the full 262,144-token window still serves on both models;
  device memory at rest not above the session's stock column at the same `--max-num-seqs`; GSM8K no more than one
  standard error below the plugin lane's same-protocol score; probe tokens held or the difference explained by a
  top-two margin.

In today's numbers the bar is: 27B TPOT <= 9.2 / 9.8 / 13.1 ms at 256 / 7,680 / 197,632 tokens, first token
<= 0.24 / 1.28 / 136 s; Qwen3.6 TPOT <= 13.2 / 14.6 / 13.7 ms, first token <= 0.58 / 4.4 / 244 s. NInfer's columns
(27B 6.3 ms / 0.050 s; Qwen3.6 1.75 ms / 0.042 s) are not the bar; they are proof the card has the headroom —
12,300 tokens a second of 27B prefill, a 15.5 ms three-row verify step — at these quantizations.

At concurrency the window is a shared budget, not a per-request promise: the cache allocation stays window-sized, and
an admitted request's context is that budget over the admission cap — 32,768 tokens at concurrency 8, 8,192 at 32 —
unless milestone 19's design picks another split. The concurrency bar, accepted by the owner on 2026-11-02 against
milestone 1's columns: per-request TPOT and total output tokens per second each within 1.5x stock at concurrency 8
and 32. Its numbers are under "Measured 2026-11-01".

## State (2026-10-29)

Measured medians at a 256-token prompt, from the two comparison sessions (both on main at `b5dddf26c`):

| TPOT / first token | native | plugin | stock vLLM 0.32.0 | NInfer |
|---|---|---|---|---|
| Qwen3.8-27B-NVFP4 | 13.7 ms / 1.34 s | 13.5 ms / 0.23 s | 7.7 ms / 0.12 s | 6.3 ms / 0.050 s |
| Qwen3.6-35B-A3B-FP8 | 19.9 ms / 5.07 s | 77.3 ms / 3.44 s | 11.0 ms / 0.29 s | 1.75 ms / 0.042 s |

Stock is faster in every timing cell both serve, on both models. The full tables, flags and "where the time goes"
sections are in each recipe's DISCOVERY on main.

**The plugin floor's failing cells.** From the 27B table, native loses to the plugin in four long-window cells:
TPOT 32.3 against 17.3 ms at 197,632 tokens and 41.7 against 16.0 at 261,632; first token 505 against 184 s at
197,632 and 815 against 266 s at 261,632. No landed PR touches them; milestones 5, 6 and 9 are their owners, and
they are the switch's hard blockers. Native also trails the plugin's first token at 256 tokens (1.34 against 0.23 s)
and 7,680 (7.78 against 5.31 s) — milestones 4 and 9 own those — and its 256-token TPOT by 0.2 ms, which the merged
step work below already more than covers. On Qwen3.6 the plugin floor fails only at first token (5.07 against
3.44 s at 256; 151 against 101 s at 7,680; 5,774 against 3,581 s at the window) — milestones 13, 14 and 15 own
those — while native beats the plugin's TPOT and GSM8K everywhere.

The diagnosis in brief:

- **27B decode** is the width-3 verify step: 32.3 ms against stock's ~18 and NInfer's 15.5, both sides accepting
  2.45 tokens a step. #1282 (merged) profiled it — two thirds in the width-3 Gated DeltaNet program — and took it to
  24.8 ms, drafted TPOT 12.1 to 9.2 ms on its probes; #1289 (merged) cut the one-row out projection's operand,
  38.5 -> 23.0 + 1.3 us a launch, drafter-off TPOT 17.05 -> 16.28 ms, drafted 9.19-9.41 ms on the probes. The probes
  flatter: their answers repeat and the drafter accepts 85% there against 68% on the protocol's text, so the same
  24.8 ms step reads ~10.5 ms TPOT on the protocol — close to the 9.2 ms cell, not yet at it. The remaining pieces:
  the coded head read once per row (3.1 ms a step; milestone 2), the three-row rotary store on one CTA (0.6 ms; 3),
  the attention partial scaling with the window (6), the width-3 weight readers at 1.0-1.4 TB/s and the rolled
  recurrence's seed read (7's tune), and the in-place state read (follow-ups, taken only if 7 misses).
- **27B long windows**: the step grows 33 -> 81 -> 97 ms at 7,680 -> 197,632 -> 261,632 positions (stock 18 -> 22).
  Native's attention reads each key-value head's cache once per query head — six times at this grouping — and the
  one-row partial scaling grows with the window (14 us a layer at 4,096 positions, 49 at three rows).
- **The 27B first-token budget is two terms.** At width W, first token is (prompt/W) chunks times the per-chunk cost
  plus the attention work, and the attention term does not depend on W. At 197,632 tokens: 3,088 chunks x ~50 ms
  ≈ 154 s of chunk cost, so attention is 505 - 154 ≈ 351 s (at the window: ~204 + ~610 of 815 s). Widening to 512
  rows alone leaves ~370 s; folding attention 3x alone leaves ~271 s; both give ~136 s — exactly the bar. The
  attention fold (5) is the bigger lever at the long prompts; the widening (9) carries the 256- and 7,680-token
  cells, where the chunk term is nearly all of the time.
- **27B prefill**: 1.01 s per 1,000 prompt tokens against the plugin's 0.69 and stock's 0.084 at a 2,048-token
  chunk. A 64-row chunk costs native ~50-55 ms: trunk 41 ms (no per-kernel profile — milestone 8), the rotary cache
  store 9 ms on one CTA (milestone 3, #1318, gives it a grid; the number waits on its gate); attention at 64 rows writes scores and probabilities to memory and grows with the window
  (4.6 ms a layer at 65,536 positions). #1295 (merged) stopped the Gated DeltaNet out projection being computed
  twice under the block amax and re-recorded both NVFP4 goldens: the 512-token attention input target 883 -> 655 us,
  the 64-row out projection piece 44.9 -> 35.7 us.
- **27B short-prompt first token is explained, not uncertain.** The vLLM session read 1.34 s at 256 tokens, the
  NInfer session 0.45 s at its 268-token chat prompt, and both are the carried-state chunk rule on the same pack: a
  chunk takes only prompt tokens before the last, so 256 tokens left 63 single steps (63 x 18.3 ms) where 268 left
  11, and the session's 257-token control measured 246 ms. #1281 (merged 2026-10-26) now takes the tail in
  decode-width chunks: 694 ms at 256 tokens, 6.83 s at 7,680. Milestone 4 (#1313) removed the remaining single
  steps; the number waits on its gate.
- **Qwen3.6 decode**: the one-row step is 19.6-19.8 ms and the drafter buys almost nothing — a verify step's three
  rows of eight picks route past the 8 device slots, and the drafted step reads 44-52 ms. The dev-box split (native
  ARCHITECTURE) put 7.8 ms of a step in the pager's table programs: the decision kernel runs one CTA, ~34 us for
  each of the ~160 paging runs (40 MoE layers x four paged buffers). The x16 split between the Gated DeltaNet
  programs, the expert programs and the paging programs is unmeasured — the NInfer session says so — and
  milestone 1 takes it.
- **Qwen3.6 prefill** is the structural gap: 50 tokens a second at 7,692 positions against stock's ~3,400 and
  NInfer's 19,600. A 64-row chunk reaches ~220 of a layer's 256 experts, so nearly every chunk copies most of the
  host half of every layer: 1.25 s a chunk, 94% of it the expert programs reading pages over the link (gate-up
  24 ms a layer — offered no tensor-core tile once paged — and down 10 ms). NInfer avoids the copies by fitting
  integer experts on the card (20.4 GiB); stock offloads only 10.5 GiB and prefills 2,048-token chunks.
- **Qwen3.6's copy floor, in numbers.** The host half is 15.9 GiB of pinned expert pages; a full-coverage chunk
  copies ~86% of it (~14 GiB), which the link moves in ~0.3 s at the ~48 GB/s the tower's block copies measured
  (0.77 GiB in ~17.5 ms). First token therefore counts full-coverage chunks: at 64-row chunks a 256-token prompt is
  four of them, ~1.2 s of copies against a 0.58 s bar cell; one 256-row chunk is ~0.3 s. The 7,680 and window cells
  need ~1,024-row chunks (7.5 chunks ≈ 2.3 s of copies against 4.4 s; 256 chunks ≈ 77 s against 244 s). This
  arithmetic is what decides 16's fallback question, and 14's gates are set against it.
- **Native's wins, protected by every gate:** the 27B's full 262,144-token window (stock refuses above 198,400);
  the least device memory on Qwen3.6 (24.5 GiB against ~29.8); GSM8K equal or better than stock on both (27B 0.710
  against 0.580; Qwen3.6 0.445 against 0.350); ready in 20-40 s from a pack against 181-236 s; the smallest host
  footprint on the 27B (3.4 GiB).

Merged since the tables: #1281 (tail chunks), #1282 (verify-step records, static prologue twins), #1288 (the
recurrence's double state write removed), #1289 (the out projection's operand cut; the audit names a piece's seams),
#1295 (contraction occurrences group operand-order-free; both NVFP4 goldens restamped and re-recorded), #1303 (a
pinned cut's remainder takes its own fork; a bare `PLACE=cut` cuts the remainder chain; `--pin-route` keeps a
recorded set), #1304 (the NInfer tables — its squash carried #1279's stock and plugin tables, so #1279 is closed,
not merged), #1293 (a cut arm's piece facts computed on the worker that builds the arm), and the tooling fixes
#1283-#1286. Milestone 1 measured the protocol once, below.

## Measured 2026-11-01 (milestones 1 and 8, #1311)

Each DISCOVERY's 2026-11-01 section holds the tables, on main at `ea17c0964`: before #1302 and #1309 changed both
recipe goldens, and with native packs that were not strict. The State above and the numbers quoted in the milestones
are the older sessions'; each gate re-reads its number against these.

The bar in these numbers: 27B TPOT <= 9.2 / 9.1 / 12.0 ms at 256 / 7,680 / 197,632 tokens, first token <= 0.17 / 1.26 /
135 s; Qwen3.6 TPOT <= 13.4 / 14.3 / 23.0 ms, first token <= 0.68 / 5.66 / 288 s.

The concurrency bar, 1.5x stock as the owner accepted it on 2026-11-02. At concurrency 8 / 32, TPOT at most and output
tokens a second at least:

| | 256-token TPOT | 256-token output | 7,680-token TPOT | 7,680-token output |
| --- | --- | --- | --- | --- |
| 27B | 13.8 / 15.2 ms | 504 / 713 | 29.0 / 36.3 ms | 184 / 187 |
| Qwen3.6 | 79.8 / 117.9 ms | 87.7 / 95.9 | 145.5 / 195.8 ms | 38.1 / 39.9 |

The plugin's cells are a low floor: it keeps the recipes' settings, which name one request, and it did not batch.

What the session changes for the milestones:

- **Qwen3.6's first token at 256 tokens went from 5.07 to 15.7 s, and no milestone owned the cause.** It is in the
  follow-ups below. The plugin floor at that cell is 3.54 s.
- **5's budget is larger.** Attention at 197,632 tokens is 544 s of the 27B's 698 s, not ~351 s of 505 s, and 6.7 ms a
  layer at 65,536 positions, not 4.6. The commit that moved it was not bisected. A 3x fold leaves ~181 s against 5's
  117 s; 117 s takes ~4.6x.
- **8 is done, and its floor says 9's gate cannot be met by width.** At 7,680 tokens attention is 1.09 s at any width,
  and 512 rows with no rotary cost at all still leaves 3.85 s against 1.26 s. Wider chunks stop paying near 256 rows.
- **11 and 12 stand, and 12's Gated DeltaNet branch does not apply.** The one-row step is 15.65 of 23.4 ms in the paging
  programs and page copies. The verify step is 44.5 of 57 ms in the expert programs and the pager, 5.5 ms in the Gated
  DeltaNet programs.
- **14's reference chunk moved.** Qwen3.6's 64-row chunk takes 1.44-1.49 s against 1.25 s, 97% expert programs. The card
  and the host memory settings both differ, and that was not separated.

## Measured 2026-11-12 (#1392)

The 27B DISCOVERY's 2026-11-12 section holds the tables: one rented x16 RTX 5090, one session, native and stock, the
pack exported under strict evidence. The plugin lane was not measured, and Qwen3.6 was not served (the host had too
little memory for its export).

The bar in these numbers: 27B TPOT <= 9.44 / 9.28 / 14.3 ms at 256 / 7,680 / 197,632 tokens, first token <= 0.32 /
1.51 / 136 s.

| 27B, native against the bar | 256 | 7,680 | 197,632 |
| --- | --- | --- | --- |
| TPOT | 9.84 ms, missed by 4% | 8.72 ms, met | 11.4 ms, met |
| first token | 0.143 s, met | 4.37 s, missed | 172 s, missed |

The full window serves (11.4 ms, 251 s), device memory at rest is 27.9 GiB against stock's 30.4, and GSM8K holds.

What the session changes for the milestones:

- **2-6 are met.** 6's gate is in its own entry below.
- **The three cells left belong to 7 and 9.** The 256-token TPOT is 7's: the verify step is 21.6 ms, 14.3 ms of it the
  Gated DeltaNet layers, and those prompts accept 2.26 tokens a step. Both first-token cells are 9's.
- **9 is two pieces of work, and width is the smaller one.** A 64-row chunk is 35.9 ms, 27.7 ms of it the Gated
  DeltaNet layers: about 15.6 ms whatever the width plus about 0.32 ms a row, an estimate from two programs, with no
  chunk above 64 rows timed. At that row cost a 256-row chunk gives about 2.9 s at 7,680 tokens, and no width gives
  less than about 2.5 s; the cell needs about 0.14 ms a row at 256 rows. At 197,632 tokens attention is 62 s at any
  width and a 256-row chunk leaves about 75 s for the rest: about 137 s against 136 s.
- **19's design is written** (`emmy/serving/native/ARCHITECTURE.md`) and waits for the owner's approval.
- **The golden has to be watched after a normal-form change.** #1365 left the 27B golden with 225 measured rows of
  415 and a strict export that stopped at its first fork. The file holds 375 of 412 now.

## The open PRs, placed

- **#1294**, beside **#1293** (both merged 2026-10-30): the placement prior prices a kernel set's work; a cut
  arm's piece facts computed on the worker that builds the arm. #1294 stops the prior keeping single-CTA arms
  that recompute a chunk's solve per
  output element (the tiny hybrid's 1e17-statement remainder; the 33-minute cold width-4 picks) and carried its
  featurizer bump and placement refit, reproduction gate green — but its own body says neither the old nor the new
  pick could be timed on the rented card (the width-16 bench worker dies with SIGSEGV, width-4 exceeds the 5,400 s
  wall budget), so it merged on the refit gate, not a speed claim. #1293 measured 575 -> 548 s (5%) on a 3-core
  host and did not measure the width-16 program it was aimed at. Neither is a milestone: evidence joins on exact
  identity, which neither changes.
- **#1290** (draft): the tiny GDN hybrid in the serving regen, capture and completion split apart, and the GPU
  tests it unblocks. #1303 (merged) was one blocker. #1294 (merged) closed part of the other, measured 2026-11-02 on
  a golden regenerated from main at #1310: the one-row programs run in 1.4 ms or less. The six chunk and verify
  programs whose kernels still hung: five were closed on 2026-11-03 by measured cuts recorded into the lane golden,
  program 44's only finishing set computes a wrong answer and stays out (the register has the numbers); the
  drafted-generation GDN test and the plugin's speculative engine test pass on the card. The switch (26) needs its
  tests green.
- **#1299** (open): lands before milestone 2, the first record milestone. It changes the cold bench and the tune DB
  version, so records taken before it would measure a bench about to move. **#1298** (merged 2026-10-30) changed
  kernel sources (identities unchanged) beyond NVFP4 through its wider alignment proof. Neither is on the 5090
  parity path (the 5090's deployed NVFP4 kernels read no pair table).
- **#1302** (merged 2026-11-01): milestone 18 — the native recipe lane, built and measured on packs prepared from
  main's goldens (each DISCOVERY's 2026-10-31 section), with both recipes' drafter rows on the RTX 5090 goldens; its
  images are not published and wait on 17's promoted rows.
- **#1309** (merged 2026-11-01): the retired tuning plan's re-record — 180 rows of both recipe goldens re-measured on
  a rented x16 RTX 5090 at the tune standard after the compiler fixes (0.918x the stored number at the median; the
  prologue64 `k_linear_mean_reduce_374241#2` row 494 -> 786 us, steady across runs). Its hardware-golden promotion
  came out before it landed: the ledger's 2026-10-31 prior item is why.
- **#1310** (draft): the promotion — every row and decision of both recipe goldens into the RTX 5090 hardware golden
  except the five kernel trees of the 2026-10-31 ledger item, both priors refit on the promoted goldens, the gate
  green; the recipes keep `prior-pending` until those five trees can follow.

## Milestones

1. **Baseline and the missing splits, one session per model, on today's main.** Re-run the DISCOVERY protocol on
   current main (post #1281/#1282/#1288/#1289/#1295/#1303): every cell of the comparison tables, stock and the
   plugin beside native. Add what the sessions left open: an nsys split of Qwen3.6's captured one-row and verify
   steps between the Gated DeltaNet programs, the expert programs, the paging programs and the page copies; the 27B
   first-token controls at 256 / 257 / 268 tokens confirming the chunk-rule story above; and the concurrency
   columns nobody has taken — stock and the plugin at concurrency 8 and 32, both models, `--max-num-seqs` raised,
   the protocol otherwise unchanged (per-request TPOT, total output tokens a second, first token, device memory).
   The plugin's concurrency cells are the floor for retiring it, and they do not exist today; the stock cells fix
   the concurrency bar's numbers with the owner. Every later gate compares against this table. Gate: the tables in
   both DISCOVERYs, one session per model; the Qwen3.6 step split in milliseconds per program class; the
   concurrency columns. Needs nothing.

2. **The 27B output head spelled W4A4, as the trunk's linears.** The coded head's measured schedule reads its codes
   once per row: 3.1 ms a verify step (1.86 ms at three rows plus two one-row drafts), 0.62 ms a one-row step.
   NVIDIA's checkpoint stores `lm_head.input_scale` and stock vLLM runs the head W4A4; spelled as the trunk's
   linears it takes the e2m1 tensor-core tile, ~0.45 ms a pass (#1282's bound), ~1.8 ms off the verify step. This
   reverses the head-stays-coded numerics choice of 2026-10-23 (#1247: logits equal to the decoded head within f32
   rounding), so the gate re-decides it on evidence. Gate: verify step down ~1.5 ms in the captured profile;
   teacher-forced logprob deltas on the probes reported; GSM8K no more than one standard error below the plugin
   lane's; rows recorded in the recipe golden. Needs 1, and #1299 in.

   The off-card half is in a draft PR, checked on the RTX 4080. The head is now the checkpoint constant
   `lm_head.weight` under the trunk's two spellers, so NVIDIA's stored `lm_head.input_scale` (0.02055, in a static
   4-bit group that names `lm_head`) makes it W4A4; its logits stay f32. At sm_120 lowering the head is two kernels, the
   rows' 4-bit quantize and the product, and the product reads as a block-scaled pair offered only the e2m1 tile.
   The native decode, prefill and verify programs build and run with it on surrogates. The 27B golden's `head-sym`
   program is the new trace, its two kernels stored with no rows, and the old head's two rows are gone; the hardware
   golden keeps its old head program and rows.

   Recorded and gated on a rented x16 RTX 5090 on 2026-11-05. The spellers had keyed the product at 512 rows (the
   coded rows lost the head's one-row hint), where the evidence picks a tile that runs 7 ms at one row; the head
   program now takes its hint again after spelling. At one row the product takes the e2m1 tile with its warps across
   the vocabulary (`w1x4`, `f1x8`, 433 us whole), a schedule a 24-row tune did not reach (scalar only, 964 us). Same
   card, main's pack against this one, both exported without strict evidence: verify step 51.3 -> 49.3 ms, its head
   3.3 -> 1.3 ms; one-row head 625 -> 448 us; plugin TPOT 11.1 -> 9.9 ms. Teacher-forced on main's greedy tokens the
   forced token's logprob moves 0.26 at most (median 0.0015), 250 of 256 picks agree; GSM8K on native 0.720 / 0.760.
   The strict audit deploys 51 of 62 twins: the 11 that do not are 3's, 4's and 5's native rows, not the head's.
   Without them a pack's decode attention takes the prior's uncut picks, 23-27 ms a step, so both steps above are
   about twice #1282's.

3. **The rotary cache store takes a grid.** The store names the whole cache (a scatter into a windowed buffer
   leaves its store no grid) and runs on one CTA: rope64 ~9 ms of a 64-row chunk (0.58 ms a layer), rope3 36-43 us
   a layer (0.6 ms a verify step), M-RoPE +122 us a layer and chunk. A lowering fix — the chunk's store spreads
   over the rows and heads it writes — so kernels re-key: restamp, re-record the rope rows, both recipes. Gate:
   the 27B chunk 50 -> ~41 ms measured in the step; the verify step down ~0.5 ms; corpus and goldens current.
   Needs 1.

   #1318 landed the lowering fix, checked on the RTX 4080. The one block did not come from the cache's size: the
   keys' and values' store reads its row from a term of its own, the ownership rule then refused the output-owning
   cut, and every recorded route ended in a copy kernel over two sibling sweeps on one block (rope64 495 us a layer,
   rope3 38-41 us). The row now rides with the store, and at sm_120 lowering every rope program of both recipes,
   M-RoPE included, gives each kernel a grid over the rows and heads it writes. Tokens are unchanged on the native
   surrogates; the corpus needed nothing. The restamp touched three goldens, all at the rope programs: the RTX 5090
   hardware golden (22 kernels re-keyed, 11 decisions and 19 rows dropped, 9 rows demoted), the 27B's (14, 6, 12, 6)
   and Qwen3.6's (8, 5, 8, 3); the one-row programs kept theirs. The dropped decisions are the cuts that left the copy
   kernel. Both priors are refit on the restamped goldens. #1324 recorded the fresh rope kernel sets of both recipes
   on the dev-box 5090; the hardware golden's rope rows are left to 17's promotion.

   Gated on a rented x16 RTX 5090 on 2026-11-06 (both DISCOVERYs), met. In the captured 27B steps the rotary store
   takes 0.09 ms of a 64-row chunk against 11.9 ms on 2026-11-01, and the chunk 38.6 ms at up to 4,096 positions
   against 54.1 ms. The verify step's three-row store takes 0.08 ms of the step against 0.6 ms. The packs before and
   after 3 and 4 give the same probe tokens on both models.

4. **Prefill ends at the prompt's last token (#1313).** After #1281 a 256-token prompt still runs 15 single steps
   on the 27B (3 on Qwen3.6): a chunk takes only prompt tokens *before the last*, because only the decode program
   runs the head. A drafted pack's chunks already run their last layer whole and the final norm; let the final chunk
   include the last prompt token and run the head and greedy selection over its last normalized row (a row pick,
   as the verify step's pick program does). 256 tokens then cost four chunks — the 257-token control's 246 ms.
   Gate: 27B first token at 256 tokens within one chunk's cost of the 257-token control; tokens unchanged on the
   probes; both models. Needs nothing.

   #1313 (merged 2026-11-02) landed the mechanics, checked on the native surrogates: every chunk now ends at its last
   row's logits and greedy token, and 256 tokens dispatch as four 64-row chunks on both models. #1324 recorded the
   row pick and the drafter-token program at each chunk width, so a strict-evidence bake of either recipe exports.

   Gated on a rented x16 RTX 5090 on 2026-11-06 (both DISCOVERYs), met. Same card, packs from the trees before and
   after #1313: the 27B's first token at 256 tokens went from 0.70 to 0.17 s, against 0.19 s for the 257-token
   control (0.15 against 0.17 s on main); Qwen3.6's went from 16.3 to 6.7 s, the control 6.7 s (6.1 against 6.2 s on
   main). Both trees give the same 64 greedy tokens on all four probes, on both models.

5. **Long-window prefill attention folds chunk by chunk.** The 64-row attention's recorded route is four pieces —
   the score GEMM to memory, the online statistics, the probabilities to memory, the P·V GEMM — and runs the value
   expectation far below bandwidth: 6.7 ms a layer at 65,536 positions, the 544 s attention term of the 698 s first
   token at 197,632 (the 2026-11-01 numbers). The fold is the softmax twist's expectation channel on tensor cores,
   FlashAttention's shape, which the tower's attention already takes as one kernel. The text attention never
   reached it, for three reasons found on 2026-11-04 by pinning the tower's schedule on the twin off the card, all
   three closed by #1321 (the attention twins of both recipes re-key; their rows record with this milestone's):
   - the twin rounded the normalized probabilities to the model dtype before the value product, so the expectation
     never joined the twist — the twin hands the FP32 probabilities to the product;
   - the causal mask compared the key coordinate against `position + arange(rows)`, a constant the trace lifts to an
     input, and loop normalization hoists that per-row term, with the fill constants, into a pointwise operand over
     the row axis, which the chunk tier refused twice (slab operands only, a mask only as a coordinate `Select`) —
     the twin takes both indices from the window's own coordinate and compares `key − row` against the position,
     the tier reads a leaf of coordinate aliases and uniform values as uniform (`Fold.uniform_leaf`), spells the
     `where` as the coordinate `Select` it lands with the position a uniform value in the predicate
     (`ir/schedule/mask.py`), and bounds the stream at the block end plus the position at runtime;
   - the chunk tier's multiplicand width followed the streamed value, so an FP8 cache asked for an fp8 fragment the
     computed weight cannot be demoted into — a 1-byte value now takes the 16-bit cell, converting per element in
     its gmem-direct fragment load (its slab declines, so an FP8 cache stages nothing).

   The twin lowers to one tensor-core kernel at the symbolic window with either cache and matches its torch twin at
   every position and window, on a one-wave launch and on a wide one
   (`tests/compiler/passes/test_chunk_position_mask.py`). It reverses a numerics choice, as 2 does for the head:
   the probabilities reach the value product in FP32 and the tensor-core cell rounds them once to bf16 or f16,
   where the old route rounded them to the model dtype in memory; the gate takes 2's probe-token and GSM8K check.
   What is left is the card, and two speed risks the fold alone does not answer:
   - **One wave.** The 64-row fold at 24 query heads is 24 CTAs on a 170-SM card. A layer at 65,536 positions is
     ~100 GFLOP, about 1 ms on the whole card and near today's 6.7 ms on a seventh of it; and the tier keeps the
     stream whole on a one-wave launch, so the causal half is not skipped either. A `REDUCE=g<n>k` split across the
     keys is the answer, and on 2026-11-04 it refused twice, probed off the card: the split declines a twisted
     carrier over a symbolic stream (its last slice overhangs the window, and the split masks an overhang by
     seeding every lift result, which breaks the tier's reading of the carrier); and the chunk tier refuses every
     split piece's causal mask, static or not, because the partition coordinate the sliced body reads is not
     uniform to its gate — so no causal attention has ever folded split on tensor cores. The fix: a twisted
     carrier masks its overhang as the additive coordinate mask the tier lands (the score takes the pivot's
     identity past the window, through the same `where` selects the twin spells, the sliced lift reading the absolute
     key unclamped), and the tier's mask gate reads a kernel's per-CTA coordinates — the contraction's batch axes
     and the split's partition — and its symbolic extents as uniform, so the causal stop lands per slice (block
     end plus position minus the slice base) and a slice past the diagonal folds nothing. Gate: the twin folds
     under `g4k` and `g8k` at both caches and matches at every position and window, one wave and wide; the
     static causal split lands the tier too. Landed and recorded on the dev-box 5090 (2026-11-04, #1324): four
     compiler bugs, two more than the probe found (a sliced load's clamp hid its row stride from the gmem-direct
     fragment loaders, a wrong answer; the partition coordinate's underscore flipped a sliced score's
     orientation, which left the FP8 piece no row). At 512 cached tokens the 64-row fold splits four ways at
     f1x16 to 29.5 us (27B) and 26.6 us (Qwen3.6), on par with the old four-piece rows (28.5 / 26.6); width 16
     folds whole at 5.1 us (old 173.7); the 27B's width 3 splits eight ways (12.5 us); width 1 has no tile fork
     and keeps the old shape, a score cut plus the key split on the remainder (7.1 / 7.4 us, old 7.3 / 6.9), which
     Qwen3.6's widths 3 and 4 take too (12.2 / 14.8 us). The unsplit fold at width 64 spills at f1x32/k4 (222
     us). Qwen3.6's rows are `FAST_MATH=0`, where the f16-accumulate cell is not offered: a pinned record on it is
     invisible to every compile, so its rows stay on the f16/f32 cell. Every root compiles strictly from its file.
     The strict serving audit of both goldens passes (2026-11-05, dev-box 5090): 62 of 62 twins (27B) and 66 of 66
     (Qwen3.6) deploy from the golden rows alone. The first round failed seven twins per file, none attention: the
     rope twins (#1318's restamp demoted their rows; its record run rode this one, 2-5 us a program, 3.4-4.7 us at
     64 rows against 500 us before) and the pick/next twins #1313 added without rows (0.6-0.9 us). A record compiles
     the family's narrower twins first from the file, so the wide rope roots record narrow to wide.
   - **The FP8 cache runs unstaged.** Its keys and values are read gmem-direct, a convert per element, through no
     shared memory and no cp.async; only the bf16 cache stages. If the unstaged form misses the gate, a staged FP8
     slab is compiler work this milestone owns, since no other does.

   Record the route in both recipes (the attention twins at every width, the split arm among the offers, the FP8
   cache unstaged), then measure. Stage it without TMA: the batching stretch pages this cache, and a launch that
   reads a paged buffer through a descriptor refuses to export. cp.async stays legal only under the page grain 19
   recommends (Decisions). Gate: the attention term of the 197,632-token first token <= 117 s, its share of the
   two-term budget — today 544 s, so ~4.6x per layer at 65,536 positions (a 3x fold leaves ~181 s and misses); the
   full-window prompt's first token <= 450 s before 9 widens the chunk; the 4,096-window cells unchanged; the
   probe-token logprob deltas and GSM8K as 2's gate. Needs 1.

   Gated on a rented x16 RTX 5090 on 2026-11-06 (both DISCOVERYs), missed. The attention term of the 197,632-token
   first token is 300 s of GPU time against 117 s, and that first token 429 s; the full window's first token is 687 s
   against 450 s. A layer at 65,536 positions takes 3.66 ms against 6.7 ms: 1.8x, where the budget needs 4.6x. The
   one-wave risk above is what happened. Every row of the route was measured at 512 cached tokens, where the 4-way key
   split wins, so a chunk runs that split in 192 blocks at every window, each block streaming about 49,000 keys at
   197,632; whether a wider split wins there is not measured (the ledger's 2026-11-06 item). The cells within 4,096
   positions did not get slower: at 256 tokens TPOT 8.7 ms, first token 0.15 s. The fold frees 2.8 GiB of device
   memory at rest on the 27B and 1.5 GiB on Qwen3.6, the old route's scores and probabilities. Native returns no
   log-probabilities and the plugin runs vLLM's attention, so 2's logprob deltas cannot see this change; the
   numerics check was greedy tokens and GSM8K. Against packs from the tree before 5, the probe tokens part within 2-62
   tokens on both models (on Qwen3.6 only the attention differs), with both continuations reading well. GSM8K holds:
   0.730 / 0.750 on the 27B, inside one standard error of the plugin lane, and 0.455 / 0.460 on Qwen3.6 against
   0.460 / 0.460. Decode attention, priced at 512 too, got slower at long windows, which 6 owns: the verify step at
   197,632 positions takes 95.6 ms, 75 ms of it attention, so TPOT there went from 26.5 to 42.9 ms.

   Gated again on a rented x16 RTX 5090 on 2026-11-07 with attention priced per window range (#1352; both
   DISCOVERYs), missed. The attention term is 200 s against 117 s, the 197,632-token first token 318 s (TPOT 23.7
   ms), the full window's 505 s against 450 s; the cells within 4,096 positions are unchanged. What is left is the
   kernel: every attention fold runs at 255 registers with local-memory spills and 17% occupancy, and the 64-row chunk
   takes 5.1 ms at 131,072 positions where one cache read per query head at bandwidth is under 1 ms. The unstaged
   FP8 cache is the second risk above; staging it is #1366.

   Gated again on 2026-11-07 with the FP8 cache staged into byte slabs on cp.async (#1366; both DISCOVERYs), met.
   The attention term is 69.9 s against 117 s, the 197,632-token first token 188 s (TPOT 14.7 ms), the full window's
   277 s against 450 s; the cells within 4,096 positions got faster. The unstaged cache was the limit: one byte load
   per element, about 1,150 a thread per 64-key chunk, each holding a register. At 131,072 positions the 64-row chunk
   takes 1.59 ms (5.1 before) and the one-row decode reads its cache at about 1.4 TB/s. GSM8K holds on both models.

6. **Long-window decode attention: one cache read per key-value group, a combine that does not rescan the
   window.** The 27B step grows +48 ms from 7,680 to 197,632 positions; stock grows +4. Two named causes: the
   attention program runs per query head, so each key-value head's cache is read six times, and the partial
   scaling after the 32-way split grows with the window. Re-spell native's attention module per key-value head
   (the group's query heads share the read) and make the split's combine run over the partials, not the window.
   The module is shared, so this also owns Qwen3.6's window TPOT cell (33.1 ms against a 13.7 bar; 16 confirms):
   new programs, recorded at one and three rows for both models, staged as 5 is. No attention row reads the cache
   through TMA today on either model, so nothing has to move off it. Gate: the captured 27B step at
   197,632 positions within ~8 ms of the 4,096-window step (about one cache read at bandwidth, ~4 ms, plus
   margin; today +48);
   Qwen3.6's window step bounded the same way at its own cache's read time, the number fixed from 1's split;
   short cells unchanged. Needs 1, 5.

   The off-card half is in a draft PR, checked on the RTX 4080. Before #1324 the decode route cut the score
   out, scanned the window again for the statistics and again for the probabilities, and split only the value
   product, every query head reading its key-value head's values. #1324 already splits the twisted fold across the
   window, so a finalize combines the slices' partial states. What it kept is the per-head spelling: its remainder ran
   per query head and per output column, re-reading the values six times and repeating the softmax rescale for every
   column. A decode or verify step now spells attention per key-value head, `[rows, kv_heads, group, ...]`, the output
   the bytes of `[rows, heads * head_dim]`. A kernel's axes follow its output's storage order, so the group (6 on the
   27B, 8 on Qwen3.6) is the axis the chunk tier folds as its rows, and no read of a key or value depends on it. The
   step's rows stay on the grid: a verify step reads each slice once per row, from CTAs launched side by side; whether
   the L2 serves those repeats is the card's question. The prefill chunks keep the per-head spelling, so 5's rows at 16
   and 64 (27B) and 4 and 64 (Qwen3.6) stand. The four new twins (`attend1`, `attend3` of both recipes) lower at
   sm_120, and on the 4080 they match the per-head twin at every position and window, whole and split four ways, with
   either cache, at one CTA per row, key-value head and slice (`f1x32` covers the 256-wide head in one warp). Both
   recipe goldens carry the new programs and their target kernels with no rows; #1324's decode routes for those
   programs went with them (19 and 18 measured rows). The hardware golden keeps its old traces, as 2's head did.
   The record waits for the per-window-range pricing (Decisions; #1352): on the 4080, under one pinned fold at one
   row, the fastest key split is 8 slices at 512 positions, 32 at 4,096 and 128 at 131,072, and the grouped spelling
   has a sixth (27B) or an eighth (Qwen3.6) of the per-head spelling's CTAs per split width, so one pick at the
   512-token hint underfills the card at 197,632. The four programs record at both window prices.

   Recorded at both prices and gated on 2026-11-07 (#1352), missed. The 27B's verify step takes 60.7 ms at 197,632
   positions, 39.6 ms of it attention, against 21.9 ms at the 4,096 window: +38.9 ms against ~8 (+73.7 before). At
   131,072 the one-row twin splits 64 ways in 581 us (27B) and 413 us (Qwen3.6), reading its cache at about a quarter
   of bandwidth; a 128-way split measured faster on every decode and verify twin, and no compile offers it (the
   ledger's 2026-11-07 item). Qwen3.6's window step was not served: a prompt that reaches the long range prefills for
   over an hour.

   Gated again on 2026-11-07 with the FP8 cache staged (#1366), missed. The verify step takes 35.5 ms at 197,632
   positions, 14.3 ms of it attention, against 21.1 ms at the 4,096 window: +14.4 ms against ~8. The step's three rows
   each read every slice: at 131,072 the three-row twin takes 487 us against the one-row twin's 205. What is left is
   one cache read per step, the step's rows folded into the fold's rows beside the group, so a slice is read once.

   Gated again on 2026-11-12 with the verify step's rows folded beside the group (#1392), met for the 27B. The
   captured verify step takes 26.4 ms at 197,632 positions, 4.5 ms of it the target's attention, against 21.6 ms at a
   window of up to 8,192: +4.8 ms against ~8. At 131,072 the three-row twin takes 183 us against 487. TPOT at
   197,632 tokens is 11.4 ms, inside the bar. Qwen3.6's twins are recorded (120 us at 131,072 against 274); its
   window step is still not served, and 16 confirms it.

7. **The width-3 Gated DeltaNet step toward its floor.** gdn3rows is 14.3 ms against a 6.9 ms floor after #1282:
   the weight readers run 1.0-1.4 TB/s, the same readers the one-row program runs 5-10 us faster (qkv 38.4 us for
   52 MB, gate_up 70 for 94.7 MB), and the recurrence keeps two ideas #1288 left (the seed read on the step's
   branch, ~12%; `coop-t` for a nested column reduce). Tune and re-record on the card, in the step, cold-checked
   (the gotchas' hot-bench trap). Gate: the verify step <= 20 ms captured; the 9.2 ms TPOT cell itself is decided
   at 10, on the protocol. Needs 2, 3.

8. **The 64-row chunk profiled kernel by kernel.** The chunk costs 50-55 ms and its 41 ms trunk has no per-kernel
   profile — the plan's largest unexplained number, and the widening's floor depends on it. Profile one captured
   chunk under nsys and write the table into the DISCOVERY: which kernels carry the 41 ms, which are weight-read
   bound and which row-bound, and the implied per-chunk floor at 256 and 512 rows. Gate: the per-kernel table,
   its kernels summing to within 10% of the measured chunk, and the named floor per candidate width. Needs 1
   (shares its session).

9. **A wider 27B prefill chunk.** Trace the trunk and native programs at 256 and 512 rows (the prefill bucket is
   an env the twins already trace under; the pack takes a width list since format 14), tune, record; keep 64 and
   the decode width for tails, and pad a prompt's last chunk to the smallest width that holds it rather than
   running a train of narrow chunks. Gate, against 8's floor, no escape: first token at 7,680 tokens <= 1.28 s
   (2x stock); the chunk term at 197,632 <= 20 s ((197,632 / W) x the measured per-chunk cost); device memory at
   rest within the standing condition at the widest kept width (native sits at 30.1 GiB against stock's 30.4, so
   the wide chunk's activations get ~0.3 GiB); the full window still serves; goldens current. Needs 3, 4, 5, 8.

   #1392 landed the pack's half: a pack holds a chunk wider than the plugin's and keeps the plugin's chunk and the
   decode width for a prompt's tail (`--prefill-size`, `SERVE_NATIVE_PREFILL`). No recipe uses it. What is left is two
   pieces of work (the 2026-11-12 section has the arithmetic):

   - **The 256-row chunk loads.** Traced at 256 rows the Gated DeltaNet layer has twelve root kernels where the
     64-row one has four: the recurrence across its four 64-token chunks adds eight. No row covers them. The
     compile's own cuts leave four FP32 workspaces of `[4, 64, 64, 48, 48, 128]` a layer, 18 GiB each, so the pack
     runs out of device memory at load, and under the 64-row kernel's recorded route one piece does not finish in
     80 s. This is a cut problem and reproduces off the card at sm_120 lowering. Done when the 256-row program
     compiles to a kernel set whose workspaces fit the standing memory condition and every piece finishes; its rows
     are then recorded on the card. The compile half is done (#1398); the records are not.
   - **The row cost of the Gated DeltaNet layers falls about 2.4x.** Width alone meets the 197,632-token cell and
     not the 7,680-token one. The layers are 27.7 ms of a 35.9 ms chunk; the in projections and the input norm, the
     state and convolution pieces and `k_slice_unsqueeze_reduce` are the row-bound ones (8's table). A tune on the
     card, at the wide width once it loads.

   The first piece loads since #1398 (2026-11-13): a program wider than one chunk traces the delta rule with the
   scan carrying only the state, and the 256-row NVFP4 layer lowers at sm_120 with no workspace above 0.18 GiB (the
   stored steps of the triangular solve, which exist once a program because scratch is packed by liveness). A
   second fix for the same fault, a tracer that clamps only the written axis of a slice write, was built and
   dropped: on top of #1398 it changes no workspace, and it re-keys the Gated DeltaNet kernels of 16 goldens on
   seven cards. Keeping only the last step of the solve was tried and reverted too: the shared-memory schedule of
   the solve needs every step stored at its own index, and without it the 16-row solve went from 59 to 214 us.

   What a rented RTX 5090 measured on 2026-11-14, on the trace before #1398, so no row of it is in a golden. The
   256-row layer's slow picks are a search problem, not a width problem: the main output kernel is 22.6 ms as
   the priors pick it and 0.84 ms after a 60-schedule tune of each of its fourteen heavy cut pieces (0.4-4 ms
   each before, 8-160 us after); the solve is 21 ms as picked and 0.14 ms tuned; `k_matmul` at 256 rows 1.9 ms
   and 13 us. Every heavy piece refuses a plain `--tune` (it is cut again) and tunes under `PLACE=fuse`. The
   pieces to record again on #1398's trace are the same ones, and the same steps apply: bench the root so its
   routing rows land in the tune DB, record the kernel set through a seed row on the root, tune each piece above
   about 25 us by name, record again. Each `emmy run` on the 79-program working golden spends about 13 minutes
   on one core before it benches, so the rows go in parallel streams. Not recorded: the two 256-row attention
   targets (7.2 ms a layer as picked; their strict record stops at a second cut fork), and no first token was
   timed at 256 rows.

   Recorded on #1398's trace the same day (second rental, the recipe golden): 25 of the 28 targets a 256-row
   chunk adds. The Gated DeltaNet layer's twelve kernels sum to about 1.06 ms at 256 rows, about 4.1 us a row,
   against about 9 us a row in the 64-row chunk of 2026-11-12. As the priors picked them the solve was 21.7 ms
   (0.14 ms recorded), triangle-times-values 1.16 ms (63 us) and the main output kernel 0.73 ms (0.30 ms). A
   record of a wide twin compiles its narrower twins first under strict evidence, so the targets are recorded
   inside a working copy of the recipe golden, never in a bare trace. Left: the two 256-row attention targets
   (they need the per-window records the 64-row ones have), one `post` kernel whose pick hangs, and then the
   strict export and the first token at 256 rows.

   The rotary term of 8's floor is gone since 3 (0.09 ms a chunk), so its "512 rows still leaves 3.85 s" no longer
   holds as written; the row cost above is what replaces it.

10. **Gate: the 27B at the bar.** The protocol re-run, all columns, one session; the DISCOVERY updated. The
    197,632 first-token cell reads as its two terms — chunk <= 20 s (9), attention <= 117 s (5) — so a miss names
    its term. Gate: every 27B bar cell green, or each miss named with its measured kernel-level floor — the owner
    decides the cell's fate here, not at the switch. Needs 2-9.

11. **Qwen3.6: the paging programs' cost out of the decode step.** Milestone 1's split sizes it; the dev-box
    figure is 7.8 ms a step of pager table programs, the decision kernel on one CTA times ~160 paging runs (40
    layers x four paged buffers: gate-up and down codes and their scales). By measurement, in order of least
    design: a parallel schedule offered for the decision program (at 256 pages it builds a single one-CTA
    schedule, so no row could change it); the per-expert scale pages folded into their codes' paging run; one
    paging run per layer over its buffers' joint list. Gate: paging overhead <= 2 ms a captured one-row step;
    tokens unchanged. Needs 1.

12. **Qwen3.6: slots sized to the verify step's picks.** A verify step's three rows route past the 8 slots, so
    the drafter buys almost nothing (a 44-52 ms drafted step against a 19.6-19.8 ms one-row step). The pager's
    split takes `min(budget, L)` slots with L the largest page list one decode launch names — today the one-row
    launch's 8 picks; make L the verify launch's 24. Same device memory: 16 resident pages become slots. If 1's
    split puts the verify step's time in the GDN programs instead, this milestone tunes those the way #1282 tuned
    the 27B's width-3 program. Gate: every page list a verify launch names fits the slots, and the captured
    verify step spends <= 2 ms in paging programs and in-place expert reads together (11's bound held at the
    verify width); the TPOT cell itself is decided at 16. Needs 1, 11.

13. **The paged gate-up piece takes a tensor-core tile.** Once its weight is paged the 64-row expert program's
    gate-up piece is offered no tensor-core tile. This is a compiler change, not a wider offer: at one expert a page
    the weight is several pages, so TMA and cp.async are both off it (Decisions), and the masked N #1261 gave
    cp.async does not help. What remains is the plain compute fill, as the paged experts' `d2/smem` rows use. So
    either that fill learns to spell this piece, or the renderer lets a cp.async copy resolve a page per chunk; the
    second would also open cp.async to a cache of several pages (19). Size it before committing to 14's date, then
    re-record the expert rows at the schedules the paged export offers. Gate: the gate-up piece
    offered and recorded on a staged tensor-core schedule, and its per-layer time at 64 rows at most twice the
    down piece's 10 ms — it moves twice the bytes — from 24 ms today; goldens current. Needs nothing; lands
    before 14.

14. **Qwen3.6: wider prefill chunks.** The copies per chunk are nearly width-free — a 64-row chunk already
    reaches ~220 of 256 experts — so a 256/512/1,024-row chunk divides the per-token copy cost by 4/8/16, the
    same amortization stock gets from 2,048-token chunks over its offload; the copy-floor arithmetic in State
    says the 7,680 and window cells need ~1,024 rows. Trace, tune and record the wide twins (trunk, experts,
    native's programs); pad the last chunk wide rather than running 64-row tails, since every full-coverage chunk
    costs the same ~0.3 s of copies. Gate, no escape: prefill at 7,680 tokens >= 1,000 tokens a second (today
    50); first token at 7,680 <= 4.4 s (2x stock); the captured chunk's split between page copies and compute
    written beside the kept width (15's reference); activation memory on the card and the 23 GB host checked at
    every kept width. Needs 11, 13.

15. **Qwen3.6: the chunk's copies overlap its compute.** Prefill knows each layer's picks after its router, one
    layer ahead of the expert launches; a second stream listing the next layer's pages (the two-slot form the
    follow-ups list holds for the tower's block copies) hides link time behind compute. An earlier dev-box
    measurement had prefetch losing at decode on a link-bound box; a wide prefill chunk computes long enough per
    layer for the overlap to pay. Capture must keep the paging programs ordered against the launches they serve.
    Gate, no escape: at 14's kept width, the captured chunk <= 1.15x the larger of its copy term and its compute
    term, the terms from 14's split. Needs 14.

16. **Gate: Qwen3.6 at the bar, or the fallback named.** The protocol re-run, all columns, one session; the
    DISCOVERY updated. If the first-token cells hold above the bar at the measured copy floor, the fallback goes
    to the owner as a recipe decision: a checkpoint whose experts fit the card whole — the model line publishes an
    NVFP4 quant, and NInfer fits with integer experts at 20.4 GiB — re-gated on GSM8K against the FP8 recipe.
    Gate: every Qwen3.6 bar cell green, or each miss named with its floor and the fallback question put. Needs 6,
    11-15.

17. **Hardware golden promotion and both prior refits, once.** Every row this plan and the retired tuning plan
    recorded on the RTX 5090 promotes with the record-and-promote flow; re-record what the restamps re-keyed and
    what #1298/#1299 changed the measurement of; refit both priors in one pass; drop the `prior-pending` tags.
    Every compiler change of milestones 2-15 and the follow-ups list's compiler correctness items (the wrong FP4
    code and the wrong-answer threshold, the recurrence roll, the unmeasured `SHARED_CARRY`) land before this, so
    the re-record measures the fixed
    kernels; the other cards' open record runs (the V100/A100/H100/4090 GDN routes) stay follow-ups, not this
    plan's scope. Gate: the prior reproduction gate green with no `prior-pending` left on these recipes; the
    weights committed. Needs 2-16, and the prior gap the ledger dates 2026-10-31: the retired plan's promotion
    (#1309, in its first form) put every recipe row into the RTX 5090 hardware golden and turned 20 gate nodes red
    on seven kernels that no refit reproduces; the PR landed its re-records without the promotion, and #1310 lands
    the promotion of everything the gate reproduces, less the five RTX 5090 kernel trees of that ledger item, with
    both priors refit. What remains of 17 is those five trees and the tag drop, which wait on the ledger item's
    feature and export work.

18. **The native lane in the recipes and the deploy path (#1302).** A recipe names the native server as its
    engine; the image carries `emmy-server`, the worker and the pack prepared from the recipe golden, so a deploy
    compiles nothing; `emmy bench` drives it at concurrency one. The plugin lane's unbuilt vLLM 0.25.1 image is
    built only if milestone 26 keeps that lane. Gate: `emmy deploy local` boots each recipe natively from its
    image with no compile; the benchmark runs; each DISCOVERY records native against the plugin lane on the same
    card. Needs 17.

19. **Continuous batching: the design.** What native lacks today, from its architecture doc: one active request
    (busy responses above it), the KV cache one request's buffer allocated at load (per-request allocation and
    reclamation and prefix reuse "not implemented"), one set of carried Gated DeltaNet state buffers, one position
    scalar a step, greedy selection over one row, host sampling as one vocabulary sort per token. The design to
    write: per-request cache regions through the pager's pages along positions, under the window-sized budget and
    the per-request context split the bar names; per-request state blocks (the plugin's slot layout); batched
    decode at the trunk's existing bucket widths (4/16/64 — already priced rows), rows padded as the plugin pads;
    per-row positions (rotary and attention already read the position from device memory); a scheduler in
    `emmy-server` admitting and retiring requests between steps and interleaving prefill chunks with decode rows;
    per-row selection and sampling, and where the one-core sampling sort stops scaling (23 owns the fix); prefix
    reuse and preemption stay out. The page grain is the design's first decision, because it decides what stays
    schedulable (Decisions). The recommendation is one page per request. A launch then reads one base address,
    which the renderer addresses as the flat buffer it replaces, so today's rows keep their transports and their
    vector loads. A region's size is then runtime bookkeeping, which is how the full window can still serve at
    concurrency one; the design confirms that. Several pages per request would load element by element and leave
    only gmem-direct reads and the plain compute fill. Lands as architecture text first, as the paged-experts design
    did. Gate: the owner approves; the design names the page grain, which programs survive unchanged, which widen,
    and which rows are new. Needs 1, for the concurrency columns its budget rests on. The first draft's order put
    the design after 10; it moved up because nothing in it has to wait for single-request parity. Revise it once
    10 lands.

    #1392 wrote the design into `emmy/serving/native/ARCHITECTURE.md`: a region of the window-sized cache per
    request, reached through a one-entry page table; a state block per request; what a batched step runs and which
    programs are new; the scheduler. Nothing of it is built. The gate is open: the owner has not approved it.

20. **Per-request cache regions and reclamation.** Today the KV cache is one key and one value buffer per layer,
    sized to the window and allocated at load, and the Gated DeltaNet state is one set of carried buffers — one
    request's. Give each admitted request a cache region (one page, as 19 recommends) and a state block,
    allocated at admission and reclaimed at retirement; the window-sized allocation does not grow. The state block
    takes the plugin's mechanism: one-page paged outputs and one launch pointing every layer's table at the
    request's block (`point_gdn_layers`). This milestone carries the compiler change the grain needs: a symbolic
    extent never counts as one page today, so the cache window has to count as one under its cap. No repository
    golden pages anything, so no stored offer moves. Paging re-keys nothing, since a kernel's identity is its
    hint-free shape. Gate: one request through a region is bit-identical to today (tokens and
    logprobs); every attention row of both goldens still realizes over the paged cache; a scripted admit-and-retire
    sequence across regions leaks nothing (device memory at rest returns to its baseline); capture replays.
    Needs 19.

21. **Batched decode at a fixed width.** N requests, one row each, through the trunk's existing decode-bucket
    programs, each row over its own cache region and state block from 20, rows padded as the plugin pads, per-row
    positions. The trunk's buckets are the easy part. Three other kinds of program are new here:
    - **Gated DeltaNet layers.** A batch row already owns independent state and history, but N carried states is a
      new buffer shape, and a buffer's shape is part of a kernel's identity.
    - **Native's own programs at N.** Norm, greedy selection, carry and the head have rows at one and three rows
      only. Rotary, attention, merge, router and combine also have the decode bucket, at 16 and 64 on the 27B and at
      4 and 64 on Qwen3.6, and each widens to whatever N the design picks.
    - **Attention over N regions.** A batched body reads several pages, so TMA and cp.async are off it (Decisions).
      The alternative is one launch per row over one page each. Measure both, as 19 scopes them.

    Their rows are recorded here and promoted at 25. Gate: N=1 bit-identical to today; each request's greedy tokens
    equal its solo run; TPOT per request at N=4 and 8 beside N solo runs; capture replays. Needs 20.

22. **Continuous batching proper.** Admission and retirement between steps; a step is a prefill chunk or a batch
    of decode rows (the plugin's own alternation at `--max-num-batched-tokens 64` is the reference shape);
    per-request page reclamation; streaming per request; busy only above the admission cap. Gate: the benchmark
    client at concurrency 8 and 32 completes on both models with every request's tokens equal to its solo run;
    the single-request bar cells re-checked at concurrency one, unregressed. Needs 21.

23. **Sampling above temperature zero at concurrency.** Native's sampler is one host-core vocabulary sort per
    token per request; at concurrency 32 that is 32 sorts a step on one core, and retiring the plugin lane means
    serving positive temperature, not only greedy. Per 19's design: per-row sampling on the device (a compiled
    program beside greedy selection) or a batched host path, whichever the measurement justifies; a seeded
    request reproduces itself. Gate: at temperature 1, concurrency 8 and 32, per-request TPOT within 1.1x the
    same cell's greedy run on both models; a fixed seed repeats its tokens; any new program's rows recorded.
    Needs 22.

24. **The drafter at concurrency.** Verify steps batch K+1 rows per request — new widths, new rows — or the
    scheduler drafts only below a measured concurrency, whichever the measurement justifies; Qwen3.6's slot list
    re-sized to the batched verify step's picks (milestone 12's rule at the new L). Gate: drafted throughput at
    concurrency 8 at or above undrafted, or the verdict that the drafter stays off above concurrency N written
    into the recipes. Needs 22.

25. **The concurrency bar met, and the batching rows promoted.** The bar's numbers were fixed with the owner from
    milestone 1's stock and plugin concurrency columns; this milestone closes the named gaps, promotes the
    batching programs' rows and refits — the second, smaller promotion: the stretch adds new programs, so nothing
    from 17 re-records. It runs after 17 because 17's compiler fixes can re-key rows 21-24 already recorded: a
    restamp keeps such a row's schedule and drops its microseconds, and this milestone re-measures those once.
    Gate: the concurrency bar met or each miss named with its reason, and native at or above
    the plugin lane's measured concurrency cells; the DISCOVERY tables carry native's concurrency columns;
    reproduction gate green. Needs 17, 23, 24.

26. **The switch.** What must be true: milestones 10 and 16 green (or their misses accepted by the owner), 25
    green, `emmy eval golden` clean on both recipes, the strict exports and the GDN GPU tests #1290 unblocks
    green, the native images published through the release flow. Then both recipes name the native lane as their
    deploy default, the plugin lane's engine blocks leave the two recipes, and the unbuilt vLLM 0.25.1 plugin
    image is never built. vLLM remains the serving integration for models the native runtime does not serve
    (Gemma 4, the V100 fork models, the embedding recipes): "completely" is bounded by native's model coverage —
    dense Qwen3/3.5-family text and vision-language, Qwen3.5-MoE — and widening that coverage is its own plan.
    Fallback if a bar cell stands missed: native becomes the default wherever it beats the plugin lane, the
    plugin lane stays deployable from its last published image for the cells it still wins, and the DISCOVERY
    records the stock gap the owner accepted. Gate: `emmy deploy` of each recipe boots native from its image; the
    plugin lane removed from both recipes or the fallback recorded; the serving docs name the native server as
    these models' path. Needs 18, 25.

Milestone 1 needs nothing; #1299 lands before 2, the first record milestone. The 27B stretch: 2 and 3
need 1; 4 needs nothing; 5 needs 1; 6 needs 1 and 5; 7 needs 2 and 3; 8 needs 1 and shares its session; 9 needs 3,
4, 5 and 8; 10 needs 2-9. The Qwen3.6 stretch runs beside it (the owner's gate order holds: 10 is decided before
16): 11 needs 1; 12 needs 1 and 11; 13 needs nothing and lands before 14; 14 needs 11 and 13; 15 needs 14; 16
needs 6 and 11-15. 17 needs 2-16 and the follow-ups list's compiler correctness items; 18 needs 17. The batching
stretch runs beside the parity work up to its promotion: 19 needs 1; 20 needs 19; 21 needs 20; 22 needs 21; 23 and
24 need 22; 25 needs 17, 23 and 24. 26 is last and needs 18 and 25.

The longest chain is eight milestones, and two paths tie at it: batching (1, 19, 20, 21, 22, 23, 25, 26) and parity
(1, 11, 14, 15, 16, 17, 18, 26). It was twelve while the whole stretch waited on the promotion. 5 and 6 do not wait
for 19's approval: they avoid TMA, which holds under any page grain, and re-record only if 19 lands on several
pages per request.

## Card time

The record and measurement milestones share rented x16 RTX 5090 sessions instead of renting one each:

- Milestone 1's two protocol sessions (one per model) also take the nsys splits, the 27B first-token controls and
  8's per-kernel chunk profile.
- 2, 3 and 5 record in one rental, and it comes first: #1321 re-keys every attention twin of both recipes (the 27B's
  attend 1/3/16/64, Qwen3.6's attend 1/3/4/64) on top of #1313's chunk kernels, so strict native export refuses
  until that rental records attention at every width — before 4's gate and before any native measurement. It
  records the head rows, both recipes' rope rows (3 re-keys Qwen3.6's rope too) and the folded prefill attention
  route with its split arm. 6's per-key-value-head re-spell re-keyed the decode attention #1324 recorded, so decode
  attention records twice. Done: 2's head rows on a rented card (#1322), 3's, 4's and 5's rows on the dev-box 5090
  (#1324), and the gates of 3, 4 and 5 on a rented x16 card on 2026-11-06. 6 and 7 share a second: decode attention
  at one and three rows for both models, at both window prices (6's PR lists what to measure), then the width-3 GDN
  tune in the step. 6's half is done (#1392, 2026-11-12); 7's tune is not. 9's wide-width records are a third, once
  the 256-row chunk loads, and 7's tune can share that rental.
- Qwen3.6's 13 and 14 share one rental; 15's overlap check rides it when the runtime lands in time, else it takes a
  short rental of its own before 16's gate session.
- 10 and 16 are each a fresh protocol session per model (the bar requires one session, all columns).
- 17 rents once per card; the batching stretch records only new programs and promotes once, at 25, after 17.

Rented 23.5 GiB hosts have known traps; the `onboard-model` skill lists them. So do containers whose CPU quota is
smaller than the CPUs they show.

## Working in parallel

Four lanes, which meet at the goldens. The parity phase absorbs four people, the batching stretch two, as long as
the changes that write a shared golden take turns.

| Lane | Milestones | Card | Golden written |
| --- | --- | --- | --- |
| 27B | 2, 3, 5, 6, 7, 8, 9 → 10 | most milestones | 27B; 3, 5 and 6 also Qwen3.6's |
| Qwen3.6 | 11, 12, 13, 14, 15 → 16 | most milestones | Qwen3.6; 11's pager rows the 5090 hardware golden |
| Compiler, off the card | the correctness items that block 17; 4 landed (#1313) | at the gates only | any lowering fix restamps all of them |
| Batching | 19, then 20-25 | 21 onward | none before 21, then new programs in both recipes |

Milestone 1 is two sessions, one per model, so two people take one model each from the start. One of the three
correctness items that block 17 reproduces off the card: the recurrence roll. The wrong FP4 code and
`SHARED_CARRY` need one. That still makes this lane the one to staff
first: 17 is on the parity path and those items are its unowned prerequisite. The `fold.lower` assert and the
`[N, 2]` loads also reproduce off the card, though neither blocks 17. 13 moved to the Qwen3.6 lane because it writes
that recipe's golden.

**One golden-writing change in flight per golden.** A record milestone ends by writing measured rows into a golden,
and a restamp re-keys kernels across the file. Two such changes in flight against one golden conflict, and the
second's rows may be re-keyed by the first. These changes therefore run one at a time, as repository-wide events: 3
(both recipes' rope rows), 5 and 6 (attention, shared by both models), and each lowering correctness fix (a restamp
of every golden). 13 runs alone against any other Qwen3.6 record. Everything else in a lane runs in parallel between
them.

The model lanes meet once at a gate: Qwen3.6's gate (16) needs the shared attention work (6) from the 27B lane.

## Developing without the rental

The owner's RTX 4080 has 16 GB and is sm_89. (The "dev box" elsewhere in this plan is a 5090 on an x1 link, not this
card.) Two rules follow, and they split the plan cleanly.

**Capability decides what builds and runs; the card's name decides what a number means.** For a program targeted
at sm_120 this card gives `realized` only — `built` and `correct` run where capability matches, so the kernels
that ship are not the kernels checked here. What it does check is the change: compile the same program for sm_89
and a lowering bug, a wrong index, a leaked region or a token mismatch shows up the same way. Measured rows never
transfer between cards, so nothing recorded here is evidence for a 5090 pick. Most correctness gates can run here,
apart from the FP4 ones below; every number on the target models is the rental.

**The surrogates already exist.** The native tests serve a tiny two-layer Qwen3.5 — one Gated DeltaNet layer and
one attention layer — a tiny Qwen3.5-MoE whose experts page against a slot budget, and a one-layer
multi-token-prediction head drafting two tokens: every architectural feature of both target models, at shapes
that fit anywhere. For real weights, native's cached generation was qualified on this card with Qwen3-0.6B, a dense
Qwen3, through `--native-checkpoint`. The 1.28x-stock figure for that model is the plugin lane's, on this card.

What each milestone can do here:

- **Local to its whole gate:** 19, 20, 21. These are token equality, leak checks and capture, which is why the
  batching lane is the one to start from this card. 19 needs no card.
- **Local for lowering, offers and correctness, with the number left to the rental:** 3, 4, 5, 6, 11, 12, 13, 14,
  15, 22, 23, 24. 4, 22 and 23 name the target models in their gates, so their mechanics run here on a surrogate and
  their verdicts do not. An offer is computed, so whether the paged gate-up piece is offered a tensor-core tile (13)
  is answerable here.
- **Rental only:** 1, 7, 8, 10, 16, 17, 18, 25, 26 — a tune, a profile, a protocol session or a promotion. 18 is
  here only because its gate boots the two recipes, which do not fit 16 GB; the lane it builds can be exercised
  locally with any small checkpoint.

Three things this card cannot do:

- **No long window on the target models.** Their four long-window cells — the switch's hard blockers — belong to
  the rental alone. A surrogate is a different matter: a hybrid holds a key-value cache only in its
  full-attention layers, so Qwen3.5-9B's is 16 KiB a token at FP8, 18 with its drafter, and a 131,072-token window
  costs 2.25 GiB. If that surrogate ever serves here with usable rows, the long-window kernels of 5 and 6 can be
  exercised for correctness at a real window length; only their numbers are the rental's.
- **No FP4 tensor cores.** sm_89 has none, so the W4A4 head (2) lowers here at `--target sm_120` and runs
  nowhere; its numerics gate is the rental's.
- **No TMA, and no cp.async over several pages at any capability.** Ada has no descriptor, so a TMA schedule never
  runs here. The paging refusal is still computed: a lowering at `--target sm_120` shows which transports a paged
  read is offered, which is how 5, 6 and 20 check it from this card.

The one real gap is a real-weight Gated DeltaNet hybrid: local GDN coverage is synthetic, and no published member
of these families fits 16 GB unquantized. A published FP8 build of Qwen3.5-9B exists and is untested. Without
measured rows it may be too slow to use, and onboarding it is its own task, not this plan's.

## Decisions

- **The margins (1.2x / 1.5x / 2x) are this plan's definition of "comparable"** until the owner revises them. They
  come from the measured tables: the 27B's drafted TPOT is at 1.2x on #1282's probes (about 10.5 ms at the
  protocol's 68% acceptance — close, not made), the long-window TPOT floor is about one cache read above stock,
  and 2x first token is reachable on the 27B with the chunk work while staying honest about Qwen3.6 (see Risks).
- **NInfer is context, not the bar.** Different checkpoints and quantizations; its value is proving the card's
  headroom at each cell.
- **The 27B gates first** (owner's call); Qwen3.6 work runs beside it but its bar decision (16) comes after 10.
- **New twins trace under the compile pin's transformers (5.19), what the serving image has installed since #1234.**
  #1277 re-traced each serving config under it and got the stored programs but one, which it dropped. A venv on
  another version traces the GDN layer differently, so record and audit runs use the pin.
- **Slower measured rows stay beside the faster ones** in every golden, as the canonical goldens keep them; a
  strict compile picks the fastest matching row either way. Routing rows stay one per arm.
- **Decode matmul rows carry only schedules that also win with a cold cache** (the gotchas' hot-bench trap).
- **NVFP4 decode tables stay as they are.** Every deployed NVFP4 linear of the 5090 goldens runs W4A4 on the
  block-scaled fp4 MMA; no deployed kernel reads the 256x2 pair table, which is why #1298 is off the parity path.
  A table-free decode would matter only if a W4A16 schedule ever wins a pick.
- **A paged cache is the same kernel, but paging shrinks what can read it.** A kernel's identity is its hint-free
  shape, so paging the cache re-keys nothing. The schedule search does read the paged declaration, though, and it
  takes transports away. TMA is off any paged buffer: the descriptor is encoded once from a fixed address, and native
  refuses to export a launch that reads a page through one. cp.async is off a buffer of several pages too, because its
  per-thread addresses cannot resolve a page per element, and a symbolic extent always counts as several. What remains
  there is gmem-direct and the plain compute fill. Today this binds little: of the 27B's 23 attention rows 22 are
  unstaged and one is `d2/smem-async`, and Qwen3.6's 19 split the same way, 18 and one. It binds the folded attention
  of 5 over a bf16 cache, which stages through cp.async, so over several pages that form may lose its speed or have
  no legal schedule; since #1366 the fold over an FP8 cache stages through cp.async too, and over several pages it
  falls back to the gmem-direct read, 2-3x slower. So 19 recommends one page per request, 20
  makes the capped window count as one page, 5 and 6 stay off TMA, and batched attention over N regions is new work in
  21. If 13's renderer change lands, cp.async opens up over several pages as well.
- **Attention is priced per window range** (owner, 2026-11-06, after 5's gate). One pick, measured at the symbolic
  window's 512-token hint, ran at every window and missed at the long ones (the ledger's 2026-11-06 item), and no
  step runs a window below 4,096: the runtime rounds every window up to a multiple of 4,096. Native compiles each
  attention program at a short and at a long window price, and the runtime takes the range's program when it captures
  a window's graph, which it already does every 4,096 positions. A kernel's identity is hint-free, so a program keeps
  its kernels and gains rows at a second binding, and the long price benches through `emmy run --golden` directly,
  with no pinned pack. One price at a long window was rejected: on the 4080 the best one-row key split moves 4x
  between 4,096 and 131,072 positions (#1329), so the short cells would pay. Two ranges first; the boundary comes from
  the card's measurements. 5's prefill and 6's decode both take it. Landed with #1352: priced at 4,096 and 131,072,
  the long range from 16,384, where its picks stop losing in the 27B's verify step on the RTX 5090.
- **Prefix caching stays out** of every comparison and out of the batching scope, as the measured sessions ran
  (`--no-enable-prefix-caching` on stock).
- **Two promotions, not one.** The single-request promotion (17) does not wait for batching: the stretch's
  programs do not exist yet, and waiting would stall parity on months of scheduler work. Batching's rows are new
  programs, promoted once at 25; nothing from 17 is re-recorded by it. 25 comes after 17, because a batching row
  recorded before 17's compiler fixes can lose its microseconds to their restamps and is re-measured there.
- **The checkpoints stay the recipes' pinned ones.** A fallback to a checkpoint that fits the card whole is a
  recipe decision the owner takes at 16 or 26, never a silent swap.

## Follow-ups — the single ledger

Items no milestone owns, each with its dated verdict. The retired tuning plan's list folded in here; after its
deletion this list is the only record. The compiler correctness items below block 17; everything else is an idea
or an open item another effort walks.

Perf ideas:

- (2026-10-23) The decode weight readers stream DRAM at 1.66-1.73 TB/s while all CTAs run (cuBLAS 1.74); the loss
  is a launch-start ramp and an uneven tail, a few us per launch; no offered WORK/TILE/STAGE schedule closes it.
  Measured, no gain (#1219, closed): early weight staging under programmatic dependent launch — strict boot 55.4
  against 55.9 tok/s, the early release costing ~10 us per gdn1 program; dynamic work handout for the tail (~0.2
  ms a token) not prototyped. Measured, no gain (#1220, closed): the GDN in_proj sibling weights joined into one
  5120x16480 read — 112.9 us against 112.5 apart, the joined producer offered no MMA (guarded store, 16480 not a
  multiple of 64); cuBLAS bounds the gain at 8-12 us a layer. Left out: the 27B TPOT cell is already at the bar's
  edge without it.
- (2026-10-20) In-place GDN state read: with one kernel the only reader, read the old state from the cache block
  and drop the gather — ~0.22 ms a token. Needs its own legality rule (each CTA reads exactly what it writes, all
  reads before the first write, no cross-CTA split of `k`); history stays gathered for the conv shift. Taken only
  if 7 misses its gate.
- (2026-10-21) GDN serving copies that remain since #1215: the state gather above, and the verify step's scatter
  of 1+k states back with `index_copy_`.
- (2026-10-22) gdn64 recurrence target `02d4dd`: 122.8 us on #1230's golden against 120.8 before it (+2 us). Not
  investigated.
- (2026-10-22) Joined cuts are measured on gdn1 only; gdn16/gdn64 compute the recurrence chunked with other
  pieces, and nothing joins them. No joined cut puts the out_proj epilogue into its FP4 MMA piece; the out_proj
  join of the statistic reduce and the residual region measured 16.6 against 17.7 us (not recorded).
- (2026-10-20) Neighbouring schedules that passed strict, worth re-checking in a tune: gdn64
  `...5f1430.m64.8105a48e7b21` (`w1x8`, bf16 `f1x1/k8`, `d4/smem-tma`), pre16 `...a7efbb.m16.fm.83c0b7f74b51`,
  gdn16 `0e5142` `...8105a48e7b21` (`d3/smem-tma`).
- (2026-10-24) The plugin lane's MTP prologue has only an any-width program, 0.73 ms a draft step; native runs
  static twins since #1282. Lives only while the plugin lane does.
- (2026-10-24) The paged tower's first image at the narrow width takes 100.7 ms against 16.7 on the device, and
  vLLM's boot profile run 6.8 s against 0.45; the block copy runs on the compute stream, where a second stream
  listing the next block into a second slot would hide it (15 reuses the idea for experts).
- (2026-10-25) Native image costs: the 1,024-token tower attention at 840 us (the head padded 72 to 80 in a
  128-wide tile); a width's first image pays its graph capture (80 ms); the image export's peak RSS 1.7 GiB above
  the text pack's. Images must keep working through every gate, but no bar cell measures them.

Open items:

- (2026-10-17) Needs a V100 record run: Qwen3.8 FP8's GDN rows (gdn16 `13e5d6`, gdn64 `353a47`) were restamped
  but not re-recorded or tuned after #1171.
- (2026-10-22) Needs a record run on each card, after #1230's restamp: Qwen3.8 FP8 dropped 11 decisions on A100,
  26 on H100, 16 on RTX 4090, 17 on V100; Gemma 4 RTX 5090 kept its decisions with 96 rows demoted to proposals.
  A strict boot of those recipes on those cards stops at a no-evidence fork until a record run measures it. 17
  promotes the 5090 only; these stay here.
- (2026-10-20) Standard lane: record the 8 missing twins only if serving ever needs `FAST_MATH=false`.
- (2026-10-25, #1282) Tooling: an isolated `--kernel REF --realization ROW` record replays the row's own
  schedule, so a tuned schedule is recorded through a seed row with no knobs and the schedule as `EMMY_KNOBS`.
  (The `--pin-route` half of that item was fixed by #1303.)
- (2026-10-25, #1282) The plugin lane's `gdn3rows` and `mtp-prologue-sym` rows, which that lane still runs.
- (2026-10-25, #1277) Qwen3.6's width-4 GDN program keeps slow recorded picks: 1.34 ms for its recurrence target
  (a 4x4096 contraction on a one-warp tile, 275 us three times), three reduces of 120, 120 and 60 us that offer
  no seam, in_proj pieces at 55 us. 12 picks this up only if 1's split blames the GDN programs.
- (2026-11-01, #1311) Qwen3.6's 4-row prefill chunk runs attention as one uncut kernel,
  `k_matmul_softmax_range_reduce_1b2f00`: 66.6 ms a layer, 131.7 ms past 4,096 positions, 88-93% of the chunk, 11.3 s of
  the 256-token first token. #1281 added the chunk, and the golden still holds only the uncut rows and no routing row
  for it at c497a0cc7; widths 1, 3 and 64 are cut. Needs a record run on the card. A cut in line with those widths would
  bring that first token near the 257-token control's 5.88 s, an estimate; no such program was built. Since #1313 a
  256-token prompt takes four 64-row chunks and never reaches this chunk; a tail past the last whole 64-row chunk
  still does. Closed by #1324: the re-keyed width-4 twin takes a score cut and a key split, 14.8 us at 512 cached
  tokens.
- (2026-11-01, #1311) Not measured: native at 8 and 32 rows, which 22 owns, and an nsys capture of Qwen3.6's longest
  prompt. The audit is green on main at `f33a86522` on an x16 5090 (2026-11-06): 62 of 62 and 66 of 66 twins.
- (2026-11-06) Every attention twin's rows are measured at 512 cached tokens, the symbolic window's hint, so a long
  window runs the pick made for 512. The 64-row fold keeps the 4-way key split in 192 blocks: 12.3 ms a layer at the
  last chunk of a 197,632-token prompt and 300 s of its first token, which is 5's miss. The verify step's three-row
  route, a score cut and a 16-way key split, takes 4.3 ms a layer at 197,632 positions, 75 ms of a 95.6 ms step. Not
  measured: whether a wider split or another route wins at long windows; `emmy run --golden --kernel` benches a
  kernel at its stored bindings only. 5 and 6 both need picks that hold at the windows their gates name. Decided:
  attention is priced per window range (Decisions). Closed by #1352, which also found that every prefill attention row
  before it was a position-zero time (a bench drew the chunk's position near zero) and that a compile ranked a kernel
  at one size by its rows at another.
- (2026-11-07, #1352) Key splits stop at 64 ways (`SPLITK_WIDTHS`). At 131,072 positions a 128-way split measured
  faster on every decode and verify twin (27B one row 475 against 581 us, Qwen3.6 286 against 413) and no compile can
  deploy it; offering it moves every split fork's ballot. Closed by #1366: with the FP8 cache staged, 128 ways
  against 64 measured +2% (one row) and +7% (three rows) on the 27B and -23% and +4% on Qwen3.6, so it is not offered.
- (2026-11-07, #1352) A split's finalize piece keeps its parent's axis table, whose cache-size axes carry the default
  hint, so its rows sit at 512 cached tokens whatever the window it was recorded at. On a stored kernel the re-hint
  never reaches those axes: a statement rebuild keeps an axis whose hint alone changed, since a hint takes no part in
  equality. Harmless to the picks, which compute the same 512, and to a golden walk, which counts a row at every size
  of its target when its kernel's buffers name none of the row's sizes. Not fixed.
- (2026-11-06) The 27B's native server leaves a median 2.6 ms between a prompt's 64-row chunk launches on an idle host
  (nsys at 7,680 tokens), about 8 s of the 197,632-token first token, and 3.7 ms while exports compiled on the host.
  Not investigated.

Open, compiler — the correctness items block 17:

- Correctness. The bf16 MMA schedule `WORK=w1x4, TILE=mma_m16n8k16_bf16_f32/f1x1/k8, STAGE=d4/smem-tma` on gdn16
  `db4f79` piece `place_6a3e623a98` gives one wrong FP4 code (flat index 86480: -6 vs -3) and its block scale;
  any single knob change is bit-identical to scalar. The golden pins `TILE=f1x2/k8` there. The non-strict
  wrong-answer check misses it: one bad element of 98304 stays under its mean-error threshold.
- Correctness. Rolling a recurrence misfires at small shapes: with 6 value heads and 8 or 12 tokens,
  `beta.transpose(1, 2).float()` is rolled by `005_roll_recurrence` into a 2-step "recurrence", leaving 3 extra
  kernels beside the fused conv parent; the real shape (48 heads, 64 tokens) is unaffected. Casting before the
  transpose avoids it. Not investigated.
- Correctness of the record flow. Tuning and recording never measure `SHARED_CARRY=1` for carried-loop kernels:
  `--tune` and `--record-greedy` left the 61-launch form of the GDN prefill triangular solve `3d261f` in a golden
  (572 against 46 us for one launch); its shared-carry rows are hand-recorded. Other carried loops may need the
  same check before 17's re-record.
- A cut arm that cannot be realized: the verify step's accept program with the later steps' M-RoPE positions as
  outputs fails in `030_cut`'s realization — `fold.lower` asserts "a store over ('v9',) writes values no one term
  defines". Native moved those outputs to the pick program; the region is a compiler bug to fix, not to route
  around again.
- `[N, 2]` pointwise loads stay 2 bytes wide because `(4a + 1) % 2` is not folded.
- The placement prior keeps a whole single-CTA remainder on the small Gated DeltaNet programs, and #1294 above is
  the attempt to price it. Measured on the dev Mac's CPU at sm_120 lowering, the tiny hybrid's one-row program: the
  prior cuts five small seams off the root and leaves the remainder at 8.6e9 serial trips per cell on one thread
  block, 736 s to rank the root fork's arms. #1303 (merged) changed what a bare `PLACE=cut` reaches — the pin cuts
  the remainder chain and only the pieces stay the prior's picks — and the 3-row verify program still keeps one
  piece of 1.3e8 trips per cell that way, so the pin alone does not close this. What moves the prior itself is a
  refit on measured cuts of such programs, or a feature that tells a piece's recompute from its size.
  (2026-11-02, #1290) Measured on main at #1310 with its refit priors, on a regenerated lane golden, each kernel
  replayed alone on an RTX 5090 (`emmy run --golden --kernel REF --realization ROW --bench`). #1294 and the refits
  closed the one-row programs (1.4 ms at most), and the 3-row verify program stays at 34 us. Six of the 11
  GDN programs held a kernel that hit the runtime's 62 s deadline, four of them on a single CTA: float32 width 4
  `k_linear_matmul_mean_reduce_b7f3d1__place_045f77378c__place_aef6553056` (grid 16 x 128); bf16 width 4
  `k_linear_matmul_mean_reduce_dcd742__place_668ca16ffb` (1 x 64) and width 16
  `k_linear_matmul_mean_reduce_0658d5__place_80fdaa9e9c` (1 x 512); float16 width 4
  `k_linear_matmul_mean_reduce_9a01eb__place_348985da70` (8 x 32), width 16
  `k_linear_matmul_mean_reduce_206776__place_463fccf612` (1 x 512) and width 2
  `k_linear_mean_reduce_0f91fc__place_08c61621a0#2` (1 x 256). So more CTAs alone do not close it.
  `S_ext_serial_cell_work` ranks the 3.46e13 and 2.71e11 kernels at the top, but the 1.34e8 one hangs while a 6.6e7
  one takes 205 ms.
  (2026-11-03, #1290) Closed by measured cuts, on a local RTX 5090 with a fresh tune DB. The replay of each kernel
  alone writes its `bench_fail` row; `EMMY_KNOBS=PLACE=cut run --golden --kernel REF --bench` repeated then cuts the
  piece that hung the round before (the failure rule takes keep-whole off it), two to three rounds each: the pieces'
  sets alone run in 34.3 ms (37), 4.4 ms (40), 60 us (41), 34.9 ms (43), 61 us (44) and 278 us (53). Program 44's
  chain needed one seam named by hand (`PLACE@place_4af66936f3=cut`): the piece it cuts runs 63 s, which the first
  launch's flat minute of watchdog grace lets through and the bench stage's 10 s budget then fails without naming
  a kernel. The whole targets, recorded strictly (`--realization SEED --bench --record-greedy`, seeded from a
  traced-only row on each root, which a record benches as no pinned comparison), run in 39.8 us (37), 39.2 ms (40),
  146 us (41), 69.8 ms (43), 145 us (44) and 265 us (53); the 35 ms single-CTA kernels left in 40 and 43 are the
  single-CTA item above. Program 37's whole-target compile does not take the chain its isolated piece measured (the
  piece sits under the file's `g16a` split, and in the program the cut recorded on its `c8670e0392` piece spells no
  offered arm), and once its descendants carried failure rows it kept re-hanging the split parent `045f77378c`; the
  isolated loop on that parent converged in two rounds, after which one more whole-target round found the 40 us set.
  Two tooling facts the records ran into: a strict record compiles the family's narrower twins unpinned from
  evidence, and the lane's rows are proposals, so the records ran on a working copy carrying the lane's own 1 us
  stand-in (`helpers.golden_document`) on every proposal, taken off again afterwards with the proposals of kernels a
  recorded set no longer runs (a stand-in would outprice the measured set in the tests); and `run --golden --program
  N --bench` segfaults in the bench worker on the width-4 chunk programs (the whole-layer Torch twin), where the
  realization path benches them. Each recorded row replays alone within its measurement. On the card the
  drafted-generation GDN test passes, the plugin's speculative engine test passes, and the plugin GDN engine test no
  longer stalls but its 5-token case generates wrong tokens with program 44's recorded set (the hand-named seam's
  chain, 145 us for a 16-row recurrence; the 1-token case, one real row, is right), so program 44's tree is back to
  its previous rows and that test stalls on it as before. The compiler lacks a numeric check of a recorded
  whole-layer set: a record's reference is the pick's own output, and the `--program` Torch twin segfaults; the
  repository's runner-shape accuracy cases (`test_gen_runner_gdn_gpu.py`, skipped for this hang) miss the same four
  cases on the previous golden as on this one, so they do not judge the cut sets. Next measured set for 44: the
  chain without the hand seam, its 63 s piece cut by the failure rule once `EMMY_KERNEL_TIMEOUT_MS` puts the
  first-launch grace under its run time. Merged with main at #1314 (59ae2eca1), the drafted GDN variant's last step
  generated one token where the test expects three. Not #1313's prefill rule: its nine-token prompt is the eight-token
  one and the attention model's first greedy token (3), and the hybrid's continuation of that prompt never repeats a
  token within 24, so no draft stands. The nine-token prompt is now the eight-token one and the model's own first
  greedy token (63 on the hybrid), whose tail repeats; the GDN variant's eight-token prompt is two whole 4-row chunks,
  so it covers the chunk that ends the prompt, as the attention variant's nine-token prompt does. Both variants pass
  on the card (33 min).
- (2026-10-31, #1309) The prior reproduction gate cannot go green on a promoted RTX 5090 hardware golden by a
  refit, so 17 needs prior and export work first. The promotion of every recipe row, with both priors refit at
  the shipped counts (schedule 400 iterations, placement 1,500, no folds), left 20 nodes red, all placement, all
  on seven kernels: `k_unsqueeze_3c3255`, `k_where_2_reduce` and `k_linear_matmul_mean_reduce_e63033__place_d476a18d79`
  (the prior picks fuse where the golden cuts), `k_linear_matmul_mean_reduce_29a8aa__place_2fe0da9b51` in three
  contexts and `k_conv1d_linear_mean_reduce_45cfea` in four (fuse or another cut set), and on the A100
  `k_linear_17ca5e` (split `g16k` where the golden records `g8k`, which main's weights reproduce). Two gaps. The
  placement dataset records both arms as golden on identical feature vectors — the same kernel fused in one
  program and cut in another (`k_where_2_reduce.001416a0cc19` fuse against `.aca2d3086a55` cut, likewise
  `k_unsqueeze_3c3255` and the `85bd73`/`e63033` place kernels), 72 such vectors in the dataset — so no fit
  reproduces both sides and the gate, which judges every program, misses one. And the export skips the conv1d
  `45cfea` pools as "recorded cut not reached", so the fit never sees pools the gate judges. Both recipe goldens
  hold these decisions on main under their `prior-pending` tags; a hardware golden's nodes the gate never skips.
  What 17 needs: a feature that separates the two programs' contexts (or one decision per kernel), and an export
  that reaches the recorded cut. The Llama 3.1 8B V100 SXM3 splits a default-count refit misses reproduce at the
  shipped counts; a refit at `emmy fit`'s defaults (200/500) underfits and reddens 28 nodes. #1310 promotes the
  rest. A piece's decisions cannot come out alone: the gate walks a root to the kernel set its recorded decisions
  leave, and without the piece's decisions the prior's own cut order carves pieces outside that set, so the five
  kernel trees come out whole (152 kernels, 78 decisions, 82 rows, 6 measured) and stay in the recipe goldens. The
  dataset a refit reads has to be exported after the goldens are final: weights fit before #1302's drafter rows
  reached the recipe goldens left four of their placement forks red once promoted, pools no fit had seen and the
  `prior-pending` tag had hidden from the gate; the re-export and refit closed them.
- (2026-10-31, #1302) The Qwen3.6 plugin lane decodes at about 0.3 tokens a second on the rented 5090s (202 ms a
  token on 2026-10-26): the card at 100% SM with no memory or PCIe traffic, py-spy in the routed-expert combine of
  the verify step. Not bisected.
- (2026-10-31, #1302) The Qwen3.6 plugin lane's first boot outlasts `emmy deploy`'s 3600 s health wait and needs
  more than 47 GiB of host memory or swap (7 compile workers beside 16.7 GiB of pinned expert pages).

## Record-run gotchas

Every record milestone above walks into the traps the `refresh-golden` skill lists under "Record-run traps". One is
this plan's own: the stored `gdn3rows` row stays whole until 17 re-records it (#1303).

## Risks

- **Qwen3.6's first-token cells may be copy-floor-bound.** The floor is the link, not a kernel: ~0.3 s of copies
  per full-coverage chunk (State), so the cells pass only at wide widths, and the window's longest prompt may stay
  above 2x stock however wide the chunk. If the floor holds, the options are the bar relaxed for those cells or
  the fallback checkpoint (16); neither blocks the switch, but the plugin floor does: native must first match the
  plugin's own first token (3.44 s at 256 tokens against native's 5.07 today), which 13 narrows and 14 is expected
  to pass, 15 holding it at the window.
- **The 27B's long-prompt first token needs the attention fold more than the widening.** The two-term budget in
  State: folding alone leaves ~271 s at 197,632, widening alone ~370 s, both ~136 s — the bar with no slack. If
  5's folded route misses its ~4.6x (the 2026-11-01 budget), the cell cannot be saved by width; if the wide-width
  kernels refuse to amortize
  (the paged gate-up precedent says schedules can refuse a new shape), the 256- and 7,680-token cells miss
  instead. 8 measures the chunk before 9 widens it, and 10 puts the result in front of the owner either way.
- **Long-window decode sits near the memory floor.** After the shared cache read (6), the budget between native
  and 1.5x stock at 197,632 tokens is a few milliseconds a step; a step today runs ~1,500-1,900 launches, and if
  per-launch overhead eats the budget, the cell misses. No milestone promises a launch-count rewrite; the risk is
  accepted and measured at 10.
- **The folded attention and the paged cache pull against each other.** 5 and 6 want tensor-core staging over the
  cache; 20 wants that cache paged, and paging takes TMA away and, over several pages, cp.async too (Decisions). One
  page per request keeps cp.async, at the cost of a compiler change in 20 that is unbuilt and of a batched attention
  in 21 that still reads several pages. If 21's one-launch-per-row form costs too much at concurrency 32 and the
  batched body has no fast legal schedule, the concurrency bar misses, and that needs 13's renderer change or the
  owner's revision at 25.
- **Continuous batching is the largest untested surface.** A scheduler in the Rust server, per-request cache and
  state, capture across varying widths, and sampling that today sorts a vocabulary per token on one core (23).
  The bar's concurrency numbers rest on milestone 1's stock columns and the owner's revision there. The switch
  cannot happen before this stretch lands — vLLM is what serves concurrent traffic today — so the whole goal's
  schedule risk concentrates here.
- **One promotion serializing on every compiler fix.** 17 waits for 2-16 and the follow-ups list's correctness
  items; a late fix re-opens it. The mitigation is AGENTS' own: recipes tag `prior-pending` where the gate needs
  it, and the promotion rents the cards once. Batching's second promotion (25) is deliberately scoped so 17 never
  re-records.
- **"Completely" is bounded by model coverage.** Retiring the plugin lane for these two recipes retires vLLM for
  them, not for the fleet; Gemma 4, the V100-fork models and the embedding recipes still serve through vLLM. If
  the owner's goal is the fleet, that is a model-coverage plan this one does not contain.
