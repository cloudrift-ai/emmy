# Native serving to parity with vLLM, and the switch

Goal, from the owner: "Have native serving working comparable to vllm so we can switch to it completely." The native
server replaces the vLLM plugin lane for the two maintained RTX 5090 recipes — `Qwen3.8-27B-NVFP4-NVIDIA` and
`Qwen3.6-35B-A3B-FP8` — once it meets the bar below, serves concurrent traffic, and ships through the recipes' images.
The 27B gates first: its gaps are kernel-level and the closer race; Qwen3.6's largest gap is structural (half its
experts live off the card). Continuous batching lives in this plan, as the stretch after single-request parity.

Written 2026-10-29 from the measured comparison tables in both recipes' DISCOVERYs (the stock-and-plugin session and
the NInfer session, merged through #1304), `emmy/serving/native/ARCHITECTURE.md`, and the open PRs. Revised the same
day, after #1289, #1295, #1303 and #1304 merged and the RTX 5090 tuning plan this plan grew out of was deleted: its
still-open items are the follow-ups list below, and its record-run traps are the gotchas. Each milestone here is one
PR, stacked on the one before where they touch the same files; a measurement milestone is a DISCOVERY PR.

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
unless milestone 19's design picks another split. The concurrency bar's numbers are fixed from the stock and plugin
columns milestone 1 measures (none exist in this line of work today); the proposal — per-request TPOT and total
output tokens per second each within 1.5x stock at concurrency 8 and 32 — is revised with the owner against that
measurement at milestone 1, not at the end of the stretch.

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
  store 9 ms on one CTA; attention at 64 rows writes scores and probabilities to memory and grows with the window
  (4.6 ms a layer at 65,536 positions). #1295 (merged) stopped the Gated DeltaNet out projection being computed
  twice under the block amax and re-recorded both NVFP4 goldens: the 512-token attention input target 883 -> 655 us,
  the 64-row out projection piece 44.9 -> 35.7 us.
- **27B short-prompt first token is explained, not uncertain.** The vLLM session read 1.34 s at 256 tokens, the
  NInfer session 0.45 s at its 268-token chat prompt, and both are the carried-state chunk rule on the same pack: a
  chunk takes only prompt tokens before the last, so 256 tokens left 63 single steps (63 x 18.3 ms) where 268 left
  11, and the session's 257-token control measured 246 ms. #1281 (merged 2026-10-26) now takes the tail in
  decode-width chunks: 694 ms at 256 tokens, 6.83 s at 7,680. Milestone 4 removes the remaining single steps.
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
not merged), and the tooling fixes #1283-#1286. Current main has not been measured on the protocol; milestone 1
does that once.

## The open PRs, placed

- **#1294, #1293** (open): the placement prior prices a kernel set's work; a cut arm's piece facts computed on the
  worker that builds the arm. #1294 stops the prior keeping single-CTA arms that recompute a chunk's solve per
  output element (the tiny hybrid's 1e17-statement remainder; the 33-minute cold width-4 picks) and carries its
  featurizer bump and placement refit, reproduction gate green — but its own body says neither the old nor the new
  pick could be timed on the rented card (the width-16 bench worker dies with SIGSEGV, width-4 exceeds the 5,400 s
  wall budget), so it merges on the refit gate, not a speed claim. #1293 measured 575 -> 548 s (5%) on a 3-core
  host and did not measure the width-16 program it was aimed at. Neither is a milestone, and no record milestone
  waits on them: evidence joins on exact identity, which neither changes.
- **#1290** (draft): the tiny GDN hybrid in the serving regen, capture and completion split apart, and the GPU
  tests it unblocks. #1303 (merged) was one blocker; the other is #1294 — the regenerated golden still holds, on 8
  of its 10 pinned GDN programs, a single-CTA remainder that does not finish in 62 s. The switch (26) needs its
  tests green.
- **#1298, #1299** (open): land before milestone 2, the first record milestone. #1299 changes the cold bench and
  the tune DB version; #1298 changes kernel sources (identities unchanged) beyond NVFP4 through its wider alignment
  proof. Neither is on the 5090 parity path (the 5090's deployed NVFP4 kernels read no pair table), but records
  taken before them would measure a bench and sources about to move.
- **#1302** (draft): milestone 18 — the native recipe lane; its images wait on 17's promoted rows.

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
   lane's; rows recorded in the recipe golden. Needs 1, and #1298/#1299 in.

3. **The rotary cache store takes a grid.** The store names the whole cache (a scatter into a windowed buffer
   leaves its store no grid) and runs on one CTA: rope64 ~9 ms of a 64-row chunk (0.58 ms a layer), rope3 36-43 us
   a layer (0.6 ms a verify step), M-RoPE +122 us a layer and chunk. A lowering fix — the chunk's store spreads
   over the rows and heads it writes — so kernels re-key: restamp, re-record the rope rows, both recipes. Gate:
   the 27B chunk 50 -> ~41 ms measured in the step; the verify step down ~0.5 ms; corpus and goldens current.
   Needs 1.

4. **Prefill ends at the prompt's last token.** After #1281 a 256-token prompt still runs 15 single steps on the
   27B (3 on Qwen3.6): a chunk takes only prompt tokens *before the last*, because only the decode program runs
   the head. A drafted pack's chunks already run their last layer whole and the final norm; let the final chunk
   include the last prompt token and run the head and greedy selection over its last normalized row (a row pick,
   as the verify step's pick program does). 256 tokens then cost four chunks — the 257-token control's 246 ms.
   Gate: 27B first token at 256 tokens within one chunk's cost of the 257-token control; tokens unchanged on the
   probes; both models. Needs nothing.

5. **Long-window prefill attention folds chunk by chunk.** The 64-row attention writes its scores and
   probabilities to memory and runs the value expectation far below bandwidth: 4.6 ms a layer at 65,536 positions,
   the ~351 s attention term of the 505 s first token at 197,632. The tower attention already runs the folded form
   — the mask a term of the score's own contraction, the expectation folding chunk by chunk on tensor cores,
   FlashAttention's shape — so spell the text prefill attention over the paged cache the same way and record the
   route. Gate: at 65,536 positions at least 3x faster per layer; the attention term of the 197,632-token first
   token <= 117 s (its share of the two-term budget; today ~351 s); the full-window prompt's first token <= 450 s
   before 9 widens the chunk; the 4,096-window cells unchanged. Needs 1.

6. **Long-window decode attention: one cache read per key-value group, a combine that does not rescan the
   window.** The 27B step grows +48 ms from 7,680 to 197,632 positions; stock grows +4. Two named causes: the
   attention program runs per query head, so each key-value head's cache is read six times, and the partial
   scaling after the 32-way split grows with the window. Re-spell native's attention module per key-value head
   (the group's query heads share the read) and make the split's combine run over the partials, not the window.
   The module is shared, so this also owns Qwen3.6's window TPOT cell (33.1 ms against a 13.7 bar; 16 confirms):
   new programs, recorded at one and three rows for both models. Gate: the captured 27B step at 197,632 positions
   within ~8 ms of the 4,096-window step (about one cache read at bandwidth, ~4 ms, plus margin; today +48);
   Qwen3.6's window step bounded the same way at its own cache's read time, the number fixed from 1's split;
   short cells unchanged. Needs 1, 5.

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
    gate-up piece is offered no tensor-core tile (TMA cannot read a page; cp.async takes a masked N since #1261,
    which the down projection already stages through). Extend the offer so a paged operand stages as the resident
    twin's does, and re-record the expert rows at the schedules the paged export offers. Gate: the gate-up piece
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
    code and the wrong-answer threshold, the dead scratch, the nested stamps, the loose rounding step, the
    recurrence roll, the unmeasured `SHARED_CARRY`) land before this, so the re-record measures the fixed
    kernels; the other cards' open record runs (the V100/A100/H100/4090 GDN routes) stay follow-ups, not this
    plan's scope. Gate: the prior reproduction gate green with no `prior-pending` left on these recipes; the
    weights committed. Needs 2-16.

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
    reuse and preemption stay out. Lands as architecture text first, as the paged-experts design did. Gate: the
    owner approves; the design names which programs survive unchanged, which widen, and which rows are new.
    Needs 10.

20. **Per-request cache regions and reclamation.** Today the KV cache is one key and one value buffer per layer,
    sized to the window and allocated at load, and the Gated DeltaNet state is one set of carried buffers — one
    request's. Give each admitted request a cache region (pages along positions, as 19 designs) and a state block,
    allocated at admission and reclaimed at retirement; the window-sized allocation does not grow. A runtime and
    pager change, no new kernel expected: rotary and attention already read positions from device memory. Gate:
    one request through a region is bit-identical to today (tokens and logprobs); a scripted admit-and-retire
    sequence across regions leaks nothing (device memory at rest returns to its baseline); capture replays.
    Needs 17, 19.

21. **Batched decode at a fixed width.** N requests, one row each, through the existing decode-bucket programs,
    each row over its own cache region and state block from 20, rows padded as the plugin pads, per-row
    positions. Gate: N=1 bit-identical to today; each request's greedy tokens equal its solo run; TPOT per
    request at N=4 and 8 beside N solo runs; capture replays. Needs 20.

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
    from 17 re-records. Gate: the concurrency bar met or each miss named with its reason, and native at or above
    the plugin lane's measured concurrency cells; the DISCOVERY tables carry native's concurrency columns;
    reproduction gate green. Needs 23, 24.

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

Milestone 1 needs nothing; #1298 and #1299 land before 2, the first record milestone. The 27B stretch: 2 and 3
need 1; 4 needs nothing; 5 needs 1; 6 needs 1 and 5; 7 needs 2 and 3; 8 needs 1 and shares its session; 9 needs 3,
4, 5 and 8; 10 needs 2-9. The Qwen3.6 stretch runs beside it (the owner's gate order holds: 10 is decided before
16): 11 needs 1; 12 needs 1 and 11; 13 needs nothing and lands before 14; 14 needs 11 and 13; 15 needs 14; 16
needs 6 and 11-15. 17 needs 2-16 and the follow-ups list's compiler correctness items; 18 needs 17. The batching
stretch: 19 needs 10 and may start beside 11-16; 20 needs 17 and 19; 21 needs 20; 22 needs 21; 23 and 24 need 22;
25 needs 23 and 24. 26 is last and needs 18 and 25.

## Card time

The record and measurement milestones share rented x16 RTX 5090 sessions instead of renting one each:

- Milestone 1's two protocol sessions (one per model) also take the nsys splits, the 27B first-token controls and
  8's per-kernel chunk profile.
- 2, 3 and 5 record in one rental: the head rows, both recipes' rope rows (3 re-keys Qwen3.6's rope too), the
  folded prefill attention route. 6 and 7 share a second: decode attention at one and three rows for both models,
  then the width-3 GDN tune in the step. 9's wide-width records are a third, after 8's floor is read.
- Qwen3.6's 13 and 14 share one rental; 15's overlap check rides it when the runtime lands in time, else it takes a
  short rental of its own before 16's gate session.
- 10 and 16 are each a fresh protocol session per model (the bar requires one session, all columns).
- 17 rents once per card; the batching stretch records only new programs and promotes once, at 25.

Rented 23.5 GiB hosts, known traps: an export with swap off needs `vm.overcommit_memory=1` or the compile's forked
arm builds fail with ENOMEM; transparent hugepages and proactive compaction off beside the pinned expert pages (the
compaction thread otherwise runs a core at 100%); few compile workers fit beside 16 GiB of pinned pages.

## Decisions

- **The margins (1.2x / 1.5x / 2x) are this plan's definition of "comparable"** until the owner revises them. They
  come from the measured tables: the 27B's drafted TPOT is at 1.2x on #1282's probes (about 10.5 ms at the
  protocol's 68% acceptance — close, not made), the long-window TPOT floor is about one cache read above stock,
  and 2x first token is reachable on the 27B with the chunk work while staying honest about Qwen3.6 (see Risks).
- **NInfer is context, not the bar.** Different checkpoints and quantizations; its value is proving the card's
  headroom at each cell.
- **The 27B gates first** (owner's call); Qwen3.6 work runs beside it but its bar decision (16) comes after 10.
- **New twins trace under the serving venv's transformers (5.14.1).** Under 5.19 the GDN layer traces differently
  (q/k norm in f32 in another layout, a divide by sqrt(128)), so a 5.19 re-trace would re-key ~260 GDN rows for no
  speed; serving is what has to beat vLLM. No 5.19 re-trace rides in on this plan.
- **Slower measured rows stay beside the faster ones** in every golden, as the canonical goldens keep them; a
  strict compile picks the fastest matching row either way. Routing rows stay one per arm.
- **Decode matmul rows carry only schedules that also win with a cold cache** (the gotchas' hot-bench trap).
- **NVFP4 decode tables stay as they are.** Every deployed NVFP4 linear of the 5090 goldens runs W4A4 on the
  block-scaled fp4 MMA; no deployed kernel reads the 256x2 pair table, which is why #1298 is off the parity path.
  A table-free decode would matter only if a W4A16 schedule ever wins a pick.
- **Prefix caching stays out** of every comparison and out of the batching scope, as the measured sessions ran
  (`--no-enable-prefix-caching` on stock).
- **Two promotions, not one.** The single-request promotion (17) does not wait for batching: the stretch's
  programs do not exist yet, and waiting would stall parity on months of scheduler work. Batching's rows are new
  programs, promoted once at 25; nothing from 17 is re-recorded by it.
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

Open, compiler — the correctness items block 17:

- Correctness. The bf16 MMA schedule `WORK=w1x4, TILE=mma_m16n8k16_bf16_f32/f1x1/k8, STAGE=d4/smem-tma` on gdn16
  `db4f79` piece `place_6a3e623a98` gives one wrong FP4 code (flat index 86480: -6 vs -3) and its block scale;
  any single knob change is bit-identical to scalar. The golden pins `TILE=f1x2/k8` there. The non-strict
  wrong-answer check misses it: one bad element of 98304 stays under its mean-error threshold.
- Correctness. The cut pass's dead scratch at width 4: one pin combination made a piece whose output nothing
  reads; #1243 dropped the pin instead of fixing the pass.
- Correctness. Nested `TILE`/`STAGE` stamps under an output-tiled root are accepted by the binder and silently
  dropped (gdn16 recurrence piece `...__place_84c2d13867__place_149fbfea8c__partial` runs serial per-cell loops).
  Likely fix: offer a nested tile only where the binder realizes it; refuse the rest at the offer.
- Correctness. The strict rule's rounding step comes from the coarsest float format in the producing chain with
  no list of which ops round: a bf16 slice cast to f32 and followed by f32-only math still gets the bf16 step.
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

## Record-run gotchas

Carried from the record runs behind the tables; every record milestone above walks into these:

- `emmy golden check` never re-traces: "current" means the stored programs still lower to the stored kernels, not
  that a fresh trace gives those programs.
- A strict record also benches the named realization's own stored row as a pinned config, and that row's schedule
  knobs replay as bare, unscoped pins on every kernel of the compile. Name a row whose knobs suit every piece (the
  big piece's own row).
- `--pin-route` on a target root's own row that has no latency pins nothing; name one of the route's piece rows as
  `--realization`.
- `GoldenFile.path_to` takes the first route that reaches a piece; a pin spelled along a sibling root's path names
  no seam. Spell pins along the target's own decisions.
- A piece with one CTA and a multi-ms or hung time has lost its split decision (a restamp can drop producer
  decisions). Cut its seams by hand with `PLACE@place_<token>/<seam>=cut`.
- Since #1303 a recorded root-most cut replays as `PLACE@step.0=cut`, and under strict evidence a record stops at a
  remainder's own fork by name instead of fusing it. The stored `gdn3rows` row stays whole until 17 re-records it.
- `--record-greedy` spells every precision gate in the new rows' pins; the golden rows carry `{"FAST_MATH": lane}`
  alone. The audit reads both spellings as one lane.
- A whole-target bench now and then falls back from CUDA graph capture and a strict record refuses it; the failed
  capture is the eager backend's, and a rerun usually records.
- `run --kernel <cut FP4 piece>` fills the piece's FP4 decode tables with random values, so tile rows fail
  correctness there; check candidates through the whole target (`--realization <target row> --pin-route --strict`).
- "benchmark run stage exceeded 10.0s of GPU time": a multi-ms pick in the pinned or greedy set; replace it with a
  kernel-scoped schedule pin, or `--warmup 2 --iters 5` for a record-only run.
- "unreproducible pin": a hand schedule pin clashes with a pinned golden row of the same realization; drop the
  hand pin.
- `--cold-cache` bench times step by ~2.05 us, too coarse to rank decode matmul schedules; cold A/B runs go under
  nsys per-launch durations.
- Hot benches of weight readers keep the weights in the 5090's 96 MiB L2 (gate_up: 25 us hot, 68 cold); a hot
  `--tune` ranks schedules that gain nothing cold. Decode rows carry cold winners only (Decisions).
- After a merge that re-keys pieces, a re-keyed piece can lose its only measured row or keep a schedule the
  re-formed piece no longer offers ("realized (off)"); re-record it on an offered schedule.
- A kernel launched early under programmatic dependent launch must not read an earlier kernel's output through
  the read-only path (`const __restrict__` compiles to `LDG.E.CONSTANT`, which `griddepcontrol.wait` does not
  make coherent); only program constants keep it under early release.

## Risks

- **Qwen3.6's first-token cells may be copy-floor-bound.** The floor is the link, not a kernel: ~0.3 s of copies
  per full-coverage chunk (State), so the cells pass only at wide widths, and the window's longest prompt may stay
  above 2x stock however wide the chunk. If the floor holds, the options are the bar relaxed for those cells or
  the fallback checkpoint (16); neither blocks the switch, but the plugin floor does: native must first match the
  plugin's own first token (3.44 s at 256 tokens against native's 5.07 today), which 13 narrows and 14 is expected
  to pass, 15 holding it at the window.
- **The 27B's long-prompt first token needs the attention fold more than the widening.** The two-term budget in
  State: folding alone leaves ~271 s at 197,632, widening alone ~370 s, both ~136 s — the bar with no slack. If
  5's folded route misses its 3x, the cell cannot be saved by width; if the wide-width kernels refuse to amortize
  (the paged gate-up precedent says schedules can refuse a new shape), the 256- and 7,680-token cells miss
  instead. 8 measures the chunk before 9 widens it, and 10 puts the result in front of the owner either way.
- **Long-window decode sits near the memory floor.** After the shared cache read (6), the budget between native
  and 1.5x stock at 197,632 tokens is a few milliseconds a step; a step today runs ~1,500-1,900 launches, and if
  per-launch overhead eats the budget, the cell misses. No milestone promises a launch-count rewrite; the risk is
  accepted and measured at 10.
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
