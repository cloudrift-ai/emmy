# Qwen3.8-27B-NVFP4 (NVIDIA) discovery

## One cache read per verify step, and the golden measured again (2026-11-12)

A verify step's attention now folds its three rows beside the query group, so every row shares one read of the
key-value cache; before, each row read every slice. Measured on one rented RTX 5090 (CloudRift; 14 vCPUs of a Ryzen 7
7800X3D, 21.5 GiB of RAM with swap on, PCIe Gen5 x16, driver 580.159, CUDA 13.3), one session, native and stock
vLLM 0.32.0 both. The client and settings are the 2026-10-25 section's; the prompts are cut again from this tree's
Markdown, so acceptance differs from the earlier sessions'. The pack exports under strict evidence from the golden
alone.

**The golden first.** On main the file held 225 measured rows of 415: #1365's normal form re-keyed the rest, and a
strict export stopped at its first fork. This session measured the demoted schedules again on the card (the file now
holds 375 measured rows of 412) and recorded the new verify programs. Three things needed more than a re-measure:

- **A split attention's combine kernel is priced at its window.** An export prices each range of windows at its own
  window, and a record stores the combine kernel's row at the stored program's hint. With that row at another size the
  split's arm has no price, the unsplit kernel's row wins, and nothing fails: a pack exported that way ran its decode
  attention unsplit (389 µs a layer against 11) and gave 10.6 ms a token at 256 tokens and 6.1 s to the first token
  at 7,680. Every attention twin's split is recorded again at its window, and each combine row is stored under the
  window it was measured at.
- **One Gated DeltaNet piece of the 64-row program had no row.** The compile's own pick for it takes 336 µs a layer;
  the schedule its sibling pieces carry takes 7.7 µs and is the row recorded.
- **Two schedules no longer realize** (a 64-row Gated DeltaNet projection piece and one alternative of the three-row
  program). The first is recorded at the schedule the compiler realizes in its place, 49.5 µs; the second is left a
  proposal, beside the plugin's any-width `pre` piece, which native does not run.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 10.7 µs (11.1) | g64k, 183 µs (205) |
| `attend3` (verify), rows folded | g32k, 12.5 µs (22.5) | g32k, 183 µs (487) |
| `attend16` | g16k, 39.1 µs (41.7) | g32k, 852 µs (974) |
| `attend64` (chunk) | g8k, 78.6 µs (73.0) | g32k, 1,601 µs (1,590) |

The 2026-11-07 times, on another card, are in brackets. The verify step's other new programs take 2.3 µs (the rotary
store, which now hands the query out per key-value head) and 0.7 µs (the unfold that puts the rows back in order).

| | native, this branch | stock vLLM 0.32.0 | native on 2026-11-07, another card |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.84 ms / 0.143 s | 7.87 ms / 0.159 s | 8.68 ms / 0.151 s |
| first token, 257-token prompt | 0.165 s | 0.127 s | 0.172 s |
| TPOT / TTFT, 7,680-token prompt | 8.72 ms / 4.37 s | 7.73 ms / 0.753 s | 8.55 ms / 4.62 s |
| TPOT / TTFT, 197,632-token prompt | 11.4 ms / 172 s | 9.54 ms / 67.9 s | 14.7 ms / 188 s |
| TPOT / TTFT, 261,632-token prompt | 11.4 ms / 251 s | does not fit | 17.7 ms / 277 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 63 / 77 / 68 / 75% | not read: its stream spaces every token | 75 / 79 / 71 / 64% |
| device memory at rest | 27.9 GiB | 30.4 GiB | 27.9 GiB |
| GSM8K strict / flexible match | 0.725 / 0.755 | | 0.715 / 0.765 |

- **Against this session's stock column.** TPOT is inside the bar at 7,680 tokens (1.13x stock, the bar 1.2x) and
  at 197,632 (1.20x, the bar 1.5x) and outside it at 256 (1.25x). The first token is ahead of stock at 256 tokens and
  behind at 7,680 (5.8x, the bar 2x) and at 197,632 (2.5x). Native holds 2.5 GiB less of the device and serves the
  whole window, which stock does not fit.
- **The long-window decode gate, met.** The captured verify step takes 26.4 ms at 197,632 positions (the target's
  attention 4.5 ms, the drafter's 0.5) against 21.6 ms at a window of up to 8,192 (attention 0.35 ms): +4.8 ms, where
  the bound is 8 ms and the 2026-11-07 step was +14.4.
- **The short cells.** The step is 21.6 ms where 2026-11-07 captured 21.1; TPOT at 256 tokens reads higher here
  because these prompts accept fewer drafts (2.26 tokens a step).
- **GSM8K holds:** 0.725 strict and 0.755 flexible match on native (standard error 0.032), against 0.715 / 0.765 on
  2026-11-07. Stock was not scored in this session, and the probe tokens were not compared.

Where the time goes, from `nsys --cuda-graph-trace=node` (one request at 7,680 tokens and one at 197,632), each
launch priced at its kernel's median:

| launches | 64-row chunk | verify step, up to 8,192 | verify step, 197,632 |
|---|---|---|---|
| Gated DeltaNet layers (48) | 27.7 ms | 14.3 ms | 14.4 ms |
| attention layers: `post` (16) | 3.0 ms | 2.2 ms | 2.3 ms |
| attention layers: `pre` (16) | 2.8 ms | 1.7 ms | 1.7 ms |
| attention | 1.2 ms | 0.35 ms | 4.5 ms |
| output head (one row; a verify step runs it three times) | 0.45 ms | 1.3 ms | 1.3 ms |
| the drafter's layer and prologue | 0.44 ms | 1.4 ms | 1.9 ms |
| state carry, rotary store, unfold, the rest | 0.3 ms | 0.3 ms | 0.3 ms |
| sum | 35.9 ms | 21.6 ms | 26.4 ms |

The 197,632-token first token is 171 s of kernels: 86 s in the Gated DeltaNet layers, 62 s in attention (69.9 s on
2026-11-07), 18 s in the attention layers' halves. A 64-row chunk's Gated DeltaNet layers take 577 µs each; the largest
launches are the MLP gate and up (74 µs), the convolution piece (71 µs), `k_slice_unsqueeze_reduce` (59 µs) and the
MLP down (54 µs).

**A 256-row chunk does not load yet.** Traced at 256 rows the Gated DeltaNet layer has twelve root kernels where the
64-row one has four: the recurrence across its four 64-token chunks adds eight. No golden row covers them, and the
compile's own cuts leave four workspaces of `[4, 64, 64, 48, 48, 128]` FP32 a layer, 18 GiB each, so the pack runs out
of device memory at load. Under the 64-row kernel's recorded route one piece of the 256-row kernel does not finish in
80 s. A wide chunk therefore needs its own recorded route before it can be timed, and no time for one is in this
section.

## The FP8 cache staged into shared memory: milestone 5 met, 6 still missed (2026-11-07)

Native attention's FP8 key-value cache now copies into shared memory, 16 bytes per copy, before the tensor-core fold
reads it; it was read one byte per load, which kept every fold at 255 registers with spills (#1366, on #1352). Recorded
and gated on the card and host of the section below (vast.ai, x16 RTX 5090, EPYC 9655, a 46-CPU quota, driver
610.57). Every attention twin was swept over its neighbouring key splits and four tiles with the cache staged, and the
compile's pick recorded. The client, prompts and settings are the section below's. The pack exports under strict
evidence, every twin compiles strictly from the golden, and `emmy eval golden --serving-config` deploys every twin from
the golden rows alone.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 11.1 µs (27.7) | g64k, 205 µs (581) |
| `attend3` (verify) | g32k, 22.5 µs (54.9) | g64k, 487 µs (1,413) |
| `attend16` | g16k, 41.7 µs (141) | g32k, 974 µs (6,516) |
| `attend64` (chunk) | g8k, 73.0 µs (195) | g32k, 1,590 µs (5,126) |

The section below's times are in brackets. The one-row decode's split piece reads its 268 MB cache in 189 µs, about
1.4 TB/s; at one warp per CTA it took 170 µs staged against 502 µs read directly. g128k, which the compiler does not
offer, measured 2% faster than g64k at one row and 7% at three.

| | this branch | the section below |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 8.68 ms / 0.151 s | 9.44 ms / 0.154 s |
| first token, 257-token prompt | 0.172 s | 0.177 s |
| TPOT / TTFT, 7,680-token prompt | 8.55 ms / 4.62 s | 9.72 ms / 4.88 s |
| TPOT / TTFT, 197,632-token prompt | 14.7 ms / 188 s | 23.7 ms / 318 s |
| TPOT / TTFT, 261,632-token prompt | 17.7 ms / 277 s | 32.7 ms / 505 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 75 / 79 / 71 / 64% | 70 / 71 / 79 / 62% |
| device memory at rest | 27.9 GiB | 27.9 GiB |
| GSM8K strict / flexible match | 0.715 / 0.765 | |

- **Milestone 5, met.** The attention term of the 197,632-token first token is 69.9 s of GPU time (`nsys
  --cuda-graph-trace=node`, 3,092 chunks), against at most 117 s (200 s in the section below). The full window's first
  token is 277 s, against at most 450 s.
- **Milestone 6, missed.** The captured verify step takes 35.5 ms at 197,632 positions (attention 14.3 ms) against
  21.1 ms at the 4,096 window (attention 0.41 ms): +14.4 ms, where the bound is about 8 ms (+38.9 in the section
  below). The three-row verify reads the cache once per row, three reads where one would do: its twin at 131,072 takes
  487 µs against the one-row decode's 205.
- **The short cells did not get slower.** The 256- and 7,680-token cells are faster on both TPOT and first token.
- **GSM8K holds,** inside one standard error (0.032) of main's 0.730 / 0.750 on 2026-11-06; the section below did not
  run it.
- Drafts accepted is the share of drafted tokens a step kept, read off the gaps between streamed tokens.

## Attention priced per range of windows: milestones 5 and 6 again (2026-11-07)

Native attention now compiles once per range of windows, priced at 4,096 and at 131,072 cached tokens, and a step runs
its window's range's program; the long range starts at 16,384 (#1352, stacked on #1329's grouped decode attention).
Recorded and gated on one rented x16 RTX 5090 (vast.ai; Gen5 x16; a 46-CPU quota on a 96-core EPYC 9655 host, driver
610.57, CUDA 13.1). The client, prompts and settings are the 2026-11-06 section's, the prompts cut from that section's
tree. The pack exports under strict evidence.

Every attention twin was swept whole and under key splits g2k to g128k, the FP8 cache unstaged, no TMA, fast math on,
and the compile's pick recorded. The compiler offers splits up to g64k; g128k measured faster on the decode twins at
131,072 (one row 475 against 581 µs, three rows 1,296 against 1,413 µs), and no compile can deploy it.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 27.7 µs | g64k, 581 µs |
| `attend3` (verify) | g16k, 54.9 µs | g64k, 1,413 µs |
| `attend16` | g8k, 141 µs | g8k, 6,516 µs |
| `attend64` (chunk) | g16k, 195 µs | g32k, 5,126 µs |

| | this branch | 2026-11-06 (main) |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.44 ms / 0.154 s | 8.70 ms / 0.15 s |
| first token, 257-token prompt | 0.177 s | 0.17 s |
| TPOT / TTFT, 7,680-token prompt | 9.72 ms / 4.88 s | 9.76 ms / 4.81 s |
| TPOT / TTFT, 197,632-token prompt | 23.7 ms / 318 s | 42.9 ms / 429 s |
| TPOT / TTFT, 261,632-token prompt | 32.7 ms / 505 s | 50.6 ms / 687 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 70 / 71 / 79 / 62% | 76 / 75 / 61 / 67% |
| device memory at rest | 27.9 GiB | 27.9 GiB |

- **Milestone 5, missed.** The attention term of the 197,632-token first token is 200 s of GPU time (`nsys
  --cuda-graph-trace=node`, 3,088 chunks), against at most 117 s (300 s on 2026-11-06). The full window's first token is
  505 s, against at most 450 s (687 s).
- **Milestone 6, missed.** The captured verify step takes 60.7 ms at 197,632 positions (attention 39.6 ms) against
  21.9 ms at the 4,096 window (attention 1.1 ms): +38.9 ms, where the bound is about 8 ms (+73.7 on 2026-11-06).
- **The short cells did not move.** The drafted step at 256 tokens takes 21.9 ms on both trees; TPOT reads higher here
  because fewer drafts were accepted on these prompts, which the attention's changed numerics move.
- **What is left is the kernels, not the pricing.** Every attention fold here runs at 255 registers with local memory
  spills and 17% occupancy. The one-row decode at 131,072 reads 268 MB of cache in 581 µs, about 0.46 TB/s; the 64-row
  chunk takes 5.1 ms there, where reading the cache once per query head at the card's bandwidth takes under 1 ms.

## The gates of milestones 3, 4 and 5 (2026-11-06)

The gates of the parity plan's milestones 3 (the rotary store's grid), 4 (prefill ends at the prompt's last token) and 5
(the folded prefill attention), native only, on one rented x16 RTX 5090 (vast.ai; Gen5 x16; a container with a 46-CPU
quota on a 384-CPU EPYC 9B14 host, driver 615.71, CUDA 13.1). Main is at `f33a86522`, after #1324 recorded the rope,
row-pick, drafter-token and attention twins. The client, prompts and settings are the 2026-11-01 section's. Two
reference packs ran on the same card, from the tree before 4 (#1311, `504165b13`) and the tree before 5 (#1318,
`57fabd6e2`); neither is strict, since rows later PRs recorded are missing from their goldens. Main's column is a
second pass on an idle host. The first pass, while pack exports compiled on the host, gave the same numbers within 4%
(413 and 684 s at the two long prompts).

On the card, `emmy eval golden --serving-config` deploys all 62 twins from the golden rows alone, the pack exports
under strict evidence, and `emmy golden check` says current, 517 kernels.

| | #1311 tree | #1318 tree | main | 2026-11-01 |
|---|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.65 ms / 0.70 s | 9.31 ms / 0.17 s | 8.70 ms / 0.15 s | 9.8 ms / 0.71 s |
| first token, 257-token prompt | 0.24 s | 0.19 s | 0.17 s | 0.24 s |
| TPOT / TTFT, 7,680-token prompt | | | 9.76 ms / 4.81 s | 10.5 ms / 7.39 s |
| TPOT / TTFT, 197,632-token prompt | | | 42.9 ms / 429 s | 26.5 ms / 698 s |
| TPOT / TTFT, 261,632-token prompt | | | 50.6 ms / 687 s | 41.9 ms / 1,154 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 71% at 256 | 73% at 256 | 76 / 75 / 61 / 67% | 70 / 66 / 88 / 57% |
| device memory at rest | 30.7 GiB | 30.7 GiB | 27.9 GiB | 30.7 GiB |
| GSM8K strict / flexible match | | | 0.730 / 0.750 | 0.720 / 0.755 |

The gates:

- **Milestone 4, met.** At 256 tokens, four 64-row chunks, the first token comes in 0.15 s against the 257-token
  control's 0.17 s; the tree before it took 0.70 s, its control 0.24 s. The trees before and after #1313 give the same
  64 greedy tokens on all four probes (`scripts/validate_serve.py`'s), on both models.
- **Milestone 3, met.** In the captured step the rotary store takes 0.09 ms of a 64-row chunk, against 11.9 ms on
  2026-11-01, and the chunk takes 38.6 ms at up to 4,096 positions, against 54.1 ms. The verify step's three-row store
  takes 0.08 ms of the step, against about 0.6 ms.
- **Milestone 5, missed.** The attention term of the 197,632-token first token is 300 s of GPU time, against at most
  117 s (544 s on 2026-11-01), and the full window's first token is 687 s, against at most 450 s. A layer at 65,536
  positions takes 3.66 ms against 6.7 ms: 1.8x, where the budget needs 4.6x. The cells within 4,096 positions did not
  get slower. GSM8K holds, inside one standard error of the plugin lane's 0.720 / 0.770 on 2026-11-01.

Where the time goes, from `nsys --cuda-graph-trace=node` on main (7,680 and 197,632 tokens; first token within 7% of
the unprofiled runs), the median launch in milliseconds:

| | 64-row chunk, up to 4,096 | 64-row chunk, 4,096 to 7,680 | 64-row chunk, all of 197,632 (mean) | verify step, 7,680 | verify step, 197,632 |
|---|---|---|---|---|---|
| rotary store | 0.09 | 0.09 | 0.09 | 0.08 | 0.08 |
| attention | 3.07 | 5.98 | 97.2 | 2.90 | 75.1 |
| the rest | 35.9 | 35.9 | 36.0 | 20.8 | 20.8 |
| the launch, start to end | 38.6 | 41.5 | 132.9 | 23.4 | 95.6 |

- **The 64-row fold takes the same split at every window.** Its rows were measured at 512 cached tokens, where the
  4-way key split wins, so a chunk runs that split in 192 blocks however many positions it holds: 0.37 ms a layer at
  4,096 positions, 3.66 ms at 65,536 and 12.3 ms at 197,568, where each block streams about 49,000 keys. Whether a
  wider split wins at long windows was not measured. The fold also frees 2.8 GiB of device memory, the old route's
  scores and probabilities.
- **Decode attention got slower at long windows.** The verify step's three-row attention runs as a score cut and a
  16-way key split of the rest, also picked at 512 cached tokens: 0.17 ms a layer at 7,700 positions and 4.3 ms at
  197,632, where it is 75 ms of a 95.6 ms step. The drafted step at 197,632 took 73 ms on 2026-11-01, so TPOT there went
  from 26.5 to 42.9 ms, and at the window's longest prompt from 41.9 to 50.6 ms. Milestone 6 re-spells this program.
- **The host leaves 2.6 ms between a prompt's chunks** (median, nsys at 7,680 tokens on an idle host), about 8 s over
  the 3,088 chunks of a 197,632-token prompt, and 3.7 ms while exports compiled on the host. Not investigated.
- **Probe tokens.** Main and the #1318 tree part at tokens 46, 9, 5 and 4 of the four probes' 64; each pack repeats its
  own tokens. Between them differ the fold and #1322's W4A4 head, which alone parted one probe at its second token.
  Where they part, both continuations read well ("a young boy named Tom" against "a young man named Tom"). Native
  returns no log-probabilities, so the margin at each parting point is not measured.

## Native, plugin and stock vLLM again, with concurrency and the chunk's kernels (2026-11-01)

The protocol of the 2026-10-25 table below, on main at `ea17c0964`, one session on one rented x16 RTX 5090 (Gen5 x16, 57
GB/s under load; 16 cores, 60 GiB of RAM, no swap, driver 595.84, CUDA 13.1, a container with no Docker).

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.8 ms / 0.71 s | 10.3 ms / 0.19 s | 7.7 ms / 0.085 s |
| TPOT / TTFT, 7,680-token prompt | 10.5 ms / 7.39 s | 10.4 ms / 5.09 s | 7.6 ms / 0.63 s |
| TPOT / TTFT, 197,632-token prompt (stock's longest) | 26.5 ms / 698 s | 11.9 ms / 180 s | 8.0 ms / 67.4 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 41.9 ms / 1,154 s | 15.7 ms / 260 s | does not fit |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 70 / 66 / 88 / 57% | 71 / 71 / 78 / 54% | 68 / 71 / 91% / - |
| TPOT with the drafter off, 256 / 7,680 | 16.2 / 16.5 ms | | |
| first token at 256 / 257 / 268 tokens | 0.71 / 0.24 / 0.42 s | | |
| device memory at rest / peak | 30.7 / 31.2 GiB | 29.9 / 30.7 GiB | 30.5 / 30.6 GiB |
| KV cache | 262,144 tokens, the window | 305,585 tokens at 0.93 | 201,383 tokens at 0.93 |
| host memory of the server at rest, peak | 3.4 GiB, 5.1 GiB | 11.9 GiB, 12.1 GiB | 5.5 GiB, 5.7 GiB |
| GSM8K strict / flexible match | 0.720 / 0.755 | 0.720 / 0.770 | 0.585 / 0.640 |
| ready after | 5 s from its pack | 1,500 s, no pack | 47 s, 270 s at its first compile |

Stock and the plugin at higher concurrency, with `--max-num-seqs` raised and nothing else changed but stock's window,
which 198,400 tokens no longer fit at these row counts (176,000 tokens here):

| | stock, 8 | stock, 32 | plugin, 8 | plugin, 32 |
|---|---|---|---|---|
| 256-token prompts: per-request TPOT | 9.2 ms | 10.1 ms | 87.7 ms | 168.9 ms |
| total output tokens a second | 756 | 1,070 | 75 | 69 |
| first token | 0.14 s | 4.75 s | 4.26 s | 74.4 s |
| 7,680-token prompts: per-request TPOT | 19.3 ms | 24.2 ms | 143.8 ms | 147.9 ms |
| total output tokens a second | 276 | 281 | 6.9 | 6.6 |
| first token | 1.42 s | 22.0 s | 261 s | 1,201 s |
| device memory peak | 30.6 GiB | 30.6 GiB | 30.8 GiB | 30.8 GiB |

Each cell runs 4 x concurrency requests at 256 tokens and 2 x concurrency at 7,680, each request a different slice of
the same Markdown. First token is the median request's, queueing included. The plugin keeps the recipe's settings, which
name one request; what stops it batching was not investigated.

What differs from the 2026-10-25 session:

- **The lanes are pip environments, not images:** vLLM 0.25.1 and this source for the plugin, vLLM 0.32.0 with
  FlashInfer 0.7.0.post1 for stock, which compiles FlashInfer at first use. That compile on 16 cores ran out of the
  container's memory, so stock's first boot and one warming request of each size ran on four cores. Nothing measured
  did.
- **The native pack is not strict.** `emmy eval golden` fails on this recipe at `ea17c0964`: 57 twins deploy from the
  golden rows and `prologue16-native` does not (`k_linear_mean_reduce_89f977` has no measured row for its `030_cut`
  fork). The pack was exported without `--strict-evidence`, so a prior decides that one fork; every other program's rows
  are the golden's. #1302 landed that fork's row after this session.
- **The plugin has no pack here,** so its ready time is a lowering of every program and does not compare.
- Native's `/health` means loaded: at its 5.4 s answer the device already held the full 30.7 GiB, and a request sent at
  once got its first token in 0.76 s.

Where the time goes, from `nsys --cuda-graph-trace=node` on native (7,680 and 197,632 tokens, under 4% slower than the
unprofiled cells):

- **Decode.** The drafted step takes 23.5 ms on native, 25 ms on the plugin and 18.1 ms on stock (TPOT times tokens a
  step), and the one-row step 16.6 ms on native. Both Emmy lanes took 32-33 ms on 2026-10-25, before #1282.
- **The 64-row chunk.** 54.1 ms at up to 4,096 positions and 60.3 ms up to 8,192; its kernels sum to 54.6 and 60.8 ms.
  The table is that chunk against the one-row step, by what each launch reads. Weight bytes are the checkpoint's. A
  launch that reads weights at the one-row rate, about 1.43 TB/s, is bound by that read; one that grows with rows is
  bound by rows.

  | launches | weights | one row | 64 rows | bound by |
  |---|---|---|---|---|
  | rotary cache store (one CTA) | | 0.06 ms | 11.9 ms | rows |
  | attention, up to 4,096 positions | | 0.74 ms | 6.3 ms | rows x positions |
  | Gated DeltaNet in projections and input norm | 4.03 GB | 3.14 ms | 12.7 ms | rows |
  | Gated DeltaNet state and convolution pieces | | 1.35 ms | 4.5 ms | rows |
  | `k_slice_unsqueeze_reduce_f670fb` | | | 2.9 ms | neither: also 2.9 ms at 16 rows |
  | Gated DeltaNet out projection | 1.51 GB | 1.03 ms | 1.7 ms | weight read, then rows |
  | MLP gate and up, Gated DeltaNet layers | 4.81 GB | 3.36 ms | 3.6 ms | weight read |
  | MLP down, Gated DeltaNet layers | 2.41 GB | 1.68 ms | 2.7 ms | weight read, then rows |
  | attention layers: q | 1.01 GB | 1.10 ms | 2.1 ms | rows |
  | attention layers: MLP gate and up | 1.60 GB | 1.07 ms | 1.2 ms | weight read |
  | attention layers: MLP down | 0.80 GB | 0.55 ms | 1.0 ms | weight read, then rows |
  | attention layers: k, v, o | 0.67 GB | 0.98 ms | 1.2 ms | weight read |
  | the rest | | 1.8 ms | 2.8 ms | |

  One read of all 16.8 GB of the trunk's weights is 11.8 ms, whatever the width.
- **The floor at a wider chunk.** Without rotary and attention the trunk takes 36.4 ms at 64 rows and 15.6 ms at one row
  without the head: 15.3 ms a chunk plus 0.33 ms a row. Two points on different programs, so an estimate, and no program
  above 64 rows was measured; the verify step sits above that line (19.7 against 16.3 ms, its head left out).

  | rows | trunk | rotary at today's rate | attention, up to 4,096 positions | chunk | per token |
  |---|---|---|---|---|---|
  | 64 | 36.4 ms | 11.9 ms | 6.3 ms | 54.6 ms | 0.85 ms |
  | 256 | 100 ms | 48 ms | 25 ms | 173 ms | 0.68 ms |
  | 512 | 184 ms | 95 ms | 50 ms | 330 ms | 0.64 ms |

  The attention term does not shrink with width: 1.09 s of a 7,680-token prompt at any width. A 7,680-token prompt at
  512 rows with no rotary cost at all is 15 x 184 ms + 1.09 s = 3.85 s; stock's whole first token there is 0.63 s. Wider
  chunks stop paying near 256 rows, because the row term is what is left.
- **A prompt's tail.** A chunk takes only prompt tokens before the last, and the tail runs in 16-row chunks, then one
  row at a time. A 256-token prompt is 3 chunks of 64, 3 of 16 and 16 single steps; 257 is 4 chunks and one step; 268 is
  4 chunks and 12 steps. The 16-row chunk costs 121.6 ms at 7,600 positions against 54-60 ms for 64 rows, and 1.91 s at
  197,600: 45% of it is attention at 7,600 and 96% at 197,600.
- **Long prompts.** Native's first token at the two long prompts is 38-42% slower than on 2026-10-25, and the plugin's
  and stock's did not move. At 197,632 tokens the 64-row chunks take 692 s: attention 544 s, all else 149 s (48 ms a
  chunk, flat). Attention grows linearly from 6.3 ms a chunk to 344 ms, 6.7 ms a layer at 65,536 positions against the
  4.6 ms measured before. The kernel's rows are the golden's, so the non-strict fork is not the cause. The commit that
  moved it was not bisected.
- **Accuracy.** The Emmy lanes score within one standard error of each other; stock 0.12-0.14 lower, as before.

## The recipe's native and plugin lanes on one RTX 5090 (2026-10-31)

Both RTX 5090 lanes of `recipe.yaml`, deployed and benchmarked by `emmy deploy local` and `emmy bench --local` on one
rented RTX 5090 (instance type `rtx59-3-24-750-xam.1`: 3 cores, 23.5 GiB of RAM, swap on), one session. The plugin lane
runs `cloudriftai/vllm-emmy:0.25.1-3454735b`, built on the card with `make vllm-emmy-image VLLM_VERSION=v0.25.1`. The
native lane runs `cloudriftai/emmy-native-qwen3.8-27b-nvfp4:0.3.23-6732a701`, baked on the card by `make native-image`:
its pack was prepared at `b60e2b02` (main at `35100514c` and this branch's drafter rows) inside the plugin image, from
the recipe golden alone under strict evidence, and the image tag's commit changes only the script that splits the pack
into image layers. Neither image is published. The
benchmark is the recipe's: 10 random-token prompts of 256 tokens, 256 output tokens with `ignore_eos`, temperature 0,
one request at a time, vLLM 0.25.1's client. Random tokens draw repetitive answers, which the drafter predicts well,
so these TPOTs sit below the ones of the tables further down; compare the two columns, not this table with those.

| | native | plugin |
|---|---|---|
| TPOT, median / P99 | 8.66 / 10.6 ms | 10.30 / 12.6 ms |
| TTFT, median / P99 | 721 / 723 ms | 198 / 361 ms |
| output tokens a second | 86.9 | 87.7 |
| tokens in / out | 2,482 / 2,560 | 2,482 / 2,560 |
| drafts accepted | the client reports none for the native server | 78%, 2.56 tokens a step |
| device memory at rest | 30.7 GiB | 30.7 GiB |
| host memory of the server at rest | 3.5 GiB | 15.8 GiB after a cold boot, 18.4 GiB after a warm one |
| ready after | 24-27 s, nothing compiled | 3,030 s cold, 437 s from its cubin cache |

Notes:

- **No compile at the native boot.** Both native boots load the pack and run the smoke request (24 s and 27 s, 0.6-0.8
  s of it the request); the server log names no compile.
- **The plugin's caches.** The bench's plugin boot took 2,927 s, a cold boot: the lane kept its cubins and packs under
  `/mnt/models`, which only `emmy deploy local` mounts, and `emmy bench` mounts `/hf_models`. The recipe now keeps them
  under `/hf_models`.
- **Prefill.** Both lanes run 64-row prompt chunks; native's first token comes 0.5 s after the plugin's at 256 tokens.
- **Memory.** Device memory is the card's total after the boot (`nvidia-smi`), host memory the server container's
  (`docker stats`), both at rest; peaks under load were not taken.

## Native and NInfer on one RTX 5090 (2026-10-29)

[NInfer](https://github.com/Neroued/ninfer) at `1882c6ad` is a C++/CUDA engine for this card that ships its own artifact
of Qwen3.8-27B. Built from its Dockerfile (CUDA 13.1.2, `sm_120a`) and run on the rented x16 RTX 5090 that measured the
Qwen3.6 recipe's table (driver 580, CUDA 13.0: the container runs with `NVIDIA_DISABLE_REQUIRE=1` and the host's
`libcuda` ahead of the image's forward-compatibility one, which a GeForce card refuses). Every cell below comes from one
session on that card. Both columns serve one request at a time with the FP8 KV cache, the 262,144-token window, the MTP
drafter at two tokens with the checkpoint's own head, images on, thinking off and no prefix cache, and get the same
prompts from the same client as the table below, with three differences NInfer forces: it serves only
`/v1/chat/completions`, so every prompt is one user message (268, 7,692 and 261,644 tokens with the template; native
reads the same message rendered by the checkpoint template from `/v1/completions`, and both servers report the same
prompt length); it has no `ignore_eos`, so neither side gets it and an answer ends at the model's end of turn or at 256
tokens; and GSM8K goes through the chat endpoint, the few-shot examples as turns. Two places in vLLM 0.25.1's client
were patched for NInfer's stream, which opens with an empty role chunk and sends `: keep-alive` comments every 5 s of a
long prefill: the first token is the first chunk with content, and comment lines are dropped (kept, the first one hid
every later chunk). Native's completions stream has neither.

| | native | NInfer |
|---|---|---|
| weights | NVIDIA's NVFP4 checkpoint (NVFP4 and FP8 linears) | unsloth's NVFP4 checkpoint as NInfer's artifact: NVFP4 MLPs in layers 0-55; row-scaled FP8 attention, Gated DeltaNet projections, last eight MLPs, head and embedding |
| TPOT / TTFT, 256-token prompt | 13.2 ms / 0.45 s | 6.3 ms / 0.050 s |
| TPOT / TTFT, 7,680-token prompt | 13.6 ms / 6.98 s | 6.5 ms / 0.62 s |
| TPOT / TTFT, 261,632-token prompt | 35.7 ms / 825 s (20 tokens) | 8.9 ms / 62.5 s (52 tokens) |
| drafts accepted at 256 / 7,680 / 261,632 | 73 / 71 / 75% | 73 / 73 / 68% |
| NInfer's own setting (three drafts, its smaller draft head): TPOT / TTFT and acceptance at 256 / 7,680 | | 5.3 ms / 0.050 s, 5.4 ms / 0.63 s; 62 / 64% of three |
| device memory at rest / peak | 30.1 / 30.6 GiB | 30.8 / 30.8 GiB (19.7 GiB of weights) |
| KV cache | 262,144 tokens, the window | 278,788 tokens, what the card has left |
| host memory of the server at rest, peak | 3.4 GiB (3.1 GiB pinned: the token-embedding table and the tower), 4.9 GiB | 2.4 GiB, 2.4 GiB |
| GSM8K strict / flexible match, chat | 0.905 / 0.905 | 0.925 / 0.935 |
| ready after | 20 s from its pack | 9-10 s |

Acceptance is the client's rule of the table below for both columns; NInfer's own counters agree within 1 point at 256
and 7,680 tokens. GSM8K's standard error is 0.019. The native column is the pack of the table below (main at
`b5dddf26c`, before #1282), exported again on this card from the recipe golden under strict evidence; its prompts run in
64-row chunks, NInfer's in 1,024-token chunks.

Where NInfer's time goes that native's does not:

- **Decode.** Both sides accept the same share of drafts, 2.45 tokens a step, so the gap is the three-row verify step:
  15.5 ms on NInfer against 32.3 ms on native (TPOT times tokens a step). #1282's profile put two thirds of native's
  step in the width-3 Gated DeltaNet program and took the step to 24.8 ms with re-recorded rows and a fixed-width
  prologue; this pack predates them. NInfer's own setting drafts three tokens from its smaller head and gains another
  17%.
- **Decode at the longest prompt.** Native's step grows to 89 ms at 261,644 positions, NInfer's to 21 ms. Native's
  attention reads each key-value head's cache once per query head.
- **Prefill.** NInfer prefills 12,300 tokens a second at 7,692 tokens and 4,200 at 261,644; native 1,100 and 317.
- **Accuracy.** The two score within one standard error of each other on GSM8K through the chat template.

## Native, plugin and stock vLLM on one RTX 5090 (2026-10-25)

One rented x16 RTX 5090 (14 cores, 23.5 GiB of RAM, swap on), one session: every cell below comes from the same card on
the same day. All three columns serve the recipe's checkpoint at its pinned revision one request at a time, with the
FP8 KV cache, the MTP drafter proposing two tokens a step, images on (one per prompt, at the recipe's bounds) and no
prefix cache. Every column gets the same text: slices of this repository's Markdown files, cut to exact token counts
with the checkpoint's tokenizer and sent raw to `/v1/completions` (no chat template, so no thinking) by `vllm bench
serve` (vLLM 0.25.1's client, `--dataset-name custom --skip-chat-template`), 256 output tokens with `ignore_eos` at
temperature 0. Medians over 10 requests at 256 prompt tokens and 4 at 7,680, one request at each long prompt. The
random-token prompts `emmy serve --bench` sends were measured first and dropped: their answers fall into repetition,
which the drafters accept 80-98% of the time, or at 261,632 tokens into more random tokens, of which both Emmy lanes
accept 36%.

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 13.7 ms / 1.34 s | 13.5 ms / 0.23 s | 7.7 ms / 0.12 s |
| TPOT / TTFT, 7,680-token prompt | 14.3 ms / 7.78 s | 14.5 ms / 5.31 s | 8.2 ms / 0.64 s |
| TPOT / TTFT, 197,632-token prompt (stock's longest) | 32.3 ms / 505 s | 17.3 ms / 184 s | 8.7 ms / 67.9 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 41.7 ms / 815 s | 16.0 ms / 266 s | does not fit |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 68 / 66 / 76 / 66% | 73 / 62 / 62 / 77% | 65 / 60 / 77% / - |
| TPOT with the drafter off, 256 / 7,680 | 18.3 / 18.7 ms | | |
| device memory at rest / peak | 30.1 / 30.6 GiB | 29.9 / 30.7 GiB | 30.4 / 30.9 GiB |
| KV cache | 262,144 tokens, the window | 305,585 tokens at 0.93 | 199,891 tokens at 0.93 |
| host memory of the server at rest, peak | 3.4 GiB, 4.8 GiB | 10.8 GiB (1.3 swapped), engine peak 9.6 GiB | 5.1 GiB (0.7 swapped), engine peak 10.9 GiB at load |
| GSM8K strict / flexible match | 0.710 / 0.770 | 0.735 / 0.760 | 0.580 / 0.635 |
| ready after | 20 s from its pack | 236 s from its pack | 181 s |

Acceptance is what the client sees: a step's tokens arrive within a millisecond of each other, so a longer gap starts a
step. vLLM's own counters agree with it within 2.5 points on both vLLM columns. GSM8K is
`scripts/run_lmeval_gate.py --num-concurrent 1`: 200 questions, 5-shot, seed 0, raw completions, the protocol of
`recipes/Qwen3.8-27B-NVFP4`; its standard error is 0.03. Host memory is the server processes' resident set and swapped
pages; native's 3.4 GiB hold 3.1 GiB of pinned pages (the token-embedding table and the vision tower's blocks).

The columns:

- **native**: `emmy-server` from the main branch after #1275 (`b5dddf26c`) over milestone 12's generation-format-13
  pack, exported from `golden/rtx5090_sm120.json` alone under strict evidence with images, the drafter, the FP8 cache
  and the 262,144-token window. Emmy compiles every program, attention included; a prompt runs in 64-row chunks. The
  pack predates #1282, whose golden rows and fixed-width prologue programs take native drafted decode from 12.1 to
  9.2 ms a token on its probes; the plugin and stock columns do not depend on it.
- **plugin**: `emmy serve --runner generate` in `vllm/vllm-openai:v0.25.1` with the same source installed, the recipe's
  engine block (fast math, strict evidence from the golden, `EMMY_GEN_EMBED_GPU=0`, `EMMY_GEN_VISION_GPU=1`,
  `--max-num-batched-tokens 64`, full decode graphs at sizes 3, 6 and 15) plus `--max-num-seqs 1 --max-model-len
  262144 --gpu-memory-utilization 0.93`. Emmy runs every decoder layer, the MTP head, the coded output head and the
  tower; vLLM schedules, drafts and runs FlashInfer attention over its cache.
- **stock**: `vllm/vllm-openai:v0.32.0`, the newest release (`sha256:befccf0e…`, the image `recipes/Qwen3.8-27B-NVFP4`
  pins), with `--max-num-seqs 1 --max-model-len 198400 --kv-cache-dtype fp8 --no-enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --gpu-memory-utilization 0.93` and the recipe's
  `--limit-mm-per-prompt` and `--mm-processor-kwargs`; the rest is vLLM's default: `FlashInferCutlassNvFp4LinearKernel`
  for the NVFP4 linears, `FlashInferFP8ScaledMMLinearKernel` for the FP8 ones, FlashInfer's Gated DeltaNet prefill
  kernel, FlashInfer attention (XQA decode over the FP8 cache), its own `Qwen3_5MTP` drafter at two tokens,
  `FULL_AND_PIECEWISE` CUDA graphs under torch.compile, and chunked prefill at 2,048 tokens a step.

What could not match:

- **The window.** Stock vLLM needs 8.98 GiB of KV cache for one 262,144-token request and has 6.99 GiB at 0.93
  utilization: it counts 20.64 GiB of weights, the 2.37 GiB token-embedding table among them, and a 2.6-2.7 GiB peak in
  its profiling pass. Text only at 0.97 it has 6.96 GiB (7.07 GiB with 512-token chunks), so it refuses the window
  however it is set, and serves at most 198,400 tokens. The 197,632-token row compares all three; the 261,632-token row
  is the two Emmy lanes alone. Both Emmy lanes keep the table in host memory.
- **The kernels.** Each side runs its own. The plugin and native run the same trunk programs, the recipe golden's
  twins; native runs its own attention, rotary and selection programs too.
- **The prefill chunk.** 64 tokens on both Emmy lanes, the width their programs are recorded at; 2,048 on stock.

Where the time goes:

- **Decode.** With two drafts a step is three rows. That step takes 32-33 ms on both Emmy lanes at the short prompts
  (TPOT times tokens a step) against stock's 18 ms, which is what native's one-row step takes with the drafter off
  (18.3 ms). It is the same programs on both Emmy lanes. #1282's profile of native's step put two thirds of it in the
  width-3 Gated DeltaNet program, whose pieces held the prior's untuned picks, and 1.1 ms in the drafter's any-width
  prologue; its re-recorded rows and fixed-width prologue take the step to 24.8 ms and drafted TPOT on its probes to
  9.2 ms.
- **Decode at long windows.** Native's step grows from 33 ms at 7,680 positions to 81 ms at 197,632 and 97 ms at
  261,632; the plugin's, with vLLM's attention, from 33 to 39-41 ms and stock's from 18 to 22 ms. Native's attention
  reads each key-value head's cache once per query head, six times at this model's grouping.
- **Prefill.** Native takes 1.01 s per 1,000 prompt tokens at 7,680, the plugin 0.69 s and stock 0.084 s. Both Emmy
  lanes prefill in 64-row chunks, about 55 ms a chunk for native and 44 ms for the plugin here; milestone 11 found
  native's extra cost in the rotary program's cache store, which runs on one CTA, and in attention at 64 rows, which
  writes its scores and probabilities to memory and grows with the window (milestone 17 items). Stock prefills 2,048
  tokens a step. At 256 tokens native's first token comes after 1.34 s with the drafter on or off: with carried Gated
  DeltaNet state a chunk takes only prompt tokens before the last, so a 256-token prompt runs three chunks and its last
  64 tokens one at a time through the decode program. A 257-token prompt, whose tokens before the last fill four
  chunks, gets its first token after 246 ms (the plugin 230 ms at 256).
- **Accuracy.** The two Emmy lanes score within one standard error of each other on GSM8K; stock scores 0.13-0.16 lower
  on the same prompts and client.
