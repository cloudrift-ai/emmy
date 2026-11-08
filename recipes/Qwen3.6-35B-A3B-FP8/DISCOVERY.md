# Qwen3.6-35B-A3B-FP8 discovery

## The FP8 cache staged into shared memory (2026-11-07)

Native attention's FP8 key-value cache now copies into shared memory before the tensor-core fold reads it (#1366, on
#1352). Recorded on the card and with the protocol of the 27B's section of the same name (fast math off); the pack
exports under strict evidence, every twin compiles strictly from the golden, and `emmy eval golden` deploys every twin
from the golden rows alone. The section below's times are in brackets.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 10.0 µs (18.0) | g64k, 111 µs (413) |
| `attend3` (verify) | g16k, 15.5 µs (29.3) | g64k, 277 µs (815) |
| `attend4` | g16k, 24.6 µs (129) | g64k, 561 µs (4,954) |
| `attend64` (chunk) | g8k, 54.4 µs (137) | g16k, 1,046 µs (3,594) |

g128k, which the compiler does not offer, measured 23% slower than g64k at one row and 4% faster at three.

| | this branch | the section below |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 20.4 ms / 5.58 s | 20.9 ms / 5.58 s |
| first token, 257-token prompt | 5.48 s | 5.62 s |
| TPOT / TTFT, 7,680-token prompt | 20.9 ms / 169 s | 21.2 ms / 167 s |
| drafts accepted at 256 / 7,680 | 76 / 79% | 75 / 85% |
| device memory at rest | 23.0 GiB | 23.0 GiB |
| GSM8K strict / flexible match | 0.460 / 0.475 | |

The short cells hold. GSM8K holds too, against main's 0.455 / 0.460 on 2026-11-06 (standard error 0.035); the section
below did not run it. The window step was not served, as in the section below: a prompt that reaches the long range
prefills for over an hour. Its verify twin's attention now takes 277 µs a layer at 131,072 positions.

## Attention priced per range of windows (2026-11-07)

Native attention now compiles once per range of windows, priced at 4,096 and at 131,072 cached tokens, the long range
from 16,384 (#1352, on #1329's grouped decode attention). Recorded on the card and with the protocol of the 27B's
2026-11-07 section (fast math off); the pack exports under strict evidence and `emmy eval golden` deploys every twin
from the golden rows alone.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 18.0 µs | g64k, 413 µs |
| `attend3` (verify) | g32k, 29.3 µs | g64k, 815 µs |
| `attend4` | g8k, 129 µs | g64k, 4,954 µs |
| `attend64` (chunk) | g8k, 137 µs | g32k, 3,594 µs |

| | this branch | 2026-11-06 (main) |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 20.9 ms / 5.58 s | 20.5 ms / 6.12 s |
| first token, 257-token prompt | 5.62 s | 6.16 s |
| TPOT / TTFT, 7,680-token prompt | 21.2 ms / 167 s | 21.7 ms / 185 s |
| drafts accepted at 256 / 7,680 | 75 / 85% | 80 / 73% |
| device memory at rest | 23.0 GiB | 23.0 GiB |

- **The short cells.** The drafted step takes 51.1 ms at 256 tokens against about 53 ms on 2026-11-06, and 55.9 ms at
  7,680 against about 53.4 ms (+5%); both prompts get their first token sooner. TPOT follows the drafts accepted, which
  the attention's numerics move on these prompts.
- **The window step was not served.** A prompt that reaches the long range takes more than an hour to prefill here
  (7,385 s at 261,632 tokens on 2026-11-01), so the verify step at a long window is read from its twin: its attention
  takes 815 µs a layer at 131,072 positions against 29.3 µs at 4,096. g128k measured 691 µs there, and the compiler
  offers splits up to g64k.

## The gates of milestones 4 and 5 (2026-11-06)

The parity plan's milestone 4 (prefill ends at the prompt's last token) and the numerics half of milestone 5 (the folded
prefill attention), native only, on the card and main of the 27B's 2026-11-06 section: one rented x16 RTX 5090 (a
container with a 46-CPU quota, driver 615.71, CUDA 13.1), main at `f33a86522`. The client, prompts and settings are the
2026-11-01 section's. Reference packs from the tree before 4 (#1311, `504165b13`) and the tree before 5 (#1318,
`57fabd6e2`) ran on the same card, not strict. Milestone 5's speed gates are the 27B's.

On the card, `emmy eval golden --serving-config` deploys all 66 twins from the golden rows alone (fast math off), the
pack exports under strict evidence, and `emmy golden check` says current, 488 kernels.

| | #1311 tree | #1318 tree | main | 2026-11-01 |
|---|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 20.9 ms / 16.3 s | 21.3 ms / 6.66 s | 20.5 ms / 6.12 s | 19.1 ms / 15.7 s |
| first token, 257-token prompt | 6.52 s | 6.65 s | 6.16 s | 5.88 s |
| TPOT / TTFT, 7,680-token prompt | | | 21.7 ms / 185 s | 20.6 ms / 200 s |
| drafts accepted at 256 / 7,680 | 81% at 256 | 80% at 256 | 80 / 73% | 90 / 74% |
| device memory at rest | 24.5 GiB | 24.5 GiB | 23.0 GiB | 24.5 GiB |
| GSM8K strict / flexible match | | | 0.455 / 0.460 | 0.460 / 0.460 |

- **Milestone 4, met.** At 256 tokens, four 64-row chunks, the first token comes in 6.12 s against the 257-token
  control's 6.16 s; the tree before it took 16.3 s, three 64-row chunks, 15 four-row chunks and three single steps. The
  trees before and after #1313 give the same 64 greedy tokens on all four probes.
- **Milestone 5's numerics.** Here only the attention differs between main and the #1318 tree. Their greedy tokens part
  at tokens 34, 2, 62 and 21 of the four probes' 64; each pack repeats its own tokens. Where they part, both
  continuations read well ("red, blue, and yellow" against "red, yellow, and blue"). Native returns no
  log-probabilities, so the margin at each parting point is not measured. GSM8K holds: 0.455 / 0.460 against 0.460 /
  0.460 on 2026-11-01, standard error 0.035.
- **The fold frees 1.5 GiB of device memory,** the old route's scores and probabilities.
- **The first token is sensitive to the host.** Main's 256- and 257-token cells measured 6.77 and 6.91 s while two
  pack exports compiled on the host, against 6.12 and 6.16 s once it was idle. The cause was not separated. Every other
  Qwen3.6 cell here ran on an idle host.
- **TPOT reads 5-7% above 2026-11-01's** at both prompts. The #1311 tree reads the same on this card (20.9 ms at 256
  tokens), so the difference is the card or the host, not the code.

## Native, plugin and stock vLLM again, with concurrency and the step's programs (2026-11-01)

The protocol of the 2026-10-26 table below, on main at `ea17c0964`, one session of 16 hours on one rented x16 RTX 5090
(Gen5 x16; 16 cores, 60 GiB of RAM, no swap, driver 595.84, CUDA 13.1, a container with no Docker), the card of the
27B's 2026-11-01 session.

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 19.1 ms / 15.7 s | 81.7 ms / 3.54 s | 11.2 ms / 0.34 s |
| TPOT / TTFT, 7,680-token prompt | 20.6 ms / 200 s | 95.5 ms / 108 s | 11.9 ms / 2.83 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 36.9 ms / 7,385 s | 94.4 ms / 3,664 s | 15.3 ms / 144 s |
| drafts accepted at 256 / 7,680 / 261,632 | 90 / 74 / 56% | 90 / 62 / 70% | 89 / 80 / 59% |
| TPOT with the drafter off, 256 / 7,680 | 22.1 / 21.8 ms | | |
| first token at 256 / 257 tokens | 15.7 / 5.88 s | | |
| device memory at rest / peak | 24.5 / 25.3 GiB | 29.8 / 31.3 GiB | 30.7 / 30.7 GiB |
| KV cache | 262,144 tokens, the window | 850,989 tokens at 0.93 | 358,723 tokens at 0.93 |
| host memory of the server at rest, peak | 16.3 GiB, 18.8 GiB | 24.0 GiB, 24.2 GiB | 15.0 GiB, 15.3 GiB |
| GSM8K strict / flexible match | 0.460 / 0.460 | 0.335 / 0.335 | 0.350 / 0.350 |
| ready after | 23-55 s from its pack | 2,250-2,320 s, no pack | 62 s, 123 s at its first compile |

Acceptance is what the client sees for every cell. Host memory is the resident set of the server's whole process group,
with the pinned expert pages or offload inside it. GSM8K's standard error is 0.033-0.035.

Stock and the plugin at higher concurrency, with `--max-num-seqs` raised and nothing else changed but stock's window at
32 rows, where 262,144 tokens ran out of device memory at boot (176,000 tokens there):

| | stock, 8 | stock, 32 | plugin, 8 | plugin, 32 |
|---|---|---|---|---|
| 256-token prompts: per-request TPOT | 53.2 ms | 78.6 ms | 447.8 ms | 599.5 ms |
| total output tokens a second | 131.5 | 143.8 | 14.9 | 18.0 |
| first token | 0.84 s | 38.0 s | 12.6 s | 288 s |
| 7,680-token prompts: per-request TPOT | 97.0 ms | 130.5 ms | 531.0 ms | 567.2 ms |
| total output tokens a second | 57.1 | 59.8 | 1.6 | 1.5 |
| first token | 7.83 s | 97.2 s | 1,174 s | 5,151 s |
| device memory peak | 29.8 GiB | 29.9 GiB | 31.2 GiB | 31.3 GiB |

Each cell runs 4 x concurrency requests at 256 tokens and 2 x concurrency at 7,680, each request a different slice of
the same Markdown. First token is the median request's, queueing included. The plugin keeps the recipe's settings, which
name one request; what stops it batching was not investigated. No request failed.

What differs from the 2026-10-26 session:

- **The lanes are pip environments, not images:** vLLM 0.25.1 with torch 2.11 and this source for the plugin, vLLM
  0.32.0 with torch 2.13 for stock. FlashInfer's first compile ran on four cores, and nothing measured did.
- **The native pack is not strict.** A strict export refuses at `ea17c0964`: `k_linear_mean_reduce_12bf99` has no
  measured row for its `030_cut` fork. `emmy eval golden` says 58 twins deploy from the golden rows and 4 do not,
  `prologue1-native`, `prologue3-native`, `prologue4-native` and `prologue64-native`, each at a `k_linear_mean_reduce`
  of that fork. So a prior decides those forks, where the 2026-10-26 pack was strict. #1302 landed that fork's row after
  this session.
- **The plugin's evidence is strict now,** on every boot. It has no pack here, so its ready time is a lowering of every
  program and does not compare; a boot at 8 or 32 rows takes 2,067 and 2,658 s.
- **Transparent hugepages are `madvise`, read-only in this container, and proactive compaction is 20;** the 2026-10-26
  boots had both off. `compact_stall` did not move during any native cell, so compaction did not stall them. Whether
  hugepages change the expert programs' reads was not tested.
- **The plugin's single-request column ran last,** after concurrency, since a 1,800 s limit cut its second boot short.

Stock's TPOT at the longest prompt, 15.3 ms against 9.1 ms, is a drafted step of 33.5 ms against 21 ms. Here the step
holds about level from 31 ms at 256 and 7,680 tokens; on 2026-10-26 the step at the longest prompt was shorter than the
29 ms at 256. One request each, and the cause was not investigated.

Where the time goes, from `nsys --cuda-graph-trace=node` on native (256 and 7,680 tokens, first token within 1% of the
unprofiled cells and TPOT 5% above at 256). The table is the median launch of each program, in milliseconds. The one-row
step comes from a boot with the drafter off and the others from a boot with it on. A page copy is the launch that moves
pages into the slots; expert programs read the slots, and the pages in place for a row the slots do not hold. The rest
is the embedding, the pre and post pieces, the combine, the heads and the greedy pick.

| | one-row step | verify step, 256 | verify step, 7,680 | 4-row chunk, 256 | 4-row chunk, 7,680 | 64-row chunk, up to 4,096 | 64-row chunk, past 4,096 |
|---|---|---|---|---|---|---|---|
| Gated DeltaNet programs | 3.43 | 5.52 | 5.52 | 31.9 | 31.9 | 11.5 | 11.5 |
| attention and rotary | 0.24 | 0.73 | 1.10 | 665.6 | 1,316.5 | 7.4 | 11.0 |
| routers | 0.48 | 1.38 | 1.37 | 1.6 | 1.6 | 0.8 | 0.8 |
| expert programs | 1.20 | 23.58 | 17.84 | 32.7 | 48.0 | 1,405.5 | 1,446.5 |
| paging programs | 7.45 | 7.44 | 7.45 | 7.5 | 7.4 | 7.5 | 7.5 |
| page copies | 8.20 | 13.51 | 12.12 | 11.9 | 13.8 | 5.4 | 4.8 |
| the rest | 1.75 | 3.74 | 3.81 | 1.2 | 1.2 | 4.1 | 4.1 |
| the launch, start to end | 23.38 | 56.95 | 50.40 | 752.4 | 1,421.3 | 1,443.0 | 1,487.8 |

- **Decode.** The drafted step takes 53.5 ms on native at 256 tokens and 50.8 ms at 7,680 (TPOT times tokens a step),
  against 229 and 215 ms on the plugin and 31 ms on stock. The one-row step's 7.45 ms of paging programs agrees with the
  7.8 ms of an earlier split on the dev box and does not move with rows, so a one-row step spends two thirds of its
  time in the pager: 15.65 ms of paging programs and page copies, then 1.2 ms in the expert programs. The verify
  step is 44.5 ms of 57 ms in the expert programs and the pager, against 5.5 ms in the Gated DeltaNet programs.
- **A prompt's tail.** A 256-token prompt is 3 chunks of 64, 15 chunks of 4 and 3 single steps; 257 is 4 chunks of 64
  and one step. 3 x 1.443 s + 15 x 0.752 s is the 15.6 s of the first token, and 4 x 1.443 s is the control's 5.88 s.
  The tail in 4-row chunks costs 11.3 s against about 1.5 s for the 63 single steps it replaced.
- **The 4-row chunk's attention.** That chunk's attention is one kernel on each of the 10 attention layers,
  `k_matmul_softmax_range_reduce_1b2f00`: 66.6 ms a layer with 4,096 positions held and 131.7 ms past that, 88-93% of
  the chunk. Every other width runs attention as cut pieces, 0.37 ms a layer at 64 rows. The golden holds rows of the
  uncut kernel only, with no routing row for it, so nothing measured offers the cut arm; its row reads 7,442 us against
  6.9, 9.7 and 26.6 us for the top-level rows at 1, 3 and 64 rows. A cut kernel in line with those would leave a 4-row
  chunk near 90 ms and the 256-token first token near the control's 5.9 s. That is an estimate from these rows; no such
  program was built.
- **Prefill.** At 7,680 tokens the 119 chunks of 64 take 174 s and the 15 chunks of 4 take 21 s, 196 s of the 200 s. The
  64-row chunk is 97% expert programs, 21.8 ms a layer for gate and up and 13.4 ms for down, reading pages over the
  link. It takes 1.44-1.49 s against 1.25 s on 2026-10-26; the card and the host memory settings both differ, and that
  was not separated. Native takes 26.1 s per 1,000 prompt tokens at 7,680 and 28.2 s at 261,632, the plugin 14.1 and
  14.0 s, stock 0.37 and 0.55 s.
- **The longest prompt.** It was not captured. Native's first token is 1,611 s slower than on 2026-10-26. If the 4-row
  attention keeps doubling with the positions held, its 15 chunks are about 640 s of that, and 4,087 chunks each about
  0.24 s slower, the captured chunk past 4,096 positions against 1.25 s, are the rest. Both terms are extrapolations.
- **Concurrency.** Stock goes from 79 total tokens a second at one row and 256 tokens to 131.5 at 8 rows and 143.8 at
  32, and from 43.7 to 57.1 and 59.8 at 7,680. The plugin goes from 10.5 to 14.9 and 18.0 at 256, and falls from 2.0 to
  1.6 and 1.5 at 7,680. The 27B's stock reaches 756 at 8 rows; what holds this one back was not investigated.
- **Accuracy.** Native scores 0.460, about three standard errors above the plugin and stock, which agree with each
  other.

## The recipe's native and plugin lanes on one RTX 5090 (2026-10-31)

Both RTX 5090 lanes of `recipe.yaml`, deployed by `emmy deploy local` and benchmarked by `emmy bench --local` on one
rented RTX 5090 (instance type `rtx59-7-48-750-xam.1`: 7 cores, 47 GiB of RAM; 64 GiB of swap added after the first
plugin boot ran out of memory, below). The plugin lane runs `cloudriftai/vllm-emmy:0.25.1-3454735b` (`make
vllm-emmy-image VLLM_VERSION=v0.25.1`), the native lane `cloudriftai/emmy-native-qwen3.6-35b-a3b-fp8:0.3.23-6732a701`
(`make native-image`): its pack was prepared at `3454735b` (main at `35100514c` and this branch's drafter rows) inside
the plugin image, from the recipe golden alone under strict evidence. Both images were built on a 3-core RTX 5090 and
copied to this card unchanged; neither is published. The benchmark is the recipe's: 10 random-token prompts of 256
tokens, 256 output tokens with `ignore_eos`, temperature 0, one request at a time, vLLM 0.25.1's client.

| | native | plugin |
|---|---|---|
| TPOT, median / P99 | 20.5 / 22.3 ms | not benchmarked: about 0.3 tokens a second (the engine's count), below |
| TTFT, median / P99 | 15.2 / 16.2 s | |
| output tokens a second | 13.3 | |
| tokens in / out | 2,482 / 2,560 | |
| device memory at rest | 24.6 GiB | 30.6 GiB |
| host memory of the server at rest | 16.4 GiB (the pinned expert pages included) | 29.5 GiB after its first boot, 41.6 GiB after the next |
| ready after | 25-34 s, nothing compiled | 447 s from its pack; the first boot, below |

Notes:

- **No compile at the native boot.** Each native boot loads the pack and answers the smoke request; the server log names
  no compile. The same image on the 3-core card gave the same benchmark (TPOT 20.7 ms, TTFT 15.3 s).
- **The plugin decodes at about 0.3 tokens a second**, on this card and on the 3-core one, so a 256-token answer takes
  about 14 minutes and the deploy's smoke request times out (690 s). The time is the GPU's: during a request it runs at
  100% of its SMs with no memory or PCIe traffic, and a py-spy profile of the engine puts its whole forward in the
  routed experts' combine, waiting on the device for the verify step's expert programs. On 2026-10-26 the same lane's
  verify step took 202 ms (the 2026-10-26 table below). Not bisected; the benchmark was not run at this rate.
- **The plugin's first boot** compiles for longer than the deploy's one-hour health wait (3,642 s, then torn down by
  the run script; the 2026-10-26 one took 68 minutes), and on 47 GiB with no swap its engine was killed for memory 64
  minutes in, while compiling the drafter, with the compile's seven worker processes beside 16.7 GiB of pinned expert
  pages.
  With swap, a boot that reused that run's programs took 971 s.
- **TTFT.** Native's first token takes 15.2 s at 256 random tokens, on this card and on the 3-core one alike, against
  4.98 s for the 256-token text prompt of the 2026-10-29 table below, on the x16 host. Each 64-row chunk reaches most
  of a layer's experts and copies most of the host half in over the link (that table's notes), and a 256-token prompt
  is four chunks; with the core count ruled out, the likely cause is a narrower PCIe link on these rentals, whose link
  width was not recorded.
- **Memory.** Device memory is the card's total after the boot (`nvidia-smi`), host memory the server container's
  (`docker stats`), both at rest; peaks under load were not taken.

## Native and NInfer on one RTX 5090 (2026-10-29)

NInfer at `1882c6ad` against native, one session on the card of the table below, with the protocol of
`recipes/Qwen3.8-27B-NVFP4-NVIDIA/DISCOVERY.md` (2026-10-29): one request at a time, the FP8 KV cache, the 262,144-token
window, the MTP drafter at two tokens with the checkpoint's own head, text only, thinking off, no prefix cache; each
prompt one user message (268, 7,692 and 261,644 tokens with the template), no `ignore_eos`, GSM8K through the chat
endpoint, and the same patched client.

| | native | NInfer |
|---|---|---|
| weights | the FP8 checkpoint, 128 of each layer's 256 experts on the card and the rest copied in from pinned host pages | NInfer's integer artifact of Qwen3.6-35B-A3B: experts Q4 gate/up and Q5/Q6 down, shared experts and mixer projections Q8; every expert on the card |
| TPOT / TTFT, 256-token prompt | 18.3 ms / 4.98 s | 1.75 ms / 0.042 s |
| TPOT / TTFT, 7,680-token prompt | 18.5 ms / 154 s | 1.90 ms / 0.39 s |
| TPOT / TTFT, 261,632-token prompt | 30.5 ms / 5,956 s (177 tokens) | 2.53 ms / 32.0 s |
| drafts accepted at 256 / 7,680 / 261,632 | 71 / 71 / 68% | 71 / 60 / 72% |
| NInfer's own setting (three drafts, its smaller draft head): TPOT / TTFT and acceptance at 256 / 7,680 | | 1.62 ms / 0.041 s, 1.70 ms / 0.39 s; 58 / 52% of three |
| device memory at rest / peak | 24.5 / 25.2 GiB | 30.9 / 30.9 GiB (20.4 GiB of weights) |
| KV cache | 262,144 tokens, the window | 854,589 tokens, what the card has left |
| host memory of the server at rest, peak | 16.2 GiB (15.9 GiB of pinned expert pages), 18.6 GiB | 2.3 GiB, 2.3 GiB |
| GSM8K strict / flexible match, chat | 0.895 / 0.910 | 0.865 / 0.885 |
| ready after | 40 s from its pack | 15-16 s |

NInfer's counters agree with the client's acceptance within 2 points. GSM8K's standard error is 0.022-0.024. The native
column is the pack of the table below.

Where NInfer's time goes that native's does not:

- **The copies.** NInfer's integer experts fit the whole model on the card; native keeps half of each layer's FP8
  experts on the card and copies the others in. Sixty-four prompt rows of eight picks each are expected to reach about
  220 of a layer's 256 experts, so each of native's 64-row chunks copies most of the host half of every layer: native
  prefills 50 tokens a second at 7,692 tokens and 44 at 261,644, NInfer 19,600 and 8,200.
- **Decode.** Both sides accept the same share of drafts at 256 tokens, 2.4 tokens a step, so the gap is the step:
  44 ms on native against 4.3 ms on NInfer (TPOT times tokens a step). Native's one-row step without the drafter took
  19.6-19.8 ms in milestone 12: a three-row verify step routes to more experts than the 8 slots hold, so the drafter
  buys native almost nothing, and NInfer's step, three rows included, is a fifth of native's one-row step. This session
  does not split native's step between the Gated DeltaNet layers and the experts.
- **Decode at the longest prompt.** Native's step grows to 72 ms at 261,644 positions, NInfer's to 6.2 ms.
- **Accuracy.** Native's FP8 checkpoint scores 0.03 above NInfer's integer artifact on GSM8K through the chat
  template, about one standard error.

## Native, plugin and stock vLLM on one RTX 5090 (2026-10-26)

One rented x16 RTX 5090 (14 cores, 23.5 GiB of RAM, 32 GiB of swap), one session from 21:21 on 2026-10-25 to 11:05 UTC
the next day: every cell below comes from the same card. The setup is the one of
`recipes/Qwen3.8-27B-NVFP4-NVIDIA/DISCOVERY.md` (2026-10-25), text only: all three columns serve the recipe's checkpoint
at its pinned revision one request at a time, with the FP8 KV cache, the MTP drafter proposing two tokens a step, no
prefix cache and the 262,144-token window. Every column gets the same text, slices of this repository's Markdown cut to
exact token counts with this checkpoint's tokenizer and sent raw to `/v1/completions` by `vllm bench serve` (vLLM
0.25.1's client), 256 output tokens with `ignore_eos` at temperature 0. Medians over 10 requests at 256 prompt tokens
and 4 at 7,680, one request at 261,632. Transparent hugepages and proactive compaction were off for every boot: on the
card before this one the kernel's compaction thread ran at 100% of a core beside the pinned expert pages.

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 19.9 ms / 5.07 s | 77.3 ms / 3.44 s | 11.0 ms / 0.29 s |
| TPOT / TTFT, 7,680-token prompt | 20.4 ms / 151 s | 78.4 ms / 101 s | 12.2 ms / 2.22 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 33.1 ms / 5,774 s | 84.7 ms / 3,581 s | 9.1 ms / 122 s |
| drafts accepted at 256 / 7,680 / 261,632 | 81 / 73 / 70% | 81 / 77 / 64% | 82 / 70 / 67% |
| device memory at rest / peak | 24.5 / 24.5 GiB, 25.1 at the longest prompt | 29.8 / 30.6 GiB | 29.8 / 31.4 GiB |
| KV cache | 262,144 tokens, the window | 856,858 tokens at 0.93 | 281,854 tokens at 0.93 |
| routed experts off the card | 15.9 GiB of pinned pages | 16.4 GiB of pinned pages | 10.5 GiB of pinned memory |
| host memory of the server at rest, peak | 16.2 GiB with the pages, 17.2 GiB (18.4 at the longest prompt) | 4.5 GiB (2.3 swapped) beside the pages, engine peak 20.5-22.0 GiB at load | 2.8 GiB (1.2 swapped) beside the offload, engine peak 13.6 GiB at load |
| GSM8K strict / flexible match | 0.445 / 0.445 | 0.360 / 0.360 | 0.350 / 0.350 |
| ready after | 30-40 s from its pack | 4,082 s cold, 356-361 s from its pack | 200 s |

Acceptance is what the client sees, as in the 27B table, except for the two vLLM cells at 261,632 tokens: there the
client got stock's 256 tokens in 74 chunks, more than three tokens a step, so those cells are vLLM's own counters.
Elsewhere the counters agree with the client within 2 points. GSM8K is `scripts/run_lmeval_gate.py --num-concurrent
1`: 200 questions, 5-shot, seed 0, raw completions; its standard error is 0.034-0.035. Host memory is the server
processes' resident set and swapped pages; the pinned expert pages are inside native's resident set and outside the
vLLM processes', so the table names them apart.

The columns:

- **native**: `emmy-server` from the main branch after #1275 (`b5dddf26c`) over a generation-format-13 pack, exported
  in 32 minutes from `golden/rtx5090_sm120.json` alone under strict evidence with the drafter, the FP8 cache, the
  262,144-token window, decode width 4, 64-row chunks and 128 of each layer's 256 experts on the card
  (`EMMY_GEN_EXPERTS_GPU=128`); the other 128 are pinned host pages copied in on a miss. Emmy compiles every program,
  attention included.
- **plugin**: `emmy serve --runner generate` in `vllm/vllm-openai:v0.25.1` with the same source installed, the recipe
  golden, `EMMY_GEN_DECODE_BUCKET=4`, `EMMY_GEN_EXPERTS_GPU=128`, `--max-num-batched-tokens 64`, full decode graphs at
  size 1, `--max-num-seqs 1 --max-model-len 262144 --gpu-memory-utilization 0.93`, and no strict evidence (below). Emmy
  runs every decoder layer and the MTP head; vLLM schedules, drafts and runs FlashInfer attention over its cache.
- **stock**: `vllm/vllm-openai:v0.32.0`, the image of the 27B table (`sha256:befccf0e…`), with `--max-num-seqs 1
  --max-model-len 262144 --kv-cache-dtype fp8 --no-enable-prefix-caching --speculative-config
  '{"method":"mtp","num_speculative_tokens":2}' --gpu-memory-utilization 0.93 --language-model-only --cpu-offload-gb
  10.5 --cpu-offload-params experts`; the rest is vLLM's default: `CutlassFp8BlockScaledMMKernel` for the FP8 linears,
  the DeepGEMM FP8 MoE backend, FlashInfer's Gated DeltaNet prefill kernel and its CUDA spec-decode kernel, FlashInfer
  attention, its own `Qwen3_5MoeMTP` drafter at two tokens, CUDA graphs under torch.compile, and chunked prefill at
  2,048 tokens a step.

What could not match:

- **The weights.** Stock vLLM 0.32.0 does not load the checkpoint on its own: without offloading it runs out of device
  memory while it loads the weights (30.55 GiB allocated, 256 MiB more asked for), since the routed experts alone are
  30.0 GiB of the checkpoint's 34.9 GiB. vLLM's UVA offload keeps 10.5 GiB of expert weights in pinned host memory and
  reads them in place over the link; with 23.8 GiB of weights on the card its cache holds 1.08 windows. Both Emmy lanes
  keep half of each layer's experts on the card and copy the others in as pages.
- **The plugin's evidence.** A strict plugin boot refused after 13 minutes of compiles, at layer 0's width-4 Gated
  DeltaNet program: `k_conv1d_linear_mean_reduce_e84cc1` has no measured row for its cut fork. The plugin column ran
  without strict evidence, the recipe golden's rows where it has them and the prior elsewhere, and its first boot
  compiled for 68 minutes; the next two loaded its pack in 6 minutes.
- **The plugin's graphs.** The plugin captures only the one-row step (milestone 12 found its fixed-slot MoE tier covers
  one row with the experts paged), so every verify step runs eager.
- **The prefill chunk.** 64 tokens on both Emmy lanes; 2,048 on stock.

Where the time goes:

- **Decode.** A step (TPOT times tokens a step) takes 52 ms on native at 256 tokens and 50 ms at 7,680, against stock's
  29 ms. Milestone 12 measured native's one-row step without the drafter at 19.6-19.8 ms TPOT, so on this model the
  drafter buys native almost nothing: a verify step's three rows route to more experts than the 8 slots hold. The
  plugin's eager verify step takes 202 ms.
- **Decode at the longest prompt.** Native's step grows to 79 ms at 261,632 positions; the plugin's stays near 194 ms,
  and stock's is 21 ms.
- **Prefill.** Native takes 19.6 s per 1,000 prompt tokens at 7,680 (22.1 s at 261,632), the plugin 13.1 s (13.7 s)
  and stock 0.29 s (0.46 s). Both Emmy lanes prefill in 64-row chunks, 1.25 s a chunk for native and 0.84 s for the
  plugin, against 44-55 ms for the 27B's dense chunks. Sixty-four rows of eight picks each are expected to reach
  about 220 of a layer's 256 experts, so nearly every chunk runs the host half of every layer's experts through the
  pages, and milestone 11 found the paged gate-up piece offered no tensor-core tile (open, compiler). A 256-token and
  a 257-token prompt both get their first token after 5.07 s, so the 27B's cost of running a prompt's last tokens
  one at a time is not what dominates here.
- **Accuracy.** Native scores 0.445 on GSM8K, about 2.5 standard errors above the plugin (0.360) and stock (0.350),
  which agree with each other.

## Assessment on 2026-10-05

Maintained, heat 35. The Qwen3.6-35B-A3B line's strongest direct quant signal (NVFP4 node 6.05M downloads at its
check); mid-size MoE with useful hardware coverage across RTX PRO 6000, A100, and H200 single cards at 262K context.
