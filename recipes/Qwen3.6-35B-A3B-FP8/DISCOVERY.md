# Qwen3.6-35B-A3B-FP8 discovery

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
