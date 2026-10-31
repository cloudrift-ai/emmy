# Qwen3.8-27B-NVFP4 (NVIDIA) discovery

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
