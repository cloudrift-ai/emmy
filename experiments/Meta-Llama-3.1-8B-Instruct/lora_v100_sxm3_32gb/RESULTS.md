# Llama 3.1 8B Instruct with a selectable LoRA on V100

## Emmy LoRA serving qualification (2026-10-05 to 2026-10-06)

**Behavior passed; performance did not.** Emmy now runs the selected rank-8 adapter in its pre- and post-attention
programs while vLLM schedules mixed base and adapter requests and runs paged attention. All 112 requests in the
one-repeat Emmy matrix completed, and fixed and multi-chunk outputs matched stock vLLM. The compiled path reached
only 2.71 output tokens/s at concurrency 16 for the adapter, versus 67.38 for stock vLLM in matched eager mode.
The release gain bar failed. This is an experimental path; no serving image was published.

The same rented Tesla V100-SXM3-32GB (SM70), driver 580.178.04, CUDA 12.9, and pinned stock image served every
row. The base checkpoint was
`NousResearch/Meta-Llama-3.1-8B-Instruct@d10aef7999a2b5ba950ab3974312feeedbfe0b77`; the adapter was
`t83714/llama-3.1-8b-instruct-limo-lora-adapter@cfccf812259ae6131253b623aaa386577c7fc791`; the tokenizer
was `hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4@db1f81ad4b8c7e39777509fac66c652eb0a52f91`.
Both servers used FP16, a 4,096-token context and batch-token limit, 16 maximum sequences, eager execution,
`FLASH_ATTN_V100`, disabled prefix caching, and `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`. Emmy used a
512-row static prefill twin, a 16-row decode twin, and strict golden evidence. The stock image was pinned to
`cloudriftai/1cat-vllm-sm70@sha256:6f34e0b247a78ca65f88f305b1f1cc52c9020ecb83a5ca21df0599676dc443d3`.

The client sent 8, 16, or 32 requests at concurrency 1, 8, or 16, respectively. It requested 512 random input
tokens and 32 forced greedy output tokens, ignored EOS, and used seeds 0–2 for three stock repeats. Tokenization
produced 511 actual input tokens per request. Emmy has one repeat per row: its large regression made the
three-repeat release gate unnecessary for this failed qualification. The stock eager means include a cold adapter
repeat at each concurrency; the warm LoRA repeats were faster. Every request in both matrices succeeded.

| Request | Concurrency | Stock eager output tok/s, 3 repeats | Emmy output tok/s, 1 repeat | Stock mean TTFT | Emmy mean TTFT | Stock mean TPOT | Emmy mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 1 | 26.21 (25.93–26.38) | 1.80 | 169.6 ms | 11.04 s | 33.89 ms | 217.26 ms |
| `limo` | 1 | 11.82 (6.13–14.77) | 1.80 | 1276.9 ms | 11.05 s | 61.87 ms | 217.15 ms |
| Base | 8 | 102.05 (98.39–104.42) | 2.59 | 1310.5 ms | 89.16 s | 38.61 ms | 306.37 ms |
| `limo` | 8 | 49.89 (19.45–65.64) | 2.60 | 3664.2 ms | 88.58 s | 107.83 ms | 318.26 ms |
| Base | 16 | 106.31 (100.59–109.59) | 2.71 | 2983.1 ms | 162.96 s | 58.89 ms | 841.07 ms |
| `limo` | 16 | 67.38 (49.07–76.95) | 2.71 | 4251.2 ms | 163.25 s | 118.17 ms | 819.01 ms |

At concurrency 16, stock eager delivered 39× the base throughput and 25× the LoRA throughput of Emmy. Its
99th-percentile first-token latency was 3.67 s for base and 6.29 s for LoRA, versus 179.11 and 178.83 s for Emmy.
Its 99th-percentile time per output token was 121 and 225 ms, versus 3,311 and 3,383 ms for Emmy. A separate
three-repeat stock run with its default graph mode reached 112.31 base and 81.07 LoRA output tokens/s at
concurrency 16; the eager rows above are the matched comparison. The stock LoRA means include one slow repeat
at each concurrency, so the range matters more than a small change in their means.

Ten fixed base and adapter API responses matched stock generated tokens exactly; the adapter changed four of
five paired answers. The maximum absolute generated-token log-probability difference was 0.01453. Two
simultaneous base and adapter requests with 1,485 input tokens each crossed multiple static prefill chunks and
matched stock generated tokens exactly. Their maximum generated-token log-probability differences were 0.00167
and 0.03274, respectively. These API checks do not compare every vocabulary logit or assess adapter task quality.

The serving golden has eight pre/post programs, 114 kernels, 26 routing decisions, and 101 rows, of which 93
have measured latency. The strict serving audit and fresh-lowering check passed for the eight twins. A saved pack
loaded 256 layer plans without tracing or compiling them again at boot. The Emmy model load still took 277 s,
versus 13 s for the matched stock eager server. A GPU memory snapshot showed 26,767 MiB used by Emmy and
26,121 MiB by stock. vLLM reported 10.59 GiB available for Emmy's KV cache, versus 11.45 GiB for stock.

The measured bottleneck is prefill. In one near-matched base-request trace, Emmy spent 13.399 s of GPU time on
583 input and 8 output tokens; stock eager spent 274.871 ms on 585 input and 8 output tokens. Two Emmy
post-attention projection kernels accounted for 8.876 s of its trace. The startup roofline probe measured
298.876 ms for one post-attention 512-row program, about 148 times its estimated floor. Static chunking
improved Emmy's one-repeat output throughput from 1.64 to 1.80 tokens/s at concurrency 1 and from about
2.15 to 2.60 at concurrency 8. The larger projection schedule still dominates whole-request time. Isolated
golden kernel timings did not predict the serving kernel time, so they are not evidence of a serving gain.

The Git LFS archive `results_v100x1.tar.gz` retains this run under `2026-10-05_emmy_lora/`: client JSON and logs,
fixed and long parity responses, GPU traces, server logs, the pack manifest, and golden audit output. The earlier
`2026-10-04_00-46-11/` stock result remains in the same archive. The release image and three-repeat Emmy
qualification remain open until the large prefill kernels improve and the identical request matrix is repeated.

## Earlier stock vLLM result (2026-10-04)

The pinned vLLM image serves the base model and the `limo` LoRA on one 32 GB V100. With a 16-sequence limit and
8,192 batched tokens, all 432 measured requests in the six-row matrix succeeded. At concurrency 16, the adapter
averaged 211.90 output tokens/s over three repeats; the base reached 342.92 output tokens/s in one repeat. The
adapter changed four of five fixed answers relative to the base. Ten simultaneous base and adapter requests matched
their sequential reference answers exactly. A separate adapter request with 58,035 input tokens and one output token
succeeded in 96.52 s. This checks a long request within the configured 65,536-token context, not the exact limit.
These were vLLM serving results. At that time, an Emmy-compiled serving image and a complete model golden remained
unqualified.

| Request | Concurrency | Success | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1 | 48/48 | 42.81 (42.80–42.81) | 123.3 ms | 22.57 ms |
| `limo` | 1 | 48/48 | 20.06 (20.06–20.07) | 190.0 ms | 48.73 ms |
| Base | 8 | 40/40 | 234.94 | 1263.2 ms | 29.22 ms |
| `limo` | 8 | 40/40 | 118.81 | 2738.9 ms | 56.85 ms |
| Base | 16 | 64/64 | 342.92 | 2639.7 ms | 36.46 ms |
| `limo` | 16 | 192/192 | 211.90 (207.60–214.06) | 3474.9 ms | 62.18 ms |

The concurrency-16 LoRA row sent 64 requests of 512 input and 256 forced output tokens in each repeat. Its
throughput was 78% higher than the tuned concurrency-8 LoRA row under twice the offered concurrency; mean TTFT rose
by 0.74 s and mean TPOT by 5.33 ms. At concurrency 16, LoRA delivered 62% of base output throughput. The single
base repeat and the different concurrency levels limit stronger speed claims.

The batch-token increase was measured separately with the same published image, 16-sequence limit, 64 prompts per
repeat, and seeds 0–2. The 4,096-token setting yielded 193.58, 206.02, and 202.77 output tokens/s. The 8,192-token
setting yielded 207.34, 214.20, and 214.19 output tokens/s: a 5.5% mean gain, with every 8,192-token repeat faster
than every 4,096-token repeat. Mean TTFT fell from 4.09 to 3.47 s. The lower-concurrency rows remained close to
their original baseline on the same published image.

The dated raw run `2026-10-04_00-46-11/` is the root of the Git LFS archive. It contains six system-only YAML
records, client and server logs, and `diagnostics/semantics/` for the fixed-response checks. The earlier baseline
and optimization trials are under `diagnostics/previous/2026-10-03_18-22-46/` in that archive.

## Question and setup

Can one 32 GB V100 serve the pinned base checkpoint and a named LoRA adapter through the same vLLM endpoint, and
what does selecting the adapter cost? The base is the public `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror at
`d10aef7999a2b5ba950ab3974312feeedbfe0b77`; its config and four weight shards were previously checked as
byte-identical to Meta's gated Llama 3.1 8B Instruct revision. The test adapter is
`t83714/llama-3.1-8b-instruct-limo-lora-adapter` at `cfccf812259ae6131253b623aaa386577c7fc791`, rank 8.

The original baseline used one Tesla V100-SXM3-32GB (SM70), driver 580.178.04, FP16 weights, 65,536-token configured
context, eight maximum sequences, and the pinned `cloudriftai/1cat-vllm-sm70` image at
`sha256:6f34e0b247a78ca65f88f305b1f1cc52c9020ecb83a5ca21df0599676dc443d3`. The image's Volta attention
backend ran with `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` and prefix caching disabled. The server advertised both
the base model and `limo` in one `/v1/models` response. Both names passed the `2 + 2` chat smoke test.

## Protocol and results

Emmy deployed a fresh server for each row and ran `vllm bench serve` against it. Prompts had 512 random input tokens,
greedy decoding, two warmup requests, and a forced output length. Concurrency 1 used 16 prompts of 128 output tokens
in three repeats; concurrency 8 used 40 prompts of 256 output tokens in one repeat. Every row loaded the adapter,
then selected either the base model or `limo` by request name. The same image and serving settings were used throughout.

| Request | Concurrency | Success | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1 | 48/48 | 42.78 (42.76–42.80) | 122.8 ms | 22.59 ms |
| `limo` | 1 | 48/48 | 20.06 (20.06–20.07) | 189.7 ms | 48.74 ms |
| Base | 8 | 40/40 | 236.38 | 1197.3 ms | 29.27 ms |
| `limo` | 8 | 40/40 | 126.05 | 1682.0 ms | 57.11 ms |

The adapter delivered 47% of base output throughput at concurrency 1 and 53% at concurrency 8. Its mean TPOT was
2.16× and 1.95× the base value, respectively. The three concurrency-1 repeats varied by at most 0.04 output tok/s
for base and 0.01 for the adapter. The concurrency-8 rows have one repeat each, so they do not establish run-to-run
variance. These figures compare request names on a server with the adapter loaded.

A separate run without an adapter used the same pinned image and V100 attention setting. It averaged 237.01 base
output tokens/s at concurrency 8, close to 236.38 on the server with the adapter loaded. It used 32 rather than 40
requests per repeat, so the comparison is directional. Load and warmup took about 137 seconds with the adapter loaded
versus 88 seconds without it; most of that difference was CUDA graph capture (40 versus 2 seconds in the raw timings).

## V100 attention failure and correction

An initial run on the same public image tag, without the paged-prefill setting, succeeded at concurrency 1 but
completed only 1/40 requests in each concurrency-8 row. Its server logs reported `RuntimeError: Shared memory exceeds
96KB: 114176 bytes` from the direct paged prefill kernel. Emmy marked those rows succeeded because the benchmark
client returned exit code zero even though 39 requests failed. The corrected run above disabled that kernel path and
all 176 measured requests succeeded. The archived `diagnostics/` directory retains the two failed client logs, their
server logs, and the initial recipe. Read client success and failure counts as well as Emmy's row status.

## Evidence and limits

Original baseline run time: 2026-10-03 18:22:46 UTC; run ID `20261003T182246Z`. All four system-only experiment
records ended in `succeeded`. Docker Engine was 29.8.1 on Ubuntu 24.04.1; the raw records carry the rest of the
machine inventory. The Git LFS archive `results_v100x1.tar.gz` preserves this earlier run under the preceding
`diagnostics/previous/` path, including:

- `v100x1_rnN-M-L-3.1-8B-I_7043c7f7d3c9.experiment.yaml` and its benchmark and server logs;
- `v100x1_rnlimo_07eef6eccf5f.experiment.yaml` and its benchmark and server logs;
- `v100x1_mc8_np40_rol256_r1_rnN-M-L-3.1-8B-I_c4cd74020277.experiment.yaml` and its logs;
- `v100x1_mc8_np40_rol256_r1_rnlimo_25237df6ee6a.experiment.yaml` and its logs;
- `benchmark.log`, `benchmark_v100_x_1.log`, and `diagnostics/` with the initial failed concurrency-8 evidence.

The original rows were run sequentially on fresh servers. The later mixed-traffic probe checked ten simultaneous
requests, but neither it nor the simple smoke test establishes adapter task quality or tool-call behavior. The later
long-request probe does not establish success at the exact configured context limit. Emmy compiler kernels were not
used by this serving image.

## Follow-up: active LoRA specialization does not preserve adapter behavior

On 2026-10-03, one freshly rented Tesla V100-SXM3-32GB repeated the concurrency-eight LoRA row three times with
seeds 0–2. Each repeat sent 40 requests with 512 random input tokens and 256 forced output tokens. The published
vLLM 1.0.0 image, checkpoint, adapter, FP16 precision, 65,536-token context, and eight-sequence limit were identical;
only `--specialize-active-lora` changed. All 240 requests completed.

| LoRA setting | Output tok/s, seeds 0–2 | Mean TPOT, seeds 0–2 | Adapter effect observed? |
| --- | --- | --- | --- |
| Default | 118.58, 128.04, 129.60 | 57.09, 55.57, 55.31 ms | Yes |
| `--specialize-active-lora` | 187.52, 217.28, 212.97 | 33.91, 32.10, 32.22 ms | No |

The apparent 1.64× average throughput gain is invalid. On five fixed prompts, the default adapter changed four
answers relative to the base model. With the specialization flag, all five adapter answers matched the base answers.
For the one answer that matched even without the flag, the sum of absolute generated-token log-probability differences
was 1.70 normally and 0.009 with specialization. A 16-iteration PyTorch GPU trace recorded 2,064 LoRA shrink
and 2,080 expand kernel launches without the flag, versus 16 of each with the flag. The trace uses a shorter four-
request probe and is diagnostic, not a throughput measurement. These observations indicate that the published image
skips most adapter updates when this flag is set. The serving recipe therefore leaves it disabled.

The archive keeps the two three-repeat client logs and system records, the profiler traces, the five prompts, and
both sets of API responses under `diagnostics/optimization_20261003/`. The printed vLLM row status and smoke answer
did not detect this failure; comparison of base and adapter responses did.

## Other optimization candidates

A newer locally built 1Cat-vLLM image from commit `96f26179bf28aaea645635b8ec6f26c98360e0c2` preserved the fixed
base and adapter answers, but was slower on the same V100 and concurrency-eight probe. It reached 98.84 output
tokens/s with its default LoRA path and 77.84 with `VLLM_LORA_ENABLE_DUAL_STREAM=1`, versus 118.58 for the published
image in the same-host reference repeat. This exploratory image was not published or selected by the recipe.

In a short PyTorch GPU trace of normal adapter serving, LoRA shrink and expand kernels accounted for about 425 ms of
GPU time. This supports the observed adapter overhead, but the trace is not a complete latency attribution. The raw
image build, repeated benchmarks, parity responses, and trace are retained in the archive's optimization diagnostics.
