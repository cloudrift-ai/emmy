# Qwen3.8-27B NVFP4 serving on one RTX 5090

Stock vLLM serving of `Inferact/Qwen3.8-27B-NVFP4` on one RTX 5090, measured against the 4x V100 SXM2 16GB
production lane of `recipes/Qwen3.8-27B-FP8`. The question: how much context and concurrency does the card's
32 GB leave after a 23 GiB checkpoint, and is prefill under concurrent requests faster than the four V100s, which
is where the production lane is slow. The recipe this experiment qualifies is `recipes/Qwen3.8-27B-NVFP4`.

## NVIDIA GeForce RTX 5090 x1

Run `20261110T181844Z`, started 2026-11-10T18:18:44Z at repository revision `9b5ebe88` on a rented CloudRift node
(`riftvm`, Ubuntu 24.04.1, kernel 6.8.0-134-generic, driver 580.159.03, Docker 29.6.1), one RTX 5090 (sm_120,
32,607 MiB). Image `vllm/vllm-openai@sha256:befccf0e…` (vLLM v0.32.0). Archive `results_rtx5090x1.tar.gz`, root
member `2026-11-10_18-18-44/`. It repeats the 2026-10-21 grid with the full-context latency lane added; every row
that ran in both lands within 3% of the earlier run.

### Protocol

Every row serves revision `6128240e` text-only through `vllm bench serve` with random prompts, temperature 0,
ignored EOS, seed 0, two warm-up requests, one repeat unless stated, client concurrency equal to the row's
value. The primary lane is the recipe's configuration: FP8 key/value cache, 0.97 memory utilization, 114,688
context, cap 16, 4,096-token prefill chunk. Two comparison lanes change one thing each on the three rows that show
prefill under concurrency (1K, 4K and 16K prompts at 16 concurrent): an 8,192-token prefill chunk (context
98,304), and the BF16 key/value cache (context 65,536). The full-context latency lane sends three non-streaming
requests, one at a time and with no warm-up, each of 114,176 prompt tokens and 512 generated tokens, under a
290-second request timeout. Every row redeploys the engine from scratch.

### Fit: what the pool holds

| Item | Value |
| --- | --- |
| Weights resident | 23.21 GiB (embed_tokens, lm_head and the Gated DeltaNet input projections stay BF16, 14.7 GiB) |
| Key/value pool, FP8 cache | 3.83 GiB = 117,666 tokens; 1.03x concurrency at 114,688 |
| Key/value pool, BF16 cache | 4.23 GiB = 66,289 tokens; 1.01x at 65,536 (FlashAttention 2 needs less workspace, each token costs twice as much) |
| Per token, 16 full-attention layers | 32 KiB FP8, 64 KiB BF16 (4 KV heads x 256 x K and V) |
| Per request, 48 Gated DeltaNet layers | 147 MiB of float32 recurrent state, whatever the prompt length (`mamba_ssm_dtype: float32` in the checkpoint config) |
| Requests resident at once, FP8 | about 18 at 1K+1K tokens, 13 at 4K+512, 6 at 16K+512, 3 at 32K+512 |
| Native window 262,144 | needs 8.2 GiB in FP8: does not fit on this card with this checkpoint |
| Load to health | weights 19.2 s, torch.compile 33.4 s, CUDA graphs 11 s, warm-up 34 s; 145 s in all |

The recurrent state, not the attention cache, is what caps concurrency: at 1,000-token prompts it is 70% of a
request's pool cost. The production lane's four V100s hold a 288,281-token pool for the same model, so every row
at 16 concurrent runs with all 16 requests resident there, while here anything longer than 4K prompts queues.

### Full-context latency

Three requests of 114,176 prompt tokens and 512 generated tokens, which together fill the 114,688 context, one at
a time: 35.33 s, 35.28 s and 35.36 s. The first is no slower than the others. The recipe carries the slowest,
rounded up: `full_context_latency_s: 36`, against the 290 s a deployed request has before Relay abandons it. No
larger context was rejected by the 290-second rule: 114,688 is the largest the pool serves, and the next sizes
tried at onboarding (131,072 and the native 262,144) were refused for memory before any request ran.

Time is not what limits the context on this card. Prefill time over the single-request ladder below fits a
linear term plus a quadratic one (8.35e-5 s per token and 1.315e-9 s per token squared) within 0.1 s at all four
points, and with 8.4 s for the 512 generated tokens it predicts this lane at 35.1 s. Extrapolated, the native
262,144 window would take about 121 s and the 290 s rule would be reached near 430,000 tokens. Both figures are
extrapolations: nothing above 114,688 can run here.

### Production comparison: 1,000-token prompts, 1,000-token outputs

Client concurrency equals the server cap. V100 numbers are the production lane's own report (one run per point).

| Cap | Output tok/s 5090 | V100 x4 | Median TTFT 5090 | V100 x4 | P99 TTFT 5090 | V100 x4 | Median TPOT 5090 | V100 x4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 209.5 | 40.8 | 342 ms | 1,024 ms | 348 ms | 2,500 ms | 18.8 ms | 92 ms |
| 8 | 425.7 | 76.8 | 561 ms | 1,686 ms | 686 ms | 4,176 ms | 18.3 ms | 96 ms |
| 16 | 643.5 | 133.8 | 850 ms | 3,310 ms | 1,973 ms | 5,828 ms | 19.7 ms | 105 ms |
| 32 | 755.9 | 235.5 | 21,962 ms | 4,778 ms | 22,694 ms | 9,917 ms | 20.8 ms | 116 ms |

One RTX 5090 serves 4.8 to 5.5 times the output throughput of the four V100s at caps 4 to 16, with time to first
token 3 to 4 times shorter at the median and per-token decode latency 5 times shorter. At cap 16 the P99 first token
is under 2 s against 5.8 s. The 16 requests fit the pool together (16 x 210 MiB = 3.3 GiB), so nothing queues.
Cap 32 does not fit (32 x 210 MiB = 6.6 GiB against 3.8 GiB): about 18 requests are resident and the rest wait
for a whole request to finish, so throughput rises 17% over cap 16 while the median first token goes from 0.85 s to
22 s, 4.6 times the V100 lane's.

### Long prompts at 16 concurrent

512-token outputs. The V100 column pairs are the production lane's 16-concurrent table (median / P99 time to first
token) and, where it measured them, its load rows (64 x 4K and 48 x 16K prompts, 512-token outputs).

| Prompt | Prompts | 5090 output tok/s | V100 x4 | 5090 median TTFT | V100 x4 | 5090 P99 TTFT | V100 x4 | 5090 TPOT | V100 x4 | 5090 batch time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 64 | 449.6 | 115.1 | 2.2 s | 6.2 s | 16.0 s | 22.3 s | 26.5 ms | 128 ms | 73 s |
| 16K | 48 | 138.9 | 39.8 | 40.0 s | 22.8 s | 52.7 s | 112.5 s | 30.7 ms | 315 ms | 177 s |
| 32K | 32 | 68.1 | | 97.4 s | 132.2 s | 113.2 s | 266.0 s | 34.8 ms | | 241 s |

This is where the pool shows. At 4K prompts 13 of 16 requests are resident; the other three wait for a slot, which
is the gap between the 2.2 s median and the 16 s P99, and still under the V100 lane's 22 s. At 16K only about six
requests are resident, so the sixteen clients are served in three waves of prefill-then-decode, and the median first
token (40 s) is 1.8 times the V100 lane's 23 s, even though the batch finishes in 177 s against about 620 s there
(48 x 512 tokens at 39.8 tok/s) and the P99 is 2.1 times shorter. At 32K three requests are resident and the
medians cross back: 97 s against 132 s, P99 113 s against 266 s. The 5090 prefills 3 to 4 times faster per token,
and the V100 lane holds 4 times the pool; which wins the median depends on whether the requests fit.

### Single-request prefill

One request at a time, 64-token outputs.

| Prompt | 5090 TTFT | Prefill rate | V100 x4 TTFT | Ratio |
| ---: | ---: | ---: | ---: | ---: |
| 16K | 1.73 s | 9,500 tok/s | 5.6 s | 3.2x |
| 32K | 4.12 s | 7,950 tok/s | 12.2 s | 3.0x |
| 64K | 10.96 s | 5,980 tok/s | 28.0 s | 2.6x |
| 112K | 25.57 s | 4,380 tok/s | 75.0 s at 128K | 2.6x per token |

Prefill falls from 9,500 to 4,400 tokens per second across the ladder because the 16 full-attention layers pay the
quadratic cost; the 48 Gated DeltaNet layers are linear in length. Single-stream decode is 60.7 tok/s (16.4 ms per
token).

### Consumer rows

| Prompt / output | Concurrency | Prompts | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 / 4,096 | 1 | 3, three repeats | 60.7 | 376 ms | 380 ms | 16.4 ms | 202 s each |
| 4,096 / 4,096 | 4 | 32 | 208.7 | 1,098 ms | 1,506 ms | 18.9 ms | 628 s |
| 4,096 / 4,096 | 8 | 64 | 405.7 | 1,348 ms | 2,961 ms | 19.4 ms | 646 s |
| 8,192 / 256 | 4 | 20 | 129.8 | 1,927 ms | 3,048 ms | 23.2 ms | 39 s |
| 256 / 256 | 64 (cap 64) | 320 | failed (see Failures) | | | | |

Repeat variation: the three single-stream repeats (seeds 0, 1, 2) agree to the hundredth of a second in duration
and within 0.7% in time to first token (375 to 377 ms at the median; the P99 column is the largest of the three);
outputs are length-fixed by the ignored EOS, so the rows are deterministic to the clock. Decode at concurrency 8 is
6.7 times the single stream at an 18% higher per-token latency.

### Comparison lanes

Each lane changes one thing against the primary lane on the three rows that show prefill under concurrency.
Primary-lane values are repeated for reading.

| Row | Lane | Pool | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Batch time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K / 1K at 16 | primary (FP8, chunk 4,096) | 117,666 | 643.5 | 850 ms | 1,973 ms | 19.7 ms | 99 s |
| | chunk 8,192 | 116,177 | crashed: 32 of 64 requests failed (see Failures) | | | | |
| | BF16 cache | 66,289 | 614.7 | 899 ms | 3,235 ms | 20.5 ms | 104 s |
| 4K / 512 at 16 | primary | 117,666 | 449.6 | 2.20 s | 16.0 s | 26.5 ms | 73 s |
| | chunk 8,192 | 116,177 | 392.1 | 3.16 s | 16.1 s | 25.2 ms | 84 s |
| | BF16 cache | 66,289 | 315.4 | 13.96 s | 16.4 s | 24.5 ms | 104 s |
| 16K / 512 at 16 | primary | 117,666 | 138.9 | 40.0 s | 52.7 s | 30.7 ms | 177 s |
| | chunk 8,192 | 116,177 | 137.2 | 41.2 s | 53.7 s | 29.8 ms | 179 s |
| | BF16 cache | 66,289 | 98.4 | 66.1 s | 78.4 s | 26.7 ms | 250 s |

Single stream on the BF16 cache (4,096 / 4,096, four prompts, one repeat): 60.5 output tok/s, 373 ms to first
token, 16.4 ms per token, against 60.7, 376 ms and 16.4 ms on the FP8 cache: the cache does not change decode.

The 8,192-token chunk is rejected. On its first row the engine ran out of memory mid-run, as it did on
2026-10-21: FlashInfer's FP4 GEMM asked for a 532 MiB MLP intermediate for an 8,192-token step with 357 MiB free,
the engine died, and 32 of the 64 requests failed. Where it survived it was no better: 13% less throughput and a
44% longer median first token at 4K prompts, because a request that arrives mid-step waits for a larger step to
finish, and the same numbers within 3% at 16K, where the pool, not the step, sets the pace. This is the production
lane's finding on its own hardware, with a crash added at this card's 0.97 utilization.

The BF16 cache is rejected for the same reason the FP8 cache was chosen: the pool. At 1K prompts it is close
(16 requests of 272 MiB just overflow 4.2 GiB, which is the P99 first token going from 2.0 s to 3.2 s). At 4K it
holds about ten requests instead of thirteen and the median first token goes from 2.2 s to 14.0 s; at 16K it
holds three or four instead of six and throughput drops 29%. Per-token decode is the same on both caches.

### Output quality

Run on separate deploys of the recipe's configuration on the same node, 2026-11-09 to 2026-11-10, outside the
harness and through the chat endpoint. None of it is in the archive.

GSM8K through `scripts/run_lmeval_gate.py --chat` (lm-eval 0.4.13, seed 0, 16 requests in flight), strict /
flexible match:

| Cache | Thinking | Questions | Generation limit | Score | Standard error |
| --- | --- | ---: | ---: | ---: | ---: |
| FP8 (the recipe) | off | 1,000 | 1,024 tokens | 0.952 / 0.953 | 0.007 |
| BF16 | off | 1,000 | 1,024 tokens | 0.950 / 0.952 | 0.007 |
| FP8 | off | 1,000 | 256 tokens (the task's own) | 0.838 / 0.849 | 0.012 |
| FP8 | on | 500 | 8,192 tokens, no `Question:` stop string | 0.866 / 0.968 | 0.015 / 0.008 |

The FP8 cache costs nothing measurable: 0.952 against 0.950 on the BF16 cache. Published scores for this model
are 95.5 to 97.4 across BF16 and other NVFP4 builds. The 0.829 this report carried before was the task's 256-token
limit cutting worked answers short of their final line, which the third row reproduces. With thinking on the
answers are as often right (0.968 flexible) but shorter, and fewer end in the `#### N` line the strict match wants.
The first two rows are `--chat-template-kwargs '{"enable_thinking": false}' --gen-kwargs '{"max_gen_toks": 1024}'
--limit 1000`; the last is `--gen-kwargs '{"max_gen_toks": 8192, "until": ["<|im_end|>"]}' --limit 500`. With the
task's own `Question:` stop string and thinking on, 164 of 500 answers came back empty.

Long-context tasks from lm-eval's RULER group (prompts built to the stated token length with the model's
tokenizer; 20 samples per task at 8K and 32K, 12 at 64K and 110K), FP8 cache, share of answers correct:

| Task | Sampling | 8K | 32K | 64K | 110K |
| --- | --- | ---: | ---: | ---: | ---: |
| Retrieval, one key (`niah_single_2`) | thinking off, temperature 0 | 1.00 | 1.00 | 1.00 | 1.00 |
| | thinking off, 0.7 / top-p 0.8 / top-k 20 / presence penalty 1.5 | 1.00 | 1.00 | 1.00 | 0.92 |
| | thinking off, 1.0 / top-p 0.95 / top-k 20 | 1.00 | 1.00 | 1.00 | 1.00 |
| | thinking on, 1.0 / top-p 0.95 / top-k 20 | 1.00 | 1.00 | 1.00 | 1.00 |
| Retrieval among distractor keys (`niah_multikey_2`) | all four settings | 1.00 | 1.00 | 1.00 | 1.00 |
| Variable tracking (`ruler_vt`) | all four settings | 1.00 | 1.00 | 1.00 | 1.00 |
| Retrieval, every line a key (`niah_multikey_3`) | thinking off, temperature 0 | | 1.00 | 1.00 | 1.00 |
| | thinking on | | 1.00 | 1.00 | 1.00 |
| Most common words (`ruler_cwe`) | thinking off, temperature 0 | | 1.00 | 1.00 | 1.00 |
| | thinking on, 4,096-token limit | | 0.00 | 0.00 | 0.08 |
| Most frequent words (`ruler_fwe`) | thinking off, temperature 0 | | 1.00 | 1.00 | 0.86 |
| | thinking on, 4,096-token limit | | 0.35 | 0.25 | 0.42 |
| Multi-document question (`ruler_qa_hotpot`) | thinking off, temperature 0 | | 0.65 | 0.83 | 0.83 |
| | thinking on | | 0.65 | 0.83 | 0.92 |

With thinking off nothing degrades with length up to 110K at any sampling setting. The one miss at 110K with the
presence penalty is an answer that began by listing the document's sections and ran out of its 256-token limit.
The multi-document misses read as answer-format mismatches ("Greenwich Village" against "Greenwich Village, New
York City") and do not grow with length. A lookup of one record among thousands of near-identical ones (ten
lookups per length, thinking off, temperature 0) is right 9, 7, 8 and 9 times of 10 at 16K, 64K, 100K and 110K;
every miss returns the neighbouring record's value. Eight 2,900- to 4,100-token reports written from a 100K prompt
at four sampling settings with thinking off are all complete and free of repeated passages.

The thinking-on rows that score near zero are empty answers, and every empty answer found in these checks has one
of two causes, both visible in the response:

- **The reasoning used the whole generation limit** (`finish_reason: length`, every completion token counted as a
  reasoning token, empty content). On the word-counting tasks the model counts thousands of words one at a time. A
  report from a 100K prompt reasoned for about 10,200 tokens before its first answer token, and 3,000 to 7,800
  tokens from an 8K prompt; a 100K prompt leaves 14,688 tokens for reasoning and answer together. The reasoning
  text is coherent planning, not a loop.
- **A stop string matched inside the reasoning** (`finish_reason: stop`, the string in `stop_reason`, empty
  content). vLLM applies a request's stop strings to the reasoning text.

No response was empty with thinking off, and a prompt over the context is refused with an error that names the
114,688 limit.

### Conclusion

The recipe keeps the FP8 cache at 0.97 utilization, 114,688 context and cap 16, and a request that fills the
context answers in 36 s. Against the production lane one RTX 5090 gives about five times the output throughput at
1,000-token prompts with first tokens three to four times sooner, prefills a lone prompt 2.6 to 3.2 times faster
at every length to 112K, and decodes five times faster per token. What it does not give is the pool: 3.8 GiB
against the V100 lane's 288K tokens, so at 16 concurrent prompts of 16K or more requests queue and the median first
token is longer than on the four V100s, while P99 and batch time stay far shorter. Cap 32 buys 17% of throughput for
a 22 s median first token, and the cap-64 deployment does not start. Output quality matches published scores for
the model and does not fall with prompt length with thinking off; with thinking on, a long prompt needs a
generation limit of many thousands of tokens or the answer comes back empty.

### Failures

- `rtx5090x1_mc64_np320_ril256_rol256_mcr64`: the engine refused to start. With 64 slots reserved the pool is
  3.6 GiB, and one 114,688-token request needs 3.68 GiB (the engine estimated 112,896 as the largest context). The
  deploy failed after 146 s and the row has no benchmark or server log. 64 slots of recurrent state would need
  9.4 GiB in any case.
- `rtx5090x1_mc16_np64_ril1000_rol1000_cl98304_…` (chunk 8,192 lane): the engine hit a CUDA out-of-memory error in
  FlashInfer's FP4 GEMM (a 532 MiB allocation with 357 MiB free) during the run and died; the client marked 32 of
  64 requests failed. The record's status is `succeeded` because the client exited normally; its numbers describe
  the 32 requests that completed and are not used above.

### Limitations

- One run per row (the single-stream row has three repeats); differences under about 5% between rows are noise.
- The V100 numbers are the production lane's own report, measured by the 1Cat vLLM fork's bench client at an
  earlier date; the metrics are defined the same way, but it is a cross-report comparison, not a same-run one.
- The benchmark client runs on the node, so time to first token excludes network latency to a real client.
- The quality checks ran outside the harness with 12 to 20 samples per long-context cell, so one miss moves a cell
  by 5 to 8 points. The FP8 and BF16 caches were compared on GSM8K only, not at long context.
- Structured output was not tested. The checkpoint's model card discussions report runaway whitespace under
  JSON-schema output for this build.
- The pool figures hold for this checkpoint's BF16 parts; a checkpoint that quantized the Gated DeltaNet input
  projections and the output head would free up to 10 GiB.

### Archive

`results_rtx5090x1.tar.gz`, root member `2026-11-10_18-18-44/`, holding `benchmark.log`,
`benchmark_rtx5090_x_1.log`, and per row `<variant>_<row-id>.experiment.yaml`, `.benchmark.log` and
`.server.log`. Node addresses are replaced with `node`.

Members: `benchmark.log`, `benchmark_rtx5090_x_1.log`, and these 24 records, each with its `.benchmark.log` and
`.server.log` beside it, except the cap-64 row, whose deploy failed before either was captured and which has only
its record:

- `rtx5090x1_mc16_np32_ril32768_rol512_35a8cb15f265.experiment.yaml`
- `rtx5090x1_mc16_np48_ril16384_rol512_cl65536_eal-m-o-m-n-b-t-4096-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_c7a8cd6e16a8.experiment.yaml`
- `rtx5090x1_mc16_np48_ril16384_rol512_cl98304_eal-m-o-kv-cache-dtype-fp8-m-n-b-t-8192-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_68db2c6d315a.experiment.yaml`
- `rtx5090x1_mc16_np48_ril16384_rol512_deb8da2aacef.experiment.yaml`
- `rtx5090x1_mc16_np64_ril1000_rol1000_bcdb2d8785e8.experiment.yaml`
- `rtx5090x1_mc16_np64_ril1000_rol1000_cl65536_eal-m-o-m-n-b-t-4096-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_7e6891e53872.experiment.yaml`
- `rtx5090x1_mc16_np64_ril1000_rol1000_cl98304_eal-m-o-kv-cache-dtype-fp8-m-n-b-t-8192-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_bbbd77177c20.experiment.yaml`
- `rtx5090x1_mc16_np64_ril4096_rol512_60e6101b4722.experiment.yaml`
- `rtx5090x1_mc16_np64_ril4096_rol512_cl65536_eal-m-o-m-n-b-t-4096-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_b884dd0885d9.experiment.yaml`
- `rtx5090x1_mc16_np64_ril4096_rol512_cl98304_eal-m-o-kv-cache-dtype-fp8-m-n-b-t-8192-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_49c2061b42ee.experiment.yaml`
- `rtx5090x1_mc1_np2_ril112000_rol64_4da7dc379d86.experiment.yaml`
- `rtx5090x1_mc1_np2_ril65536_rol64_b1630eb43283.experiment.yaml`
- `rtx5090x1_mc1_np3_nw0_ril114176_rol512_sc1_srts290_4de011adbb57.experiment.yaml`
- `rtx5090x1_mc1_np3_ril4096_rol4096_r3_548c024bfcd1.experiment.yaml`
- `rtx5090x1_mc1_np4_ril16384_rol64_180c86a146b3.experiment.yaml`
- `rtx5090x1_mc1_np4_ril32768_rol64_4686fd60b3ca.experiment.yaml`
- `rtx5090x1_mc1_np4_ril4096_rol4096_cl65536_eal-m-o-m-n-b-t-4096-n-e-p-c-e-a-t-c-t-c-p-qwen3-c-reas-p-qwen3_07ad491a044b.experiment.yaml`
- `rtx5090x1_mc32_np128_ril1000_rol1000_mcr32_1ff414d2f5bb.experiment.yaml`
- `rtx5090x1_mc4_np16_ril1000_rol1000_de426393875d.experiment.yaml`
- `rtx5090x1_mc4_np20_ril8192_rol256_r1_e50840c2119a.experiment.yaml`
- `rtx5090x1_mc4_np32_ril4096_rol4096_r1_133fc85a7e32.experiment.yaml`
- `rtx5090x1_mc64_np320_ril256_rol256_mcr64_55c75e962ff4.experiment.yaml`
- `rtx5090x1_mc8_np32_ril1000_rol1000_329926b05470.experiment.yaml`
- `rtx5090x1_mc8_np64_ril4096_rol4096_r1_888c34fc6d63.experiment.yaml`
