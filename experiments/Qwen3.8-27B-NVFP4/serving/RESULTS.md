# Qwen3.8-27B NVFP4 serving on one RTX 5090

Stock vLLM serving of `Inferact/Qwen3.8-27B-NVFP4` on one RTX 5090, measured against the 4x V100 SXM2 16GB
production lane of `recipes/Qwen3.8-27B-FP8`. The question: how much context and concurrency does the card's
32 GB leave after a 23 GiB checkpoint, and is prefill under concurrent requests faster than the four V100s, which
is where the production lane is slow. The recipe this experiment qualifies is `recipes/Qwen3.8-27B-NVFP4`.

## NVIDIA GeForce RTX 5090 x1

Run `20261021T210320Z`, started 2026-10-21T21:03:20Z on a rented CloudRift node (`riftvm`, Ubuntu 24.04.1,
kernel 6.8.0-134-generic, driver 580.159.03, Docker 29.6.1), one RTX 5090 (sm_120, 32,607 MiB). Image
`vllm/vllm-openai@sha256:befccf0e…` (vLLM v0.32.0). Archive `results_rtx5090x1.tar.gz`, root member
`2026-10-21_21-03-20/`.

### Protocol

Every row serves revision `6128240e` text-only through `vllm bench serve` with random prompts, temperature 0,
ignored EOS, seed 0, two warm-up requests, one repeat unless stated, client concurrency equal to the row's
value. The primary lane is the recipe's configuration: FP8 key/value cache, 0.97 memory utilization, 114,688
context, cap 16, 4,096-token prefill chunk. Two comparison lanes change one thing each on the three rows that show
prefill under concurrency (1K, 4K and 16K prompts at 16 concurrent): an 8,192-token prefill chunk (context
98,304), and the BF16 key/value cache (context 65,536). Every row redeploys the engine from scratch.

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
| Load to health | weights 18.7 s, torch.compile 33.7 s, CUDA graphs 11 s, warm-up 35 s; 146 s in all |

The recurrent state, not the attention cache, is what caps concurrency: at 1,000-token prompts it is 70% of a
request's pool cost. The production lane's four V100s hold a 288,281-token pool for the same model, so every row
at 16 concurrent runs with all 16 requests resident there, while here anything longer than 4K prompts queues.

### Production comparison: 1,000-token prompts, 1,000-token outputs

Client concurrency equals the server cap. V100 numbers are the production lane's own report (one run per point).

| Cap | Output tok/s 5090 | V100 x4 | Median TTFT 5090 | V100 x4 | P99 TTFT 5090 | V100 x4 | Median TPOT 5090 | V100 x4 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 213.9 | 40.8 | 340 ms | 1,024 ms | 344 ms | 2,500 ms | 18.4 ms | 92 ms |
| 8 | 429.9 | 76.8 | 559 ms | 1,686 ms | 678 ms | 4,176 ms | 18.1 ms | 96 ms |
| 16 | 630.1 | 133.8 | 846 ms | 3,310 ms | 1,994 ms | 5,828 ms | 20.1 ms | 105 ms |
| 32 | failed (see Failures) | 235.5 | | 4,778 ms | | 9,917 ms | | 116 ms |

One RTX 5090 serves 4.7 to 5.6 times the output throughput of the four V100s at the same cap, with time to first
token 3 to 4 times shorter at the median and per-token decode latency 5 times shorter. At cap 16 the P99 first token
is under 2 s against 5.8 s. The 16 requests fit the pool together (16 x 210 MiB = 3.3 GiB), so nothing queues.

### Long prompts at 16 concurrent

512-token outputs. The V100 column pairs are the production lane's 16-concurrent table (median / P99 time to first
token) and, where it measured them, its load rows (64 x 4K and 48 x 16K prompts, 512-token outputs).

| Prompt | Prompts | 5090 output tok/s | V100 x4 | 5090 median TTFT | V100 x4 | 5090 P99 TTFT | V100 x4 | 5090 TPOT | V100 x4 | 5090 batch time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4K | 64 | 446.6 | 115.1 | 2.2 s | 6.2 s | 16.1 s | 22.3 s | 26.7 ms | 128 ms | 73 s |
| 16K | 48 | 138.5 | 39.8 | 40.2 s | 22.8 s | 53.0 s | 112.5 s | 30.9 ms | 315 ms | 178 s |
| 32K | 32 | 68.1 | | 97.4 s | 132.2 s | 113.2 s | 266.0 s | 34.7 ms | | 241 s |

This is where the pool shows. At 4K prompts 13 of 16 requests are resident; the other three wait for a slot, which
is the gap between the 2.2 s median and the 16 s P99, and still under the V100 lane's 22 s. At 16K only about six
requests are resident, so the sixteen clients are served in three waves of prefill-then-decode, and the median first
token (40 s) is 1.8 times the V100 lane's 23 s, even though the batch finishes in 178 s against about 620 s there
(48 x 512 tokens at 39.8 tok/s) and the P99 is 2.1 times shorter. At 32K three requests are resident and the
medians cross back: 97 s against 132 s, P99 113 s against 266 s. The 5090 prefills 3 to 4 times faster per token,
and the V100 lane holds 4 times the pool; which wins the median depends on whether the requests fit.

### Single-request prefill

One request at a time, 64-token outputs. The streamed probe column is the exploration deploy's planted-passphrase
request (same configuration, chat endpoint, one request), which retrieved the phrase at every size.

| Prompt | 5090 TTFT (bench) | Prefill rate | Streamed probe | V100 x4 TTFT | Ratio |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16K | 1.72 s | 9,500 tok/s | 1.67 s | 5.6 s | 3.2x |
| 32K | 4.13 s | 7,934 tok/s | 3.95 s | 12.2 s | 3.0x |
| 64K | 11.07 s | 5,923 tok/s | 10.78 s | 28.0 s | 2.5x |
| 112K | 25.90 s | 4,325 tok/s | 25.73 s | 75.0 s at 128K | 2.5x per token |

Prefill falls from 9,500 to 4,300 tokens per second across the ladder because the 16 full-attention layers pay the
quadratic cost; the 48 Gated DeltaNet layers are linear in length. Single-stream decode is 60.9 tok/s (16.4 ms per
token) at every prompt length measured.

### Consumer rows

| Prompt / output | Concurrency | Prompts | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 / 4,096 | 1 | 3, three repeats | 60.9 | 375 ms | 376 ms | 16.3 ms | 202 s each |
| 4,096 / 4,096 | 4 | 32 | 205.2 | 1,090 ms | 1,498 ms | 19.2 ms | 639 s |
| 4,096 / 4,096 | 8 | 64 | 405.4 | 1,344 ms | 2,956 ms | 19.4 ms | 647 s |
| 8,192 / 256 | 4 | 20 | 129.2 | 1,920 ms | 3,033 ms | 23.4 ms | 40 s |
| 256 / 256 | 64 (cap 64) | 320 | failed (see Failures) | | | | |

Repeat variation: the three single-stream repeats (seeds 0, 1, 2) agree to the hundredth of a second in duration
and within 0.5% in time to first token; outputs are length-fixed by the ignored EOS, so the rows are deterministic
to the clock. Decode at concurrency 8 is 6.7 times the single stream at a 19% higher per-token latency.

### Comparison lanes

Each lane changes one thing against the primary lane on the three rows that show prefill under concurrency.
Primary-lane values are repeated for reading.

| Row | Lane | Pool | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | Batch time |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1K / 1K at 16 | primary (FP8, chunk 4,096) | 117,666 | 630.1 | 846 ms | 1,994 ms | 20.1 ms | 102 s |
| | chunk 8,192 | 116,177 | crashed: 32 of 64 requests failed (see Failures) | | | | |
| | BF16 cache | 66,289 | 623.3 | 894 ms | 3,195 ms | 20.3 ms | 103 s |
| 4K / 512 at 16 | primary | 117,666 | 446.6 | 2.19 s | 16.1 s | 26.7 ms | 73 s |
| | chunk 8,192 | 116,177 | 398.0 | 3.13 s | 15.9 s | 24.8 ms | 82 s |
| | BF16 cache | 66,289 | 317.2 | 13.86 s | 16.3 s | 24.3 ms | 103 s |
| 16K / 512 at 16 | primary | 117,666 | 138.5 | 40.2 s | 53.0 s | 30.9 ms | 178 s |
| | chunk 8,192 | 116,177 | 137.3 | 41.1 s | 53.6 s | 29.8 ms | 179 s |
| | BF16 cache | 66,289 | 98.4 | 66.1 s | 78.4 s | 26.7 ms | 250 s |

Single stream on the BF16 cache (4,096 / 4,096, four prompts, one repeat): 59.4 output tok/s, 372 ms to first
token, 16.8 ms per token, against 60.9, 375 ms and 16.3 ms on the FP8 cache: the cache does not change decode.

The 8,192-token chunk is rejected. On its first row the engine ran out of memory mid-run: FlashInfer's FP4 GEMM
asked for a 532 MiB MLP intermediate for an 8,192-token step with 357 MiB free, the engine restarted, and 32 of the
64 requests failed. Where it survived it was no better: 11% less throughput and a 43% longer median first token at
4K prompts, because a request that arrives mid-step waits for a larger step to finish, and the same numbers within
2.5% at 16K, where the pool, not the step, sets the pace. This is the production lane's finding on its own
hardware, with a crash added at this card's 0.97 utilization.

The BF16 cache is rejected for the same reason the FP8 cache was chosen: the pool. At 1K prompts it is even
(16 requests of 272 MiB just overflow 4.2 GiB, which is the P99 first token going from 2.0 s to 3.2 s). At 4K it
holds about ten requests instead of thirteen and the median first token goes from 2.2 s to 13.9 s; at 16K it
holds three or four instead of six and throughput drops 29%. Per-token decode is the same on both caches.

### Capability and accuracy checks

Run on the exploration deploy of the same configuration, outside the harness (see the recipe report for the
commands): a plain chat answer ("12 times 12" gives "144"), a tool call with `tool_choice: auto` returning
structured `tool_calls` (`get_weather`, `{"city": "Oslo"}`), and a thinking request landing 45 reasoning tokens in
the `reasoning` field. GSM8K through the chat endpoint with thinking off, 1,000 questions, seed 0
(`scripts/run_lmeval_gate.py --chat --chat-template-kwargs '{"enable_thinking": false}' --limit 1000`): FP8 cache
0.829 strict / 0.849 flexible, BF16 cache 0.845 / 0.859, standard error 0.012 and 0.011. With thinking on the same
gate scores 0.395 at 200 questions on either cache, because the hidden reasoning spends the task's 256-token budget.

### Conclusion

The recipe keeps the FP8 cache at 0.97 utilization, 114,688 context and cap 16. Against the production lane one
RTX 5090 gives about five times the output throughput at 1,000-token prompts with first tokens three to four times
sooner, prefills a lone prompt 2.5 to 3.2 times faster at every length to 112K, and decodes five times faster per
token. What it does not give is the pool: 3.8 GiB against the V100 lane's 288K tokens, so at 16 concurrent
prompts of 16K or more requests queue and the median first token is longer than on the four V100s, while P99 and
batch time stay far shorter. Cap 32 and cap 64 are not useful on this card: the 147 MiB per request of recurrent
state means 18 requests fill the pool at 1K prompts, and the cap-64 deployment did not boot.

### Failures

- `rtx5090x1_mc32_np128_ril1000_rol1000_mcr32`: the SSH session carrying the benchmark client from the operator's
  machine to the node timed out during the main run ("Read from remote host: Operation timed out"); the server
  was healthy before and after. No measurement; the row is kept as failed. The pool arithmetic says cap 32 only
  adds queueing (32 x 210 MiB = 6.6 GiB against 3.8 GiB).
- `rtx5090x1_mc64_np320_ril256_rol256_mcr64`: the engine ran out of memory after allocating the pool, in the
  FlashInfer warm-up that follows profiling (a 128 MiB workspace with 69 MiB free), and the health check timed out
  after 3,600 s. 64 slots of recurrent state would need 9.4 GiB in any case.
- `rtx5090x1_mc16_np64_ril1000_rol1000_cl98304_…` (chunk 8,192 lane): the engine hit a CUDA out-of-memory error in
  FlashInfer's FP4 GEMM (`mm_fp4`, a 532 MiB allocation with 357 MiB free) 100 s into the run and restarted; the
  client marked 32 of 64 requests failed ("Never received a valid chunk to calculate TTFT"). The record's status is
  `succeeded` because the client exited normally; its numbers describe the 32 requests that completed and are not
  used above.

### Limitations

- One run per row (the single-stream row has three repeats); differences under about 5% between rows are noise.
- The V100 numbers are the production lane's own report, measured by the 1Cat vLLM fork's bench client at an
  earlier date; the metrics are defined the same way, but it is a cross-report comparison, not a same-run one.
- The benchmark client runs on the node, so time to first token excludes network latency to a real client.
- GSM8K at 1,000 questions bounds the FP8 cache's accuracy cost to within about 3 points at 95% confidence; it says
  nothing about long-context quality beyond the planted-passphrase retrievals at 16K to 112K.
- The pool figures hold for this checkpoint's BF16 parts; a checkpoint that quantized the Gated DeltaNet input
  projections and the output head would free up to 10 GiB.

### Archive

`results_rtx5090x1.tar.gz`, root member `2026-10-21_21-03-20/`, holding `benchmark.log`,
`benchmark_rtx5090_x_1.log`, and per row `<variant>_<row-id>.experiment.yaml`, `.benchmark.log` and
`.server.log` (the two failed rows have no benchmark stanza). Node addresses are replaced with `node`.

Members: `benchmark.log`, `benchmark_rtx5090_x_1.log`, and these 23 records, each with its `.benchmark.log` and
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

