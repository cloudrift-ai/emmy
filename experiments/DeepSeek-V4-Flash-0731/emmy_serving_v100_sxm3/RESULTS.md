# DeepSeek V4 Flash 0731 — the recipe's serving workload on the Emmy image

The workload of `recipes/DeepSeek-V4-Flash-0731` at its exact envelope, run per exact GPU platform. Each platform
section below describes the archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** How fast does the recipe's configuration serve: the Emmy serving image at a 131,072-token context, a
memory share of 0.80, the reasoning and tool-call parsers and the prefix cache on?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_serving_v100_sxm3 --ssh <host>` against a
pre-allocated host. The engine block is the recipe's, field for field: image
`cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-992de5c8`, TP8 × PP2, fp16, fp8 KV cache, block size 256, 4,112
batched tokens, at most 8 sequences. Two points, with greedy decoding and `ignore_eos`:

- one request at a time: 8 prompts of 2,048 random tokens, 128 output tokens, 2 warm-up requests;
- 8 concurrent requests: 32 prompts of 1,024 random tokens, 64 output tokens, 8 warm-up requests.

Each point is its own deployment, and the client runs 3 repeats against it from seeds 731, 732 and 733. Spreads below
are the sample standard deviation over those 3.

The prompts share nothing, so the prefix cache has nothing to serve, with one exception the client makes: its
warm-up sends the first prompt, which is then a full cache hit when it is measured. That is one prompt in 8 at the
first point and one in 32 at the second. The first point is therefore read by its median time to first token.

**Run.** Timestamp `2026-10-07T05:41:36Z`, run ID `20261007T054136Z`, repository revision `4f9dba86d` with this recipe
(committed with this report). Both rows `succeeded`. Every request completed: 24 at the first point and 96 at the
second, 0 failed.

| Point | Row id |
| --- | --- |
| one request, 2,048 in / 128 out | `83ca98957aec` |
| 8 concurrent, 1,024 in / 64 out | `4fb56fa4308f` |

Archive members sit under `2026-10-07_05-41-36/`: `benchmark.log`, `benchmark_v100_x_16.log`, and per row
`<variant>_<row id>.{experiment.yaml,benchmark.log,server.log}`. In the archive copy the host address is replaced
with `v100-host`.

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GiB RAM, 16× Tesla V100-SXM3-32GB behind NVSwitches, driver 580.159.03, nvcc 12.9.86, vLLM
`1.2.3.dev87+gd76126608`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

**Result.**

| Point | Metric | Value |
| --- | --- | ---: |
| one request, 2,048 in / 128 out | Median TTFT | 3,040 ± 4 ms |
| | Mean TPOT | 120.25 ± 0.01 ms |
| | Output throughput | 7.10 ± 0.00 tok/s |
| | Total token throughput | 120.67 ± 0.02 tok/s |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 21.86 ± 0.17 tok/s |
| | Total token throughput | 371.7 ± 3.0 tok/s |
| | Mean TTFT | 7,826 ± 478 ms |
| | P99 TTFT | 11,043 ± 384 ms |
| | Mean TPOT | 246.9 ± 10.2 ms |
| | Median ITL | 194.95 ± 0.09 ms |

Model load and warm-up takes 340 to 352 s, 214 s of it loading weights.

**Repeat variation.** The one-request point repeats within 0.2%. At 8 concurrent, throughput spreads 0.8% and the
mean time to first token 6% (8.34, 7.75, 7.39 s), which is where prompts land among the prefill steps.

**Comparison.** The A/B against the plain fork (`../emmy_ab_v100_sxm3`) measured this image at the envelope it was
warmed at: context 4,096, memory share 0.90, prefix cache off. This is a directional comparison, since the envelope
and the number of prompts differ.

- One request: 3,040 ms to first token here against 3,035 ms there, and 120.25 ms per output token against 119.39.
- 8 concurrent: 21.86 tok/s against 21.50, with one cached prompt in 32 here; 194.95 ms between tokens against
  190.98.

So the longer context, the lower memory share and the parsers cost about 1% per output token for one request and 2%
per decode step at 8 concurrent, and nothing in time to first token.

What each server logs: a pack hit on all 16 workers, no engine error, the eight Triton kernels the fork compiles once
during the first requests, and one 17-token step on the symbolic path, the smoke-test request. The reported prefix
cache hit rate is 15 to 18%, all of it the warm-up prompt.

**Conclusion.** At the recipe's envelope the image serves one 2,048-token request with its first token after 3.04 s
at 120 ms per output token, and 8 concurrent 1,024-token requests at 21.9 output tokens per second. That is the
image's speed at its warm envelope to within 2%.

**Limitations.**

- Two points with random-token prompts; one prompt per repeat is a warm-up cache hit.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
- Long prompts, concurrency over long prompts, tool calls, reasoning and answer quality at this envelope were checked
  by probes outside this archive; the recipe's `RESULTS.md` reports them.
