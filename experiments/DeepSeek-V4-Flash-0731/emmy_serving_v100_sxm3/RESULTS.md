# DeepSeek V4 Flash 0731 — the recipe's serving workload on the Emmy image

The workload of `recipes/DeepSeek-V4-Flash-0731` at its exact envelope, run per exact GPU platform. Each platform
section below describes the archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** How fast does the recipe's configuration serve: the Emmy serving image at the checkpoint's full
1,048,576-token context, a memory share of 0.80, the reasoning and tool-call parsers and the prefix cache on?

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

**Run.** Timestamp `2026-10-07T08:46:32Z`, run ID `20261007T084632Z`, repository revision `4f9dba86d` with this recipe
(committed with this report). Both rows `succeeded`. Every request completed: 24 at the first point and 96 at the
second, 0 failed.

| Point | Row id |
| --- | --- |
| one request, 2,048 in / 128 out | `83ca98957aec` |
| 8 concurrent, 1,024 in / 64 out | `4fb56fa4308f` |

Archive members sit under `2026-10-07_08-46-32/`: `benchmark.log`, `benchmark_v100_x_16.log`, and per row
`<variant>_<row id>.{experiment.yaml,benchmark.log,server.log}`. In the archive copy the host address is replaced
with `v100-host`.

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GiB RAM, 16× Tesla V100-SXM3-32GB behind NVSwitches, driver 580.159.03, nvcc 12.9.86, vLLM
`1.2.3.dev87+gd76126608`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

**Result.**

| Point | Metric | Value |
| --- | --- | ---: |
| one request, 2,048 in / 128 out | Median TTFT | 3,043 ± 5 ms |
| | Mean TPOT | 123.32 ± 0.01 ms |
| | Output throughput | 6.95 ± 0.00 tok/s |
| | Total token throughput | 118.12 ± 0.01 tok/s |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 20.68 ± 0.32 tok/s |
| | Total token throughput | 351.5 ± 5.5 tok/s |
| | Mean TTFT | 8,028 ± 383 ms |
| | P99 TTFT | 11,105 ± 332 ms |
| | Mean TPOT | 265.0 ± 11.0 ms |
| | Median ITL | 219.2 ± 1.4 ms |

Model load and warm-up takes 353 to 354 s, 216 to 220 s of it loading weights.

**Repeat variation.** The one-request point repeats within 0.2%. At 8 concurrent, throughput spreads 1.5% and the
mean time to first token 5% (8.36, 7.61, 8.11 s), which is where prompts land among the prefill steps.

**Comparison.** Three other measurements of the same workload shape, none of them in this archive, so each
comparison is directional.

| Configuration | One request: median TTFT | Mean TPOT | 8 concurrent: output | Median ITL |
| --- | ---: | ---: | ---: | ---: |
| this recipe: Emmy image, context 1,048,576, share 0.80 | 3,043 ms | 123.3 ms | 20.68 tok/s | 219.2 ms |
| the plain fork at its former recipe: context 1,048,576, share 0.90 | 3,778 ms | 152.6 ms | 20.82 tok/s | 205.0 ms |
| Emmy image, context 131,072, otherwise this recipe | 3,040 ms | 120.3 ms | 21.86 tok/s | 195.0 ms |
| Emmy image at its warm envelope (the A/B, `../emmy_ab_v100_sxm3`) | 3,035 ms | 119.4 ms | 21.50 tok/s | 191.0 ms |

The fork row and the 131,072 row are one run each of this recipe file with only the engine block changed
(2026-10-07, three repeats per point, every request completed). The A/B row has the prefix cache off, a 0.90 share
and fewer prompts per repeat, and reports the mean time to first token.

- Against the fork at the same context: the first token of one request in 0.81× the time, 0.81× the time per output
  token, the same throughput at 8 concurrent (0.99×), and a decode step 7% longer. That is the A/B's result at a
  4,096-token context, repeated at 1M.
- The full context costs decode speed in both images. Against the 4,096-token A/B, this image's decode step at 8
  concurrent is 15% longer and the fork's 17% (205.0 against 175.3 ms). Time to first token does not move.
- Against the 131,072-token setting, the 1M context costs 3% per output token for one request and 5% of the
  throughput at 8 concurrent.

What each server logs: a pack hit on all 16 workers, no engine error, the eight Triton kernels the fork compiles once
during the first requests, and one 17-token step on the symbolic path, the smoke-test request. The reported prefix
cache hit rate is 15 to 18%, all of it the warm-up prompt.

**Conclusion.** At the recipe's envelope the image serves one 2,048-token request with its first token after 3.04 s
at 123 ms per output token, and 8 concurrent 1,024-token requests at 20.7 output tokens per second. At the same
context the plain fork takes 24% longer per output token for one request and is level at 8 concurrent.

**Limitations.**

- Two points with random-token prompts; one prompt per repeat is a warm-up cache hit.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
- The comparisons are single runs outside this archive, not interleaved rounds. The interleaved comparison of the
  two images is the A/B, at a 4,096-token context.
- Long prompts, concurrency over long prompts, tool calls, reasoning and answer quality at this envelope were checked
  by probes outside this archive; the recipe's `RESULTS.md` reports them.
