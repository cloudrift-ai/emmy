# DeepSeek V4 Flash 0731 — what the prefix cache is worth on the Emmy serving image

One image, served with vLLM's prefix cache on and off, run per exact GPU platform. Each platform section below
describes the archive named in it and nothing else.

## NVIDIA Tesla V100 SXM3 32GB × 16 — `results_v100x16.tar.gz`

**Question.** The Emmy serving image starts with the prefix cache off. How much faster does it serve with the cache
on when prompts share a prefix, and what does turning it on cost when they share nothing?

**Protocol.** `emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_prefix_cache_v100_sxm3 --ssh <host>` against a
pre-allocated host. Every row serves `cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-992de5c8` at the envelope it
was warmed at: TP8 × PP2, fp16, fp8 KV cache, block size 256, context 4,096, 4,112 batched tokens,
`gpu_memory_utilization` 0.90, at most 8 sequences. The image's start script passes `--no-enable-prefix-caching`; a
flag given to the container comes after it, so `--enable-prefix-caching` turns the cache on. Each server log confirms
its setting.

Three arms, each at two points, with greedy decoding and `ignore_eos`:

| Arm | Cache | One request at a time | 8 concurrent requests |
| --- | --- | --- | --- |
| shared, cache on | on | 1,536 shared + 512 own tokens | 768 shared + 256 own tokens |
| shared, cache off | off | the same prompts | the same prompts |
| nothing shared, cache on | on | 2,048 own tokens | 1,024 own tokens |

The one-request point sends 4 prompts with 128 output tokens after 2 warm-up requests. The 8-concurrent point sends 8
prompts with 64 output tokens after 8 warm-up requests. The shared part is `benchmark.random_prefix_len`: the client
draws it once per run, so every request of a repeat starts with the same tokens. The cache holds whole 256-token
blocks, so the shared lengths are multiples of 256.

Every row is a fresh deployment, and the client runs 3 repeats against it from seeds 731, 732 and 733, each with its
own prompts and its own shared prefix. Spreads below are the sample standard deviation over those 3.

The client's warm-up sends its first prompt, once per warm-up request. With the cache on this has two effects:

- the shared prefix is already cached when measurement starts, so the shared arm measures warm hits only;
- that first prompt is itself a full hit when it is measured: one of 4 prompts at the one-request point, one of 8 at
  the other. In the arm that shares nothing it is the only hit.

**Run.** Timestamp `2026-10-06T03:45:52Z`, run ID `20261006T034552Z`, repository revision `b6a2bc451` plus the
`benchmark.random_prefix_len` workload field and this recipe (committed with this report). All 6 rows `succeeded`.
Every request completed: 12 per one-request row and 24 per concurrent row, 0 failed.

| Arm | One request (row id) | 8 concurrent (row id) |
| --- | --- | --- |
| shared, cache on | `7ad213240cf1` | `3529c22bfed6` |
| shared, cache off | `be98257cc0e1` | `c7e87e950b9a` |
| nothing shared, cache on | `8035571be3a7` | `2bd2f7972a19` |

Archive members sit under `2026-10-06_03-45-52/`:

- `benchmark.log` and `benchmark_v100_x_16.log`;
- per row, `<variant>_<row id>.{experiment.yaml,benchmark.log,server.log}`, 18 files.

A variant name ends in `-e-p-c` for the cache on and `-n-e-p-c` for the cache off, and carries `rpl<shared tokens>`. In
the archive copy the host address is replaced with `v100-host`.

**Machine and software.** Ubuntu 24.04.1, kernel 6.8.0-124-generic, Intel Xeon Platinum 8168 (80 logical CPUs),
1,338 GiB RAM, 16× Tesla V100-SXM3-32GB behind NVSwitches, driver 580.159.03, nvcc 12.9.86, vLLM
`1.2.3.dev87+gd76126608`. Model revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

**Result.**

| Point | Metric | Shared, cache off | Shared, cache on | On / off | Nothing shared, cache on |
| --- | --- | ---: | ---: | ---: | ---: |
| one request, 2,048 in / 128 out | Median TTFT | 3,038 ± 1 ms | 1,135 ± 4 ms | 0.37× | 3,027 ± 4 ms |
| | Mean TTFT | 3,039 ± 2 ms | 1,052 ± 6 ms | 0.35× | 2,469 ± 5 ms |
| | Output throughput | 7.02 ± 0.01 tok/s | 7.89 ± 0.01 tok/s | 1.12× | 7.25 ± 0.00 tok/s |
| | Mean TPOT | 119.54 ± 0.03 ms | 119.36 ± 0.01 ms | 1.00× | 119.53 ± 0.01 ms |
| 8 concurrent, 1,024 in / 64 out | Output throughput | 21.49 ± 0.32 tok/s | 33.41 ± 0.63 tok/s | 1.56× | 22.50 ± 0.40 tok/s |
| | Mean TTFT | 8,304 ± 145 ms | 3,195 ± 97 ms | 0.38× | 7,602 ± 686 ms |
| | Mean TPOT | 244.1 ± 5.6 ms | 192.3 ± 2.9 ms | 0.79× | 238.5 ± 4.5 ms |
| | Median ITL | 190.99 ± 0.03 ms | 190.87 ± 0.02 ms | 1.00× | 191.25 ± 0.02 ms |

At the one-request point the median is the number for a request whose prefix is cached: the mean also holds the one
prompt in four that is a full hit.

**Repeat variation.** The one-request point repeats within 0.4% in every arm. At 8 concurrent, throughput spreads
1.5% with the cache off and 1.9% with it on. The arm that shares nothing spreads 9% in mean time to first token at 8
concurrent (8.39, 7.29, 7.13 s), which is where its one cached prompt lands among the prefill steps.

**Comparison.** The first two arms are a direct comparison: same image, host, prompts and flags, one switch apart.

- With three quarters of each prompt shared, the time to first token falls to 0.37× for one request and 0.38× at 8
  concurrent. It does not fall to a quarter: a 512-token prefill takes 1.14 s where a 2,048-token one takes 3.04 s, so
  a prefill step has a fixed cost of roughly 0.4 to 0.5 s.
- Throughput rises 12% for one request, where 128 output tokens take 15 s either way, and 56% at 8 concurrent, where
  prefill was the larger share of each request.
- Decode does not change: 119.4 ms per output token for one request and 190.9 ms between tokens at 8 concurrent,
  against 119.5 and 191.0 with the cache off. The lower mean time per output token at 8 concurrent comes from
  shorter prefill steps interrupting decode, not from a faster decode step.
- Turning the cache on costs nothing that this run can see. For one request, a prompt that shares nothing reaches
  its first token in 3,027 ms (median) against 3,038 ms with the cache off.
- The arm that shares nothing is not a clean control at 8 concurrent. One of its 8 prompts is the warm-up's and a
  full hit, which is where its 5% higher throughput comes from.
- A full hit is not free. vLLM recomputes at least one token and caches whole blocks, so the last 256-token block is
  computed again. From the mean and median of the arm that shares nothing, the fully cached 2,048-token prompt
  reaches its first token in about 0.8 s; this is derived, not measured per request.

The cache-off arm reproduces the A/B's Emmy numbers for the same request sizes (`../emmy_ab_v100_sxm3`): 3,039 ms
against 3,035 ms to first token for one request, and 21.49 against 21.50 tok/s at 8 concurrent.

What each server logs is the same in all six rows and the same as in the A/B: a pack hit on all 16 workers, no new
kernel compile, no error, and one 17-token step on the symbolic path, the smoke-test request. The reported prefix
cache hit rate is 0% with the cache off and 65 to 70% in the shared arm.

**Conclusion.** On this image the prefix cache does what its share of the prompt allows. With 75% of each prompt
shared and already cached, requests reach their first token in a bit over a third of the time, and 8 concurrent
requests deliver 56% more output tokens per second. Decode speed is unchanged, and turning the cache on shows no cost
when nothing is shared.

**Limitations.**

- One share of the prompt (75%) at two points. The gain scales with the shared share and with how much of a
  request's time is prefill.
- The shared prefix is always warm, and prompts are random tokens. A real workload misses on first use and evicts
  under memory pressure; neither is measured.
- Generated text was not compared between cache on and cache off. The harness records timings only, so this run does
  not show that a cached prefix gives the same output.
- The fork was not run with its cache on, so this says nothing about Emmy against the fork in that mode.
- Every row is a fresh deployment, so the numbers are warmed but not sustained load.
