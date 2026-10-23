# GLM-5.2-FP8 serving throughput

**Question.** On one 8-GPU node, which engine and which split of the single replica gives the most sustained output
tokens/s on 32,768-token random prompts with 4,096 output tokens? The 756 GB of block-FP8 weights fit once on such a
node, so the choice is how to split one replica: tensor parallelism (TP), TP with expert parallelism (EP), or
data-parallel attention with EP, each with or without MTP speculative decoding.

**Protocol.** `vllm bench serve` random dataset, 32,768 input and 4,096 output tokens, seed 0, temperature 0, ignore
EOS, unique prompts, no prefix reuse, two warm-up requests. Every lane serves 40,960 tokens of context. The client
concurrency equals the server's total admission limit and doubles until output tokens/s stops rising or requests
are preempted. The figure of merit is output tokens/s at that peak with zero failed requests.

## NVIDIA H200 141GB x8

Measured 2026-10-22 to 2026-10-23 on one GCP spot node. The final run is `2026-10-23_00-19-23` (run ID
`20261023T001923Z`); the exploration runs before it and the datacenter rows after it are in the same archive.

**Answer.** vLLM `v0.32.0` with data-parallel attention on 8 ranks, the experts split across them (DP8+EP8), an FP8
KV cache and MTP with 3 draft tokens, 2,048 batched tokens per rank, `--gpu-memory-utilization 0.95`, 8 requests per
rank (64 for the node): **1,316.7 output tokens/s** (mean of three repeats, 1,299.6 to 1,329.9), 11,850.7 total
tokens/s, 0.32 requests/s, 0 failed requests, at concurrency 64. The same lane without MTP gave a mean of 1,108.1.

**System.** 8x NVIDIA H200 141GB (driver 580.178.04, NVLink), Ubuntu 24.04, 224 CPU threads. `vllm/vllm-openai:v0.32.0`
(CUDA 13.0, bundled DeepGEMM for the FP8 linears); `lmsysorg/sglang:v0.5.22`. The vLLM client `vllm bench serve` from
`v0.32.0` measured every lane, SGLang's included (`benchmark.client_image`).

**KV capacity, as the engines report it.** Under data parallelism each rank keeps its own requests' cache; under
TP8 every rank keeps a copy of all of it. It sets each lane's ladder.

| Lane | Weights per GPU | KV per GPU | Tokens | 36,864-token sequences on the node |
| --- | ---: | ---: | ---: | ---: |
| vLLM TP8, 0.9, 131,072-token limit | 88.78 GiB | 31.64 GiB | 366,784, shared | 9.9 |
| vLLM TP8, 0.9, 40,960-token limit | 88.78 GiB | 33.13 GiB | 384,064, shared | 10.4 |
| vLLM DP8+EP8, 0.9, 8,192 batched tokens | 106.01 GiB | -2.3 GiB | none: does not start | 0 |
| vLLM DP8+EP8, 0.95, 2,048 batched tokens | 106.01 GiB | 20.56 GiB | 238,272 per rank | 51.7 |
| the same, FP8 cache | 106.01 GiB | 20.56 GiB | 395,968 per rank | 85.9 |
| the same, FP8 cache and MTP 3 | | | 320,128 per rank | 69.5 |
| the same, FP8 cache and 4,096 batched tokens | | | 283,712 per rank | 61.6 |
| the same, FP8 cache and DeepEP low-latency | | | 337,984 per rank | 73.3 |
| SGLang DP8+DeepEP, 0.9 | 103.98 GB | 20.18 GB | 233,856 per rank | 50.7 |
| SGLang DP8+DeepEP, FP8 cache, 32,768-token chunk, 0.85 | 103.98 GB | 13.28 GB | 264,256 per rank | 57.3 |

**Final run: the selected lane and its comparison, three repeats each** (repeat `i` uses seed `i`; 192 requests at
concurrency 64 per repeat).

| Lane | Output tok/s per repeat | Mean output tok/s | Mean total tok/s | Req/s | Mean TPOT | MTP acceptance | Failed |
| --- | --- | ---: | ---: | ---: | ---: | --- | ---: |
| FP8 cache + MTP 3 (`vllm-dp8ep-fp8kv-mtp3`) | 1,320.8 / 1,329.9 / 1,299.6 | 1,316.7 | 11,850.7 | 0.32 | 39.96 ms | 92.4%, 3.77 tokens per step | 0 |
| FP8 cache (`vllm-dp8ep-fp8kv`) | 1,062.8 / 1,043.8 / 1,217.7 | 1,108.1 | 9,973.0 | 0.27 | 47.6 ms | | 0 |

MTP adds 18.8% on the means, and its repeats spread 2.3% against 16% without it.

**Exploration lanes, one run each** (output and total tokens/s with zero failed requests unless noted).

| Lane | Client concurrency | Server limit | Requests | Output tok/s | Total tok/s | Req/s | Mean TPOT | Notes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| vLLM DP8+EP8, 0.9, 8,192 batched tokens | 16 to 64 | | | | | | | did not start: no KV memory |
| vLLM DP8+EP8 (`vllm-dp8ep`) | 16 | 2 per rank | 64 | 418.8 | 3,769.5 | 0.10 | 28.4 ms | |
| vLLM DP8+EP8 (`vllm-dp8ep`) | 32 | 4 per rank | 128 | 661.9 | 5,957.3 | 0.16 | 36.5 ms | |
| vLLM DP8+EP8 (`vllm-dp8ep`) | 64 | 8 per rank | 192 | 871.6 | 7,844.0 | 0.21 | 48.1 ms | 26 preemptions |
| vLLM TP8 (`vllm-tp8`) | 8 | 8 | 32 | 395.2 | 3,556.7 | 0.10 | 17.9 ms | |
| SGLang DP8+DeepEP (`sglang-dp8ep`) | 32 | 32 | 128 | 661.1 | 5,950.0 | 0.16 | 28.7 ms | |
| vLLM DP8+EP8, FP8 cache (`vllm-dp8ep-fp8kv`) | 64 | 8 per rank | 192 | 1,211.7 | 10,905.3 | 0.30 | 46.9 ms | |
| vLLM DP8+EP8, FP8 cache | 80 | 10 per rank | 240 | 1,000.2 | 9,002.0 | 0.24 | 63.0 ms | no preemption |
| vLLM DP8+EP8, FP8 cache | 96 | 12 per rank | 288 | 1,077.2 | 9,694.9 | 0.26 | 67.3 ms | at least 16 preemptions |
| vLLM DP8+EP8, FP8 cache | 128 | 16 per rank | 384 | | | | | cancelled before its client started |
| vLLM DP8+EP8, FP8 cache, MTP 3 (`vllm-dp8ep-fp8kv-mtp3`) | 64 | 8 per rank | 192 | 1,330.5 | 11,974.3 | 0.32 | 39.8 ms | |
| vLLM DP8+EP8, FP8 cache, 4,096 batched tokens (`vllm-dp8ep-fp8kv-mbt4096`) | 64 | 8 per rank | 192 | 987.1 | 8,883.9 | 0.24 | 49.3 ms | 21 preemptions |
| vLLM DP8+EP8, FP8 cache, DeepEP low-latency (`vllm-dp8ep-fp8kv-deepepll`) | 64 | 8 per rank | 192 | 754.8 | 6,793.4 | 0.18 | 64.9 ms | 96 cancelled |
| SGLang DP8+DeepEP, FP8 cache, 32,768-token chunk, 0.9 | 64 | 64 | 192 | | | | | did not start: CUDA out of memory |
| SGLang DP8+DeepEP, FP8 cache, 32,768-token chunk, 0.85 (`sglang-dp8ep-fp8kv`) | 64 | 64 | 192 | 1,118.2 | 10,063.8 | 0.27 | 36.6 ms | |

**What decided it.**

- *Data parallelism beats TP by the cache.* TP8 decodes fastest per request (17.9 ms TPOT) but keeps a copy of the
  whole MLA cache on every rank, about ten 36,864-token sequences, so its ladder ends at 8 and gives 395.2 output
  tokens/s. DP8+EP8 spreads the cache over the ranks and reaches 871.6 with a BF16 cache. TP8 with EP was not run:
  EP changes how the experts are split, not the replicated cache, so it has the same ten-sequence ceiling.
- *Memory on a DP rank.* Each rank holds all attention weights, 106 GiB in all. At 0.9 and vLLM's default 8,192
  batched tokens the profiled activation peak (the all-gathered MoE input of 8 ranks) leaves no room for the cache;
  0.95 with 2,048 batched tokens leaves 20.56 GiB.
- *The FP8 cache* holds 1.66 times the tokens. It passed the accuracy gate below and lifted the peak by 39% at
  concurrency 64.
- *The peak is at 64.* Above it, TPOT grows faster than concurrency: 80 gave 1,000.2 without preemption, 96 gave
  1,077.2 with preemption. Single runs of three waves differ by up to 16% (the final run's three repeats without
  MTP), mostly from how vLLM's balancer spreads requests over the ranks and from the last wave's tail, so 64 is the
  peak these runs support, not a sharp optimum.
- *MTP* wins on the means and steadies the runs. Its acceptance here is high because greedy continuations of
  random-token prompts with EOS ignored become repetitive. On natural text (the GSM8K prompts on the same server) it
  accepted 77.5% of drafts, 3.33 tokens per step, so most of the gain should carry over. The gain is moderate even at
  92% acceptance because 32,768 of every 36,864 tokens are prompt: a step that prefills a 2,048-token chunk on any
  rank holds every rank at the expert-parallel layers.
- *Engines.* At concurrency 32 SGLang and vLLM tie (661.1 against 661.9). With FP8 caches at 64 SGLang gave 1,118.2,
  8% under vLLM's single run, inside the spread. SGLang decodes faster and prefills more slowly, and its cache at 0.85
  is a third smaller than vLLM's at 0.95 (it ran out of memory at 0.9). With the engines this close, vLLM is kept, as
  the brief prefers on a tie, and the remaining variants were spent on it. SGLang with MTP was not measured.
- *Backend and chunk.* DeepEP low-latency forces 256 batched tokens per rank: prefill takes 128 steps per prompt and
  output falls 38%. A 4,096-token chunk costs 112,256 tokens of cache per rank, preempts at 64 and falls 19%.
- *Balancing.* With a per-rank limit of concurrency / 8, vLLM's data-parallel balancer left some ranks with waiting
  requests while others idled (concurrency 16: mean TTFT 27 s, almost all queueing). The limit was kept equal to
  concurrency / 8 throughout, as the protocol asks.

**Correctness of the selected lane** (raw responses in each run's `probes/`). Chat answers "The capital of France is
Paris, and the capital of Japan is Tokyo." With thinking on, the reasoning lands in `reasoning_content` and the
answer `9` in `content` (`glm47` parser). A `get_weather` request returns a structured `tool_calls` entry with
`{"city": "Paris"}` (`glm47` parser). A 40,384-token request (the 40,960-token context minus a 512-token answer
budget) recalls the code planted at its start, `7341`. The same probes passed on every lane they were run on: the
BF16 and FP8 DP lanes, MTP, and SGLang with an FP8 cache.

**Accuracy gate for lossy lanes, declared before any was run.** An FP8 KV cache (or any other lossy option) may be
selected only if, on the same server shape, GSM8K 5-shot strict-match over the first 400 test questions (lm-eval
0.4.12, `/v1/completions`, seed 0, greedy) scores no more than 0.03 below the BF16-cache lane (about two standard
errors of the difference), and the 40,384-token needle still returns the right code.

| Lane | KV cache | GSM8K strict-match (400) | Needle at 40,384 tokens | Result |
| --- | --- | ---: | --- | --- |
| `vllm-dp8ep` | BF16 | 0.950 ± 0.011 | `7341` | reference |
| `vllm-dp8ep-fp8kv` | FP8 (`fp8_ds_mla`) | 0.940 ± 0.012 | `7341` | passes (-0.010) |
| `vllm-dp8ep-fp8kv-mtp3`, final run's server | FP8 (`fp8_ds_mla`) | 0.9475 ± 0.011 | `7341` | passes (-0.0025) |

**The selected configuration on the datacenter rows** of `prompts/onboard-model/benchmark.md` (run
`2026-10-23_01-54-21`; the admission limit follows each row, concurrency / 8 per rank, at least 1; all rows 0 failed).

| Input / output | Concurrency | Requests | Output tok/s | Total tok/s | Req/s | Mean TTFT | Mean TPOT | MTP tokens per step |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 / 4096 | 1 | 8, three repeats | 147.4 / 142.0 / 157.1 | 294.9 / 284.1 / 314.1 | 0.04 | 0.61-0.66 s | 6.2-6.9 ms | 3.56-3.93 |
| 4096 / 4096 | 16 | 128 | 1,238.2 | 2,476.4 | 0.30 | 5.0 s | 11.2 ms | 3.78 |
| 4096 / 4096 | 32 | 256 | 1,578.9 | 3,157.7 | 0.39 | 5.1 s | 18.1 ms | 3.78 |
| 8192 / 256 | 16 | 80 | 254.1 | 8,385.7 | 0.99 | 4.1 s | 44.9 ms | 3.39 |
| 1024 / 1024 | 64 | 320 | 2,299.5 | 4,599.0 | 2.25 | 1.5 s | 24.6 ms | 3.56 |
| 256 / 256 | 256 | 1,280 | 3,961.2 | 7,922.4 | 15.47 | 1.2 s | 55.1 ms | 3.20 |

These rows are a profile of the selected configuration, not a search: no other lane ran them. The 256/256 row at
1,280 requests and the 8192/256 row at 80 finish in under 90 s each, so their figures include more ramp-up and tail
than the longer rows. Shorter outputs make MTP's acceptance fall (3.20 to 3.39 tokens per step against 3.78).

**Limitations.** Random-token prompts make MTP look better than natural text does (92% against 77.5% acceptance).
With three waves per run the measurement includes ramp-up and tail; single runs vary by up to 16%. The context is
qualified at 40,960 tokens, as the workload needs; the native 1,048,576 cannot fit one data-parallel rank (395,968
tokens with the FP8 cache). Accuracy was checked on GSM8K only, at short context. Not measured: TP8 with EP (same
cache ceiling as TP8), dual-batch overlap, DeepEP high-throughput, SGLang with MTP, and a cache-reuse lane.

**Raw runs**, all in `results_h200x8.tar.gz` (node addresses replaced with `node`): `2026-10-22_18-57-28` (DP8 at
0.9, did not start; its records stop at `deploy`/`staging` because the health wait was stopped by hand),
`2026-10-22_19-15-05` (vLLM DP8 BF16 ladder, probes, GSM8K), `2026-10-22_20-28-33` (SGLang), `2026-10-22_20-51-44`
(TP8), `2026-10-22_21-08-57` (FP8-cache ladder, probes, GSM8K, metrics poll; the 128 rung's record stops at
`benchmark`), `2026-10-22_22-10-25` (MTP), `2026-10-22_22-31-11` (SGLang FP8 cache at 0.9, failed),
`2026-10-22_22-38-23` (FP8 cache at 80), `2026-10-22_23-05-57` (SGLang FP8 cache at 0.85, probes),
`2026-10-22_23-28-16` (DeepEP low-latency; the 96 rung's record stops at `deploy`), `2026-10-22_23-54-39` (4,096-token
chunk), `2026-10-23_00-19-23` (final run, probes, GSM8K and MTP acceptance counters) and `2026-10-23_01-54-21`
(datacenter rows).
