# Ornith-1.5-35B-A3B-FP8 serving qualification

**Question.** How should the FP8 Ornith-1.5-35B-A3B checkpoint be served on one datacenter card when it is only
allowed a fraction of that card, and which concurrency is the right operating point?

**Protocol.** One vLLM v0.30.0 server per row (`--gpu-memory-utilization 0.75`, `--max-model-len 262144`,
text-only, qwen3_xml tool parser, qwen3 reasoning parser), the server's admission limit set to the row's client
concurrency, and the datacenter workload matrix from `prompts/onboard-model/benchmark.md`: random prompts with
seed 0, temperature 0, ignore-EOS, two warm-up requests, unique prompts (no prefix reuse). All rows share the
recipe in this directory; `--filter` selects one.

## NVIDIA H100 80GB x1

Run `20260929T192754Z`, started 2026-09-29T19:27:54Z, last row completed 2026-09-29T20:22Z. Six of six rows
succeeded, no failed requests, no preemption logged in any row.

| Purpose | Input/output tokens | Concurrency (client = server) | Requests | Output tok/s | Median TTFT | P99 TTFT | Median TPOT | P99 TPOT | Duration | Peak KV use |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single stream, repeat 1 | 4096/4096 | 1 (server 32) | 8 | 170.8 | 238 ms | 413 ms | 5.49 ms | 7.79 ms | 192 s | 1.0% |
| Single stream, repeat 2 | 4096/4096 | 1 (server 32) | 8 | 193.1 | 235 ms | 250 ms | 5.05 ms | 5.46 ms | 170 s | 1.0% |
| Single stream, repeat 3 | 4096/4096 | 1 (server 32) | 8 | 196.3 | 234 ms | 242 ms | 5.04 ms | 5.05 ms | 167 s | 1.0% |
| Moderate batch | 4096/4096 | 16 | 128 | 1,531.2 | 727 ms | 1,295 ms | 10.29 ms | 10.83 ms | 342 s | 15.5% |
| Higher batch | 4096/4096 | 32 | 256 | 2,365.7 | 694 ms | 2,390 ms | 13.34 ms | 13.63 ms | 443 s | 31.0% |
| Long input, short output | 8192/256 | 16 | 80 | 842.1 | 937 ms | 2,306 ms | 15.52 ms | 17.59 ms | 24 s | 15.5% |
| Balanced shared service | 1024/1024 | 64 | 320 | 3,825.7 | 369 ms | 1,273 ms | 16.54 ms | 16.74 ms | 86 s | 28.4% |
| Saturated short turns | 256/256 | 256 | 1,280 | 7,624.5 | 410 ms | 2,120 ms | 31.37 ms | 32.14 ms | 43 s | 94.2% |

TTFT at concurrency above 1 is end-to-end and includes admission queueing: the two concurrency-16 rows briefly
held 3 and 6 waiting requests; every other row admitted its whole client load. "Peak KV use" is the highest
`GPU KV cache usage` the engine logged during the row; the pool is 22.7 GiB, 1,157,545 tokens, 4.42 full-context
sequences (the hybrid model's linear-attention state pages live in the same pool, which is why 256 short
sequences fill 94% of it).

**Repeat variation.** The three single-stream repeats differ by the first one only: 170.8 tok/s and a P99 TPOT
of 7.79 ms against 193-196 tok/s and 5.05-5.46 ms afterwards. The server logs a just-in-time compile of the
FlashInfer linear-attention prefill on its first requests, which the two warm-up requests did not fully cover.
Repeats 2 and 3 agree within 1.6% on throughput and 0.2% on median TPOT. Every other row ran once.

**Comparisons.** Going from 16 to 32 concurrent 4K/4K streams raised output throughput 1.55x for a 30% TPOT
increase; 64 streams of 1K/1K reached 3.8K tok/s at 16.5 ms per token; 256 streams of short turns doubled that
throughput again but also doubled per-token latency to 31 ms and ran the pool at 94%. Single-stream decode is
5.0 ms per token (about 195 tok/s) with a 235 ms first token on a 4K prompt.

**Conclusion.** The 0.75 share holds the model with room to spare: 34.4 GiB of weights, a 22.7 GiB KV pool, and
about 20 GiB of the card untouched. Concurrency 64 is the recommended admission limit for this share: it keeps
per-token latency at or under 17 ms across the 1K-4K workloads while delivering 2.4-3.8K output tok/s, and it
stays far from the pool's edge that the 256-stream row reached. `recipes/Ornith-1.5-35B-A3B-FP8/recipe.yaml`
pins that configuration.

**Limitations.** One run per multi-stream row; the long-input row is short (24 s) and its TTFT is dominated by
prefill of 8K prompts at 16-way concurrency. No prefix-cache lane was measured, although prefix caching is on in
the engine by default; agent traffic with shared system prompts will see lower TTFT than this no-reuse matrix.
No comparison engine or Emmy lane was run: the Emmy serving image was not qualified for this checkpoint (see the
recipe's `RESULTS.md`).

**System.** GCP a3-highgpu-1g: 1x NVIDIA H100 80GB HBM3 (81,559 MiB, driver 580.173.02), Intel Xeon Platinum
8481C (26 logical CPUs), 230 GiB RAM, Ubuntu 24.04.5, kernel 7.0.0-1011-gcp, Docker 29.8.1, nvcc 12.9.41 on the
host. Engine image `vllm/vllm-openai:v0.30.0`, digest
`sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`, build fingerprint
`vllm-0.30.0-84b1640b`, FLASH_ATTN attention backend, compressed-tensors FP8 quantization,
`torch.compile` and CUDA graphs on. Model revision `fab11c26e2325a42f4b32da0249c819a0bade1b1`. Repository
revision `a98fd4f851be305a225ffce7bac2ad61b2892c8a`, clean.

**Status.** All rows `succeeded`. Archive: `results_h100x1.tar.gz`, root member `2026-09-29_19-27-54/`, holding
the six `*.experiment.yaml` records plus each row's `*.benchmark.log` and `*.server.log`, the run's
`benchmark.log` and `benchmark_h100_x_1.log`. Row ids: `2fa990f3f3a0` (single stream), `baec19fc359d` (16),
`150405da9de0` (32), `24c4ab4e4a24` (8K/256), `04a0751b1020` (64), `29baf62bed8d` (256).

Reproduce one row on a supplied host:

```bash
emmy bench experiments/Ornith-1.5-35B-A3B-FP8/serving --ssh user@host \
  --filter "engine.llm.max_concurrent_requests=64"
```
