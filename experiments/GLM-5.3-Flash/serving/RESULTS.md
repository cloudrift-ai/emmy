# GLM-5.3-Flash serving experiment

## Protocol

Serving benchmark of `zai-org/GLM-5.3-Flash` (321B MoE, 18B active, native FP8 checkpoint,
~321B total params) on **2x AMD Instinct MI350X** (gfx950, ROCm 7.2, 288 GB HBM3e each, 576 GB total)
with tensor parallelism 2. Engine is vLLM (ROCm nightly image pinned by digest, vLLM 0.3.1.dev),
with `VLLM_ROCM_USE_AITER=1`, auto tool-choice and the `glm47` tool/reasoning parsers. Server context
length is 65536. All lanes use `random` seeded prompts with `--ignore-eos` off (server-side sampling),
no prefix-cache reuse. The model's native context is 1,048,576 tokens; the 65536 cap is a capacity
choice, not an architecture limit (the KV pool holds ~7.0M tokens).

The model revision is pinned: `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.

## AMD Instinct MI350X x2

- **Run timestamp:** 2026-09-23_12-00-32
- **Archive:** `results_mi350xx2.tar.gz`
- **Row status:** 3/3 succeeded (0 failed requests across all lanes)

### Measurements

| Lane | In/Out tokens | Concurrency | Requests | Out tok/s | Total tok/s | Peak out tok/s | Median TTFT (ms) | P99 TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) | Duration (s) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Single-stream long-context | 4096/4096 | 1 | 8 | 59.90 | 119.81 | 63.00 | 314.38 | 4200.59 | 16.51 | 16.56 | 547.02 |
| Moderate batch | 4096/4096 | 32 | 64 | 893.64 | 1787.29 | 1088.00 | 7655.76 | 29023.14 | 33.22 | 35.48 | 293.34 |
| Long-input short-output | 8192/256 | 1 | 20 | 51.81 | 1709.76 | 62.00 | 543.85 | 4002.65 | 16.42 | 16.58 | 98.82 |

### Observations

- **Decode is MoE-18B-active-rate-bound.** At concurrency 1 the model sustains ~60 output
  tok/s (median TPOT ~16.5 ms); adding 32 concurrent requests raises total output to 894 tok/s
  but roughly doubles per-request TPOT to ~33 ms, the expected trade for a model whose active
  compute is independent of batch until the attention/MLP batches fill.
- **Prefill scales well.** The 8192-input lane hits ~1700 total tok/s (prefill-dominated), and a
  single 65536-token request prefill was observed at ~550 tok/s in a separate probe, so long
  context is not a bottleneck on this platform.
- **The 32-concurrency lane is throughput-optimal but not latency-friendly.** Median TTFT reaches
  7.7 s (P99 29 s) because 64 long 4096/4096 requests arrive together. For a consumer single-user
  deployment the concurrency-1 lane is the right operating point; a short-turn high-throughput
  service can use concurrency 32 if the latency budget allows.
- **No OOM, no preemption, no failures** in any lane over 92 total requests. The 65536 context cap
  is comfortable at this scale.

### Selected recipe point

The recommended serving recipe is the concurrency-1 / 4096-in / 4096-out configuration: median
TTFT 314 ms, median TPOT 16.5 ms, 59.9 out tok/s single-stream, which matches the intent for a
single-user or interactive serving deployment of this model.

### System

- **GPU:** 2x AMD Instinct MI350X VF (0x75b0), gfx950, ROCm 7.2
- **Image:** `vllm/vllm-openai-rocm@sha256:d5da8f963f0571a171f6817cc108bc2371a39528a11aeba5a00cb78177fcb4e5`
  (vLLM 0.3.1.dev311, PyTorch 2.12.0+git6bbd260, ROCm 7.2.53211)
- **Deployment:** TP=2, `gpu_memory_utilization=0.9`, `context_length=65536`,
  `VLLM_ROCM_USE_AITER=1`, auto tool-choice, `glm47` parsers, `--chat-template-content-format=string`
- **Model revision:** `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`
- **KV cache:** 7,005,798 tokens (100.76 GiB per GPU), max concurrency at 65536 tokens = 106.9x

### Limitations

- Single measurement per lane (no repeats beyond what the request count provides); a multi-rep
  sweep would bound run-to-run variance.
- The ROCm nightly is pinned by sha (immutable), but the nightly tag itself moves, so a
  re-onboard picks up newer engine behavior. The pinned digest is the reproducibility anchor.
- Context beyond 65536 was not benched as a serving lane; the 1M native context is supported but
  capacity-bound at high concurrency and was validated for reachability, not throughput, at that
  length.
