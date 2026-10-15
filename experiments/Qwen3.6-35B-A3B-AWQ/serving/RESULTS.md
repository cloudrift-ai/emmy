# Qwen3.6-35B-A3B-AWQ serving experiment

Shared serving-protocol root for QuantTrio/Qwen3.6-35B-A3B-AWQ on Volta V100 hardware: the 1Cat/vLLM sm70
fork serves the AWQ Qwen3.5-MoE (qwen3_5_moe) text checkpoint with stock attention; no Emmy lane applies (the
AWQ quantization has no Emmy serving path). One measured platform so far: the 1x V100 SXM3 32GB single-replica
profile.

## Protocol

- Engine: 1Cat/vLLM sm70 image, TP=1, `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100`, fp16 compute over AWQ 4-bit
  weights (quant_method="awq", group 128, zero_point, gemm); layers 0, visual, linear_attn/self_attn,
  shared_expert, mlp.gate and MTP stay unquantized FP16.
- Server: `--max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser qwen3_coder
  --reasoning-parser qwen3`, `--default-chat-template-kwargs {"preserve_thinking": true}`,
  gpu_memory_utilization 0.88.
- Client: `vllm bench serve` (random dataset, seed 0), greedy, forced output length, no prefix-cache reuse,
  one row per platform; 2 warmups by deploy smoke + engine warmup before the recorded run.
- Capability probes (chat/completions and completions against the same deployment): context, tool calling,
  reasoning capture, coherence.

## NVIDIA Tesla V100 SXM3 32GB x1

Measured 2026-10-11 (run ID `20261011T101558Z`, single recorded row, one repeat; variability not measured).

| Item | Value |
| --- | --- |
| Repo revision | `bb7ee02e` (Emmy 0.3.22) |
| Model revision (HF) | `119886a1072372348f73ef0df2d801cdcc0f455b` |
| Image | `cloudriftai/1cat-vllm-sm70:1.0.0` (only public tag of the registry) |
| GPU / driver | Tesla V100-SXM3-32GB, 32768 MiB, driver 580.178.04, CUDA 12.9.86, sm_70 |
| Host | Ubuntu 24.04.1, x86_64, 4 vCPU, 83 GiB RAM |
| Context served | 32,768 (KV pool 38,016 tokens at gpu_mem_util 0.88) |
| KV headroom | 4.20x max 32K concurrency (server-reported); weights 22.3 GB, KV 3.46 GiB |

Workload: 32 prompts, 1,000 input / 1,000 forced output tokens (random-1000 corpus, `random_range_ratio` 0),
concurrency 4, greedy, 1,500 s steady-state window. Recorded row: `v100x1_5835c0e10702`.

| Metric | Value |
| --- | --- |
| Completed / failed | 32 / 0 |
| Output tok/s | 212.7 (peak 256.0) |
| Total tok/s | 425.5 |
| req/s | 0.21 |
| TTFT mean / median / P99 (ms) | 2635 / 664 / 16785 |
| TPOT mean / median / P99 (ms) | 16.18 / 16.14 / 16.39 |
| ITL mean / median / P99 (ms) | 16.18 / 16.13 / 18.91 |

Interpretation: decode is compute-bound at ~16.2 ms/token with an 87 tok/s single-stream ceiling, consistent
with the 35B-AWQ weights moving through one 900 GB/s card at 3B active experts per token; the bench's 212.7
output tok/s is the 4-concurrent sustained point (5.2x the single-stream baseline). TTFT is prefill-dominated
and queue-sensitive: median 664 ms at the first admitted wave, P99 16.8 s under the saturated four-way queue
(a 1,000-token prefill is ~16 s of single-stream prefill time, so the P99 is queueing plus prefill, not an
anomaly). Concurrency 4 was chosen as the server's reported max-concurrency point (4.20x at 32K) minus margin;
the consumer matrix's higher-concurrency rows (8/64) and the 256/256 short-turn row cannot run cleanly inside a
38K-token KV pool at the 8,448 total-token floor, and the 4096/4096 rows exceed the served context, so they
do not inform this deployment and are documented as inapplicable.

Capability probes (same deployment, greedy):

| Probe | Result |
| --- | --- |
| Coherence | "The capital of France is" -> " Paris, a city renowned for its rich history..." (completions); smoke "2+2" -> "4" |
| Context | 11,016 input tokens (30K cap) answered coherently in 2.5 s wall, finish stop; 32K single-request fits the 38,016-token KV pool per server report |
| Tool calling | `get_weather` prompt with `preserve_thinking` template returned structured `tool_calls` (finish `tool_calls`); budget-limited prompts (max_tokens <= 512) instead emit thinking prose and never reach the tool format - raise the completion budget or disable thinking for tool traffic |
| Reasoning | `reasoning` field populated when the chat template opens the think block (`enable_thinking: true`); without it the reasoning lands in `content` |
| Multimodal | checkpoint is vision-capable (vision_config present); qualified text-only, `input_modalities` unset |

Limitations: one row and one repeat, so no repeat-variability evidence; peak concurrent requests reached 8
against max 4 (oversampling of the random corpus); the served 32K context is a cap, the checkpoint's native
`max_position_embeddings` is 262,144; the image predates 1Cat v1.2.x (0.15.x-0.21.x vLLM base) and no newer
tag is published in this registry.

Raw evidence: `results_v100x1.tar.gz` -> `2026-10-11_10-15-58/` (benchmark logs, server log, system-only
experiment records). Reproduce:

```bash
emmy bench --filter "deploy.gpu=NVIDIA Tesla V100 SXM3 32GB" \
  --ssh <target> experiments/Qwen3.6-35B-A3B-AWQ/serving
```
