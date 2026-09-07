# Qwen3.8-27B at FP16 on eight V100 SXM2 16GB

Qualified 2026-09-05 against repository revision `01588556cef50c202e5370927e6788f039550801`.

This is the official Qwen3.8-27B checkpoint served on Volta. It is the only one of the three Qwen3.8 Volta lanes with
no quantization kernel anywhere in the path — Volta has no bfloat16, so the BF16 checkpoint is served as FP16 — which
makes it the cleanest evidence that the architecture itself runs on sm_70.

## What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPUs | 8 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.126.20 |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP8, context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Backends | FLASH_ATTN_V100 attention, Triton Gated DeltaNet prefill |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 1,631.13 s |
| Output token throughput | 19.62 tok/s |
| Total token throughput | 39.24 tok/s |
| Peak output token throughput | 29.00 tok/s |
| Median TTFT | 54,249.55 ms |
| Mean / P99 TTFT | 49,388.50 / 61,649.02 ms |
| Median TPOT | 147.98 ms |
| Mean / P99 TPOT | 154.65 / 178.23 ms |
| Median inter-token latency | 145.17 ms |

Deploy from container create to health took 545.9 s. The benchmark row ran 1,631 s, over the 20-minute per-variant
cap; it is reported rather than discarded because all 32 requests succeeded, and the overrun is the measurement, not
an error.

## Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Tokyo` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}` |
| Reasoning separation | Pass — the `reasoning` field is populated and `content` holds only the answer |
| Context fill | Pass at 60,295 tokens — a planted marker was retrieved. See the caveat below. |

As on the other Volta lanes, reasoning requires `chat_template_kwargs: {"enable_thinking": true}`; Qwen3.8 defaults
thinking off and otherwise reasons inline in `content`.

## Fit

55.6 GB of BF16 weights give a min-to-serve near 72.3 GB against 8 x 16,384 MiB = 137.4 GB of platform capacity. Four
cards hold only 68.7 GB and cannot take the weights, so eight is the smallest admissible platform on this fleet, not
a throughput preference. TP8 divides cleanly: 24 attention heads / 8 = 3, 48 linear-attention value heads / 8 = 6,
16 linear-attention key heads / 8 = 2; the 4 key/value heads are replicated because 8 % 4 == 0.

## Limitations

- **The advertised 262,144 context is memory-backed but not validated end to end.** The engine genuinely allocates
  for it — the measured KV pool holds 293,427 tokens, 1.12x concurrency at full context — and this is the only one of
  the three Volta lanes that needed no context reduction to start. But a prompt near the full window did not finish
  prefilling within 20 minutes and was abandoned; retrieval is validated at 60,295 tokens, which completed quickly.
  Prefill cost grows faster than linearly here, so on this hardware the practical ceiling is **prefill time, not
  memory**. Treat 262,144 as the allocated window and roughly 60k as the size with end-to-end evidence behind it.
- **This is the slowest of the three Volta lanes.** 19.62 tok/s output and a 148 ms median TPOT (about 6.8 tokens/s
  per stream) are roughly half the 4-bit lane's throughput, which is expected: FP16 moves four times the weight bytes
  per token that Int4 does, and the host exposes no NVLink — `nvidia-smi topo` reports PHB between every pair — so an
  eight-way all-reduce crosses PCIe on every layer. A median TTFT of 54 s at concurrency 4 makes this a batch or
  background configuration, not an interactive one.
- **The engine image cannot build its own Volta Gated DeltaNet kernel.** 48 of the 64 layers are Gated DeltaNet, so
  that path is unavoidable. `flash_qla_sm70_gdn_strided` is a JIT torch extension whose build fails at
  `fatal error: cusparse.h: No such file or directory`, because the runtime layer ships the sources without the CUDA
  development headers. Selecting the Triton GDN *prefill* backend is not enough — the GDN decode path reaches for
  flash_qla independently — so the recipe sets `VLLM_SM70_GDN_DECODE_FLASHQLA=0`. The Volta-native kernel is
  therefore being skipped, and a purpose-built sm_70 image that prebuilds it is the most likely source of a speed-up.
- **Only the Volta platform was qualified.** The discovery shell this recipe replaces proposed one RTX PRO 6000
  Blackwell Max-Q and one H200 141GB. Both remain unmeasured, not unsuitable — the model fits either far more
  comfortably than it fits eight 16 GB cards, and the numbers here must not be read across to them.
- **The image tag is misleading by name.** It is tagged for DeepSeek-V4-Flash, but the wheel is the whole 1Cat-vLLM
  fork, selected because it is the first published sm_70 image new enough for Qwen3.8: its `qwen3_5` config defaults
  `partial_rotary_factor` to 0.25 and its Gated DeltaNet layer accepts `output_gate_type: "swish"`. The older
  `cloudriftai/1cat-vllm-sm70:1.0.0` knows neither and would silently apply full rotary to a model declaring a
  quarter.

## Emmy

Not evaluated. This run was scoped to serving qualification only, so no compiler coverage was traced, no golden was
produced, and Emmy eligibility is **unevaluated** rather than established. There is no Emmy lane in the recipe and no
Emmy comparison in the numbers above — every figure here is the stock 1Cat-vLLM fork serving lane.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B/serving --ssh USER@HOST
```
