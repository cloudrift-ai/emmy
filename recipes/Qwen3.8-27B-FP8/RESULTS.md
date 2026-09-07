# Qwen3.8-27B-FP8 on four V100 SXM2 16GB

Qualified 2026-09-05 against repository revision `01588556cef50c202e5370927e6788f039550801`.

The interesting property of this lane is that **Volta has no FP8 arithmetic at all**. The checkpoint is served
through the 1Cat-vLLM fork's TurboMind dequantization path, the same route the DeepSeek-V4-Flash lane uses on this
engine. Every number below is a property of that dequantization path, not of hardware FP8.

## What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| GPUs | 4 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.126.20 |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP4, context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Backends | FLASH_ATTN_V100 attention, Triton Gated DeltaNet prefill, TurboMind FP8 dequantization |
| Workload | 16 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 16 / 0 |
| Benchmark duration | 492.87 s |
| Output token throughput | 32.46 tok/s |
| Total token throughput | 64.93 tok/s |
| Peak output token throughput | 48.00 tok/s |
| Median TTFT | 23,072.16 ms |
| Mean / P99 TTFT | 18,972.64 / 23,124.08 ms |
| Median TPOT | 100.98 ms |
| Mean / P99 TPOT | 104.33 / 122.43 ms |
| Median inter-token latency | 92.94 ms |

Deploy from container create to health took 445.5 s.

## Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Tokyo` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}` |
| Reasoning separation | Pass — the `reasoning` field is populated and `content` holds only the answer |
| Context fill | Pass at 60,295 tokens — a planted marker was retrieved. See the caveat below. |

Reasoning requires `chat_template_kwargs: {"enable_thinking": true}`; Qwen3.8 defaults thinking off and otherwise
reasons inline in `content`.

## Fit

30.9 GB of FP8 weights give a min-to-serve near 40.2 GB against 4 x 16,384 MiB = 68.7 GB of platform capacity. Two
cards hold only 34.4 GB and cannot take the weights, so four is the floor. TP4 divides cleanly: 24 attention heads
/ 4 = 6, 4 key/value heads / 4 = 1.

## How it compares on this host

All three Qwen3.8 Volta lanes were measured on the same machine, same engine image, same 1,000/1,000 workload at
client concurrency 4. The Int4 and FP16 lanes ran 32 prompts; this one ran 16 to stay inside the 20-minute
per-variant cap, so **total throughput is not directly comparable** — the per-token latencies are.

| Lane | Cards | Weights | Output tok/s | Median TPOT | Median TTFT |
| --- | ---: | ---: | ---: | ---: | ---: |
| GPTQ Int4, TP2 | 2 | 19.6 GB | 40.21 | 96.76 ms | 2,884 ms |
| FP8, TP4 (this) | 4 | 30.9 GB | 32.46 | 100.98 ms | 23,072 ms |
| FP16, TP8 | 8 | 55.6 GB | 19.62 | 147.98 ms | 54,250 ms |

The ordering tracks bytes moved per weight — 0.5, 1, and 2 — and the widening tensor-parallel degree. Since the host
exposes no NVLink (`nvidia-smi topo` reports PHB between every pair), each doubling of the parallel degree adds
PCIe all-reduce traffic on every layer, which is why the widest lane is also the slowest.

## Limitations

- **The advertised 262,144 context is memory-backed but not validated end to end.** The measured KV pool holds
  292,125 tokens at that window, but retrieval is validated at 60,295 tokens. On the FP16 lane a near-full-window
  prompt did not finish prefilling within 20 minutes; prefill cost grows faster than linearly, so the practical
  ceiling on this hardware is prefill time rather than memory. Treat 262,144 as the allocated window.
- **A 23-second median TTFT** at concurrency 4 makes this a batch or background configuration, not an interactive
  one.
- **FP8 here is emulation.** Volta has no FP8 units; throughput reflects TurboMind dequantization and should not be
  read as an FP8 hardware result, nor compared against FP8 numbers from Hopper or Blackwell parts.
- **The engine image cannot build its own Volta Gated DeltaNet kernel.** 48 of the 64 layers are Gated DeltaNet.
  `flash_qla_sm70_gdn_strided` is a JIT torch extension whose build fails at `fatal error: cusparse.h: No such file
  or directory`, because the runtime layer ships the sources without the CUDA development headers. Selecting the
  Triton GDN *prefill* backend is not enough — the GDN decode path reaches for flash_qla independently — so the
  recipe sets `VLLM_SM70_GDN_DECODE_FLASHQLA=0`. The Volta-native kernel is therefore skipped, and a purpose-built
  sm_70 image that prebuilds it is the most likely source of a speed-up across all three lanes.
- **Only the Volta platform was qualified.** The discovery shell this recipe replaces proposed one RTX PRO 6000
  Blackwell Max-Q and one H200 141GB; both remain unmeasured, not unsuitable, and both have native FP8 hardware.

## Emmy

Not evaluated. This run was scoped to serving qualification only, so no compiler coverage was traced, no golden was
produced, and Emmy eligibility is **unevaluated** rather than established. There is no Emmy lane in the recipe and no
Emmy comparison in the numbers above — every figure here is the stock 1Cat-vLLM fork serving lane.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
