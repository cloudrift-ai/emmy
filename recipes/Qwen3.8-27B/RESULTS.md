# Qwen3.8-27B at FP16 on eight V100 SXM2 16GB

## 2026-10-01 verification

Verification run against repository revision `7c317658681415c73a929493bb9a803151486db5`, on the previously
qualified revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The recipe is unchanged in substance: same image,
same flags, same serving shape. This run re-measured the lane, re-ran the capability checks, and re-confirmed
the fit and Emmy eligibility positions.

### What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPUs | 8 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.178.04 (upgraded from 580.126.20) |
| Host | `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM |
| Interconnect | `nvidia-smi topo` reports NV1/NV2 between intra-node GPU pairs, PHB between cross-node pairs — the platform's all-reduce still crosses at least one PCIe hop per 8-way collective |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP8, FP16 (`--dtype half`), context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 837.32 s |
| Output token throughput | 38.22 tok/s |
| Total token throughput | 76.43 tok/s |
| Peak output token throughput | 44.00 tok/s |
| Median TTFT | 702.14 ms |
| Mean / P99 TTFT | 672.61 / 902.57 ms |
| Median TPOT | 104.90 ms |
| Mean / P99 TPOT | 104.09 / 108.41 ms |
| Median ITL | 107.86 ms |

Deploy from container create to health took 131.5 s; benchmark wall time 888.4 s. The previous run took 545.9 s to
deploy and 1,631 s to benchmark. Both improved substantially on this host.

Measured KV pool: 279,171 tokens, 1.06x maximum concurrency at full context (previously 293,427 / 1.12x).

### Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Paris` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}` |
| Reasoning separation | Pass — `reasoning` field populated (~255 chars), `content` holds the worked answer `1161` for `27*43` |
| Context fill | Pass at 84-token prompt — a planted marker was retrieved (wall 0.8 s). A 60k-context probe was not re-run this cycle due to time budget; see the 2026-09-05 limitation below |

### Delta versus the 2026-09-05 qualification

Output throughput doubled (19.62 → 38.22 tok/s); TTFT dropped from a median of 54,249 ms to 702 ms; TPOT dropped
from 148 ms to 105 ms. The host is materially different — the previous run recorded PHB for every GPU pair; this
host reports NV1/NV2 for intra-node pairs, which removes the dominant all-reduce bottleneck at the same time the
driver was upgraded. The workload, model, image, and serving shape are identical; the delta is the platform.
The run remains over the 20-minute per-variant cap — 837 s against a 1,200 s target — and is retained because all
32 requests succeeded. This is still a batch and background configuration, not an interactive one.

### Limitations

- **The 60k context probe is stale.** This cycle re-checked coherence at 84 tokens but did not re-run the 60,295-token
  retrieval, so the end-to-end context ceiling for this host has to be inherited from the 2026-09-05 run until it is
  re-run. The allocated 262,144 window remains memory-backed (KV pool of 279,171 tokens, 1.06x concurrency).
- **Only the Volta platform is qualified.** The discovery shell also proposed RTX PRO 6000 Blackwell Max-Q and
  H200 141GB; both remain unmeasured, not unsuitable.
- **The interconnect topology is a hybrid.** `nvidia-smi topo` reports NV links between some GPU pairs and PHB
  between others. The 8-way all-reduce still has at least one PCIe hop per step, which limits the throughput
  gain from the NV topology even though it is real.

### Emmy eligibility

Still **ineligible**, first failing gate unchanged: `emmy/serving/twins.py` refuses to trace the Gated DeltaNet
layers of this checkpoint (`blocks whose token mixer is not attention … have no serving program yet`), which is
the gate `emmy serve` must pass to substitute an Emmy-compiled kernel for the vLLM attention seam. The compiler's
lowering half is demonstrated — sibling V100 goldens for quantized Qwen3.8-27B (`recipes/Qwen3.8-27B-AWQ-INT4/`,
`-GPTQ-Int4/`, `-FP8/`, `-EXL3/`) contain `v100_sm70.json` measured on this card — but no serving runner deploys
the GDN recurrence, so there is no Emmy recipe and no Emmy lane in the numbers above.

## 2026-09-05 initial qualification

Qualified 2026-09-05 against repository revision `01588556cef50c202e5370927e6788f039550801`.

This is the official Qwen3.8-27B checkpoint served on Volta. It is the only one of the three Qwen3.8 Volta lanes with
no quantization kernel anywhere in the path — Volta has no bfloat16, so the BF16 checkpoint is served as FP16 — which
makes it the cleanest evidence that the architecture itself runs on sm_70.

### What was measured

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

### Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Tokyo` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}` |
| Reasoning separation | Pass — the `reasoning` field is populated and `content` holds only the answer |
| Context fill | Pass at 60,295 tokens — a planted marker was retrieved. See the caveat below. |

As on the other Volta lanes, reasoning requires `chat_template_kwargs: {"enable_thinking": true}`; Qwen3.8 defaults
thinking off and otherwise reasons inline in `content`.

### Fit

55.6 GB of BF16 weights give a min-to-serve near 72.3 GB against 8 x 16,384 MiB = 137.4 GB of platform capacity. Four
cards hold only 68.7 GB and cannot take the weights, so eight is the smallest admissible platform on this fleet, not
a throughput preference. TP8 divides cleanly: 24 attention heads / 8 = 3, 48 linear-attention value heads / 8 = 6,
16 linear-attention key heads / 8 = 2; the 4 key/value heads are replicated because 8 % 4 == 0.

### Limitations

- **The advertised 262,144 context is memory-backed but not validated end to end.** The engine genuinely allocates
  for it — the measured KV pool held 293,427 tokens, 1.12x concurrency at full context — and this is the only one of
  the three Volta lanes that needed no context reduction to start. But a prompt near the full window did not finish
  prefilling within 20 minutes and was abandoned; retrieval was validated at 60,295 tokens, which completed quickly.
  Prefill cost grows faster than linearly here, so on this hardware the practical ceiling is **prefill time, not
  memory**. Treat 262,144 as the allocated window and roughly 60k as the size with end-to-end evidence behind it.
- **This was the slowest of the three Volta lanes.** 19.62 tok/s output and a 148 ms median TPOT (about 6.8 tokens/s
  per stream) were roughly half the 4-bit lane's throughput, which is expected: FP16 moves four times the weight bytes
  per token that Int4 does, and the host recorded then exposed no NVLink — `nvidia-smi topo` reported PHB between
  every pair — so an eight-way all-reduce crossed PCIe on every layer. A median TTFT of 54 s at concurrency 4 made
  this a batch or background configuration, not an interactive one.
- **The engine image cannot build its own Volta Gated DeltaNet kernel.** 48 of the 64 layers are Gated DeltaNet, so
  that path is unavoidable. `flash_qla_sm70_gdn_strided` is a JIT torch extension whose build fails at
  `fatal error: cusparse.h: No such file or directory`, because the runtime layer ships the sources without the CUDA
  development headers. Selecting the Triton GDN *prefill* backend is not enough — the GDN decode path reaches for
  flash_qla independently — so the recipe sets `VLLM_SM70_GDN_DECODE_FLASHQLA=0`. The Volta-native kernel is
  therefore being skipped, and a purpose-built sm_70 image that prebuilds it is the most likely source of a speed-up.
- **The image tag is misleading by name.** It is tagged for DeepSeek-V4-Flash, but the wheel is the whole 1Cat-vLLM
  fork, selected because it is the first published sm_70 image new enough for Qwen3.8: its `qwen3_5` config defaults
  `partial_rotary_factor` to 0.25 and its Gated DeltaNet layer accepts `output_gate_type: "swish"`. The older
  `cloudriftai/1cat-vllm-sm70:1.0.0` knows neither and would silently apply full rotary to a model declaring a
  quarter.

### Emmy

Not evaluated on this cycle. The compiler's lowering half is demonstrated by the sibling V100 goldens for
quantized Qwen3.8-27B, but no serving runner deploys the GDN recurrence, so there is no Emmy lane in the numbers
and no Emmy comparison.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B/serving --ssh USER@HOST
```
