# Qwen3.8-27B GPTQ Int4 on two V100 SXM2 16GB

Qualified 2026-09-05; re-verified 2026-09-29 against repository revision
`044c84f11f016d7eec85a7d7f7e3cdb201344764` on the same checkpoint, image, and platform.

This is the 4-bit Qwen3.8 lane for Volta. It exists because the Qwen3.8-27B wave is almost entirely
`compressed-tensors` 4-bit builds, whose kernels gate at "Min capability: 75" and refuse to run on sm_70. The
checkpoint served here is real GPTQ (`quant_method="gptq"`, 4-bit, `group_size=128`, `desc_act=false`, `sym=true`),
prepared and tested by its author on a V100.

## What was measured

| Item | Value |
| --- | --- |
| Model | `Max73333/Qwen3.8-27B-GPTQ-Int4-V100@d5a18cc1477e301e50d3fe4167fbf76db9337edc` |
| GPUs | 2 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.178.04 |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP2, context 65,536, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 929.61 s |
| Output token throughput | 34.42 tok/s |
| Total token throughput | 68.85 tok/s |
| Peak output token throughput | 44.00 tok/s |
| Median TTFT | 11,817.77 ms |
| Mean / P99 TTFT | 10,799.34 / 13,241.68 ms |
| Median TPOT | 105.00 ms |
| Mean / P99 TPOT | 105.50 / 115.00 ms |
| Median inter-token latency | 103.08 ms |

Startup on this platform is not free: weights load in 9.25 s, `torch.compile` takes 119.94 s and CUDA graph
capture another 125.0 s, for 372.63 s from container start to health.

## Capability checks

All four protocol gates were exercised against the deployed server, not inferred from startup:

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns the correct one-word answer to a capital-city question |
| Tool calling | Pass — returns a structured `tool_calls` entry, `get_weather{"city": ...}` |
| Reasoning separation | Pass — the engine's `reasoning` field is populated and `content` holds only the answer |
| Context fill | Pass — a planted marker was retrieved from a prompt that fills the full 65,536-token KV window; the server enforces the ceiling per request (any request whose input plus requested output exceeds 65,536 is rejected) |

**Reasoning requires an explicit opt-in.** Qwen3.8 defaults thinking *off*, unlike Qwen3-0.6B. Without
`chat_template_kwargs: {"enable_thinking": true}` the model reasons inline in `content` and the `reasoning` field
stays empty, which is easy to misread as a broken reasoning parser. The parser is fine; the request has to ask.

## Fit

19.6 GB of GPTQ 4-bit weights, so min-to-serve is about 25.5 GB against 2 x 16,384 MiB = 34.4 GB of platform
capacity. One 16 GB card cannot hold the weights at all, which is what sets TP2 as the floor rather than a
throughput choice. TP2 divides cleanly: 24 attention heads / 2 = 12, 4 key/value heads / 2 = 2.

## Limitations

- **Context is 65,536, not the advertised 262,144.** Measured on this platform: 262,144 tokens need 8.16 GiB of KV
  against a 4.57 GiB pool, and 131,072 need 4.16 GiB against 3.06 GiB once the Triton Gated DeltaNet path takes its
  workspace. At 65,536 the pool holds 93,206 tokens, giving 1.42x concurrency at full context — four request slots
  queue rather than run fully parallel on long prompts.
- **Per-stream decode is slow.** A 105.00 ms median TPOT is roughly 9.5 tokens/s per stream; the 34.42 tok/s figure
  is aggregate across four. The host exposes no NVLink — the pair connects over PCIe — so the TP2 all-reduce crosses
  it on every layer.
- **The engine image cannot build its own Volta GDN kernel.** Qwen3.8-27B is 48 Gated DeltaNet layers out of 64, so
  that path is unavoidable. `flash_qla_sm70_gdn_strided` is a JIT torch extension and its build fails at
  `fatal error: cusparse.h: No such file or directory`, because the runtime layer ships the sources without the CUDA
  development headers. Selecting the Triton GDN *prefill* backend is not sufficient — the GDN decode path reaches for
  flash_qla independently — so the recipe also sets `VLLM_SM70_GDN_DECODE_FLASHQLA=0`. A purpose-built sm_70 image
  that prebuilds that extension would remove this workaround and is the most likely source of a speed-up, since the
  Volta-native kernel is the one being skipped.
- **The image tag is misleading by name.** It is tagged for DeepSeek-V4-Flash, but the wheel is the whole 1Cat-vLLM
  fork. It was selected because it is the first published sm_70 image new enough for Qwen3.8: its `qwen3_5` config
  defaults `partial_rotary_factor` to 0.25 and its Gated DeltaNet layer accepts `output_gate_type: "swish"`. The
  older `cloudriftai/1cat-vllm-sm70:1.0.0` knows neither, and would silently apply full rotary to a model that
  declares a quarter.
- **This checkpoint is a community quantization**, not a vendor release, and carries no upstream support guarantee.

## Emmy

`golden/v100_sm70.json` is the committed compiler evidence for this checkpoint on **the exact card this run used**:
2 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0. As of this verification the compiler team re-keyed and
compacted the file (kernel-set retargeting and dropping unmeasured rows), so it now carries a small set of fully
measured layer-0 targets — one representative of each structural kernel in the Gated DeltaNet archetype — rather than
the wider layer-0 sweep recorded in the 2026-09-05 version.

Re-verified against the current compiler on the requested platform in this run:

- `emmy golden check`: every stored target is the fresh lowering (the file is current at repository revision
  `044c84f1`).
- Row-by-row strict decode (GPU-free): all stored rows still equal an enumerated schedule, no red rows.
- Deployable O3 replay on the requested card: the golden's measured rows deploy under strict evidence. Every stored
  single-kernel target that measures matches its recorded reference within ~2%; the recorded fork rows (which this
  session does not re-measure) are correctly refused by strict evidence rather than fabricated.

| target | eager us | emmy us (this run) | vs eager |
| --- | ---: | ---: | ---: |
| `k_cumsum_reduce` | 102 | 37 | **2.8x faster** |
| `k_matmul_c42469` | 2,627 | 2,592 | 1.01x |
| `k_matmul_reduce_95eda6` | 7,198 | 6,139 | **1.17x faster** |
| `k_linear_matmul_mean_reduce` | ~9,516 | 9,516 | parity |

**The golden does not qualify serving.** Every serving figure above is the stock 1Cat-vLLM fork, which reads none of
these kernels. The golden is durable compiler evidence for Volta, not a measurement of the deploy.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-GPTQ-Int4/serving --ssh USER@HOST
```
