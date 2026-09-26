# Qwen3.8-27B GPTQ Int4 on two V100 SXM2 16GB

Qualified 2026-09-05 against repository revision `01588556cef50c202e5370927e6788f039550801`.

This is the 4-bit Qwen3.8 lane for Volta. It exists because the Qwen3.8-27B wave is almost entirely
`compressed-tensors` 4-bit builds, whose kernels gate at "Min capability: 75" and refuse to run on sm_70. The
checkpoint served here is real GPTQ (`quant_method="gptq"`, 4-bit, `group_size=128`, `desc_act=false`, `sym=true`),
prepared and tested by its author on a V100.

## What was measured

| Item | Value |
| --- | --- |
| Model | `Max73333/Qwen3.8-27B-GPTQ-Int4-V100@d5a18cc1477e301e50d3fe4167fbf76db9337edc` |
| GPUs | 2 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.126.20 |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP2, context 65,536, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 795.82 s |
| Output token throughput | 40.21 tok/s |
| Total token throughput | 80.42 tok/s |
| Peak output token throughput | 44.00 tok/s |
| Median TTFT | 2,884.09 ms |
| Mean / P99 TTFT | 2,427.91 / 2,923.30 ms |
| Median TPOT | 96.76 ms |
| Mean / P99 TPOT | 97.14 / 99.27 ms |
| Median inter-token latency | 96.24 ms |

Startup on this platform is not free: weights load in 8.6 s, but `torch.compile` takes 105.0 s and CUDA graph
capture another 110.0 s, for 317.9 s from container start to health.

## Capability checks

All four protocol gates were exercised against the deployed server, not inferred from startup:

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Tokyo` for a capital-city question |
| Tool calling | Pass — returns a structured `tool_calls` entry, `get_weather{"city": "Paris"}` |
| Reasoning separation | Pass — the engine's `reasoning` field is populated and `content` holds only the answer |
| Context fill | Pass — a planted marker was retrieved from a 60,295-token prompt, 92% of the advertised 65,536 |

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
- **Per-stream decode is slow.** A 96.76 ms median TPOT is roughly 10 tokens/s per stream; the 40.21 tok/s figure is
  aggregate across four. The host exposes no NVLink — `nvidia-smi topo` reports PHB between every pair — so the TP2
  all-reduce crosses PCIe on every layer.
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

`golden/v100_sm70.json` carries **48 recorded rows over 46 targets** of decoder layer 0 — the Gated DeltaNet
archetype, 48 of the model's 64 layers. **36 beat eager PyTorch, 2 are at parity, 8 lose.** Recorded on **one V100
SXM3 32GB**, not on the SXM2 16GB pair the serving lane above ran on: both are sm_70 and the kernels are the same,
but the card is a different SKU, so read the golden as compiler evidence for Volta rather than as a measurement of
that deploy. **The golden does not qualify serving** — every serving figure above is the stock 1Cat-vLLM fork, which
reads none of these kernels.

Layer 0 traces to 99 distinct kernels. The 46 recorded are what one GPU-pair session could measure and verify; the
coverage is a subset by time, not by selection, and it deliberately spans both ends of the ranking so the file
records what loses as well as what wins.

**The largest wins.** The export-unrolled delta rule dominates, and the two GPTQ projections carry the most absolute
time:

| target | eager us | emmy us | vs eager |
| --- | ---: | ---: | ---: |
| `k_slice_unsqueeze_reduce_6bb9a7` | 2,630 | 5 | 525.25x |
| `k_slice_unsqueeze_reduce_55cc08` | 2,641 | 14 | 189.85x |
| `k_reshape_pointwise_a69add` | 24,469 | 193 | 126.84x |
| `k_slice_unsqueeze_reduce_cd0f4d` | 2,648 | 34 | 78.51x |
| 14 x `k_copy_*_base_pointwise` | ~1,085 | ~16 | ~68x |
| `k_linear_pointwise_fa224d` (`in_proj_qkv`) | 25,020 | 1,392 | **17.98x** |
| `k_linear_pointwise_944f3d` | 15,229 | 1,011 | **15.07x** |

Both projections carry **two rows**: the schedule above and the greedy the compiler picks unaided, which is 66x and
47x slower. The greedy takes `WORK=w16x2` with 1024-thread blocks — one wave of very fat blocks on an 80-SM card —
where `w8x1` at 12-25% occupancy wins. Occupancy is not the signal; the single wave is. Both rows are measured, and
a strict-evidence compile ranks them fastest-first.

**What still loses, and why each one does.** These are four different kinds of fact, not one:

| target | eager us | emmy us | vs eager | why |
| --- | ---: | ---: | ---: | --- |
| `k_slice_unsqueeze_reduce_c47f85` | 3,230 | 4,068 | 0.79x | the O(*k*) tail — upstream of scheduling |
| `k_slice_unsqueeze_reduce_334002` | 3,314 | 4,767 | 0.70x | the O(*k*) tail — upstream of scheduling |
| `k_matmul_reduce_b686cf` | 18 | 35 | 0.53x | unprobed; 17 us |
| `k_matmul_pointwise_227f1b` | 54 | 197 | 0.27x | 0.30x in all three Qwen slots — a shared pricing item |
| `k_cumsum_reduce` | 4 | 29 | 0.14x | no parallel scan tier exists in the tree |
| `k_reshape_3457df` | 2 | 30 | 0.06x | eager's reshape is a VIEW; Emmy materializes |
| `k_linear_pointwise_796083` | 33 | 591 | 0.06x | `in_proj_z`'s decode cone does not fuse into its linear |
| `k_transpose_272034` | 3 | 101 | 0.03x | eager's transpose is a VIEW; Emmy materializes |

`k_mul_85_pointwise` (1.01x) and `k_slice_unsqueeze_reduce_492362` (0.96x) are recorded as parity.

The two **view** rows are not tuning gaps: torch gets a transpose or a reshape by re-striding, at no cost, and Emmy
writes the tensor out. The question there is why the graph materializes them at all, and no pin addresses it. The
**scan** is a missing schedule family: `ScanOp` lifts to a serial-prefix loop, `_reduction_domain` returns one plan
and `split_pending` refuses, so no cross-CTA fork is even offered — only a new family would move it. The
**unfused decode cone** is a fusion asymmetry: `in_proj_qkv` fuses its whole int4 cone into its linear and
`in_proj_z` does not, so the second materializes the dequantized weight into global memory as its own kernel. And
the **O(*k*) tail** is the trace shape: `torch.export` unrolls the Gated DeltaNet chunk loop, so unrolled chunk *k*
carries O(*k*) work where eager's batched form amortizes flat. Across the 30 recorded chunks Emmy climbs from 5 us
to 4,767 us while eager moves only 2,630 to 3,314 — the curve is the evidence, and no pin removes work the traced
program contains.

**Five targets carry no row at all**, for three different reasons. `k_matmul_264a4c`, `k_matmul_reduce_08ec89` and
`k_matmul_reduce_00fcbf` hang: 60 s per launch on iteration 0, nothing measured.
`k_linear_matmul_mean_reduce_f10138` never reaches the GPU — over forty minutes in the compiler's kernel-set
pricing, which resolves every cut arm by nested resolution before any launch. And `k_matmul_reduce_c9784e` is a
defect rather than a gap: measured with the watchdog raised it runs **5,182,147 us against 3,717 us eager** at grid
12 on an 80-SM card, returns half its output as zeros, and disagrees with itself between runs. It is filed for its
own fix.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-GPTQ-Int4/serving --ssh USER@HOST
```
