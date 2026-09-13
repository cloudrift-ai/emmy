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

`golden/v100_sm70.yaml` carries **114 recorded rows over 15 targets** of decoder layer 0 — the Gated DeltaNet
archetype, 48 of the model's 64 layers. 101 of those rows are measured; the other 13 are the routing seeds that name
a kernel set and take their price from the rows they list. They were recorded on **one V100 SXM3 32GB**, not on the
SXM2 16GB pair the serving lane above ran on. Both are sm_70 and the kernels are the same, but the card is a
different SKU, so read the golden as compiler evidence for Volta rather than as a measurement of that deploy.

**The golden does not qualify serving.** Every serving figure above is the stock 1Cat-vLLM fork, which reads none of
these kernels. What the golden establishes is that this checkpoint's dominant decoder path compiles, runs correctly
and is mostly faster than eager PyTorch on Volta.

Fourteen targets are the export-unrolled delta rule and one is the `in_proj_qkv` GPTQ linear with its whole int4
decode cone fused in. Against eager, at the shapes recorded:

| target | eager us | emmy us | vs eager |
| --- | ---: | ---: | ---: |
| `k_slice_unsqueeze_reduce_d90fea` | 3,842 | 316 | 12.14x |
| `k_linear_reduce_5542a0` (`in_proj_qkv`) | 25,014 | 2,306 | 10.84x |
| `k_slice_unsqueeze_reduce_006570` | 3,876 | 536 | 7.23x |
| `k_slice_unsqueeze_reduce_cc2c8e` | 3,924 | 647 | 6.07x |
| `k_slice_unsqueeze_reduce_d4b52b` | 3,973 | 834 | 4.77x |
| `k_slice_unsqueeze_reduce_287e6f` | 4,032 | 997 | 4.04x |
| `k_slice_unsqueeze_reduce_d126a6` | 4,084 | 1,652 | 2.47x |
| `k_slice_unsqueeze_reduce_99707e` | 4,205 | 2,052 | 2.05x |
| `k_slice_unsqueeze_reduce_7e169c` | 4,265 | 2,560 | 1.67x |
| `k_slice_unsqueeze_reduce_124b26` | 4,338 | 3,113 | 1.39x |
| `k_slice_unsqueeze_reduce_0d1c72` | 4,421 | 3,780 | 1.17x |
| `k_slice_unsqueeze_reduce_6169c0` | 4,554 | 4,570 | 1.00x |
| `k_slice_unsqueeze_reduce_955202` | 4,627 | 5,454 | 0.85x |
| `k_slice_unsqueeze_reduce_1faf06` | 3,260 | 5,977 | 0.55x |
| `k_matmul_reduce_b686cf` | 18 | 35 | 0.51x |

**The three chunks that lose are upstream of scheduling.** Eager sits flat between 3,260 and 4,627 us across the
whole family while Emmy climbs monotonically with chunk index, because the Gated DeltaNet reference loops over chunks
in Python: `torch.export` unrolls it, so unrolled chunk *k* carries O(*k*) work where eager's batched form amortizes
every chunk into one flat cost. The cut is already the best partition available of the wrong amount of work, and no
pin, tile, fold or placement removes work the traced program contains.

**Every row is recorded behind a `PLACE` cut**, because the greedy elects one fused kernel at grid 1 — a single CTA
for the whole reduce, 648,716 us on `006570` against the cut's 536. The cut was always offered and always cheaper; it
was never picked, for a pricing reason diagnosed and fixed separately. A recorded row outranks the prior, which is
what makes this file the thing that keeps a deploy off the grid-1 pick.

**What the golden does not cover.** Layer 3's full attention (16 of 64 layers) is traced but carries no row: its
fused nest opens a serial cone that does not finish compiling. `embed_tokens`, the final norm, `lm_head` and the two
MTP programs are not traced. Of layer 0's 118 kernels, 15 are here — the ones that dominated the layer.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-GPTQ-Int4/serving --ssh USER@HOST
```
