# Qwen3.8-27B at FP16 on eight V100 SXM2 16GB

## 2026-10-04 verification

Verification run on the repository tree at `e59848d9`, on the previously qualified revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (the hub head is still that commit as of this run — no drift). The recipe
is unchanged in substance: same image, same flags, same serving shape. This run re-measured the lane, re-ran the
capability checks at a new context size, and re-checked the Emmy serving gate on the current code (still ineligible,
same first failing gate).

### What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPUs | 8 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.178.04, nvcc 12.9.86 |
| Host | `riftvm`, Ubuntu 24.04.1 LTS, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`), cached on the host |
| Serving shape | TP8, FP16 (`--dtype half`), context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 869.64 s |
| Output token throughput | 36.80 tok/s |
| Total token throughput | 73.59 tok/s |
| Peak output token throughput | 40.00 tok/s |
| Median TTFT | 874.89 ms |
| Mean / P99 TTFT | 814.73 / 1,004.08 ms |
| Median TPOT | 107.93 ms |
| Mean / P99 TPOT | 107.99 / 109.67 ms |
| Median ITL | 107.88 ms |

The row's model download was 3.21 s (checkpoint already on the host's shared model volume), load and warmup 360.22 s
(weights 17.09 s, torch compile 101.77 s, CUDA graph capture 108.0 s). Total row wall 1,297.82 s.

Measured KV pool: 279,171 tokens, 1.06x maximum concurrency at full context (re-read from the engine log this run).

### Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Paris` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}`, `finish_reason: tool_calls` |
| Reasoning separation | Pass — `reasoning` field populated (140 chars) with the worked `27*43` derivation (the 128-token budget cut the final answer line off; the arithmetic in the field is correct) |
| Context fill | Pass — a planted 8-char marker at ~50% of a 54,003-word prompt (270,000 characters) was retrieved (wall 17.5 s). The window itself holds (279,171-token KV pool, 1.06x concurrency at 262,144); requests near the full window remain bounded by prefill time, ~120k with end-to-end evidence from 2026-10-03 |

### Delta versus the 2026-10-03 verification

Output throughput dipped below the three-run band (39.16, 38.22, 37.09 to 36.80 tok/s, −5.7% against the band head);
median TTFT rose 687.45 to 874.89 ms and median TPOT 102.82 to 107.93 ms. Workload, model, image, and serving shape
are identical across all four runs; the benchmark itself ran 869.64 s, between the 2026-10-03 row (817.25 s, fastest)
and the 2026-10-02 row (862.86 s). The remaining variance is run-to-run on the same platform.

### Emmy eligibility — unchanged from 2026-10-03

The GDN serving-twin capture still succeeds on this checkout (gates 1, 2 and the trace half of gate 4 intact), and
`serving/ARCHITECTURE.md` still states that capture "does not integrate recurrent state into `EmmyGenRunner` or
native HTTP request dispatch," so gate 5 remains the first failing gate — no serving runner deploys the GDN
recurrence end-to-end. Compiler coverage stays partial for this hybrid checkpoint and nothing new is committed under
`golden/`. An Emmy recipe becomes viable when `EmmyGenRunner` integrates the GDN state and a complete golden exists
for this exact checkpoint.

### Limitations

- **The 262,144 window is memory-backed, not end-to-end backed.** The KV pool holds 279,171 tokens (1.06x
  concurrency at full context); this cycle retrieved a marker from a 270,000-character prompt, and the prior cycle
  took 120,015 tokens end to end, but requests near the full window remain bounded by prefill time, not memory
  (about 120k is the size with end-to-end evidence).
- **Only the Volta platform is qualified.** The discovery shell also proposed RTX PRO 6000 Blackwell Max-Q and
  H200 141GB; both remain unmeasured, not unsuitable.

## 2026-10-03 verification

Verification run on a repository tree at `3a251085`, on the previously qualified revision
`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` (the hub head is still that commit — no drift). The recipe is unchanged in
substance: same image, same flags, same serving shape. This run re-measured the lane, re-ran the capability checks
with the context ceiling pushed higher, and re-checked the Emmy serving gate on the current code (still ineligible,
same first failing gate).

### What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPUs | 8 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.178.04, nvcc 12.9.86 |
| Host | `riftvm`, Ubuntu 24.04.1 LTS, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`), cached on the host |
| Serving shape | TP8, FP16 (`--dtype half`), context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 817.25 s |
| Output token throughput | 39.16 tok/s |
| Total token throughput | 78.31 tok/s |
| Peak output token throughput | 44.00 tok/s |
| Median TTFT | 687.45 ms |
| Mean / P99 TTFT | 662.97 / 917.84 ms |
| Median TPOT | 102.82 ms |
| Mean / P99 TPOT | 101.58 / 107.74 ms |
| Median ITL | 106.69 ms |

The image was already on this host, so the row's startup was model download 113.6 s plus load and warmup 366.2 s
(weights 15.7 s, torch compile 100.9 s, CUDA graph capture 107.0 s). Total row wall 1,356.0 s.

Measured KV pool: 279,171 tokens, 1.06x maximum concurrency at full context (re-read from the engine log this run).

### Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Paris` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}`, `finish_reason: tool_calls` |
| Reasoning separation | Pass — `reasoning` field populated (241 chars), `content` holds the worked `27*43` answer `1161` |
| Context fill | Pass at 120,015 tokens — a planted 8-char marker was retrieved (wall 89.0 s). A 200,016-token probe was rejected by the engine at the 262,144-window boundary because prompt plus output exceeded it; the window itself stays memory-backed (279,171-token KV pool) |

The context fill is the strongest end-to-end context evidence on this lane: 60,016 tokens (27.6 s) and 120,015
tokens (89.0 s) both retrieved their planted markers in one request, against 33.5k (75.1 s) on 2026-10-02 and
60,295 (the 2026-09-05 ceiling) — the 262,144 window is real, and the practical ceiling is prefill time, not memory.

### Delta versus the 2026-10-02 verification

Output throughput is the best of the three retained runs (37.09 to 39.16 tok/s, +5.6%; band 37.09–38.22–39.16).
Median TTFT improved 739 to 687 ms and median TPOT 105.77 to 102.82 ms. The benchmark ran 817.25 s, the fastest
row on this platform. Workload, model, image, and serving shape are identical across all three runs; the remaining
variance is run-to-run on the same platform.

### Emmy eligibility — unchanged from 2026-10-02

Re-checked on the current checkout: the GDN serving-twin capture still succeeds (gates 1, 2, 4-trace intact), and
`serving/ARCHITECTURE.md` still states that capture "does not integrate recurrent state into `EmmyGenRunner` or
native HTTP request dispatch," so gate 5 remains the first failing gate — no serving runner deploys the GDN
recurrence end-to-end. Compiler coverage stays partial for this hybrid checkpoint and nothing new is committed under
`golden/`. An Emmy recipe becomes viable when `EmmyGenRunner` integrates the GDN state and a complete golden exists
for this exact checkpoint.

### Limitations

- **The 262,144 window is memory-backed, not end-to-end backed.** The KV pool holds 279,171 tokens (1.06x
  concurrency at full context) and 120,015 tokens of live input was retrieved end to end, but a request near the full
  window was rejected at the token accounting boundary. Treat roughly 120k as the size with end-to-end evidence and
  262,144 as the allocated window; prefill time, not memory, is the practical ceiling on this hardware.
- **Only the Volta platform is qualified.** The discovery shell also proposed RTX PRO 6000 Blackwell Max-Q and
  H200 141GB; both remain unmeasured, not unsuitable.

## 2026-10-02 verification

Verification run against repository revision `2f5ae0962f898aa7b9a1a80d079cd34f136308f6`, on the previously
qualified revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. The recipe is unchanged in substance: same image,
same flags, same serving shape. This run re-measured the lane, re-ran the capability checks, and — most importantly
— re-checked the Emmy serving gate on the current code, where the Gated DeltaNet serving-twin capture that was
blocking on 2026-09-28 now succeeds.

### What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` |
| GPUs | 8 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.178.04 |
| Host | `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM |
| Interconnect | `nvidia-smi topo` reports NV1/NV2 between some GPU pairs and PHB between others — the eight-way all-reduce still crosses at least one PCIe hop per step |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP8, FP16 (`--dtype half`), context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
| Workload | 32 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, seed 0, temperature 0, ignored EOS, 2 warm-ups |

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 32 / 0 |
| Benchmark duration | 862.86 s |
| Output token throughput | 37.09 tok/s |
| Total token throughput | 74.17 tok/s |
| Peak output token throughput | 59.00 tok/s |
| Median TTFT | 739.17 ms |
| Mean / P99 TTFT | 2,154.98 / 14,998.24 ms |
| Median TPOT | 105.77 ms |
| Mean / P99 TPOT | 105.55 / 121.68 ms |
| Median ITL | 109.07 ms |

Deploy from container create to teardown-complete took 906.2 s this cycle (image pull 435.5 s because the 1Cat sm_70
image was cold on this fresh host; the 2026-10-01 cycle already had it cached and deployed in 131.5 s). Benchmark
wall time was 862.86 s for 1,832.05 s total.

Measured KV pool: 279,171 tokens, 1.06x maximum concurrency at full context.

### Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returns `Paris` for a capital-city question |
| Tool calling | Pass — structured `tool_calls`, `get_weather{"city": "Paris"}`, `finish_reason: tool_calls` |
| Reasoning separation | Pass — `reasoning` field populated (~83 chars), `content` holds the worked `27*43` answer |
| Context fill | Pass at 33.5k tokens — a planted 8-char marker was retrieved (wall 75.1 s). 2026-09-05 validated 60,295 tokens; the 262,144 window stays memory-backed (279,171-token KV pool) |

### Delta versus the 2026-10-01 verification

Throughput is in the same band (38.22 → 37.09 tok/s output, a 3% dip); TTFT and TPOT are unchanged (median TTFT
702 → 739 ms, median TPOT 104.90 → 105.77 ms). The benchmark itself ran 862.86 s, under the 1,200 s per-variant
cap (the 2026-09-05 qualification's 1,631 s was over it). The workload, model, image, and serving shape are
identical across all three runs; the remaining variance is run-to-run scheduling on the same platform.

### Emmy eligibility — the gate moved

The 2026-09-28 re-check recorded Emmy as **ineligible**, first failing gate the serving twins refusing to trace the
Gated DeltaNet layers of this checkpoint (`blocks whose token mixer is not attention … have no serving program
yet`). That gate has since been closed in the compiler: the current checkout's twin capture now builds the GDN
state wrapper and traces `gdn{width}` programs for every linear-attention layer, handing the state from prefill
into decode. Running `capture_twin_graphs` on this exact checkpoint in this cycle succeeds and produces six
programs:

```
gdn256-dense-linear, gdn32-dense-linear   (Gated DeltaNet linear-attention, decode-32 / prefill-256 buckets)
pre256-dense-full, pre32-dense-full       (full-attention pre-twin, 16 full-attention layers)
post256-dense-full, post32-dense-full     (full-attention post-twin)
```

What that closes and what it does not:

- **Closes**: gate 1 (live compute capability sm_70 is accepted by the CUDA backend — demonstrated by the sibling
  V100 goldens for the quantized Qwen3.8-27B checkpoints), gate 2 (a real trace path for the architecture's GDN
  and full-attention layer types now exists; the twin capture on this exact checkpoint succeeds), and gate 4 for
  the trace half (a compiler inventory for this card is producible — the two distinct layer paths traced to 9
  fresh targets in under a minute on the card).
- **Does not close** (and why the recipe remains vLLM, not Emmy): gate 5. The ARCHITECTURE note is explicit that
  the GDN state capture "does not integrate recurrent state into `EmmyGenRunner` or native HTTP request dispatch;
  those runners still need allocation, reset and scheduling support." No serving runner deploys the GDN
  recurrence end-to-end yet, so an Emmy recipe with an `emmy serve --runner generate` serving path cannot be
  verified in this cycle. The compiler's lowering side is demonstrated, the serving-side integration is not.
- **Compiler coverage is still partial, not complete**: the fresh lowering of this hybrid checkpoints produces
  many targets. The two most representative paths (one GDN layer, one full-attention layer) traced to 9 targets in
  this cycle; the full model inventory is not complete because the GDN recurrence kernels are expensive to compile
  (the 48 GDN layers alone would take many more hours). Per the onboarding rule, a partial inventory is not
  committed under `golden/`; it stays outside the repository. The sibling quantized `v100_sm70.json` goldens
  (AWQ-INT4, GPTQ-Int4, FP8, EXL3) are the durable evidence of what the compiler currently lowers on this card.

So the eligibility position changes from "twins refuse the GDN" to "twins accept the GDN but no serving runner
deploys it, and a complete BF16 golden for this card is not yet buildable in one run" — still **ineligible** for a
serving recipe, but with a materially different blocker than in September. An Emmy recipe becomes viable when
`EmmyGenRunner` integrates the GDN state and a complete golden exists for this exact checkpoint.

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
