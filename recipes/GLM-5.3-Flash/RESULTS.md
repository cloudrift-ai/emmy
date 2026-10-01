# GLM-5.3-Flash — recommended serving recipe

**Qualified on 2x AMD Instinct MI350X (gfx950, ROCm 7.2), TP=2.**
Date measured: 2026-09-23. Repo revision at run: main @ `agents/model-discovery-...` working tree.
Model revision: `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (~321B MoE, 18B active, native FP8).

## Engine and image

- **Engine:** vLLM (ROCm), pinned by immutable digest
  `vllm/vllm-openai-rocm@sha256:d5da8f963f0571a171f6817cc108bc2371a39528a11aeba5a00cb78177fcb4e5`
  (vLLM 0.3.1.dev311, PyTorch 2.12.0+git6bbd260, ROCm 7.2.53211). No stable per-version ROCm tag
  exists for vLLM >= 0.29 (the model's first supported version); `nightly` pinned by sha is the
  only reproducible anchor.
- **Why vLLM:** the `glm5_next` architecture (hybrid KDA linear attention + sparse MLA, MoE,
  hyper-connections) is vLLM-native and is the only engine with a documented AMD/ROCm path for this
  family (vLLM PR #53906 sparse-MLA, #55239 ROCm MTP). MTP was not enabled (unverified on 2-GPU MI350X).
- **Launch knobs:** `VLLM_ROCM_USE_AITER=1`, `--enable-auto-tool-choice
  --tool-call-parser glm47 --reasoning-parser glm47 --chat-template-content-format=string`,
  `gpu_memory_utilization=0.9`, `context_length=65536`, `max_concurrent_requests=32`.

## Serving configuration

- **GPU:** 2x AMD Instinct MI350X (288 GB HBM3e each, 576 GB total), TP=2.
- **Context:** 65536 (capacity choice; native context is 1,048,576, KV pool holds ~7.0M tokens).
- **Precision:** the checkpoint's native mixed FP8 (weights) + BF16 (attention/embeddings/HC
  layers). This is the stored representation — no dequantization or repacking performed. `--kv-cache-dtype
  fp8` was not set, so KV is BF16 (safer default; a quality measurement would be owed before
  enabling it).
- **Fit arithmetic:** 321B total params x 1 byte (FP8) = ~321 GB weights (~306 GiB on disk);
  min-to-serve x1.3 = ~397 GiB, which fits the 576 GB platform with ~178 GB for KV cache/activations.

## Validated protocol

- **Reasoning:** separated into the engine `reasoning` field (e.g. 2+2 -> content `4`, 39 tokens
  reasoning). Thinking is always on; `reasoning_effort` only changes depth.
- **Tool calling:** returns structured `tool_calls` (validated with a `get_weather` function;
  arguments parsed correctly).
- **Long context:** a 30K-input request returns 200 OK with coherent output at ~550 tok/s prefill;
  4096/4096 requests serve to completion at 65536 context with no OOM or preemption.
- **Multimodal:** the checkpoint is multimodal (image+video vision tower), but `multimodal_mode: auto`
  resolved to the text-serving path here — the vision tower is present in the config but was not
  benched. Image/video input is not validated in this run.

## Measured serving (recommended lane)

Selected point: **concurrency 1, 4096 input / 4096 output** (the recommended single-stream
operating point). From the complete 3-lane measurement (run 2026-09-23_12-00-32, 92 requests,
0 failures):

| Lane | Concurrency | Out tok/s | Total tok/s | Median TTFT | Median TPOT | P99 TPOT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| **4096/4096 (selected)** | 1 | **59.90** | 119.81 | **314 ms** | **16.5 ms** | 16.6 ms |
| 4096/4096 (batch) | 32 | 893.64 | 1787.29 | 7656 ms | 33.2 ms | 35.5 ms |
| 8192/256 (long-in) | 1 | 51.81 | 1709.76 | 544 ms | 16.4 ms | 16.6 ms |

The single-stream median TPOT of 16.5 ms is the intrinsic decode rate for the 18B-active MoE; the
32-concurrency lane demonstrates the throughput ceiling (894 out tok/s, peak 1088) at roughly 2x
per-request TPOT.

## Emmy eligibility and compiler coverage

**Emmy is INELIGIBLE for this checkpoint on this platform.** First failed gate: the live compute
capability is **AMD/gfx950 (ROCm)**, not an NVIDIA `sm_*` capability accepted by Emmy's CUDA
backend (`ptxas`/sm_75+). The supplied host has no CUDA toolkit and no nvcc, so kernel build and
measurement cannot run here. The architecture is additionally not traceable by Emmy's current
checkout: zero references to `glm5_next`/`Glm5Next`; `find_text_decoder` requires a `rotary_emb`
attribute, which GLM-5.3-Flash does not have (`qk_rope_head_dim=0`, no RoPE); the MHC/hyper-connection
seam uses `hc_attN_fn`/`hc_ffn_fn` module names that differ from the `attn_hc` pattern Emmy detects.
Uncovered seams: KDA linear-attention state recurrence, sparse-MLA with q/k lora compression,
hyper-connection (hc_mult=4) layout, and the MTP draft layer. The 288-expert MoE block itself is
expressible in Emmy's `build_moe_split_wrapper`, but the attention side is not. **No model golden is
committed** (trace would be incomplete, and the platform cannot build/measure regardless).

**Emmy release image:** not applicable (ineligible), so no `EMMY_FAST_MATH` policy applies.

## Reproduction

```bash
emmy bench experiments/GLM-5.3-Flash/serving --ssh USER@HOST --ssh-key PATH
```

The experiment recipe keeps the 3-lane matrix; each row deploys and benches the pinned image. The
selected lane is the concurrency-1 4096/4096 row (filter `--filter "deploy.gpu=*MI350X*"` and the
`mc1` variants when narrowing).

## Limitations

- ROCm engine is a nightly pinned by sha; the underlying engine (vLLM 0.3.1.dev) is newer than the
  first documented 0.29.0 support, so behavior may shift between onboarding runs.
- MTP speculative decoding not enabled (unverified on 2-GPU MI350X); single-stream decode could be
  faster with MTP if it were validated.
- Multimodal input (image/video) not benched.
- `--kv-cache-dtype fp8` not enabled (would reduce KV for more context but owes a quality check).
- Single measurement per lane; a multi-rep sweep would bound variance.
- The NVIDIA B200 x4 matrix row remains untested (outside this platform's scope).
