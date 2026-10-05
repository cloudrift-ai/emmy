# BAAI/bge-m3 — V100 SXM2 16GB Onboarding

## 2026-10-05 — Onboarding attempt (FAILED)

**Platform:** 1× NVIDIA Tesla V100 SXM2 16GB (sm_70, 16384 MiB, driver 580.178.04)
**Model revision:** `5617a9f` (BAAI/bge-m3, main branch)
**Repository revision:** `agents/model-discovery-37114388896`

### Fit gate — PASSED

| Quantity | Value |
|---|---|
| Total parameters | ~566M (1024 × 24 layers, XLM-RoBERTa) |
| dtype | fp32 (stored), served fp16 |
| Weight VRAM (fp16) | ~1.13 GB |
| Min-to-serve (×1.3) | ~1.47 GB |
| Platform capacity | 16384 MiB |

Weights fit with ample headroom. Context 8192 × 16 attention heads × 128 head-dim × fp16 = 25.7 MB per
sequence KV — negligible for a pooling model (no KV cache needed for prefill-only serving).

### First failed gate — vLLM image for sm_70 pooling

**Gate:** Serving (section 3)

Stock vLLM (`vllm/vllm-openai`) has dropped sm_70 from its CUDA build. The only in-repo image with sm_70
support is `cloudriftai/1cat-vllm-sm70`, a source-pinned 1Cat-vLLM Volta fork. Two problems:

1. **No pooling support verified.** The 1Cat-vLLM fork is generate/AWQ focused. Its published tags
   (`1.0.0`, `1.2.2-cloudrift`) have no bge-m3 pooling test. The fork's vLLM base version may not include
   `BgeM3EmbeddingModel` (added in vLLM v0.10.x, March 2026) or the RoBERTa CUDA-graph position-ids fix
   (PR #37884, March 2026).
2. **No prebuilt image for the source pin.** `docker/1cat-vllm-sm70/Dockerfile` builds
   `emmy-round2-laguna-fp8-sm70:96f26179bf28` on demand — this is a multi-step source build (torch 2.10.0
   cu128 + vLLM from source + flash-attention-v100 wheel + emmy runtime extension) that exceeds the time
   budget for this onboarding run.

### What would unblock it

- A published `cloudriftai/1cat-vllm-sm70` tag whose vLLM base includes `BgeM3EmbeddingModel` (v0.10+)
  and the RoBERTa position-ids fix (post-March 2026), verified on a V100 with `--runner pooling
  --enforce-eager --dtype half`.
- Alternatively, a stock vLLM image that still ships sm_70 kernels with bge-m3 pooling support.

### Emmy eligibility

**Ineligible** — gate 1 (compute capability). The V100 SXM2 16GB has sm_70. Stock vLLM has dropped sm_70
support. The 1Cat-vLLM fork provides sm_70 generate support but its pooling capability is unverified and
no prebuilt image exists for the in-repo source pin. Emmy's `EmmyEmbedModel` targets causal-trunk
embedding models (e.g. Qwen3-Embedding); bge-m3 is a bilateral XLM-RoBERTa encoder, not a causal-trunk
model, so the Emmy serving path has no runner for this architecture.

### Golden

No golden committed. sm_70 is accepted by Emmy's CUDA backend (see `experiments/Qwen3-0.6B/paged_cache/
golden/v100_sm70.json` for a working example), but the bilateral XLM-RoBERTa encoder has no Emmy trace
path in this checkout. A representative layer trace would be a compiler addition, not a qualification
measurement.

### Recipe

**File:** `recipes/bge-m3/recipe.yaml`
**Tags after:** `onboarding`, `untested`, `onboarding-failed`
**Change:** added `onboarding-failed` tag. No other changes.
