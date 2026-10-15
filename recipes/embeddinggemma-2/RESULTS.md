# google/embeddinggemma-2 results

## 2026-10-09 — onboarding failure: NVIDIA Tesla V100 SXM2 16GB, 1x

No engine produced a valid recipe on this platform. The failure is upstream model support, not hardware or capacity.

### Platform

| Item | Value |
| --- | --- |
| Hardware | 1x NVIDIA Tesla V100 SXM2 16GB (sm_70 Volta), driver 580.178.04, CUDA 580, Ubuntu 24.04 |
| Repository revision | `a138700e5` |
| Model revision | `914f7f89142e33e77833254d9c9b90c3cef7303b` |
| Checkpoint | 744,371,512 parameters, BF16 safetensors (`total_params` from HF safetensors metadata) |
| Architecture | `EmbeddingGemma2Model` (model_type `embedding_gemma2`), multimodal (text + vision + audio), transformers 5.18.0.dev0 |
| Native context | 8,192 tokens (text tower) |
| VRAM fit | 0.74 GB weights x 1.3 = 0.97 GB min-to-serve << 16,384 MiB — passes by a wide margin |

### First failed gate: engine support

The probe deployed on the supplied V100 node used `vllm/vllm-openai:v0.31.0` with `transformers==5.16.1` (the
first stable release that unvends the config mapping for this architecture family) and `--runner pooling`.
The probe image was a local FROM-extension of the pinned tag (`emmy-eg2-tf5161:v0.31.0-tf5161`), built and
then torn down on the node after the run.

vLLM's API server rejected the model at engine-config construction:

```
pydantic_core.ValidationError: 1 validation error for ModelConfig
  Value error, The checkpoint you are trying to load has model type `embedding_gemma2` but Transformers
  does not recognize this architecture. This could be because of an issue with the checkpoint, or because
  your version of Transformers is out of date.
```

Root cause, established from primary sources:

- `EmbeddingGemma2Model` landed in vLLM upstream via PR #60254 (commit `02b83919aa2e`), merged 2026-10-06, two
  days after v0.31.0 was built (2026-10-05). v0.31.0's tree does not contain `vllm/model_executor/models/
  embedding_gemma2.py`; the compare against v0.31.1rc0 puts v0.31.0 three commits behind it, with the support
  commit among those three. No vLLM stable release after v0.31.0 existed at run time.
- The checkpoint's config maps to `embedding_gemma2` / `gemma4_text` / `gemma4_vision` / `gemma4_audio`.
  `transformers==5.16.1` (verified inside the probe image) ships `gemma` through `gemma4_unified_assistant`
  under `transformers/models/` but not `embedding_gemma2`; the mapping only appears in the 5.17/5.18 line.
  vLLM's own vendored fallback was removed in v0.31.1rc0 (commit `2c99ee933382`), which is why a transformers
  upgrade beyond 5.16.1 against v0.31.0 is not a supported path.
- The 1Cat/vLLM sm70 fork in `docker/1cat-vllm-sm70/` is based on upstream vLLM at commit `644d8a7cd05e`
  (2026-07-19), months before the architecture landed. The published image tag `1.0.0` was retained for other
  checkpoints but cannot be assumed to know this model. The fork's newer source commit `d76126608` referenced
  by the `1cat-vllm-deepseek-v4-flash-0731` tag is not resolvable public, so no published cloudriftai/1Cat/vLLM
  image was verified to carry support; none was used.

The probe therefore reached engine construction but never loaded weights or served a request; context and
modality were never exercised, which is expected — the checkpoint cannot be registered by any available engine
image.

### What would unblock this

Any one of:

1. vLLM releases v0.31.1 stable or later with `EmbeddingGemma2Model` (already in `main`). A plain
   `vllm/vllm-openai` image from that release would need sm_70 kernels added by 1Cat/vLLM for this platform
   (Volta is not in stock vLLM's `CUDA_ARCH_LIST`).
2. A 1Cat/vLLM sm70 image rebuilt on a post-`02b83919aa2e` upstream base, with a transformers version that
   resolves `model_type embedding_gemma2` (5.17 or 5.18 line).
3. An out-of-tree vLLM model registration for `EmbeddingGemma2Model` (text tower + vision tower + pooling
   head), which is a new plugin, not a reuse of the existing `--hf-overrides {"architectures":["EmmyEmbedModel"]}`
   path — that class is a single-tower, attention-free text pooling shell and does not implement the multimodal
   fusion this checkpoint requires.

Emmy eligibility was not evaluated: the ineligibility is upstream model support, which applies to the whole
Emmy stack as well (the serving plugin is built on the same vLLM model registry). No compiler golden was
committed — the architecture has no trace or runner path in this checkout, and the checkpoint's BF16 /
multimodal shape has no matching loader, so a golden would be partial at best. The golden's eventual form
depends on which unblock lands first.

The recipe is tagged `onboarding-failed`; nightly selection skips it until an explicit retry succeeds.
