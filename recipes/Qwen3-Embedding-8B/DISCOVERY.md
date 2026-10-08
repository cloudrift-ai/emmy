# Qwen3-Embedding-8B discovery

## Assessment on 2026-10-08

Best-effort (embed) recipe, heat 38. No prior note existed; this is the baseline creation for it.

The Qwen3-Embedding-8B node holds the established high-performance end of the Qwen embedding line: it serves on a
single RTX 5090 or 4090 at a pinned 4096-token context (the model natively supports 32k), compared side-by-side
against stock vLLM pooling in the recipe. It inherits family stability from the Qwen embedding recipes, but this
cycle's bounded sources surfaced no independent trending, OpenRouter, or arena signal for the 8B node specifically.

The one material current signal in the embedding space this week points AWAY from the Qwen line: google/embeddinggemma-2
(0.74B multimodal embedding, 1,066 likes, refreshed 2026-10-06, trending on the Hugging Face API at the 2026-10-08
check) is the fresh embedding release competing for the current-momentum slot. The Qwen line's own 0.6B node also did
not independently resurface as a top-trending item this cycle. That is why the current-momentum embed slot in the
maintained set sits with the 0.6B recipe while this 8B node stays best-effort — a stable, high-performance variant
without a direct attention breakout of its own.

Serving context for the decision: one `NVIDIA GeForce RTX 5090` or `NVIDIA GeForce RTX 4090`, `gpu_count: 1`, at 4096
context (see recipe.yaml). No new deployment was authored here; the onboarding sizing is owned by the fit subagents when
this recipe is later refreshed.
