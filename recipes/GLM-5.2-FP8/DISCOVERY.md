# GLM-5.2-FP8 discovery

## Assessment on 2026-10-23

Best-effort (emmy-blocked), first note. zai-org/GLM-5.2-FP8 is the official block-FP8
checkpoint of the GLM-5.2 generation: ~750B total / ~40B active, ~756 GB of weights, so a
single replica fits 8x H200 141GB (the recipe's matrix) or 8x B200 - unlike the raw BF16
GLM-5.2 checkpoint, which no rentable platform shape can hold. The recipe serves it with
vLLM data-parallel attention (8 ranks, experts split across ranks, FP8 KV cache, MTP with 3
draft tokens) at 40,960 context (see recipe.yaml and its inline comments); the highest
sustained throughput measured at 32,768-in / 4,096-out prompts is held in
`experiments/GLM-5.2-FP8/serving` per the recipe's own comment.

The `emmy-blocked` tag and heat 0 reflect a serving-side qualification that the Emmy runtime
does not yet fully pass for this checkpoint (distinct from the Qwen3.8-27B Gated-DeltaNet
blocker noted in the sibling GLM-5.3 notes); exact blocking mechanism should be confirmed
against `experiments/GLM-5.2-FP8/` at refresh time rather than inferred here. No fresh
trending, OpenRouter, or arena signal for this specific FP8 node surfaced in this run's
bounded checks (bounded, not confirmed absent); the broader GLM line's attention is currently
on GLM-5.3(-Flash), so this 5.2-FP8 node is a distinct, larger-capability deployment (8x
H200) rather than a demand leader.

Keep this note stable until the emmy-blocked reason is settled or a substantial signal changes
the standing.
