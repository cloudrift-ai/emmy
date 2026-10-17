# Step-5-Preview discovery

## Assessment on 2026-10-17

Onboarding shell, heat 90. No prior note existed; this is the baseline creation for it.

stepfun-ai/Step-5-Preview is StepFun's Step-5 flagship open release: a 604B-total / 27B-active MoE vision-language
model with 1M context (verified from config: step5_vl, 92 text layers, 352 routed experts top-8, 64 attention heads,
https://huggingface.co/stepfun-ai/Step-5-Preview). It is one of this run's strongest current serving signals:
OpenRouter's stepfun/step-5-preview route was #2 by volume at the 2026-10-16 cut (17.1T/wk, +430% week; 20.3T/30d,
new entrant) and the HF repository exists (created 2026-09-20). That rides the StepFun agentic/1M-context wave.
Two caveats the record should carry: the repository is FP8 (about 565B F8 + 40B BF16) and its license reads
"StepFun License 1.0" (not a standard open license), both of which onboarding should settle; and the node's own HF
download counter is low (~572 at the 2026-10-17 check) because the serving route, not the repo, is where the volume
is.

Serving context for the record: the recipe matrix is 8x NVIDIA H200 141GB or 8x NVIDIA B200 (see recipe.yaml). Not
yet onboarded or benchmarked.

Keep this note stable until onboarding settles the license and serving footprint, or a larger shift in the Step-5
wave appears.
