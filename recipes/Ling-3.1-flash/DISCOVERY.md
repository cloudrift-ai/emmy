# Ling-3.1-flash discovery

## Assessment on 2026-10-17

Onboarding shell, heat 78. No prior note existed; this is the baseline creation for it.

inclusionAI/Ling-3.1-flash is inclusionAI's (Ant Group) new large flagship MoE: 572B total / 25B active,
hybrid-linear (63 KDA + 9 Gated-MLA), 1M context, MIT license; Hugging Face confirms an FP8 repository (about
570B FP8 + 1.5B BF16, https://huggingface.co/inclusionAI/Ling-3.1-flash). It is the successor to the Ling-3.0-flash
node and the Ling family's current serving wave (Ling-3.0-flash's 3.0 cut is the older generation). At the 2026-10-17
check the node was very fresh (created 2026-10-13, about 115 likes, about 1,276 30-day downloads) and it did not
surface in this run's OpenRouter top-10 or LMArena capture (not visible, not confirmed absent), so the shell heat
rests on the verified config, the new-first-party-lab signal, and the fleet-fitting 1M-context MoE slot rather than a
measured serving volume — it is the one newly verified large open MoE this cycle without a complete recipe.

Serving context for the record: the recipe matrix is 8x NVIDIA H200 141GB or 4x NVIDIA B200 (see recipe.yaml) —
sizing is owned by the fit subagents when refreshed. Not yet onboarded or benchmarked.

Keep this note stable until an independent OpenRouter or arena signal for the Ling-3.1 node appears, or the recipe is
onboarded.
