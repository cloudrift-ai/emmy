# MiniMax-M3 discovery

## Assessment on 2026-10-09

Onboarding shell, heat 82. No prior note existed; this is the baseline creation for it.

MiniMaxAI/MiniMax-M3 is a new 427B-total / 23B-active multimodal open MoE with 1M-token context and a claimed SWE-bench of 80.5 — the biggest new open wave this cycle. It is Hugging Face verified: roughly 205K downloads and 1,566 likes under a `minimax-community` license
(https://huggingface.co/MiniMaxAI/MiniMax-M3). This run's bounded OpenRouter and LMArena captures did not surface a route or arena row for the base M3 (not visible, not confirmed absent), and no high-engagement thread surfaced, so the high shell heat rests on the verified
config, license, and download momentum rather than a measured serving volume.

Serving context for the record: the recipe matrix is 16x H100, 16x H200, or 16x B200 — a full-node deployment. Not yet onboarded or benchmarked.

Uncertainty: the `minimax-community` license and the 16-GPU footprint are onboarding constraints to settle; the exact served checkpoint (base vs Flash) is owned by the `onboard-model` skill.
