# GLM-5.3 discovery

## Assessment on 2026-10-05

Onboarding shell, heat 85. Newest Z.ai flagship, verified: the Hugging Face zai-org checkpoint exists (created
2026-10-01/02, MIT); LMArena shows glm-5.3-max at 1478 (overall #27, #2 open) on the Oct 2, 2026 board; community
quant demand is live (an uncensored EXL3 quant already trending at 191 with 220 likes in ~4 days). Two caveats:
no non-Flash z-ai/glm-5.3 route surfaced on OpenRouter in this run's capture (only glm-5.3-flash, top-5 open), and the
Z.ai author share dipped 14% WoW to 6.6% — the family is strong but not monotonically rising week-to-week.
Not yet onboarded or benchmarked; the 8x B200 matrix is an assumption for onboarding.

## Assessment on 2026-10-16

Onboarding shell, heat 85 (held). The wave behind this node re-verified and re-measured on 2026-10-16 across two
independent sources, reversing the prior "not monotonically rising" reading: OpenRouter (data through 2026-10-15) now
shows a dedicated z-ai/glm-5.3 route at 4.83T/wk (+46% week) and z-ai/glm-5.3-flash at 12.9T/wk (+19% month), with the
Z.ai author share up +107% week-over-week — a non-Flash base route has surfaced, resolving the earlier caveat; and
Hugging Face re-verified the zai-org/GLM-5.3 checkpoint at ~1,803,969 downloads, 2,231 likes, with an active
quant/abliteration ecosystem forming (https://huggingface.co/zai-org/GLM-5.3). Two corrections for onboarding: the
served checkpoint lives under zai-org, and at ~753B total in FP8 the 8x B200 matrix (~1,437 GiB) does not hold the
weights — a full-node 16x B200 or equivalent capacity is the admissible deployment, not 8x. Not yet onboarded or
benchmarked.
