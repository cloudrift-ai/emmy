# GLM-5.2 discovery

## Assessment on 2026-10-05

Onboarding shell, heat 25. Earlier verified read stands: BF16 753B total — unservable in the fleet (largest node is
16 GPUs, which cannot hold BF16 weights, so the 16x B200 matrix is arithmetic, not a rentable deployment). Community
attention has shifted to the GLM-5.3 releases: LMArena showed glm-5.2-max at 1476 (#32, MIT) on the Oct 2, 2026 board
while glm-5.3-max held 1478 (#27, #2 open) — strong as a sibling, but the 5.2 checkpoint itself is not the served
open model. Not yet onboarded or benchmarked.
