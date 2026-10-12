# clef discovery

## Assessment on 2026-10-12

Onboarding shell, heat 75. Cloudflare/clef is Cloudflare's "System One" structured-output decision model: a 27.4B BF16
post-train of Qwen/Qwen3.8-27B (27,356,728,560 total parameters per Hugging Face metadata), 131K native context, Apache-2.0,
served as image-text-to-text. It is one of this run's strongest current Hugging Face signals: at the 2026-10-12 check it
held a trendingScore of 630 with 2,050 likes (https://huggingface.co/Cloudflare/clef), created 2026-09-30. That rides the
dominant Qwen3.8-27B derivative wave (base Qwen/Qwen3.8-27B trendingScore 361, 6,843,108 30-day downloads, 17,532 likes,
https://huggingface.co/Qwen/Qwen3.8-27B), rather than an independent Qwen3.8 node. No OpenRouter route or LMArena row
surfaced in this run's bounded captures for the clef node (not visible, not confirmed absent), and no high-engagement
thread surfaced, so the shell heat rests on the fresh HF trending plus the Qwen3.8 family momentum rather than a measured
serving volume.

Serving context for the record: the recipe matrix is 1x NVIDIA B200, 1x NVIDIA H200 141GB, or 2x NVIDIA A100 80GB (see
recipe.yaml). Not yet onboarded or benchmarked.

Keep this note stable until an independent OpenRouter/arena signal for the clef node appears, or the recipe is onboarded.
