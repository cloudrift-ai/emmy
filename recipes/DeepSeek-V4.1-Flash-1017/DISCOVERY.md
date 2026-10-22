# DeepSeek-V4.1-Flash-1017 discovery

## Assessment on 2026-10-22

Onboarding shell, heat 95 (first note — no prior note existed for this recipe).

deepseek-ai/DeepSeek-V4.1-Flash-1017 is the 2026-10-17 revision of the runaway #1 open-weight wave. Verified
this run from the Hugging Face API (https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017): created
2026-10-17, trendingScore 1956 — the platform's top overall text-generation trending — 2,229 likes, 42,122
30-day downloads, MIT license, image-text-to-text. The revision is a 763B-total MoE (the same 763B-class body as
the DeepSeek-V4.1-Flash base) served at 1M context.

Three independent current signals agree:

- OpenRouter: deepseek/deepseek-v4.1-flash remains the top-volume open-weight route (about 18.5T tokens/wk at the
  latest capture) and the 1017 revision is the node feeding that route.
- Hugging Face: the 1017 node is the #1 trending checkpoint overall; a community abliterated FP8
  (dealignai/DeepSeek-V4.1-Flash-1017-UNCENSORED-FP8) is itself trending (trendingScore 193), which shows active
  downstream demand on exactly this revision.
- The 1017 release supersedes the base deepseek-ai/DeepSeek-V4.1-Flash node for fresh attention; the base node stays
  as the serving route and a separate onboarding shell.

Sizing: the checkpoint is published in FP8/8-bit, so roughly 763 GB of weights with a min-to-serve near 1 TB, which
is why the existing matrix is 16x H200, 4x B200, or 8x B200 (a full-node deployment). This run does not author or
change that matrix; sizing is owned by the fit subagents when the recipe is refreshed.

Uncertainty: parameter count is not in the repository metadata (the 763B figure is from the model-card / family
read), so confirm the exact total at onboarding. Not yet onboarded or benchmarked.

Keep this note stable until onboarding settles the parameter count and serving footprint, or a larger shift in the
DeepSeek wave appears.
