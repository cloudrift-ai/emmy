# Beam-1.0-501B-A23B discovery

## Assessment on 2026-10-23

Onboarding shell (generate), first note. New Reflection 501B-total / 23B-active MoE
(~Apache-2.0 open weights, text-generation), created ~2026-10-07; Hugging Face verified this
run: trendingScore ~223 (in the platform-wide trending top-15), 232 likes, 3,984 30-day
downloads (https://huggingface.co/reflection/Beam-1.0-501B-A23B). The exact parameter total
was not re-read from config.json in this bounded capture; the 501B/23B figure comes from the
repository's name, not its measured metadata, and must be confirmed from `config.json` at
onboarding time.

Not yet in OpenRouter's top-10 weekly volume list and no LMArena row surfaced this cycle
(bounded check, not confirmed absent). The stored shell heat (70) rests on the verified repository, the fresh first-party-lab
release signal, and the fleet-fitting large-MoE slot rather than a measured serving volume;
no independent Reddit/HN high-engagement thread surfaced (Reddit source was blocked,
403/429, this run).

Serving context for the record: 8x NVIDIA B200 (see recipe.yaml) - sizing is owned by the fit
subagents when this recipe is refreshed. Not yet onboarded or benchmarked.

Keep this note stable until an independent OpenRouter or arena signal for the Beam-1.0 node
appears, or the recipe is onboarded.
