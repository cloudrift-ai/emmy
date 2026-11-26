# MiniMax-M3.1-Flash discovery

## Assessment on 2026-10-26

Onboarding shell, heat 88. No prior note existed; this is the baseline creation for it.

MiniMaxAI/MiniMax-M3.1-Flash is MiniMax's 10/17 successor to MiniMax-M3, created 2026-10-17. Verified this run from
the Hugging Face API (https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash): a full open-weight checkpoint, ~220B
total (mxfp8 + U8 shards), 7,355 downloads and 360 likes as of this run, under the `minimax-community` license. The
stored serving read — OpenRouter top-3 weekly (13.7T/wk, new entrant) — is a bounded read this run could not
re-measure: OpenRouter's keyless volume endpoints were unreachable, so the route number stands unverified rather
than re-measured; the repo's own modest counters (9 days old at check) reflect freshness, not failure of demand, and
the MiniMax M3 wave it sits in is still active (the M3 base sits off the trending list, not deleted). A distinct node
from the existing M3 shell (16-GPU class). Matrix: 1x MI350X, 8x H100, or 4x H200 (see recipe.yaml). Not yet
onboarded or benchmarked.

Keep this note stable until an OpenRouter/arena signal for the M3.1-Flash node is re-measured, or the recipe is
onboarded.
