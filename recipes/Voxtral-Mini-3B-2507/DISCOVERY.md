# Voxtral-Mini-3B-2507 discovery

## Assessment on 2026-10-08

Best-effort (audio+text) recipe, heat 35. No prior note existed; this is the baseline creation for it.

mistralai/Voxtral-Mini-3B-2507 is a small Apache-2.0 speech model serving transcription, translation, and audio Q&A
on one 32 GB Volta card. The recipe serves it at FP16 (Volta has no bfloat16) through the 1Cat sm_70 vLLM audio fork,
at a 32,768-token context with the Mistral tokenizer and tool-call parser. It is an existing, runnable, useful recipe
for the fleet's only audio-in slot — the recipe is maintained for that speech coverage, not for a current demand wave.

No new independent signal this cycle: the 2026-10-08 bounded Hugging Face, OpenRouter, and LMArena checks surfaced no
fresh trending, serving-volume, or arena row for the Voxtral-Mini-3B node specifically. (The one new voice entry on
OpenRouter this week, microsoft/MAI-Voice-2.1-Flash, is a different lab's new voice model and does not change this
node's standing.) The heat therefore reflects the recipe's stable, serving-driven value rather than a demand breakout;
it stays best-effort.

Serving context for the decision: one `NVIDIA Tesla V100 SXM3 32GB`, `gpu_count: 1`, audio and text input at 32,768
context (see recipe.yaml). No deployment was authored here; onboarding sizing is owned by the fit subagents when the
recipe is later refreshed.
