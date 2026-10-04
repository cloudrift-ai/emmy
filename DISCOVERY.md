# Model discovery

This note keeps the research behind the nightly model lifecycle decisions. Recipes hold the current heat, rationale,
and lifecycle tag. The [October 4 discovery run](https://github.com/cloudrift-ai/emmy/actions/runs/37194785326) is the
starting point; its source investigations were not retained in the repository, so the observations below are the
agent's recorded assessment and need source checks before they are treated as current facts.

## Current decisions

The run kept ten complete recipes maintained. It favored current serving demand, embedding coverage, and a spread of
GPU sizes. Its maintained set includes GLM-5.3-Flash and DeepSeek-V4-Flash-0731; Qwen3-Embedding-0.6B; Qwen3-30B,
Qwen3.5-397B, and Qwen3.6-35B-A3B; DeepSeek-V4-Pro-NVFP4; and three Qwen3.8-27B variants serving on different
hardware. The exact model IDs, scores, and rationales are in the recipes.

The run moved the full-precision Qwen3.8-27B recipe to best-effort while maintaining its quantized variants. Its
recorded reason was the eight-V100 deployment footprint and the smaller-card coverage of the quantized recipes. This
is a serving-priority decision, not a claim that the full-precision checkpoint fails qualification.

The next review should verify the current demand claims behind these choices against dated Hugging Face, OpenRouter,
arena, and community sources. The previous run's source links and measured values are missing here. Add those facts
when verified, and keep the existing assessment when the evidence has not materially changed.
