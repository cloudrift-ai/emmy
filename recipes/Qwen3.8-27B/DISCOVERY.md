# Qwen3.8-27B discovery

## Assessment on 2026-10-04

Keep the full-precision recipe at heat 75 and best-effort. Demand for the model is strong: the [official Hugging Face
page](https://huggingface.co/Qwen/Qwen3.8-27B) showed about 6.8 million downloads in the preceding month and 16,900
likes when checked. Qwen describes a 27B dense vision-language model with a native 262,144-token context and 48 Gated
DeltaNet layers among 64. The download count measures interest in this checkpoint, not demand for this exact recipe.

The full-precision recipe requires eight V100 16 GB GPUs. Its [October 2 verification](RESULTS.md) measured 37.09
output tokens/s for the text serving workload on that platform. The official
[FP8 checkpoint](https://huggingface.co/Qwen/Qwen3.8-27B-FP8) has substantial adoption too; the repository's
FP8 recipe serves on four V100s. That smaller GPU footprint supports keeping a quantized variant in the maintained
set while this full-precision deployment stays best-effort. The model's popularity alone does not justify promoting
every serving variant.

The same verification found that Emmy can trace the Gated DeltaNet layers, but its generation runner does not yet
serve the recurrent state end to end. This is a qualification blocker, separate from the demand and heat assessment.
The current recipe rationale also mentions an arena ranking and the absence of an official 35B-A3B sibling. Those
claims were not independently measured in this review; verify them before using them to change the decision.

Keep this note stable until substantial evidence changes demand, the serving tradeoff, or the blocker. Routine changes
in download counts, rank, or wording do not warrant an edit.

## Assessment on 2026-10-17

Heat 85 (held from the earlier read, now restated). Two substantive points. First, a factual correction to this note
and to the prior "27B dense" phrasing: the checkpoint re-fetched 2026-10-17 from the
[safetensors metadata](https://huggingface.co/Qwen/Qwen3.8-27B) is the `Qwen3_5ForConditionalGeneration` (qwen3_5)
architecture, 64 layers, 27.78B total parameters, image-text-to-text — it is the VLM with 48 Gated DeltaNet layers
among the 64, at 262,144-token context and Apache-2.0. It is not a dense-text model. The 2026-10-17 check showed
7,516,626 30-day downloads and 17,869 likes (trendingScore 326, roughly HF trending #14 overall); the strong
local-run derivative wave (Swift-Qwen3.8, ThinkingCap, Underdog-Saluki 2-bit, and the Qwen3.8-Flash-Next 125B
breakout story at 939 points on 2026-10-04) is what is driving the attention. Second, the Oct 2 serving
qualification blocker (the generation runner does not serve the Gated DeltaNet recurrent state end to end) still
applies and is the reason this full-precision recipe stays best-effort while its FP8 sibling stays maintained —
popularity alone does not justify promoting every serving variant. Not a new demand signal; the demand read was and
is strong.
