# Qwen3.8-27B-FP8 discovery

## Assessment on 2026-10-05

Maintained, heat 62. Benefits from the 27B line's download utility (base Qwen/Qwen3.8-27B at 6,821,761 30-day
downloads and 16,960 likes, https://huggingface.co/Qwen/Qwen3.8-27B) and FP8 serving value on four V100s or one H200.
Arena positioning is weaker than the GLM-5.3/DeepSeek-V4.1-Flash leaders (qwen3.8-27b 1438 on the Oct 2 LMArena
board). Sibling recipes cover the consumer (AWQ) and Volta (GPTQ-Int4) paths; this recipe is the general-purpose FP8
deployment.
