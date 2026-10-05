# Qwen3.8-27B-NVFP4 discovery

## Assessment on 2026-10-05

Best-effort recipe (compiler-qualified, serving-blocked), heat 55. The Qwen3.8-27B NVFP4 ecosystem is strong: the base
Qwen/Qwen3.8-27B held 6,821,761 30-day downloads and 16,960 likes
(https://huggingface.co/Qwen/Qwen3.8-27B), and the verified NVFP4 nodes carry the bulk of the quant volume
(unsloth/Qwen3.8-27B-NVFP4 at 2,365,557 30-day downloads,
https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). The serving-blocked tag means engine support for the NVFP4 path is
not yet qualified; the specific Inferact checkpoint also did not resurface in this run's bounded identity checks
(bounded search, not deletion evidence).

Keep this note stable until serving is qualified or substantial new NVFP4-specific evidence lands.
