# Qwen3.8-27B-GPTQ-Int4 discovery

## Assessment on 2026-10-05

Maintained, heat 40. Derivative of the dominant Qwen3.8-27B line (base Qwen/Qwen3.8-27B at 6,821,761 30-day downloads
and 16,960 likes, https://huggingface.co/Qwen/Qwen3.8-27B). This Max73333 GPTQ-Int4 checkpoint is the smallest fully
open deployment of the 27B model in the fleet: Int4 weights (about 8.1GB plus overhead) on 2x V100 16GB at 64K
context. Useful low-VRAM Volta coverage for the top HF wave; the 2x16GB total (32GB) is no larger than the
serving-blocked 1x32GB EXL3 sibling, so it keeps the maintained slot for the line's Volta path.
