# Qwen3.8-27B-EXL3 discovery

## Assessment on 2026-10-05

Best-effort (compiler-qualified, serving-blocked), heat 25 (raised from 15 — named observation: the checkpoint was
verified this cycle at 172 likes and 66,045 30-day downloads, created 2026-08-14,
https://huggingface.co/turboderp/Qwen3.8-27B-exl3, showing real quant-ecosystem interest; serving remains blocked).
EXL3 is a llama.cpp-family format, so the serving blocker plus the Qwen3.8 Gated DeltaNet gap (the runner does not yet
serve recurrent state end to end, per the full-precision recipe's October 2 verification) keep this recipe off the
maintained path.
