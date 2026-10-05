# Edge0-35B-A3B-preview discovery

## Assessment on 2026-10-05

Onboarding shell, heat 72. HF API at the 2026-10-05 check: 80,790 30-day downloads, 3,565 likes, created 2026-09-08
(https://huggingface.co/Edge0/Edge0-35B-A3B-preview). This is a 4-bit MLX adapter + LoRA pack built on
Qwen/Qwen3.6-35B-A3B, not a full standalone checkpoint — the deployment matrix (1x H200) assumes the underlying 35B
model plus the adapter load. The likes count corroborates real community interest in A3B edge serving, and the prior
breakout read stands.

Uncertainty: adapter-only packaging may complicate a clean vLLM/SGLang deployment until onboarding.
