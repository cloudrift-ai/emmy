# MiMo-V2.6-Flash-RL onboarding

## 2026-09-30 — AMD Instinct MI350X x4

Onboarding did not produce a serving recipe. A stock vLLM ROCm nightly served a coherent text response, but the
required MXFP4 MoE path had no stable, published ROCm image qualified for this checkpoint. Tool calling, multimodal
input, and the claimed long context were not validated. Emmy's CUDA backend cannot run on this AMD GPU, so no golden
was produced. The failed run's experiment files were incomplete and were not retained. A stable, pinned ROCm image
with the model's MoE path and full capability checks would allow an explicit retry.

Evidence: [September 30 qualification run](https://github.com/cloudrift-ai/emmy/actions/runs/36698998344).

## 2026-09-27 — AMD Instinct MI350X x4

The earlier attempt also failed. The available vLLM ROCm image could not load the checkpoint because its attention
sink, DiffKV, and vision paths depended on CUDA-only code. The weights fit the rented GPU memory, but no valid serving
recipe or Emmy golden resulted.

Evidence: [September 27 qualification run](https://github.com/cloudrift-ai/emmy/actions/runs/36310621687).
