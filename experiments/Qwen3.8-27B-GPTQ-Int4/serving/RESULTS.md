# Qwen3.8-27B GPTQ Int4 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B-GPTQ-Int4/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x2

- Archive: `results_v100x2.tar.gz`, archived root `2026-09-29_10-59-30/`.
- Run timestamp `2026-09-29T10:59:30Z`, completed `2026-09-29T11:24:08Z`; repository revision
  `044c84f11f016d7eec85a7d7f7e3cdb201344764`.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4 (12 logical CPUs), 109.7 GB RAM.
- GPUs: two NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.178.04,
  UUIDs `GPU-b6e63246-059f-6237-c010-1dfb569ade18` and `GPU-c22b6752-98c0-303a-6c06-ba56fedbb4c3`.
  PCIe topology (the tensor-parallel all-reduce crosses it, no NVLink).
- Model: `Max73333/Qwen3.8-27B-GPTQ-Int4-V100@d5a18cc1477e301e50d3fe4167fbf76db9337edc`.
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`), TP2.
- Controls: seed 0, temperature 0, ignored EOS, 2 warm-ups, text-only path (`--language-model-only`), context 65,536,
  TP2, Triton GDN prefill backend, `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Rows executed: 1 of 1 succeeded, 0 failed requests.
- System software: CUDA/nvcc 12.9.86, cuBLAS 12.9.2.10, Docker 29.8.1.

| Row | Input / output | Concurrency | Prompts |
| --- | ---: | ---: | ---: |
| `v100x2_mc4_np32_ril1000_rol1000` | 1,000 / 1,000 | 4 | 32 |

Startup cost recorded for this row: weights load 9.25 s, `torch.compile` 119.94 s, CUDA graph capture 125.0 s,
model load and warm-up 372.63 s in total.

The archive contains the per-row system-only experiment record, the client benchmark log, the engine server log, and
the executed recipe snapshot.

### Capability gates exercised against the deployed server

Recorded because each is a property of this engine image on this platform, verified on the live server during this
verification run (2026-09-29):

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returned the correct one-word answer to a capital-city question |
| Tool calling | Pass — returned a structured `tool_calls` entry `get_weather{"city": ...}` |
| Reasoning separation | Pass — with `enable_thinking: true` the engine's `reasoning` field is populated and `content` holds only the answer (no explicit opt-in required the reasoning path in this run; the parser field separates correctly) |
| Context fill | Pass — a planted marker was retrieved from a prompt that fills the full 65,536-token KV window (the server rejects any request whose input plus requested output exceeds 65,536, confirming the ceiling is enforced, not startup-only) |

### Configuration attempts / known platform behaviors

Recorded because each is a property of this engine image on this platform:

| Attempt | Outcome |
| --- | --- |
| context 262,144 (native) | KV gate: 8.16 GiB needed against a 4.57 GiB pool; engine ceiling 145,040 |
| context 131,072, Triton GDN | KV gate: 4.16 GiB needed against a 3.06 GiB pool; engine ceiling 95,648 |
| `--gdn-prefill-backend flashqla_sm70` | JIT build of `flash_qla_sm70_gdn_strided` fails at `fatal error: cusparse.h` |
| Triton prefill, flash_qla decode left enabled | same JIT build failure, reached from the GDN decode path |
| prompt > 65,536 − requested-output tokens | HTTP 400: `maximum context length is 65536 … requested N output … at least X input` — the context ceiling is enforced per request |

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-GPTQ-Int4/serving --ssh USER@HOST
```
