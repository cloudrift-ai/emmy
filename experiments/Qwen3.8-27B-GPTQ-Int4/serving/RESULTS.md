# Qwen3.8-27B GPTQ Int4 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B-GPTQ-Int4/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x2

- Archive: `results_v100x2.tar.gz`, archived root `2026-09-05_01-19-15/`.
- Run timestamp `2026-09-05T01:19:15Z`, completed `2026-09-05T01:40:28Z`; repository revision
  `01588556cef50c202e5370927e6788f039550801`, staged source clean.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-51-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM.
- GPUs: two NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.126.20,
  UUIDs `GPU-e34b3197-e72f-6bad-42ec-6098b55c4d85` and `GPU-b6e63246-059f-6237-c010-1dfb569ade18`.
  The host reports PHB between every GPU pair, so the tensor-parallel all-reduce crosses PCIe.
- Model: `Max73333/Qwen3.8-27B-GPTQ-Int4-V100@d5a18cc1477e301e50d3fe4167fbf76db9337edc`.
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: seed 0, temperature 0, ignored EOS, 2 warm-ups, text-only path (`--language-model-only`), context 65,536,
  TP2, Triton GDN prefill backend, `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Rows executed: 1 of 1 succeeded, 0 failed requests.

| Row | Input / output | Concurrency | Prompts |
| --- | ---: | ---: | ---: |
| `v100x2_mc4_np32_ril1000_rol1000` | 1,000 / 1,000 | 4 | 32 |

Startup cost recorded for this row: weights load 8.6 s, `torch.compile` 105.0 s, CUDA graph capture 110.0 s,
model load and warm-up 317.9 s in total.

The archive contains the per-row system-only experiment record, the client benchmark log, the engine server log, and
the executed recipe snapshot.

### Configuration attempts that did not reach health

Recorded because each one is a property of this engine image on this platform, not of the workload:

| Attempt | Outcome |
| --- | --- |
| context 262,144 (native) | KV gate: 8.16 GiB needed against a 4.57 GiB pool; engine ceiling 145,040 |
| context 131,072, Triton GDN | KV gate: 4.16 GiB needed against a 3.06 GiB pool; engine ceiling 95,648 |
| `--gdn-prefill-backend flashqla_sm70` | JIT build of `flash_qla_sm70_gdn_strided` fails at `fatal error: cusparse.h` |
| Triton prefill, flash_qla decode left enabled | same JIT build failure, reached from the GDN decode path |

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-GPTQ-Int4/serving --ssh USER@HOST
```
