# Qwen3.8-27B-FP8 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B-FP8/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x4

- Archive: `results_v100x4.tar.gz`, archived root `2026-09-05_02-53-27/`.
- Run timestamp `2026-09-05T02:53:27Z`; repository revision `01588556cef50c202e5370927e6788f039550801`, staged
  source clean.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-51-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM.
- GPUs: four NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.126.20. The host
  reports PHB between every GPU pair, so the tensor-parallel all-reduce crosses PCIe.
- Model: `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: seed 0, temperature 0, ignored EOS, 2 warm-ups, text-only path (`--language-model-only`), context
  262,144, TP4, Triton GDN prefill backend, FLASH_ATTN_V100 attention backend, `VLLM_SM70_FP8_TURBOMIND=1`,
  `VLLM_SM70_QUANT_BACKEND=turbomind`, `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Measured KV pool: 292,125 tokens, 1.11x maximum concurrency at full context.
- Rows executed: 1 of 1 succeeded, 0 failed requests.

| Row | Input / output | Concurrency | Prompts |
| --- | ---: | ---: | ---: |
| `v100x4_mc4_np16_ril1000_rol1000` | 1,000 / 1,000 | 4 | 16 |

This row uses 16 prompts rather than the 32 used on the other two platforms. The 32-prompt form of the same workload
ran 1,631 s on the FP16 lane, over the 20-minute per-variant cap; halving the request count is the prescribed remedy
and this row completed in 492.87 s. Request count therefore differs from the FP16 and Int4 rows and the totals are
not directly comparable, though the per-token latencies are.

Startup cost recorded for this row: deploy 445.5 s in total from container create to health.

The archive contains the per-row system-only experiment record, the client benchmark log, the engine server log, and
the executed recipe snapshot.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
