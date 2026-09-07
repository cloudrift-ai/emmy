# Qwen3.8-27B FP16 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x8

- Archive: `results_v100x8.tar.gz`, archived root `2026-09-05_01-41-34/`.
- Run timestamp `2026-09-05T01:41:35Z`; repository revision `01588556cef50c202e5370927e6788f039550801`, staged
  source clean.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-51-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM.
- GPUs: eight NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.126.20. The host
  reports PHB between every GPU pair, so the eight-way tensor-parallel all-reduce crosses PCIe.
- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`, BF16 weights served as FP16 (`--dtype half`).
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: seed 0, temperature 0, ignored EOS, 2 warm-ups, text-only path (`--language-model-only`), context
  262,144, TP8, Triton GDN prefill backend, FLASH_ATTN_V100 attention backend,
  `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Measured KV pool: 293,427 tokens, 1.12x maximum concurrency at full context.
- Rows executed: 1 of 1 succeeded, 0 failed requests.

| Row | Input / output | Concurrency | Prompts |
| --- | ---: | ---: | ---: |
| `v100x8_mc4_np32_ril1000_rol1000` | 1,000 / 1,000 | 4 | 32 |

This row ran 1,631.13 s, over the 20-minute per-variant benchmark cap. It is retained because it completed with all
32 requests successful and no failures; the overrun is a property of decode speed on this platform, recorded here
rather than corrected by shrinking the workload.

Startup cost recorded for this row: deploy 545.9 s in total from container create to health.

The archive contains the per-row system-only experiment record, the client benchmark log, the engine server log, and
the executed recipe snapshot.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B/serving --ssh USER@HOST
```
