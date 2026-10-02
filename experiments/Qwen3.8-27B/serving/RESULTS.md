# Qwen3.8-27B FP16 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x8

### 2026-10-01 run (current archive)

- Archive: `results_v100x8.tar.gz`, archived root `2026-10-01_10-07-45/`.
- Run timestamp `2026-10-01T10:07:45Z`; repository revision `7c317658681415c73a929493bb9a803151486db5`.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 48 logical CPUs, 409 GiB RAM.
- GPUs: eight NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.178.04. The host
  reports NV1/NV2 between some GPU pairs and PHB between others, so the eight-way tensor-parallel all-reduce still
  crosses at least one PCIe hop per step.
- Model: `Qwen/Qwen3.8-27B@1d4bf0f2ff6012fd82039d0dd7c60c0`, BF16 weights served as FP16 (`--dtype half`).
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: seed 0, temperature 0, ignored EOS, 2 warm-ups, text-only path (`--language-model-only`), context
  262,144, TP8, Triton GDN prefill backend,
  `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Measured KV pool: 279,171 tokens, 1.06x maximum concurrency at full context.
- Rows executed: 1 of 1 succeeded, 0 failed requests.

| Row | Input / output | Concurrency | Prompts | Duration | Output tok/s | Median TPOT |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| `v100x8_ea16f6fcc69d` | 1,000 / 1,000 | 4 | 32 | 837.32 s | 38.22 | 104.90 ms |

The row ran 837.32 s (wall 888.4 s including teardown), under the 20-minute per-variant benchmark cap. Startup
cost recorded for this row: deploy 131.5 s from container create to health.

### 2026-09-05 run (superseded archive, preserved for the report cross-comparison)

- Archive root `2026-09-05_01-41-34/`; run timestamp `2026-09-05T01:41:35Z`; repository revision
  `01588556cef50c202e5370927e6788f039550801`.
- Host: same model, kernel 6.8.0-51-generic, driver 580.126.20; the topology then reported PHB between every GPU
  pair.
- Model, image, and controls: identical to the current run.
- Measured KV pool: 293,427 tokens, 1.12x maximum concurrency at full context.
- Row `v100x8_mc4_np32_ril1000_rol1000` at the same workload: 1,631.13 s, 19.62 tok/s output, median TPOT 147.98 ms.
  Over the 20-minute per-variant benchmark cap; retained because all 32 requests succeeded.
- Deploy cost: 545.9 s from container create to health.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B/serving --ssh USER@HOST
```
