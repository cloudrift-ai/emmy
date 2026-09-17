# Gemma 4 12B kernels at sequence length 512 — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Do the kernel tables of the article "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published 2026-08-01) still hold
on the current compiler: the five projections of a decoder layer and the sliding layers' causal attention at sequence
length 512, in the standard lane (FP32 accumulation) and under `EMMY_FAST_MATH=1` (FP16 accumulation with periodic
promotion into FP32)?

### Status

All 12 rows succeeded (6 kernels x 2 lanes), one run ID, every Emmy row passed the run command's scaled correctness
check against eager. Fast-math reproduces the last committed run of the article's recipe on every projection. The
standard lane is at or above it. Attention does not reproduce: 0.92x eager in both lanes against 0.97x and 1.03x.

### Protocol

Each row replays one hand-recorded working golden from `golden/` with `emmy run --golden FILE --realization SEED
--bench --bench-backends eager,tcompile,emmy` at deployable `-O3`, harness defaults (10 warmups, 100 iterations), from
an empty tune DB, online prior and cubin cache, so the golden's rows alone decide what deploys. Nothing traces, tunes
or pins. Latency is the captured whole-program forward; for a split projection that includes the finalize kernel.
PyTorch is pinned to 2.13.0, the article's version, in a venv of the recipe's own. `golden/README.md` records how each
row was found.

The correctness gate is the default scaled check, not `--strict`. At the down projection's accumulation depth
(K = 15360) PyTorch's default FP16 GEMM leaves 9.6% of its own elements outside `rtol=atol=1e-3` of an FP64 product,
so `--strict` rejected the FP32-accumulate lane for eager's error: 189,701 mismatching elements of 1,966,080. The
reference forward now runs with torch's reduced-precision reductions off, which leaves 2,650; those are near-zero
sums of 15,360 terms that no flat tolerance holds for an FP16 output. Fast-math is a precision trade and fails
`--strict` by design (relative L2 error near 3.3e-4).

### Measurements

Whole-program latency in microseconds; ratios are eager / Emmy within the same task, so above one favors Emmy.

| Kernel | Eager | torch.compile | Emmy standard | Emmy fast-math | Standard | Fast-math |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` 512x3840 @ 3840x4096 | 90.2 | 90.1 | 90.0 | 65.4 | 1.00x | 1.38x |
| `kv_proj` 512x3840 @ 3840x2048 | 49.2 / 47.1 | 49.2 / 47.1 | 52.3 | 37.0 | 0.94x | 1.27x |
| `o_proj` 512x4096 @ 4096x3840 | 96.1 / 95.5 | 82.1 | 83.9 | 68.0 | 1.15x | 1.40x |
| `mlp_gate_up` 512x3840 @ 3840x30720 | 632.2 / 629.4 | 602.9 / 606.5 | 628.5 | 403.3 | 1.01x | 1.56x |
| `mlp_down` 512x15360 @ 15360x3840 | 308.3 / 309.4 | 305.9 / 307.7 | 308.2 | 230.3 | 1.00x | 1.34x |
| `attention` causal (1, 16, 512, 256) | 35.0 | 34.9 | 38.2 | 38.1 | 0.92x | 0.92x |

Where two baseline values appear they are the standard-lane and the fast-math task's own measurements; each ratio
uses its own task's eager.

Per-kernel device time of the split projections, partial then finalize: `q_proj` 82.3 + 5.5 (standard), 58.8 + 4.4
(fast-math); `kv_proj` 44.8 + 5.5, 32.1 + 3.1; `o_proj` 78.0 + 4.2, 61.9 + 4.2; `mlp_down` 304.7 + 5.3, 227.5 + 4.2.
`mlp_gate_up` and `attention` are single kernels.

### Comparison with the article and with its last committed run

| Kernel | Article standard | Article fast-math | 2026-08-06 run standard | 2026-08-06 run fast-math | This run |
| --- | ---: | ---: | ---: | ---: | --- |
| `q_proj` | 1.16x | 1.59x | 0.91x (99 us) | 1.38x (65) | 1.00x, 1.38x |
| `kv_proj` | 1.08x | 1.34x | 0.94x (50) | 1.21x (39) | 0.94x, 1.27x |
| `o_proj` | 1.26x | 1.61x | 1.10x (87) | 1.41x (68) | 1.15x, 1.40x |
| `mlp_gate_up` | 1.02x | 1.58x | 1.00x (631) | 1.58x (400) | 1.01x, 1.56x |
| `mlp_down` | 1.00x | 1.34x | 1.00x (309) | 1.33x (232) | 1.00x, 1.34x |
| `attention` | 0.97x | 1.03x | 0.97x (36) | 1.03x (34) | 0.92x, 0.92x |

The 2026-08-06 column is the committed run of the article's own recipe
(`experiments/gemma-4-12B/kernels_rtx5090/2026-08-06_02-08-56_6d16017e`), which shares this run's harness: captured
whole-program latency against eager on this card. It is the like-for-like reference.

The article's projection ratios are not comparable to either run. Its numbers were the rows of the standalone golden
of its day, whose `emmy_us` was the partial kernel alone and whose reference was a separately measured cuBLAS time
(97.6 us for `q_proj`, where eager measures 90). The article's 84.2 us for `q_proj` is this run's 82.3 us partial
kernel; the finalize kernel of the split is extra, and the harness reported 99 us for that schedule on 2026-08-06 and
98.5 us today. The fast-math column is less affected because its two-way splits finalize faster.

### What the numbers say

Fast-math reproduces on every projection: 1.27x to 1.56x eager against 1.21x to 1.58x in the committed run, each
kernel within 5%. The hybrid accumulation still buys 1.24x to 1.56x over Emmy's own FP32 lane.

The standard lane is level with eager on three projections, 6% behind on the narrow `kv_proj` and 15% ahead on
`o_proj`. `q_proj` moved from 0.91x to 1.00x because a four-way split beats the article's eight-way one on the current
compiler (88 us against 106 us in one sweep); the other winners are the article's schedules or a neighbouring split.
torch.compile picks a faster GEMM than eager for `o_proj` and `mlp_gate_up`; Emmy's standard lane trails it by 2% and
4% there.

Attention regressed. The committed run measured 36 us and 34 us against eager's 35; this run measures 38.2 and 38.1.
The article's schedule was FlashAttention-2's single-slab form with 64-key chunks, which today measures 41.2 us; the
best current rows are the two- and three-slot TMA rings over 32-key chunks. The FP16-accumulate value product the
article's fast-math row used was refused outright at head width 256 until this change; it is offered again and is the
fast-math row here, level with the FP32 ring (37.8 against 38.2 us in the sweep) and worth 5% on the single-slab
geometry (41.2 to 39.0 us), close to the article's 6%. The remaining distance is the base kernel on that geometry,
not the accumulator.

Without the goldens the same programs deploy the prior's pick: 93 to 556 us for the projections (1.2x to 2x slower
than eager) and 246 us for attention.

### Repeat variation

The projections were measured in two independent `emmy bench` runs an hour apart (`2026-09-17_17-13-13`, before the
attention row existed, and this one) and once by hand. Emmy's whole-program latency agrees within 1.5% across all
three on every row except `kv_proj` standard (50.3, 52.3 and 53.7 us). Eager moves more: `kv_proj` reads 47.1 or
49.2 us depending on the task. Attention has this run and its two recording runs (38.2 and 38.0 us).

On this display-attached card a longer measurement loop (50 warmups, 300 iterations) reads eager about 10% slower
than the harness defaults do, so only ratios within one task compare.

### Limitations

- One card, one host, single runs per row; the repeat evidence above is three separate sessions, not a designed
  repeat protocol with an interval.
- Six kernels at one sequence length. The article's 277-case catalog is not part of this experiment.
- Eager is the framework and vendor-library reference, not a cuBLAS kernel time; the article's cuBLAS column cannot
  be reproduced from the harness.
- Rows were chosen by hand sweep over a few neighbours of the article's schedules, not by a search. A better row may
  exist; a worse one cannot deploy, because the golden decides.
- Fast-math rows pass the scaled check only. They trade accumulation precision by design.

### System

| Item | Value |
| --- | --- |
| Run | `2026-09-17_17-56-52`, 12 rows, all `succeeded` |
| Source revision | `8505fe5867f45511a0b914e475c211b6bbf78f99`, clean tree |
| Host | `kenshin`, Ubuntu 24.04.2 LTS, kernel 7.0.0-28, AMD Ryzen 9 9950X3D |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02, display attached |
| Toolchain | CUDA 13.0 (nvcc V13.0.88), PyTorch 2.13.0, Triton 3.7.1, cupy-cuda12x 14.2.0 |
| Article stack | driver 580.159.03, CUDA 13.0, PyTorch 2.13.0+cu130 |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-17_17-56-52/`: `benchmark.log`,
`benchmark_rtx5090_x_1.log`, and per row `<variant>_<row id>.experiment.yaml` plus `<variant>_<row id>_artifacts.tar.gz`
holding the bench JSON and log, `status.txt`, `golden.sha256`, `requirements.freeze.txt`, `nvidia-smi.txt` and the
cubin cache. Variants: `rtx5090x1_k{q-p,kv-p,o-p,m-g-u,m-d,attention}_l{std,fm}`.
