# Gemma 4 12B kernels at sequence length 512 — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Do the kernel tables of the article "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published 2026-08-01) still hold
on the current compiler: the five projections of a decoder layer and the sliding layers' causal attention at sequence
length 512, in the standard lane (FP32 accumulation) and under `EMMY_FAST_MATH=1` (FP16 accumulation with periodic
promotion into FP32)?

### Status

All 12 rows succeeded (6 kernels x 2 lanes), one run ID (`2026-09-17_23-41-07`), every Emmy row passed the run
command's scaled correctness check against eager, and every row replayed under `--strict-evidence`: the golden's rows
decided every fork, nothing fell to the prior. Fast-math reproduces the last committed run of the article's recipe
on every projection. The standard lane is at or above it. Attention does not reproduce: 0.92x and 0.92x eager
against 0.97x and 1.03x.

### Protocol

Each row replays one hand-recorded working golden from `golden/` with `emmy run --golden FILE --realization SEED
--bench --strict-evidence --bench-backends eager,tcompile,emmy` at deployable `-O3`, harness defaults (10 warmups,
100 iterations), from an empty tune DB, online prior and cubin cache, so the golden's rows alone decide what deploys
and a fork they do not decide fails the row. Nothing traces, tunes or pins. Latency is the captured whole-program
forward; for a split projection that includes the finalize kernel.
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
| `q_proj` 512x3840 @ 3840x4096 | 90.1 | 90.1 | 89.8 | 64.7 | 1.00x | 1.39x |
| `kv_proj` 512x3840 @ 3840x2048 | 47.1 | 47.1 | 50.1 | 36.9 | 0.94x | 1.28x |
| `o_proj` 512x4096 @ 4096x3840 | 96.1 | 82.0 | 84.0 | 68.2 | 1.14x | 1.41x |
| `mlp_gate_up` 512x3840 @ 3840x30720 | 633.2 / 628.0 | 603.0 / 608.1 | 628.6 | 402.2 | 1.01x | 1.56x |
| `mlp_down` 512x15360 @ 15360x3840 | 307.9 / 310.0 | 309.1 / 306.2 | 310.6 | 231.8 | 0.99x | 1.34x |
| `attention` causal (1, 16, 512, 256) | 35.0 / 34.9 | 35.0 / 34.8 | 38.1 | 38.0 | 0.92x | 0.92x |

Where two baseline values appear they are the standard-lane and the fast-math task's own measurements; each ratio
uses its own task's eager.

Per-kernel device time of the split projections, partial then finalize: `q_proj` 81.7 + 5.5 (standard), 58.5 + 4.4
(fast-math); `kv_proj` 42.6 + 5.5, 32.2 + 3.1; `o_proj` 77.9 + 4.2, 62.2 + 4.2; `mlp_down` 303.3 + 5.2, 226.7 + 4.2.
`mlp_gate_up` and `attention` are single kernels.

### Comparison with the article and with its last committed run

| Kernel | Article standard | Article fast-math | 2026-08-06 run standard | 2026-08-06 run fast-math | This run |
| --- | ---: | ---: | ---: | ---: | --- |
| `q_proj` | 1.16x | 1.59x | 0.91x (99 us) | 1.38x (65) | 1.00x, 1.39x |
| `kv_proj` | 1.08x | 1.34x | 0.94x (50) | 1.21x (39) | 0.94x, 1.28x |
| `o_proj` | 1.26x | 1.61x | 1.10x (87) | 1.41x (68) | 1.14x, 1.41x |
| `mlp_gate_up` | 1.02x | 1.58x | 1.00x (631) | 1.58x (400) | 1.01x, 1.56x |
| `mlp_down` | 1.00x | 1.34x | 1.00x (309) | 1.33x (232) | 0.99x, 1.34x |
| `attention` | 0.97x | 1.03x | 0.97x (36) | 1.03x (34) | 0.92x, 0.92x |

The 2026-08-06 column is the committed run of the article's own recipe
(`experiments/gemma-4-12B/kernels_rtx5090/2026-08-06_02-08-56_6d16017e`), which shares this run's harness: captured
whole-program latency against eager on this card. It is the like-for-like reference.

The article's projection ratios are not comparable to either run. Its numbers were the rows of the standalone golden
of its day, whose `emmy_us` was the partial kernel alone and whose reference was a separately measured cuBLAS time
(97.6 us for `q_proj`, where eager measures 90). The article's 84.2 us for `q_proj` is this run's 81.5 us partial
kernel; the finalize kernel of the split is extra, and the harness reported 99 us for that schedule on 2026-08-06 and
98.5 us today. The fast-math column is less affected because its two-way splits finalize faster.

### What the numbers say

Fast-math reproduces on every projection: 1.28x to 1.56x eager against 1.21x to 1.58x in the committed run, each
kernel within 5%. The hybrid accumulation still buys 1.24x to 1.56x over Emmy's own FP32 lane.

The standard lane is level with eager on three projections, 6% behind on the narrow `kv_proj` and 14% ahead on
`o_proj`. `q_proj` moved from 0.91x to 1.00x because a four-way split beats the article's eight-way one on the current
compiler (88 us against 106 us in one sweep); the other winners are the article's schedules or a neighbouring split.
torch.compile picks a faster GEMM than eager for `o_proj` and `mlp_gate_up`; Emmy's standard lane trails it by 3% and
4% there.

Attention regressed. The committed run measured 36 us and 34 us against eager's 35; this run measures 38.1 and
38.0.
The article's schedule was FlashAttention-2's single-slab form with 64-key chunks, which today measures 41.2 us; the
best current rows are the two- and three-slot TMA rings over 32-key chunks. The FP16-accumulate value product the
article's fast-math row used was refused outright at head width 256 until this change; it is offered again and is the
fast-math row here, level with the FP32 ring (37.8 against 38.2 us in the sweep) and worth 5% on the single-slab
geometry (41.2 to 39.0 us), close to the article's 6%. The remaining distance is the base kernel on that geometry,
not the accumulator.

Without the goldens the same programs deploy the prior's pick: 93 to 556 us for the projections (1.2x to 2x slower
than eager) and 246 us for attention.

### Repeat variation

The projections were measured in six `emmy bench` runs over seven hours (`2026-09-17_17-13-13` before the
attention row existed, `17-56-52`, `18-32-59`, `21-02-23`, `23-23-36` and this one) and once by hand. Emmy's
whole-program latency agrees within 1.5% across all of them on every row except `kv_proj` (standard 50.1 to 53.7 us,
fast-math 36.9 to 38.4). Eager moves more: `kv_proj` reads between 47.1 and 51.2 us depending on the task.
Attention has five bench runs (38.1 to 38.3 us standard; 38.0 to 38.3 fast-math) and its two recording runs (38.2
and 38.0). The archived run is the last one, the first replayed under `--strict-evidence`; the earlier runs did not
carry the flag and measured the same rows.

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
| Run | `2026-09-17_21-02-23`, 12 rows, all `succeeded` |
| Source revision | `b79ac6707bbc65174070beed019bdcdef52a1bac`, clean tree |
| Host | `kenshin`, Ubuntu 24.04.2 LTS, kernel 7.0.0-28, AMD Ryzen 9 9950X3D |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02, display attached |
| Toolchain | CUDA 13.0 (nvcc V13.0.88), PyTorch 2.13.0, Triton 3.7.1, cupy-cuda12x 14.2.0 |
| Article stack | driver 580.159.03, CUDA 13.0, PyTorch 2.13.0+cu130 |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-17_21-02-23/`: `benchmark.log`,
`benchmark_rtx5090_x_1.log`, and per row `<variant>_<row id>.experiment.yaml` plus `<variant>_<row id>_artifacts.tar.gz`
holding the bench JSON and log, `status.txt`, `golden.sha256`, `requirements.freeze.txt`, `nvidia-smi.txt` and the
cubin cache. Variants: `rtx5090x1_crtx5090_k{q-p,kv-p,o-p,m-g-u,m-d,attention}_l{std,fm}`.

## NVIDIA GeForce RTX 4090 x1 (`rtx4090x1`)

### Question

The same six kernels on the article's second card. The article reports the 4090 as a per-kernel catalog only (the
model does not fit beside a KV cache) and gives one number to reproduce directly: the causal attention scoreboard,
41.0 us for torch's SDPA against 37.1 us (FP32) and 33.8 us (hybrid accumulation) for Emmy.

### Status

All 12 rows succeeded, one run ID (`2026-09-17_23-41-08`), every Emmy row passed the scaled correctness check and
replayed under `--strict-evidence`. Fast-math is 1.12x to 1.55x eager on every kernel. The standard lane is ahead of
eager on four kernels and behind on the two whose split or narrow output the card handles worse (`kv_proj` 0.88x,
`mlp_down` 0.92x). The attention scoreboard does not reproduce in full: Emmy is ahead of SDPA in both lanes, but by
3% and 12% rather than 11% and 21%.

### Protocol

The 5090 protocol, over SSH to a pre-allocated host (`riftuser@118.163.199.138`, driver 580.159.03, CUDA 13.3, PyTorch
2.13.0+cu130) with the six 4090 goldens; `golden/README.md` records the rows. The host was idle apart from the
run, and it is not display-attached.

### Measurements

| Kernel | Eager | torch.compile | Emmy standard | Emmy fast-math | Standard | Fast-math |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `q_proj` 512x3840 @ 3840x4096 | 109.7 / 109.2 | 101.3 / 101.0 | 105.7 | 71.9 | 1.04x | 1.52x |
| `kv_proj` 512x3840 @ 3840x2048 | 52.2 / 51.3 | 51.0 / 50.9 | 59.1 | 43.9 | 0.88x | 1.17x |
| `o_proj` 512x4096 @ 4096x3840 | 120.1 / 120.0 | 106.8 / 107.1 | 110.9 | 77.2 | 1.08x | 1.55x |
| `mlp_gate_up` 512x3840 @ 3840x30720 | 848.7 / 836.1 | 730.1 / 733.7 | 766.1 | 546.8 | 1.11x | 1.53x |
| `mlp_down` 512x15360 @ 15360x3840 | 389.8 / 387.3 | 408.9 / 401.4 | 426.0 | 295.3 | 0.92x | 1.31x |
| `attention` causal (1, 16, 512, 256) | 42.4 / 42.5 | 42.3 / 42.4 | 41.1 | 37.9 | 1.03x | 1.12x |

`mlp_down` is the one split row on this card (`g2k`): 418.8 + 5.1 us standard, the fast-math row is unsplit. Every
other kernel is one launch.

### Comparison with the article

| Kernel | Article (2026-07 golden rows, kernel vs cuBLAS) | This run (whole program vs eager) |
| --- | --- | --- |
| `q_proj` | 1.02x, 1.23x | 1.04x, 1.52x |
| `kv_proj` | 0.86x, 0.84x | 0.88x, 1.17x |
| `o_proj` | 1.05x, 1.34x | 1.08x, 1.55x |
| `mlp_gate_up` | 0.99x, 1.11x | 1.11x, 1.53x |
| `mlp_down` | 0.99x, 1.34x | 0.92x, 1.31x |
| `attention` | 1.11x, 1.21x (SDPA 41.0 us) | 1.03x, 1.12x (SDPA 42.4 us) |

The article's projection column is the card's golden of the day (`recipes/gemma-4-12B-it/golden/rtx4090_sm89.yaml`,
kernel time against a separately measured cuBLAS time), the same non-comparable reference as on the 5090; its
committed 4090 run (`experiments/gemma-4-12B/kernels_rtx4090`) preserved 62 of 160 rows with no lane attribution
and is not usable as a reference. The attention row is the article's own scoreboard.

### What the numbers say

Fast-math is the card's story. The FP16-accumulate tiles win by 1.17x to 1.55x with no TMA and no split, 1.31x to
1.53x on the three widest kernels, ahead of the 5090's own fast-math ratios on `q_proj` and `o_proj`. The standard
lane pays for the card's weaker synchronous transport: with `cp.async` staging only, the FP32 tiles are at or above
eager on the wide outputs and 8% to 12% behind on `kv_proj` and `mlp_down`, where the 5090's cross-CTA splits do
not help either (every split lost on this card except `mlp_down`'s two-way one). torch.compile is a stronger
baseline here than on the 5090: it beats eager by 8% to 16% on `q_proj`, `o_proj` and `mlp_gate_up`, and Emmy's
standard lane trails it there by 4% to 5%.

Attention: the FP16-accumulate value product, refused at head width 256 before this change, is worth 8% on this
card (41.1 to 37.9 us), close to the article's 9% (37.1 to 33.8). The FP32 row itself is 4 us behind the article's,
so the whole scoreboard is 4 us behind: the distance is the base kernel, as on the 5090, not the accumulator.

### Limitations

- One card, one host, one run per row; the three sweep passes agree with the run within 3% on every row except
  `mlp_down` (415 us in the sweep, 426 here) and `mlp_gate_up` (796 to 766), which moved with eager.
- The article's 139-case catalog is not part of this experiment.
- The old 4090 golden rows were recorded on a compiler four months older, against cuBLAS; the comparison column is
  indicative only.


## Re-tune on RTX 4090 and RTX 5090 (`rtx4090x1`, `rtx5090x1`, 2026-09-22/23)

### Question

Are the twelve committed schedules still the best the current compiler offers on each card, and by how much do the
re-found schedules move each kernel against eager? The tuner was not used: it is known to be broken, so the
schedules were found by hand-pinned sweeps, the way the goldens were first recorded.

### Status

All 12 goldens swept on both cards, 24 lanes, about 4,600 hand-pinned rows benched against eager, every kept row
clean under the run's integrity flags (realized-vs-pinned knobs, arithmetic-intensity floor, wrong-answer check).
21 lanes were written back through `--record-greedy` with the recorded greedy equal to the sweep winner; the three
5090 lanes whose winner is a cross-CTA split keep their previous rows (see Findings). Each golden was measured whole
on one box.

### Protocol

For every target and lane the compiler's own fork tree was enumerated (`enumerate_graph`; ~17k rows for a
projection in the standard lane, ~32k under fast-math, ~130k / ~410k for the two-site attention) and swept in three
passes of `emmy run --golden FILE --realization SEED --bench --bench-backends eager,emmy --ab KNOBS` at deployable
`-O3`: tile fragment x K step x worker split (144 pins: the previously recorded rows first, then a grid over 12 CTA
layouts and every mma atom the lane offers, wide tiles also at each cross-CTA split `g2k`/`g4k`/`g8k`), then staging
ring x reduce partition around the top three (48 pins), then rasterization x deeper K step (24 pins). Pass 1 ranked
at 5 warm-ups / 30 iterations; passes 2 and 3 and the write-back at 10 / 100. The write-back ran each lane against a
working copy holding only that lane's seed and a fresh per-lane tune DB seeded by one bench of the winner, because
the measured-evidence pick is regime-blind (below). Boxes: vast.ai RTX 4090 (driver 595.71.05, 32 cores) and RTX
5090 (driver 580.105.08, 16 cores), both CUDA 13.0.88 (the version the previous rows were recorded with) and
PyTorch 2.14.0+cu130 in a fresh venv; source revision `1159502e`. The raw pins and A/B records of every pass are in
`sweeps_rtx4090x1_2026-09-22.tar.gz` and `sweeps_rtx5090x1_2026-09-22.tar.gz`.

### Measurements

Whole-program latency in microseconds. "previous" is the golden's recorded deploy at its whole-set price (the
routing row when it deploys a split); "vs previous" compares the sweep's best with that deploy, both measured on the
same box, so it isolates the schedule from the box.

RTX 5090:

| golden | lane | eager | previous | re-measured best | vs eager | vs previous | best schedule |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q_proj` | std | 88.8 | 86.8 | 89.9 | 0.99x | 0.97x | the previous row (`f2x4/k2 g4k`) |
| `q_proj` | fm | 88.7 | 62.7 | **62.4** | 1.42x | 1.01x | `w4x2 f16_f16/f2x8/k4 d2/smem-tma` |
| `kv_proj` | std | 46.6 | 47.8 | 49.1 | 0.95x | 0.97x | `w2x4 f16_f32/f2x2/k4 d2/smem-tma gm8` |
| `kv_proj` | fm | 46.6 | 35.2 | **34.6** | 1.35x | 1.02x | `w2x8 f16_f16/f2x2/k4 d2/smem-tma/p2 gm8` |
| `o_proj` | std | 94.7 | 81.1 | 81.6 | 1.16x | 0.99x | `w2x4 f16_f32/f2x2/k4 d2/smem-async/p2 gm8` |
| `o_proj` | fm | 94.8 | 73.8 | **65.4** | 1.45x | **1.13x** | `w4x2 f16_f16/f2x8/k4 d2/smem-tma` |
| `mlp_gate_up` | std | 577.2 | 562.3 | **553.6** | 1.04x | 1.02x | `w4x4 f16_f32/f4x4/k4 d2/smem-tma/p2 gm8` |
| `mlp_gate_up` | fm | 575.6 | 419.5 | **360.0** | 1.60x | **1.17x** | `w4x2 f16_f16/f4x8/k4 d2/smem-tma/p2 gm8` |
| `mlp_down` | std | 294.2 | 287.1 | 287.0 | 1.03x | 1.00x | tie with the previous row (`f2x4/k2 g4k`) |
| `mlp_down` | fm | 298.3 | 221.1 | **210.0** | 1.42x | 1.05x | `w4x2 f16_f16/f4x8/k4 g8k d2/smem-tma` (candidate, not written) |
| `attention` | std | 30.8 | 38.1 | **32.1** | 0.96x | **1.19x** | value site `d2/smem-async`, score site `d3/smem-tma` |
| `attention` | fm | 30.8 | 38.0 | **29.1** | **1.06x** | **1.31x** | same transport, score tile `f16_f16/f1x4/k2` |

RTX 4090:

| golden | lane | eager | previous | re-measured best | vs eager | vs previous | best schedule |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `q_proj` | std | 133.4 | 105.7 | 127.6 | 1.05x | 0.83x | `w1x4 f16_f32/f4x4/k4 d2/smem-async/p2 gm8` |
| `q_proj` | fm | 133.4 | 71.8 | 88.6 | 1.51x | 0.81x | the previous row |
| `kv_proj` | std | 64.0 | 59.3 | 70.2 | 0.91x | 0.84x | `w2x2 f16_f32/f2x4/k2 d4/smem-async/p2 gm8` |
| `kv_proj` | fm | 64.0 | 43.7 | 50.5 | 1.27x | 0.87x | `w1x4 f16_f16/f4x4/k4 d4/smem-async gm8` |
| `o_proj` | std | 142.0 | 111.1 | 135.9 | 1.05x | 0.82x | `w1x4 f16_f32/f4x4/k4 d2/smem-async/p2` |
| `o_proj` | fm | 146.4 | 77.0 | 95.9 | 1.53x | 0.80x | `w4x2 f16_f16/f2x8/k4 d2/smem-async/p2` |
| `mlp_gate_up` | std | 967.7 | 800.5 | 944.1 | 1.02x | 0.85x | the previous row |
| `mlp_gate_up` | fm | 968.7 | 542.7 | 607.7 | 1.59x | 0.89x | `w2x2 f16_f16/f4x8/k2 d2/smem-async/p2 gm8` |
| `mlp_down` | std | 474.1 | 416.6 | 492.5 | 0.96x | 0.85x | `w1x4 f16_f32/f4x4/k4 d2/smem-async/p2 gm8` |
| `mlp_down` | fm | 475.1 | 284.6 | 320.5 | 1.48x | 0.89x | `w2x2 f16_f16/f4x8/k4 d2/smem-async/p2` |
| `attention` | std | 53.2 | 41.0 | 50.5 | 1.05x | 0.81x | the previous row |
| `attention` | fm | 54.4 | 37.9 | 50.0 | 1.09x | 0.76x | the previous row |

### What the numbers say

On the 5090 the fast-math lane — the one the article's claim rests on — is better than the previous deploy in four
of six goldens (2–17%, geometric mean 1.11x) and ties the other two; the standard lane is a wash (1.02x). The wins come
from two families the hand passes had not reached: the register-staged ring with rasterization (`/p2`, `gm8`) on the
projections, and a mixed transport on attention (value site on the asynchronous ring, score site on the three-slot
TMA ring) that puts this kernel at or above eager for the first time. `mlp_down` fast-math has a 5% better split
(`g8k` over `g2k`) that could not be written back (Finding 3) and is left as a candidate.

On the 4090 every lane trails the previous *number* by 10–25%, but so does every previous schedule re-measured on
this box, and so does eager (x1.18–1.45; the attention kernel x1.45): that is the box, not the schedules. Against
the previous schedules re-measured here, the picks are better in six lanes (`kv_proj` both, `o_proj` std, `q_proj`
std, `mlp_down` std, `mlp_gate_up` fm; 3–13%) and tie in the other six, so the 4090 goldens carry the better row
where one was found and the previous row otherwise. Fast-math against eager on this stack: 1.37x geometric mean on
the 5090, 1.40x on the 4090.

### Findings

- **The measured-evidence pick is regime-blind.** In every standard-lane run the greedy deployed the fast-math
  `f16_f16` kernel once the tune DB held fast-math rows (the run tags that greedy `lane=fm` although
  `EMMY_FAST_MATH` was unset), on both cards. A perf row's `knobs` record the regime but the `context_key` the pick
  joins on is the same for both regimes of one kernel. Correctness still passes at the scaled tolerance, so nothing
  flags it. Hand pins are unaffected; the write-back isolates regimes as described in Protocol.
- **A hand-pinned cross-CTA split cannot be written back.** `--ab "…REDUCE=g4k…"` records the piece and the finalize
  receipts but no row that spells the kernel-set arm, so `--record-greedy --strict-evidence` fails on the cut fork
  (`EvidenceError: … no measured row spells a kernel-set arm`) and without strict evidence lands on a prior-driven
  unsplit kernel. Unsplit winners write back exactly. The three affected 5090 lanes keep their previous rows.
- **The prior's own pick for fused attention hangs at 8k context** (`HungKernelError` after the 60 s watchdog on
  `prefill_gqa-b1-s8192` of the sibling attention experiment), and a hung greedy takes every A/B row of that run
  down with it. Pinning the run's greedy to a known row with `EMMY_KNOBS` is the workaround; a compile of that
  shape with an empty tune DB deploys a kernel that hangs.
- The recorded rows' `identity` did not steer a greedy on `1159502e` when the canonical golden was passed as
  `--golden`: the run still deployed the prior's pick. Worth a check against the recipe's `--strict-evidence`
  replay before the next reproduction.

### Limitations

Eager is PyTorch 2.14.0, not the 2.13.0 the recipe pins, so the ratios are read against this stack only; the recipe
itself was not re-run. The boxes are rented and differ from the hosts of the previous sections, which is why
absolute numbers are not compared across them. Pass 1 ranked at 30 iterations; every winner carries a 100-iteration
measurement from passes 2–3 and a second one from the write-back.

### Archive

`sweeps_rtx4090x1_2026-09-22.tar.gz`, `sweeps_rtx5090x1_2026-09-22.tar.gz`: per golden and pass, the pin list, the
`--json` A/B record and the run log; the recorded per-lane goldens and their record logs; the per-golden summaries.
