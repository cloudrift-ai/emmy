# Quantized checkpoint kernels — results

## RTX 5090 x1 (`rtx5090x1`)

**Question.** Does Emmy compile and strictly execute every recorded post-fusion target of one `Qwen/Qwen3-0.6B-FP8`
layer — block-scaled FP8: e4m3 weights under one scale per 128x128 block, activations quantized per token and per
128-wide K group — at deployable `-O3`, replaying the goldens committed under `golden/`?

**Scope of this run.** Only the two `format=fp8-block` rows ran (`--filter format=fp8-block`), after both block-FP8
goldens were re-recorded under per-group activation scales (PR #784). The NVFP4, AWQ and Trellis rows were not run
and have no snapshot here.

**Protocol.** Each row copies its committed golden as the working golden and, for every unmeasured seed realization,
runs `emmy run --golden ... --realization <seed> --bench --strict --no-record-nodes --warmup 5 --iters 20
--bench-backends eager,emmy` with nvcc at `-O3`, a 60 s run budget and a 120 s kernel watchdog. Each seed deploys the
receipt recorded beside it. Decode (seq 1) targets are runnable frontend programs, compared against eager on the
same graph algebra; prefill (seq 512) targets are Loop targets, which have no eager twin and are checked against
same-input greedy replay.

**Result.** Both rows succeeded: all 9 seeds passed strict with no integrity flags.

| row | target | Emmy | eager |
| --- | --- | ---: | ---: |
| seq 1 | input RMSNorm + group maximum (`k_mean_a995ea`) | 7.49 us | 114.64 us |
| seq 1 | activation quantize (`k_mul_1_dynamic_fp8_value_pointwise`) | 0.65 us | 4.09 us |
| seq 1 | post-attention norm weight broadcast | 0.62 us | 10.24 us |
| seq 512 | input RMSNorm + group maximum (`k_mean_ce52ae`) | 2.24 us | — |
| seq 512 | activation quantize, attention input (`k_mul_1_pointwise`) | 1.00 us | — |
| seq 512 | activation quantize, MLP input (`k_mul_11_pointwise`) | 1.00 us | — |
| seq 512 | k-norm weight broadcast | 1.05 us | — |
| seq 512 | input-norm weight broadcast | 1.28 us | — |
| seq 512 | post-attention norm weight broadcast | 1.28 us | — |

Each value is one strict bench of 20 iterations after 5 warmups, so there is no repeat variation to report. The eager
column is PyTorch op by op on the same program; it is the correctness oracle, not a speed baseline.

**Conclusion.** The block-FP8 layer's norm and activation-quantize kernels compile and run correctly at both lengths.
The seq-1 norm kernel deploys its hand-pinned cooperative schedule (`WORK=t512`, `REDUCE=coop`), half its greedy's
15.9 us.

**Limitations.** The goldens do not cover the layer's projections. With per-group activation scales, the q, k, v and
o projections fuse into the attention kernel and the gate, up and down projections into one MLP kernel whose down
projection recomputes the gate and up products inside its operand; neither runs within the lane budget at either
length, and both are left out of the goldens. The seq-512 causal-mask kernel is left out as well: its output is -inf
by construction, which the strict check refuses. The previous decode golden covered the q and k projections; that
coverage returns with a fusion change. Projection performance is measured separately, on isolated linears against
vLLM's block-FP8 kernels, by `scripts/bench_quant_linear.py --format fp8-block`.

**Run.** Timestamp 2026-09-11T03:38:27Z, run ID `20260911T033827Z`, `emmy bench --local` on the host `kenshin`: one
NVIDIA GeForce RTX 5090 (driver 580.173.02), AMD Ryzen 9 9950X3D, Ubuntu 24.04.2, kernel 7.0.0-28. The lane's staged
venv ran torch 2.11.0 and cupy-cuda12x 14.1.1 against the staged checkout at `1e7bcc7d6`.

**Archive.** `results_rtx5090x1.tar.gz`, root member `2026-09-11_03-38-27/`: the two row records
(`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s1-...-sl1_66e855959ca3.experiment.yaml`,
`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s512-...-sl512_dd64bb1c8f8b.experiment.yaml`), each row's
`*_artifacts.tar.gz` (working golden, seed list, per-seed strict JSON and log, cubins, `pip freeze`, status), and the
two benchmark logs.

## Re-tune on RTX 5090 (`rtx5090x1`, 2026-09-22)

### Question

Are the nine committed block-FP8 schedules still the best the current compiler offers on this card, and by how much
do re-found schedules move each kernel? The tuner was not used: it is known to be broken, so the schedules were found
by hand-pinned sweeps, the way the goldens were first recorded.

### Status

Both goldens swept, 9 lanes (these goldens carry only the standard lane), about 190 hand-pinned rows benched, every
kept row clean under the run's integrity flags (realized-vs-pinned knobs, wrong-answer check). 8 lanes were written
back through `--record-greedy`; the seq-512 norm kernel keeps its previous row (see below). Each golden was measured
whole on one box.

### Protocol

For every target the compiler's own fork tree was enumerated (`enumerate_graph`) and swept in three passes of
`emmy run --golden FILE --realization SEED --bench --bench-backends eager,emmy --ab KNOBS` at deployable `-O3`.
None of these targets is a tensor-core matmul, so each pass used the pointwise/reduce grid: the previously recorded
row first, then worker split x tile fragment x cross-CTA split (`g2k`/`g4k`/`g8k`), capped at 48 pins, passes 2
and 3 around the top three rows measured so far. Pass 1 ranked at 5 warm-ups / 30 iterations; passes 2 and 3 and the
write-back at 10 / 100. The write-back ran each lane against a working copy holding only that lane's seed and a fresh
per-lane tune DB seeded by one bench of the winner. Box: vast.ai RTX 5090 (driver 590.48.01, CUDA 13.0.88, PyTorch
2.14.0+cu130 in a fresh venv); source revision `1159502e`. The raw pins and A/B records of every pass are in
`sweeps_rtx5090x1_2026-09-22.tar.gz`.

### Measurements

Whole-program latency in microseconds. "previous" is the golden's recorded receipt; "vs previous" compares the sweep's
best with that number. Eager is the frontend program op by op where the target has one (the seq-512 Loop targets
mostly do not).

| golden | target | eager | previous | re-measured best | vs previous | best schedule | now recorded |
| --- | --- | ---: | ---: | ---: | ---: | --- | --- |
| seq 1 | `k_mean_a995ea` (input RMSNorm + group max) | 131.1 | 7.49 | 7.60 | 0.99x | the previous row (`t512`, `coop`) | same row, 7.65 |
| seq 1 | `k_mul_1_dynamic_fp8_value_pointwise` | 4.1 | 0.65 | 0.66 | 0.99x | tie between `f2` and untiled, split or not | untiled, 0.67 |
| seq 1 | `k_p_post_attention_layernorm_weight_bc_pointwise` | 12.3 | 0.62 | 0.75 | 0.82x | the previous row (`f2`) | same row, 0.75 |
| seq 512 | `k_p_attn_k_norm_weight_bc_pointwise` | 16.4 | 1.04 | 1.16 | 0.89x | the previous row (`f4`) | same row, 1.16 |
| seq 512 | `k_p_input_layernorm_weight_bc_pointwise` | 16.4 | 1.28 | **1.15** | **1.11x** | `f2` (split or not) | `f2`, 1.16 |
| seq 512 | `k_p_post_attention_layernorm_weight_bc_pointwise` | 16.4 | 1.28 | **1.15** | **1.11x** | `f2` (split or not) | `f2`, 1.16 |
| seq 512 | `k_mean_ce52ae` (input RMSNorm + group max) | — | 2.23 | 14.36 | 0.16x | no pin reproduces the receipt (below) | previous row kept |
| seq 512 | `k_mul_1_pointwise` | 16.4 | 1.02 | 1.12 | 0.91x | `f2` | `f2`, 1.12 |
| seq 512 | `k_mul_11_pointwise` | — | 0.99 | 0.99 | 0.99x | `f2` (split or not) | `f2`, 1.00 |

### What the numbers say

These kernels sit at the launch floor (about 1 us), so there is little to find. The sweep confirms six of the nine
previous schedules and moves three seq-512 tiles from `f4` to `f2`: on the two norm-weight broadcasts that is 10%
in the same-box comparison; on the attention-input quantize `f2` beats `f4` re-measured here but still trails the
previous number. Every unchanged schedule re-measures 1–20% slower than its recorded number on this box, which is
the box, not the schedule, as on the 4090 in `../gemma4_kernels/RESULTS.md`.

The seq-512 norm kernel is the exception. Its committed receipt strict-replays here at 2.3 us as one kernel of 512
CTAs x 128 threads, but the same knobs as a hand pin (`WORK=t128,REDUCE=coop`) realize a two-kernel program at
14.4 us: the target holds two kernels that spell their worker and reduce knobs the same way, so a pin is
site-ambiguous and the sweep could not reach the receipt. The previous row stays.

Cross-CTA splits (`g4k`/`g8k`) tie the unsplit row within 0.01 us on every pointwise target. The recorded unsplit
rows were kept: a hand-pinned split records no row that prices its kernel-set arm, so `--record-greedy` cannot write
it back (Finding 3 in `../gemma4_kernels/RESULTS.md`), and a tie is no reason to trade a strict-replayable receipt
for a sweep-derived row.
