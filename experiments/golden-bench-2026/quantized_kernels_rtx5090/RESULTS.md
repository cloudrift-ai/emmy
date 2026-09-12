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
vLLM's block-FP8 kernels, in `evaluation_results/2026-09-10_fp8-block-linear-rtx5090.md`.

**Run.** Timestamp 2026-09-11T03:38:27Z, run ID `20260911T033827Z`, `emmy bench --local` on the host `kenshin`: one
NVIDIA GeForce RTX 5090 (driver 580.173.02), AMD Ryzen 9 9950X3D, Ubuntu 24.04.2, kernel 7.0.0-28. The lane's staged
venv ran torch 2.11.0 and cupy-cuda12x 14.1.1 against the staged checkout at `1e7bcc7d6`.

**Archive.** `results_rtx5090x1.tar.gz`, root member `2026-09-11_03-38-27/`: the two row records
(`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s1-...-sl1_66e855959ca3.experiment.yaml`,
`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s512-...-sl512_dd64bb1c8f8b.experiment.yaml`), each row's
`*_artifacts.tar.gz` (working golden, seed list, per-seed strict JSON and log, cubins, `pip freeze`, status), and the
two benchmark logs.
