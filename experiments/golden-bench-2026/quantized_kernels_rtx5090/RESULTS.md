# Quantized checkpoint kernels — results

## Current NVFP4, AWQ and Trellis reproduction (2026-10-13)

All four recipe rows reached a terminal state. NVFP4 decode and AWQ succeeded; NVFP4 prefill and Trellis failed.
Eight NVFP4 targets passed an independent eager or FP64 accuracy check. AWQ used the CLI's same-input greedy
reference convention, which does not independently establish agreement with the model. Three targets remain blocked.
This is compiler coverage evidence, not a model-serving speedup or a comparison with specialized vendor kernels.

Each row traces layer zero of its pinned checkpoint, then attempts every distinct post-fusion target at deployable
O3 and standard math. The CLI compares the selected program with its available reference using five warmups and
20 captured iterations, strict rtol=atol=0.001, a 60-second benchmark run budget and a 120-second kernel watchdog.
It preserves other target results after a failure. There is one measurement per target and no process-repeat
distribution. No failed target was removed or rerun alone, and no tolerance was relaxed.

| Format | Sequence length | Terminal status | Target results |
| --- | ---: | --- | --- |
| NVFP4 | 1 | Succeeded | 5/5 strict eager passes |
| NVFP4 | 512 | Failed | 3/5 strict passes through the existing FP64 comparison; one binding error and one run-budget failure |
| AWQ | 1 | Succeeded | 1/1 executed; self-reference only, no independent accuracy check |
| Trellis / EXL3 | 1 | Failed | 0/1: malformed lift binding before measurement |

The successful target timings below are captured whole-forward measurements of that target's selected kernel set.
They must not be added into a layer latency: traced targets can overlap or serve different outputs. Eager uses the
same target program and generated inputs, not an optimized quantized serving implementation. AWQ has no eager
timing. The largest NVFP4 decode target takes about 303 ms despite passing accuracy, so success alone does not
establish useful performance.

| Format / length | Target | Emmy, µs | Eager, µs | Accuracy reference |
| --- | --- | ---: | ---: | --- |
| NVFP4 / 1 | `k_sdpa_linear_mean_reduce_7f374c` | 1272.448 | 653.285 | Eager |
| NVFP4 / 1 | `k_linear_mean_reduce_1ac1ce` | 302766.296 | 467.135 | Eager |
| NVFP4 / 1 | `k_mean_e2759d` | 83.859 | 122.833 | Eager |
| NVFP4 / 1 | `k_linear_reduce_983ccf` | 93.574 | 3058.576 | Eager |
| NVFP4 / 1 | `k_linear_e0ea9f` | 56.296 | 1481.222 | Eager |
| NVFP4 / 512 | `k_mean_203eb4` | 792.352 | 336.955 | FP64 comparison |
| NVFP4 / 512 | `k_linear_reduce_122933` | 254.696 | 3989.094 | FP64 comparison |
| NVFP4 / 512 | `k_linear_fa07f3` | 92.944 | 1783.443 | FP64 comparison |
| AWQ / 1 | Whole traced target, nine launches | 21575.411 | — | Same-input greedy only |

NVFP4 decode's largest direct absolute difference is 0.00048828125. The three prefill checks do not pass by exact
agreement with eager: their direct maximum absolute differences are 0.03125, 16.0 and 0.015625 respectively.
The existing FP64 check finds the candidate no less accurate than eager, using per-output maximum error and the
existing mean-error allowance. Packed values and scale carriers are decoded before comparison. Every per-output
error is retained in the JSON; these are not silently relabeled as elementwise passes. AWQ reports a same-input
greedy pass with zero error, but the CLI compares the saved output with itself in this case. It has no pinned or
isolated replay row. This self-reference validates neither the output independently nor repeated-run equality.

NVFP4 prefill's attention target, `k_sdpa_linear_mean_reduce_6f3014`, reads the unbound value `a15` during
materialization and produces no timing JSON. Its `k_linear_mean_reduce_5a06a3` target selects five kernels,
including an E2M1 tensor-core/TMA kernel, but exceeds the 60-second benchmark run budget. Its JSON records
`bench_fail`, no captured latency and failed strict checks. This budget failure is not a numerical mismatch or
proof that a single kernel hit the 120-second watchdog. Trellis's target fails because its lift binds zero
parameters for one operand result component. Its failed row and raw diagnostic remain in the archive.

The checkpoints are `nvidia/Qwen3-8B-NVFP4@ccd10a893cbca613259517c3efe08e151ddf2b8e`,
`hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4@db1f81ad4b8c7e39777509fac66c652eb0a52f91`, and
`riftstack/Laguna-S-2.1-exl3@45c1f34a8c3ae3025a8bce2b1decea4991a6a239`. The run used the refitted priors at
revision `d413e243705c1b6075a2ad8e37e69817aa7c123a`, PyTorch 2.13.0+cu130, Transformers 5.14.1, CUDA 13.0.88 and
driver 580.178.04 on `kenshin`: one 32 GB RTX 5090, Ryzen 9 9950X3D and Ubuntu 24.04.2. This is the local desktop
GPU, not a dedicated headless host. The protocol-input hashes match the named source revision.

The later reduction partial-merge type fix at `d3d02f3c2` leaves these measured kernels unchanged. Of 45 recorded
kernel executions, two use register-partial reduction; both carry FP32 state, and their freshly compiled CUDA
source hashes exactly match the recorded hashes. The other schedules do not reach that path, and these programs
have no scan observers. The recipe explicitly disables evidence recording, so the fresh-golden evidence fix also
does not change its execution. This source check does not turn any failed target into a successful reproduction.

The invocation is `20261013T085231Z`. Archive `results_rtx5090x1.tar.gz` is rooted at `2026-10-13_08-52-31/`,
with SHA-256 `2309875652eb90d275931f3192944028cd909955fc433120836507add7796b9f`. It contains four terminal
system-only records and their declared artifact bundles, logs, working goldens, package freezes and provenance.
The NVFP4 `verification/` directories contain one JSON per completed target; AWQ's single JSON is `verification`.
Pre-measurement failures are recorded in `verification.log` and the row status. Separate diagnostic directories
retain both earlier complete attempts; `source-impact-d3d02f3c2/` preserves the source comparison above. All 369
archived files were compared byte for byte with the collected evidence; 429 text entries, including nested
artifacts, passed the credential-pattern scan.

## Historical block-FP8 study

**Retired rows (2026-09-28).** The block-FP8 rows and their two goldens are gone from the recipe. Maximal fusion now
traces the `Qwen/Qwen3-0.6B-FP8` layer as one whole-layer kernel, so no fresh lowering writes the norm and
activation-quantize targets those goldens stored, and a re-record would need a hand-made cut route for the FP8 layer.
The run below is kept as the record of what they measured.

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

**Archive.** Historical version of `results_rtx5090x1.tar.gz` in Git, root member `2026-09-11_03-38-27/`: the two
row records
(`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s1-...-sl1_66e855959ca3.experiment.yaml`,
`rtx5090x1_ffp8-b_gqwen3-06b-fp8-b-s512-...-sl512_dd64bb1c8f8b.experiment.yaml`), each row's
`*_artifacts.tar.gz` (working golden, seed list, per-seed strict JSON and log, cubins, `pip freeze`, status), and the
two benchmark logs.
