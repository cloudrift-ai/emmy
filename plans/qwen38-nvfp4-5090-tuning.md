# Qwen3.8 NVFP4 on the RTX 5090: tuning log

Working log of the tuning branch `tune/qwen38-nvfp4-5090` (off main d362e4ab). Goal: strict serving of
`recipes/Qwen3.8-27B-NVFP4` boots from the golden (`emmy eval golden` reports 9 of 9 twins in the fast-math lane),
then end-to-end decode faster than stock vLLM.

## Resume here

1. Working copy of the golden: `W.json` beside the record scripts (box `~/claude/tune/run1/`). The canonical file is
   written only by promotion through `GoldenFile.edit`.
2. Recording, per target: `emmy run --golden W --realization <piece row> --bench --pin-route --warmup 5 --iters 20`,
   then the same with `--record-greedy --strict --bench-backends eager,tcompile,emmy`, twice (the second record
   picks the fastest schedule the first two benched).
3. Audit after each batch, in the serving venv (transformers 5.14.1): `emmy eval golden --golden W --serving-config
   <copy of serving-rtx5090.env naming W>`.

## Why main deploys 0 of 9 twins strictly

- Attention twins (pre16, pre64, pre-sym, post16, post64, post-sym): the refused root is a stored kernel under another
  C name (the C name hashes the body at fusion time, the file keeps its old `ref`). Its stored route is offered, but
  every piece row is a proposal, so the route has no price. Measuring the pieces fixes it.
- GDN twins: under transformers 5.19 the traced layer differs (q/k norm in f32 in another layout, a divide by
  sqrt(128)), so the roots match no stored kernel. Under 5.14.1 the trace equals the stored programs: gdn1 deploys,
  gdn16 and gdn64 stop at unmeasured pieces of stored kernels (`k_linear_matmul_mean_reduce_db4f79`,
  `k_conv1d_linear_mean_reduce_5f1430`).

## Open gaps (not fixed here)

- Transformers versions disagree: `pyproject.toml` pins 5.19 for the compile extras, the serving venv (vLLM 0.23.0)
  runs 5.14.1, and the golden programs were traced under 5.14.1. `emmy eval golden` from a 5.19 venv reports the GDN
  twins as broken although serving (5.14.1) traces the stored programs. This branch keeps 5.14.1 as the trace
  reference: serving is what has to beat vLLM, and a 5.19 re-trace re-keys about 260 GDN rows for no speed.
- `emmy golden check` never re-traces: "current" means the stored programs still lower to the stored kernels, not
  that a fresh trace of the model gives those programs.

## Bugs and workarounds

- `--pin-route` on a target root's own row pins nothing when the row carries no latency (the root has no path of its
  own), so the fused kernel is benched (post16 root: hung kernel after 2000 ms). Workaround: name one of the route's
  piece rows as the realization; the bench still compiles the whole target.
- `--record-greedy` spells every precision gate in the new rows' pins (`FAST_MATH`, `FAST_EXP`, `F16_MMA_F32_ACC`,
  `FP8_MMA`); the audit reads both spellings as one lane.

- Strict whole-target correctness refuses FP4 targets that are more accurate than eager on average. The FP64 fallback
  (`_exact_error_comparison` in `emmy/commands/run.py`) asks the candidate's max error to be no larger than eager's
  on every output, with no slack for one output rounding step. pre64 (`k_linear_mean_reduce_fc8e7b`, the stored
  `w2x2, e2m1/f1x2/k8` schedule): output `reshape_3` max 0.0305 vs eager 0.0384, mean 0.0022 vs 0.0050, passes;
  output `reshape` max 0.0247 vs eager 0.0222, mean 0.0015 vs 0.0027, fails. Same numbers in both lanes. post-sym
  (`k_linear_reduce_ee6ca2`): the quantized output `mul_3_static_fp4_scale_bits` has max 32 vs 16 (one code step),
  mean 0.127 vs 0.204, fails. The quantize-node adoption (#1132) covers only quantize nodes that are not graph
  outputs. Open: blocks pre64, post-sym and likely pre-sym.
- A whole-target bench now and then falls back from CUDA graph capture ("graph-capture fallback"), and a strict record
  refuses it. Re-running the record passed (post64).
- Prior picks that are clearly bad, so `--tune 10` or the stored tuned row applies: pre-sym projection piece scalar
  `t512, coop` at 40.3 ms; post-sym gate/up piece `w1x1, e2m1/f1x1/k8, d8/smem-async` at 4.4 ms (the stored proposal
  `w4x1, e2m1/f1x2/k4, d2/smem-async` from #1120 is the one to measure).

## Progress

- FM lane, measured and recorded on W: post16 (66-72 us whole target vs torch.compile 501 us), post64 (97 us vs
  513 us), pre16 (28 us vs 165 us). CPU audit of W in the serving venv: 4 of 9 FM twins deploy (gdn1, post16, post64,
  pre16).

## Decisions

- Trace reference: transformers 5.14.1 (see Open gaps).
