# Qwen3.8 NVFP4 on the RTX 5090: tuning log

Working log of the tuning branch `tune/qwen38-nvfp4-5090` (off main d362e4ab). Goal: strict serving of
`recipes/Qwen3.8-27B-NVFP4` boots from the golden (`emmy eval golden` reports 9 of 9 twins in the fast-math lane),
then end-to-end decode faster than stock vLLM.

## Resume here

1. Working copy of the golden: `W.json` beside the record scripts (box `~/claude/tune/run1/`). The canonical file is
   written only by promotion through `GoldenFile.edit`.
2. Recording, per target: `emmy run --golden W --realization <piece row> --bench --pin-route --warmup 5 --iters 20`,
   then the same with `--record-greedy --strict --bench-backends eager,tcompile,emmy`, twice (the second record
   picks the fastest schedule the first two benched). For a target whose pieces another target shares, pass the
   route as `EMMY_KNOBS` from `route_knobs.py <W> <root>` (see Bugs).
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
  Evidence in bf16 ULPs at the element (FP64 values dumped at the comparison): pre64 `reshape` Emmy's worst element
  is 0.79 ULP off (exact 4.194, ULP 0.03125), eager's worst is 1.42 ULP off (exact 2.616, ULP 0.0156); the two
  maxima differ by 0.08 ULP at Emmy's element. A scalar schedule of the same piece (`WORK=t512, REDUCE=coop`) matches
  eager almost bit for bit (768 of 393216 elements differ) and passes, so the gap is the FP4 tile skipping eager's
  bf16 rounding, not the order of the shared K+V fold. post-sym: every disagreement is exactly one code; where Emmy
  and eager disagree Emmy equals FP64 in 73% (scales) and 79% (FP4 codes) of the elements; Emmy misses FP64 on
  17350 scales, eager on 29559. gdn64 (`k_conv1d_linear_mean_reduce_5f1430`): `type_as` misses eager by one bf16
  ULP on 114 of 327680 elements; the FP64 fallback then fails on output `_to_copy_7` (max 0.00343 vs eager 0.00323,
  mean equal). The acceptance rule is Ivan's decision; nothing is recorded without `--strict`.
- A whole-target bench now and then falls back from CUDA graph capture ("graph-capture fallback"), and a strict record
  refuses it ("backend comparison did not use CUDA graph capture"). A rerun records (post64, gdn16 `2bdde8`); the box
  script `retry.sh` re-runs the record up to three times. The cause is not known yet: the bench worker's stderr is
  not shown.
- Stale seam spellings from `--pin-route`: `GoldenFile.path_to` takes the first route that reaches a piece. gdn16's
  `k_linear_matmul_mean_reduce_db4f79` and gdn64's `k_linear_matmul_mean_reduce_02d4dd` share pieces, so the path of
  such a piece runs through the other target, and the hand pin spells seams of the other root ("PLACE pin ... names
  no seam of any kernel in this compile"). With the remaining cuts one piece ran fused and hung (2000 ms).
  Workaround: the box script `route_knobs.py` spells the pins along the target's own decision tree and passes them
  as `EMMY_KNOBS`.
- gdn64 `5f1430`: with its route applied, the remainder piece `__place_<token of 53bc49c5a7>` gets no schedule (one
  CTA, 16.2 ms of 16.5 ms): "measured DB rows, none matches any offered candidate", and the prior falls back to the
  serial pick. Workaround tried: a kernel-scoped pin `WORK@place_<token>=t128,REDUCE@place_<token>=coop`, the
  stored proposal for that piece.
- Prior picks that are clearly bad, so `--tune 10` or the stored tuned row applies: pre-sym projection piece scalar
  `t512, coop` at 40.3 ms; post-sym gate/up piece `w1x1, e2m1/f1x1/k8, d8/smem-async` at 4.4 ms (the stored proposal
  `w4x1, e2m1/f1x2/k4, d2/smem-async` from #1120 is the one to measure).

## Progress

- FM lane, measured and recorded on W: post16 (66-72 us whole target vs torch.compile 501 us), post64 (97 us vs
  513 us), pre16 (28 us vs 165 us). CPU audit of W in the serving venv: 4 of 9 FM twins deploy (gdn1, post16, post64,
  pre16).
- FM lane, GDN targets recorded on W: gdn16 `0e5142` (150-170 us vs torch.compile 81-87 us, slower: a tuning target),
  gdn16 `2bdde8` (67 us vs 510 us), gdn64 `e6d1e1` (88 us vs 525 us). gdn16 `db4f79` and gdn64 `5f1430` re-run with
  the pins above.
- The producer piece of each route had lost its split (#1171's restamp dropped the GDN producer decisions), so it got
  no schedule and ran on one CTA (gdn16: hung past 2000 ms; gdn64: 16.2 ms). Producer splits added as hand pins on the
  fresh piece tokens (box `knobs-db4f79.txt`, `knobs-5f1430.txt`): gdn16 `PLACE@place_54138af7d7/{map.1/reduce,
  map.2/inner, map.3/reduce}=cut`, gdn64 `PLACE@place_c534464056/{map.1/reduce, map.2/reduce, map.3/inner,
  map.3/inner.1/map.3/map}=cut`. Every piece then gets a parallel schedule: gdn16 `db4f79` 512 us in the bench,
  212-215 us whole target; gdn64 `5f1430` 345 us vs torch.compile 103 us (slower: a tuning target).
- Neither records yet, both on the strict rule: gdn16 `db4f79`'s pinned row fails the same-input-greedy check on one
  quantized output element (`view_16_static_fp4_scale_bits`, -12 vs -11, one code, 1 of 6144); gdn64 `5f1430` fails
  strict eager on `type_as` (2 of 327680 elements, one bf16 ULP) and then the FP64 max rule. The same-input-greedy
  check of pinned rows needs the one-code rule for quantized outputs too.

## Decisions

- Trace reference: transformers 5.14.1 (see Open gaps).
