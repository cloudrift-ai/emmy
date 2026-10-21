# Qwen3.8 NVFP4 on the RTX 5090: tuning plan

Goal: strict serving of `recipes/Qwen3.8-27B-NVFP4` boots from the golden (`emmy eval golden` reports 9 of 9 twins
in the fast-math lane), then end-to-end decode faster than stock vLLM.

## State

- Done: the RTX 5090 golden deploys all 9 fast-math twins under strict evidence, and a strict boot serves. Final boot
  (2026-10-20): about 56 tok/s steady decode; TTFT 0.17 / 0.25 / 0.74 s on the short (warm) / medium / long prompts.
  Stock vLLM 0.23.0, same shape: 63.2 tok/s, TTFT 0.08 / 0.30 / 1.24 s. GSM8K (200 questions, 5-shot): emmy 0.450,
  stock 0.470, within one standard error (0.035).
- Not done: decode is still slower than stock vLLM. The standard lane (`FAST_MATH=false`) deploys 1 of 9 twins
  (gdn1); serving runs the fast-math lane (`SERVE_ENV="EMMY_FAST_MATH=1"`), so no standard-lane rows were recorded.

## Open follow-ups

- Decode gap to stock vLLM (56 vs 63 tok/s). The largest known pieces:
  - Decode weight readers stream DRAM at 1.66-1.73 TB/s while all CTAs run (cuBLAS 1.74 TB/s); the loss is a ramp
    at launch start and an uneven tail at launch end, a few us per launch. No offered WORK/TILE/STAGE schedule
    closes it (gate_up 67.8-70.5 us across CTA counts and pipeline depths).
  - Measured, no gain (#1219, closed): loading each kernel's first weight stages before `griddepcontrol.wait` and
    releasing the next kernel at entry. Strict boot 55.4 tok/s against 55.9 on main; per captured program gdn1 271.4
    against 262.0 us, pre16 and post16 within 1 us. The weight loads gain nothing, and the early release costs about
    10 us per gdn1 program (also with the release moved after the kernel's own wait). A microbenchmark of cold
    back-to-back launches had predicted 0.4-0.5 ms per token. Handing out work tiles dynamically to shrink the tail
    (about 0.2 ms per token) is not prototyped.
  - Measured, no gain (#1220, closed): the GDN in_proj sibling weights (qkv, z, a, b) joined into one 5120x16480
    weight read by one launch. Its best tuned row is 112.9 us against 112.5 us for the three kernels in serving,
    because the joined producer gets no tensor-core schedule: its guarded store to each weight's own workspace
    leaves no MMA atom offered, and 16480 is not a multiple of 64 or 128, so cp.async/TMA fills refuse the masked
    N tile. cuBLAS bounds the gain at about 8-12 us per layer (104.4 us joined against 116.7 apart, cold). To
    revisit: an MMA tile for the joined producer, a padded constant or masked-N fills, and a route of its own
    (after #1206 the joined root forms `map.1/map.1/map.1/reduce.1`, so the apart route no longer spells it).
- Qwen3.8 FP8 on the V100: its GDN rows (gdn16 `13e5d6`, gdn64 `353a47`) were restamped but not re-recorded or
  tuned after #1171.
- 10 serving GPU tests (`test_gen_pack_gpu`, `test_gen_runner_gpu`, `test_attention_split_gpu`) stop with
  `EvidenceError: strict evidence: kernel 'k_linear_mean_reduce_af84cd' (node 'view_3') has no measured evidence for
  its 030_cut fork`. One of them, re-run on main fd9633b9, fails the same way; the cause is not investigated.
- Standard lane: record the 8 missing twins if serving ever needs `FAST_MATH=false`.
- Search policy: the prior or the greedy fallback can pick a single-CTA `Tile[]` arm over a large contraction (gdn64
  `k_linear_matmul_mean_reduce_73983d__place_68111979ee`: one thread for a 64x6144x5120 linear plus the recurrence,
  38.9 s per call; gdn16 `d29945__place_54138af7d7`, same shape). A non-strict boot then takes hours. No guard
  rejects such arms or prices them by work (about 20-50 lines plus a test; not done).
- Tuning and recording never measure `SHARED_CARRY=1` for serial (carried-loop) kernels: `--tune` and
  `--record-greedy` left the 61-launch form of the GDN prefill triangular solve `3d261f` in the golden (572 us against
  46 us for one launch). The shared-carry rows are hand-recorded. Other carried loops may need the same check.
- Stale measured rows that greedy trusts: the `SHARED_CARRY=0` rows of `3d261f` say 218.5 / 217.2 us, but a re-bench
  on 2026-10-19 gave 572-574 us. Re-bench or re-record rows whose kernels changed launch shape since they were
  measured.
- Compiler bug (wrong answer, not fixed): the bf16 MMA schedule `WORK=w1x4, TILE=mma_m16n8k16_bf16_f32/f1x1/k8,
  STAGE=d4/smem-tma` (192 CTAs, 48 KiB shared memory) on gdn16 `db4f79` piece `place_6a3e623a98` gives one wrong FP4
  code (flat index 86480 of `view_16_static_fp4_bits`: -6 vs -3) and its block scale. Changing any single knob gives
  output bit-identical to the scalar schedule. The prior picks this schedule greedily; the golden pins `TILE=f1x2/k8`
  there instead (15.2 us). The non-strict wrong-answer check misses it: its quantized threshold is a mean error above
  0.5% of the peak, and one bad element in 98304 stays under it.
- Compiler bug (performance, worked around in the golden): `_cut._abstracted` numbers a contraction's free-index
  placeholders in operand order, by first appearance in the lowered body. Under the block amax the weight operand
  comes first, elsewhere the activation does, so `_group_contractions` splits one contraction into two groups at M>1
  and the amax occurrences keep their own copy of the contraction (the GDN out_proj + rmsnorm + FP4 quant kernels
  recomputed out_proj as a scalar loop, 9-10 ms per layer). The golden cuts the second contraction group
  (`PLACE@map.2/map.2/reduce.1/inner`) and splits the fused sibling producer, at the cost of a second out_proj MMA
  per layer (about 20 us gdn64, 11 us gdn16). Fix: group by an operand-order-free numbering and lay the workspace out
  in the first member's order (about 40-60 lines); it would save about 1.2 ms per 64-token chunk.
- Strict rule, known loose spot: a float output's rounding step comes from the coarsest float format its producing
  chain computes in, and the rule has no list of which ops round and which only move data. So a bf16 slice cast to
  f32 and followed by f32-only math still gets the bf16 step, looser than the f32 math needs. The mean-error rule
  still applies to such outputs.
- `roofline.py` `time_program_us` reports a single cold first run when that run exceeds 10x the floor. In the full
  64-layer boot pre16 and post16 showed 4.26 and 3.88 ms against 44-51 and 134 us steady state. The cause (first
  touch or placement in the full boot) is not investigated.
- Rows of neighbouring schedules that passed strict and are worth re-checking: gdn64 `...5f1430.m64.8105a48e7b21`
  (`w1x8`, bf16 `f1x1/k8`, `d4/smem-tma`), pre16 `...a7efbb.m16.fm.83c0b7f74b51` (FP4 tile, `w1x4`, `d4/smem-tma`),
  gdn16 `0e5142` `...8105a48e7b21` (`d3/smem-tma`).

## Decisions

- Trace reference: transformers 5.14.1. `pyproject.toml` pins 5.19 for the compile extras, the serving venv
  (vLLM 0.23.0) runs 5.14.1, and the golden programs were traced under 5.14.1. Under 5.19 the GDN layer traces
  differently (q/k norm in f32 in another layout, a divide by sqrt(128)), so `emmy eval golden` from a 5.19 venv
  reports the GDN twins as broken although serving traces the stored programs. Serving is what has to beat vLLM, and
  a 5.19 re-trace re-keys about 260 GDN rows for no speed.
- Slower measured rows stay beside the faster ones, as in the canonical goldens (several measured rows for 89 of 253
  measured kernel+pins groups; every `.tuned-*` row keeps its base row). A strict compile picks the fastest matching
  row either way. Routing rows stay one per arm.
- Decode matmul rows carry only schedules that also win with a cold cache (see Gotchas: hot benches).
- NVFP4 decode tables stay as they are. On the tuned golden every NVFP4 linear of the 9 fast-math programs runs as
  W4A4 (4-bit weights and 4-bit activations) on the block-scaled fp4 MMA
  (`mma.sync...kind::mxf4nvf4.block_scale...e2m1.e2m1.f32.ue4m3`). TMA copies the packed weight and activation bytes
  into shared memory and the tensor core reads them unchanged; the activation encode uses ALU compares. No deployed
  kernel reads the 256x2 pair table (`*_f4_pairs`) or the `EMMY_F4_LUT_*` fragment table, and no scalar kernel reads
  NVFP4 data (the scalar weight readers in gdn1 read bf16 in_proj weights). A table-free decode (byte load, nibble
  extract, `cvt.rn.f16x2.e2m1x2`) therefore gains 0 ms/token. Every fp4-MMA kernel still takes the pair table as an
  unread parameter, a few bytes per launch. A table-free decode would matter only if a W4A16 schedule (4-bit weights,
  bf16 activations) wins a pick. CUDA 13.0, the recording toolchain, has only the f16x2 convert on sm_120a, so a bf16
  decode there takes an extra f16x2-to-bf16x2 step (exact for e2m1 values); CUDA 13.3 adds `cvt.rn.bf16x2.e2m1x2`.

## Gotchas

- A kernel launched early (programmatic dependent launch) must not read an earlier kernel's output through the
  read-only path: `griddepcontrol.wait` makes the earlier writes visible to coherent loads only, and inputs declared
  `const __restrict__` compile to `LDG.E.CONSTANT`. On #1219 that read a stale RMS-norm sum of squares (gdn1
  launches 16 and 17). Only program constants may keep `__restrict__` / `__ldg` under early release.
- `emmy golden check` never re-traces: "current" means the stored programs still lower to the stored kernels, not
  that a fresh trace of the model gives those programs.
- A strict record also benches the named realization's own stored row as a pinned config, and that row's schedule
  knobs replay as bare, unscoped pins on every kernel of the compile. Name a row whose knobs suit every piece of the
  route (the big piece's own row); a small piece's row can leave the big piece with no measured candidate or with a
  pick over the 10 s GPU bench budget.
- `--pin-route` on a target root's own row that has no latency pins nothing, so the fused kernel benches (a hung
  kernel shows up as `HungKernelError ... did not complete within 2000 ms`). Name one of the route's piece rows as
  `--realization`; the bench still compiles the whole target.
- `GoldenFile.path_to` takes the first route that reaches a piece. gdn16 `db4f79` shares pieces with gdn64 `02d4dd`,
  so a hand pin spelled along that path can name seams of the other root (`PLACE pin ... names no seam of any kernel
  in this compile`). Spell the pins along the target's own decisions.
- A piece with one CTA and a multi-ms or hung time has lost its split decision (a restamp can drop producer
  decisions). Cut its seams by hand with `PLACE@place_<token>/<seam>=cut`.
- `--record-greedy` spells every precision gate in the new rows' pins (`FAST_MATH`, `FAST_EXP`, `F16_MMA_F32_ACC`,
  `FP8_MMA`); the audit reads both spellings as one lane. The golden rows carry `{"FAST_MATH": lane}` alone.
- A whole-target bench now and then falls back from CUDA graph capture, and a strict record refuses it ("backend
  comparison did not use CUDA graph capture"). The capture that fails is the eager PyTorch backend's
  (`cudaErrorStreamCaptureInvalidated`); a rerun usually records.
- `run --kernel <cut FP4 piece>` fills the piece's FP4 decode tables (`*_f4_pairs`) with random values, so tile rows
  fail correctness there and `--tune` counts every tile row as a failure. Check candidates through the whole target
  (`--realization <target row> --pin-route --strict`).
- `benchmark run stage exceeded 10.0s of GPU time`: a multi-ms pick in the pinned or greedy set. Replace the pick
  with a kernel-scoped schedule pin, or for a record-only run use `--warmup 2 --iters 5`.
- `unreproducible pin: ... realized ...`: a hand schedule pin clashes with a pinned golden row of the same
  realization. Drop the hand pin before the strict record.
- `--cold-cache` bench times come in steps of about 2.05 us, too coarse to rank decode matmul schedules. Cold A/B
  runs go under nsys and read per-launch durations instead.
- Hot benches of weight readers keep the weights in the 96 MiB L2 of the 5090 (gate_up, 100 MB: 25 us hot, about
  4 TB/s; 68 us cold). A hot `--tune` can rank schedules that gain nothing cold (post16 gate_up hot winner
  `w1x2 f1x4/k4`: -10 us hot, 0 cold).
- After a main merge that re-keys pieces, a re-keyed piece can lose its only measured row, or keep a row whose
  schedule the re-formed piece no longer offers (`WORK=t512` then fails as "realized (off)"). Re-record it on an
  offered schedule.
