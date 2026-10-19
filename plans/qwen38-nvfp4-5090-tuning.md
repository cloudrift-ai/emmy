# Qwen3.8 NVFP4 on the RTX 5090: tuning log

Working log of the tuning branch `tune/qwen38-nvfp4-5090` (off main d362e4ab). Goal: strict serving of
`recipes/Qwen3.8-27B-NVFP4` boots from the golden (`emmy eval golden` reports 9 of 9 twins in the fast-math lane),
then end-to-end decode faster than stock vLLM.

## Resume here

State on 2026-10-17 11:00 UTC: `tune/strict-rule` is merged (215bbc72). FM lane recorded for post16, post64, pre16,
pre64, post-sym, pre-sym, gdn16 `0e5142` / `2bdde8` / `db4f79`, gdn64 `e6d1e1`. The CPU audit (`audit.sh a8`) shows 8 of
9 FM twins deploying (all but gdn64) and 1 of 9 STD twins (gdn1; the STD lane waits until FM is 9 of 9 and the
strict boot works). Left in the FM lane: gdn64 `5f1430` (still fails strict under the new rule; waits on a rule question to Ivan). Pins are rewritten (11:06 UTC): the 84 rows that spelled all four precision gates now carry the lane alone, `{"FAST_MATH": lane}`, through `GoldenFile.edit` (script `~/claude/tune/pins_rewrite.py`); the audit of the rewritten copy (`logs/audit-a7.log`) is unchanged at 7 of 9 FM. Backup before the rewrite: `W-after-sym.json`. Rows recorded later may spell all four gates again; rerun the script before promotion.

### Layout on the box (all paths under `/home/riftuser`)

- Worktree of this branch: `claude/work-tune`. Every command below runs from it with `PYTHONPATH=$PWD`, after
  `. ~/claude/env.sh` (nvcc 13.0, `gpu_run`, GPU lock `~/gpu.lock`).
- Run directory `claude/tune/run1/`:
  - `W.json`: the working copy of the golden, which every record writes into. `W-start.json` is main's file and
    `W-before-gdn.json` the copy before the GDN records. The canonical golden is written only at the end, by
    promotion through `GoldenFile.edit`.
  - `fm.db` and `std.db`: the tune DBs, one per lane. `cubins/`: the cubin cache. `logs/`: one log per step,
    `d1-*`, `d2-*`, `d3-*`, `retry-*`.
  - `serving-W.env`: `recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env` with `SERVE_GOLDEN_FILE` pointing at `W.json`.
  - Scripts:
    - `rec.sh LANE R TAG`: bench (D1, `--bench --pin-route --warmup 5 --iters 20`), then two strict records (D2, D3,
      `--record-greedy --strict --pin-route --bench-backends eager,tcompile,emmy --warmup 5 --iters 20`).
      `KNOBS=...` exports `EMMY_KNOBS` for all three steps; `NO_D3=1` skips D3.
    - `rec2.sh R TAG`: for a target with a clearly bad prior pick. D1 benches with `SCHED` (kernel-scoped schedule
      pins on the bad piece) on top of `KNOBS`, so the tune DB holds that measurement; D2 and D3 record without
      `SCHED`. Retries the record up to twice more while CUDA graph capture falls back. FM lane only.
    - `retry.sh R TAG`: the strict record alone, up to 3 attempts. FM lane only; honours `KNOBS`.
    - `audit.sh TAG`: CPU-only `emmy eval golden` of `W.json` in the serving venv, which prints each refused fork.
      Log: `logs/audit-TAG.log`.
    - `route_knobs.py W ROOT`: prints the hand pin of ROOT's own recorded decisions (see Bugs: `path_to`).
    - `route_check.py W ROOT`: replays ROOT's stored routes and lists stored cut keys the fresh parent no longer
      offers.
  - Pin files: `knobs-db4f79.txt`, `knobs-5f1430.txt` (route plus producer split), `sched-presym.txt`,
    `sched-postsym.txt` (the stored #1120 schedule `w4x1, mma_m16n8k64_e2m1_f32/f1x2/k4, d2/smem-async` on the bad
    piece's fresh token).
- Diagnostic tools in `claude/tune/diag/`:
  - `fork_diag.py <golden> <emmy args...>`: runs `emmy` and, at every strict-evidence refusal, prints the fork
    root's identity, the stored kernel with that identity, its rows (measured or proposal) and routes, and the
    offered cut keys. With `DIAG_CPU=1` it fakes the RTX 5090 context, so `eval golden` runs without the GPU.
  - `site/sitecustomize.py`: prepend `~/claude/tune/diag/site` to `PYTHONPATH`. It also loads in bench workers,
    whose stderr is hidden. `DIAG_PROOF=<file>` appends every strict correctness proof to that file;
    `DIAG_DUMP=<dir>` saves the candidate, eager and FP64 arrays at every FP64 comparison (one npz per call);
    `DIAG_LOG=<file>` sends INFO and above from every process to that file.
  - `ulp.py <npz>...`: errors against FP64 in bf16 ULPs.
  - `trace_diff.py <dir>`: writes fresh and stored twin programs, for diffing.
  - Seam listing: `SEAMS_TOKEN=<piece token> python ~/claude/restore/box-24700/claude/split/seams_of.py compile
    --golden W --realization R --target sm_120 --ir kernel` (with the route in `EMMY_KNOBS`) prints the seams the
    cut pass offers on that piece.

### Ordered plan (FM lane)

Each record needs the GPU and waits for the lock. A step is done when its log shows `D2 rc=0` or `D3 rc=0`, plus
`[record-evidence] ... routing rows`. Run each step in a named tmux session and tee its output to `~/logs/NAME.log`.

1. pre64:
   `~/claude/tune/run1/rec.sh fm pre64-dense-full@nvfp4.k_linear_mean_reduce_fc8e7b.m64.fm.0e776646b137 pre64-fm`.
   Expected: the stored schedule `w2x2, e2m1/f1x2/k8` on the projection piece, about 50 us whole target vs
   torch.compile about 180 us. Before the strict-rule change it failed with `strict eager correctness failed:
   output 'reshape_3' ...` (see Bugs).
2. post-sym:
   `SCHED="$(cat ~/claude/tune/run1/sched-postsym.txt)" ~/claude/tune/run1/rec2.sh
   post-sym-dense-full@nvfp4.k_linear_reduce_ee6ca2.dynamic.fm.40467afbe9f4 postsym-fm`.
   Expected: the gate/up piece `__place_159c0d8622` runs the FP4 tile instead of the prior's 4.4 ms pick
   (torch.compile about 1170 us for the target).
3. pre-sym:
   `SCHED="$(cat ~/claude/tune/run1/sched-presym.txt)" ~/claude/tune/run1/rec2.sh
   pre-sym-dense-full@nvfp4.k_linear_mean_reduce_260a91.dynamic.fm.4dc0d982c2e1 presym-fm`.
   Expected: the projection piece `__place_1ae0859772` runs the FP4 tile instead of scalar `t512 coop` at 40.3 ms
   (torch.compile about 420 us). If either tuned schedule is still clearly slow, run `EMMY_FAST_MATH=1
   EMMY_TUNE_DB=~/claude/tune/run1/fm.db EMMY_KNOBS=PLACE=fuse emmy run --golden ~/claude/tune/run1/W.json --kernel
   <piece ref> --bench --tune 10 --warmup 5 --iters 20`, then step 2 or 3 again without `SCHED` (but see the
   `--kernel` FP4 gotcha below).
4. gdn16 `db4f79`:
   `KNOBS="$(cat ~/claude/tune/run1/knobs-db4f79.txt)" ~/claude/tune/run1/rec.sh fm
   gdn16-dense-linear@nvfp4.k_linear_matmul_mean_reduce_db4f79.m16.fm.b80a4d98af94 gdn16-db4f79-fm`.
   Expected: 16 kernels, 212-215 us whole target. Before the change it failed with `failed exact-pin integrity:
   ... strict same-input-greedy correctness failed: output 'view_16_static_fp4_scale_bits'` (one code).
5. gdn64 `5f1430`:
   `KNOBS="$(cat ~/claude/tune/run1/knobs-5f1430.txt)" ~/claude/tune/run1/rec.sh fm
   gdn64-dense-linear@nvfp4.k_conv1d_linear_mean_reduce_5f1430.m64.fm.f8d147aa0e10 gdn64-5f1430-fm`.
   Expected: 12 kernels, about 345 us vs torch.compile 103 us (a tuning target). Before the change it failed with
   `strict eager correctness failed: output 'type_as'` (one bf16 ULP).
6. Audit: `~/claude/tune/run1/audit.sh a5`. It runs on the CPU in the serving venv (`~/claude/venv-serving`,
   transformers 5.14.1), so it needs no lock. Goal: `FAST_MATH=true: 9 twin(s) deploy from the golden rows
   alone, 0 do not`. A twin that still fails names its refused fork; its `DIAG` lines say which stored kernel and
   rows that fork maps to. Record that target the same way, then audit again. The `FAST_MATH=false` (STD) line comes
   second.
7. Check the new rows' pins. `--record-greedy` spells all four precision gates. The audit's `_lane` reads both
   spellings as one lane, but rewrite them to `{"FAST_MATH": true}` before promotion, as #1172 did.
8. Strict boot of W, under the lock (strict boot takes 541-900 s):
   ```bash
   . ~/claude/env.sh; cd ~/claude/work-tune; S=~/claude/tune/serve; mkdir -p $S
   export PYTHONPATH=$PWD EMMY_GEN_DECODE_BUCKET=16 EMMY_GEN_PREFILL_BUCKET=64 EMMY_GEN_M1_TIER=0 EMMY_FAST_MATH=1 \
     EMMY_TUNE_DB=$S/serve.db EMMY_CUBIN_CACHE=$S/cubin
   unset EMMY_NVCC_FLAGS EMMY_COLD_CACHE EMMY_KNOBS EMMY_PACK_DIR
   gpu_run bash -c '~/claude/venv-serving/bin/emmy serve Inferact/Qwen3.8-27B-NVFP4 \
     --revision 6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 --golden ~/claude/tune/run1/W.json --strict-evidence \
     --dtype bfloat16 --max-num-seqs 1 --max-model-len 4096 --max-num-batched-tokens 64 --no-enable-prefix-caching \
     --host 127.0.0.1 --port 8080 > ~/claude/tune/serve/server.log 2>&1 & s=$!
     until curl -sf http://127.0.0.1:8080/health >/dev/null; do kill -0 $s || exit 1; sleep 5; done
     ~/claude/venv-serving/bin/python ~/claude/restore/box-24700/claude/serve/probe.py ~/claude/tune/serve/results.json 2
     kill -TERM $s; wait $s'
   ```
   `probe.py` reads `short.txt`, `medium.txt` and `long.txt` from its own directory and writes decode tok/s and TTFT
   per prompt into the results file. Report all three prompts. This is a boot check, not a tuning claim. For
   comparison: main's strict boot on 2026-10-16 gave 31.7 tok/s, and stock vLLM gives 63 tok/s.

### Failure signatures and fixes

- `backend comparison did not use CUDA graph capture; deployed Emmy timing was not captured`: the whole-target bench
  fell back from graph capture now and then, cause not found. Rerun the record (`retry.sh`; `rec2.sh` retries by
  itself).
- `--pin-route` on a target root's own row that has no `latency` pins nothing. The fused kernel then benches, and a
  hung kernel shows up as `HungKernelError ... did not complete within 2000 ms` on the root name. Name one of the
  route's piece rows as `--realization` (the realizations above already do).
- `PLACE pin ... names no seam of any kernel in this compile`: a hand pin spelled along another target's route.
  `GoldenFile.path_to` takes the first route that reaches a piece, and gdn16 `db4f79` shares pieces with gdn64
  `02d4dd`. Spell the pins with `route_knobs.py W ROOT`, which walks only ROOT's own decisions.
- A piece with one CTA (grid 1 in the bench table, no knobs) and a multi-ms or hung time: its split decision is gone.
  List its seams with `seams_of.py`, then cut them with `PLACE@place_<token>/<seam>=cut` (as in `knobs-*.txt`).
- `deploy: node ... has N measured DB row(s) ... but none matches any of the offered candidates`: the prior then
  picks; check that the pick in the bench table is sane.
- `pkill -f <pattern>` inside `ssh ... '<command>'` also matches the ssh command line and kills the ssh session
  (exit 255). Kill a job by its tmux session or by PID.
- `run --kernel <cut FP4 piece>` fills the piece's FP4 decode tables (`*_f4_pairs`) with random values, so tile
  rows fail correctness there and `--tune` counts every tile row as a failure. Check candidates through the whole
  target (`--realization <target row> --pin-route --strict`).
- `benchmark run stage exceeded 10.0s of GPU time`: a multi-ms pick in the pinned or greedy set. Replace the pick
  (`rec2.sh` with `SCHED`), or for a record-only run use `--warmup 2 --iters 5`.
- `unreproducible pin: ... realized ...`: a hand schedule pin clashes with a pinned golden row of the same
  realization. The run's own greedy result still counts. Drop the hand pin before the strict record (`rec2.sh`
  does).

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

## Findings

- Search policy: the prior or the greedy fallback can pick a single-CTA `Tile[]` arm over a large contraction (gdn64
  `k_linear_matmul_mean_reduce_73983d__place_68111979ee`: one thread for a 64x6144x5120 linear plus the recurrence,
  38.9 s per call; gdn16 `d29945__place_54138af7d7`, same shape). A non-strict boot then takes hours. No guard
  rejects such arms or prices them by work (about 20-50 lines plus a test; not done).
- Boot audit: a stuck graph-capture sync shows up as a hang in the next program (gdn1 waited behind about 31 min of
  queued gdn64 work). Main's #1204 runs the first pass under the hung-kernel watchdog and names the program.
- `roofline.py` `time_program_us` reports a single cold first run when that run exceeds 10x the floor. In the full
  64-layer boot pre16 and post16 showed 4.26 and 3.88 ms against 44-51 and 134 us steady state. The cause
  (first touch or placement in the full boot) is not investigated.
- Compiler bug (wrong answer, not fixed): the bf16 MMA schedule `WORK=w1x4, TILE=mma_m16n8k16_bf16_f32/f1x1/k8,
  STAGE=d4/smem-tma` (192 CTAs, 48 KiB shared memory) on gdn16 `db4f79` piece `place_6a3e623a98` gives one wrong FP4
  code (flat index 86480 of `view_16_static_fp4_bits`: -6 vs -3) and its block scale. Changing any single knob gives
  output bit-identical to the scalar schedule. The prior picks this schedule greedily; W pins `TILE=f1x2/k8` there
  instead (15.2 us). The non-strict wrong-answer check misses it: its quantized threshold is a mean error above 0.5%
  of the peak, and one bad element in 98304 stays under it. Repro: box `~/claude/strict/r2/` (`wa16.sh`).
- Rows of neighbouring schedules that passed strict and are worth re-checking: gdn64 `...5f1430.m64.8105a48e7b21`
  (`w1x8`, bf16 `f1x1/k8`, `d4/smem-tma`), pre16 `...a7efbb.m16.fm.83c0b7f74b51` (FP4 tile, `w1x4`, `d4/smem-tma`),
  gdn16 `0e5142` `...8105a48e7b21` (`d3/smem-tma`).
- `--cold-cache` bench times come in steps of about 2.05 us (69.6, 71.6, 73.7, 75.7 ...), too coarse to rank decode
  matmul schedules. Cold A/B runs go under nsys and read per-launch durations instead (decode-matmul scripts
  `abn.sh`, `nsys_variants.py`; laptop `tmp-claude/tune-5090/decode-matmul/`).
- Hot benches of weight readers keep the weights in the 96 MiB L2 of the 5090 (gate_up, 100 MB: 25 us hot, about
  4 TB/s; 68 us cold). A hot `--tune` can therefore rank schedules that gain nothing cold (post16 gate_up hot winner
  `w1x2 f1x4/k4`: -10 us hot, 0 cold). Decode matmul rows in W carry only schedules that also win cold.
- FP4 decode weight readers reach about 1.30-1.47 TB/s cold (gate_up 68 us for 100 MB, q 27 us for 35 MB), against
  1.60-1.64 TB/s for a bf16 cuBLAS GEMV. No offered WORK/TILE/STAGE schedule closes the gap; it looks like a fixed
  ramp and tail of about 5 us per launch.
- One fused GDN in_proj kernel (qkv, z, a/b in one 168 MB read) is out of reach: the fuse arm
  (`PLACE@place_f2a9222bb8/map.{1,2,3}/inner=fuse`) forms three sequential contractions in one `Tile[]` (N=1, one
  CTA). It needs horizontal fusion (one grid over the concatenated N=16480, or block-range dispatch). cuBLAS gives
  102.6 us for one GEMV against 112.6 us for three, so the gain is about 0.4-0.5 ms per token.

- Strict rule, known loose spot: a float output's rounding step comes from the coarsest float format its producing
  chain computes in (`0f637e57`), and the rule has no list of which ops round and which only move data. So a bf16
  slice cast to f32 and followed by f32-only math still gets the bf16 step, which is looser than the f32 math needs.
  The mean-error rule still applies to such outputs.
- Graph-capture fallback of whole-target benches: the capture that fails is the eager PyTorch backend's
  (`CUDA graph capture failed for Eager PyTorch (... cudaErrorStreamCaptureInvalidated)`, seen with `DIAG_LOG` in the
  bench worker, box `run1/logs/retry-g64r-1.diag`); emmy then re-benches every backend uncaptured and the strict
  record refuses. A rerun usually captures (`run1/retry-diag.sh`).

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

- After the strict-rule merge (215bbc72): pre64 recorded (D2, 50.6 us whole target vs torch.compile 177 us; four
  piece rows measured, pins `{"FAST_MATH": true}`, one routing row). D3 refused on the graph-capture fallback; the D2
  record stands.
- post-sym and pre-sym first refused through the exact-pin check: a strict record also benches the named
  realization's own stored row as a pinned config (`golden_configs`, `emmy/commands/compile.py`), and that row's
  schedule knobs replay as bare, unscoped pins on every kernel of the compile (`_sample_replay_knobs` and
  `pinned_knobs` in `emmy/commands/run.py`). The realizations named first were rows of small pieces:
  - post-sym `...40467afbe9f4` (`WORK=, REDUCE=`): on the gate/up piece `__place_159c0d8622` those pins leave only
    `WORK=""` candidates, none matches the measured w4x1 row, and strict evidence refuses ("2 measured DB row(s) ...
    none matches any of the 8192 offered candidates"). Without those pins the compile offers the schedule and the
    measured pick takes it (checked on the CPU with a hook on `greedy._db_measured_pick`).
  - pre-sym `...4dc0d982c2e1` (`WORK=, TILE=f4`): the same pins on the projection piece give a schedule that
    exceeds the 10 s GPU bench budget.
  Fix: name the realization by the big piece's own row, whose schedule is the one to deploy. Both recorded that way:
  - pre-sym via `...260a91.dynamic.fm.80c06af6e052` (w4x1 `e2m1/f1x2/k4, d2/smem-async` on the projection piece):
    D3 recorded, 484 us whole target vs torch.compile 411 us (projection piece 460-467 us; a tuning target).
  - post-sym via `...ee6ca2.dynamic.fm.30fb6a0ce3ab.tuned-native034` (the #1120 row on the gate/up piece): D2 and D3
    recorded, 863-865 us whole target vs torch.compile 1173-1175 us (gate/up piece 772-786 us).
  Gotcha for any record: name a row whose knobs suit every piece of the route, because they pin every piece.

- gdn16 `db4f79` recorded (D2 and D3, 204 us whole target; torch.compile has no time for this root, so the seed row
  keeps no latency). Pins: `knobs-db4f79.txt` plus `WORK@place_6a3e623a98=w1x4,
  TILE@place_6a3e623a98=mma_m16n8k16_bf16_f32/f1x2/k8, STAGE@place_6a3e623a98=d4/smem-tma` on all three steps (box
  script `rec-db4f79-fix.sh`); that piece runs 15.4-15.6 us. The prior's greedy pick there (`f1x1/k8`, `d4/smem-tma`,
  `w1x4`) computes a wrong value (a compiler bug, found by the strict-rule agent); it must not be recorded, and no row
  with it is in W.json. The 6 new rows that spelled all four precision gates were rewritten with `pins_rewrite.py`.

- gdn1 FM tuning merged into W (2026-10-19). The GDN-recurrence run (`~/claude/tune/gdnrec/`, on a copy of W taken
  before gdn16 `db4f79` was recorded) tuned the seven slow pieces of `k_conv1d_linear_mean_reduce_970a92` with
  `--tune 40` and recorded them strictly: 102.3 -> 21.1 us per layer for those pieces, 190 -> 103 us whole target.
  Merged as a diff against its start copy (script `~/claude/tune/gdnrec/merge_gdn1.py`; W's gdn16/gdn64 rows
  untouched): 6 new FM rows (`74f91b727368`, `2c6f6570cde2`, `b45a65917987`, `bdf21fa2a230`, `1ee9c517392c`,
  `c423591da0e6`), the proposal `803fd6b8d8f1` now measured, 7 FM rows re-measured. Dropped: 11 slower FM rows of the
  same kernels and pins (one measured row per kernel and pins), among them the `f1x2` rows of `2a032d7af1#2` and
  `fcd8ee3ca5#2` and the `t4 coop` row of `cc0f560fe6`; and the record's second routing row for `970a92`, which
  spells the stored arm minus five alias seams (the stored 19-key row stays). Backup: `run1/W-before-gdn1-merge.json`.
  `emmy golden check`: current (215 kernels). Audit `gdn1m`: FM 8 of 9 (only gdn64 refuses), STD 1 of 9, as before.
  Strict whole target on W, golden rows only: 107 us (`logs/whole-gdn1m.log`, script `run1/whole-gdn1.sh`; the 4 us
  over 103 is the `1af9d13697` matmul piece, 50.2 vs 46.7 us, same pick). The target row's stored latency (217 us FM)
  is stale; a `--record` run would refresh it. The 7 freeze files of those tune runs are in
  `emmy/compiler/pipeline/search/freezes/`.

- Decode matmul FM tuning merged into W (2026-10-19). The decode-matmul agent (`~/claude/tune/mm/`, on a copy of W
  taken before the gdn1 merge) tuned five FP4 weight readers with a hot `--tune 40` and then 9 cold A/B runs per kernel
  (cold times from nsys, see Findings). It recorded only schedules that also win with a cold cache: a hot bench of a
  weight reader reads its weights from the 96 MiB L2, so hot winners can gain nothing in serving. New FM rows (merged
  as a diff, script `~/claude/tune/mm/merge_rows.py`; `tried` set to 40, the schedules the hot search measured, since
  each record ran on its own empty tune DB):
  gdn1 gate_up `k_linear_reduce_139b4a.fm.2e518ef89d7f.tuned-ddc2eb777b09` (`w1x4, e2m1/f1x4/k4, d2/smem-tma`, 25.6 us
  hot, 68.6 us cold); post16 gate_up `...bcbff6.m16.fm.b9a2ae642f62.tuned-ddc2eb777b09` (same schedule, 26.3 us);
  pre16 `...a7efbb.m16.fm.83c0b7f74b51.tuned-9b40cdeaf62b` (`w1x4, e2m1/f1x4/k8, d2/smem-tma`, 16.1 us); post16
  `...abb89f.m16.fm.c7eaa1f33b03.tuned-bfaaca0312e1` (`w1x4, e2m1/f1x2/k8, d3/smem-async`, 21.5 us); post16
  `...5cd5f4.m16.fm.6ba125ea160f.tuned-7602e8eac7c0` (`w1x4, e2m1/f1x2/k8, d2/smem-async`, 9.4 us). Dropped: 7 slower
  FM rows of the same kernels and pins (the base rows those names extend, plus `...bcbff6...fc5631774a92` and the
  `t512 coop` row of `abb89f`); proposal rows stay. Backup: `run1/W-before-mm-merge.json`. The 7 freeze files of those
  tune runs are in `emmy/compiler/pipeline/search/freezes/`.
  Checks: `emmy golden check` current (215 kernels); audit `mm`: FM 8 of 9 (only gdn64 refuses), STD 1 of 9; strict
  CPU compiles of programs 6 (gdn1), 1 (post16) and 4 (pre16), FM lane, empty tune DB, pick all five schedules
  (read from the `08_tile_schedule.kernels.txt` dump, `--dump-dir`).

- Rows kept beside faster ones (2026-10-19): the 18 slower FM rows the gdn1 and decode matmul merges dropped are back
  in W (script `~/claude/tune/mm/readd.py`, backup `run1/W-before-readd.json`). Rule followed: the canonical golden's
  practice, not a repo rule (none says slower measured rows go). The canonical file holds several measured rows for
  89 of its 253 measured kernel+pins groups (49 of them under one row name), and all 54 `.tuned-*` rows keep their base
  row. A strict compile picks the fastest matching row either way. Routing rows stay one per arm: the canonical file
  has no re-spelled duplicate arm, so the second 970a92 row stays dropped.
  Checks after the re-add: `emmy golden check` current (215 kernels); audit `readd` FM 8 of 9 (only gdn64 refuses),
  STD 1 of 9; strict CPU compiles of programs 6, 1 and 4 (FM, empty tune DB, `run1/picks.sh`) still pick the five
  decode matmul schedules.

- FM lane 9 of 9 (2026-10-19), after the strict-rule change (`0f637e57`, merged as `e5d14864`):
  - gdn64 `5f1430` recorded with `knobs-5f1430.txt` (`rec.sh`, realization `...5f1430.m64.fm.f8d147aa0e10`): D2 and
    D3 fell back from graph capture; the second `retry-diag.sh` attempt recorded (344 us whole target vs
    torch.compile 103 us and eager 275 us; the `w1x2, bf16 f4x4/k8, d2/smem-tma` piece `__place_526ad89b7f` is 336 us
    of it, a tuning target). 7 new rows, pins rewritten to `{"FAST_MATH": true}`.
  - The audit then stopped at gdn64's matmul root (`k_linear_matmul_mean_reduce_02d4dd`, fresh name `..._73983d`): its
    producer piece (`__place_68111979ee`, stored `__place_683a8f4894`) ran as one CTA (the 38.9 s/call pick in
    Findings) with no measured row. Box `knobs-02d4dd.txt` = `route_knobs.py W k_linear_matmul_mean_reduce_02d4dd`
    plus the producer split `PLACE@place_68111979ee/{map.1/reduce, map.2/inner, map.3/reduce}=cut` (the same three
    seams as gdn16's `place_54138af7d7`). `rec.sh` with realization `k_linear_matmul_mean_reduce_02d4dd.m64.fm.54df000e1fa2`:
    D1 299 us, D2 recorded 291 us whole target (no torch.compile time for this root). D3 refused only because the
    short realization name now matches 4 rows (D2's new rows extend it); the D2 record stands. 3 new rows, pins
    rewritten.
  - `emmy golden check`: current (219 kernels). Audit `g64b`: `FAST_MATH=true: 9 twin(s) deploy from the golden rows
    alone, 0 do not`; STD 1 of 9. Backups: `run1/W-before-gdn64-5f1430.json`, `run1/W-before-gdn64-02d4dd.json`.

## Decisions

- Trace reference: transformers 5.14.1 (see Open gaps).
