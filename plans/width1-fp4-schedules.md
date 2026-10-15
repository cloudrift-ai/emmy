# Width-1 NVFP4 schedules: MLP down projection and GDN out_proj

Status: recorded in the canonical golden on main 45e754bc (four rows: both kernels, both lanes); `emmy golden check`
current, `emmy eval golden` 9/9 in both lanes, ncu done, strict serving measured (36.0 decode tok/s against 29.2). The
second box's GPU dropped off the bus right after the serving profile; see "Serving".

Width 1 is the decode width of Qwen3.8-27B-NVFP4: one token per step. Two FP4 linears of its gdn1 program (the GDN
layer at one token) are slow:

- the MLP down projection `k_linear_7f95ba` (serving name `k_linear_7e3c4b`): 5120 outputs, K = 17408, 50.2 MB of
  FP4 weights and scales;
- the GDN out_proj piece `k_linear_mean_reduce_6fee7a__place_154114dfce` (serving name `…71791f__place_3a41271e16`):
  5120 outputs, K = 6144, 17.7 MB.

"Before" below is the schedule the golden records today. "After" is a schedule with the native FP4 tensor-core tile
`mma_m16n8k64_e2m1_f32` (written `e2m1` below).

## Why the recorded schedules are slow

The golden records cooperative scalar reductions: `WORK=t512,REDUCE=coop` for the down projection and
`WORK=t256,REDUCE=coop/r4` for out_proj. They decode every FP4 element in software: byte loads of the code and its
e4m3 block scale, one load from a 512-entry decode table in global memory, and fp8/f16/bf16 conversions before the
multiply. ptxas emits about 52 instructions per element for the down projection and about 43 for out_proj. The down
projection's 89 M elements then cost about 145 M warp instructions, about 89 µs at the RTX 5090's full issue rate
(170 SMs, 4 warp instructions per clock, 2.4 GHz); its bench time is 107 µs. Instruction issue bounds both kernels,
not DRAM, which is why ncu reads them at 22–25% of DRAM peak.

The tile has been legal at width 1 since #1108 and #1133, but the golden never measured it. The 2026-10-12/13 record
run kept one row per kernel, the greedy pick, and the schedule prior ranks the scalar rows first (the first tile row
is rank 9 of 10753 for the down projection, rank 4 of 5701 for out_proj).

## RTX 5080 laptop measurements (not the target card)

The RTX 5080 laptop has the 5090's architecture (sm_120), a 48 MiB L2 and far less bandwidth. Its read rate is
0.57 TB/s, measured with a cuBLAS BF16 GEMV over a 512 MiB matrix and a `sum` over 512 MiB of BF16 (repeat runs move
it between 0.48 and 0.61 TB/s). The read floor is the kernel's bytes at that rate: 88 µs for the down projection,
31 µs for out_proj. "% of floor" is the floor divided by the measured time.

Cold cache (`emmy run --kernel … --bench --cold-cache`: L2 evicted before each launch, so weights come from DRAM as
in serving), fast-math lane, median of three runs of 100 launches, range in brackets:

| Kernel | Before row | After row | Before µs | After µs | Speedup | Before % of floor | After % of floor |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| down | `t512, coop` | `w1x1, e2m1/f1x4/k8, d2/smem-tma` | 383 (358–383) | 86 (75–101) | 4.5× | 23% | 102% |
| out_proj | `t256, coop/r4` | `w4x1, e2m1/f1x1/k8, d3/smem-async` | 121 (117–137) | 41 (39–42) | 3.0× | 26% | 76% |

Other after rows, same runs, against the same before rows:

| Kernel | After row | Before µs | After µs | Speedup | After % of floor |
| --- | --- | ---: | ---: | ---: | ---: |
| down | `w1x1, e2m1/f1x8/k8, d2/smem-tma` | 383 | 88 (84–94) | 4.3× | 100% |
| down | `w1x4, e2m1/f1x1/k8, d2/smem-tma` (the width-16 twin's row) | 383 | 97 (92–114) | 3.9× | 90% |
| out_proj | `w2x1, e2m1/f1x1/k8, d2/smem-async` | 121 | 42 (41–47) | 2.9× | 74% |

Warm bench (no eviction), for comparison only; most weights stay in the 48 MiB L2, so these beat the DRAM floor:

| Kernel | Before µs | After µs (best after row) | Speedup |
| --- | ---: | ---: | ---: |
| down | 318–345 | 40.8 (`w1x1, f1x4/k8`) | about 8× |
| out_proj | 95–106 | 19–20 (`w4x1, f1x1/k8`) | about 5× |

A `--tune 40` on the down projection measured no failing or flagged row. Its best rows keep one warp per block with
4 or 8 fragments along N (`w1x1, f1x4` or `f1x8`) rather than the twin's four warps with one fragment each. The
standard-math lane gives about the same timings.

Caveats: single cold rows vary by up to 30% between runs, the read rate itself moves by ±12%, and the card once
reported a software power-cap throttle. The roughly 4× gap to the before rows is robust; the ranking among the top
after rows is not, and the 5090 decides it. ncu does not run on the laptop (not installed, and the driver restricts
performance counters to root), so "% of floor" comes from cold timings, not DRAM byte counts.

## RTX 5090 measurements

Box with one RTX 5090, CUDA 13.0 (nvcc 13.0.88, as the golden), main ba00a2c6. Read rate: 1.68 TB/s with a cuBLAS
BF16 GEMV over 512 MiB, 1.62 TB/s with a `sum` over 512 MiB. The read floor at 1.68 TB/s is 29.9 µs for the down
projection (50.2 MB) and 10.5 µs for out_proj (17.7 MB). "% of floor" is the floor divided by the measured time.

`emmy run --kernel <name> --bench --warmup 10 --iters 100 --ab <row>`, single runs, both lanes (FM / STD); cold is
`--cold-cache` (L2 evicted before each launch). The 5090's L2 is 96 MiB, so warm times can beat the DRAM floor.

| Kernel | Before row | After row | Lane | Before µs warm / cold | After µs warm / cold | Speedup warm / cold | Before / after % of floor (cold) |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| down | `t512, coop` | `w1x2, e2m1/f1x2/k8, d2/smem-tma` | FM | 103.7 / 120.8 | 29.2 / 38.9 | 3.6× / 3.1× | 25% / 77% |
| down | `t512, coop` | `w1x2, e2m1/f1x2/k8, d2/smem-tma` | STD | 104.1 / 120.5 | 27.9 / 38.6 | 3.7× / 3.1× | 25% / 77% |
| out_proj | `t256, coop/r4` | `w8x1, e2m1/f1x1/k8, d2/smem-async` | FM | 32.8 / 47.1 | 11.8 / 16.4–18.4 | 2.8× / 2.6–2.9× | 22% / 57–64% |
| out_proj | `t256, coop/r4` | `w8x1, e2m1/f1x1/k8, d2/smem-async` | STD | 33.0 / 46.8 | 11.8 / 17.6–18.1 | 2.8× / 2.6–2.7× | 22% / 58–60% |

The down projection's after row is the best of `--tune 40` in the FM lane (28.1 µs warm; no failing or flagged row in
either lane's 40). Close behind, within the run-to-run spread: `w1x1, f1x4/k8, d2` or `d3/smem-tma` (28.9–29.3 µs
warm, 38.9 cold) and the width-16 twin's `w1x4, f1x1/k8, d2/smem-tma` (28.7–29.8 µs warm, 40.9 cold). For out_proj,
`w4x1, f2x1/k8, d2/smem-async` ties the after row (12.0 µs warm, 18.2–18.4 cold); `w4x1, f1x1/k8` rows take
13–14 µs warm. Split-K (`REDUCE=g2k`, a cross-block split with a combine kernel) takes 550–1080 µs. Five out_proj pins
did not realize (`w8x1` with d3 or d4, `w16x1, f1x1/k8`, `w4x1, f2x1/k8, d3`, `w8x1, f2x1/k8`). The out_proj times
come from `--kernel`, whose wrong-answer flags are void for this piece (see the harness section below).

Against the profile's serving times (116 µs and 38 µs per launch, 48 launches each per token), the cold after times
suggest about (116 − 39) × 48 + (38 − 18) × 48 ≈ 4.7 ms per token less; this is an estimate until the serving run.

Strict correctness on the 5090 (`--realization … --bench --strict`; out_proj with `--pin-route` and the schedule
pinned on its piece as `…@place_3a41271e16`): every after row passes in both lanes, with the same errors against FP64
as on the RTX 5080 (down 0.059 / 0.0059 worst / mean against eager PyTorch's 0.091 / 0.0103; out_proj layer output
0.034 / 0.0036 against 0.041 / 0.0062). Inside its target, out_proj's after row takes 19.6 µs (FM) and 20.1 µs (STD)
against the before row's 41.3 and 52.0 µs.

### Recording state

Recorded on the second RTX 5090 box under main 45e754bc, from a working copy of the canonical golden:

| Row | Schedule | Before µs (scalar row) | After µs | `tried` |
| --- | --- | ---: | ---: | ---: |
| `gdn1-dense-linear@nvfp4.k_linear_7f95ba.m1.fm.ab7077b55776.30d0b06d6934` | `w1x2, e2m1/f1x2/k8, d2/smem-tma` | 107.3 | 29.0 | 1 |
| `gdn1-dense-linear@nvfp4.k_linear_7f95ba.m1.ab7077b55776.30d0b06d6934` | same | 106.5 | 28.1 | 1 |
| `gdn1-dense-linear@nvfp4.k_linear_mean_reduce_6fee7a.m1.fm.c36811a256a3.93c9779d7ebf` | `w8x1, e2m1/f1x1/k8, d2/smem-async` | 33.3 | 11.9 | 1 |
| `gdn1-dense-linear@nvfp4.k_linear_mean_reduce_6fee7a.m1.c36811a256a3.93c9779d7ebf` | same | 41.1 | 11.8 | 1 |

- Identity: under 45e754bc both kernels keep their exact identity (down `30d0b06d6934657a6129f821`, out_proj piece
  `93c9779d7ebf…`, computed from the stored Loop IR in the old working copy and in the canonical file). The out_proj
  parent `k_linear_mean_reduce_6fee7a` and two of its other pieces did re-key (#1170 binds the fixed FP32 scalars),
  and the linear piece's token in a compile is now `place_61fdab96bf` (it was `3a41271e16`).
- Down: `--record-greedy` on the target row under the after row's global pins, as before.
- Out_proj: a proposal row (the tile schedule, no measurement) was added beside each scalar row in the working copy,
  then recorded with the skill's isolated route,
  `--kernel k_linear_mean_reduce_6fee7a__place_154114dfce --realization <proposal row> --bench --record-greedy --strict`.
  That route takes the schedule from the named row, so no hand pin clashes with the row's knobs (the cause of the
  earlier "unreproducible pin: STAGE= realized d2/smem-async"). Its reference is the same-input greedy run of the same
  schedule, so the random decode tables of the `--kernel` harness do not flag it.
- `tried: 1`: main's tune-DB version moved since the first box, so the first compile re-created both lane DBs empty
  ("tables written by another emmy — re-created empty"). The 41-row tune of the down projection and the out_proj
  evidence row from the first box were not carried over; `tried` counts only this run's rows.
- Whole-target check from the working copy with an empty tune DB (`--bench-backends eager,tcompile,emmy --strict
  --strict-evidence --no-record-evidence --record`): the greedy pick is the tile row in both lanes. Down target 30.2 µs
  FM / 29.1 STD (torch.compile 117 / 119); out_proj target 19.5 / 19.9 µs (torch.compile 30 / 30). The canonical file
  takes only the four rows above; the whole-target latencies stayed in the working copy.
- The promoted rows' pins were re-spelled by hand; see the follow-up below.

ncu (`--clock-control none`, L2 flushed before each launch by ncu's default cache control), fast-math lane, two
launches each:

| Kernel | Before µs | After µs | Before % of DRAM peak | After % of DRAM peak | DRAM bytes read (both) |
| --- | ---: | ---: | ---: | ---: | ---: |
| down | 116–121 | 37 | 23–25% | 76–77% | 50.2 MB |
| out_proj | 40–41 | 15–16 | 25% | 62–67% | 17.7 MB |

### RTX 5090 GPU loss

At 15:24 box time (uptime 2556 s), during that unpinned out_proj `--record-greedy` (fast-math lane, our job alone on
the GPU under the box's lock), the kernel log shows `NVRM: … GPU lost from the bus [NV_ERR_GPU_IS_LOST]`, GSP RPC
failures and an FSP timeout; a re-attach 16 s later fails with `RmInitAdapter failed! (0x22:0x56:776)`. `nvidia-smi`
then finds no device. The same after row had run cleanly several times just before (A/B, strict and evidence runs
in both lanes). An out-of-bounds kernel usually reports Xid 13 or 31 rather than bus loss, but the timing matches, so
the next box first runs that row alone under `compute-sanitizer`.

Result on the second box: `compute-sanitizer --tool memcheck` over the out_proj target with the after row pinned on
the piece reports `ERROR SUMMARY: 0 errors` in both lanes, and the unpinned strict-evidence run of the same target
that picks the after row from evidence ran cleanly in both lanes. No out-of-bounds access explains the loss.

## RTX 5090 expectation (written before the 5090 measurements)

| Kernel | Before µs (measured) | 5090 read floor µs | After µs (estimate, not measured) |
| --- | ---: | ---: | ---: |
| down | 116 serving, about 107 bench | 30 | about 30–35 |
| out_proj | 38 serving, 33 bench | 10.5 | about 14 |

The floor is the kernel's bytes at 1.69 TB/s, the 5090's measured cuBLAS GEMV read rate. The estimate assumes the
after rows reach the same share of the floor they reached cold on the 5080 (about 100% for the down projection, about
76% for out_proj), and for the down projection at worst the 35 µs its width-16 twin serves at for the same bytes.

## Numerics

The bench's loose check flags a row whose output differs from the before row's output, on the same input, by more
than 5% of the output's peak. The down projection's after rows pass it. Out_proj's after rows, run inside their whole
target, differ by 6.7% of peak and are flagged.

`--strict` compares each row with eager PyTorch at rtol and atol 1e-3, and where they disagree with an FP64
evaluation of the same program. Every after row tried passes in both lanes, and it is closer to FP64 than eager
PyTorch is:

| Kernel, output | After: worst / mean error vs FP64 | Eager PyTorch: worst / mean error vs FP64 |
| --- | ---: | ---: |
| down, `add_10` | 0.059 / 0.0059 | 0.091 / 0.0103 |
| out_proj target, layer output `add_7` | 0.034 / 0.0036 | 0.041 / 0.0062 |

The before rows match eager PyTorch exactly. The traced program rounds every dequantized weight and activation element
to bf16 before the multiply; the tile multiplies the FP4 codes and scales exactly and accumulates in f32. So the 6.7%
flag on out_proj marks the before row's rounding, not an error in the tile.

## `--kernel` fills a cut piece's FP4 decode tables with random values

Reproducer (any card with sm_120, fast-math lane):

```bash
emmy run --golden <copy of the Qwen3.8 golden> --kernel k_linear_mean_reduce_6fee7a__place_154114dfce --bench \
  --ab 'WORK=w4x1,TILE=mma_m16n8k64_e2m1_f32/f1x1/k8,STAGE=d3/smem-async,REDUCE='
```

Expected: the tile row matches the before row to within the bf16 rounding above. Observed: `wrong-answer: rel err
8.971 vs greedy output`, the same 8.971 for every tile schedule. The isolated program of a cut piece takes every
tensor as a graph input, the FP4 decode tables (`*_f4_pairs`, bf16[256,2]) and the per-tensor scales included, and
the bench fills inputs with random normal values. The scalar kernel reads the random tables; the tile hardcodes the
e2m1 values. The down projection, an uncut kernel, keeps its tables and weights as named checkpoint constants, which
the bench synthesizes correctly, so it is not affected.

Consequence: `--kernel` can time an out_proj tile row but cannot vouch for it, and a `--tune` through `--kernel`
counts every tile row as a failure. Check and record out_proj through its target instead:
`--realization gdn1-dense-linear@nvfp4.k_linear_mean_reduce_6fee7a.m1[.fm].c36811a256a3 --pin-route`, with the
schedule pinned on the piece as `WORK@place_3a41271e16=…`, `TILE@place_3a41271e16=…` and so on (the piece's token in
that compile is `3a41271e16`, not the golden ref's `154114dfce`). The fix is a harness change: the isolated program
should keep a cut piece's constant tensors as constants.

## Serving

The 2026-10-13 strict setup (recipe golden at 44986bca, `--strict-evidence`, decode bucket 16, prefill chunk 64, M1
tier 0, fast math, no prefix cache, eager), under nsys with a profiler-API capture range; temperature 0, 256 tokens
max. "Before" is the 2026-10-13/14 measurement on main a951e6c0 (another box, same card model and driver). Main also
moved from a951e6c0 to 45e754bc in between, so not all of the difference is this PR's; no 45e754bc baseline was
measured before the GPU dropped.

| Prompt tokens | Before decode tok/s | After decode tok/s | Before TTFT s | After TTFT s |
| ---: | ---: | ---: | ---: | ---: |
| 25 (warm) | 29.5 | 36.1 | 0.61 | 0.54 |
| 227 | 29.2 | 36.7 (36.0 on a repeat) | 1.05 | 1.25 (0.90 on a repeat) |
| 1027 | 29.2 | 36.0 (35.9 twice more) | 1.19 | 1.15 |

Greedy outputs against the earlier run's token ids: the 25- and 227-token prompts give the same tokens; the
1027-token prompt diverges at output token 12 ("reviewed" becomes "examined"), with the same answer (branches 24,
23, 22 at 1888, 1851, 1814 visitors) and 199 tokens instead of 229. The tile's products are not rounded to bf16 (see
"Numerics"), so a greedy divergence is expected; the decode rate does not depend on it. Boot took 1082 s (before: 541 s, with no
nsys and a warm cubin cache on that box).

Decode profile, 48 steps of the 1027-token request (the 2026-10-09 method, `analyze.py` from the handoff):

| Per step | Before ms | After ms |
| --- | ---: | ---: |
| Step | 34.49 | 27.92 |
| MLP down `k_linear_7e3c4b` (48 launches) | 5.57 (116 µs each) | 1.64 (34.2 µs each) |
| GDN out_proj `…__place_61fdab96bf` (48 launches; before `…71791f__place_3a41271e16`) | 1.82 (37.8 µs each) | 0.66 (13.7 µs each) |
| GPU idle | 1.29 | 1.37 |

The two kernels account for 5.1 of the 6.6 ms the step lost.

GPU loss on the second box: the nsys capture stopped after its 50 iterations at 10:24:17; the engine then failed
with `CUDA error: unspecified launch failure` at 10:24:25, and the kernel log shows `GPU lost from the bus
[NV_ERR_GPU_IS_LOST]` again (`nvidia-smi`: no devices). This was the third VM on this host to lose its GPU. Both losses
on our runs came right after many clean launches of the same rows, the sanitizer is clean, and this one happened in a
serving run, so the kernels are an unlikely cause; the host is the likelier one.

## Follow-up: `--record-greedy` spells every precision gate in a row's pins; `emmy eval golden` matches pins exactly

The down projection's record (`EMMY_KNOBS="PLACE=fuse,WORK=w1x2,TILE=mma_m16n8k64_e2m1_f32/f1x2/k8,STAGE=d2/smem-tma,REDUCE=,LOOPIFY=0,RASTER=,SHARED_CARRY=0"
emmy run --golden W.json --realization gdn1-dense-linear@nvfp4.k_linear_7f95ba.m1.fm.ab7077b55776 --bench
--record-greedy`, with `EMMY_FAST_MATH=1`) wrote its row with
`"pins": {"FAST_MATH": true, "FAST_EXP": true, "F16_MMA_F32_ACC": true, "FP8_MMA": true}`. Every other row of the
golden spells `{"FAST_MATH": true}`. The two spellings are one regime (`regime_pins` agrees: an omitted gate follows
`FAST_MATH`). The out_proj proposal rows were written by hand in the same spelling, and the isolated record kept it.

`emmy eval golden` takes a lane's rows by exact equality (`row.pins == dict(pins)`, pins from the serving config), so
it would have left all four rows out of its audit. Reproducer (CPU only) on the file as first promoted, 85932305:

```
lane FAST_MATH=True: 2 tile rows in the regime, 0 kept by the lane filter; pins as stored: {'FAST_MATH': True, 'FAST_EXP': True, 'F16_MMA_F32_ACC': True, 'FP8_MMA': True}
lane FAST_MATH=False: 2 tile rows in the regime, 0 kept by the lane filter; pins as stored: {'FAST_MATH': False, ...}
```

The audit would still pass on the scalar rows, so it would say nothing about the new rows. Not run on the GPU in that
spelling. Commit 44986bca re-spelled the four rows' pins by hand to `{"FAST_MATH": <lane>}` (checked equal under
`regime_pins`); the same filter then keeps 2 rows per lane, and `emmy eval golden` passes 9/9 in both lanes. The fix
belongs in one place: either the writer stores the umbrella spelling, or the audit compares `regime_pins`.
