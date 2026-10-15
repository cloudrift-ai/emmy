# Width-1 NVFP4 schedules: MLP down projection and GDN out_proj

Status: RTX 5090 measurements and strict correctness checks are done for both kernels, on main ba00a2c6. Nothing is
recorded in the canonical golden yet: the down projection's rows exist only in a working copy, and the out_proj
record run ended when the GPU dropped off the bus (see "RTX 5090 GPU loss" below).

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

- Down projection: `--record-greedy` under the after row's pins wrote rows
  `gdn1-dense-linear@nvfp4.k_linear_7f95ba.m1[.fm].ab7077b55776.30d0b06d6934` (28.64 µs FM, 28.58 µs STD,
  `tried: 41`) into a working copy of the golden. The same run rewrote other rows' bookkeeping in that copy, so only
  these rows will be promoted, after a diff against the canonical file.
- Main 45e754bc (#1170, #1177) rewrote the canonical golden's latency bookkeeping but left both target kernels'
  Loop IR unchanged, so the recorded down rows still name the right kernel. The next box starts from the current
  canonical file and promotes only those two rows into it.
- Out_proj: a `--record-greedy` on the target row with `@place_3a41271e16` pins is refused: "unreproducible pin:
  STAGE= realized d2/smem-async". Taking the target row as the seed with piece pins does not work. A strict A/B
  run stored the after row as tune-DB evidence; the unpinned `--record-greedy` that followed picked it and failed
  with "unspecified launch failure" as the GPU dropped.

### RTX 5090 GPU loss

At 15:24 box time (uptime 2556 s), during that unpinned out_proj `--record-greedy` (fast-math lane, our job alone on
the GPU under the box's lock), the kernel log shows `NVRM: … GPU lost from the bus [NV_ERR_GPU_IS_LOST]`, GSP RPC
failures and an FSP timeout; a re-attach 16 s later fails with `RmInitAdapter failed! (0x22:0x56:776)`. `nvidia-smi`
then finds no device. The same after row had run cleanly several times just before (A/B, strict and evidence runs
in both lanes). An out-of-bounds kernel usually reports Xid 13 or 31 rather than bus loss, but the timing matches, so
the next box first runs that row alone under `compute-sanitizer`.

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

## Next on the RTX 5090

Bench the after rows above and the twin's row per kernel, cold and warm, in both lanes; tune the down projection
through `--kernel` and out_proj through its target; record the winners with `--record-greedy`; run ncu on before and
after rows; measure strict serving against 29.2 tok/s.
