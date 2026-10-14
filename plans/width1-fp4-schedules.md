# Width-1 NVFP4 schedules: MLP down projection and GDN out_proj

Status: CPU findings and early measurements on an RTX 5080 laptop. Nothing is recorded in the golden yet; the RTX 5090
measurements and the record run are pending.

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

## RTX 5090 expectation

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
