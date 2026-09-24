# Neptune, modern PyTorch, and Emmy on A100

## Conclusion

The compiler now matches FlashAttention-2 on causal and GQA prefill and is ahead of Neptune on both, on the same
NVIDIA A100-SXM4-40GB the Neptune paper used. Re-measured on 2026-09-10 after the chunk-loop density change (the
staged slab swizzle hoisted per lane, each chunk folded at the advanced pivot), with no golden retuned: causal prefill
is 1.02x of eager and 0.87x of Neptune by geometric mean over the eight lengths, GQA prefill 1.03x and 0.88x, each
ahead of Neptune at seven of eight lengths. Global prefill, whose stream has no early stop, improved from 1.54x to
1.17x of eager and is ahead of Neptune at six of eight lengths (0.94x) but still trails FlashAttention-2. GQA decode,
replayed from the split-KV goldens, is 0.81x of Neptune and ahead at all eight lengths; on the paper's scale the
family's geometric mean over Inductor is 0.27 for Emmy against 0.34 for Neptune. Causal decode, the one family the
compiler could not schedule at all, is now ahead of Neptune too: binding the size-one query row it had lost lets it
reach the fragment tiers, and its eight re-recorded goldens are 0.84x of Neptune by geometric mean, ahead at six of
eight lengths, each shape 2.5x to 6.8x faster than the row it replaces. That column is a manual re-recording rather
than a lane rerun; the section below says what that costs.

The two earlier implementation results still stand under the new numbers: the causal early stop on the chunk tier
(masked chunks skipped rather than folded), which halved the long causal and GQA rows, and the split-KV partial that
stores the chunk tier's row states whole, which is what put GQA decode ahead. The remaining gaps are global prefill's
schedule (its goldens pin a weaker row than causal's; the causal geometry measured 10% faster at 2048 keys and 12%
slower at 512, so the retune is per length) and causal decode's register pressure, which is what keeps its two
longest lengths level with Neptune rather than ahead.

## Prefill and GQA decode after the chunk-loop density change

The three prefill families and GQA decode were re-measured on 2026-09-10 on the same host at revision
`3394fd03cbeac90b76e812875d2812b801c683d6`, after the compiler learned to hoist the staged slab swizzle per lane and
to fold each attention chunk at the advanced pivot, so the chunk's P·V accumulates straight into the carrier. No golden
was retuned: the lane replayed the committed rows, and the difference to the previous section is the compiler's.

The lane ran through `emmy bench --local --filter lane=emmy`, one invocation for `operator=prefill_*` and one for
`operator=decode_gqa`, each setup measured twice at deployable `-O3` with one warmup and 15 captured iterations and
strict eager correctness at `rtol=1e-3, atol=1e-3`. All 32 setups completed both repetitions and every Emmy
measurement passed correctness. Each latency below is the mean over the two repetitions of the minimum of 15; Neptune
is the replay experiment's mean of 15 (`paper-baselines.csv`). For GQA decode the Emmy latency is the replay's kernel
sum for the full kernel-set receipt row, as in the split-KV section below.

| Operator | Emmy / eager | Emmy / `torch.compile` | Emmy / Neptune | Emmy wins vs Neptune | previous Emmy / eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.17x | 1.17x | 0.94x | 6/8 | 1.54x |
| Prefill causal | 1.02x | 1.02x | 0.87x | 7/8 | 1.41x |
| Prefill GQA | 1.03x | 1.03x | 0.88x | 7/8 | 1.44x |
| Decode GQA | 0.04x | 0.34x | 0.81x | 8/8 | 0.05x |

Lower ratios favor Emmy; the summaries are geometric means over the eight lengths, and the previous column is the
section below this one, measured 2026-09-09.

**Prefill causal**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.4 | 23.8 | 24.3 | 26.6 | 1.02 | 0.92 | 1.15 |
| 512 | 58.6 | 58.8 | 58.7 | 54.2 | 1.00 | 1.08 | 1.13 |
| 1024 | 124.7 | 122.4 | 120.3 | 129.3 | 1.02 | 0.96 | 1.42 |
| 2048 | 315.6 | 335.4 | 336.4 | 375.4 | 0.94 | 0.84 | 1.39 |
| 4096 | 1056.3 | 1042.4 | 1040.9 | 1129.8 | 1.01 | 0.93 | 1.50 |
| 8192 | 3438.6 | 3362.8 | 3360.8 | 4389.3 | 1.02 | 0.78 | 1.56 |
| 16384 | 11872.8 | 11354.1 | 11311.1 | 16581.8 | 1.05 | 0.72 | 1.60 |
| 32768 | 47180.8 | 44305.4 | 44499.5 | 61207.6 | 1.06 | 0.77 | 1.63 |

**Prefill GQA**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 36.1 | 36.1 | 36.3 | 35.8 | 1.00 | 1.01 | 1.06 |
| 512 | 69.7 | 70.6 | 70.2 | 81.3 | 0.99 | 0.86 | 1.31 |
| 1024 | 197.8 | 193.1 | 192.3 | 217.8 | 1.02 | 0.91 | 1.41 |
| 2048 | 578.0 | 567.0 | 561.4 | 619.2 | 1.02 | 0.93 | 1.51 |
| 4096 | 2017.8 | 1925.6 | 1922.0 | 2113.7 | 1.05 | 0.95 | 1.56 |
| 8192 | 6015.5 | 5710.8 | 5708.3 | 8522.9 | 1.05 | 0.71 | 1.58 |
| 16384 | 23493.6 | 22295.0 | 22136.3 | 28992.0 | 1.05 | 0.81 | 1.62 |
| 32768 | 93751.8 | 88208.9 | 88154.1 | 100425.6 | 1.06 | 0.93 | 1.60 |

**Prefill global**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 22.9 | 20.1 | 20.0 | 24.1 | 1.14 | 0.95 | 1.31 |
| 512 | 57.7 | 54.0 | 53.6 | 64.9 | 1.07 | 0.89 | 1.30 |
| 1024 | 160.8 | 149.6 | 148.5 | 191.8 | 1.07 | 0.84 | 1.36 |
| 2048 | 556.0 | 454.7 | 452.6 | 551.7 | 1.22 | 1.01 | 1.59 |
| 4096 | 1999.9 | 1721.3 | 1716.2 | 2001.6 | 1.16 | 1.00 | 1.54 |
| 8192 | 6612.5 | 5508.6 | 5492.7 | 7502.6 | 1.20 | 0.88 | 1.61 |
| 16384 | 25008.6 | 20663.3 | 20809.2 | 29564.6 | 1.21 | 0.85 | 1.63 |
| 32768 | 103875.1 | 81058.8 | 84664.8 | 92427.5 | 1.28 | 1.12 | 2.11 |

**Decode GQA** (the replay's kernel sum; the whole-forward capture, eager and Inductor from the same lane)

| keys | Emmy, kernel sum (us) | Emmy, whole (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / Inductor | Emmy / Neptune | previous kernel sum (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 9.7 | 11.4 | 74.8 | 17.2 | 13.1 | 0.57 | 0.74 | 9.2 |
| 512 | 11.1 | 12.5 | 110.0 | 27.5 | 13.6 | 0.40 | 0.81 | 17.5 |
| 1024 | 13.1 | 14.7 | 206.3 | 28.6 | 16.7 | 0.46 | 0.79 | 17.3 |
| 2048 | 14.7 | 16.3 | 392.2 | 41.8 | 19.8 | 0.35 | 0.74 | 18.5 |
| 4096 | 22.3 | 24.3 | 915.5 | 94.7 | 30.2 | 0.24 | 0.74 | 25.2 |
| 8192 | 45.0 | 46.7 | 1509.4 | 131.1 | 52.9 | 0.34 | 0.85 | 42.6 |
| 16384 | 78.7 | 80.7 | 3511.8 | 312.8 | 87.8 | 0.25 | 0.90 | 82.3 |
| 32768 | 137.0 | 141.6 | 6739.5 | 564.7 | 143.2 | 0.24 | 0.96 | 146.5 |

Each decode setup's replay directory holds one record per golden row; the finalize's receipt and the routing row
pin the partial kernel's knobs only through the route and replay it on a scalar tile there (hundreds of
microseconds to milliseconds), exactly as in the previous archive, so the table reads the full kernel-set receipt.
The reference arm's Emmy number is again not reported for decode: without golden evidence the greedy deploys the
prior's pick. The two `op-g` rows (global and GQA prefill) share one task directory name, so the GQA row's artifact
archive also carries the global row's evidence files; each setup file was read once.

The run-to-run spread on this host is the one noted below: eager itself moved by up to 25% between separate
processes at 2048 keys during the development measurements, so the same-process ratios are what the tables rest on,
and short-row ratios within a few percent of 1.00 are ties.

- Prefill run: `2026-09-10_07-23-33`, run ID `20260910T072333Z`; GQA-decode run: `2026-09-10_07-41-21`, run ID
  `20260910T074121Z`
- Git revision: `3394fd03cbeac90b76e812875d2812b801c683d6`; dirty: false
- Host, GPU, driver and toolkit as in "Run and system" below; PyTorch 2.13.0 with CUDA 13.0 in the staged repo's
  venv
- Archive: `results_a100x1.tar.gz` (both run directories with their records, per-row artifacts and logs); SHA-256
  `41115796b3f07a6958b505fa4ffeef922d5c9057a9832fb3a9eed6cf67401872`

## Causal decode after the bound row

Causal decode was the one family the compiler could not schedule. A query of one token per head leaves no axis for
the loop nest to iterate, so normalization inlined that coordinate as a constant and the contraction that remained
shared every coordinate it had with the value it multiplies. `TileOp.contracts` refuses such a term — a B that moves
with the row it is contracted against is no slab per tile — so the family fell to the per-cell tier, where each of
the 128 output channels re-ran the score's own 128-step contraction. No pin in the schedule space avoided that, so
the committed goldens were the best rows of a space that could not express the kernel.

Binding the coordinate back as an extent-one axis gives the term a row, and decode then traces to the same fused
chunk-tier kernel prefill uses. All eight goldens were re-traced and re-recorded on 2026-09-10 on the same
A100-SXM4-40GB host. Five of the eight also take a cross-CTA key-range split, whose width scales to hold roughly 256
keys per partition.

| Sequence | Emmy (us) | Eager (us) | `torch.compile` (us) | Neptune (us) | Emmy / Neptune | Split | Emmy before (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 256 | 8.817 | 22.528 | 19.456 | 15.162 | 0.58x | none | 24.680 |
| 512 | 15.360 | 27.648 | 24.576 | 19.027 | 0.81x | none | 38.841 |
| 1024 | 18.725 | 35.840 | 30.720 | 27.484 | 0.68x | `g8k` | 73.169 |
| 2048 | 42.583 | 48.128 | 41.984 | 45.496 | 0.94x | `g8k` | 148.541 |
| 4096 | 63.982 | 70.144 | 62.976 | 74.266 | 0.86x | `g16k` | 294.468 |
| 8192 | 117.617 | 119.296 | 110.592 | 122.328 | 0.96x | `g32k` | 607.300 |
| 16384 | 220.910 | 218.112 | 203.264 | 216.662 | 1.02x | `g64k` | 1384.960 |
| 32768 | 410.820 | 409.088 | 390.144 | 410.271 | 1.00x | `g64k` | 2775.040 |

By geometric mean over the eight lengths Emmy is 0.84x of Neptune and 0.74x of eager, ahead of Neptune at six of
eight lengths and level with it at the two longest. Every shape is 2.5x to 6.8x faster than the row it replaces. The
family that was 3.5x behind Neptune is now ahead of it.

**How these numbers were produced, and what that costs.** The Emmy column is not a lane rerun. Each value is the
routing row of a re-recorded golden — the kernel-set total from `emmy run --golden … --bench --record-greedy`'s
isolated re-bench, equal to the sum of its per-kernel receipts — measured during a manual pin sweep, not by
`emmy bench` on this recipe. The eager, `torch.compile` and Neptune columns are unchanged from the archived lane. So
every ratio in the table pairs a fresh Emmy measurement against a reference measured in a different session, and the
references are not neutral: this host measured eager at 31-35 us on the 2048-key shape during the sweep, against the
48.128 us the archived lane recorded. Taking that faster eager instead would move the 2048 row from 0.88x of eager to
about 1.3x. Rerunning `emmy bench … --filter lane=emmy --filter operator=decode_causal` would put both halves in one
process and settle it; until that runs, read the Emmy-versus-Neptune column as indicative rather than as a paired
measurement.

**The two schedule facts the sweep established**, both of which make a naive sweep of this family misleading:

- The geometry is a matched diagonal. The score tile's column count must equal the carrier's chunk width (`16 * k`).
  Off it — chunk 64 against 128 columns, or 128 against 64 — the identical kernel measures about 9300 us at 2048
  keys, 160x slower.
- A `TILE@map.1/twist` pin does not resolve on a split PIECE, whose tree spells the route `@twist`. The pin is
  ignored without complaint and the compiler picks that piece's geometry itself. Pinning only the one spelling made
  the split look harmful (137 us at 1024 keys, 511 at 4096); pinning both holds the geometry fixed, and the split is
  then worth 1.4x to 2x at every length.

## Split-KV GQA decode

The eight GQA-decode goldens now pin FlashAttention-2's split-KV: the key range splits across `n` CTAs
(`REDUCE@map.1/twist=g<n>k`, about 256 keys per CTA, `n` = 2 at 256 keys through 32 at 16384 and 32768), the partial
keeps the fused kernel's tensor-core chunk tier and writes its three carrier states to an f32 workspace, and a
serial finalize merges them per cell. This became recordable when the chunk tier learned to store a per-row carried
state whole (the pivot and the denominator broadcast into a fragment beside the expectation); before, the split's
partial fell to scalar tiles and the fused single-wave row ran on eight CTAs. The lane re-ran over the new goldens at
revision `38f7f6d9e` on 2026-09-10 on the same host: all eight setups completed both deployable-O3 replays and
both strict source references. Each Emmy latency is the replay's kernel sum, the mean over the two repetitions of the
minimum of 15.

| keys | Emmy replay, kernel sum (us) | Emmy whole (us) | Inductor, lane (us) | Inductor, paper baseline (us) | Neptune (us) | Emmy / Inductor | Neptune / Inductor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 9.2 | 10.3 | 15.5 | 25.6 | 13.1 | 0.36 | 0.51 |
| 512 | 17.5 | 19.1 | 19.5 | 39.9 | 13.6 | 0.44 | 0.34 |
| 1024 | 17.3 | 18.9 | 26.8 | 41.5 | 16.7 | 0.42 | 0.40 |
| 2048 | 18.5 | 20.0 | 41.9 | 57.3 | 19.8 | 0.32 | 0.35 |
| 4096 | 25.2 | 26.9 | 92.4 | 95.2 | 30.2 | 0.26 | 0.32 |
| 8192 | 42.6 | 43.8 | 132.1 | 164.9 | 52.9 | 0.26 | 0.32 |
| 16384 | 82.3 | 84.4 | 310.3 | 314.9 | 87.8 | 0.26 | 0.28 |
| 32768 | 146.5 | 149.7 | 555.5 | 615.9 | 143.2 | 0.24 | 0.23 |

geometric mean over 8 lengths: Emmy / Inductor = 0.312, Neptune / Inductor = 0.335

Emmy is ahead of Neptune at 5 of the eight lengths and at 0.93x of its time by geometric mean; on the paper's
scale (Neptune's mean of 15 over Inductor's minimum, `paper-baselines.csv`) the family reads 0.31 for Emmy
against 0.34 for Neptune. The lane's own Inductor column is the reference arm's whole-forward capture in the
same process; the paper-baseline column is the earlier replay experiment's, which the paper table normalizes by.
The reference arm's Emmy number is not reported for decode: without golden evidence the greedy deploys the prior's
pick. The 64-way split at 32768 keys recorded well by hand pin but is outside the cut pass's offered widths, so the
unpinned deploy fell through to a scalar loop; the golden pins the 32-way split, which deploys.

The raw lane records are in `results_a10040_split_kv_decode_gqa.tar.gz`; `emmy_decode_gqa_lane.csv` carries the table
as numbers.

## Manually tuned Emmy on A100 40GB

The full run measured revision `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`. A second complete GQA-decode row at
`1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c` replaces its 2048 schedule with the qualified fused schedule. No result
uses `emmy tune`: each golden was selected manually, checked with five warmups and 20 measurements, then accepted only
after two fresh correct measurements.

All 40 shapes completed two deployable-O3 strict golden replays and two source-reference measurements. Every source
run passed eager correctness at `rtol=1e-3, atol=1e-3`. Each latency is the arithmetic mean of the two repetitions;
each repetition reports the minimum of 15 captured GPU measurements. Decode latency sums the realized golden kernels.

| Operator | Emmy / eager | Emmy / `torch.compile` | Emmy / Neptune | Emmy wins vs Neptune |
| --- | ---: | ---: | ---: | ---: |
| Prefill global | 1.54x | 1.54x | 1.23x | 0/8 |
| Prefill causal | 1.41x | 1.42x | 1.20x | 0/8 |
| Prefill GQA | 1.44x | 1.44x | 1.23x | 0/8 |
| Decode causal | 3.12x | 3.47x | 3.54x | 0/8 |
| Decode GQA | 0.05x | 0.42x | 0.93x | 5/8 |

Lower ratios favor Emmy. The corresponding per-shape Emmy latency is:

| Sequence | Prefill global (us) | Prefill causal (us) | Prefill GQA (us) | Decode causal (us) | Decode GQA (us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 26.435 | 28.032 | 38.183 | 24.680 | 9.200 |
| 512 | 70.473 | 66.731 | 93.510 | 38.841 | 17.500 |
| 1024 | 203.366 | 172.459 | 274.432 | 73.169 | 17.300 |
| 2048 | 728.576 | 462.592 | 844.800 | 148.541 | 18.500 |
| 4096 | 2670.592 | 1582.592 | 3007.488 | 294.468 | 25.200 |
| 8192 | 8174.592 | 5226.496 | 8929.280 | 607.300 | 42.600 |
| 16384 | 33263.617 | 17671.679 | 35146.751 | 1384.960 | 82.300 |
| 32768 | 172432.899 | 70041.088 | 137703.423 | 2775.040 | 146.500 |

After the early stop moved from the loop IR to the chunk tier (revision `4598f8f17`), the same goldens replayed on the
same card: causal prefill measured 28.0 us at 256 keys (28.032 above), 61.8 at 512 (66.731), 1233.9 at 4096 (1582.592)
and 4542.5 at 8192 (5226.496); GQA prefill measured 2337.8 at 4096 (3007.488). Repeated replays of the 256-key row spread
from 22.5 to 29.4 us and the 512-key row from 55.0 to 66.9, so the short rows carry a run-to-run spread of about 20% on
this VM. The tables above keep the full-run values.

`paper-emmy-a10040.csv` contains the exact 40 Emmy, eager, Inductor, and Neptune values behind both tables; its
GQA-decode Emmy column carries the split-KV lane values above, which replace the earlier fused and split rows.

## Historical tuned Emmy decode-causal follow-up

The follow-up measured the eight committed decode-causal goldens at revision
`5642d020259d0e09d49cbdab04e8e96408616b3e`. All eight shapes completed two deployable-O3 golden replays and two
strict source-reference invocations: 32/32 required invocations succeeded. Every replay realized exactly the two
expected golden schedules, every source comparison passed, and the largest Emmy absolute error was `2.59e-4` under
the experiment's `1e-3` tolerance.

Each latency below is the arithmetic mean of two independently launched repetitions; each repetition reports the
minimum of 15 captured GPU measurements. Replay latency is the sum of the two golden kernels. The final column compares
the same-input untuned greedy Emmy latency recorded beside each replay, so values above 1.00x favor the tuned schedules.

| Sequence | Tuned replay (us) | Eager (us) | `torch.compile` (us) | Untuned greedy Emmy (us) | Tuned vs greedy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.522 | 13.102 | 12.651 | 73.179 | 2.98x |
| 512 | 38.306 | 17.262 | 16.603 | 149.285 | 3.90x |
| 1024 | 59.048 | 24.064 | 22.357 | 311.296 | 5.27x |
| 2048 | 125.897 | 38.912 | 33.280 | 640.000 | 5.08x |
| 4096 | 274.970 | 57.856 | 53.248 | 1281.024 | 4.66x |
| 8192 | 535.962 | 93.696 | 88.576 | 2551.808 | 4.76x |
| 16384 | 1152.640 | 163.328 | 175.616 | 5081.600 | 4.41x |
| 32768 | 2334.208 | 323.072 | 316.416 | 9318.400 | 3.99x |

Across the eight shapes, tuned replay was 4.32x faster than untuned greedy Emmy by geometric mean. It remained 3.82x
slower than eager and 4.02x slower than `torch.compile`. This row qualifies decode-causal only; it does not qualify the
other four Emmy operator families or change the broader Neptune comparison.

## Historical starter comparison

This table retains the full starter sweep. Its Emmy column describes the original untuned run; the tuned decode-causal
follow-up above is reported separately so historical failures are not rewritten.

The Neptune columns report `Inductor latency / Neptune latency`, so values above 1.00x favor Neptune. "Best" selects
the fastest measured manual or tuned Neptune schedule; "manual" uses only the artifact's fixed manual schedules. Each
summary is the geometric mean over eight sequence lengths from 256 through 32768.

| Operator | Inductor vs eager | Neptune vs Inductor, best | Neptune vs Inductor, manual | Full tunes | Valid Emmy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.02x | 0.84x | 0.82x | 8/8 | 8/8 |
| Prefill causal | 1.05x | 0.90x | 0.89x | 5/8 | 7/8 |
| Prefill GQA | 1.05x | 0.86x | 0.84x | 6/8 | 7/8 |
| Decode causal | 1.01x | 1.07x | 0.99x | 7/8 | 0/8 |
| Decode GQA | 7.67x | 3.01x | 2.90x | 8/8 | 0/8 |

Best-available Neptune beat Inductor on 15 of the 40 individual shapes: two prefill, five decode-causal, and all eight
decode-GQA setups. The decode-GQA speedup increased with context, from 1.68x at sequence 256 to 4.54x at sequence
32768. By contrast, Neptune's best prefill results ranged from 0.76x to 1.19x Inductor across individual shapes.

## Alignment with the Neptune paper

The [Neptune paper](https://arxiv.org/abs/2510.08726) reports the geometric-mean speedup of Neptune over the fastest
manually optimized library in Table 4. To compare with that table, the replay values below use the arithmetic mean of
the 15 projected GPU times for each implementation and then the geometric mean across the eight sequence lengths.
This is the closest aggregation available in the durable Nsight exports to the paper's mean-of-15 kernel rule. Values
above 1.00x favor Neptune.

| Operator | Paper, A100 | A100 40GB replay | Difference |
| --- | ---: | ---: | ---: |
| Prefill global | 0.84x | 0.80x | -5.1% |
| Prefill causal | 0.81x | 0.78x | -4.0% |
| Prefill GQA | 0.80x | 0.79x | -1.3% |
| Decode causal | 0.99x | 1.03x | +3.7% |
| Decode GQA | 1.24x | 1.21x | -2.4% |

The five comparable library results agree within 1.3--5.1%. Both the paper and this exact-card replay find that
optimized libraries beat Neptune on the three common prefill operators, while Neptune is competitive on causal
decode and ahead on GQA decode.

### Reconstructing the normalized paper table

`paper-baselines.csv` extracts the 40 shared shape measurements used by the paper. For each shape, the Neptune value
is the arithmetic mean of 15 projected GPU ranges for each available manual or tuned schedule, followed by selection
of the lower schedule mean. The Inductor value is the minimum of 15 captured, whole-forward CUDA-event measurements.
The per-family reproduced value is the geometric mean of `neptune_mean_us / inductor_min_us` over the eight sequence
lengths. This gives 1.20, 1.14, 1.15, 0.99, and 0.34 for global prefill, causal prefill, GQA prefill, causal decode,
and GQA decode, respectively. Lower values favor Neptune because current Inductor is normalized to one.

`paper-table.csv` records the published library-relative ratios, their artifact replay, and the displayed values after
putting both results on the current Inductor scale. For each operator, the bridge is

```text
paper Neptune / current Inductor
  = replayed Neptune / current Inductor
  * replayed library / Neptune
  / published library / Neptune
```

The Neptune paper publishes only two-decimal family ratios. The final column in `paper-table.csv` therefore records
the displayed paper value directly; recomputing from the rounded intermediate columns can differ by 0.01.

To audit `paper-baselines.csv`, use the sibling `compiler_neptune_replay_a100/results_a10040.tar.gz` archive. Neptune
ranges are under `nsys-stats/<operator>-b1-s<sequence>.csv`; current PyTorch JSON rows are under
`evidence/emmy-tcompile/json/`. Warmup ranges are excluded.

The paper's Table 2 reports Neptune relative to Triton, FlexAttention, TVM, and Mirage, rather than to the manually
optimized libraries. The pinned artifact revision leaves its TVM runners disabled, so this experiment cannot claim a
complete reproduction of every Table 2 cell. Its PyTorch 2.6 runners also select specialized SDPA, cuDNN, or CUTLASS
paths; they are not equivalent to the full-graph PyTorch 2.13 lane added here.

The paper does not publish per-shape latency tables or raw plot data. Its absolute attention results are throughput
plots at sequence length 8192 over varying batch sizes. Representative measurements from the 40GB replay are below,
reported as `Neptune / torch.compile` in microseconds. Neptune is the mean of 15 projected GPU ranges; Inductor is the
mean of two independently launched minimum-of-15 measurements.

| Operator | Sequence 2048 | Sequence 32768 |
| --- | ---: | ---: |
| Prefill global | 551.7 / 453.6 | 92,427.5 / 84,806.1 |
| Prefill causal | 375.4 / 334.8 | 61,207.6 / 45,048.8 |
| Prefill GQA | 619.2 / 563.2 | 100,425.6 / 88,315.4 |
| Decode causal | 45.5 / 38.9 | 410.3 / 395.8 |
| Decode GQA | 19.8 / 57.3 | 143.2 / 615.9 |

The absolute trend is coherent with the ratios: prefill remains close but favors current PyTorch, causal decode
converges toward parity, and Neptune's decode-GQA advantage grows with context length.

## Published artifact coverage

All ten operator families and all eight sequence lengths produced Nsight profiles. Sixty-four of 80 tuning jobs
completed their 128-trial search; 16 reached the 30-minute per-setup limit. Timed-out rows still profile the available
manual and partial tuned schedules and are not described as fully tuned.

The speedup column compares the fastest available Neptune schedule with the fastest valid non-Neptune runner in the
published artifact. Values above 1.00x favor Neptune. The artifact uses PyTorch 2.6, so this table characterizes
Neptune's original comparison environment; the modern Inductor comparison above is the more relevant baseline.

| Published operator | Full tunes | Profiles | Neptune vs fastest valid artifact runner | Validity note |
| --- | ---: | ---: | ---: | --- |
| Prefill global | 8/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton rows at 256–1024 |
| Prefill causal | 5/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton runner |
| Prefill GQA | 6/8 | 8/8 | 0.76x | Excludes the mismatching Tri Dao Triton runner |
| Decode causal | 7/8 | 8/8 | 1.04x | No cross-runner mismatch |
| Decode GQA | 8/8 | 8/8 | 1.21x | No cross-runner mismatch |
| Prefill ALiBi | 6/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Decode ALiBi | 8/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Prefill softcap | 4/8 | 8/8 | 1.04x | Compared with Flex |
| Decode softcap | 8/8 | 8/8 | 4.96x | Compared with Flex |
| Prefill windowed | 4/8 | 8/8 | 0.68x | Compared with CUTLASS |

The harness treats a Neptune manual schedule as its correctness reference. Agreement from the other runners supports
the non-ALiBi rows, but it is not an independent oracle for Neptune. ALiBi is therefore excluded rather than assigning
the disagreement to either side.

SoftCap decode is the remaining performance-reproduction outlier. This replay reports 4.96x over Flex, while the
paper's A100 compiler table reports 1.86x over its best compiler baseline. The modern PyTorch lane does not implement
SoftCap, so this result should not be treated as reproduced until that difference is explained.

## Protocol and limitations

- Neptune ran revision `3aa55c12ac822337e630b809b0d9eabb11eee5d3` in the pinned image
  `evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`.
  The artifact environment recorded PyTorch 2.6.0 with CUDA 12.4 and Nsight Systems 2025.3.1.
- The current Emmy lane reconstructs global, causal, and GQA attention for prefill and decode through
  `emmy run -c ... --bench`. It uses PyTorch 2.13.0 with CUDA 13.0, deployable O3, full-graph Inductor in
  `max-autotune-no-cudagraphs` mode, one warmup, 15 measured iterations, and strict correctness.
- The old starter lane used untuned Emmy and retained its failures. The current table replaces only that Emmy result;
  it does not rewrite the historical run below.
- The paper comparison uses the arithmetic mean of 15 projected Neptune GPU ranges. The PyTorch/Emmy lane uses the
  mean of two independent minimum-of-15 CUDA-event measurements. Both are GPU-time measurements from separate
  processes and software environments, so their ratios are kernel-level evidence rather than an end-to-end result.
- The softcap, ALiBi, and windowed families have no current PyTorch/Emmy twin in this experiment. Their table only
  reproduces the runners shipped in Neptune's artifact.

## Run and system

- Chunk-loop density lane (prefill and GQA decode): 4/4 rows succeeded on 2026-09-10 at revision `3394fd03c`, same
  host as below
- Split-KV GQA-decode lane: 8/8 setups succeeded on 2026-09-10 at revision `38f7f6d9e`, same host as below
- Status: 5/5 full rows succeeded; the corrected GQA-decode row also succeeded
- Full run: `20260909T081524Z`; corrected GQA-decode run: `20260909T093117Z`
- Full-run Git revision: `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`; dirty: false
- Corrected-row Git revision: `1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c`; dirty: false
- Host: `bench-codex-a100-0908-0933-d43f`; Ubuntu 24.04.4 LTS; kernel `6.17.0-1022-gcp`
- CPU: Intel Xeon at 2.20 GHz, x86_64, 12 logical CPUs; memory: 89616363520 bytes
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, UUID `GPU-be299b90-0ff5-e1b4-db53-28465b6f874b`
- NVIDIA driver: `580.173.02`; host NVCC: `12.9.41`; host cuBLAS: `12.9.0.13`
- Docker client/server: `29.8.0` / `29.8.0`

The host was supplied for this work and remains running.

## Historical starter run and system

- Status: succeeded
- Result timestamp: 2026-08-16T00:41:38Z; run ID: `20260816T004138Z`
- Experiment row: `compiler_neptune_emmy_pytorch_a100_recovery/a100x1`; row ID: `e246bb6279fd`
- Git revision: `2550211d9c93e522ea4f9eb81e39735f4ab64d07`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; host NVCC: `12.9.86`; host cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

The source run (`20260815T040818Z`) completed all Neptune work in 71393.66 seconds, then failed because the host lane
started outside the staged repository. The successful 2353.12-second recovery verified the immutable source archive's
SHA-256 (`775fb71d3eac78f0371c1014b9945b29d17f41202347f8115e8703db5a4c14ca`), retained its failed status, and ran
only the missing host lane. The durable `recipe.yaml` contains the corrected working directory for clean future runs.

## Durable files

- Exact A100 40GB comparison: `paper-emmy-a10040.csv` (prefill and GQA-decode Emmy, eager and Inductor columns from
  the 2026-09-10 lane; causal decode's Emmy column re-recorded 2026-09-10 after the bound row, its eager, Inductor
  and Neptune columns still from the 2026-09-09 run)
- Causal-decode bound-row sweep: `results_a10040_decode_causal_bound_row.tar.gz`; SHA-256
  `e92d9b0f49963597c94931cc0ad3edf44f3dbbe37272965177750c264f79d9b5`. Holds the pin-sweep and split-width tables, every
  `--record-greedy` and trace log behind the eight re-recorded goldens, the environment freeze and the revision.
  Logs and tables only — this was a manual sweep, so it carries no experiment record, JSON rows or Nsight profiles,
  and the lane has NOT been rerun for this family.
- 2026-09-10 prefill and GQA-decode lane: `results_a100x1.tar.gz`
- Split-KV GQA-decode lane: `emmy_decode_gqa_lane.csv` and `results_a10040_split_kv_decode_gqa.tar.gz` (per-setup JSON
  records, logs and the status table)
- Neptune paper reconstruction: `paper-baselines.csv` and `paper-table.csv`
- Current system records and composite task artifacts: five rows under `2026-09-09_08-15-24/` and the corrected
  GQA-decode row under `2026-09-09_09-31-17/`, both retained in the raw-results archive
- Starter experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Tuned decode-causal experiment record: `a100x1_lemmy_od-c_9f8816b4a4fb.experiment.yaml`; SHA-256
  `07a4b79cf046bfb16b766bc830974dc42f5cc291c87874c3de7f25d7fb7b81d3`
- Raw-results archive: `results.tar.gz`; SHA-256
  `0871843d4d8eb9232c3260143f7da63322a2978318895a54a570a9f91f6dafc8`
- Archived roots: `2026-08-16_00-41-38/`, `2026-08-24_22-35-24/`, `2026-09-09_08-15-24/`, and
  `2026-09-09_09-31-17/`
- Starter composite task artifact: `a100x1_artifacts.tar.gz`; SHA-256
  `015951d7cccf187c69dd2712bcaf966f3de179b53508942312a0e8e6cc31e4b5`
- Tuned decode-causal composite task artifact: `a100x1_lemmy_od-c_9f8816b4a4fb_artifacts.tar.gz`; SHA-256
  `6c288facacb05cf46b20c4d7be8a6bf56c1495a96ffdbe58c79f41b744412d4b`
- Raw evidence includes 80 `.nsys-rep` profiles, 80 CSV exports, all tune/profile logs, 40 modern PyTorch JSON rows,
  Emmy dumps and logs, all current replay/reference JSON rows, environment freezes, runner hashes, source/recovery
  status files, and all run records/logs.
