# Attention: findings and what to try next (memo, revised 2026-09-10 after PR #772)

## Where things stand

A100 40GB, the Neptune paper's card, experiment lane of 2026-09-10 (`compiler_neptune_emmy_pytorch_a100`, no golden
retuned), geometric means over 256-32768 keys, lower favors Emmy:

| family | Emmy / eager FA-2 | Emmy / Neptune (wins) | before PR #772 |
| --- | ---: | ---: | ---: |
| prefill causal | 1.02x | 0.87x (7/8) | 1.41x / 1.20x |
| prefill GQA | 1.03x | 0.88x (7/8) | 1.44x / 1.23x |
| prefill global | 1.17x | 0.94x (6/8) | 1.54x / 1.23x |
| decode GQA | 0.04x | 0.81x (8/8) | 0.05x / 0.93x |
| decode causal (256-2048 re-recorded; see the eager-reference caveat) | 1.29x | 0.94x | was 3.09x / 3.26x |

RTX 5090: the cp.async causal row at 32 heads / 2048 keys / hd 128 runs 1.13x faster than eager (242 vs 274 us), at
16 heads 1.28x; the article's one-wave shape (16 heads, 512 keys, hd 256, causal) is 0.92x of eager (38 vs 35 us).

## Findings from PR #772

- **The chunk loop was issue-bound, and two things made it so.** sm_80 SASS of the recorded causal row (`f1x16/k8`
  + `f1x16/k4`, `d1/smem-async`, `w4x1`): 2361 instructions per 128-key chunk per warp for 256 HMMA, 835 of them
  integer address math, 255 registers with spills. The swizzle re-applied per load was the address math: the XOR is
  linear over bit-disjoint parts, so `swz(lane + base) = (swz(lane) ^ swz(col)) + row·ldm` and the lane's part is one
  hoisted value (`swizzled_slab_index`; 2361 → 1728, spills gone). The chunk-local P·V fragment was the rest: folding
  the chunk at the advanced pivot lets the mma accumulate into the carrier (1728 → 1640, 237 registers). The
  measuring loop: `emmy compile … --target sm_80 --ir cuda`, strip the `=== ` header, `nvcc --cubin -arch=sm_80 -O3`,
  `cuobjdump -sass`, count opcodes in the largest backward-branch loop. Ten seconds per iteration, no A100 needed.
- **The geometry is still right.** Pin sweep on the A100 at 32 heads / 2048 keys after the change: fast math buys 1%,
  `f1x16/k4` + `f1x8/k8` loses 8%, `f2x16/k2` and `/k4` (32 rows per warp, FA-2's block) still spill at 255 registers
  and lose 40-50%, `d2/smem-async` rings lose 13-16%. The wider warp tile is the lever left (below), and it is a
  register problem: the hoisted query fragments cost 64 registers at `f2`, on top of O (128) and the score chunk.
- **A descending causal launch order (longest row blocks first) measured 0-2%** on the A100 at 2048 and 1024 keys and
  on the 5090, inside the run-to-run spread; dropped. The A100 VM's eager reference itself moves up to 25% between
  processes: compare same-process Emmy/eager ratios only.
- **The key-range split on a CAUSAL stream returns wrong answers, on main too.** `_sliced_contraction`
  (`lowering/tile/_split.py`) σ-reindexes the carrier's operands to absolute keys but not its own lift body, where
  the coordinate masks live, so every partition past the first masks slice-local keys as absolute and the early stop
  bounds the slice the same way. Substituting the lift body alone turns the partition coordinate into a free read
  of the score's prefix cone, which the chunk tier's gate refuses ("STAGE pin … does not resolve"): the sliced lift
  must bind it, the way `_sliced_edge` extends a computed edge's params. The GQA decode goldens are non-causal and
  unaffected.
- **`WORK=+p1` on the chunk tier double-arrived** because the tier runs every warp through the uniform ring
  (`pipelined_kloop` takes no `workers`); the aux band decoded onto warp 0's ids and re-issued the elected TMA
  arrive. A chunked carrier is no longer eligible for a producer band (`classic.py`, `producer_eligible`).
- **Global prefill's goldens pin a weaker row** (`f1x16/k4` + `f1x8/k4`, key unstaged, 256-16384 keys). The causal
  geometry measured 497.7 vs 555.5 us at 2048 keys (eager 455) but 50.4 vs 44.8 at 512, so its retune is per length.
- **Two lane artifacts, not deploy issues.** A decode setup's replay directory holds one record per golden row, and
  the finalize's receipt and the routing row replay the partial on a scalar tile (hundreds of microseconds); the
  full kernel-set receipt is the number, as in the previous archive. The two `op-g` rows (global, GQA prefill) share
  one task directory name, so the second row's artifacts carry the first's evidence files.

## What to try next, by expected paper impact over effort

### 1. Global prefill: retune the A100 goldens per length

Sweep each length (256-32768) over the causal geometry, the recorded row, `d2/smem-async` and `FAST_MATH`, and
record the winner with `--record-greedy` into a traced working golden (drop the seed row; the lane needs exactly
one measured row per file). Expected: the family joins causal and GQA at parity with FA-2. Effort: an afternoon on
the A100, no compiler change.

### 2. The softmax arithmetic: FA-2's one FFMA and one MUFU per score

The loop is 1640 instructions for 256 HMMA; about 800 are still float ops on 64 scores per lane: `expf`'s range
reduction (~5 per score), the scale multiply, the `+ zero` the mask's keep branch adds, the `− pivot` subtraction,
and 66 ISETP for the mask on every chunk. FA-2 does `exp2(s · scale·log2e − m · scale·log2e)`: one FFMA, one MUFU.
Three steps, each measurable with the SASS loop above:

- Inline the SDPA scale and mask constants as literals. The decomposition (`010_sdpa.py`) makes them f16 constant
  buffers, so the render sees `x + in2` and cannot drop an additive identity; the render already has a
  `literal_constants` path that production never feeds. With the zero a literal, the `FragmentMask` keep branch of an
  identity is nothing (−64 FADD).
- Fold the scale into the exponent. With the scale a known positive literal, the row max commutes with it: reduce
  the raw score, keep the pivot in raw units and instantiate the pattern as one FFMA against `scale·log2e` and
  `exp2` (the `FAST_MATH` intrinsic, which measured 1% alone — the win is the folding). Roughly −300 instructions,
  which is where the loop stops being issue-bound with margin. This is a reading of the prefix cone (a positive
  uniform product through a `maximum`), so it belongs in the chunk tier's residence, not in the recipe.
- Then confine the per-element mask to the chunks that can hold a masked element (FA-2's guard on the chunk's and
  the block's extreme coordinates). It measured nothing on the old kernel because everything else was the
  bottleneck; re-measure after the two steps above.

### 3. Beyond FA-2: the 32-row warp tile

At 16 rows per warp every warp drains the whole K and V chunk, so smem read traffic per row·key is 2x FA-2's and L2
traffic is 2x too (64-row CTAs). `f2x16` fixes both and spills. Two ways to make it fit: re-read the query per chunk
from a slab (FA-2's `sQ`, 32 KB; +25% smem traffic but −64 registers) or from L1 (the pre-hoist form, one gmem
fragment load per K step, cheap at `k2`), and the exp folding above, which shrinks the per-chunk temporaries. Try
`f2x16/k2` first (score chunk 32 registers). Expected: 10-20% over FA-2 on the A100 prefill families.

### 4. Plain decode: the bound row — DONE in part, and here is where it stopped

The lift now binds the elided query coordinate back as an extent-one axis whenever a contraction owns no row
(`lowering/tile/_row.py`), which subsumed and replaced `_implicit_unit_row`. Decode then traces to ONE fused
chunk-tier kernel instead of two per-cell ones. Re-recorded A100 40GB goldens, against the archived
Emmy and Neptune columns, in raw microseconds:

| keys | committed before | Neptune | now | split |
| ---: | ---: | ---: | ---: | --- |
| 256 | 24.7 | 15.2 | **8.8** | none |
| 512 | 38.8 | 19.0 | **15.4** | none |
| 1024 | 73.2 | 27.5 | **18.7** | `g8k` |
| 2048 | 148.5 | 45.5 | **42.6** | `g8k` |
| 4096 | 294.5 | 74.3 | **64.0** | `g16k` |
| 8192 | 607.3 | 122.3 | **117.6** | `g32k` |
| 16384 | 1385.0 | 216.7 | 220.9 | `g64k` |
| 32768 | 2775.0 | 410.3 | 410.8 | `g64k` |

0.84x of Neptune by geometric mean, ahead at six of eight lengths and level at the two longest; 2.5x to 6.8x faster
than the rows replaced. Quote the golden's ROUTING row, not the `TOTAL` line of a `--record-greedy` log: the routing
row is the isolated re-bench and the sum of its receipts (117.6 us at 8192), while `TOTAL` is the in-process figure
measured alongside everything else (143.6). Reading `TOTAL` made 8192-32768 look 1.16-1.25x behind Neptune when they
are level. Read the eager caveat before calling the family on raw microseconds at all: this box's eager reference
does not match the archived lane's.

The reading that mattered was not `_inner_free` or `_node_refusal` — neither is reached. It is `TileOp.contracts`:
with the query coordinate gone, the term's only shared axis is the HEAD, `left_axes` is empty, and a B that moves
with its row is no slab per tile, so the catalog never offers a fragment.

What the measurement says to do next, in order:

- **Registers.** The split row runs at 128 registers and 25% occupancy — the fused row's 254 registers against the
  255 spill wall is gone. Still not memory-bound: 33.5 MB at 2048 keys is a 21.5 us roofline against 42.6 measured.
  Item 2's exp folding is the remaining register lever.
- **Settle the eager reference before claiming the family.** This box measures eager at 31-33 us on the 2048-key
  decode where the archived lane measured 48.1. So 42.6 us beats Neptune's recorded 45.5 on raw microseconds and
  loses to it on the eager-normalized ratio (1.4x against 0.95x). Re-run Neptune on the same box before either
  number goes in a paper.
- **The key-range split is the biggest single lever — and pin BOTH route spellings or the measurement is a lie.**
  A `TILE@map.1/twist` pin does not resolve on a split PIECE, whose tree spells the route `@twist`; the pin is
  ignored without complaint and the compiler picks that piece's geometry itself. Pinning both spellings holds the
  geometry fixed (`f1x8/k8` + `f1x16/k8` realized on every row below) and the answer is then consistent:

  | keys | unsplit | `g8k` | `g16k` |
  | ---: | ---: | ---: | ---: |
  | 1024 | 28 | **20** | refused |
  | 2048 | 64 | **39** | 45 |
  | 4096 | 136 | 69 | **66** |

  1.4x to 2x at every length. Sweeping with only the one spelling produced 137 us at 1024 and 511 at 4096 and
  looked like evidence AGAINST the split; it was measuring the piece's own geometry choice. The `g16k` refusal at
  1024 is honest — 64 keys per partition leaves the staging depth nothing to resolve against.

- **Registers.** The split row runs at 128 registers and 25% occupancy — the fused row's 254 registers against the
  255 spill wall is gone. Still not memory-bound: 33.5 MB at 2048 keys is a 21.5 us roofline against 42.6 measured.
  Item 2's exp folding is the remaining register lever.
- **Settle the eager reference before claiming the family.** This box measures eager at 31-33 us on the 2048-key
  decode where the archived lane measured 48.1. So 42.6 us beats Neptune's recorded 45.5 on raw microseconds and
  loses to it on the eager-normalized ratio (1.4x against 0.95x). Re-run Neptune on the same box before either
  number goes in a paper.
- **A split row is NOT a controlled measurement, and this invalidated two readings before it was noticed.** A
  `TILE@map.1/twist` pin does not resolve on a split PIECE — the piece's tree spells the route `twist`, so the
  compiler picks the piece's geometry itself and the pin is silently ignored. Every split row below therefore varies
  geometry and split together:

  | keys | unsplit | `g4k` | `g8k` | `g16k` |
  | ---: | ---: | ---: | ---: | ---: |
  | 1024 | 28 | — | 137 | 3498 |
  | 2048 | 57 | 192 | **44** | 52 |
  | 4096 | 123 | — | 511 | 335 |

  At 2048 the compiler's own choice for the piece (`f1x4/k4` + `f1x8/k4`, chunk 64 — on the diagonal) beat the
  unsplit row and is what the committed golden records. At 1024 and 4096 its choice was bad. So the recorded 2048
  win is real as a measurement and unexplained as a mechanism: whether the split helps, or whether the piece simply
  landed on a better geometry, is not resolved. To sweep it properly, pin BOTH route spellings (`@map.1/twist` and
  `@twist`) so one of them binds whichever shape the placement takes.
- **The geometry is a matched diagonal, not a cross.** The score tile's column count must equal the carrier's chunk
  width (`16 * k`). Off it — 64/128 or 128/64 — the same kernel measures about 9300 us, 160x worse. Any sweep that
  crosses `TILE@map.1/twist` against `TILE@map.1/twist.1/inner` freely wastes most of its rows.
- **Re-record the other seven lengths.** Only 2048 was re-recorded. A golden cannot take this change by replay:
  `005_replay_lowered` restores stored Tile IR and never reaches the lift, and a replayed row re-applies its own
  recorded pins. Re-recording is `emmy trace --loop-targets -c … -o work.yaml`, then
  `run --golden … --realization … --bench --record-greedy` under the pins.
- **A bug the widened space exposes.** Some candidates abort the bench worker with `scratch buffer … has no
  consuming launch (dead scratch)`. `emmy tune` routes around it by pinning `bench_fail`, but it is a real refusal
  that should not be reachable.

### 5. Fix the causal split, then the one-wave shape

Bind the partition coordinate in the sliced carrier's lift (finding above); a test at (1, 16, 512, 256) causal with
`g2k` against the fused row is the acceptance. Then the one-wave shape: the serial finalize at that shape costs as
much as the fused kernel (37 vs 38 us), so the FA-2 form (O partial plus one LSE scalar per row) needs the per-row
workspace layout the chunk tier's whole-state store makes possible, before the split can pay there.

### 6. Split-KV housekeeping

- Batch 8 at 32768 keys on the 5090 did not record: the greedy worker returned no outputs (the eager reference's
  broadcast is 4.3 GB per operand). A record path that checks accuracy against FA-2 instead of the materialized
  broadcast would fit.
- The finalize is serial per cell; at 64 partials per cell (32768 keys, `g64k`, now offered) it is 8 us of the 140. A
  band over the partials only pays when cells are few; revisit if a split lands on a small-output reduce.
- A pinned compile of a 2048-key attention shape spends about 110 s in the deterministic resolve on the dev box
  (the pinned enumeration under `validate_pins`), which sets the pace of every hand measurement; worth profiling.
