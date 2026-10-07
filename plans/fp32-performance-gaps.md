# FP32 performance gaps after the FP32 hardware goldens

Status: open. Found while recording FP32 rows for the V100, A100, H100 and RTX 5090 hardware goldens (2026-10-06).
That work left FP32 attention out of the goldens on purpose, and it left the gaps below unclosed. Every number here is
one RTX 5090 run of `emmy run --bench` unless it names another card.

Goal: an FP32 program compiled by the greedy, with no measurement of its own shape in scope, runs within 10% of eager
PyTorch for the kernel types the hardware goldens cover, attention included.

Done (2026-10-07): both priors refit on FP32 32-row and 512-row Linear rows recorded at K = 2048-6144 on the RTX 5090
and A100, with every computed feature in both priors and a last-wave fill feature. The held-out
`nn.Linear(3584, 3584)` now picks a split K at 32 rows (RTX 5090 21 µs, A100 64 µs; eager 152 / 86 µs). At 512 rows
it still trails eager: RTX 5090 262 µs vs 253, A100 1,061 µs vs 936, where the best swept tiles reach 214 and 870 µs.
The prior prefers a `f4x4` register tile where the best is `f4x12` (RTX 5090) or `f4x6` (A100); more FP32 512-row
evidence at other widths is the next step. Refit again after the attention fix, so the attention rows join the fit.

## 1. FP32 attention

`F.scaled_dot_product_attention` at `(1, 8, 512, 128)`, FP32, no mask:

| Route | Emmy | Eager |
| --- | --- | --- |
| Greedy today (fused, prior schedule) | 6,674 µs | 83 µs |
| Fused, best of a 50-row `--ab` sweep | 3,535 µs | 83 µs |
| Score cut (`PLACE@map.1/twist.1/inner=cut`), one bare schedule for both pieces | ~166 µs | 83 µs |
| Score cut, each piece at its own best schedule (not pinnable from the CLI) | ~150 µs | 83 µs |

The V100 and A100 give the same picture: the fused kernel's best row is 13-90x slower than eager.

### Why the fused kernel is slow

The fused kernel is the online-softmax (flash) form, and the Loop IR is right. The scalar lowering of the twist is not:

- **The score is recomputed for every output column.** The CTA's grid covers `(batch·head, query row, head dim)`.
  Every thread computes the full 128-wide `Q·K` dot product for its query rows, so the score for one `(row, key)` is
  computed once per column tile of the output: 32 times at `TILE=f4x4`, from global memory, with no staging.
- **The softmax statistics are kept per output element.** The running max and sum depend on `(row, key)` only, but the
  twist keeps one copy per output column (`acc1__c0_0 … acc1__c3_3`), so the kernel evaluates two `exp` per
  `(row, key, column)`: 128 times what the softmax needs.
- **The twist's contraction takes no staging.** `STAGE=d2/smem-tma` on either site refuses ("STAGE pin does not resolve
  for this contraction"), so K and V tiles are re-read from global memory by every thread.

### Why the cut route is not 1x either

The score cut gives a tiled `Q·Kᵀ` kernel (25-31 µs, close to an FP32 batched matmul) and a softmax·V kernel that
still carries the statistics per output column — ~120 µs of the ~150 µs.

### Plan

1. **Hoist the twist's statistics out of the free axes they do not depend on.** The max and sum of a softmax twist
   are functions of the reduction's own row; the output column axis (`a6` above) must not replicate them. Keep one
   copy per row in the thread's register tile, shared across its columns, and across the CTA's column threads through
   shared memory. This is a lowering fix in the twist's scalar tier. Expected: the softmax·V piece drops to about one
   FP32 matmul of the same size (~25 µs), the cut route to ~55 µs, under eager.
2. **Stage the twist's operands.** Make `STAGE` resolve for the scalar twist, so K and V tiles go through shared
   memory like a scalar matmul's operands. Needed for the fused route to be competitive at all.
3. **Share the score across the CTA in the fused kernel.** With (1) and (2), the fused kernel still recomputes the
   score per column tile. The FA-2 scalar form computes one `(query tile × key tile)` score block per CTA into shared
   memory, then multiplies it by V. This is the scalar counterpart of the warp tier's P→A repack. Do it only if the
   measured cut route from (1) is not already at eager speed; the cut route may be the right FP32 answer.
4. **Let the CLI pin cut pieces apart.** A bare `TILE`/`WORK` applies to every piece of a cut, and a scoped key on the
   twist piece (`TILE@twist`) does not resolve because the twist is that piece's root. Recording the best route needs
   one schedule per piece. Today the only way is a record receipt per piece.
5. **Record FP32 attention rows on the four cards**, with the cut decision measured (routing rows), plain and causal,
   head width 64 and 128 — the four cases this PR dropped. Then the placement prior learns that FP32 attention cuts.

Do not fix this by adding a fusion gate (AGENTS.md): the fused region stays maximal, the cut and the lowering are the
fix.

## 2. Large FP32 matmuls trail cuBLAS by 10-17%

Every recorded case passes a strict-evidence compile from its card's golden and matches eager. Reductions, softmax,
RMSNorm and pointwise match or beat eager on every card, and the 32-row projections mostly beat it through a split K.
The square matmuls and the 512-row projections do not reach cuBLAS SGEMM. Cases more than 10% slower than eager, from
the strict-evidence replay (µs, Emmy / eager):

| Card | Case | Emmy | Eager | Gap |
| --- | --- | --- | --- | --- |
| V100 | `linear.down.h4096` (and `.dynM`) | 5,172 | 4,491 | +15% (+17%) |
| V100 | `linear.o_proj.h4096` (and `.dynM`, `.m32`) | 1,417 | 1,243 | +14% (+17%, +11%) |
| V100 | `matmul.square.2048` | 1,367 | 1,201 | +14% |
| V100 | `linear.gate_up.h4096.dynM` | 9,155 | 8,275 | +11% |
| A100 | `linear.qkv.h4096` | 3,565 | 3,111 | +15% |
| A100 | `matmul.square.4096`, `linear.gate_up.h4096` (and `.dynM`) | 7,982 | 7,238 | +10% |
| H100 | `matmul.square.4096` | 3,106 | 2,662 | +17% |
| H100 | `matmul.square.2048`, `linear.down.h4096` | 392 | 340 | +15% |
| H100 | `linear.o_proj.h4096` | 388 | 345 | +12% |
| RTX 5090 | `matmul.square.2048` | 289 | 254 | +14% |

The grid each case was swept over: thread tiles `t16x8`, `t16x16`, `t32x8`, `t32x16`; register tiles `f2x8`, `f4x4`
through `f4x12`, `f8x4`, `f8x8`; 2-4 staging stages; no split, no raster. The winners cluster at the grid's largest
tiles (`t32x16` with `f4x8`), so the grid may simply be too small.

Plan:

1. Extend the grid on one case per card (H100 `matmul.square.4096`, V100 `linear.down.h4096`): wider register tiles
   (`f8x8`, `f8x16`), `RASTER`, 5-stage staging. If a row closes the gap, re-record the class.
2. If the grid does not close it, profile the best row against cuBLAS's SGEMM with `ncu` (`emmy run --profile`):
   shared-memory bank conflicts on the transposed operand, the register-level K prefetch cuBLAS does, and occupancy at
   the 120+ registers the `f4x8` tile takes. Each is a lowering fix in the scalar tile tier, never a fusion change.

## 3. Tooling gaps the recording ran into

- **No sweep command.** After the tuner was removed there is no command that runs a grid of schedules for one golden
  realization, checks the top rows against eager and records the winner. This PR did it with a shell loop around
  `emmy run --ab` and `--record-greedy`. A small `emmy run --golden F --realization N --sweep GRID` would replace it.
- **An `EMMY_KNOBS` pin leaks into every `--ab` row that does not spell the same knob.** A sweep under a baseline pin
  must spell every knob in every row, or the rows silently measure the baseline's value.
- **`--record-greedy` needs every kernel-set fork pinned by hand.** A Linear's weight layout (`LAYOUT=folded`), a
  biased Linear's placement (`PLACE=fuse`) and attention's placement all fail strict evidence unless pinned, and the
  error names the fork, not the pin that would decide it.
