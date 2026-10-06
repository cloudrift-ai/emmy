# FP32 performance gaps after the FP32 hardware goldens

Status: open. Found while recording FP32 rows for the V100, A100, H100 and RTX 5090 hardware goldens (2026-10-06).
That work left FP32 attention out of the goldens on purpose, and it left the gaps below unclosed. Every number here is
one RTX 5090 run of `emmy run --bench` unless it names another card.

Goal: refit both priors on the FP32 rows (deferred from the PR that recorded them), then an FP32 program compiled by the greedy, with no measurement of its own shape in scope, runs within 10% of eager
PyTorch for the kernel types the hardware goldens cover, attention included.

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

## 2. Other gaps the sweeps could not close

(filled in from the sweep results at the end of this PR)

## 3. Tooling gaps the recording ran into

- **No sweep command.** After the tuner was removed there is no command that runs a grid of schedules for one golden
  realization, checks the top rows against eager and records the winner. This PR did it with a shell loop around
  `emmy run --ab` and `--record-greedy`. A small `emmy run --golden F --realization N --sweep GRID` would replace it.
- **An `EMMY_KNOBS` pin leaks into every `--ab` row that does not spell the same knob.** A sweep under a baseline pin
  must spell every knob in every row, or the rows silently measure the baseline's value.
- **`--record-greedy` needs every kernel-set fork pinned by hand.** A Linear's weight layout (`LAYOUT=folded`), a
  biased Linear's placement (`PLACE=fuse`) and attention's placement all fail strict evidence unless pinned, and the
  error names the fork, not the pin that would decide it.
