# Attention: what to try next (memo, 2026-09-10, revised after the chunk-loop density PR)

Where the A100 Neptune comparison stands: GQA decode is ahead of Neptune (0.31 against 0.34 of Inductor's time on
the paper's scale); causal prefill at 2048 keys is now 1.07x of FlashAttention-2 (the three prefill families are
being re-measured by the experiment lane); causal decode is 3.5x behind. The RTX 5090 is at parity or ahead on every
recorded family except plain decode, which has no golden. The items below are ordered by expected paper impact over
effort. Done and dropped: the per-chunk swizzle address math (hoisted per lane), the chunk-local P·V accumulator (the
chunk now folds at the advanced pivot), and the 64-way split offer; a descending causal launch order measured 0-2%
and was not carried; fast math measured 1% on the A100 prefill row.

## 1. Global prefill on the A100: retune the goldens per length

The global family's goldens (256-16384 keys) pin `f1x16/k4` + `f1x8/k4` with the key unstaged; the causal row's
geometry (`f1x16/k8` + `f1x16/k4`, both `d1/smem-async`) measured 497.7 us against 555.5 at 2048 keys (eager 455)
but 50.4 against 44.8 at 512, so the retune is per length, not one pin. The family is the one still behind
FlashAttention-2 (1.07-1.28x of eager) after the 2026-09-10 lane; causal and GQA prefill are at parity.

## 2. Plain decode (one query row per head): keep the unit query axis as the fragment's M

`decode_causal` forms the twisted carrier and even takes the split, but the placement drops the extent-1 query axis
(`_workspace_axes` and the placement's free set), leaving (head, head width) as the free pair. The fragment's M would
be the head axis, whose keys differ per row, so no tensor-core atom binds and the carrier folds on the scalar tier
(A100: 3.5x behind Neptune). Keeping the unit query axis as a masked M row (15 of 16 rows idle, which is what
FlashAttention-2's decode does) lets the chunk tier bind, and the same key-range split then applies. Touch points:
the placement's free-axis rule for unit extents and `_node_refusal`'s "no output-axis pair" reading.

## 3. Split-KV housekeeping

- `WORK` left unpinned picks `w8x1+p1` for the partial and the producer warp double-arrives on the TMA barrier: its
  `_gid` folds onto warp 0 (`threadIdx.x % block_threads`). No golden uses `+p1`; fix its elected-thread condition
  or drop the variant from the offer.
- Batch 8 at 32768 keys on the 5090 did not record: the greedy worker returned no outputs (the eager reference's
  broadcast is 4.3 GB per operand). A record path that checks accuracy against FlashAttention-2 instead of the
  materialized broadcast would fit.
- The finalize is serial per cell; at 64 partials per cell (32768 keys, `g64k`) it is 8 us of the 140. A band over
  the partials only pays when cells are few, so this is fine for decode; revisit if a split lands on a small-output
  reduce.

## 4. Prefill one-wave shapes (the article's Gemma shape)

Tried on 2026-09-10 (RTX 5090, `REDUCE@map.1/twist=g2k` and `g4k` on (1, 16, 512, 256) causal): the split rows
return WRONG answers (relative error 1.1 against the fused row), and the serial finalize alone costs as much as the
whole fused kernel (37 us against 38). The wrong answer is the split's: `_sliced_contraction` σ-reindexes the
carrier's operands to absolute keys but not its own lift body, where the coordinate masks live, so every partition
past the first masks slice-local keys as if they were absolute (and the causal early stop bounds the slice the same
way). Substituting the lift body alone is not enough — the partition coordinate becomes a free read of the score's
prefix cone, which the chunk tier's gate refuses — so the sliced carrier's lift must bind it, as `_sliced_edge`
binds it for a computed operand edge. Until then no causal split can deploy; the GQA decode goldens are non-causal
and unaffected.

The causal early stop cannot shorten a launch of fewer CTAs than SMs; FlashAttention-2 splits the key range there.
With split-KV now recordable for the fused carrier, try `REDUCE@map.1/twist=g2k` on (1, 16, 512, 256): the workspace
is 3 states × cells × 4 B per split (24 MB at that shape), so the split must beat the fused 39 us by more than the
workspace round trip; if it does not, the FA-2 form (O partial plus one LSE scalar per row) needs a per-row
workspace layout, which the chunk tier's whole-state store makes possible.
