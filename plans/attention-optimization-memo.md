# Attention: what to try next (memo, 2026-09-10)

Where the A100 Neptune comparison stands after the split-KV PR (#767): GQA decode is ahead of Neptune (0.31 against
0.34 of Inductor's time on the paper's scale), the three prefill families are 1.20-1.23x behind, causal decode is
3.5x behind. The RTX 5090 is at parity or ahead on every recorded family except plain decode, which has no golden.
The items below are ordered by expected paper impact over effort.

## 1. Prefill on the A100: the chunk loop is issue-bound

ncu on causal prefill (32 heads, 2048 keys, hd 128): Emmy 477 us against FlashAttention-2's 334, same FMA-pipe
utilization (28.7% vs 29.5%), same 255 registers (56 bytes of spills), 1.6x the load/store instructions, SM active
35% vs 45%. The sm_80 SASS of the recorded row (`f1x16/k8` + `f1x16/k4`, `d1/smem-async`, `w4x1`) puts about 2360
instructions in the chunk loop per 128-key chunk per warp against 256 HMMA — two warps per sub-partition issue more
slots than the tensor pipe needs, so the pipe waits on issue. Where the instructions are:

- ~835 integer ops (SHF / IMAD / IADD3 / LOP3 / LEA): the swizzled ldmatrix addresses are recomputed per chunk from
  `threadIdx` and the ring slot (`emmy_swizzle_b128(slot·rows·cols + lane terms)`), because the swizzle mixes the
  slot term with the lane term non-linearly and nvcc cannot hoist it. FlashAttention-2 keeps one per-lane smem
  pointer per slot. Hoist the per-lane, per-slot base addresses out of the chunk loop (the V100 matmul PR did this for
  its operands) — likely the single largest lever, and it applies to every staged ring kernel.
- ~800 float ops (FFMA / FADD / FMUL), twice FlashAttention-2's softmax arithmetic per key: the scale multiply is a
  separate pass over the score fragment where FA-2 folds it into the `exp2` FFMA with log2e; the per-chunk merge
  computes the rescale per element where a per-row alpha times the accumulator suffices; and the causal mask runs
  its 76 ISETP per chunk on every chunk (the doc notes confining it measured nothing on its own, but it is part of
  the issue budget once the address math is gone).
- Deeper cp.async rings, 32-row warp tiles (`f2x16`), eight warps and fast math all measured slower than the recorded
  row on this card, so do not spend pin sweeps there; the geometry is right, the code density is not.

Expected: the 5090 reaches parity with the same kernel family on a TMA ring (address math is the TMA unit's), which
suggests the A100 gap closes when the cp.async lane stops issuing per-chunk address arithmetic.

## 2. Plain decode (one query row per head): keep the unit query axis as the fragment's M

`decode_causal` forms the twisted carrier and even takes the split, but the placement drops the extent-1 query axis
(`_workspace_axes` and the placement's free set), leaving (head, head width) as the free pair. The fragment's M would
be the head axis, whose keys differ per row, so no tensor-core atom binds and the carrier folds on the scalar tier
(A100: 3.5x behind Neptune). Keeping the unit query axis as a masked M row (15 of 16 rows idle, which is what
FlashAttention-2's decode does) lets the chunk tier bind, and the same key-range split then applies. Touch points:
the placement's free-axis rule for unit extents and `_node_refusal`'s "no output-axis pair" reading.

## 3. Split-KV housekeeping

- `SPLITK_WIDTHS = (2, 4, 8, 16, 32)`: a 64-way split records under a pin but never deploys from evidence (the A100
  fell to a 1.6 s scalar loop at 32768 keys until the golden pinned 32). Either offer 64 or make a route row whose
  arm is not offered fail loudly at the evidence pick.
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

The causal early stop cannot shorten a launch of fewer CTAs than SMs; FlashAttention-2 splits the key range there.
With split-KV now recordable for the fused carrier, try `REDUCE@map.1/twist=g2k` on (1, 16, 512, 256): the workspace
is 3 states × cells × 4 B per split (24 MB at that shape), so the split must beat the fused 39 us by more than the
workspace round trip; if it does not, the FA-2 form (O partial plus one LSE scalar per row) needs a per-row
workspace layout, which the chunk tier's whole-state store makes possible.
