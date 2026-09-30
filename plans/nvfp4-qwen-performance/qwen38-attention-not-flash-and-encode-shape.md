# Qwen3.8 attention repeats probability computation; encode launches a block per byte

## Summary

Two forms of repeated work appear in the inspected Qwen3.8 NVFP4 layer: attention recomputes each softmax
probability for every output column, and activation encoding recomputes each group's maximum for every packed byte.
CPU-only follow-up confirmed a missed softmax rewrite after Q/K cuts. Repeating the rewrite exposes partial carriers,
but the resulting three kernels use no tensor-core instructions and the attention consumer runs only 16 active
threads. This is not a practical route for Qwen3.8 NVFP4 serving.

The checkpoint has 16 full-attention layers out of 64, at indices 3, 7, 11, and so on. The examples below use layer 3
at 16 tokens after cuts split the fused kernel into smaller kernels, called pieces, exposing its projections to fp4
tensor-core schedules.

## Observed and expected Tile IR

**Observed P·V**, from the saved layer-3, 16-token dump with the full-projection cut and fp4 pin at `a98fd4f8`:

```text
=== 13: k_sdpa_linear_mean_reduce_c6c239__place_1108b7f44e ===
    place  free=(a1, a0, a2, a3, a4)  grid=(a1, a0, a2, a3, a4)
    Fold[a5 in 0..16] reduce
    ├─ operand[in6]: load mul_10_static_fp4_scale_bits__place_5b87de7ca7_0[a0, a3, a4, a5]   ‹materialized›
    ├─ operand[v5]: Fold  free   ‹computed›
    │  ├─ operand[in5]: load mul_10_static_fp4_scale_bits__place_ff1a5c8a30_0[a1, ((6 * a0) + a2), a5]   ‹materialized›
    │  ├─ operand[-1e+09, 0, 0.0625, in3, in4]: Fold  free  ‹pointwise›   ‹computed›
    │  │  └─ lift: λ(a0, a1, a2) -> (-1e+09, 0, 0.0625, in3, in4)
    │  │       in3 = load mul_10_static_fp4_scale_bits__place_19eeb2b186_0[a1, ((6 * a0) + a2)]
    │  │       in4 = load mul_10_static_fp4_scale_bits__place_3814a75921_0[a1, ((6 * a0) + a2)]
    │  └─ lift: λ(in5, -1e+09, 0, 0.0625, in3, in4, a1, a5) -> (v5)
    │       v0 = multiply(0.0625, in5)
    │       v1 = 0 when ((a5 <= a1))
    │            -1e+09 when ((a1 < a5))
    │       v2 = add(v0, v1)
    │       v3 = subtract(v2, in4)
    │       v4 = exp(v3)
    │       v5 = divide(v4, in3)
    ├─ init: (0)
    ├─ lift: λ(a5, in6, v5) -> (v6)
    │    f32 v6 = multiply(in6, v5)
    └─ combine: λ(acc0, acc0__o) -> (acc0)
         acc0 = add(acc0, acc0__o)
    outputs
    └─ mul_10_static_fp4_scale_bits__place_1108b7f44e_0[a1, ((6 * a0) + a2), a3, a4] = acc0

```

The output-column axes `a3` and `a4` are in `grid`. Every output repeats the `exp` and `divide` inside the key
reduction (`a5`). The generated workspace names contain `static_fp4_scale_bits`, but these loads hold V, scores,
softmax sums and maxima; the names do not describe their contents.

**Expected recognition, composed Tile IR**, based on the actual successful control dump described below. `q`, `k`
and `v` denote materialized attention inputs after projection, normalization and RoPE. `…` omits ordinary contraction
arithmetic, softmax rescaling and helper definitions; this is an excerpt, not a complete runnable IR file:

```text
    place  free=(a0, a1, a6)  unmapped
    Fold  free
    ├─ operand[acc1, acc3, acc5__sum]: Fold[a2 in 0..16] contraction  ⟨twist=softmax⟩   ‹computed›
    │  ├─ operand[acc0]: Fold[a3 in 0..256] contraction   ‹computed›
    │  │  ├─ operand[in4]: load q[0, a0, a1, a3]   ‹materialized›
    │  │  ├─ operand[in3]: load k[0, (a0 / 6), a2, a3]   ‹materialized›
    │  │  …
    │  ├─ operand[in9]: load v[0, (a0 / 6), a2, a6]   ‹materialized›
    │  ├─ init: (-1e+30, 0, 0)
    │  ├─ lift: λ(a2, acc0, in9, a1) -> (v5, one, in9)
    │  │    v3 = multiply(acc0, 0.0625)
    │  │    one = 1.0
    │  │    v4 = 0 when ((a2 <= a1))
    │  │         -1e+09 when ((a1 < a2))
    │  │    v5 = add(v3, v4)
    │  ├─ combine: λ(acc1, acc3, acc5__sum, acc1__o, acc3__o, acc5__sum__o) -> (acc1, acc3, acc5__sum)
    │  │    acc1__o__gn = maximum(acc1, acc1__o)
    │  │    …
    │  │    acc3 = add(acc1__o__acc3_sa, acc1__o__acc3_sb)
    │  │    acc5__sum = add(acc1__o__acc5__sum_sa, acc1__o__acc5__sum_sb)
    │  │    acc1 = copy(acc1__o__gn)
    │  …
    └─ lift: λ(acc3, acc5__sum) -> (acc5)
         acc5 = divide(acc5__sum, acc3)
    outputs
    └─ attention[0, a0, a1, a6] = acc5
```

Here `acc1`, `acc3` and `acc5__sum` carry the running maximum, denominator and weighted output together across keys.
The division is after that contraction. This shows the desired **pre-scheduling** carrier, not a proven flash kernel:
compatible tensor-core schedules and reuse across output-column tiles must still be checked in scheduled Tile IR and
CUDA. Dense attention retains quadratic query/key work.

**Observed encode**, from the saved gate/up full-projection-cut dump. Only omitted lines are replaced with `…`:

```text
=== 4: k_linear_reduce_2d1c79__place_03cba86402 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   t128
    Fold  free
    ├─ operand[acc0]: Fold[a2 in 0..16] reduce   ⟨REDUCE=coop⟩   ‹computed›
    │  ├─ operand[in5]: load mul_13_static_fp4_scale_bits__place_f98562e225_0[a0, (a1 / 8), a2]   ‹materialized›
    │  ├─ operand[in6]: load mul_13_static_fp4_scale_bits__place_f98562e225_1[a0, (a1 / 8), a2]   ‹materialized›
    │  ├─ init: (-1e+30)
    │  ├─ lift: λ(a2, in5, in6) -> (v9)
    │  │    f16 v1 = copy(in5)
    │  │    f32 v2 = copy(v1)
    │  │    …
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = maximum(acc0, acc0__o)
    └─ lift: λ(acc0, a0, a1) -> (v39)
         in0 = load mul_13_static_fp4_scale_2[0, 0, 0, 0]
         v0 = multiply(in0, 6)
         v10 = divide(acc0, v0)
         f8e4m3 v11 = to_f8e4m3(v10)
         …
         f4e2m1x2 v39 = copy(v38)
    outputs
    └─ mul_13_static_fp4_bits[0, a0, a1] = v39
```

`a1` selects a packed byte, yet the reduction reads group `a1 / 8`. Eight byte outputs therefore repeat one group's
maximum. **Composed expected Tile IR** below instead makes `a1` the group and `a3` the byte within it (extent eight).
The unchanged SiLU, scale conversion and packing arithmetic is omitted:

```text
    place  free=(a0, a1)  grid=(a0, a1)
    work   t128
    Fold  free
    ├─ operand[acc0]: Fold[a2 in 0..16] reduce   ⟨REDUCE=coop⟩   ‹computed›
    │  ├─ operand[in5]: load mul_13_static_fp4_scale_bits__place_f98562e225_0[a0, a1, a2]   ‹materialized›
    │  ├─ operand[in6]: load mul_13_static_fp4_scale_bits__place_f98562e225_1[a0, a1, a2]   ‹materialized›
    │  …
    │  └─ combine: λ(acc0, acc0__o) -> (acc0)
    │       acc0 = maximum(acc0, acc0__o)
    └─ lift: λ(acc0, a0, a1, a3) -> (v11, v39)
         …
         in7 = load mul_13_static_fp4_scale_bits__place_ca3a5e6d66_0[a0, ((8 * a1) + a3)]
         …
         f4e2m1x2 v39 = copy(v38)
    outputs
    ├─ mul_13_static_fp4_scale_bits[0, a0, a1, 0] = v11
    └─ sweep(a3) mul_13_static_fp4_bits[0, a0, ((8 * a1) + a3)] = v39
```

The group reduction no longer depends on the byte sweep. This is one way to express the required reuse; the eventual
worker layout is not prescribed. Generated CUDA must confirm that the maximum is evaluated once per group.

## Observed details

1. **P·V recomputes softmax probabilities.** None of the three initially tested cut sets (the
   default, a projection-only cut, and the full-projection cut) keeps scaled dot-product attention as one kernel with
   an online softmax. The later Q/K-cut investigation below finds a skipped rewrite and an impractical partial route. P·V becomes a reduce over keys with one 128-thread block per output element (batch × head ×
   query × head-dim column, 98,304 blocks at 16 tokens). Each block recomputes `exp(score − max) / sum` for every key,
   so probability evaluation runs once per head-dim column (256 times per score) instead of being reused. The max and
   sum have already been materialized; this is not a fresh reduction of the entire softmax in each block.
   - Four pieces compute this product. Two are printed as `reduce` (`…__place_1108b7f44e`, `…__place_d20db92cbe`), at
     111–149 µs each.
   - Two are printed as `contraction` (`…__place_129c440787`, `…__place_54c95851bf`); with the fp4 TILE pinned, the
     scheduler leaves them without a schedule.
   - The four-fold copy is likely the same recompute as in [the producer-duplication
     report](fp4-encode-recomputes-producer.md); this link between causes is unverified. The cut
     prints two of them as `reduce`, not `contraction`; that may be why the scheduler offers them no tensor-core
     schedule.
   - This repeats work within attention; it does not change its query/key arithmetic complexity.
2. **Activation-encode kernels launch one block per packed output byte.** A 4-bit re-encode of a matmul output runs
   `WORK=t128, REDUCE=coop`, with one 128-thread block per byte cooperatively reducing its 16-value group. The group's
   index is `byte_index / 8`, so eight neighboring byte outputs repeat the same maximum reduction.
   - For the gate/up output at 16 tokens that is 139,264 blocks, 151.7 µs, for a job that moves about 0.5 MB.
   - At 512 tokens, o_proj's encode takes 1.6 ms, and gate/up's takes 7.2 ms with the fp4 TILE pinned (0.65 ms under
     the default pick).

## Reproduce

All commands run from the repository root inside `nix develop`, with a fresh tune DB.

```sh
M=Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462   # ~26 GB download on first use
rm -f /tmp/q38.db; export EMMY_TUNE_DB=/tmp/q38.db
./venv/bin/emmy trace $M --layer 3 --seq-len 16 --target sm_120 -o /tmp/l3_s16.golden.json

# 1. Attention kernel (q/k/v projections, q/k norm, RoPE, SDPA, output gate, encode): full-projection cut + fp4 cell.
#    The compile takes about 14 s.
ATTN_CUT='PLACE@map.1/map=cut,PLACE@map.1/map.1/inner=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.2/inner.1/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.1/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map.11/map.1/reduce.1/inner=cut,PLACE@map.1/map.2/inner.2/map.2/reduce=cut,PLACE@map.1/map.2/inner.2/map.3/reduce=cut,PLACE@map.1/map.3/inner=cut,PLACE@map.1/map.4/reduce.1/inner=cut,PLACE@map.1/map.4/reduce.2/inner=cut,PLACE@map.1/map.4/reduce.2/inner.1/map.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.1/reduce.1/inner=cut,PLACE@map.2/map.1/reduce.2/inner=cut,PLACE@map.2/map.1/reduce.2/inner.1/map.1/inner=cut'
EMMY_KNOBS="$ATTN_CUT,TILE=mma_m16n8k64_e2m1_f32/f1x1/k8,STAGE=d3/smem-async" \
  ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json --realization k_sdpa_linear_mean_reduce_c6c239 --ir tile
# the same knobs with `emmy run … --bench --no-record-evidence` give per-piece times and block counts

# 2. Gate/up + SiLU + encode, full-projection cut + fp4 cell; the bench shows the encode piece's launch shape:
EMMY_KNOBS='PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,TILE=mma_m16n8k64_e2m1_f32/f1x2/k8,STAGE=d3/smem-async' \
  ./venv/bin/emmy run --golden /tmp/l3_s16.golden.json --realization k_linear_reduce_2d1c79 --bench --no-record-evidence
```

Kernel names are the ones this trace gives at commit `a98fd4f8`. `emmy golden kernels /tmp/l3_s16.golden.json` prints
each kernel's Loop IR as one JSON line, and its `"name"` fields list the current names.

## Launch evidence

The gate/up encode piece in the bench table:


```
Kernel                                       us      %    grid  block   …  WORK  …  REDUCE
k_linear_reduce_2d1c79__place_03cba86402  151.7  25.9%  139264    128   …  t128  …  coop
```

   Here `grid` is the thread-block count and `block` is threads per block. 139,264 = 16 tokens × 8,704 packed bytes
   per row (17,408 values, two per byte).

## Flash verdict and CPU-only investigation (2026-09-28)

**Keep the bug report.** There is a confirmed missed recognition opportunity in the actual Qwen3.8 NVFP4 cut path,
not just three unsuccessful schedule choices. The additional pass route described below also fails the practical
check: its generated CUDA uses scalar projections and almost serial attention. This does not prove that every possible
cut fails. A workaround must yield a usable kernel set for the checkpoint, including its fp4 projections.

At `a98fd4f8`, `020_twisted` applies the softmax recipe before cuts. `Fold.fuse` matches the score expressions and
reduction axes, then joins the maximum, denominator and weighted-output states. A TILE pin schedules the resulting
contraction; it does not create this carrier.

**Actual checkpoint: a rewrite is skipped after cuts.** Cutting the final Q and K operand expressions produces three
pieces, still without a softmax carrier. The consumer retains output sweeps. In `_row.reformed`, the following early
return preserves those sweeps but also skips the later call to `rewrite_twisted`:

```python
if any(store.sweep for store in piece.output_specs):
    return piece
```

Applying the existing rewrite once more to that consumer produces two distinct partial carriers: `(max, denominator)`
and `(max, weighted output)`, with recipe channels `(0,)` and `(1,)`. This was first checked in memory and then
reproduced through the ordinary compile CLI by adding a second `t` pass:

```sh
QK_CUT='PLACE@map.1/map.2/inner.2/map.1/inner.1/map=cut,PLACE@map.1/map.2/inner.2/map.1/inner.2/map=cut'
EMMY_KNOBS="$QK_CUT" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json \
  --realization k_sdpa_linear_mean_reduce_c6c239 --target sm_120 --passes tp --ir tile
# No twist=softmax.
EMMY_KNOBS="$QK_CUT" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json \
  --realization k_sdpa_linear_mean_reduce_c6c239 --target sm_120 --passes tpt --ir tile
# Partial carriers appear; this stops before scheduling.
```

The latter dump contains these actual lines at separate sites; each carrier has two states, not the complete three:

```text
    │  │  │  ├─ operand[acc26, acc39]: Fold[a4 in 0..16] reduce  ⟨twist=softmax⟩   ‹computed›
    │  │  │  ├─ operand[acc13, acc71__sum]: Fold[a4 in 0..16] contraction  ⟨twist=softmax⟩   ‹computed›
```

**Practical check: this route still does not deliver usable flash.** Continuing the same Q/K cuts and repeated
recognition pass through scheduling and CUDA emission succeeds:

```sh
EMMY_KNOBS="$QK_CUT" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json \
  --realization k_sdpa_linear_mean_reduce_c6c239 --target sm_120 --passes tpth --ir tile
EMMY_KNOBS="$QK_CUT" ./venv/bin/emmy compile --golden /tmp/l3_s16.golden.json \
  --realization k_sdpa_linear_mean_reduce_c6c239 --target sm_120 --passes tpthkc --ir cuda
```

Both Q/K producers select `TILE=f2x4`, a scalar tile. No `mma`/`wmma` instruction or helper call appears in any of the
three emitted kernels. The consumer has no worker layout in scheduled Tile IR; its CUDA admits only 16 active threads
and loops serially over heads, head dimension and the 5120-wide projections. These are actual emitted lines, with
omissions marked:

```cuda
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (_gid < 16) {
        int a0 = _gid;
        // …
        for (int a35 = 0; a35 < 24; a35++) {
            float acc74 = 0.0f;
            for (int a36 = 0; a36 < 256; a36++) {
                float acc73 = 0.0f;
                for (int a37 = 0; a37 < 5120; a37++) {
                    // …
```

This structural result is enough to reject this attempt as a practical serving workaround; no latency estimate or
GPU run is needed. Further cuts or schedules might improve it, but no such complete route was established here.

The early return explains why these opportunities are missed in the normal cut path. It does not by itself explain
all failed score matches or establish that changing the return is a safe, sufficient fix.

**Reduced case: computed Q/K score matching is also fragile.** Plain causal GQA at the same 24/4 heads, 16 tokens and
256 head dimension forms a complete carrier. Put Q/K/V projections inside the traced module, and the following
smaller f16 example forms none. Its projection input width is reduced to 64; this isolates recognition and is not a
replacement for the checkpoint repro:

```sh
PROG='
class M(nn.Module):
    def forward(self, x, wq, wk, wv):
        q = F.linear(x, wq).view(1, 16, 24, 512)[..., :256].transpose(1, 2)
        k = F.linear(x, wk).view(1, 16, 4, 256).transpose(1, 2)
        v = F.linear(x, wv).view(1, 16, 4, 256).transpose(1, 2)
        return F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
M()(torch.randn(1, 16, 64).half(), torch.randn(12288, 64).half(),
    torch.randn(1024, 64).half(), torch.randn(1024, 64).half())'
EMMY_KNOBS= ./venv/bin/emmy compile --target sm_120 --passes dolfnst --ir tile -c "$PROG"
# No twist=softmax.
```

For comparison, the materialized-input control forms the complete `(max, denominator, weighted output)` carrier:

```sh
EMMY_KNOBS= ./venv/bin/emmy compile --target sm_120 --passes dolfnst --ir tile \
  -c 'F.scaled_dot_product_attention(torch.randn(1,24,16,256).half(), torch.randn(1,4,16,256).half(), torch.randn(1,4,16,256).half(), is_causal=True, enable_gqa=True)'
```

The reduced dump spells one score contraction with computed K then Q operands and another with Q then K. The matcher
compares their canonical Fold forms, while the orientation rule preserves the former's order for computed operands.
An in-memory diagnostic that consistently orders single-product contractions restores the denominator carrier, but
still not the full weighted-output carrier. It also does not restore the actual checkpoint's carrier by itself.
Thus operand ordering contributes to the reduced failure; treating it as the sole cause of the checkpoint failure
would be unverified.

All checks above used CPU tracing, IR rewriting and compilation. No GPU correctness or performance result is implied.

## Compare

Not checked: whether shipped goldens reach a flash form for full attention, and by which route. Candidates are
`recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` and the Qwen3-0.6B goldens under `experiments/Qwen3-0.6B/*/golden/`
and `experiments/golden-bench-2026/kernels/golden/`.

## Fix criteria

1. **Attention.** With some cut set, at 16 and 512 tokens:
   - A documented cut and schedule reaches fused attention for this checkpoint's grouped-query layout. Tile IR carries
     the running max, sum and weighted output across key blocks, and both score and P·V contractions carry compatible
     tensor-core TILEs (`mma_…`). The number of kernels or query tiles is not prescribed; legal tiling and split-KV
     remain available.
   - No piece reduces over keys with `exp`/`divide` in its body while it grids the head-dim axis.
   - P·V is computed once, not four times.
   - Compare attention and whole-kernel-set latency before and after at 16 and 512 tokens, on the same hardware and
     inputs with correctness checked. Check that probability evaluation is reused across output columns. Attention
     still requires work proportional to queries × keys × head dimension; flash changes data movement and reuse, not
     that arithmetic bound.
2. **Encode.**
   - Compute each 16-value group's maximum once and reuse it for its eight packed bytes and stored scale. Distribute
     groups across threads or warps without dedicating a mostly idle 128-thread block to every byte. Do not require
     one exact block count or layout.
   - Compare encode and total kernel-set time at 16 and 512 tokens under matching measurement conditions. Record
     launch geometry, work per block, and correctness. Treat the reported microseconds as baselines; explain any
     latency regression rather than imposing an unmeasured absolute target.
- **Correctness:** the outputs match the current kernel set on the same inputs (`emmy run … --ab`, whose wrong-answer
  check compares both). This check drops out when the reference compile disagrees with itself (the bench prints
  "wrong-answer reference unusable"). A fixer must confirm the check actually ran.
