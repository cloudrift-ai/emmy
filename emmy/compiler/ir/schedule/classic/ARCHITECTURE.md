# Classic schedule against the best known kernels

The classic schedule family binds a kernel's axes to the grid, the CTA, its warps and threads, and their registers.
The [schedule model](../ARCHITECTURE.md) owns how its sites, choices and compatibility rules work. This file asks one
question of that space: for each kind of kernel a model needs, can a row of knobs spell the best hand-written kernel
known? Each section gives the best known kernel, pseudocode for its structure, the row that spells it, and a verdict:

- **Reproduces** — a row spells the same structure, and it realizes.
- **Better** — a row measured faster than the reference kernel on the same card.
- **Gap** — a part of the structure no row can spell. The gap is named, so it can be closed or left on purpose.

A gap is a statement about the space, never about fusion: a slow fused kernel is fixed by a cut, a schedule or
lowering coverage.

## Where the numbers come from

Times are microseconds on one card. "Recorded" is the fastest row that card's hardware golden holds for the program,
which is the row a compile deploys, beside the eager PyTorch, `torch.compile` or cuBLAS time stored with the program.
A program's first row can be slower than its fastest one, and `emmy golden list` prints the first. "Pinned" is a hand
pin run once with `emmy run --bench` on an RTX 5090 (CUDA 13.0, November 2026). A pinned time shows that a structure
realizes and roughly what it costs; it is not evidence, and two pins compare only inside one run. Shapes of the
hardware golden's programs used below:

| Program | Shape |
| --- | --- |
| `pointwise.n16384`, `reduce.k8192`, `softmax.k8192`, `rms_norm.k8192` | f32, 512 rows |
| `matmul.square.4096` | f16, 4096 x 4096 x 4096 |
| `matmul.qkv.h4096` | f16, 512 x 4096 times 4096 x 12288 |
| `attention.hd128.fused` | f16, 1 x 8 heads x 512 x 128, queries = keys |
| `attention.hd256.causal.fused` | f16, 1 x 16 heads x 512 x 256, causal |

## The knobs, in the terms of a hand-written kernel

| Hand-written kernel says | Row spells |
| --- | --- |
| Warps per CTA and their layout over the output tile | `WORK=w<M>x<N>` |
| A dedicated load warp beside the compute warps | `WORK=…+p<n>` (needs a TMA stage) |
| Threads per CTA of a kernel without tensor cores | `WORK=t<N>x<M>`, or `t<N>` for one cooperating group |
| Tensor-core instruction and accumulate dtype | the atom in `TILE=<atom>/…` |
| Warp tile, in instructions per warp | `TILE=…/f<FM>x<FN>` |
| K step of the main loop | `TILE=…/k<bk>`, in units of the atom's K |
| Elements per thread, held in registers | `TILE=f<n>` or `f<n>x<m>` |
| Shared-memory pipeline: stages and how they fill | `STAGE=d<depth>/smem`, `smem-async` (cp.async) or `smem-tma` |
| Register double buffer of the fragments | `STAGE=…/p<n>` |
| Threads that share one reduction, and its combine | `REDUCE=coop` (lane shuffle, then shared memory) |
| Vector load of adjacent elements per thread | `REDUCE=coop/v<n>` |
| Independent accumulators per thread | `REDUCE=…/r<n>` |
| Threads swept along the output of a product | `REDUCE=coop-t` |
| Reduction split across CTAs, second kernel combines | `REDUCE=g<n>k` |
| Reduction split across CTAs, atomic add combines | `REDUCE=g<n>a` |
| Order of CTAs over the output, for cache reuse | `RASTER=gm<G>` or `gn<G>` |
| Which work is fused into the kernel | `PLACE@<seam>=cut` or `fuse` |

A CUTLASS tile of 128 x 128 x 32 on four warps is therefore `WORK=w2x2`, `TILE=mma_m16n8k16_f16_f32/f4x8/k2`: each
warp holds 4 x 8 instructions of 16 x 8, and the K step is two atoms of 16.

## Summary

| Kernel | Best known | Verdict |
| --- | --- | --- |
| Pointwise map | PyTorch `gpu_kernel`, 16-byte loads | Reproduces |
| Row reduction | PyTorch `Reduce.cuh`, CUB block reduce | Reproduces |
| Reduction to one value | CUB `DeviceReduce` | Reproduces the two-kernel form |
| Softmax, RMSNorm, LayerNorm | Quack, FlashInfer | Reproduces; gap: row in shared memory |
| Prefix scan | CUB decoupled look-back, PyTorch scan | Better than PyTorch on rows; gap: scan across CTAs |
| Top-k, sort, sampling | Air Top-K, FlashInfer sampling | Gap |
| Gather, scatter, cache append | FlashInfer page append | Reproduces gather; gap: scatter-add |
| Dense product, Ampere style | CUTLASS multistage | Reproduces; better with f16 accumulate |
| Dense product, load warp and TMA | CUTLASS warp-specialized | Reproduces the tile; gap: persistent launch, ping-pong |
| Dense product, `tcgen05` | CUTLASS sm_100 | Gap |
| Split over K | split-K, Stream-K | Reproduces split-K; gap: Stream-K |
| Product with few rows | vLLM dot-product and skinny kernels | Better than cuBLAS |
| FP8 and NVFP4 weights | CUTLASS, B12x | Reproduces the atom and the transport; not compared |
| 4-bit integer weights | Marlin | Gap |
| Mixture of experts | B12x resident kernel, grouped product | Gap |
| Convolution | cuDNN | Reproduces implicit product; gap: Winograd, FFT |
| Attention, prefill | FlashAttention-2 | Better for full attention; gap: causal, head size 256 |
| Attention, Hopper | FlashAttention-3 | Reproduces the tile; gap: ping-pong, overlap |
| Attention, `tcgen05` | FlashAttention-4 | Gap |
| Attention, decode | Flash-Decoding, FlashInfer | Better than FlashAttention-2 |
| Attention over a paged cache | FlashInfer | Gap: async copies of a paged operand |
| Recurrence with a matrix state | FlashInfer Gated DeltaNet | Gap |
| Recurrence, one channel per thread | Mamba, RWKV | Reproduces RWKV; no evidence for Mamba |
| Small triangular inverse | FlashInfer `CollectiveInverse` | Gap |

## Pointwise map

**Best known.** PyTorch's `gpu_kernel`: 128 threads, each owning 8 contiguous elements read as aligned 16-byte
vectors. The kernel is bound by memory bandwidth; Quack reports 90% of an H100's peak with the same rule of one
128-bit load per thread. A chain of maps is one kernel only under a compiler (Inductor); eager PyTorch launches each.

```
grid = ceil(N / 1024); block = 128
thread t of CTA c:
    base = (c * 128 + t) * 8
    for u in 0, 4:                              # unrolled
        x = load_16_bytes(in + base + u)        # one aligned vector per input
        store_16_bytes(out + base + u, f(x))    # f is the whole fused chain
```

**Row.** `TILE=f<n>` hands a thread 2, 3, 4 or 8 contiguous elements of the inner axis, and the strip's consecutive
loads and stores fold into vector accesses (`VECTORIZE_LOADS`, `VECTORIZE_STORES`, on by default). Fusion is maximal,
so the chain is one kernel by construction.

**Verdict: reproduces.** `pointwise.n16384`, RTX 5090, recorded `TILE=f2`: 11.0 against `torch.compile` 11.6 and
eager 12.5. RTX 4090, `TILE=f4`: 16.1 against 20.2.

## Row reduction

**Best known.** One CTA per row. Each thread folds its share with several independent accumulators and vector loads;
lanes combine by shuffle, warps through shared memory. Narrow rows take one warp each, several rows per CTA
(FlashInfer's query/key norm).

```
grid = rows; block = 128..512
thread t:
    acc[0..4] = identity
    for i in t*V .. K step block*V:             # V adjacent elements, one vector load
        acc[i % 4] = combine(acc[i % 4], fold(load_vector(row + i)))
    a = combine(acc[0..4])
    a = shuffle_butterfly(a)                    # across the 32 lanes
    a = shared_memory_tree(a)                   # across the warps
thread 0: out[row] = a
```

**Row.** `WORK=t<coop>`, `REDUCE=coop/v<n>/r<n>`: `coop` threads per row, `v` adjacent elements per lane and step,
`r` accumulators. The combine is derived: a shuffle up to one warp, then shared memory. `WORK=t<coop>x<cells>` packs
several rows into a 128-thread CTA when `coop` is at most a warp.

**Verdict: reproduces.** `reduce.k8192`, RTX 5090, recorded `WORK=t128 REDUCE=coop`: 3.3 against `torch.compile` 3.4
and eager 6.1. A sum over the same shape pinned at `WORK=t128 REDUCE=coop/v4`: 3.2.

## Reduction to one value

**Best known.** CUB `DeviceReduce`: a first kernel writes one partial per CTA (256 threads, 16 items each, vector
loads), a second kernel reduces the partials. PyTorch does it in one launch: the last CTA to finish, found by an
atomic counter, combines the partials.

```
kernel 1, CTA c:  partial[c] = block_reduce(items of tile c)     # the row reduction above
kernel 2, one CTA: out = block_reduce(partial[0..grid])
```

**Row.** `REDUCE=g<n>k/coop` is CUB's form: `n` CTAs per output and a deferred combine kernel. `REDUCE=g<n>a` is the
one-launch form for a sum: each CTA adds its partial into the output atomically.

**Verdict: reproduces** the two-kernel form. Two limits: the cut pass offers 2 to 32 CTAs per output unpinned, where
CUB sizes the grid to the input; and the one-launch form exists for additive combines only, so a maximum always pays
the second kernel.

## Softmax, RMSNorm, LayerNorm

**Best known.** Quack (the reference PyTorch now measures against) and FlashInfer. One CTA per row, or several narrow
rows per CTA. The row is copied once into shared memory, so global memory is read once however many statistics the
operation needs; registers read it back as 128-bit vectors. Rows of 65k and more reduce across a cluster of CTAs
through distributed shared memory. FlashInfer's large-vocabulary softmax instead splits the row over CTAs and merges
(maximum, denominator) pairs in a second kernel.

```
grid = rows; block = 512
row_smem = async_copy(row)                      # the only read of global memory
m = block_reduce(max, row_smem)                 # shuffle, then shared memory
d = block_reduce(sum, exp(row_smem - m))
thread t: store_vector(out + t*V, exp(row_smem[t*V ..] - m) / d)
```

**Row.** The statistics are reduce sites and the map is fused, so the row is the row reduction's:
`WORK=t512 REDUCE=coop`.

**Verdict: reproduces, within 1.1 times the reference.** RTX 5090: `softmax.k8192` is recorded at 8.1
(`WORK=t512 REDUCE=coop`) against `torch.compile` 7.3, and `rms_norm.k8192` at 8.5 (`WORK=t512 REDUCE=coop/r4`)
against 7.6. No faster row was found. In one run on the softmax, a vector run per lane was slower (`coop/v4` 10.1,
`coop/v8` 9.3, against 8.2 for the recorded row), and so were 256 threads (11.2) and one warp per row (`WORK=t32x4`,
10.8), which belongs to narrow rows.

**Gap: the row in shared memory.** A `STAGE` pin on a cooperative reduce does not realize, so every statistic and the
final map read the row from global memory again. This is the likely remainder on both programs. A reduction across a
cluster of CTAs has no spelling either; `REDUCE=g<n>k` covers that case with a second kernel, as FlashInfer does.

## Prefix scan

**Best known.** For one long sequence, CUB `DeviceScan`: a single pass in which each CTA reduces its tile, publishes
the total, looks back over its predecessors' totals, then scans and writes. For rows, PyTorch scans a pair of
shared-memory windows per step and carries the running total along the row.

```
CTA c (single pass, decoupled look-back):
    local = block_reduce(tile c);  publish(c, local)
    prefix = identity
    for p in c-1 down to 0:                     # stops at the first CTA that published a full prefix
        prefix = combine(status(p).value, prefix)
        if status(p).is_full_prefix: break
    publish_full(c, combine(prefix, local))
    out[tile c] = block_scan(tile c, seed = prefix)
```

**Row.** A scan root takes `REDUCE=coop` up to one warp, every lane keeping one inclusive state, and packs rows:
`WORK=t32x4 REDUCE=coop`.

**Verdict: better than PyTorch on rows.** A cumulative sum over 512 x 8192 f32, pinned at that row, runs 11.7 against
eager 52.

**Gap: a scan across CTAs.** A scan is never split over the grid, and a lane's vector run (`/v<n>`) is refused on a
scan. One sequence of millions of elements therefore runs on one warp, where the look-back kernel uses the whole
card.

## Top-k, sort, sampling

**Best known.** Radix selection: Air Top-K (the basis of `cub::DeviceTopK`) and RadiK build per-CTA digit histograms
in shared memory, merge them with atomics, and keep only the candidates of the selected digit, all passes in one
launch. FlashInfer samples top-p, top-k and min-p by rejection: each round reads the row once and tests pivots.

```
candidates = row
for digit from most significant:                # one launch; passes separated by a grid barrier
    hist = merge_atomic(per_CTA_histogram(candidates, digit))
    b = bucket where the running count crosses k
    emit(elements in buckets above b);  candidates = elements in bucket b;  k -= count above b
```

**Row.** None. The index of a maximum compiles as two maximum reductions in one kernel, which covers greedy decoding.

**Verdict: gap.** Selection needs a histogram (a scatter-add, which has no lowering), a loop whose trip count depends
on the data, and a barrier across CTAs inside one launch. None is a schedule choice; the first is tracer and lowering
work, the other two would be new kernel forms.

## Gather, scatter, cache append

**Best known.** FlashInfer's page append: one CTA per request, a thread per vector of a head, each copying 16 bytes
or more of the key and the value to an address read from the page table.

```
grid = requests; block = (head_dim / V, heads)
thread (x, h): p = page_table[request];  store_vector(cache[p, h, x*V ..], load_vector(src[h, x*V ..]))
```

**Row.** A gather or an indexed copy is a map whose load or store index is read from memory, scheduled as a map:
`TILE=f4`.

**Verdict: reproduces** gather and indexed copy; the recorded copy kernels of the paging programs run under a
microsecond. **Gap: scatter-add.** A store that combines into its target has no lowering.

## Dense product, Ampere style

**Best known.** CUTLASS's multistage kernel, which cuBLAS's Ampere-class kernels resemble. Every warp does the same
work: cp.async copies of A and B tiles run several K steps ahead into swizzled shared memory, `ldmatrix` loads
fragments, `mma.sync` accumulates in f32 registers, and the epilogue stores the tile. CTAs are ordered so that
neighbours reuse an operand from the L2 cache.

```
grid = (M / 128) * (N / 128), rastered in groups; block = 4 warps, each owning a 64 x 64 tile
for s in 0..stages-1: cp_async(slab[s] <- A, B at K step s)
for k in 0..K step 32:
    cp_async(slab[(k/32 + stages - 1) % stages] <- A, B at k + (stages-1)*32)
    wait_for(slab[k/32 % stages])
    for kk in 0..32 step 16:
        a = ldmatrix(A slab, kk);  b = ldmatrix(B slab, kk)     # next fragments loaded one step ahead
        acc[4][8] += mma_m16n8k16(a, b)
store(C tile <- acc)
```

**Row.** `WORK=w2x2 TILE=mma_m16n8k16_f16_f32/f4x8/k2 STAGE=d3/smem-async/p2 RASTER=gm8`. Volta's two-stage kernel
is the `mma_m8n8k4` atom with `STAGE=d2/smem`. Without tensor cores (f32 operands) the same loop is
`WORK=t32x16 TILE=f4x8`: 512 threads, each holding a 4 x 8 register tile.

**Verdict: reproduces; better with f16 accumulate.** `matmul.square.4096`, RTX 5090:

| Row | Time |
| --- | --- |
| cuBLAS | 641 |
| Pinned CUTLASS shape, `w2x2 …/f4x8/k2 d3/smem-async` | 681 |
| Recorded, `w2x4 mma_m16n8k16_f16_f32/f2x4 d4/smem-tma` | 627 |
| Recorded under fast math, `w4x2 mma_m16n8k16_f16_f16/f4x8/k4 d2/smem-tma` | 465 |

The f16-accumulate atom changes rounding and cuBLAS has no such mode, so the last row is a different contract, taken
only where the compile allows it. A small product is level: 512 x 512 f16 is recorded at 3.3 against `torch.compile`
3.6 and cuBLAS 6.1. The f32 product at 4096 is recorded at 1895 against `torch.compile` 2187 on the RTX 5090, and at
2728 against cuBLAS 2459 on the RTX 4090.

## Dense product, load warp and TMA

**Best known.** CUTLASS's warp-specialized kernels. On Hopper one warp issues TMA copies into a ring guarded by
barriers, and two groups of four warps issue `wgmma`, which reads its operands from shared memory by descriptor. In
the ping-pong form each group owns its own output tile, so one group's epilogue overlaps the other's products. The
launch is persistent: one CTA per SM takes tiles from a scheduler. On Blackwell consumer cards (sm_120) there is no
`wgmma`; the same skeleton runs with `mma.sync` in the compute warps (B12x: eight compute warps laid out 4 x 2 and
one load warp, 288 threads).

```
grid = one CTA per SM (persistent); block = 1 load warp + 2 groups of 4 compute warps
load warp:      for each tile from the scheduler, for k: wait(empty[slot]); tma_copy(A, B -> slot); arm(full[slot])
compute group g: for each tile assigned to g:
                    for k: wait(full[slot]); acc += wgmma_m64nNk16(desc(A slot), desc(B slot)); release(empty[slot])
                    wait_turn(g); store(C tile <- acc)          # the other group computes meanwhile
```

**Row.** `WORK=w4x2+p1 TILE=mma_m16n8k16_f16_f32/f2x8/k4 STAGE=d2/smem-tma` is the sm_120 layout. On Hopper the atom
is `wgmma_m64n<N>k16_f16_f32` with `WORK=w<4k>x1`, one instruction row per warp (`f1x<C>`), `k4`, and a stage on
every operand; `STAGE=…/p2` keeps two instruction groups in flight.

**Verdict: reproduces the tile, the transport and the load warp.** The sm_120 pin above realizes with a 288-thread
block and runs `matmul.square.4096` in 691 on the RTX 5090, against 627 for the recorded row without a load warp.

**Gap: persistent launch and ping-pong.** On the H100 the recorded `wgmma` rows trail the reference:
`matmul.square.4096` 229 against cuBLAS 176, `matmul.qkv.h4096` 99.5 against `torch.compile` 73.6. Three parts of
the reference have no spelling: a persistent CTA with a tile scheduler (a row's CTA owns exactly one output tile),
two compute groups on different tiles with overlapped epilogues, and a cluster of CTAs sharing one TMA copy. The
epilogue is also a plain register store, never a TMA store.

## Dense product, `tcgen05`

**Best known.** CUTLASS on Blackwell datacenter cards (sm_100). One thread issues the product; the accumulator lives
in tensor memory, not in registers; a pair of CTAs computes one 256 x 256 atom; separate warps schedule, load,
multiply and run the epilogue.

```
load warp:     tma_copy(A, B -> slot)                           # each CTA of the pair loads half
mma warp:      one thread: tcgen05_mma(tmem_acc, desc(A slot), desc(B slot))
epilogue warps: reg = tmem_load(tmem_acc);  store(C tile <- reg)
```

**Verdict: gap.** No `tcgen05` atom is registered, and tensor memory and the two-CTA atom have no site in the
schedule. An sm_100 card takes the `mma.sync` and TMA rows above.

## Split over K

**Best known.** Split-K divides the K range of each output tile over several CTAs and reduces the partials. Stream-K
gives every CTA an equal share of all inner iterations, so a CTA may end in the middle of a tile and the next one
finishes it and adds the partial. It needs one tile size per precision and no per-shape heuristic.

```
split-K:   CTA (tile, s): partial[tile, s] = product over K range s;  kernel 2: C[tile] = sum_s partial[tile, s]
Stream-K:  CTA c: for iteration i in [c * total / grid, (c+1) * total / grid):
               tile, k = divmod(i, iterations_per_tile);  acc += A[tile, k] * B[tile, k]
               if k is the tile's last: C[tile] = acc + partials left by earlier CTAs;  acc = 0
           leftover acc -> partial[tile]
```

**Row.** `REDUCE=g<n>k` is split-K with the combine kernel; `REDUCE=g<n>a` adds partials into the output atomically,
as B12x does for bf16 outputs. The cut pass decides it, so the pieces are scheduled like any kernel.

**Verdict: reproduces split-K. Gap: Stream-K.** A CTA's share is a fixed part of one tile's K range. Marlin's
variant, where a CTA takes a run of work that spans column slices and CTAs sharing a slice combine in order, is the
same gap.

## Product with few rows

**Best known.** vLLM's kernels for one to a few rows, where a tensor-core tile wastes most of its rows. The
dot-product kernel gives one CTA per output column: 128 or 256 threads stripe K and combine by shuffle and shared
memory. The skinny kernel makes 1 to 4 output columns per CTA with 8-wide loads and unrolls the rows, so a weight is
loaded once for all of them. For wider outputs a cluster of CTAs splits K and reduces through distributed shared
memory.

```
grid = N (one output column per CTA); block = 128
thread t:
    for r in rows: acc[r] = 0
    for k in t*8 .. K step 128*8:
        w = load_vector(W[n, k .. k+8])         # the weight row is contiguous along K
        for r in rows: acc[r] += dot(w, load_vector(x[r, k .. k+8]))
out[r, n] = block_reduce(acc[r])
```

**Row.** `WORK=t<coop> REDUCE=coop/v<n>` is the dot-product kernel. `REDUCE=coop-t` sweeps the lanes along the output
instead, which keeps the loads contiguous when the weight is stored K-major. More rows take a register tile:
`WORK=t16x8 TILE=f4x4 STAGE=d2/smem-tma`.

**Verdict: better than cuBLAS.** One row through a 2048-wide expert projection, RTX 5090, recorded
`WORK=t64 REDUCE=coop`: 3.2 against cuBLAS 8.2. Linear layers at 32 rows are recorded at 0.2 to 0.9 of
`torch.compile`'s time (`linear.sq.k3840.m32`: 20.0 against 68.3). None of this was measured against vLLM's own
kernels, and the cluster reduction has no spelling.

## FP8 and NVFP4 weights

**Best known.** The same warp-specialized skeleton with a narrower operand: `mma.sync` with `e4m3` operands (K of 32
per instruction) for FP8, and the block-scaled instruction with `e2m1` codes (K of 64) and a scale per 16 elements for
NVFP4, the scales copied by TMA beside the codes. B12x is the fastest NVFP4 kernel on sm_120; a tuned one reaches 83%
of the card's peak. Activations are quantized by a separate kernel first.

```
load warp:     tma_copy(A codes, A scales, B codes, B scales -> slot)
compute warps: acc += mma_m16n8k64_block_scale(a_codes, b_codes, a_scales, b_scales)     # f32 accumulate
epilogue:      store(C tile <- acc)
```

**Row.** `TILE=mma_m16n8k32_e4m3_f32/…` for FP8 and `TILE=mma_m16n8k64_e2m1_f32/…` for NVFP4, staged with
`smem-tma` or `smem-async`; the codes and the scales are copied as complete K-contiguous rows. The activation's group
scale is a per-chunk statistic of the consuming kernel, or a kernel of its own under a cut.

**Verdict: reproduces the atom and the transport; not compared.** Recorded rows exist on the RTX 5090 (for example
`WORK=w2x2 TILE=mma_m16n8k64_e2m1_f32/f1x1/k8 STAGE=d4/smem-tma`). The reference kernels live on the serving path, so
no kernel-level comparison against B12x exists. Whether the load warp composes with the block-scaled stage was not
checked.

## 4-bit integer weights

**Best known.** Marlin. Eight 4-bit weights per 32-bit word, shuffled offline into fragment order; the kernel
dequantizes in registers with two bit operations and one multiply-add, then issues `mma.sync`. A four-stage cp.async
pipeline evicts weights from the L2 cache first, and the grid is one CTA per SM taking equal runs of work.

```
w = packed_word(B slab)                         # 8 weights, already in fragment order
b = (bit_select(w, lo), bit_select(w >> 4, hi)) * scale - zero       # in registers
acc += mma_m16n8k16(a, b)
```

**Verdict: gap.** The packed operand a contraction recognizes is a byte decoded through a table of value pairs, or
an FP8 byte. A 4-bit integer code dequantized in registers has no recorded row, and the striped grid is the Stream-K
gap.

## Mixture of experts

**Best known.** Group the tokens by expert so that every output tile reads one expert's weights, then run a product
whose tile scheduler walks the groups (CUTLASS grouped product, vLLM's Triton kernel). B12x goes further on sm_120:
one resident kernel routes, packs rows, quantizes activations, runs gate/up, the activation, down, and the weighted
scatter, with a grid barrier between phases.

```
order = sort tokens by expert;  groups = segments of order
CTA takes (expert e, M tile of group e, N tile):
    acc += mma(x[order[M tile]], W[e][:, N tile])               # a gather on A, one weight for the tile
    out[order[M tile], N tile] += route_weight * acc
```

**Row.** The expert program is one fused kernel over (token, chosen expert): the expert id indexes the weights,
and gate/up, the activation and down are fused. Its recorded rows on the RTX 5090, 64 tokens with 8 experts each,
are `WORK=t512 REDUCE=coop` at 680 and a tensor-core tile at 790.

**Verdict: gap.** A tile whose rows read different experts cannot share a weight fragment, which is why the
cooperative row wins over tensor cores. Grouping needs a sort or a count per expert (the scatter-add gap) and a tile
that takes its expert from the group. No comparison with the reference kernels was measured.

## Convolution

**Best known.** cuDNN picks per shape among an implicit product (the kernel computes input addresses instead of
building the unfolded matrix), Winograd for 3 x 3 filters, and FFT.

```
implicit product:  A[m, k] = input[n, c, y*stride + r, x*stride + s]   with m = (n, y, x), k = (c, r, s)
                   then the dense product above, with this address computation in the A loads
```

**Row.** A dense or depthwise 1-D convolution and a dense 2-D convolution lower to a contraction whose operand is
read through an index map, so the dense product's rows apply.

**Verdict: reproduces the implicit product. Gap: Winograd and FFT**, which are other algorithms and not schedules.
No row was compared with cuDNN.

## Attention, prefill

**Best known.** FlashAttention-2. One CTA per block of query rows; warps are stacked along the rows, so no warp reads
another's results. For each block of keys the CTA computes scores on tensor cores, updates the row maximum and
denominator in registers, rescales the output accumulator, and adds the weighted values. Keys and values are copied
with cp.async one block ahead: the next keys during the value product, the next values during the score product. A
causal CTA starts at its diagonal block and masks only that one. Tiles are 128 query rows by 32 to 128 keys.

```
grid = (query blocks, batch, heads); block = 4 warps, 32 query rows each
Q = stage(query block)
for j in key blocks (causal: only up to the diagonal):
    cp_async(K[j+1]);  S = mma(Q, K[j]^T)                       # scores
    m' = max(m, rowmax(S));  P = exp2(scale * S - m');  l = l * exp2(m - m') + rowsum(P)
    cp_async(V[j+1]);  O = O * exp2(m - m') + mma(P, V[j]);  m = m'
store(O / l)
```

**Row.** The chunk tier is this kernel. `WORK=w4x1` stacks the warps; `TILE@…/twist` names the value product, its `k`
the key block; `TILE@…/twist.1/inner` names the score product; `STAGE@…/twist` is the transport of keys and values.
FlashAttention-2's own tile at head size 128, 128 query rows by 64 keys, is:

```
WORK=w4x1
TILE@map.1/twist=mma_m16n8k16_f16_f32/f2x16/k4
TILE@map.1/twist.1/inner=mma_m16n8k16_f16_f32/f2x8/k4
STAGE@map.1/twist=d1/smem-async
```

**Verdict: reproduces; better for full attention.** RTX 5090, PyTorch's attention is FlashAttention-2:

| Program | Recorded | Fast math | PyTorch |
| --- | --- | --- | --- |
| `attention.hd64.fused` | 7.4 | 7.1 | 8.7 |
| `attention.hd128.fused` | 13.1 | 11.5 | 16.2 |
| `attention.hd128.causal.fused` | 13.5 | 12.5 | 11.4 |
| `attention.hd256.causal.fused` | 40.2 | 34.8 | 29.4 |

A bench of the deployed pick in November 2026 agrees: full attention at head size 128 ran 15.1 against PyTorch 18,
the causal one 15.5 against 13. The recorded rows hold one instruction row per warp (16 query rows, `f1x…`) on four
warps, not FlashAttention-2's tile. That tile realizes but needs every register and spills: pinned on
`attention.hd128.fused` it runs 37.

**Gap: causal and head size 256.** The causal kernel is 1.18 times PyTorch's time and head size 256 is 1.37 times.
The schedule is the same as the reference's; the difference measured on an A100 was in the code of one key block
(shared-memory addresses computed again per block, twice the softmax arithmetic, a mask test per element), which is
lowering work and not a missing row.

## Attention, Hopper

**Best known.** FlashAttention-3. A load warp group drives TMA; one compute group per 64 query rows issues `wgmma`,
scores with both operands in shared memory and the value product with the weights in registers. Two compute groups
alternate products and softmax under barriers (ping-pong), and inside a group the softmax of one key block runs while
the next score product is in flight. It reports 1.5 to 2 times FlashAttention-2 on an H100.

```
load group:      tma_copy(K[j], V[j] -> ring)
compute group g: S_next = wgmma_async(Q, K[j+1]^T)              # in flight during the softmax below
                 wait_turn(g);  P, m, l = online_softmax(S)     # the other group multiplies meanwhile
                 O = O * rescale + wgmma(P in registers, V[j])
```

**Row.** The chunk tier with `wgmma` atoms:

```
WORK=w4x1
TILE@map.1/twist=wgmma_m64n128k16_f16_f32/f1x16/k4
TILE@map.1/twist.1/inner=wgmma_m64n64k16_f16_f32/f1x8/k4
STAGE@map.1/twist=d2/smem-async
```

**Verdict: reproduces the tile and the instruction.** H100, recorded: `attention.hd128.fused` 10.7 against PyTorch's
FlashAttention-2 at 10.9, and 9.6 under fast math; head size 256 causal 19.3 against 17.1.

**Gap: ping-pong and overlap.** One compute group per CTA, no second group to alternate with, and no softmax
overlapped with a product in flight. Parity with FlashAttention-2 on this card therefore means the reference's
reported 1.5 to 2 times is still open; FlashAttention-3 itself was not measured.

## Attention, `tcgen05`

**Best known.** FlashAttention-4 on sm_100: accumulators in tensor memory, two softmax warp groups with one thread
per row, a correction group, one group for products and TMA, and a rescale only when the row maximum grows by more
than a threshold. It reports 71% of a B200's peak.

**Verdict: gap**, for the reason of the `tcgen05` product: no atom and no tensor memory.

## Attention, decode

**Best known.** Flash-Decoding and FlashInfer. With few query rows a grid of query blocks leaves the card idle, so
the keys are split over CTAs; each CTA writes a partial output and its log-sum-exp, and a second kernel merges them.
The query heads that share one key/value head are packed into the rows of one tile. FlashInfer chooses the split at
run time from the number of CTAs the card can hold. With one query row it uses no tensor cores: threads span the head
size with vector loads.

```
CTA (kv head h, key chunk s): Q = queries of the heads sharing h         # packed into the tile's rows
    O_s, m_s, l_s = flash_attention(Q, K[chunk s], V[chunk s])
kernel 2: m = max_s m_s;  l = sum_s l_s * exp(m_s - m);  out = sum_s O_s * exp(m_s - m) / l
```

**Row.** The split is the cut pass's: `PLACE=fuse REDUCE@map.1/twist=g8k` beside the chunk tier's tiles and
`WORK=w4x1` gives pieces of about 256 keys and a combine kernel. The form without tensor cores is
`WORK=t8x16 REDUCE=coop/v16`.

**Verdict: better than FlashAttention-2.** 64 query heads over 8 key/value heads, 2048 keys, measured in September
2026: RTX 5090 12.3 against FlashAttention-2 15.3; A100 17.9 against Inductor 42. Two limits: the split count is
fixed when the program is compiled, from evidence, and unpinned it stops at 32 pieces.

## Attention over a paged cache

**Best known.** FlashInfer reads keys and values through a page table, each thread resolving the page of the vector
it copies with cp.async.

**Verdict: gap.** A paged operand is read straight from global memory. TMA encodes one base address on the host, and
the cp.async stage cannot resolve a page per element, so both transports are refused on a buffer of several pages.

## Recurrence with a matrix state

**Best known.** FlashInfer's Gated DeltaNet kernel: one launch, one CTA per (sequence, head). Three warp groups share
shared memory under barriers: one loads and stores with TMA, one does the key/key work, one the query/key work. Each
chunk of 64 rows inverts a unit lower-triangular matrix by blocks on tensor cores and then updates a 128 x 128 f32
state. The decode kernel streams the state in tiles through a two-stage pipeline.

```
grid = sequences * heads; block = load group + two compute groups
for each chunk of 64 rows:
    load group:    tma_copy(q, k, v, gates -> shared memory)
    group A:       T = strict_lower(k k^T * decay);  T = block_inverse(I + T)        # tensor cores, see below
    group B:       u = T (v - k S);  out = q S + (q k^T * decay) u
    both:          S = decay * S + k^T u                                             # state stays in the CTA
store(out, S)
```

**Row.** A static loop over one matrix state, with maps and additive products, takes the register schedule:
`WORK=w<M>x1`, an f16 atom with `f1x<N>` covering the state's columns, and `STAGE=d1/reg`, which keeps the state in
registers across all steps of one launch. Any other recurrence is one launch per step over a state buffer.

**Verdict: gap.** The recurrence of a 27B-class layer (48 heads, head size 128), RTX 5090, 256 rows: FlashInfer 38
(80 the way vLLM calls it); Emmy partly tuned 2,338; eager PyTorch 309; `torch.compile` 216. Three parts are missing.
The chunk's work is nine kernels where the reference fuses one launch per head. The solve is traced as a 63-step
forward substitution in f32, not a blocked inverse on tensor cores. And warp groups with different roles inside one
CTA have no spelling beyond the load warp. At 64 rows the program does not compile: the cut pass fails on one body
whose normal form does not settle, a compiler bug.

## Recurrence, one channel per thread

**Best known.** RWKV's kernel: one thread per (batch, channel), a serial loop over time with a few scalars of state.
Mamba's selective scan is an associative scan over pairs `(a, b)` combined as `(a1 * a0, a1 * b0 + b1)`, run as a
block scan per channel with the running prefix carried between chunks of the sequence.

```
RWKV:  thread (b, c): for t in 0..T: state = step(state, x[b, t, c]);  out[b, t, c] = read(state)
Mamba: CTA (b, d): for each chunk: (a, b)[0..chunk] = block_scan(pairs of chunk, seed = carried);  carried = last
```

**Verdict: reproduces RWKV**, as the per-cell serial fold. **No evidence for Mamba:** a scan with a twisted combine
is a scan root and takes the prefix scan's row, but no such program has been traced or measured.

## Small triangular inverse

**Best known.** FlashInfer's `CollectiveInverse`: a 64 x 64 unit lower-triangular matrix inverted in place in shared
memory by one group of four warps. Diagonal 8 x 8 blocks are eliminated one thread per row; then blocks merge 8 to
16 to 32 to 64, each merge two small tensor-core products.

```
for each 8 x 8 diagonal block: gauss_eliminate(block)                               # one thread per row
for size in 8, 16, 32:
    for each pair (L11, L21, L22) of that size:  L21 = -L22 * (L21 * L11)           # two mma products
```

**Verdict: gap.** A triangular solve reaches the compiler as its serial substitution, and no rewrite recognizes it. A
blocked inverse would have to be traced as such, or added as a rewrite rule; the schedule could then tile its
products.
