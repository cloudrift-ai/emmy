# Hopper: the warp-group MMA (`wgmma`) tier for GEMM and attention

Status: stages 1-3 landed 2026-09-10 (PR #775): the six atoms, the legality rules, the five kernel-IR statements with
the generated PTX prelude, the warp-group drain, and the corpus case `matmul/f16-wgmma-ss-tma-sm90.yaml`, which
passes offered, realized, built and correct on the rented H100. Stage 0 became the golden-bench H100 lane in the
same PR (the mma.sync tier's numbers). Stage 4 (attention: Q staged once, P in registers) and stage 5 (H100
hardware golden, freeze rows, the paper) are open.

## Goal

Reach cuBLAS-class GEMM and better-than-FA-2 attention on H100 through the existing schedule algebra: a family of
warp-group atoms in `TILE`, one legality rule on the warp grid, two on `STAGE`, and the asynchronous instruction
discipline in the kernel IR. No new codec, no change to `WORK`, no second pipeline.

Acceptance, measured on one H100 SXM with the CLI (`emmy run --bench`, `--golden FILE`), same process for both arms:

| kernel | shape | target |
| --- | --- | --- |
| bf16 GEMM, fp32 accumulate | 4096 × 4096 × 4096 | ≥ 0.9× cuBLAS latency (cuBLAS reaches 70-80% of the 989 TFLOPS peak). Measured 2026-09-10 at 2048³ f16, K-contiguous weight, strict: n256 `w8x1 f1x32/k4 d3/smem-tma` 34.0 µs vs cuBLAS 24.6 (0.73×); the best mma.sync row 69.3 |
| causal prefill attention, hd 128 | batch 1, 32 heads, 2048 keys | ≤ 0.7× eager FA-2 latency (about 1.4× faster) |

Beyond that is FA-3's overlap of softmax with the two GEMMs, which is a separate scheduling change (see "Out of scope").

## Where things stand

What exists and carries over unchanged:

- The TMA transport (`lowering/kernel/_stage.py`, `TmaTransport`): box copies with the 128-byte hardware swizzle onto
  a per-slot mbarrier ring, `pick_swizzle_atom` choosing the swizzle, and the cp.async software swizzle that matches it
  bit for bit. The 128-byte swizzle is the canonical layout `wgmma` descriptors expect for 16-bit K-major operands,
  and its 64-element row equals the `k4` chunk of the current cell.
- The producer band (`Work.producer`, `_producer_band_kloop`): elected TMA arrive, "empty" mbarrier release ring,
  named barrier over the compute band, `SetMaxNReg` redistribution (24 producer / 240 consumer registers).
- The atom registry (`ir/atom.py`): a logical warp cell whose `instruction_shape` may differ from its `shape` (the
  Volta precedent), `target_feature` gating atoms per target, and `c_to_a_repack`, the flag the chunk tier asserts
  for the P→A register handoff.
- The arch-specific compile (`backend/cuda/program.py`, `_nvrtc_options(arch_specific=True)` → `sm_90a`), already
  used by TMA and the block-scaled fp4 cell.
- Grouped rasterization, the cross-CTA `REDUCE` split, and the online prior's hardware regime features.

What the current instruction set buys on H100: `mma.sync` peaks at roughly two thirds of the `wgmma` rate, real
kernels land near half of peak for GEMM and at FA-2's ~35% for attention. So the ceiling, not the tuning, is the
problem, and every measurement of the current tier on an H100 is a baseline, not a result.

## Design

### `WORK`: unchanged; the worker stays a warp

Decision, 2026-09-10: no new worker kind. The `wgmma` cell is a 16-row WARP cell that the hardware executes four
warps at a time, the way the Volta atom is a logical 16 × 16 cell the PTX spells as four `m8n8k4`. `WORK` keeps
fixing the launch on its own (`w8x1` is 256 threads whatever the atom), and the atom supplies the cooperation width.

Why not a `wg` kind with a 128-lane atom: per warp, the `m64nN` accumulator is exactly `N / 8` of today's `m16n8` C
fragments laid side by side, register for register, so the epilogue, the softmax rescale and the P→A repack keep
their lane maps unchanged. A 128-lane worker would re-express every one of those layouts over 128 lanes, add a kind
to `Work`, and buy geometries nobody uses (groups side by side along N, when one cell already spans 256 columns).

What reusing `w` costs, as legality rules on a kernel that selects a `wgmma` atom:

- **The group leaks into the grid.** The four warps of a group must have contiguous ids and stack along M. The unit
  decode is N-fastest (`m_unit · units_n + n_unit`), so the grid must be `w<4k>x1`: `w4x1`, `w8x1`, `w12x1`, …
- **Prefer more warps over a taller fragment grid.** At `f1` along M the row of warp `wm`'s fragment is
  `16·wm + lane/4`, identical to `mma.sync`. At `f2` a warp's second fragment sits 64 rows down, not 16, because it
  is the group's second cell; the layout carries that map, and enumeration need not offer it at first.
- **The producer band is whole groups on sm_90.** `setmaxnreg` is a warp-group instruction, so `+p<n>` must be a
  multiple of four warps there. That rule is needed under either design and also settles today's `w4x1+p1`, which
  is off-spec on sm_90; stage 0 measures whether it even runs.

`derive_workers` and the launch-configuration agreement check (`choices.py`) are untouched.

### `TILE`: the warp-group atoms

New `AtomKind` entries `wgmma_m64n<N>k16_{f16,bf16}_f32` with logical `shape = (16, N, 16)`, `instruction_shape =
(64, N, 16)`, `lanes = 32`, `fragment_layout = "wgmma"`, `target_feature = "has_wgmma"`. Start with
`N ∈ {64, 128, 256}`; the instruction accepts any multiple of 8 up to 256, and more widths can be added when a
measurement asks for one. The accumulator costs `N / 2` registers per thread per cell, which the existing
fragment-register budget check reads through the atom's register count.

Per warp, the `m64nNk16` accumulator is the `m16n8` C layout repeated `N / 8` times along N, and the register-A form
takes the `m16n8k16` A layout, so `c_to_a_repack` is true for the family and the chunk tier's P→A handoff keeps its
lane map. The fragment grid `f<R>x<C>` stacks cells per warp as today; `k<B>` groups `B` k16 steps, and under the
128-byte swizzle the K chunk of a K-major operand must be one swizzle row (`k4` for 16-bit), which is a `STAGE`
legality rule, not a new spelling. Every warp emits the same descriptor and the same `wgmma` call; the instruction is
`.aligned` across the group, which uniform SIMT code over the compute warps satisfies by construction.

`atoms_for` appends the family after the `mma.sync` atoms so no existing option-0 moves. `Context.has_wgmma` is
`compute_capability[0] == 9`: datacenter Blackwell (sm_100) has a different instruction and consumer Blackwell
(sm_120) has neither, so this is an equality on the major, like the fp4 gate.

### `STAGE`: two legality rules, no new spelling

- A `wgmma` B operand must be staged in shared memory (`d<D>/smem-tma` or `d<D>/smem-async` with the matching
  swizzle); the direct `""` stage is refused on that edge. The same holds for A unless A is the register-resident
  output of the carrier, which is exactly the P operand of P·V: staged A means the shared-memory form, unstaged
  carrier A means the register form. Nothing new to spell; the form falls out of the edge.
- The K chunk equals one swizzle row of the slab's swizzle mode.

For 16-bit types both K-major and MN-major operands are legal through the descriptor's transpose bit, so a `[K, N]`
row-major B (a plain GEMM), an `[N, K]` weight (a linear layer) and V in P·V all stage without an in-kernel transpose.
FP8 is K-major only; it is out of scope here.

### Kernel IR: the asynchronous instruction discipline

New statements in `ir/kernel/ir.py`, rendered through prelude wrappers like the `mma.sync` path (inline PTX in
`emmy_wgmma_m64n128k16_bf16_f32_ss(...)` and `_rs(...)`):

- `WgmmaDescriptor`: the 64-bit shared-memory matrix descriptor (start address, leading and stride byte offsets,
  swizzle mode, base offset) built from a slab name, a ring slot and a k step. Replaces `LdmatrixLoad` on a
  descriptor-fed operand.
- `WgmmaAsync`: one `wgmma.mma_async` cell, shared-memory or register A, with the scale-d flag for the first step
  of a fresh accumulator and the transpose bits.
- `WgmmaFence`, `WgmmaCommit`, `WgmmaWait(n)`.

The rule they enforce: a fence before the first cell after anything else touched the accumulator (the epilogue, the
softmax rescale), a commit after a chunk's cells, and a wait before any read of the accumulator and before the
slot's release on the "empty" mbarrier. The first version waits to zero at the end of every chunk. `wait_group 1`
with the release lagging one chunk is a follow-up once the plain form is correct. The redundant-sync pass
(`110_drop_redundant_syncs.py`) and the ldmatrix pairing pass (`096_pair_ldmatrix_loads.py`) must treat the wait as a
barrier on the accumulator and must not reach across the fence.

### Lowering: the warp-group leaf

In `lowering/kernel/_atom.py` the staged drain for a `wgmma` atom builds descriptors per (slot, k step) instead of
ldmatrix fragments, and issues one `WgmmaAsync` per cell of the fragment grid. Both K loops (`pipelined_kloop` and
`_producer_band_kloop`) get the fence / commit / wait around the drain and move the release after the wait. The
warp-level leaf and the warp-group leaf should share one "operand source" seam (register fragment or descriptor) rather
than fork the drain; the audit at finalization checks that the ldmatrix-only assumptions were replaced, not
duplicated.

Attention on the chunk tier under `wgmma`: Q is staged once into a slab and read through a descriptor (FA-3's sQ),
which retires the 64 hoisted query registers the memo names as the register problem; K arrives by TMA; P stays in
registers as the register-form A; V arrives by TMA and is read through the transposed descriptor. Budget at 240
consumer registers, hd 128, 128-key chunk: O 64, S 64, packed P 32. The chunked carrier is not eligible for a
producer band today (`producer_eligible` in `classic.py`, after PR #772's double-arrive finding, because
`pipelined_kloop` takes no `workers`); attention on H100 needs the band, so making the chunked carrier run through
`_producer_band_kloop` is part of the attention stage.

### Target and evidence plumbing

- `program.py`'s arch-specific predicate widens from "uses TMA or the fp4 cell" to "uses TMA, the fp4 cell or a
  `wgmma` cell".
- `hardware_id` already separates H100 from H200 by product name. `features()` gains nothing unless the prior needs
  the atom family as a knob feature; the `MMA_*` expansion reads the atom's shape and dtypes, which covers it.
- A hardware golden `h100_sm90.yaml` next to the other cards' files, and H100 rows in the node freeze
  (`scripts/freeze_node_store.py`, the `collect-node-data` skill).

## Stages

Each stage lands as its own PR. Development runs the named tests only; the full suite, lint and docs belong to
finalization, per AGENTS.md.

### Stage 0: an H100 and a baseline, before any compiler change

1. Rent one H100 SXM (`emmy vm create gpu --gpu H100 --gpu-count 1`). Keep the host bundle notes the A100 work left.
2. Run the current tier there: the 2048³ and 4096³ bf16 GEMM goldens' rows, the attention memo's shapes, with and
   without the producer band, `--strict-evidence` off. Record Emmy vs cuBLAS and vs eager SDPA in the memo, with which
   SDPA backend eager dispatched to (FA-2, or cuDNN's FA-3-class kernel if PyTorch enabled it on this card).
3. Confirm whether `+p1` under `w` runs at all on real sm_90 (the `setmaxnreg` question above).

Verify: numbers in `plans/attention-optimization-memo.md`, and a one-line statement of the ceiling the current tier
reaches on H100. This is the "remeasure before building on a number" step; it also tells the paper what it can say
today.

### Stage 1: the spelling and the atoms (no GPU)

The `wgmma` atom family, `Context.has_wgmma`, the `w<4k>x1` grid rule, the whole-group producer band on sm_90, the two
`STAGE` legality rules, the register budget through `N / 2`.

Verify: an atom test next to `tests/compiler/passes/test_fp8_mma.py` (offered at (9, 0), absent at (8, 0), (10, 0)
and (12, 0)), `test_classic_schedule_domains.py` (a bf16 GEMM at (9, 0) enumerates `w4x1` and `w8x1` rows with the
family and no `w2x4` row; the direct stage on the B edge and a `+p1` band are refused with a message).

### Stage 2: the statements and the descriptor (compile-only)

The five kernel IR statements, the prelude wrappers, the descriptor encoding, the `sm_90a` predicate.

Verify: a descriptor-encoding test against known values for a 64 × 64 bf16 128-byte-swizzled K-major slab and its
MN-major sibling (the CUTLASS `GmmaDescriptor` encodings are the reference); a render test that builds the emitted
kernel with `nvcc --cubin -arch=sm_90a` on the dev box, which needs no Hopper card. A layout test that the software
swizzle's XOR (`software_swizzle`) produces the same 8-row × 128-byte core-matrix order the descriptor declares, so a
cp.async-filled slab is legal too.

### Stage 3: GEMM

The warp-group leaf on both K loops, shared-memory form for A and B, fence / commit / wait, release after the wait.

Verify: a realization case `cases/matmul/bf16-wgmma-ss-tma-sm90.yaml` declaring capability (9, 0) so `offered` and
`realized` run on any box and `built` / `correct` run on the H100; then on the H100, `emmy run --bench` at 2048³ and
4096³ bf16 against cuBLAS, sweeping `w4x1` / `w8x1`, `N ∈ {128, 256}`, `d3`-`d5`, `+p4`, and record the winner
with `--record-greedy` into the working golden, then into `h100_sm90.yaml`. Acceptance: the GEMM row of the table
above.

### Stage 4: attention

Q staged once, K and V by TMA, P in registers, V through the transposed descriptor, the chunked carrier on the
producer band.

Verify: a case `cases/attention/bf16-wgmma-causal-chunk-sm90.yaml` at (9, 0); correctness on the H100 against
eager at (1, 32, 2048, 128) causal and GQA; the memo's shapes benched. If the row lands below the target, the
SASS loop from the memo (`emmy compile --target sm_90a --ir cuda`, `nvcc --cubin`, `cuobjdump -sass`, count the
chunk loop) says whether it is issue-bound on the exp path, in which case the memo's item 2 (one FFMA and one MUFU
per score) is the next change, not more `wgmma` work.

### Stage 5: evidence and the paper

The H100 hardware golden, H100 rows in the node freeze, model goldens for the serving smoke model and one serving
model the recipes already target on H100, and a prior refit if rank correlation on the H100 rows says so. Then the
paper's evaluation gains its Hopper numbers, or its abstract loses the word.

## Out of scope, deliberately

- FA-3's overlap: two consumer warp groups alternating through named barriers, and issuing the next QK^T before the
  current softmax. This is the last 15-20% on attention and it is a schedule choice over the groups, not part
  of the atom.
  It is the warp-specialized enumeration the paper lists as the open gap.
- Persistent kernels with a tile scheduler, cluster multicast, TMA-store epilogues: a few percent each.
- FP8 `wgmma` (K-major only, so V needs an in-kernel transpose) and the sm_100 instruction family.

## Risks

- **A layout mismatch is a silent wrong answer.** The descriptor and the slab must agree on the swizzle, the core
  matrix order and the base offset. Stage 2's encoding test and stage 3's `correct` node are the guard; never bench a
  row before its `correct` node is green.
- **The pinned resolve at attention shapes takes about 110 s** on the dev box (memo, item 6). Every hand measurement
  in stage 4 pays it; profiling it first may pay for itself.
- **The exp unit.** H100 doubled tensor throughput and not the special-function unit, so the softmax costs twice as
  much relative to the GEMMs as on A100. Without the memo's exp folding the attention target may be out of reach
  even with a correct `wgmma` tier.
- **Hardware cost.** Stages 0, 3, 4 and 5 need the rented H100; stages 1 and 2 do not. Batch the GPU work.
