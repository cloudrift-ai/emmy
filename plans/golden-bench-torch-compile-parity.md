# Golden-bench kernels: match or beat torch.compile on five cards (2026-09-27)

Supersedes the Qwen3-0.6B layer next-steps plan. The corpus is `experiments/golden-bench-2026/kernels`: Qwen3-0.6B
layer 0 at sequence lengths 1 and 512. Since #897 the layer lowers to one fused kernel deployed through a cut route,
so "every kernel" means every (card, shape) cell of that layer, timed end to end.

## Goal and exit criterion

For each of the ten cells — V100, A100, H100, RTX 4090, RTX 5090, each at s1 and s512 — Emmy's end-to-end time is at
or below `torch.compile`'s, measured in the same process, with the committed golden replayed UNPINNED from a fresh tune
DB at `-O3`, `EMMY_FAST_MATH=0`, and the replay matching eager. An approximate match is enough: a cell whose replay
fails `--strict` on a few outputs still counts, and the failing count is written in the scoreboard. A cell counts only
when all three hold. The plan is done when the lane (`emmy bench` on the recipe) confirms all ten, run once at the end.

Accepted exception (2026-09-28): V100 s1 at 72 µs vs torch.compile 61. It is launch-bound (16 launches at a ~1.8 µs
floor each); unsplit GEMVs, fewer splits and a consumer-summed split all lost on the V100's 80 SMs.

Called level (2026-09-28): A100 s512 at 182-184 µs vs torch.compile 180-181, inside run-to-run noise. The rest is
in the plain GEMMs (~4 µs each vs cuBLAS): shared-memory bank conflicts in the f2x4 drain's fragment loads, more load
instructions per mma, and no 96-row (non-power-of-two, masked M edge) tile. Compiler work, not a sweep.

## Scoreboard (µs, end to end)

| card | s512 Emmy | s512 torch.compile | gap | s1 Emmy | s1 torch.compile | gap |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| H100 | 141.5 | 80 | 1.77× | 88 | 32 | 2.75× |
| A100 40GB | 323 | 193 | 1.67× | 198 | 51 | 3.9× |
| V100 SXM2 | 855 | 638 | 1.34× | no golden | — | — |
| RTX 4090 | no golden | — | — | no golden | — | — |
| RTX 5090 | no golden | — | — | no golden | — | — |

Main after #932 and #933, before #930; eager and `torch.compile` re-measured 2026-09-27 on the same hosts. #930 re-forms
the q/k pieces, so its goldens drop one row each until phase 0 re-records them; a sweep-recorded A100 s1 file on #930
ran 82 µs, the target for that re-record. The 4090 and 5090 have had no golden since the old nine-target set was
retired; their last numbers belong to kernels that no longer exist.

## What bounds each shape

- **s512 is compute bound, and the GEMMs are most of it.** The six projections are about 16 GFLOP. H100 after #933:
  65.8 µs of GEMM pieces against a whole-layer `torch.compile` of 80; the A100's 186 µs of GEMMs run at about a
  quarter of the card; the V100's largest GEMM piece sits at 15% occupancy, stalled on global loads. Attention is
  second (A100 42 µs, H100 22, V100 89 after #932). Norms, RoPE, residuals and the cooperative reduce are a tail of
  small pieces, each a few µs.
- **s1 is bandwidth and launch bound.** The layer reads about 31 MB of weights: about 9 µs on the H100 and 21 on the
  A100 at full bandwidth. The route has about twenty pieces; at 1-2 µs of launch and tail per piece, the piece count
  alone costs more than the weight read. `torch.compile` gets 32 / 52 µs with fewer, bandwidth-saturating kernels.

## Phases

### 0. Land the open work and set the baseline

1. Landed: #931 (split-row decode check), #932 (Volta attention staging), #933 (H100 wgmma rows). #930 (split
   pricing, one weight read for q/k) lands with its re-formed pieces unrecorded.
2. Re-record the rows #930's restamp dropped. Each is a q/k piece #930 re-forms:
   - golden-bench kernels: one row in each of the five goldens (s1 and s512 on the A100 and H100, s512 on the V100);
   - `recipes/Qwen3.8-27B-AWQ-INT4/golden/v100_sm70.json` on a V100 SXM3, and
     `recipes/Qwen3.8-27B-GPTQ-Int4/golden/v100_sm70.json` on a V100 SXM2 16GB (the card each file names);
   - the V100 s512 attention row and whole-set row #932 wrote by hand from a replay: record them with
     `--record-greedy` in the same pass.

   Record from a sweep, never from the prior. The first attempt (2026-09-27) recorded from the prior and failed: at s1
   the prior's schedule for the new q/k piece made the A100 layer slower than main (235 vs 198 µs); at s512 on all
   three cards and on both Qwen3.8 routes the prior chose a pathological split for the re-formed piece, and the runs
   hung or hit the bench time cap. Sweep the pinned route with `--ab` candidates into a fresh tune DB — split factors at
   s1, `WORK` shapes at s512, thread tiers on the Qwen3.8 routes with a 600 s run cap and a 60 s per-launch timeout —
   then `--record-greedy` from a copy of that DB, promote, and verify with an unpinned `--strict` replay. A Qwen3.8
   piece that cannot get back to its old row's time is a regression of #930 to fix, not a row to accept. The prior's
   pathological split pick is itself a finding: find why it prices that split cheap.
3. Create the missing goldens: V100 s1, RTX 4090 s1/s512, RTX 5090 s1/s512. Trace, sweep `WORK` and splits into a
   fresh tune DB, record, and add the rows to the recipe.
4. Fill the scoreboard with same-process eager and `torch.compile`, and a per-piece table per cell from the recorded
   rows. Also record `torch.compile`'s own kernel list and per-kernel time (profiler) for each cell — that is the
   per-piece bar.

Exit: every cell has a committed golden, a strict-clean replay, and a measured gap.

### 1. Correctness gate at s512

Model-level `--strict` fails on main at s512 on 28-40 of 524,288 outputs, one f16 step off. Fix the rounding bugs
(f16 ops contracted into fma: fixed; the softmax scale stored at f16 where torch uses f32: in progress), not the
tolerance. After both, 4-6 outputs stay off, the same spread as between torch's own attention backends, so a green
`--strict` is not required: the reference stays eager, and each cell's remaining `--strict` failures are documented.

### 2. s512 GEMM pieces — the largest lever on every card

Bar: per piece, the same GEMM through `torch.mm` (cuBLAS) at the piece's shape and layout. A piece is done at ≤1.1× of
it.

- **H100.** wgmma rows landed in #933 (`w4x1`, n64, `d4/smem-async`). Open: TMA staging was refused on one piece;
  5 stages and a producer warp group are illegal on these pieces. Find why and make them legal, then sweep. Also make
  an illegal pin fail loudly instead of falling to a 366 µs prior pick.
- **A100, RTX 4090, RTX 5090.** `mma.sync` with cp.async. Profile the slowest piece with `ncu` (sudo) before sweeping:
  earlier A100 work found deeper rings and wider warp tiles lose, so the gap may be in the cp.async lowering, not the
  schedule space. The 5090 has TMA; try it there.
- **V100.** Latency bound at 15% occupancy. Sweep tile and pipeline (`d2` with the blocking vector copy that #932 made
  legal for attention), watching registers.
- **Short grids.** The corpus linears have short grids at s512 (a wider N tile halved them on the H100). Evaluate
  split-K or stream-K for the pieces whose grid is under one wave; #930's pricing now lets a measured split win.

### 3. s512 attention

Bar: `torch.compile`'s attention kernel time from phase 0. H100 (22 µs): the wgmma attention stage of the Hopper plan
(Q staged once, P in registers). A100 (42 µs): the known cp.async / per-chunk softmax gap; profile first. V100
(89 µs): deeper staging (depth 2) is refused today. 4090/5090: measure first.

### 4. s512 small-piece tail

Count pieces per cell and compare with `torch.compile`'s kernel count. Fusion stays maximal; the lever is the route:
prefer cut routes that keep norms, RoPE and residuals as prologues or epilogues of the GEMM pieces. Where the route
exists but loses on evidence, the fix is the prologue/epilogue lowering, not a fusion gate. Also check the
cooperative reduce piece (6.4 µs on the H100).

### 5. s1 — bandwidth and piece count

Bar: weight bytes / DRAM bandwidth per card, and `torch.compile`'s time.

1. **Every GEMV at ≥80% of DRAM bandwidth.** Per piece: bytes read / time. Splits are now priced (#930); re-sweep
   split factors per card.
2. **Fewer pieces.** At s1 the norm, RoPE and residual work is tiny; routes that keep it inside the GEMV pieces win on
   launch count. Compare the route's piece count with `torch.compile`'s.
3. **The first-launch zero-init memset** (about 4 µs per replay on the H100) that the compiler cannot move into an
   earlier kernel because it is in the program's first launch. Find a place for it or drop the need (a split that
   writes its partials without a zeroed accumulator).
4. **Graph replay floor.** Measure an empty route's replay per card, so the gap that is left is attributable.

### 6. Close out

Re-record every golden on the final compiler, replay each cell unpinned under `--strict`, then run the lane once per
card and write the RESULTS.md section.

## Parallel work

One agent per card owns its host and its cells; compiler fixes land as separate PRs from whichever agent finds them,
and the others rebase. Every agent works in its own worktree and writes nothing to the main checkout. Cards: H100 and
A100 on GCP, V100 SXM2 and RTX 4090 on CloudRift. The RTX 5090 needs a rented card: the dev box's 5090 is shared.

## Measurement rules

- Unpinned replay of the committed golden, fresh `EMMY_TUNE_DB`, eager and `torch.compile` in the same process.
- `nvidia-smi` for foreign processes before every measurement; long warmup on the V100.
- Quick single-target runs while tuning; the lane only once, at the end.
- After any Loop IR change, piece rows go stale: expect a per-piece re-record on each card.
- A slower golden row after a compiler change is a finding, never something to re-record green.
