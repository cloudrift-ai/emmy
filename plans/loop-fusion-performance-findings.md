# Loop fusion: performance findings from the golden refresh (2026-09-24)

Found while refreshing every golden after loop fusion started deciding regions from the graph and rolling every
unrolled recurrence. The refresh records a decent baseline, not the best one; everything below is left for future PRs.
Numbers are from the refresh agents' runs on each card and are single measurements unless stated.

## Whole layers are one kernel now

- **What changed.** The Qwen3-0.6B layer (kernels golden-bench goldens) lowers to ONE kernel at s1 and s512, where the
  old set had 9 targets. Qwen3.8-27B V100 layers regroup the same way (1 of 31-91 targets survive per file), and the
  DeepSeek-V4 post-attention block lowers to 5 kernels per width instead of about 35.
- **The unscheduled greedy cannot run them.** It hangs past the 60 s bench watchdog on the whole-layer kernel (V100,
  A100, H100). Every such target needs a hand-picked cut route today.
- **Cut routes are far slower than the old kernel set.** Qwen3-0.6B, best routes found so far:

  | Card | Shape | Route | Now | Old kernel set |
  | --- | --- | --- | ---: | ---: |
  | H100 | s512 | all 31 seams cut | 2294 us | 64 us |
  | H100 | s1 | 9 cuts (+ q/k projection and q/k norm stat) | 168.6 us | 17 us |
  | A100 | s1 | 9 cuts | 530 us | ~31 us |

  With every seam cut, the score and softmax pieces run as scalar code (200-630 us each on the H100). At s1 the q/k
  projection piece is 87 us because attention-internal seams carry the o_proj column axis, which makes the Q/K/P
  workspaces 128x redundant. The greedy also picks bad schedules for the pieces: an A100 s1 V projection at 7.8 ms and
  gate/up at 329 us, where a GEMV should take single-digit microseconds.
- **More cuts are not better.** On a Qwen3.8 GDN kernel, 2 cuts gave 35.8 ms and all 4 depth-1 cuts gave 215 ms.

Next step: make the greedy (or the prior) choose a route for a whole-layer kernel that reproduces the old kernel
boundaries (norm, qkv, attention, o_proj, MLP), and seed each piece with the matching old schedule family.

## The recurrence roller on GDN (Qwen3.8-27B, V100)

- The rolled recurrence `slice_unsqueeze` kernel runs at 129 ms. The unrolled chunk kernels it replaces totalled about
  20 ms. It writes all 61 step states.
- `matmul_reduce_81b2ae` became serial with heads as its only free axis: the greedy emits 48 threads for about 45 G
  scalar MACs, it has no cuttable seams, and WORK/REDUCE pins do not change it. Left unrecorded in the refresh.
- `copy_65_steps0` is serial too.

Next step: keep the roller's smaller IR but give the rolled loop parallel work (heads x batch x value columns), or leave
short recurrences unrolled when the rolled kernel loses its parallelism.

## GDN whole-layer pre and post kernels (Qwen3.8-27B, every quant, V100)

- **Pre kernel** (`k_conv1d_linear_mean_reduce_*`: norm, in_proj, conv1d): the root has no free axis and compiles to
  `if (_gid < 1)`, one thread for the whole layer. No single cut removes that root. Left unrecorded in the refresh.
- **Post kernel** (`k_linear_matmul_mean_reduce_*`: out_proj + MLP): one thread per token, and it recomputes the GDN
  output and out_proj (5120x6144) inside every MLP element. Five cuts at depth <= 2 give 6 kernels whose pieces still
  carry 5120x6144x64x128 nests; only a deep composed cut at every projection would separate them.

Both look like fusion decisions a schedule cannot repair: fusion should not merge a producer whose recompute cost
grows with the consumer's extent, and a region should keep a free axis at its root.

## Compiler errors hit while cutting

- `PLACE@map.4/map=cut` on the GDN pre kernel (AWQ) crashes the compile: `ValueError: Lambda body reads ['_r0'] it does
  not bind`. The 4-seam depth-1 composition hits the same error on AWQ, and on GPTQ it compiles but a piece hangs.
- A tune-DB perf row with knobs `{LOOPIFY: '0'}`, written by a fallback record, makes the next compile of that kernel
  raise "register schedule accepts only WORK, TILE and STAGE". Workaround: a fresh tune DB.

## Composed cuts: the V projection piece loses its name (fixed or in progress)

Cutting both the attention output and the V projection of the s512 layer makes the V-projection piece carry the
delegated zero prologue of the attention-output atomic buffer, and its kernel name comes out with no base
(`__zp1048576`). It then writes shared memory out of bounds (CUDA_ERROR_ILLEGAL_ADDRESS on H100 and V100). Being fixed
in the golden-refresh PR; recorded here because the s512 goldens were refreshed around it.

## Tooling gaps that slowed the refresh

- `emmy tune` does not work, so every schedule was pinned by hand with `EMMY_KNOBS` and `--record-greedy`.
- Seam paths from a separate tile lift (`golden._lifted_target`) are spelled differently from the compile's own cut
  pass (`map.1/inner.1/...` vs `map.1/inner.2/...`), so pins built from it silently do not apply. There is no CLI that
  lists a kernel's cuttable seams in the spelling `EMMY_KNOBS` accepts; the agents hooked `cuttable_seams` inside the
  compile.
- Multi-millisecond kernels need `--warmup 10 --iters 20`, or the isolated re-bench exceeds the 10 s GPU cap.
- Fast math is the default, so a std-lane `--record-greedy` needs `EMMY_FAST_MATH=0` explicitly; the refresh-golden
  skill only mentions `EMMY_FAST_MATH=1`.
