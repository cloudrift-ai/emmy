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

  The gap is piece schedules, not duplicated work. The fused Loop IR reads q, k, the score and the softmax statistics
  through `col // 128` of the o_proj column, and the cut already stores them per head (the seam's `strides`), so each
  cut piece computes its value once per head. With every seam cut, the score and softmax pieces still run as scalar
  code (200-630 us each on the H100); the s1 q/k projection piece (about 2 MMAC) takes 87 us; the greedy picks an A100
  s1 V projection at 7.8 ms and gate/up at 329 us, where a GEMV should take single-digit microseconds. The only
  duplicated work left is GQA's 2x: k and v pieces are indexed per q head (16), not per KV head (8).
- **More cuts are not better.** On a Qwen3.8 GDN kernel, 2 cuts gave 35.8 ms and all 4 depth-1 cuts gave 215 ms.

Next step: fusion stays maximal, so the gains come from cuts. Make the greedy (or the prior) pick cuts by duplicated
work: rank each seam by how often its producer is recomputed (the extent of consumer axes it sits under but does not
depend on) times its cost, cut the largest first, and seed each piece with the matching old schedule family.

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

- **Pre kernel is a lowering gap, not a cut choice** (FP8, `7c37f0` and the width-64 `24b33c`): the root's stores share
  no axis (`free=()`), none of its depth-1 seams owns an output, and the `to_7` gate (`sigmoid(linear_2(normed
  hidden))`, 48x512 dot products of 5120) has no cuttable seam. After any cut set the root still computes that gate and
  copies every workspace into the 9 outputs on one thread, and hangs. It needs a grid for a multi-store root or a seam
  for the gate cone. The best route found (8 cuts) leaves 8 sane gridded pieces plus that root.

Fusion stays maximal, so both need cuts that remove the duplicated work: a cut that computes out_proj (and the GDN
output) once instead of per MLP element, and one that lifts the norm and in_proj out from under the one-thread root.

## Compiler errors hit while cutting

- DeepSeek-V4 post1 (V100): the unpinned greedy of `k_linear_matmul_softmax_mean_reduce_bcf52a` emits CUDA that nvcc
  rejects ("identifier v148 is undefined" at `add_45[a10*4096+a27] = v148`). A cut route avoids it.
- Several DeepSeek-V4 `linear_softmax` kernels fail `--strict` only with "did not use CUDA graph capture"; accuracy
  against eager passes, so the refresh recorded them without `--strict`.
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
- `EMMY_KNOBS` cannot pin one piece of a cut set apart from the others: a bare key applies to every kernel of the set.
  Recording a stored piece schedule over the greedy's choice needs the tune DB to price the alternatives.
- Multi-millisecond kernels need `--warmup 10 --iters 20`, or the isolated re-bench exceeds the 10 s GPU cap.
- Fast math is the default, so a std-lane `--record-greedy` needs `EMMY_FAST_MATH=0` explicitly; the refresh-golden
  skill only mentions `EMMY_FAST_MATH=1`.
- Under a route's `PLACE` pins, `--record-greedy` ignores piece proposals (a proposal is not evidence) and picks atomic
  split-K for Gemma post projection pieces (40 / 76 us) over the stored `w2x2` schedules (13.3 / 73 us).
- A seam listing shows the raw axes (`CutSite.axes`, e.g. the 2048 o_proj column) and not the strides the cut applies,
  so redundancy estimated from it is wrong by the stride (the refresh agents read 128x where there was none).
- The recurrence roller turns a plain chain of repeated same-shape pointwise steps (`x = tanh(x) + 0.5*x`, seven
  times) into a serial state kernel, which is then a kernel boundary. Found while writing the maximal-fusion tests.
- Qwen3-0.6B s512 P.V piece (A100, all seams cut): `out[q, ch] = sum_k P[q, ch/128, k] * V[ch, k]` runs as a scalar
  loop at about 800 us of the layer's 1464 us. With the head folded into the channel axis no mma schedule realizes on
  it. A lowering gap.
- With faster split pieces measured in the tune DB, an unpinned greedy pick still chose unsplit o_proj and down
  projections, and the strict replay reports "measured DB row(s) ... none matches any of the 4 offered candidates" for
  split partials.
