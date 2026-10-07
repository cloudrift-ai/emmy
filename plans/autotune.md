# Autotuning: from one kernel to a whole program

Status: open, 2026-10-06. `emmy run --bench --tune N` (PR on `feature/run-tune`) is the minimal working version: it
tunes the one scheduled kernel of a program. This plan records why it is built the way it is and what comes next.

## What the minimal version does

The program's one scheduled kernel is enumerated under the greedy kernel set, so the search only ever proposes rows
the compiler offers. The schedule prior ranks every row and its ten best are measured first. After that, each batch
of eight is the rows with the highest expected improvement under a Gaussian process fit to the log latencies
measured so far. The process sees each knob value split into its parts (an atom, a fragment count, a `k` step, a
staging depth), so two rows that differ in one tile dimension sit close together. Every row is benched exactly as an
`--ab` row is. A row that fails to compile, does not realize, fails to bench or is flagged (a wrong answer among
them) counts as a failure. Clean rows land in the tune DB, so the next compile picks the fastest. Nothing is written
to a golden file; the existing `--record` / `--record-greedy` flow does that.

## Why this algorithm

The old tuner (MCTS with an online prior, removed in PR #1000) spent most of its measurements far from the best: over
90% of benches after warmup were at least 2× slower than the best row. A configuration is a flat set of coupled
knobs, and its speed is known only once every knob is set, so a tree that fixes knobs one at a time is the wrong
shape. Earlier experiments with the prior as a stand-in for latency also showed that coordinate descent works on
some kernels and not on others, where knobs have to change together.

Three experiments led to the current choice. Scripts and raw results stayed in a session scratchpad; the numbers
below are what they showed.

**1. The prior as the objective (53 kernels from the repository goldens, median 21k rows, largest 174k).** Each
search ran on the prior's own scores, with 8 seeds. Cells give how often the best row found was among the space's 10
best:

| Method | 50 benches | 200 benches |
| --- | --- | --- |
| Bayesian optimization | 50% | 91% |
| Coordinate descent (one knob part at a time) | 42% | 77% |
| Multi-start local search | 41% | 79% |
| Genetic algorithm | 11% | 79% |
| Simulated annealing | 24% | 66% |
| Random | 6% | 20% |

The prior's own error near the top is large: it ranks the recorded golden row at the 6.5th percentile of its
kernel's space (median), and in only 27% of kernels inside the top 1%. Taking the prior's top N alone reached the top
10 in 32% of runs at 200 benches once that error was simulated.

**2. Start points (51 kernels, prior as the objective).** Golden rows of related kernels (the same kernel family,
same card first), mapped into the target's space, land at the 1.4th percentile (median); the best of five at the
0.06th. Starting from them reached the top 10 in 49% of runs after 10 benches, against 5% from the prior's top 10.
Adding the prior as a decaying weight on expected improvement (πBO) did not help at any budget.

**3. Real latency on the RTX 5090 (four single-kernel programs, 100 benches, 2 seeds).** Best µs found:

| Kernel | Greedy pick | Local search | Bayesian optimization |
| --- | --- | --- | --- |
| fp16 matmul 1024² (`matmul.square.1024`) | 15.6 | 13.8–14.3 | 14.5–15.1 |
| f32 attention, s512 (corpus `qwen3emb/sdpa-s512`) | 6980 | 6417 | 4280–4430 |
| fused norm-linear (corpus `fused/norm-linear-f16-warp-masked-m`) | 9.47 | 2.11 | 2.11 |
| f32 linear (corpus `qwen3emb/matmul-qproj-s512`) | 39.2 | 39.1 | 39.3 |

Starting from sibling golden rows and starting from the prior's top 10 gave the same results on real latency. That
contradicts experiment 2, which judged sibling rows on the prior's scores, so the minimal version uses the prior's
start only. Local search stalls in a local optimum on the attention kernel; Bayesian optimization does not, and it
loses only a few percent on the matmul. The winning rows of the matmul, attention and norm-linear runs re-benched
clean under the wrong-answer check.

## Limits of the minimal version

- **One scheduled kernel.** A bare pin reaches every kernel of a program, so a program with several is refused.
  `--kernel NAME` tunes one kernel of a layer golden as its own program instead (seconds per compile, against about
  100 s for the layer on the dev-box 5090); since #1077 its rows file under the identity the layer compile reads.
  The kernel times differently alone than in its layer, so a winner still needs a whole-layer re-bench.
- **The kernel set is fixed** at the greedy pick. Cut and split arms are not searched.
- **The whole space is enumerated.** Measured up to 285k rows (35 s to enumerate, 22 s to score). Spaces past the
  enumerator's budget would need a walk that projects a requested row onto the nearest offered one; the row-to-leaf
  walk the pool sampler uses is the starting point.
- **One process per call.** Each `--tune` call re-traces, compiles the greedy pick and benches it. Measurements from
  an earlier call are not reused as observations, even though they are in the tune DB.

## Next steps

1. **Tune every kernel of a program in one call.** `--kernel` does one kernel by hand; a program-wide tune would walk
   its kernels, spend the budget by each kernel's share of the program's time, and re-bench the layer at the end.
2. **Search kernel-set arms.** Price each cut or split arm as the sum of its pieces' best measured rows, tune the
   pieces of the few arms the placement prior ranks first, and keep the cheapest. This is the outer choice the old
   two-level tuner made, now over measured pieces.
3. **Start from the tune DB.** Feed rows already measured on the exact kernel to the process as observations, and add
   golden rows of related kernels as start points where a kernel has them. Re-measure on more kernels whether sibling
   starts help on real latency.
4. **Train the prior on every measured row.** Each pool now has one positive and unmeasured negatives, some of them as
   fast as the winner. The rows a tune measures are real losers and near-winners; feeding them to the fit is the
   cleanest way to sharpen the prior near the top. Nightly refresh owns refits.
5. **Measure the method on more kernels.** Four kernels on one card is a small sample. Repeat experiment 3 on a model
   layer's kernels once step 1 lands, on the 5090 and one other card.
6. **Bound the cost of large spaces** with the nearest-row walk above, and drop full enumeration when it lands.
