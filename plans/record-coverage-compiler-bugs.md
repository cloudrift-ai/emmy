# Compiler and bench bugs the full-coverage record run found

Status: open, 2026-11-22. Found while recording every hardware-golden program on every card. Bugs 1 and 2 were found
on the RTX PRO 6000 Server Edition; their programs were left out of the goldens. Bugs 3 to 6 showed up later in the
same run. None was fixed in the record PR: 1 and 2 need a lowering or codegen change that moves kernels on other
cards, and 3 to 6 change what the bench or the record measures.

## 1. Reduce loop not in canonical Loop IR (DeepSeek-V4-Flash)

**What fails.** Every tune of 8 kernels from the DeepSeek-V4-Flash V100 golden's programs 8, 9, 10 and 11 raises
`ValueError: reduce loop 'aN' is not in canonical Loop IR` in `scan_from_loop` (the tile pass's lift from Loop IR).
No row is measured, so a strict record of those programs fails. Kernels: `k_979e61c3fcbf`, `k_6f12b304e87b`,
`k_307d89c138a8`, `k_ecd8be1b721f`, `k_571832e20a57`, `k_f0c0617edbb8`, `k_073f8b9d560e`, `k_2f5512540b71`.

**Reproducer.**

The same error shows on two paths, on two cards. On the PRO 6000 Server Edition every tune of the 8 kernels raises
it. On the A100 the walks tune fine as one kernel, but a strict record of the whole program raises it while the
greedy expands its `030_cut` route candidates (`kernel_set` → `_route_candidates` → `parallel_expand`), on all four
programs and on every retry. So the failing body is one a cut route writes, not the stored kernel.

```bash
emmy compile --golden recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json --program 8 --ir torch -o _data/p8.json
emmy trace _data/p8.json -o _data/W.json --append        # into a scratch copy of the card's hardware golden
emmy run --golden _data/W.json --realization <program 8's seed row> --bench --record-greedy --strict-evidence
emmy run --golden _data/W.json --kernel k_979e61c3fcbf --bench --tune 40
```

On the A100 the seed kernel is `k_386084c79684` and the loop the error names is `a8`.

On the RTX 4090 it shows a third way: an unpinned whole-target compile of `k_073f8b9d560e` (recorded under the source
golden's arm) raises it in `_route_candidates` → `parallel_expand` → `_cut.realize` → `_join_pieces` →
`_splice_pieces` → `lift_kernel`, while building a joined cut splice, with `EMMY_FAST_MATH=1` and no pins. A CPU scan
of the 4090 working golden's 770 kernels raised it for 5 kernels lowered on their own: `k_2f65af085a5c`,
`k_7f47a8a301f7`, `k_f20a172513b9`, `k_d8c5f1cd17b7`, `k_54265a53efa9`. A kernel lowered on its own needs no GPU, so
the lift of one of these is the smallest reproducer:

```bash
emmy compile --golden <4090 working golden> --kernel k_2f65af085a5c
```

The error is CPU-side (it fires while lifting the loop), so the tune is not needed to see it: the first schedule of
`--tune` reaches it. Check whether a plain compile of the kernel under the failing schedule reproduces it without a
GPU.

**Where to look.** The check refuses an `Accum` that carries a `base` or a `dtype`. Canonical Loop IR has neither,
so one of these holds:

- a pass before the tile pass (a cut, a split, or a schedule knob the tune sets) writes an `Accum` with a base or a
  dtype and does not normalize it away;
- normalization keeps that form for this body (then canonical form and this check disagree, and the check is wrong);
- the stored body in the golden already carries it (then a restamp should have caught it).

Find which `Accum` it is and which step wrote it. A fix that changes the stored kernels is a lowering change:
restamp the goldens it moves and say which in the PR.

**Done when.** `--tune 40` on each of the 8 kernels measures rows, a strict record of programs 8–11 passes on a
card, and a test pins the failing shape (a realization-corpus case if a pinned schedule shows it, else a unit test
on the lift).

## 2. CUDA_ERROR_ILLEGAL_ADDRESS in a greedy baseline

**What fails.** Two kernels from RTX 5090 golden programs fault the GPU with `CUDA_ERROR_ILLEGAL_ADDRESS` in their
greedy baseline on the PRO 6000 Server Edition (sm_120):

- `k_584fbca2982c` (rtx5090 program 123): faults in its own **unpinned** greedy. This is generated code that reads
  or writes out of bounds, the serious one.
- `k_6d8ebf7490dd` (rtx5090 program 176): faults only under `EMMY_KNOBS="PLACE=fuse,REDUCE="` (the unsplit whole-kernel
  tune). Its unpinned greedy takes the split arm and runs. The unsplit schedule is still wrong code, and any
  evidence or prior that picks it would serve it.

**Reproducer.**

```bash
emmy run --golden emmy/compiler/pipeline/search/golden/records/rtx5090_sm120.json --kernel k_584fbca2982c --bench
EMMY_KNOBS="PLACE=fuse,REDUCE=" emmy run --golden emmy/compiler/pipeline/search/golden/records/rtx5090_sm120.json \
  --kernel k_6d8ebf7490dd --bench
```

It is not sm_120-only. On the V100 (sm_70) both programs fault in their **unpinned** greedy on every bench, with the
error naming no kernel, so nothing gets disqualified and neither program ever benches clean. The V100 reproducer: the
same commands on a scratch copy of the program traced into the V100 working golden. Run
under `compute-sanitizer --tool memcheck` to get the faulting instruction and address; `EMMY_DUMP_DIR` gives the
kernel's PTX to match it against.

**Where to look.** Compare the faulting access with the kernel's index bounds: a tile edge with no guard, a staging
buffer sized for the wrong tile, or a split or cut piece that reads past a boundary. Bisect by pinning the greedy's
decisions one at a time to the next-best value until the fault goes away; that names the knob whose lowering is wrong.

**Done when.** Both kernels run clean under memcheck under the failing schedule, `--strict` passes against eager,
and a realization-corpus case pins the schedule that faulted (it must pass `correct`). Re-record the two programs on
the Server Edition and the 5090.

## 3. A cut piece's knob is recorded at one site address and restamped at another

**What fails.** The record writes a cut piece's `REDUCE` key at the site address the piece has inside its seed
program (`REDUCE@map.1/inner`). The restamp lifts the piece alone, where the same site is `@inner`, re-spells the
key, and drops the row's time, so the row becomes a proposal. Re-recording reproduces it every time. Seen on the SE
(`k_1be59c38438d` piece `k_3fff7d74df56`, the w1x4 row at 12.1 µs) and once on the 5090. The SE route stays priced
only because the piece has another measured row in the fresh spelling (w1x2, 17.0 µs).

**Reproducer.** Record the seed on the SE working golden, then `emmy golden restamp` the file: the w1x4 row loses its
time.

**Done when.** The record and the restamp spell a cut piece's site the same way, and a test records a cut piece's
keyed row and restamps it with no change.

## 4. A bench worker that respawns loses the tune's inputs

**What fails.** After one hung kernel trips the 2 s launch watchdog, the bench worker respawns, and every later
pinned row of that tune fails with `InputsCacheMissError: run_inputs ... not cached in this worker (respawned?)`. The
tune measures nothing after that point. It lost 4 tunes on the SE (`k_f595a539af1a`, `k_eb32a4713740` among them).

**Done when.** A respawned worker gets the run's inputs again, and a test that kills the worker mid-tune still
measures the rows after it.

## 5. A standalone piece draws `position` near zero

**What fails.** When an attention piece that reads `position` is benched alone, the input draw finds no query or key
tensors to derive the positions from, so `position` is drawn standard-normal, near zero. Nearly every causal chunk is
then masked and skipped. On the RTX 4090, `k_e81edb934a14` runs 3.2 µs alone against 16.2 µs inside its program. A
tune of such a piece is up to 4x optimistic, and its candidate rows look faster than the golden's.

**Where to look.** The position draw in the CUDA backend's compare module. A piece has the sizes its parent program
gave it; the positions can be drawn from those.

**Done when.** A piece benched alone draws positions in the range its program draws, and its tune time is within
noise of its in-program time.

## 6. The record does not pick a faster tuned schedule

**What fails.** On the H100 and the A100, a tune at a piece's own sizes finds a faster schedule, but recording the seed
again keeps the slower one. H100: `k_9c0f10f0f3f8` piece `k_62c28411cac8` tuned `WORK=t32` at 6,060 µs, and the
record kept the golden's `t64` at 9,898 µs (`k_8f532d791362`: 5,963 vs 10,018). A100: the `pre-sym` and `post16` tunes
found fast rows for every piece, but the seeds' recorded time did not move.

**Where to look.** Either the tune rows are filed under other sizes or another regime than the piece has inside its
program, so the pick never sees them, or the golden's imported row wins the pick. Compare the tune row's join key with
the key the record's pick asks for.

**Done when.** A record after a tune picks the tuned row when it is faster, and a test pins that.
