# Two compiler bugs the full-coverage record run found

Status: open, 2026-11-19. Found on the RTX PRO 6000 Server Edition card while recording every hardware-golden
program on every card. Both programs were paused there and left out of that card's golden. Neither bug was fixed in
the record PR, because each needs a lowering or codegen change that moves kernels on other cards.

## 1. Reduce loop not in canonical Loop IR (DeepSeek-V4-Flash)

**What fails.** Every tune of 8 kernels from the DeepSeek-V4-Flash V100 golden's programs 8, 9, 10 and 11 raises
`ValueError: reduce loop 'aN' is not in canonical Loop IR` in `scan_from_loop` (the tile pass's lift from Loop IR).
No row is measured, so a strict record of those programs fails. Kernels: `k_979e61c3fcbf`, `k_6f12b304e87b`,
`k_307d89c138a8`, `k_ecd8be1b721f`, `k_571832e20a57`, `k_f0c0617edbb8`, `k_073f8b9d560e`, `k_2f5512540b71`.

**Reproducer.**

Export program 8 of `recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json` as a traced-IR file (the record run did
this with `emmy compile --golden PATH --program 8`), trace it into a scratch golden for the card, then:

```bash
emmy trace _data/p8.ir.json --target sm_120 -o _data/p8.json
emmy run --golden _data/p8.json --kernel k_979e61c3fcbf --bench --tune 40
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

Run each on an sm_120 card (an RTX 5090 or a PRO 6000), and on another card to see whether it is sm_120-only. Run
under `compute-sanitizer --tool memcheck` to get the faulting instruction and address; `EMMY_DUMP_DIR` gives the
kernel's PTX to match it against.

**Where to look.** Compare the faulting access with the kernel's index bounds: a tile edge with no guard, a staging
buffer sized for the wrong tile, or a split or cut piece that reads past a boundary. Bisect by pinning the greedy's
decisions one at a time to the next-best value until the fault goes away; that names the knob whose lowering is wrong.

**Done when.** Both kernels run clean under memcheck under the failing schedule, `--strict` passes against eager,
and a realization-corpus case pins the schedule that faulted (it must pass `correct`). Re-record the two programs on
the Server Edition and the 5090.
