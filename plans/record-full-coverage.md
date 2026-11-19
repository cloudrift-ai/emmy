# Record every hardware-golden program on every card we have, and refit the priors with labeled negatives

## Goal

Every hardware golden holds every program any hardware golden holds, each group measured, with candidates beside it.
The priors are refit on that data, with the candidate files read as labeled negatives. Done means: one PR, the
reproduction gate green with the new weights, `eval prior` no worse than today's weights on today's pools, and
better on the new ones.

## Start

- Starts after #1429 merges, from fresh `main`. One branch, `feature/record-full-coverage`, one draft PR opened with
  the first push. Everything lands there; finalization runs once at the end.
- Cards in scope, one hardware golden each:

  | Card | Where | Golden |
  |---|---|---|
  | RTX 5090 | dev box (+ CloudRift 5090 as a second worker) | `rtx5090_sm120` |
  | RTX 4090 | CloudRift | `rtx4090_sm89` |
  | V100 SXM2 | CloudRift | `v100_sxm2_sm70` |
  | A100 | GCP `bench-kn1386-a100-…` | `a100_sm80` |
  | H100 | GCP `bench-knptx-h100s-…` | `h100_sm90` |
  | PRO 6000 Server Edition | GCP `bench-vm-1117-1433-9617` (SPOT) | new file |

- Skipped, no card: RTX 4080, V100 SXM3, PRO 6000 Max-Q, PRO 6000 Workstation. Their goldens are only restamped if
  a code fix moves them.
- Excluded targets: the Qwen3.6-35B-A3B FP8 sparse expert kernels on the RTX 5090 (lowering gap), and the 13 RTX
  4090 groups whose numbers disagree with their golden (checked by hand before anything is re-recorded there).

## Agents

One lead (this session) and seven agents. Each card agent owns one card's files only — its hardware golden, its
candidate file, its freeze file, and the model goldens of that card — so no two agents write the same file. Each
works in its own worktree on its own branch cut from the PR branch; the lead merges card branches into the PR
branch as they finish a phase.

| Agent | Work |
|---|---|
| card-5090, card-4090, card-v100, card-a100, card-h100, card-pro6000 | phases 1–3 on that card |
| fit | phase 4 code, on the dev box; runs beside the recording |

Small issues found on the way (a lowering crash, a CLI gap, a strict record that refuses) are fixed at once:

- A fix the agent can make in under an hour, with a test, goes in as its own commit on the agent's branch. The lead
  cherry-picks it onto the PR branch and the other branches rebase onto it.
- A fix that moves kernels on other cards (a lowering change) goes through the lead only, so every card re-traces
  from the same compiler. Agents pause recording on the affected programs until it lands.
- A gap that needs real design work is written down, its groups skipped, and reported at the end.

## Phase 1 — every program on every card

1. The program set is the union of two parts, deduplicated, recomputed on fresh `main`:
   - the 247 unique programs of the hardware goldens (today a card lacks: 5090 39, A100 157, H100 165, 4090 175,
     V100 SXM2 179, Server Edition all 247);
   - the 192 programs under 1,000 nodes that only model goldens hold. These bring families no hardware golden has:
     Gemma 4 (40), Qwen3.5-122B (32, tiny), Qwen3.8 NVFP4 (39), Qwen3.8 FP8 (14), Flash-Next NVFP4 (18), Qwen3.6
     FP8 (16), DeepSeek-V4 with its u8 weights (12), OLMoE (9), Voxtral (9).
   Left out: the 15 whole-layer twins of model goldens (1,700–2,800 nodes). Their kernels repeat the pieces added
   above, and each costs hours on the older cards. They stay on their recipes' cards.
   No program is dropped by dtype: bf16, FP8 and NVFP4 programs all compile for sm_70, sm_80, sm_89 and sm_90
   (software decode below sm_89), and real recipes serve FP8 and NVFP4 on V100.
2. Each card agent exports the missing programs from the golden that holds them
   (`compile --golden PATH --program N --ir torch -o FILE.json`) and appends them on the card
   (`trace FILE.json -o GOLDEN --append`). The Server Edition golden starts from an empty trace.
3. Check: `emmy golden check` on the card's file shows the new programs and nothing else changed.

## Phase 2 — measure

Per card, in this order:

1. **Pilot.** Tune one affine-recurrence group with `--tune 40` and read how long it takes and what it finds. This
   sets the tune budget for the ~130 recurrence groups per card, which have no candidates yet.
2. **Proposals and gaps.** Groups whose rows hold no measurement — #1429's A100 and V100 proposals among them —
   tune and record first.
3. **New programs.** Tune each group of the programs phase 1 added, then `--record-greedy` under strict evidence.
   Narrow targets first, one tune DB per card.
4. **Stale goldens.** Groups where a candidate was measured more than 5% faster: re-record. If the faster row is
   numerically wrong (the suspected Ministral FP8 rows on the 4090), drop it instead.
5. Every tune's freeze file is copied back from the VM and committed with the card's branch.

Tune per group, not per program: one tune DB per card, so a kernel two programs share is tuned once and the second
record picks its rows. The 1,700–1,900-node GDN layer programs go last on the A100 and V100 SXM2, under a time cap:
#1429 found their 64-row set at 15 ms on the A100, and the V100 did not finish.

Rough size: about 7,000 tunes for the hardware-golden part, about 230 GPU-hours; the model-golden part adds an
estimated 50–60%. The lead counts the real groups after phase 1 and before any tune. The pilot in step 1 corrects the
time per tune; if one card runs far longer than the rest, the lead splits its program list with the second 5090 or a
second VM of that card (asked first).

## Phase 3 — candidates

On each card's branch, after its phase 2: `emmy candidates build` for its file, then `emmy candidates check`. A
group with fewer than 3 measured non-best schedules gets a short extra tune. Check: the candidate tests pass for
that card.

## Phase 4 — the fit change (agent `fit`, in parallel with phases 1–3)

1. The export reads each group's candidates from the candidate files. A candidate more than 1.1× slower than the
   best golden row is a labeled negative. A faster candidate is not a positive; it is reported as a stale golden.
2. The dataset format holds the negative ids. Bump its version.
3. The trainer always includes the labeled negatives and leaves them out of the uniform draw. The uniform draw also
   stops drawing measured slower rows, as its docstring already says.
4. Turn hard-negative mining back on only if CV shows it helps now that negatives are labeled.
5. Tests: the export labels, the trainer's draw, the version bump. Check on today's data: CV against today's
   weights, and the reproduction gate.

Failures as negatives come later, not in this PR.

## Phase 5 — consolidate and refit on berserk

1. The lead merges every card branch and the fit branch into the PR branch. Restamp check over every repository
   golden and candidate file: nothing changes.
2. Copy the tree to berserk (no compiler there: prebuilt extensions, `TMPDIR=~/tmp`). Run the README "Fit the
   priors" steps: import with `--repository`, export both spaces, fit both, `eval prior` both.
3. Compare the new weights with today's on the same datasets (`--compare-to`), per card and per space. Look at the
   pools the new prior gets wrong.
4. If a miss traces to a feature that cannot tell two schedules apart, fix the featurizer, bump its version, and
   refit. Repeat until the misses left are ones no feature would fix.
5. Run the reproduction gate on the final weights. Commit the weights.

## Phase 6 — finalize

AGENTS.md steps 8–28 over the whole diff: audit, docs, `make test`, `make lint`, PR body revised twice, ready for
review. The PR body names every group skipped, every row dropped, and anything that got slower.

Then tear down the VMs this job rented (the PRO 6000 Server Edition), and ask before tearing down the others.

## Risks

- A SPOT preemption on the Server Edition loses in-flight tunes; its agent copies the freeze file back often.
- A compiler fix mid-job re-keys kernels already recorded; this is why such fixes go through the lead and paused
  programs are re-recorded after.
- The model goldens' `v100_sm70` files name a V100 SXM2 16GB; the hardware golden `v100_sm70` is the V100 SXM3,
  out of scope. Check the CloudRift V100's memory before recording into a 16GB file.
- The cold deploy draws 8192 rows while the fit and gate score smaller draws; not fixed here, noted in the PR.
