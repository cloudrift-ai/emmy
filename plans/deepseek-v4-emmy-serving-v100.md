# DeepSeek V4 Flash 0731 through `emmy serve` on 16× V100 (TP8 × PP2)

Goal: serve `deepseek-ai/DeepSeek-V4-Flash-0731` through the Emmy vLLM plugin on the 16× V100 SXM3 host — Emmy
compiled kernels for the hyper-connection stream mixing, norms, shared expert and routed experts; the pinned 1Cat
sm_70 fork (`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`) supplying paged MLA attention and the
serving shell — then A/B against the plain 1Cat container at an equal serving envelope.

43 layers, `hc_mult` 4, 256 routed experts at top-6 plus one shared, 3 hash-router layers. At TP8 × PP2 the first
stage owns layers 0–21 and the second 22–42.

## Where it stands

**The repository golden serves it.** Measured 2026-09-15 on the host from `main` (7e9336e6) plus #807,
`--strict-evidence`, the repository golden as the only measured evidence in the process, single stream, greedy:
2.03 s per output token (steady steps inside 1.3%), 6.1 s to first token on a short prompt and 44.2 s at 2,275
input tokens, health in twelve minutes. The 0.899 s per token of 2026-09-12 was measured against a host-local
golden that is not in the repository; the difference between the two is one election, named below. Against the
pinned fork on the same host the repository arm is 13.8× slower per output token and about 12× slower to first
token — directional, because the two arms have not yet run at one envelope (Stage 5). The evidence, the protocol
and the caveats are in `experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16/RESULTS.md`; that
report is the durable record, not this file.

**Main serves, and decode is 12× faster (2026-09-18).** `main` at `483e4cb7` — #813's retune, #818, #827, #826 —
did not boot from the golden as merged: #813's rows had three empty receipts in the symbolic post twin (an empty
schedule row is no deploy evidence), receipts for eight of seventeen route pieces at m16 and m4096, and the four
`k_div_35` rows red since its codec change, all invisible to the pooled strict decode. PR #833 re-records the three
post twins, the two M=1 post cuts and the division kernel on the serving host and drops the empty rows. Booted from
that file, strict, empty tune DB: 0.267 s per output token (3.75 tok/s), 3.7 s to first token on a short prompt
and 29.3 s at 2,155 input tokens, health in seventeen minutes; the M=1 tier deployed for the first time since #804,
3.4 ms per layer against the 69.8 ms the m16 twin cost a single-token step the day before. Against the pinned fork
on the same host the repository arm is 1.8× slower per output token. One regression: the m4096 pre-attention twin
measured 25.4 ms against 2.73 ms on the same three-kernel cut. The cause was the residual's receipt, not the
compiler — a cooperative reduce the binder ignored until #813 — and the receipt is re-recorded as the serial row,
2.82 ms from `main`. The report has the numbers.

**The re-recorded golden needed one more row out before it booted (2026-09-19).** `main` at `cab3b735` with the
golden as merged died 24 minutes in: the m16 pre-attention set held a cooperative row and a cross-CTA split row of
one kernel, tied at 3.89 µs; #843 retimed the cooperative row to 5.73 µs, the split row won, and its pieces have no
rows, so strict refused the m16 decode twins and the runner rejected the engine's step budget. With that one row
dropped the boot serves: 0.266 s per output token, `pre.chunk.m4096` 3.10 ms per layer, and the long prompt's time
to first token unchanged at 29.3 s — a single request under a 4,096-token context never makes the exactly-4,096-token
step a chunk twin takes, so its prefill rides the symbolic twins. Health took 29 minutes: the first compile on each
rank now takes about 800 s on this tree, in two boots running, against 107 s before; cause not found.

Stages −1, 1 (#651), 2 (#656) and 3 (#662) are done, and Stage 0's question — whether the compiler can produce a
schedule fast enough to serve this model — is answered yes. Gate (c) passed on the repository golden at 7e9336e6 +
#807: the server boots, answers, and its completions are coherent. Gate (d)'s greedy token-ID half: two of four
prompts agree with the fork on all 32 token ids, the other two diverge at near-ties of about 0.2 nats, and one of
those agreed in August; its layer-level tensor half was never run and an HF eager reference for a 156 GB checkpoint
stays impractical here.

## The correction this plan owes its reader

Earlier revisions of this file concluded that what held serving was a compiler gap in three named programs, and that
"measuring them harder will not help". **That was wrong, and it pointed at the wrong work.** Every one of those
programs was fixed by recording a placement cut, with no new compiler capability:

| program | before | after | how |
| --- | ---: | ---: | --- |
| `pre` chunk m4096 | 1,923,598 µs | 3,238 µs | four `PLACE@…=cut` |
| `post1` `9e578e` @ m1 | 42,278 µs | 9,866 µs | placement cuts |
| `post1` `4e26cc` @ m1 | 29,988 µs | 4,839 µs | six `PLACE@…=cut` (#799) |
| `pre1` `k_linear_mean_reduce_7fce9f` @ m1 | 29.7 s | 419 µs | the same four cuts as `pre` m4096 (#795) |

The mechanism is the opposite of what was written here. These schedules were already expressible and already on the
ballot; the defect is that **one bad measured row is binding under strict evidence**. A shape for which nobody
recorded a cut elects whatever single candidate it has, however slow. `pre1` at M=1 had exactly one measured row, at
29.7 s, and single-token decode ran at a flat ~61 s per layer until #795 recorded a better one.

Two properties made it hard to see, both fixed by #795. The boot roofline audit exempted any program whose floor sat
under `MIN_FLOOR_US`, so a mispick there was invisible however much it cost — a small floor bounds what a healthy
program costs, never a broken one. And the static M=1 decode tier deployed unconditionally even when its twins were
slower than the bucket twins they replace; it now drops when it does not measure faster.

**The diagnostic that generalizes:** read the golden's measured rows per (program, shape), take the minimum per
kernel, and treat a shape carrying one candidate as the smell. `emmy compile --golden PATH --realization NAME --ir
loop` then prints the Loop IR, where the recurring defect on this model is a dot product sitting inside a sweep it
does not depend on.

The same lesson held once more in a different place. #801 reported that the expert compile died at plan
construction with a `RangeOp` node, and read it as a lowering gap in the mxfp4 path. It was the constant-fold pass
leaving a two-lane range unfolded under a broadcast (#807 fixes it), and it was invisible to every replay command
because neither `emmy compile --ir cuda` nor a `--golden --realization` replay reaches plan construction. What
serving runs is `CudaBackend.compile` followed by `plan_from_graph` on the twin's Graph IR, and that is the call to
reproduce a boot failure with.

## What is left

### Kernel quality — the post family, and one election that goes the wrong way

The pre family is closed. #793 took the four pre-attention targets to 28.8 µs / 1.79 ms / 8.70 ms / 5.82 ms at
m1 / m16 / m4096 / dynamic; the boot audit read `pre.chunk.m4096` at 2.7 ms per layer through 2026-09-17 and the
pre decode programs stay under its threshold. That is what halved time to first token at 2.3k input tokens. On
2026-09-18 the same three-kernel cut measured 25.4 ms from `main`. The residual's receipt spelled a cooperative
reduce the binder ignored when the row was recorded and honours since #813: 128 threads on a four-element reduce.
The m16 and dynamic residuals carried the same spelling. All three are re-recorded as the serial row the old
figures measured — 2.82 ms at m4096 — and no boot has run on that file yet.

**One election is wrong before anything is recorded.** The boot audit reads `post.decode.m1` at 44.7 ms per layer
(750× its floor) from the repository golden; the host-local golden the 2026-09-12 numbers came from elected about
18 ms (294×) for the same program. #799 recorded that cut into the repository, and the election does not take it.
Forty-three layers of the difference is the 1.1 s per token between 0.899 s and 2.03 s. Since 2026-09-18 the M=1
tier deploys from `main` and `post.decode.m1` reads 3.4 ms per layer (57× its floor) from the rows #833 recorded, so
that election is settled; the expert M=1 twin is still refused and rides the wider tier — see "Owed regardless".
One mechanism that makes a recorded cut lose is now known: a schedule row carrying the identity a cut fork is
offered on reads as the fused kernel's own receipt and prices the fused arm with a piece's timing. Whether it
explains 2026-09-15 is open; that tree predates #804.

The post-family rows that table used to list were retuned by #813 on 2026-09-17: the Sinkhorn mixing kernel
`06fabe` @ m4096 from 454.8 ms to 1.07 ms, `dba017` @ m1 / m16 / dynamic / m4096 from 30.6 / 46.9 / 97.5 / 106
ms to 0.81 / 3.58 / 11.5 / 40.5 ms, `9e578e` @ m1 to 0.37 ms, `3836f9` @ dynamic from 117.8 to 8.48 ms and `8e1e80`
@ m16 to 16.6 ms; the dynamic-width Sinkhorn twin `d2b070` (74 ms) cannot take the cut and stays open, a reshape
lowering lockout #813 names. None of those rows deployed as recorded: #813's post sets had empty receipts and
receipts for eight of seventeen route pieces, and #833 re-recorded the three post twins on the serving host — the
dynamic twin at 64.9 ms, m16 at 9.9 ms, m4096 at 536.8 ms, the last because the nine pieces #813 never recorded
take the prior's schedules. The 2026-09-18 boot audit reads `post.decode.m16` at 13.6 ms and `post.chunk.m4096` at
599 ms per layer.

The restamp is done (#826), so recording is unblocked: a row recorded now is written once, against main's spellings.

### Stage 4 — image and release plumbing (not started)

No baked image exists, so every boot pays its compile. Build FROM the immutable 1Cat digest with `cupy-cuda12x` under
its own image identity — do not inherit the Makefile's default version/tag for a 1Cat 1.2.3 base — labelling the 1Cat
digest and source SHA, Emmy SHA, checkpoint revision and CUDA/NVRTC versions. The serve env plumbing carries the
fork's `VLLM_SM70_*` variables along with `--tensor-parallel-size 8 --pipeline-parallel-size 2` and
`--distributed-executor-backend mp`. The pinned config `docker/vllm-emmy-serve/models/deepseek-v4-flash-0731.env`
exists since #768 and is the single source for the twin widths; a headroom sweep on the host seals its memory values
(do not author widths off-host).

→ verify: `make serve-config / serve-goldens / serve-warm / serve-image / serve-verify` on the host; the baked image
cold-starts offline and EVERY one of the 16 workers reports its pack hit (today's verify accepts one `pack hit` line,
which is insufficient), the cubin set is unchanged, and no request-time Triton JIT occurs. Build and verify only;
registry publication is a separate approval. `serve-goldens` runs the strict decode of the golden, which
completes in minutes since #804 and passes since #826.

Measured envelope to plan the sweep against (gate (c), `--max-model-len 4096 --kv-cache-dtype fp8 --block-size 256`,
`--gpu-memory-utilization 0.90`): 30.8 GiB resident on a first-stage card and 31.75 on a second-stage one of 32, KV
cache 76,337 tokens on the first stage and 78,722 on the second (2026-09-15). KV capacity is not derivable from a
bytes/token constant here — sliding layers cache a 128-token window and the compressed layers cache compressed
entries.

### Stage 5 — the A/B (the deliverable)

Both arms now produce serving numbers, but not comparably: they ran in separate invocations, at different shapes, and
at different envelopes (`gpu_memory_utilization` 0.80 against 0.90, prefix caching off against on). A real comparison
needs one `emmy bench` run over both arms at one envelope with their order alternated inside each repeat, the way the
RTX 5090 gemma-4 experiment balances time and thermal drift, against immutable image digests and one checkpoint
revision. Profile in a separate run — profiling the fork's multi-stream execution perturbs the A/B — and include a
per-phase split (expert dispatch against attention against stream mixing) so Stage 6's hypothesis is grounded.

### Stage 6 — MXFP4 expert inputs (optional)

Only worth building if Stage 5's profile shows expert weight streaming dominates and a fused-unpack GEMM can
plausibly beat TurboMind's on Volta. Main already spells native MXFP4 expert twins; what this checkpoint still needs
is its declaration and orientation mapped onto that spelling (`quant_method: fp8` with `expert_dtype: fp4`, packed as
`w1.weight I8 [out, in/2]` with `.scale [out, in/32]`), plus tuning.

### Owed regardless

`emmy run --golden` cannot carry a correctness verdict for these twins: the twin draws its eps and count constants as
random inputs so every replay is non-finite, and the same-input reference is the route under test. #795 hit this
directly and had to fall back to reading serving output. What is missing is a finite-input replay per twin and an
independent reference on `run --golden` — the loop-IR CPU runner exists but is unexposed.

**The strict decode completes now.** The seven `k_div_*` `PLACE=cut` rows at m1 that #797 recorded took over
29 minutes each before #804 — the pre-#804 compiler sits at full CPU on the first of them for 20+ minutes — and
take 1.3 s each on main; the whole file decodes in about two and a half minutes. The goldens gate can protect this
file again; the only red rows are the four `k_div_35` rows, whose two cooperative reduces #813's codec no longer
allows together ("a second scheduled root on a projection its outputs do not partition by root"). They stay red on
purpose: #833 records the kernel again beside them (one cooperative-thread seam, 18 / 3 / 57 µs at dynamic / m16 /
m4096 against 3.5 / 3.7 / 7.3 with both), so the gate keeps naming the codec change until someone decides whether
the pair should be offered again.

**Three ways a green strict decode still fails to deploy, and what the M=1 tier needed (2026-09-18).** The strict
decode is pooled — a row passes when any kernel of its set enumerates it — while the deploy keys every row under the
kernel it decides. Three gaps came out of five boots, and a restamp owes all three checks before a boot. Same-shaped
twins must be told apart by their `S_*` signature, not mint order (#826). A measured schedule row that carries the
identity a cut fork is offered on prices the fused arm, so an empty plain row can elect a fused kernel that runs for
minutes (#826). And a receipt whose schedule row is empty is dropped by the evidence index, so a piece with only an
empty receipt has no measured row at its cut fork; the same audit must count the route's minted pieces against the
set's receipts, because a cut arm is eligible only when every piece has one (#833). The M=1 tier came back with the
M=1 division cut recorded (its piece builds since #827, 10 µs) and the `9e578e` cut recorded again at 406 µs; the
residual refusal of #799's cut did not recur. Still refused: the expert M=1 and m256 twins, whose cut minted one
kernel where it minted two and whose prior schedule runs about 4 s per launch — `emmy tune`, not a bench. The boot's
roofline audit still has no time limit. A fourth gap came out of the 2026-09-19 boot: a measured row that spells a
split or a cut whose pieces have no rows can win its fork on a retime alone, and strict then refuses at the pieces
instead of taking the next measured arm — audit every such row for piece coverage, not only the placement routes. The
symbolic post twin is measured (2026-09-19): linear in width, 276 ms per layer at 2,155 tokens, 11.9 s of the 29.3 s
to first token. Seven post receipts that spelled two cooperative reduces with one block per output cell are
re-recorded as the serial row (dynamic 64.9 → 40.3 ms at the hint, m16 9.9 → 8.8 ms, m4096 537 → 258 ms), and the boot
on that file reads 24.9 s to first token at 2,155 tokens. To bench a dynamic row at another width, bench a scratch
copy of the golden with its hint rewritten; the election does not change. The three `t256 coop-t` pieces were the
block's plain matmuls with the tile site left empty; tiled on tensor cores with staging they run 42–67× faster, and
the m4096 twin's matmul piece 51× faster under a better tile (2026-09-19): symbolic post twin 40.3 → 4.84 ms at the
hint and 18.8 ms at 2,155 tokens, m4096 post twin 258 → 121 ms, time to first token at 2,155 tokens 24.9 → 18.3 s. A
kernel-scoped A/B is a scratch golden with one receipt respelled, benched strict; sixteen cards run sixteen candidates
at once. The symbolic post twin is now 0.8 s of that 18.3 s, so the next prefill item is a measurement, not a kernel:
where the other 17 s go (experts, attention, first-request warm-up). A decode step is now measured (2026-09-19,
torch-profiler trace, the model serves eager): per token about 126 ms of Emmy kernels, 96 ms of all-reduce at a
millisecond per call over PCIe (the host's, not the arm's), 22 ms of attention. The boot audit's `post.decode.m1`
figure is an uncaptured launch loop and overstates serving. The M=1 post lead's two fully serial pieces are
re-recorded thread-parallel (851 → 52 µs per layer) and time per output token is 0.235 s, 1.6× the fork. The expert
program is done as far as rows go (2026-09-20): the M=1 expert twin offers no schedule knob under its cut and its
residual runs 1.04 s per launch, so it needs tile or reduce sites from the compiler, not a row; single-token steps
ride the m16 expert twin, whose two tiles are re-recorded (864 → 430 µs per call) for 0.214 s per output token, 1.46×
the fork. Next decode items, by what the trace says they cost per token: the `9e578e` cut's pieces, 19 ms; the two M=1
pre-attention pieces, 17 ms, whose 128 threads each recompute the 16,384-element statistic and for which no offered
cut was found by hand; `k_div_4` and `k_div_30`, 12 ms. Then the m16 post twin's four serial single-block kernels, 7
of its 8.8 ms, the same serial pattern as the M=1 pieces; the static m4096 post twin, 3.4× slower than the symbolic
twin at the same width, whose three largest `r4` pieces are 72 of its 121 ms; then the experts and Stage 4.

**The golden does not deploy on `main` any more (2026-09-22/23).** #864's restamp dropped 17 rows and keyed split rows
onto their leads' identities, so `main` at `dab7bce9` refuses all five twins under strict evidence; every twin was
re-recorded from `main`'s tree with the pieces covered again (post m16 8.8 → 6.0 ms per layer, post symbolic 4.8 → 3.6
ms at the 512 hint, post m4096 121 → 85 ms; the experts pay the removed Volta depth-2 ring: m16 430 → 719 µs, symbolic
2.4 → 22.6 ms because its root piece offers no tensor-core tile now) and the file decodes and audits clean, yet the
strict boot dies at the symbolic expert twin because serving's fresh trace no longer lowers to the golden's kernels:
between `9607133e` and `dab7bce9` the fusion of this model's post block changed from 36 kernels per twin, with the two
big softmax-matmul kernels every round tuned, to 45 with them split apart by #863, the reduction-dependency and
coordinate normalization. Stored Loop IR replays (the decode gate, `emmy run --golden`) cannot see that; `emmy trace
--serving-twins` on the tree, diffed against the golden's kernel families, can, and must precede any record.
Cooperative reductions also run 2–30× slower on `main` for the same spelling. The re-recorded file is kept on the host
and not committed. The follow-up found two causes (2026-09-23): the merge rule's copy-drop from #863, reverted on
draft PR #875, which restores this model's kernel set but moves Qwen3.8's the other way (#861 recorded on the
post-#863 sets), and #863's Loop IR normalization, for which this golden's stored loops were never re-lowered. Next:
the author settles the rule, then the golden's programs are re-lowered and every row re-keyed; only if the rule stays
as it is does a re-tune from the fresh capture follow.

**Restamped onto the fresh lowering (2026-09-23).** Each stored target was replaced by the kernel a fresh lowering of
its own program writes on `main` plus #875 (byte-identical to the serving inventory), every row re-keyed, 105 rows
re-spelled onto the closest site of the re-associated trees or re-measured, 14 dropped. The deploy's evidence index
and a per-twin kernel-coverage audit are the acceptance tests now, not the pooled decode gate, and a new gate keeps
every golden's stored targets equal to a fresh lowering (every other golden but Qwen3.5-122B and two hardware files is
stale and listed as a strict xfail). 465 measured rows, 448 live in the evidence index (365 before), 324 of 335 twin
kernels covered (283 of 330 before), nine of nine twins elect strict. Boot36b serves strict: TTFT 0.73 s warm at 5
tokens and 5.5 s at 2,155, TPOT 0.56 s (0.215 on boot33); the width-16 post twin runs 8 ms per layer, the width-4,096
one 104 ms; the probe's text degraded and the m16 expert twin's reproducer check reports 9,931 outliers over a budget
of 4 — a correctness audit of the m16 expert and post twins on this tree comes before any tuning. Third cause of the
boot deaths since 2026-09-22: #868 made fast math the default and serving publishes no precision pin, so rows recorded
under `FAST_MATH: False` are off-regime in every worker and the index is empty — boot with `EMMY_FAST_MATH=0` (now in
the boot script), or re-record under fast math. Open compiler findings: cooperative reduction on the fresh divide
kernels does not compile (a duplicate declaration, GPU-free repro in the report) and serial is 36-250× slower than the
old cooperative rows; the transposed cooperative reduce is gone from this model; the prior's cooperative picks cost
10-100× until a register split at the top reduce is pinned by hand. Next: the compiler defect, then the Qwen3.8
goldens and the other stale goldens each need the same restamp on their card.

## Operations handoff

The host's address is deliberately absent from this repo; it lives in the operator's notes and is used only inside
commands.

**Layout.** `~/emmy-serve` is the serving checkout the boot container installs with `pip install -e .` — a plain
directory, not a git checkout, so sync it with `rsync -a --delete --exclude __pycache__ emmy/ HOST:~/emmy-serve/emmy/`
before recording anything. Its emmy must be new enough to read the golden in play: an older one rejects a current
model golden with `unknown field(s): eager_us`. A boot can equally mount a revision-named copy (`~/emmy-main-<sha>`,
the 2026-09-15 boot's `~/serve-evidence/boot20.sh` does), which leaves the shared checkout alone and says what ran.
Serving evidence and golden copies live in `~/serve-evidence/`. The compiler tree for tuning work is
`~/emmy-durations/` with its own `./venv`, and `py-spy` there is the tool that names a stalled program
(`sudo -n ~/emmy-durations/venv/bin/py-spy dump --locals --pid PID`, one dump per rank, simultaneously).

**Never touch** `~/.cache/emmy/autotune.db` (the real tune DB), `~/emmy`, `~/emmy-dsv4`, `~/emmy-fix-backup`,
`~/emmy-durations/_verify/gap3-tune/` (partial rows that regress the election — never merge that DB), or
`~/.cache/emmy/verify3/` (another user's live tuning session). Another user tunes on this host: run `nvidia-smi
--query-compute-apps=gpu_uuid,pid,process_name --format=csv` before every launch, use only devices nobody holds,
never kill a foreign process, and delete nothing when done. `pkill -f` over ssh matches its own remote command line
and kills the session — ship a script and run it by path.

**Reproducing a boot failure without a boot.** A twin's Graph IR is the golden's `programs[i]` entry
(`graph_from_wire`, then `specialize_program` for a static width); serving compiles it with `CudaBackend.compile`
and then `plan_from_graph`, and only that pair shows a plan-time failure — `emmy compile --ir cuda` and a `--golden
--realization` replay stop before it. Host CPU compiles need `~/emmy-durations/venv` with `PYTHONPATH` set to the
tree under test and `LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12`, and they need a card visible: golden
evidence is keyed by the live card's name, so `CUDA_VISIBLE_DEVICES=""` empties it and every fork reports no
measured row. Pick an idle device and leave it visible.

**Two deadlines bound any forward pass** under two-stage pipelining, because the second stage blocks on the first for
the whole traversal. `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300) is raisable. The NCCL collective watchdog at
600 s is a compiled-in constant with no environment override, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` does not
suppress it. A forward that needs more than 600 s cannot be measured at PP2 at all.

**Pin mechanics.** Under `EMMY_KNOBS` the cut pass visits only the root: every `PLACE@seam=cut` that resolves on the
root joins one composed decision, nested cuts are spelled from the root, a key naming no root seam is silently
skipped, and a bare `PLACE=fuse` fuses everything unaddressed. A `PLACE` pin REPLACES the whole placement decision
rather than adding to it — pinning a subset of a program's cuts silently discards the rest. A GPU-less `--target
sm_70` compile on a Mac featurizes with the default card's 170 SMs and elects differently from the live V100's 80, so
every election replay runs on the host CPU.

**Rules that bind every round.** The prior must never decide a production election; the golden must carry a measured
row for every kernel; `--strict-evidence` is the gate; no pricing floor, bound, clamp or hand-edited price, ever; no
benchmark scripts (`emmy run --bench --json`, `emmy compile`, `emmy tune` only — a missing capability is a flag to
add). Every harness fix ships as a minimal PR per AGENTS.md with a red-then-green test, and the goldens gate is
compared against pristine `origin/main` per-file — several files including this model's are red on `main`, so
identical counts mean pre-existing. Never re-record a red row to make it green. On a CUDA-less Mac never run `make
test-durations`; hand-insert a `tests/durations.json` entry at the CI-measured value if the durations gate fires.
The launcher leaves prefix caching on, so a repeated prompt's time to first token is a cache hit; cold numbers come
from a prompt the server has not seen.

## Risks

- Kernel quality is open-ended, and the twins sit on the fusion and tile-lowering path, so any rewrite there can
  re-block serving without touching this model's code. Treat a green gate as revision-scoped evidence; #804 is the
  worked example, staling 87 rows of a clean file in one merge.
- A single unrecorded shape is enough to make the model unservable, as `pre1` at M=1 was. Recording is not a
  finishing step here; it is the mechanism.
- A recorded row is not an elected row: the repository golden carries #799's M=1 cut and elects a kernel 2.5× slower.
- Per-hit-expert dispatch at top-6 across 43 layers is a known latency wall (~0.23 ms per launch of framing); the
  fixed-slot tier covers only single-token decode and is excluded on an expert shard.
- Replicated stream-mixing, norm and shared-expert compute across 8 tensor-parallel ranks wastes most of that
  compute. Small next to the experts, but it caps the ceiling.
- Everything pins to one 1Cat image; a fork bump reopens the weight-mapper and attention-API assumptions.
