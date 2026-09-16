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

**Main has moved under that measurement.** #804 (the Loop IR identity re-key) merged 2026-09-15 and left 87 of this
golden's 372 rows stale — decoded row by row on main (6eb37188) on 2026-09-16: the file had no stale row before
#804, #804 alone produces the 87, and #807 adds none. Among them are all five post-family placement cuts, #799's
M=1 cut included, each failing on a seam path that no longer resolves; the fastest expert row at every width (the
best live row is 12× slower at M=1 and 65× at the dynamic width); the pre family's best M=1 row; and
`post1.k_div_43_reduce` at M=1, which has no live measured row left, so a strict boot from main should refuse
there (unconfirmed — it needs a host boot). Seven of the 87 are old-spelling duplicates of a row that still
decodes; the other 80 are real losses. The 2.03 s above describes 7e9336e6 + #807, not main.

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
m1 / m16 / m4096 / dynamic; the boot audit now reads `pre.chunk.m4096` at 2.7 ms per layer and the pre decode
programs stay under its threshold. That is what halved time to first token at 2.3k input tokens.

**One election is wrong before anything is recorded.** The boot audit reads `post.decode.m1` at 44.7 ms per layer
(750× its floor) from the repository golden; the host-local golden the 2026-09-12 numbers came from elected about
18 ms (294×) for the same program. #799 recorded that cut into the repository, and the election does not take it.
Forty-three layers of the difference is the 1.1 s per token between 0.899 s and 2.03 s. On main the question has
moved: #799's cut is one of the 87 stale rows, so it is not on the ballot at all. Restamp first, then ask why a
recorded cut loses, ahead of recording anything new.

Then the rows. By the diagnostic above, every single-candidate row over 1 ms in the golden (372 realizations over
152 configs) is a post kernel:

| row | best measured |
| --- | ---: |
| `post4096.k_matmul_reduce_06fabe` @ m4096 | 453.5 ms |
| `post4096.k_linear_softmax_mean_matmul_reduce_4682df` @ m4096 | 106.3 ms |
| `post-sym.k_linear_softmax_mean_matmul_reduce_dba017` @ m4096 / dynamic / m16 / m1 | 106.3 / 97.5 / 46.9 / 30.5 ms |
| `post-sym.k_matmul_reduce_d2b070` @ dynamic | 73.8 ms |
| `post16.k_linear_softmax_mean_matmul_reduce_20ed9d` @ m16 | 46.9 ms |
| `post16.k_matmul_reduce_9fab3b` @ m16 | 1.85 ms |

Per-program sums of the best rows: `post4096` 560 ms, `post-sym` 356 ms, `post16` 49 ms, everything else under
9 ms.

- `post4096.k_matmul_reduce_06fabe` — the largest single cost anywhere in this model, inside `post.chunk.m4096`
  (688 ms per layer, 352×). This is what holds time to first token. Its nest is the same defect in an extreme form:
  every load comes from one 4×4 matrix per index, and the program recomputes that matrix's row sums at about eight
  nesting levels, so it runs `4096 × 4^k` iterations over sixteen values.
- `4682df`, `dba017` and `20ed9d` are `4e26cc` at other widths: structurally identical nests differing only in the
  outermost token loop, and the same cut family binds on them. #799's six cuts are the template. `post.decode.m16`
  (68 ms per layer, 1,145×) is off the single-stream decode path but on any concurrent one, and on prefill.

Recording waits on the restamp of the 87 stale rows (see "Owed regardless"): #804 has landed, but it accepted this
file's breakage instead of restamping it, and a row recorded against the stale spellings would be written twice.
The discovery half — which seams, what they measure — does not depend on identity and can run on a host-local
working golden meanwhile.

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
completes in minutes since #804 but reports the 87 stale rows, so Stage 4 waits on the restamp.

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
file again, and what it reports today is the 87 stale rows.

**The restamp of those 87 rows is owed, and it is the next step.** #804 restamped 600 identity fields across the
repository but accepted this file's rows as breakage. 371 of its 372 rows carry a stored `identity:` that deploy
joins on, so a restamp must replay each cut under its pins and map the seam paths and child identities to their new
spellings — re-lifting targets is not enough, and re-recording is not the answer either: the measurements are good,
only the spellings moved. Two rows do not even report a verdict; they raise a `KeyError` inside the replay, where a
kernel's offered keys are recorded through the declared-keys path without a matching offered-pairs entry. That is a
one-line fix with a red-then-green test, and it ships first so the gate reads all 372 rows. A fresh strict boot from
main comes after the restamp and before any new recording; the boot20 numbers do not describe main.

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
