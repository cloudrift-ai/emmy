# DeepSeek V4 Flash 0731 through `emmy serve` on 16× V100 (TP8 × PP2)

Goal: serve `deepseek-ai/DeepSeek-V4-Flash-0731` through the Emmy vLLM plugin on the 16× V100 SXM3 host — Emmy
compiled kernels for the hyper-connection stream mixing, norms, shared expert and routed experts; the pinned 1Cat
sm_70 fork (`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`) supplying paged MLA attention and the
serving shell — then A/B against the plain 1Cat container at an equal serving envelope.

43 layers, `hc_mult` 4, 256 routed experts at top-6 plus one shared, 3 hash-router layers. At TP8 × PP2 the first
stage owns layers 0–21 and the second 22–42.

## Where it stands

**It serves.** Measured 2026-09-12 on the host, `--strict-evidence`, single stream, greedy: 0.899 s per output token
(31 steady steps inside 1.3%), 7.05 s to first token on a short prompt and 88.2 s at 2,405 input tokens. Against the
pinned fork on the same host the fork is 6.1× faster per output token and 23× faster to first token — directional,
because the two arms have not yet run at one envelope (Stage 5). The evidence, the protocol and the caveats are in
`experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16/RESULTS.md`; that report is the durable record,
not this file.

Stages −1, 1 (#651), 2 (#656) and 3 (#662) are done, and Stage 0's question — whether the compiler can produce a
schedule fast enough to serve this model — is answered yes. Gate (c) passes: the server boots, answers, and its
completions are coherent. Gate (d)'s greedy token-ID half passed against the fork on three of four prompts, the
fourth diverging at a near-tie; its layer-level tensor half was never run and an HF eager reference for a 156 GB
checkpoint stays impractical here.

## The correction this plan owes its reader

Earlier revisions of this file concluded that what held serving was a compiler gap in three named programs, and that
"measuring them harder will not help". **That was wrong, and it pointed at the wrong work.** Every one of those
programs was fixed by recording a placement cut, with no new compiler capability:

| program | before | after | how |
| --- | ---: | ---: | --- |
| `pre` chunk m4096 | 1,923,598 µs | 3,238 µs | four `PLACE@…=cut` |
| `post1` `9e578e` @ m1 | 42,278 µs | 9,866 µs | placement cuts |
| `post1` `4e26cc` @ m1 | 30,016 µs | 5,347 µs | placement cuts (measured, **not recorded** — see below) |
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

## What is left

### Kernel quality — two targets, both the signature above

- `post1.k_linear_softmax_mean_matmul_reduce_4e26cc` @ m1 — **one** measured row, 29.9 ms, and it dominates
  `post.decode.m1` (293× its floor, 17 ms per layer, most of the 0.899 s per token). Its 5.6× cut was benched and
  correctness-checked bit-exact in an earlier round and never reached the repo golden. Cheapest remaining win.
- `post4096.k_matmul_reduce_06fabe` — 454.8 ms, the largest single cost anywhere in this model, inside
  `post.chunk.m4096` (317×, 619 ms per layer). This is what puts time to first token 23× off the fork.

`post.decode.m16` sits at 1,149× (68 ms per layer). It serves widths 2–16, so it is off the single-stream
path but on any concurrent one.

### Stage 4 — image and release plumbing (not started)

No baked image exists, so every boot pays its compile. Build FROM the immutable 1Cat digest with `cupy-cuda12x` under
its own image identity — do not inherit the Makefile's default version/tag for a 1Cat 1.2.3 base — labelling the 1Cat
digest and source SHA, Emmy SHA, checkpoint revision and CUDA/NVRTC versions. The serve env plumbing carries the
fork's `VLLM_SM70_*` variables along with `--tensor-parallel-size 8 --pipeline-parallel-size 2` and
`--distributed-executor-backend mp`. A headroom sweep on the host seals
`docker/vllm-emmy-serve/models/deepseek-v4-flash-0731.env` (the sweep creates it; do not author widths off-host).

→ verify: `make serve-config / serve-goldens / serve-warm / serve-image / serve-verify` on the host; the baked image
cold-starts offline and EVERY one of the 16 workers reports its pack hit (today's verify accepts one `pack hit` line,
which is insufficient), the cubin set is unchanged, and no request-time Triton JIT occurs. Build and verify only;
registry publication is a separate approval.

Measured envelope to plan the sweep against (gate (c), `--max-model-len 4096 --kv-cache-dtype fp8 --block-size 256`,
`--gpu-memory-utilization 0.90`): 30.8 GiB resident on a first-stage card and 31.75 on a second-stage one of 32, KV
cache 78,730 tokens on the first stage and 81,190 on the second. KV capacity is not derivable from a bytes/token
constant here — sliding layers cache a 128-token window and the compressed layers cache compressed entries.

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

## Operations handoff

The host's address is deliberately absent from this repo; it lives in the operator's notes and is used only inside
commands.

**Layout.** `~/emmy-serve` is the serving checkout the boot container installs with `pip install -e .` — a plain
directory, not a git checkout, so sync it with `rsync -a --delete --exclude __pycache__ emmy/ HOST:~/emmy-serve/emmy/`
before recording anything. Its emmy must be new enough to read the golden in play: an older one rejects a current
model golden with `unknown field(s): eager_us`. Serving evidence and golden copies live in `~/serve-evidence/`. The
compiler tree for tuning work is `~/emmy-durations/` with its own `./venv`, and `py-spy` there is the tool that names
a stalled program (`sudo -n ~/emmy-durations/venv/bin/py-spy dump --locals --pid PID`, one dump per rank,
simultaneously).

**Never touch** `~/.cache/emmy/autotune.db` (the real tune DB), `~/emmy`, `~/emmy-dsv4`, `~/emmy-fix-backup`,
`~/emmy-durations/_verify/gap3-tune/` (partial rows that regress the election — never merge that DB), or
`~/.cache/emmy/verify3/` (another user's live tuning session). Another user tunes on this host: run `nvidia-smi
--query-compute-apps=gpu_uuid,pid,process_name --format=csv` before every launch, use only devices nobody holds,
never kill a foreign process, and delete nothing when done. `pkill -f` over ssh matches its own remote command line
and kills the session — ship a script and run it by path.

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

## Risks

- Kernel quality is open-ended, and the twins sit on the fusion and tile-lowering path, so any rewrite there can
  re-block serving without touching this model's code. Treat a green gate as revision-scoped evidence.
- A single unrecorded shape is enough to make the model unservable, as `pre1` at M=1 was. Recording is not a
  finishing step here; it is the mechanism.
- Per-hit-expert dispatch at top-6 across 43 layers is a known latency wall (~0.23 ms per launch of framing); the
  fixed-slot tier covers only single-token decode and is excluded on an expert shard.
- Replicated stream-mixing, norm and shared-expert compute across 8 tensor-parallel ranks wastes most of that
  compute. Small next to the experts, but it caps the ceiling.
- Everything pins to one 1Cat image; a fork bump reopens the weight-mapper and attention-API assumptions.
