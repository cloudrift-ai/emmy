# DeepSeek V4 Flash 0731 through `emmy serve` on 16× V100 (TP8 × PP2)

Goal: serve `deepseek-ai/DeepSeek-V4-Flash-0731` through the Emmy vLLM plugin on the 16× V100 SXM3 host — Emmy
compiled kernels for the hyper-connection stream mixing, norms, shared expert and routed experts; the pinned 1Cat
sm_70 fork (`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`) supplying paged MLA attention and the
serving shell — then A/B against the plain 1Cat container at an equal serving envelope.

43 layers, `hc_mult` 4, 256 routed experts at top-6 plus one shared, 3 hash-router layers. At TP8 × PP2 the first
stage owns layers 0–21 and the second 22–42.

## Where it stands (2026-09-24)

`main` at `1494e4be` carries this model's golden restamped onto the compiler's fresh lowering: #875 restored the
fusion #863 had changed, #878 re-lowered every stored target and re-keyed the rows, and a gate
(`test_stored_targets_are_the_fresh_lowering`) now decodes every repository golden's stored targets against a fresh
lowering so the drift cannot return unseen. The file holds 454 measured rows over 149 targets; the deploy's evidence
index reads 437 of them, and 318 of the 335 kernels the nine serving twins mint have a measured row. Of the seventeen
without, eleven are the single-token expert and pre twins' pieces, whose refusal that tier survives by riding the
wider one, and six are the pieces #883 re-shaped in the three post twins, whose refusal ends the boot. The durable
record of every boot, election and A/B is
`experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16/RESULTS.md`; this file only says what is left.

It serves strict from the repository golden, but slower and worse than the tree before #863 did. Boot36b (main plus
#875 at #869, 2026-09-23, fast math pinned off): 0.73 s to first token warm at 5 prompt tokens and 5.5 s at 2,155;
0.56 s per output token, against 0.215 s on the last boot of the old tree and about 0.15 s for the pinned fork. Its
completions are worse than that boot's, the width-16 expert twin's election reports its random-input reproducer 9,931
outliers over a budget of 4, and the width-4,096 post twin's reproducer returns NaN. Boot37 (`main` as merged,
2026-09-24) did not serve: seventeen minutes in, the symbolic post twin's compile refused at one of the pieces #883
re-shaped — the piece opens a cut fork of its own, a kernel-set decision strict cannot take without a row, and a post
twin has no wider tier to ride — so `main` as merged does not boot strict until the three post leads are recorded
again, and their cut receipts carry the old pieces' time until then.

Three things hold the numbers. Fast math became the default (#868) while serving published no precision pin, so every
boot needed the regime pinned off by hand — closed by `serve --golden` publishing the rows' regime to its workers.
Cooperative reduction on the fresh divide kernels does not compile (`float v125` declared once per half of the split
accumulator), so those kernels run serial, 36 to 250 times slower than the old cooperative rows, and the transposed
cooperative reduce is offered on no kernel of this model; that is most of the 0.215 → 0.56 s. And the prior's
cooperative picks cost 10 to 100 times wherever nothing is measured.

Stages −1 to 3 are done, and Stage 0's question — can the compiler serve this model — is answered yes. Gate (c),
coherent completions, passed on the old tree and is open again on `main`. Gate (d)'s greedy token-ID half: two of four
prompts agree with the fork on all 32 token ids, two diverge at near-ties of about 0.2 nats; its layer-level half
never ran, and an HF eager reference for a 156 GB checkpoint stays impractical here.

## What is left, in order

1. **Record the three post leads on `main`** (`post-sym`, `post16`, `post4096`): the two pieces #883 re-shaped in each
   and their cut receipts. Started 2026-09-24 as rec37 on the host from the tree at `1494e4be`; until the rows land,
   no boot from `main` serves strict.
2. **Correctness on `main`.** Boot `main` with those rows, probe it, and find the wrong kernel by switching twin
   families off one at a time. Suspects: #863's reduction normalization, the cooperative-reduce codegen defect, the
   register-split rows pinned on the post twins during the restamp. Nothing else is worth measuring until the output
   is right. `emmy run --golden` cannot carry the verdict: the twins draw their eps and count constants as random
   inputs, so every replay is non-finite and the only reference is serving output — a finite-input replay per twin and
   an independent reference on `run --golden` are still owed.
3. **The cooperative-reduce codegen defect**, then the transposed cooperative reduce. GPU-free repro:
   `EMMY_KNOBS="REDUCE@map.1/map.1/reduce=coop,WORK=t128" emmy compile --golden
   recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.yaml --realization
   post16.k_div_50_reduce.d81043a46649.m16.4e18a355bd66 --target sm_70 --ir cuda`. This is what brings decode back
   near 1.5× the fork.
4. **Stage 4 — image and release plumbing.** Bake FROM the immutable 1Cat digest with `cupy-cuda12x` under its own
   image identity — not the Makefile's default version/tag for a 1Cat 1.2.3 base — labelled with the 1Cat digest and
   source SHA, Emmy SHA, checkpoint revision and CUDA/NVRTC versions; carry the fork's `VLLM_SM70_*` variables with
   `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend mp`. The pinned config
   `docker/vllm-emmy-serve/models/deepseek-v4-flash-0731.env` (#768) is the single source for the twin widths; a
   headroom sweep on the host seals its memory values. Verify with `make serve-config / serve-goldens / serve-warm /
   serve-image / serve-verify` on the host: the baked image cold-starts offline, every one of the 16 workers reports
   its pack hit (today's verify accepts one line, which is insufficient), the cubin set is unchanged, no request-time
   Triton JIT. Build and verify only; registry publication is a separate approval. Envelope to plan the sweep against
   (gate (c), `--max-model-len 4096 --kv-cache-dtype fp8 --block-size 256 --gpu-memory-utilization 0.90`): 30.8 GiB
   resident on a first-stage card and 31.75 on a second-stage one of 32, KV cache 76,337 tokens on the first stage and
   78,722 on the second; KV capacity is not a bytes-per-token constant here, since sliding layers cache a 128-token
   window and the compressed layers cache compressed entries.
5. **Stage 5 — the A/B, the deliverable.** One `emmy bench` run over both arms at one envelope with their order
   alternated inside each repeat, the way the RTX 5090 gemma-4 experiment balances time and thermal drift, against
   immutable image digests and one checkpoint revision. Profile in a separate run — profiling the fork's multi-stream
   execution perturbs the A/B — with a per-phase split (expert dispatch, attention, stream mixing) so Stage 6's
   hypothesis is grounded.
6. **Stage 6 — MXFP4 expert inputs**, only if Stage 5's profile shows expert weight streaming dominates and a
   fused-unpack GEMM can plausibly beat TurboMind's on Volta. `main` spells native MXFP4 expert twins; this checkpoint
   needs its declaration mapped onto that spelling (`quant_method: fp8` with `expert_dtype: fp4`, packed as `w1.weight
   I8 [out, in/2]` with `.scale [out, in/32]`), plus tuning.

Owed beside the list: the boot's roofline audit has no time limit (one mispicked program hung a boot for six hours);
the expert M=1 twin offers no schedule knob under its cut and its residual runs 1.04 s per launch, so it needs tile or
reduce sites from the compiler, not a row; the dynamic-width Sinkhorn twin cannot take its cut, a reshape lowering
lockout #813 names.

## What every round has taught

- **One bad measured row is binding under strict evidence.** A shape with one recorded candidate elects it however
  slow: `pre1` at M=1 ran 29.7 s per layer until a cut was recorded, and no compiler capability was missing. Read the
  golden's rows per program and shape, take the minimum per kernel, and treat a shape carrying one candidate as the
  smell; `emmy compile --golden PATH --realization NAME --ir loop` prints the Loop IR, where the recurring defect on
  this model is a dot product inside a sweep it does not depend on.
- **A green strict decode is not a deploy.** The decode is pooled — a row passes when any kernel of its set enumerates
  it — while the deploy keys every row under the kernel it decides. Before any boot, run the deploy's own evidence
  index under the recorded regime and, per twin, the kernels the route mints against the rows that vouch for them: a
  cut arm is eligible only when every piece has a row, a row spelling a split or a cut whose pieces have no rows can
  still win its fork on a retime, an empty schedule row is no evidence, and a schedule row carrying the identity a cut
  fork is offered on prices the fused arm with a piece's timing.
- **A compiler change can re-shape the kernels a golden names.** #863 changed this model's fusion (36 → 45 kernels per
  post twin) and normalized the stored Loop IR, #864 dropped rows in a restamp, #883 split a fused grid pair in two
  pieces per post twin. `emmy trace --serving-twins` on the tree, diffed against the golden's kernel families, sees
  the first before a record; the fresh-lowering gate sees the second on any machine. A row on a kernel whose body
  changed is dropped, not re-keyed, and if the new kernel opens a cut fork the twin refuses until it is recorded
  (boot37); a row whose kernel only changed identity is re-keyed and re-anchored — its OFF sites completed against the
  new kernel, or a strict election realizes a 28 ms cooperative reduce under a serial row.
- **Never re-record a red row to make it green**, and the prior must never decide a production election.
  `--strict-evidence` is the gate; no pricing floor, bound, clamp or hand-edited price; no benchmark scripts (`emmy
  run --bench --json`, `emmy compile`, `emmy tune` only — a missing capability is a flag to add). Every harness fix
  ships as a minimal PR with a red-then-green test. The launcher leaves prefix caching on, so a repeated prompt's time
  to first token is a cache hit; cold numbers come from a prompt the server has not seen.
- **Reproducing a boot failure without a boot.** A twin's Graph IR is the golden's `programs[i]` entry
  (`graph_from_wire`, then `specialize_program` for a static width); serving compiles it with `CudaBackend.compile`
  and then `plan_from_graph`, and only that pair shows a plan-time failure — `emmy compile --ir cuda` and a `--golden
  --realization` replay stop before it. Host CPU compiles need `~/emmy-durations/venv` with `PYTHONPATH` set to the
  tree under test and `LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12`, and a card visible: golden evidence is
  keyed by the live card's name, so `CUDA_VISIBLE_DEVICES=""` empties it.
- **Pin mechanics.** Under `EMMY_KNOBS` the cut pass visits only the root: every `PLACE@seam=cut` that resolves on the
  root joins one composed decision, a key naming no root seam is silently skipped, and a `PLACE` pin REPLACES the
  whole placement decision rather than adding to it. A GPU-less `--target sm_70` compile on a Mac featurizes with the
  default card's 170 SMs and elects differently from the live V100's 80, so every election replay runs on the host
  CPU.
- **Two deadlines bound any forward pass** under two-stage pipelining, because the second stage blocks on the first
  for the whole traversal: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300) is raisable; the NCCL collective
  watchdog at 600 s is compiled in, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` does not suppress it. A forward that needs
  more than 600 s cannot be measured at PP2 at all.

## Operations handoff

The host's address is deliberately absent from this repo; it lives in the operator's notes and is used only inside
commands.

**Layout.** `~/emmy-serve` is the serving checkout the boot container installs with `pip install -e .` — a plain
directory, not a git checkout, so sync it with `rsync -a --delete --exclude __pycache__ emmy/ HOST:~/emmy-serve/emmy/`
before recording anything. A boot can equally mount a revision-named copy (`~/emmy-main-<sha>`; the 2026-09-24 boot's
`~/serve-evidence/boot37.sh` mounts `~/emmy-main-1494e4b`), which leaves the shared checkout alone and says what ran.
Serving evidence, golden copies, boot, record and probe scripts live in `~/serve-evidence/`. The compiler tree for
tuning work is `~/emmy-durations/` with its own `./venv`, and `py-spy` there is the tool that names a stalled program
(`sudo -n ~/emmy-durations/venv/bin/py-spy dump --locals --pid PID`, one dump per rank, simultaneously).

**Never touch** `~/.cache/emmy/autotune.db` (the real tune DB), `~/emmy`, `~/emmy-dsv4`, `~/emmy-fix-backup`,
`~/emmy-durations/_verify/gap3-tune/` (partial rows that regress the election — never merge that DB), or
`~/.cache/emmy/verify3/` (another user's live tuning session). Another user tunes on this host: run `nvidia-smi
--query-compute-apps=gpu_uuid,pid,process_name --format=csv` before every launch, use only devices nobody holds, never
kill a foreign process, and delete nothing when done. `pkill -f` over ssh matches its own remote command line and
kills the session — ship a script and run it by path. On a CUDA-less Mac never run `make test-durations`; hand-insert
a `tests/durations.json` entry at the measured value if the durations gate fires.

## Risks

- Kernel quality is open-ended, and the twins sit on the fusion, cut and tile-lowering path, so any rewrite there can
  re-block serving without touching this model's code. Treat a green gate as revision-scoped evidence; #804 staled 87
  rows of a clean file in one merge, #863 changed the fusion, #883 re-shaped two pieces per post twin.
- A single unrecorded shape is enough to make the model unservable, as `pre1` at M=1 was. Recording is not a finishing
  step here; it is the mechanism.
- A recorded row is not an elected row: a bad price beside it, or a piece without a row, can make a cut lose.
- Per-hit-expert dispatch at top-6 across 43 layers is a known latency wall (~0.23 ms per launch of framing); the
  fixed-slot tier covers only single-token decode and is excluded on an expert shard.
- Replicated stream-mixing, norm and shared-expert compute across 8 tensor-parallel ranks wastes most of that compute.
  Small next to the experts, but it caps the ceiling.
- Everything pins to one 1Cat image; a fork bump reopens the weight-mapper and attention-API assumptions.
