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
record of every boot, election and A/B was
`experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16/RESULTS.md` until #896 removed this lane from
`main` on 2026-09-24; it survives in `main`'s history at bc58445ef, and what of it the work still rests on is
condensed under "The record" below.

It serves strict from the repository golden, but slower and worse than the tree before #863 did. Boot36b (main plus
#875 at #869, 2026-09-23, fast math pinned off): 0.73 s to first token warm at 5 prompt tokens and 5.5 s at 2,155;
0.56 s per output token, against 0.215 s on the last boot of the old tree and about 0.15 s for the pinned fork. Its
completions were worse than that boot's and the width-16 expert twin's random-input check reported 9,931 outliers over
a budget of 4; both are #829's staging defect, fixed by #893 (item 2), after which boot39 serves coherent text at the
same timings. The width-4,096 post twin's reproducer still returns NaN. Boot37 (`main` as merged, 2026-09-24) did not
serve: seventeen minutes in, the symbolic post twin's compile refused at one of the pieces #883 re-shaped — the piece
opens a cut fork of its own, a kernel-set decision strict cannot take without a row, and a post twin has no wider tier
to ride — so `main` as merged did not boot strict until the three post leads were recorded again (boot38, item 1
below); their cut receipts still carry the old pieces' time.

Three things hold the numbers. Fast math became the default (#868) while serving published no precision pin, so every
boot needed the regime pinned off by hand — closed by `serve --golden` publishing the rows' regime to its workers. The
fresh divide kernels run serial, 36 to 250 times slower than the old cooperative rows and up to 40 times slower than
eager: #863 lowers them to chains of nested four-element folds recomputed per element, so the one long reduce axis the
transposed cooperative reduce needs is gone (the build defect that hid this, a name declared twice under any
cooperative reduce, is fixed by #895); that is most of the 0.215 → 0.56 s. And the prior's cooperative picks cost 10
to 100 times wherever nothing is measured.

Stages −1 to 3 are done, and Stage 0's question — can the compiler serve this model — is answered yes. Gate (c),
coherent completions, passed on the old tree, was red on `main` from #829 to #893 without anyone seeing it, and is
green again with #893. Gate (d)'s greedy token-ID half: two of four prompts agree with the fork on all 32 token ids,
two diverge at near-ties of about 0.2 nats; its layer-level half never ran, and an HF eager reference for a 156 GB
checkpoint stays impractical here.

## What is left, in order

1. **Boot `main` with the six pieces recorded.** Done 2026-09-24: the pieces #883 re-shaped are recorded serial, at
   199 and 178 µs on the symbolic twin, 6 and 6 µs at width 16 and 1.4 and 1.6 ms at width 4,096 (the old
   register-split row cost 39.7 ms and was never in the deployed route), all three post leads elect strict from the
   file, and boot38 serves in thirteen minutes at the same numbers as boot36b: 0.71 s to first token warm at 5 prompt
   tokens, 5.5 s at 2,155, 0.53–0.58 s per output token, the width-4,096 post twin at 104 ms per layer, the
   completions degraded.
2. **Correctness on `main`.** Found and fixed 2026-09-24 (#893): since #829 the expert cut piece's compute-filled B
   slab staged the producer edge's last result for every channel, so both tensor-core B tiles held the up half and the
   twins computed the up projection twice; the width-16 expert twin's random-input check failed by 70% of the output's
   peak and single-token decode served noise. Bisected by pinning the twin's recorded tile on one host tree per
   revision; fixed in the fold's channel walk and the fill; both expert twins pass on the fixed tree under their
   recorded rows. Boot39 from that tree serves coherent completions at boot38's timings, so gate (c) is green again.
   Still owed: a finite-input replay per twin and an independent reference on `run --golden`, and a boot that reads
   the election's check instead of printing it as a warning.
3. **The transposed cooperative reduce on the fresh divide kernels.** The build defect under any cooperative reduce is
   fixed (#895, 2026-09-24: the one-value-per-name sweep now reaches nested scopes), and the six divide leads build
   and run under it on the V100 — at the serial rows' speed, because the only site the fresh kernels offer is the
   four-element inner reduce. The rows that made these kernels fast before #863 (7.6 µs at width 16, 49 µs symbolic,
   32 µs at width 4,096, against 270 µs, 1.5 ms and 7.8 ms serial today, and eager PyTorch's 150 to 200 µs) were
   `coop-t` over one long reduce axis, and the kernels #863 lowers to have none: their Tile IR is a chain of nested
   four-element folds recomputed per element of the sweep around them (a `coop-t` pin is quietly taken as `coop`). The
   item is the reduction normalization that fragments the axis, not a schedule; undoing it for these kernels is what
   brings decode back near 1.5× the fork.
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
- **Three more things that each cost a day.** A `perf` row times one CUDA op, so under a schedule that splits a kernel
  it is a fragment, and only the boot audit's ratio says whether the program moved. `emmy tune` defaults to a 2 s
  cumulative bench budget, which a kernel near 1 s per launch exhausts on warm-up, recording zero valid latencies
  while appearing to search; `EMMY_BENCH_RUN_TIMEOUT_S` raises it. A replay must use the golden the boot uses: benched
  against a file with no rows for it, a kernel the boot runs in 42 ms hung for a minute.
- **Two deadlines bound any forward pass** under two-stage pipelining, because the second stage blocks on the first
  for the whole traversal: `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS` (default 300) is raisable; the NCCL collective
  watchdog at 600 s is compiled in, and `TORCH_NCCL_ASYNC_ERROR_HANDLING=0` does not suppress it. A forward that needs
  more than 600 s cannot be measured at PP2 at all.

## The record

What the deleted results report established, kept here because the work continues from it. Every Emmy number is a
strict boot from the repository golden on the 16× V100 host unless a host-local file is named; the probe is greedy,
single stream, streamed and timestamped per chunk, 5 prompt tokens → 33 out and the long passage (2,275 tokens through
2026-09-17, 2,155 on later boots) → 9 out, with prefix caching on, so the repeat column is a cache hit. Evidence for
each boot is on the host under `~/serve-evidence/bootNN-*`.

### The fork's baseline

`emmy bench` on 2026-09-10 (run `20260910T174153Z`, fifteen rows, all succeeded): TP8 × PP2, `gpu_memory_utilization`
0.80, `max_model_len` 4096, `max_num_batched_tokens` 4096, block size 256, float16 with `deepseek_v4_fp8` weights and
an FP8 KV cache, prefix caching off; 57,594 KV tokens; greedy with `ignore_eos`. Image
`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (digest `sha256:276240257b22…c65bc1`, vLLM
1.2.3.dev87), model revision `7872f01b`, driver 580.159.03, nvcc 12.9.86.

| Concurrency | Input → output | Output tok/s, mean ± SD over 5 repeats | Mean TPOT | Mean TTFT |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 128 → 128 | 33.71 ± 3.24 (29.03 – 38.31) | 221.6 ms | 2.50 s |
| 4 | 1024 → 256 | 19.99 ± 0.39 | 179.7 ms | 5.41 s |
| 1 | 2048 → 512 | 6.46 ± 0.00 | 147.7 ms | 3.77 s |

The single-stream row is the one to compare against: its time per output token agrees to 0.2 ms across repeats, while
the eight-way row spans 30% on noise alone. The earlier 30.79 tok/s figure was measured at a 1,048,576-token context
the Emmy arm cannot hold and is no baseline.

### The Emmy boots

| Date | Tree, golden | TPOT | TTFT 5 tok (cold / repeat) | TTFT long (cold / repeat) | Health | What changed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 09-11 | first strict boot | — | — | — | 949 s init | serves; without `--strict-evidence` a blown-out prior (exponent 996 against a peak near 28) scored every candidate alike |
| 09-12 | host-local golden | 0.899 s | 7.05 s / — | 88.2 s at 2,405 / — | — | four cuts on `pre1` at M=1: 29.7 s → 474 µs, TPOT 5.565 → 0.899 s |
| 09-15 | `7e9336e6` + #807, repo golden | 2.03 s | 6.14 / 3.37 s | 44.2 / 16.0 s | 12 min | the repository golden serves for the first time; `post.decode.m1` elects 44.7 ms where the host-local file elected 18 |
| 09-17 | `3b5cc4ca`, #826 golden | 3.30 s | 6.33 / 3.60 s | 45.5 / 16.6 s | 15 min | after #804's re-key; strict refuses the M=1 tier, decode rides width 16 |
| 09-18 | `483e4cb7`, #833 rows | 0.267 s | 3.73 / 0.96 s | 29.3 / 2.70 s | 17 min | the M=1 tier deploys again; post m1 3.4 ms per layer; `pre.chunk.m4096` regressed to 25.4 ms |
| 09-19 | `cab3b735`, one split row dropped | 0.266 s | 3.74 / 0.97 s | 29.3 / 2.68 s | 29 min | #843's serial residuals; the first compile per rank took 800 s (cause never found) |
| 09-19 | seven post receipts serial | 0.267 s | 3.67 / 0.92 s | 24.9 / 2.48 s | 21 min | `post.chunk.m4096` 599 → 318 ms |
| 09-19 | three matmul pieces tiled | 0.267 s | 3.69 / 0.92 s | 18.3 / 2.21 s | 22 min | `post.chunk.m4096` 318 → 173 ms; the symbolic post twin 172.6 → 18.8 ms at 2,155 tokens |
| 09-19 | M=1 post pieces on threads | 0.235 s | — | 18.3 / 2.22 s | 14 min | 851 → 52 µs per layer |
| 09-20 | `9607133e`, expert m16 re-tiled | 0.214 s | — | 18.2 / 1.98 s | 25 min | 1.46× the fork; the last coherent boot before #829's defect reached the golden's rows |
| 09-22 | `dab7bce9`, rec34 | died | | | 3.5 min | #863 changed the fusion: serving's kernels are not the golden's |
| 09-23 | main + #875, restamped, fast math off | 0.56 s | 3.50 / 0.73 s | 68.7 / 5.53 s | 25 min | serves; text degraded; #868's default had killed three boots before |
| 09-24 | `1494e4be` as merged | died | | | 17 min | #883's re-shaped pieces open cut forks no row spells |
| 09-24 | + the six pieces' rows (#892) | 0.53 – 0.58 s | 3.50 / 0.71 s | 68.0 / 5.52 s | 13 min | serves; text degraded |
| 09-24 | + #893 | 0.53 – 0.58 s | 3.49 / 0.70 s | 68.1 / 5.54 s | 27 min | coherent again |

The first decode step of a request costs more than a steady one (1.85 s against 0.90 on 09-12, 4.2 s against 2.03 on
09-15): each layer's programs are CUDA-graph captured on first use. The first compile on each rank is the cold
evidence index of a new compiler fingerprint, about 800 s; a repeated tree does not pay it.

### Where the time went

The boot's roofline audit, first layer of each stage, per layer:

| Boot | `pre.chunk.m4096` | `post.decode.m1` | `post.decode.m16` | `post.chunk.m4096` |
| --- | ---: | ---: | ---: | ---: |
| 09-11 | 64,604× (1.92 s) | 1,273× | 1,154× | 343× |
| 09-12, host-local | 107× (3.2 ms) | 294× (18 ms) | 1,156× | 321× (619 ms) |
| 09-15 | 2.74 ms | 44.7 ms | 68.3 ms | 688 ms |
| 09-18 | 25.4 ms | 3.2 – 3.6 ms | 13.6 ms | 599 ms |
| 09-19, tiled | 3.10 ms | 3.2 – 3.9 ms | 12.5 ms | 172.8 ms |
| 09-23 / 09-24 | 3.12 ms | not deployed | 7.41 – 8.06 ms | 104 ms |

A decode step, profiled on 09-19 with torch's profiler over eleven single-stream steps on all sixteen workers (the
model serves eager, because the hyper-connection routed combine host-syncs): per token about 126 ms of Emmy kernels
(3,300 launches), 96 ms of NCCL all-reduce at a millisecond per call over PCIe (the host's, not the arm's), 22 ms of
the fork's sparse attention and 36 ms of everything else. The boot audit's `post.decode.m1` figure is an uncaptured
launch loop and overstates what serving pays. The symbolic post twin is linear in width, about 128 µs per token before
the tiling and 18.8 ms per layer at 2,155 tokens after it; a single request under a 4,096-token context never makes
the exactly-4,096-token step a chunk twin takes, so its prefill rides the symbolic twins and the m4096 twins matter
only once concurrent prompts fill a step.

### What made the kernels fast

- **Placement cuts that hoist a loop-invariant dot out of a sweep.** `pre4096` computed four 16,384-long dot products
  once per output channel, 8,192× redundant work: four `PLACE@…=cut` took it from 1,923,598 to 3,238 µs (594×, max abs
  1.2e-4 against the greedy pick). `pre1` at M=1 is the same kernel and took the same cuts, 29.7 s → 474 µs. `post1`'s
  `9e578e` ran sixteen long dots on one thread, 42,278 → 9,866 µs; its `4e26cc` evaluated sixteen logits 352 times,
  30,016 → 5,347 µs; both bit-exact. Recorded with `run --golden --bench --record-greedy`, the election takes them on
  price with nothing pinned.
- **Serial residuals.** A `WORK: t128, REDUCE: coop` receipt recorded while the binder ignored the cooperative reduce
  carried the serial kernel's time; when #813 honoured it, 128 threads shared a four-element reduce: `pre4096` 25.1 →
  2.82 ms per layer once re-recorded serial (224 µs), `pre16` 1.55 → 1.27 ms. Seven post receipts with two cooperative
  reduces and one block per output cell, the same way: `post-sym` 64.9 → 40.3 ms at the 512 hint, `post16` 9.88 → 8.85
  ms, `post4096` 537 → 258 ms.
- **Tensor-core tiles on the block's plain matmuls**, from kernel-scoped A/Bs (a scratch golden with one receipt
  respelled, sixteen cards running sixteen candidates): `WORK: w2x2, TILE: mma_m8n8k4_f16_f32/f4x4/k8, STAGE: d2/smem`
  on the 4,096 → 2,048 pieces (9,457 → 142 µs) and `w2x4 f4x2/k8 d2/smem` on the 2,048 → 4,096 piece (17,244 → 414 µs)
  and the m4096 matmul (140,438 → 2,745 µs); staging is what makes the tile pay (895 µs unstaged against 142). The
  symbolic post twin went 40.3 → 4.84 ms at its hint, the m4096 twin 258 → 121 ms.
- **The M=1 post lead's two serial pieces on threads:** its residual `WORK: t256` 668 → 15.5 µs, its sum of squares
  `t256 coop` 150 → 3.1 µs; the set 851 → 52 µs per layer.
- **The width-16 expert twin re-tiled at depth 1** from forty-eight candidates: gate/up `w2x1 f1x1/k8 d2/smem` 554 →
  265 µs, down `w2x4 f1x1/k8 d1/smem` 310 → 159 µs (the wide `f4x4/k8` tile that won the post matmuls is the worst row
  here: the expert rows are narrow); the twin 864 → 430 µs, TPOT 0.235 → 0.214 s. #864 then removed the depth-2 ring
  on Volta (wrong answers on nine of sixteen measured grids), and the re-tile at depth 1 measures 490 µs; the symbolic
  expert twin's root piece offers no tensor-core tile since then (20.4 ms against 1.8 ms) and is what a single
  request's prefill pays.
- **`k_div_35`**: the pre-#813 rows put two cooperative reduces on seams the codec no longer allows together; one
  `coop-t` seam measured 16.7 / 2.5 / 53.5 µs at dynamic / m16 / m4096 against main's picks of 140 / 4.7 / 492 µs.
- **The restamp round (09-22, rec34, on `dab7bce9`)**, per layer against the old file on the #860 tree: post m1 406 →
  449 µs, post m16 8,847 → 5,966, symbolic post 4,836 → 3,625 at the 512 hint, post m4096 120.6 → 84.6 ms, expert m16
  430 → 719 µs, expert symbolic 2.44 → 22.6 ms, expert m4096 22.5 → 14.2 ms. Cooperative reductions ran 2 – 30× slower
  on that tree than on `9607133e` for the same spelling (a `coop-t` t256 piece 61.8 → 133.5 µs; a single `coop` t128
  residual 303 µs → 11 ms), which is why serial wins everywhere since.

### Correctness

Gate (c) passed on 09-12 ("Red, blue, and green are three classic colors…"), and every re-record through 09-20 left
the long prompt's completion unchanged word for word, which was the only correctness evidence a post twin had. Gate
(d)'s greedy half, the four-prompt corpus of 2026-08-26 at temperature 0 against the fork's dumps: code 32/32, medium
32/32, short 5/32 (` Spain` −1.114 against ` Italy` −1.324, the same near-tie as in August), long 1/32 (` is` −1.075
against `.` −0.869, which agreed on all 32 in August); its layer-level half never ran. Gate (c) was red from #829
(09-20) to #893 without anyone seeing it, because the election's random-input check prints as a non-fatal warning and
no boot reads it; the width-4,096 post twin's check returns NaN since 09-17 and is still unexplained.

### Compiler defects met on the way

Fixed: the constant-fold pass left a two-lane range unfolded under a broadcast, so the expert compile died at plan
construction and no replay command reaches that step (#807); the cut splicer renamed a workspace read but not the
values derived from it (#827); a schedule row carrying the identity a cut fork is offered on priced the fused arm with
a piece's timing, and same-shaped twins were told apart by mint order (#826); an empty receipt row was dropped from
the evidence index (#834); a split row that outbid its sibling on a retime, whose pieces had no rows (#849); the merge
rule's copy-drop that changed this model's fusion (#875); the stored loops never re-lowered for #863's normalization
(#878); serving publishing no precision pin after #868 made fast math the default (#891); the compute-filled B slab
staging one producer edge's last result for every channel (#893); the one-value-per-name sweep stopping at nested
scopes (#895). Open: a replay from another compiler tree wipes the one-fingerprint identity store (draft #851); #863's
normalization lowers the divide kernels to chains of nested four-element folds (item 3); the M=1 expert twin offers no
schedule knob under its cut and its residual runs 1.04 s per launch; the m256 expert twin is absent from the golden
because the runner's expert prefill tier is a hardcoded constant the serving config cannot declare; the dynamic-width
Sinkhorn twin cannot take its cut (a reshape lowering lockout #813 names).

### Not established

The fork and Emmy numbers are directional, not a balanced A/B: separate invocations, different envelopes (the fork at
`gpu_memory_utilization` 0.80 with prefix caching off, Emmy at 0.90 with it on), different prompt shapes, the Emmy
rows one repeat each from direct HTTP requests with no experiment record. No baked Emmy image exists, so the
single-image two-entrypoint mechanism the A/B needs is not built. Correctness beyond greedy agreement and coherent
completions was never measured, and no tensor-level comparison has run.

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
