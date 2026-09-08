# DeepSeek V4 Flash 0731 through `emmy serve` on 16× V100 (TP8 × PP2)

Goal: boot `deepseek-ai/DeepSeek-V4-Flash-0731` through the Emmy vLLM plugin on the 16× V100 SXM3 host — Emmy compiled
kernels for the hyper-connection stream mixing, norms, shared expert and routed experts; the pinned 1Cat sm_70 fork
(`cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`) supplying paged MLA attention and the serving shell —
then A/B against the plain 1Cat container at an equal serving envelope.

**Expectation setting, up front.** At this seam the Emmy arm replaces the fork's already-fused mHC kernels and its
TurboMind MXFP4 MoE. With fp8 experts (2× the bytes, no fp8 tensor cores on Volta) and per-hit-expert dispatch, the
first working arm should be expected to LOSE the A/B. The deliverable of stages 1–5 is a *working, measured* lane and
the integration machinery; a *competitive* lane additionally needs stage 6 (MXFP4 expert inputs), and even that is a
hypothesis until profiled. The A/B must rerun the 1Cat baseline at the same shorter context — the published
30.79 tok/s was measured at a 1M-token envelope the Emmy arm cannot hold.

## Current state (already landed on this branch)

- The attention-sublayer seam: `pre(streams[T, hc·H]) → x[T, H]`, `post(attn_out[T, H], streams) → (mixed, xn,
  mix[T, hc])`, routed experts on the MoE third seam, closed by `place_routed_streams`. CPU-proven against the eager
  HF layer (sliding/HCA/CSA × hash/top-k). One twin profile per model (attention and routing are outside the twins).
- Serving-twin capture + `eval golden` audit accept DeepSeek V4; on the target host the `pre` and `expert` twins
  realize 20/20 (m1 / m32 / m4096 / dynamic) on sm_70.
- Checkpoint facts: DeepSeek-native names (`layers.N.attn.wq_a` …), trunk fp8-e4m3 with `F8_E8M0` [128, 128] block
  `.scale` siblings, routed experts MXFP4 (`I8 [out, in/2]` nibbles + e8m0 per-32 scales), `hc_mult` 4, 43 layers,
  256 experts × top-6 + 1 shared, 3 hash-router layers. The snapshot ships a lossless MXFP4 → fp8+e8m0[128,128] cast
  (`inference/convert.py`).
- Fork facts: `DeepseekV4Attention.forward(positions, normed_x[T, H]) → [T, H]` is one self-contained sublayer whose
  projections, compressors, indexer, FP8 paged insert and grouped output projection are fused with its own paged
  caches (SWA + compressor + indexer cache layers registered by prefix); sm_70 requires `--dtype half` and the
  `VLLM_SM70_*` env. There is no API accepting externally computed compressed latents.

## Memory budget per GPU (32 GB; TP8 × PP2 ⇒ 21/22 layers per stage; 256 experts sharded 8-way per TP group)

Worst (22-layer) stage, 32 experts/rank/layer, `w1+w2+w3` = 25.17 MB values per expert:

| item | fp8 experts | MXFP4 experts (+e8m0 per-32 scales) |
| --- | ---: | ---: |
| routed experts (sharded) | ~17.7 GB | ~9.4 GB |
| shared experts + hc/norm params (replicated) | ~0.6 GB | ~0.6 GB |
| fork attention weights (TP-sharded + replicated parts) | ~1.0 GB | ~1.0 GB |
| PP0 embedding (full vocab, fp16, runner-resident today) | ~1.1 GB | ~1.1 GB |
| Emmy activation arenas (capacity 4096 × hc·H carrier) | ~1.5 GB | ~1.5 GB |
| vLLM + CUDA context + fork workspaces | ~2–3 GB | ~2–3 GB |

KV capacity is NOT derivable from a bytes/token constant: sliding layers cache a 128-token window and HCA/CSA layers
cache compressed entries, which is how the recorded MXFP4 baseline allocates 4.2M tokens of KV per stage. Measure at
the first full boot (weight bytes, arenas, non-Torch allocations, GPU blocks, KV tokens per stage). Qualify initially
at 4K–32K `--max-model-len`; attempt the recipe's 1M only after that measurement proves capacity on both stages.

## Stage −1 — freeze and probe the pinned fork contract (cheap; before everything)

Resolve the base image to an immutable digest and record its source SHA. Inside that exact image: contract-test the
attention class (constructor args, forward signature, cache-layer registration and prefix rules, `topk_indices_buffer`
and aux-stream ownership, parameter `weight_loader`s, `WeightsMapper`, quantization config); install the Emmy wheel +
`cupy-cuda12x`; probe `nvrtcVersion` and compile a trivial sm_70 kernel through the same cupy path Emmy uses (add the
CUDA-12 `libnvrtc` preload only if that probe fails, using the path present in the image); confirm the unchanged plain
1Cat model still boots. Stop the plan here if a required fork API is absent.

### Stage −1 findings (2026-08-25, executed on the target host)

- Image pinned: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6
  b03d6df188e5c65bc1`, labels confirm build commit `d76126608…` / model revision `7872f01b…` / target GPU V100 SXM3.
- Source integrity: all 69 files of `vllm/models/deepseek_v4` + the MLA backends + the attention layer package are
  byte-identical to the `d76126608` checkout (per-file sha256) — the studied API is the shipped API.
- In-image stack: python 3.12.13, vLLM `1.2.3.dev87+gd76126608`, torch `2.10.0+cu129` — NVRTC in-image is **12.9**,
  which still targets sm_70 (deprecation warning only), so the host-venv NVRTC-13 preload problem does NOT apply
  inside the image.
- Contract confirmed: `DeepseekV4Attention.__init__(vllm_config, prefix, topk_indices_buffer, aux_stream_list)`,
  `forward(positions, hidden_states, llama_4_scaling)`; `DeepseekV4MLAModules` fields as studied;
  `_is_exact_sm70_cuda()` True on the card; `DeepseekV4SWACache(head_dim, window_size, dtype, prefix, cache_config)`;
  MLA attention + indexer cache are `AttentionLayerBase` (prefix-registered KV specs); the fp4/fp8 `WeightsMapper`
  builders exist; `VLLM_SM70_*` env keys and `VLLM_MULTI_STREAM_GEMM_TOKEN_THRESHOLD` (1024) present;
  `ModelRegistry.register_model` (the OOT plugin hook) available.
- The emmy wheel + `cupy-cuda12x` install into the image cleanly beside vLLM's pins, and `emmy.serving.register` is
  importable there.

## Stage 0 — unblock the twins (compiler; hard prerequisite) — **all three twins COMPILE; `post4096` RUNTIME blocks**

The first round closed (2026-08-25, below). Loop fusion was then rewritten under it, and the same class of pathology
came back on both twins ("Round two"). Round two is now closed too — `pre16` runs in 8.18 s and `post4096` compiles,
builds and launches — but round three found the remaining blocker one layer down: the plan the greedy selects for
`post4096` contains a serially-impractical kernel, so a boot still stalls in the first prefill forward.

### Round one — the `post` twin (2026-08-25) — closed

The stall was never in the graph: it was a compiler pathology that main fixed while this plan was being written.

**Diagnosis (2026-08-25).** A parameterized repro (capture + lower the `post` twin, `sm_70` forced) isolated it by
elimination: hidden size 32→1024 flat at 2.3 s, `hc_sinkhorn_iters` 2→20 linear (+0.13 s/iter), `hc_mult` 2/3/4 flat,
live CUDA device present or absent identical. The graph is byte-for-byte the same everywhere (644 IR nodes). What
differed was the compiler revision: at `5646a0796` (pre-merge) the lowering does not terminate (>240 s locally, 3 days
on the host); at merged main it takes 2.3 s locally and **7.3 s on the V100**. Reverting `71dfa184d` (#595, fresh-name
scans) and `4443f5d98` (#597, memoized splicer walks) leaves it fast, so the fix is `1ee503099` — **Bound fusion
construction work (#602)**, which rewrites exactly `merge_region` / `build_merged_region`, the two frames directly
above the stalled `_ensure_dep` / `fresh` calls in every py-spy sample. It cannot be reverted in isolation because
later commits build on it.

**Realization (2026-08-25, on the target host at merged main).**

| step | result |
| --- | --- |
| `emmy trace --serving-twins` (real checkpoint, pinned serving env) | ~60 s → **12 graphs, 380 distinct kernels** |
| `emmy run --golden` over the inventory | **1520/1520** (380 × m1/m32/m4096/dynamic), exit 0, 550 s |
| errors / tracebacks | none |
| non-finite fragment outputs | 10 / 1520 (0.66 %), all `post` at the widest width |

The non-finite outputs are a property of random-input fragment replay for this checkpoint, not a regression: the
already-qualified, published golden (`recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.yaml`) replays 279/279 exit 0
with **4 / 279 (1.4 %)** non-finite under the same harness — a higher rate than the new inventory's. `run --strict`
cannot adjudicate a Loop-IR fragment either way (no Torch twin and no independent greedy reference; it reports
"same-input greedy reference is unavailable"), so real correctness evidence comes from the CPU seam equivalence test
today and from real-weight parity at Stage 3.

Two environment notes for later stages — the first CORRECTED 2026-09-01: the old NVRTC-13 bench-worker constraint
no longer applies (workers inherit the full environment, and the nvcc path dropped its NVRTC fallback). The real
sm_70 trap is torch cu130 shipping no sm_70 kernels at all; `torch==2.13.0+cu126` fixes it, and `--bench` / `tune`
then run fine in the plain host venv — no 1Cat image needed. The inventory is untuned (no knobs or timings) and is
therefore NOT promoted to the canonical path; it regenerates in minutes (post-#691 rewrite: 12 graphs, 152 distinct
kernels, 3 m 44 s on the host).

### Round two — fusion rewritten under the twins (2026-08-28/29) — CLOSED

**Make loop fusion maximal and multi-output (#648, `bff3e3444`) landed after gate (c) passed** and broke both remaining
twins. All three defects are now fixed:

| twin | at `ab1ad4592` (pre-#648) | at `bff3e3444` (#648) | now |
| --- | --- | --- | --- |
| `expert16` | compiles | 11 same-scope redeclarations, nvcc rejects | fixed by #671 |
| `post16` | compiles | lowering never terminates | fixed by #676, ~57 s on the V100 |
| `pre16` | 3 kernels, 0.001 s | 1 kernel, never returns | fixed; **8.18 s verified at main+#692** |

`pre16` lowered to a single kernel recomputing the loop-invariant RMSNorm sum-of-squares under a crossed product
(≈ 4.4 × 10¹² iterations/thread) because `_close_projection` sank the sibling reduce into a nested contraction's
evaluation domain. The fix needed provider closure generalized from "direct body-member host" to lexical environments
for every `Fold` occurrence, so an operand-edge capture closes into a dependent seam (`CutSite.requires`) — landed via
#682 (provider-closed statistics seams) and #688 (one scoped-lambda `Closure` concept); the residual placement-cut
correctness bugs ride #692 (also PR #686's standalone form). Re-verified 2026-08-31 on `claude/attr-v100`
(main + #692): `pre16` builds 1 kernel and `run_once` returns in 8.182 s.

### Round three — `post4096`'s placement (2026-08-30/31) — compile CLOSED by #692, runtime OPEN

With `pre16` fixed, the TP8×PP2 boot got past compile and died in whole-program capture on a cut-workspace
`KeyError`; fixing that (piece inputs declared from the lossy lowered view — `loaded_buffers` is the honest reader)
exposed the real problem: `post4096`'s fused monster `k_linear_softmax_matmul_mean_reduce_3052e1` does **2⁵⁵ worst
per-thread serial trips** (`block_threads=1`) — the recomputation blowup of maximal fusion — and no evidence could
steer placement away from it. PR #692 fixes the whole chain, each step verified on the host:

- **Attribution**: one hang condemned every kernel in the terminal (70 failures / 7 distinct errors); the watchdog
  names the culprit, so only that kernel earns the `bench_fail` row (re-tune: 15/15 rows correct).
- **Disqualification**: a kernel whose every measured variant failed prices its structural arm at `inf`, matched
  exactly.
- **The composed cut compiles**: the consumer piece's IR was scope-inverted (normalize hoisted a fold whose subtree
  captures body-defined names; ILP replication renamed `deps()`-channel reads) — 17 nvcc errors → 0; the two-cut
  plan builds and launches at 2³⁰ trips.
- **The composed cut is on the ballot**: the unpinned fork offered plain seams only (2 of the monster's 33); now
  every seam is offered with its transitive `requires` closure as one composed arm, and the feared recursion
  explosion is measured convergent.

**Measured (2026-08-31, V100): an unpinned `post4096` compile with clean attributed evidence selects a composed cut
on its own — 52 kernels, worst 2³⁷ (was 2⁵⁵), placement terminating in 327 s.** Still not servable: 2³⁷ serial
trips is hours per launch (the 2³⁰ two-cut variant ran 2.6 h without completing before being killed). Two named
gaps stand between here and a boot that serves, both follow-ups to #692:

1. **Partition the monster — LANDED (#693/#694).** A chain-form root's DIRECT body members (the piece's
   workspace-rsqrt captures feeding the retained reduce, exactly the monster's shape) now offer and realize
   cooperative/ILP partitions: the reduce tier binds a provider chain ahead of a strided cooperative/ILP fold
   sharing one lane axis, closing lane-distributed. A fold nested deeper, and any member of a sweep- or
   streamed-store-carrying kernel, still keep the serial fold — an offer-side decision, not a remaining capability
   gap. Realization is corpus-ratcheted (a cooperative reduce row on a composed-cut chain-form piece).
2. **Evidence electing the route — MECHANISM PROVEN on the host (2026-09-01), measurement still owed.** The first
   tune pass over the new ballot could not measure the monster (below), but its 9 attributed `bench_fail` rows
   alone flipped the greedy: the worst piece dropped 2³⁸ → 2³⁰ per-thread serial trips (256×) and three
   partitioned reduces were elected (placement 857 s vs 616 s baseline). The elected consumer piece then failed nvcc — an
   order-blind seam-capture accounting bug in the composed cut (a deeper occurrence of the same workspace `Load`
   masked the shallower read, so the piece read the name before any definition); fixed by resolving seam captures
   in program order with a realize-time no-read-before-definition guardrail (PR #700; all 12 pieces of the real
   plan now compile, plan name-identical). What still blocks a MEASURED election: the tune's hardcoded compile
   budget (12 s/74 s) is far under the >160 s these variants need, the monster exposes 45 site-local `REDUCE@`
   knobs, and `--dump-dir` crashes on these targets — so no successful measurement exists yet, only
   disqualifications, and no online prior was written. The monotone serial-work prior feature remains the
   cold-start answer. NOTE: the host's tune DB now carries those 9 rows, so any unpinned compile there elects the
   partitioned route — intended, now that #700 makes it build.
3. **Price the recomputation so the statistics piece gets elected — LANDED (#702), then the clamp REVERTED by
   review decision; the statistics piece elected under it measured 8.6 ms on the host.** Even the elected 2³⁰ route
   ran past the 60 s bench watchdog
   per launch: the dominant cost was re-evaluating the mHC statistics subtree 16,384× (4096 carrier positions × 4
   streams) inside the consumer piece's sum-of-squares reduce. Characterized GPU-free: the materializing seam
   (`PLACE@a8`, the gate's fn-projection) was OFFERED and priced away — the offline cold-start proxy gave the
   fused 2³⁰-trip nest 4.29e-37 µs against the cut arm's 1.02e-17, with zero weights on any structural feature.
   #702 answered with pricing: the nest-aware `S_ext_serial_cell_work` stamp, the `D_serial_cell_work` fit signal,
   and a guarded clamp at the kernel-set Σ that priced a kernel at least its serial-work lower bound past 1 ms.
   The clamp was reverted on review: a lower bound on the prior's estimate is not how the deploy path elects —
   the prior must not decide a production election at all, and where it does, the missing golden or measured row
   is the defect. The stamp and fit signal stay (the prior can learn serial work), as do the two fixes that rode
   the same PR: disqualification signatures survive featurizer vocabulary growth (the stamp alone had silenced
   the host DB's 9 `bench_fail` rows) and `SearchDB` schema v4 dropping stale `lowering` chains keyed pre-stamp.
   Under the clamp, replayed on the pinned twins + host-DB copy, the greedy elected the same 12-piece plan plus
   exactly the `PLACE@a8` statistics piece — 13 kernels, the consumer drops 2³⁰ → 2¹⁶ and the route's worst piece is 2¹⁹
   per-thread trips (the unaided fused monster was 2³⁸).
4. **Make the elected pieces fast — OPEN; the path is now recorded goldens, not pricing.** Measured on the host
   2026-09-02: the elected route benched completely for the first time (every earlier attempt died on the 60 s
   watchdog; the clean run needed `--warmup 3 --iters 10` plus `EMMY_BENCH_RUN_TIMEOUT_S` — a third budget knob
   beside the two the tune report named) — 13 kernels, and the whole `post4096` forward **23.24 s**, not
   servable, so gate (c) was not attempted. One piece (`__place_8a9a1fe058`, 13.2 s, 57 %) re-read both FFN
   weight matrices per output element through serial 4096-trip hidden-dim reductions, and two 16384-grid sweep
   pieces added 9.3 s. A measured tune pass (2026-09-03, three passes, 58 ok rows) found no faster row for those
   pieces — every arm hung the watchdog — so the round turned to materializing the matmul contributions.
   What stands from that round (all GPU-free, pinned twins + host-DB copy): **(a)** the (A) verdict — the
   contribution seams (`PLACE@a29`, the linear_2/linear_3 contraction; `PLACE` on the activation cone) and the
   mHC statistics seams are ON the ballot and realize; seam capability is not the gap, and the uncalibrated
   proxy should not be the decider. **(b)** the materialized shape is right: a route whose contribution
   consumer is a pure staged mma with no serial hidden-dim walk, the walk living once in a producer piece, and
   the statistics computed once per row. **(c)** two blockers characterized on post-#699 main, each its own
   item: the residual root's output sweeps were all promoted into its placement (the emitted kernel decoded a
   2^56-thread linear grid — un-launchable) — FIXED by #723 (only the kernel's shared output sweep promotes; a
   sibling output nest's axis stays a sweep, and the residual launches at its 4096-row grid, 16 blocks × 256
   threads, again), and the contribution producer still recomputes the carrier chain
   per (row, a28) cell (the next seam, `PLACE@…inner` on the `a32` contraction, is on its ballot). Pricing
   floors beyond #702's are OUT by review decision: a golden must carry every kernel of the route, and strict
   evidence then keeps the prior from deciding at all. The path to serving: make `emmy tune` able to measure
   this family — the tune dead-end sink (#705), the retry-policy and regime-pin fixes in flight, the bench
   budget knobs, and a flag to seed the deploy election's route as a measured proposal — then tune the serving
   twins on the host, record the golden, and boot under strict evidence.
   **Measured (2026-09-07, host round three): a 12-kernel route at 2.10 s per `post4096` forward** (11× off
   the 23.24 s), benched completely under a pin set of 11 seams, recorded by `run --record-greedy` into the
   working golden (one composed routing row, two split rows, 12 receipts, every one from measuring), and the
   strict compile from the file alone emits the measured program byte-identical — with #739, without which a
   recorded composed cut replayed as its first seam only. Three findings that redirect the rest of this item:
   **(d)** the route was found by pinned characterization on the host CPU (27 s per probe), not by the loop or
   the tune: the offline prior prices unmeasured arms at 1e-27 … 1e+67 µs, so every measured piece loses to any
   arm holding an unmeasured kernel, the tune's 623 measured rows collapsed the election to six kernels, and the
   online checkpoint fitted on this family is quarantined — the sequential loop and the tune cannot be resumed on
   this target as they stand. **(e)** the one recompute behind every hang since round one is the mixed stream's
   RMS statistic, whose provider seam sits inside the gate statistic's subtree; once it is a piece the consumer
   reads it back instead of walking 16384 trips per cell. **(f)** 93 % of the 2.10 s is two kernels the route
   left naive — the A-operand reduction (1.37 s, a matmul computed as 8 million 128-lane reductions) and the
   one-thread-per-row root residual (0.59 s on 16 blocks) — both on the root's own ballot; cutting them is the
   next measurement, before any route for the other twins. The route has NO correctness verdict yet: `--strict`
   failed on non-finite values in the random-input replay (the round-one property above) and this twin has no
   eager reference, so a finite-input check is owed with it.
   **Measured (2026-09-08, host round four): 0.245 s per `post4096` forward** — 19 kernels, 8.6× off round
   three and 95× off the 23.24 s of 2026-09-02 — and it is the UNPINNED strict-evidence election of what the
   working golden deploys, stable byte-for-byte after every other twin's rows were recorded beside it. The path
   there: six more seams (the twist and exp contractions, the RMS statistic, both A cones, the mixed-stream
   store) at 2.68 s, a `REDUCE=` pin turning the A-operand contraction per-cell at 0.465 s, and one shared twist
   A-cone seam at 0.319 s, with the file's own unpinned election beating each pinned route. **(g)** 75 % of the
   remaining time is one contraction, the gate/up projection at 0.18 s, and it was NOT a tile site.
   Diagnosed on the host 2026-09-08 and FIXED by PR #752: the carrier is two independent matmuls the loop
   fusion put in one nest — each channel multiplies its own A cone by its own weight slab — and no tier folds
   such a carrier whole, since a tile holds one A fragment against a B slab per channel. `TileOp.contracts`
   refuses, no TILE row is enumerated, and no pin reaches an mma. Round four's stated cause is wrong on both
   halves: the two A cones are not duplicates of one value — they share all three of their own operand edges
   and read DIFFERENT components of one shared producer, acc42 and acc44 — and a commuted product could not
   have hidden a duplicate anyway, because a commutative `Assign`'s arguments are re-sorted on the canonical
   names whenever a `Lambda` is rebuilt, which makes exactly the two spellings that report quoted alpha-equal.
   Normalization now hands back one fold per state under a projection re-exposing the state tuple; on the host
   the kernel's a28 and a29 sites go from 0 tile sites to 22, and the contracting site carries a TILE knob in
   its family. MEASURED the same day, same target and width: the refusing contraction goes from 179.7 ms in one
   per-cell kernel to 15.4 ms in two kernels of 7.72 ms, both on `mma_m8n8k4`, and **the whole `post4096` forward
   from 238.8 ms to 74.9 ms (3.2x) as the file's own UNPINNED strict-evidence election** — the number a deploy
   picks. The pinned steps on the way, against round four's own 455.4 ms under that pin set: 292.8 ms with the
   pins repaired and a cut per child, 152.0 ms with round four's shared-cone seam beside them, then 74.9 ms
   unpinned, where every kernel takes its fastest measured row. Nothing else in the set moved more than noise.
   Two things the round taught about re-measuring a route across this change: the seam
   paths inside a carrier that comes apart shift one level (9 of the 11 Rstar pins resolved unchanged), and the
   route's single cut now names ONE of the two children, so a cut per child is needed or the other stays fused
   into its parent and recomputes its input per cell — the first two benches hung at the 180 s first-iteration
   deadline on exactly that. Artifacts: `_verify/item3/route{11,12,13}.json`, `E{1,2,3}SPLIT.pins`,
   `working.split-e{2,3}.yaml`, `autotune.split-e3.db`, `per-state.patch`; `working.yaml` and the tree itself were
   restored to their round-four state after each run.
   The nvcc refusal at `post` m1 is the same class of name collision, located this round: the per-cell
   ILP replication suffixes two different accumulators to one name, so `float acc0__c0_0 = 0.0f;` is emitted
   twice in one scope (8 collisions in the recorded reproduction). **(h)** the correctness verdict is not
   obtainable with today's CLI: the twin draws its eps and count constants as random inputs, so every replay is
   non-finite, and the same-input reference is the pinned route itself; a finite-input replay per twin and an
   independent reference (the loop-IR CPU runner, unexposed) on `run --golden` are the missing flags. **(i)** all
   151 serving-width realizations of the twins file ran unpinned and 123 carry measured routes (expert 4/4, pre
   2/4 with `pre4096` on a three-seam pin, post 118/144); the 28 missing are the second large `post` kernel
   family at m1 / m32 / dynamic (hangs unpinned — the `post4096` pin treatment), five small `post` reduces at m1
   (two refused by nvcc on a same-scope accumulator redeclaration, reproduced GPU-free; three retry-arm timeouts)
   and 18 rows lost to concurrent `--record-greedy` writers of one file. The `pre` twin's unpinned election is
   the round-two recomputation class at every width, and its contractions also refuse an mma. Routing rows are
   now priced at the launches their decision produced (#741), which is what let a measured split rank against
   the unsplit receipt.

**Consequence for the stages below.** Gate (c) passed at `ab1ad4592` and still does not reproduce: a boot now
compiles end to end and the recorded route is a measured 0.245 s per `post4096` prefill forward (no longer a
watchdog unknown) — the boot's roofline audit runs each program 4×, so serving needs the gate/up contraction
tiled first, and a strict boot needs a measured route for every serving twin: 28 of 151 realizations still lack
one, so `emmy serve --strict-evidence` would raise at their first fork. Stage 4 cannot warm or bake until that
lands and gate (c) is re-run on the host, and the golden re-record should follow it, not precede it.

### Operations handoff — how the Stage 0 host loop is run (for a fresh session or another machine)

The V100 host's address is deliberately absent from this repo; it lives in the operator's notes and is used only
inside commands. Everything else a continuing session needs is here.

**Host layout.** The compiler tree is `~/emmy-durations/` — a plain export of `origin/main` (no `.git`) with its own
`./venv`; sync it with `git archive origin/main | ssh HOST tar -x -C STAGING` then `rsync -a --delete --exclude venv
--exclude _verify --exclude _tune --exclude 'durations*' --exclude emmy_ml.egg-info --exclude '*.db*' --exclude
__pycache__ --exclude .git STAGING/ ~/emmy-durations/`. All loop artifacts sit in `~/emmy-durations/_verify/item3/`:
`working.yaml` (the working golden — the measured routes; `working.routeN.yaml` / `working.final4.yaml` are its
snapshots), `autotune.db` (the tune-DB COPY every command points at; `autotune.iterNstate.db`, `autotune.routeNstate.db`,
`autotune.roundNstart.db`, `autotune.tune{1,2}state.db` are byte snapshots taken through SQLite's backup API),
`online.json` (the online-prior checkpoint), `twins.yaml` (the captured serving twins, unchanged), the per-iteration
`iterN.{cu,log}`, the pinned probes `*.pins` + `*.cu`, and the GPU-free reproductions `nvcc-refused-k_div_11_reduce.cu`
(the same-scope accumulator redeclaration) and `tile_probe.py` (the two-channel contraction that is not a tile site).
The round reports (`item3-host-report.md`, `item3-resume-report.md`, `item3-round3-report.md`,
`item3-round4-report.md`) are untracked, under `.superpowers/` in the agent worktrees on the operator's Mac; this plan
carries their conclusions.

**Never touch** `~/.cache/emmy/autotune.db` (the real tune DB; backup `~/autotune.db.bak-2026-09-02`), `~/emmy`,
`~/emmy-dsv4`, `~/emmy-fix-backup`, `~/emmy-durations/_verify/gap3-tune/` (partial rows that regress the election —
never merge that DB), or `~/.cache/emmy/verify3/` (another user's live tuning session). Another user tunes kernels on
this host: run `nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv` before every launch, use only
devices nobody holds, never kill a foreign process, and delete nothing when done.

**Invocation.** Every command runs from `~/emmy-durations` with `PATH=/usr/local/cuda-12.9/bin:$PATH
CUDA_HOME=/usr/local/cuda-12.9 LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12 HF_HOME=/hf_models
HF_HUB_OFFLINE=1 EMMY_TUNE_DB=$HOME/emmy-durations/_verify/item3/autotune.db
EMMY_ONLINE_FILE=$HOME/emmy-durations/_verify/item3/online.json` and the budgets `EMMY_KERNEL_TIMEOUT_MS=30000
EMMY_FIRST_ITER_TIMEOUT_MS=180000 EMMY_BENCH_COMPILE_TIMEOUT_S=900 EMMY_BENCH_RUN_TIMEOUT_S=1800
EMMY_BENCH_WALL_TIMEOUT_S=3600`; recording needs `--warmup 5 --iters 20`. The launchers in `_verify/item3/` carry all
of it: `pin.sh NAME "K=V,…"` (a CPU-only pinned compile of the `post4096` target into `NAME.cu`, ~27 s — the way a
route is shaped; `nest.py` / `ktable.py` read a dump's per-kernel grid and serial depth), `route.sh N PINSNAME` (bench +
`--record-greedy` of `PINSNAME.pins` under `EMMY_KNOBS`, one device, writes the route into `working.yaml` and the
per-kernel rows into the DB copy), `strict.sh NAME` (the gate: `compile --golden working.yaml --realization …
--strict-evidence --target sm_70 --ir cuda`, no pin, must emit the measured program byte-identical), `twins.sh LIST
DEVICE` (unpinned strict bench + `--record-greedy` per realization, one writer per device), `iter.sh N` (the unpinned
evidence-loop iteration — do not resume it, see finding (d)), `evidence_probe.py` / `ballot_probe*.py` (what the
evidence index and each kernel-set fork see). Run long commands with `nohup … &` and poll; snapshot the DB before any
kill.

**Pin mechanics.** Under `EMMY_KNOBS` the cut pass visits only the root: every `PLACE@seam=cut` that resolves on the
root joins one composed decision, nested cuts are spelled from the root (`map.4/map.2/inner…`), a key naming no root
seam is silently skipped, and a bare `PLACE=fuse` fuses everything unaddressed. `Rstar.pins` (round three, 11 seams),
`B6.pins`, `E1.pins` (+`REDUCE=`), `F1.pins` are the measured sets; the file's own unpinned strict election beat every
one of them once their rows were recorded. Target realization:
`post4096.k_linear_softmax_matmul_mean_reduce_9716a1.f86b6dbe35b7.m4096`, `--target sm_70`. A GPU-less `--target
sm_70` compile on a Mac featurizes with the default card's 170 SMs and elects differently from the live V100 (80), so
every election replay runs on the host CPU.

**Rules that bind every round.** The prior must never decide a production election; the golden must carry a measured
row for every kernel; `--strict-evidence` is the gate; no pricing floor, bound, clamp or hand-edited price, ever; no
benchmark scripts (`emmy run --bench --json`, `emmy compile`, `emmy tune` only — a missing capability is a flag to
add); every harness fix ships as a minimal PR per AGENTS.md with a red-then-green test and the goldens gate compared
against pristine `origin/main` per-file counts (all 11 files are red on `main` since #691/#699 — identical counts mean
pre-existing). On a CUDA-less Mac never run `make test-durations`; hand-insert a `tests/durations.json` entry at the
CI-measured value if the durations gate fires. In an agent worktree, symlink the main checkout's `venv` and prefix
`PYTHONPATH=$PWD`.

## Stage 1 — loader lane: read the published checkpoint (CPU-testable) — **DONE (#651)**

Extend the quantized split loader (`load_quantized_split` + `loader/quant.py`) with one DeepSeek-native lane:

1. Key translation native → HF (`attn.wq_a` → `self_attn.q_a_proj` etc.) — reuse transformers' checkpoint
   conversion mapping for `deepseek_v4`, not a hand-rolled copy. `.scale` joins `_scale`/`_scale_inv` as a sibling
   form.
2. e8m0 scales: `F8_E8M0` reads as f32 `2^(e−127)` (torch's `.float()` on `float8_e8m0fnu` is the conversion).
3. Routed experts: per-expert `w1/w3` (gate/up) and `w2` (down) stack into the E-leading store the expert programs
   feed from. **What landed instead of the planned fp8 cast:** the loader keeps the published MXFP4 bytes and views
   them onto the uint8 blocks/scales carrier the expert programs already bind (`expert_dtype: fp4` selects it over
   the fp8 trunk declaration), so serving deploys `@mxfp4` expert programs — half the expert bytes per rank, and no
   cast to keep byte-exact. Two consequences, both found at the first real boot: the expert spelling had to learn
   this checkpoint's `F.linear` weight layout, and the twin lane still spells `@f8e4m3` (below).
4. `expert_range=(lo, hi)` filter so a rank reads only its expert shard (a PP stage's full fp8 expert set is
   ~138 GB of host RAM otherwise), alongside the existing `layer_range`.
5. Trunk fp8: dequantize to fp16 values at load (existing lane), including the grouped `wo_a`'s [128,128] blocks.

→ verify: a synthetic tiny native-format checkpoint (tests, mirroring `test_load_quantized_split_*`) loads to a twin
whose eager forward matches `load_dequantized_state_dict` values; the fp4→fp8 cast matches the reference converter in
raw fp8 payload bytes AND raw e8m0 scale bytes across every fp4 nibble code, block-boundary exponents, and random
tensors; `expert_range` applies before stacking/conversion and peak host memory proves no rank materializes non-local
experts; on the host, one real layer's loaded expert slice matches the reference converter's output.

## Stage 2 — runner: DeepSeek widths + TP expert sharding — **DONE (#656)**

1. Seam plumbing in `EmmyGenRunner.from_model`: DeepSeek `_meta` (no `q_proj`; carrier `hc·H`, attention width `H`),
   the 3-output post program (`mixed`, `xn`, `mix`) routed through `_route_post_device` (per-output dest shapes —
   the rider path currently assumes residual-width outputs), `place_routed_streams` as the combine closer,
   `input_ids` reaching the hash-router layers' gate, embed broadcast to `hc_mult` streams before layer 0, and
   `final_norm` = `hc_head` collapse + RMSNorm. The carrier contract is explicit: `runner.carrier_size = hc_mult·H`
   sizes the activation arenas AND the plugin's PP intermediate-tensor factory (which today allocates
   `config.hidden_size`); every PP boundary transports the flattened carrier; only the last rank applies `hc_head`.
2. TP expert sharding: `moe["inputs"]` holds the local expert slice + a global→local index map; the router runs
   replicated (same weights, deterministic); `combine_routed_experts` skips non-local experts; the plugin all-reduces
   the routed `[T, H]` partial (vLLM's tensor-parallel all-reduce) before `place_routed_streams`. `mixed`/`xn` stay
   replicated compute. Eager routed path only at first; the fixed-slot capture tier (per-rank tables) is a follow-up,
   so the first boots serve `--enforce-eager`.

→ verify: a 2-rank CPU/GPU unit test proving sharded-combine + all-reduce equals the single-rank oracle
(`combine_routed_experts` full set); a PP2 test asserting the intermediate-tensor factory, send/receive tensors and
residual arenas all use `hc_mult·H` while attention and expert inputs stay `H`; the existing gen_runner GPU stitch
tests stay green; a single-GPU tiny-config DeepSeek stitch test (seam programs + torch attention stand-in) matches
eager.

## Stage 3 — plugin: host the fork's attention inside `EmmyGenModel` — **in-repo work DONE (#662)**

1. A DeepSeek branch that constructs the fork's `DeepseekV4Attention` per layer (needs `vllm_config`, the shared
   `topk_indices_buffer`, aux streams, unique prefixes so its SWA/compressor/indexer cache layers register in the
   static forward context and get KV allocations), instead of vLLM `Attention` + RoPE.
2. Weight routing: an explicit ownership table over checkpoint keys — fork attention (via the pinned fork's
   `WeightsMapper` + each destination's own `weight_loader`: `fused_wqa_wkv` ← `wq_a`+`wkv`, `compressor.
   fused_wkv_wgate` ← `wkv`+`wgate`, `attn_sink` head-narrowing, indexer params), Emmy trunk/shared/routed programs,
   `lm_head` ← `head.weight`, embedding ← `embed.weight`. Loading fails loudly if a fork-owned attention parameter is
   missing, double-loaded, or an attention checkpoint key stays unclaimed. Fork attention keeps its pinned fp8 quant
   config; Emmy owns trunk/expert conversion. Speculative/MTP serving is rejected at boot.
3. Forward: carrier `[T, hc·H]` as the PP intermediate tensor; per layer `pre → fork attention (positions, x) →
   post → local routed combine → all-reduce → place`; `hc_head` + norm on the last rank.
4. Optional de-risk hybrid, decided UP FRONT (it must be algebraically exact): Emmy `post` already places the
   shared expert, so a hybrid may only call a fork operation of the form `native_routed(xn, input_ids) → routed[T,H]`
   — the fork's routed experts WITHOUT its shared expert and mHC placement. If the pinned fork only exposes the full
   `DeepseekV4MoE` (shared expert included), either add a verified Emmy `post` variant that omits the shared expert,
   or drop the hybrid. Silently keeping both double-counts the shared expert.

→ verify, in grades: (a) a one-process attention contract test — unique absolute layer prefixes, every fork cache
spec registered; (b) a TP2×PP2 small-config distributed test; (c) the TP8×PP2 target-host boot serving mixed
prefill/decode requests (not just `/health`); (d) layer-level numerical agreement vs the eager seam at predefined
atol/rtol, then deterministic greedy token-ID agreement on a fixed corpus spanning HCA/CSA/hash layers, multiple
expert destinations, PP transport, and mixed scheduling — token IDs either agree or they do not. Also verify
`emmy serve`'s MoE probe recognizes `n_routed_experts` (or pin capture sizes + eager in the release config).

### Stage 3 findings (2026-08-26, PR #662)

- Items 1–3 landed; gates (a), the weight-loading gate, and (b) are green in the pinned image: construction and
  per-absolute-layer cache registration across all three attention layer types, the attention ownership table
  against the real fork modules, and a REAL-engine parity gate — the same checkpoint served single-rank and
  TP2×PP2 produces identical greedy token ids modulo numerical ties (exact ids stay the real checkpoint's gate,
  where logits are decisive).
- Load-bearing surprises, each encoded in code/tests/docs: a vLLM process re-registers `deepseek_v4` onto its own
  rope-only config class process-wide, so the loader reloads with Transformers' same-named native class; the
  fork's kernels accept only the published geometry (compressor head 512 / indexer head 128 / 128-aligned fp8
  group outputs / 128×128 blocks, and no invented `compress_rates`); compiled twins may hand outputs back in
  their accumulation dtype, normalized at the seam by the runner; and the machine-wide GPU file lock deadlocks
  multi-rank serving (a rank holds it inside its combine while its pending collective waits on the peer queued
  behind the same lock) — it is now scoped per physical device UUID.
- The remaining gates move to the on-host block beside Stage 4's image work: (c) the TP8×PP2 target-host boot
  serving mixed prefill/decode, and (d) real-checkpoint layer-level numerics plus greedy token-ID agreement.

### Gate (c) — PASSED (2026-08-26, real checkpoint at TP8 × PP2, at `ab1ad4592`)

**Does not reproduce on current main** — see Stage 0 round three. The result below stands as evidence that the seam,
the loader and the plugin are correct; re-running it needs `post4096`'s selected plan to be executable at serving
speed (the partitioned composed-cut route).

`deepseek-ai/DeepSeek-V4-Flash-0731` serves through `EmmyGenModel` on the 16× V100 SXM3 host, in the pinned 1Cat
image, at TP8 × PP2 with `--max-model-len 4096 --kv-cache-dtype fp8 --block-size 256` and eager execution:

| | |
| --- | --- |
| Boot (engine init → serving) | ~19 min: 55 s load, ~5 min compile (warm cubin cache), ~12 min profile + KV alloc |
| KV cache | 78,730 tokens (PP0 stage) / 81,190 (PP1), from 12.99 GiB free after residents |
| Resident per card | 30.8 GiB (PP0) / 31.75 GiB (PP1) of 32, at `--gpu-memory-utilization 0.90` |
| Mixed prefill/decode | 8 concurrent requests, prompts 5–361 tokens, outputs 8 and 128, 544 output tokens in 101.9 s |
| Output | coherent and correct: "The capital of France is" → " Paris. The capital of Spain is Madrid. …" |

**Gate (d) — greedy token-ID half PASSED.** Against the plain 1Cat arm at the same shape and revision, on a fixed
four-prompt corpus at temperature 0 × 32 tokens: three prompts agree on every token id (including a 361-token prompt
that reaches the compressed/indexed attention layers, and a code prompt); the fourth diverges at token 6 on a near-tie
(the fork's own top two are 0.125 nats apart, and Emmy's logprob for its pick is within 0.089 of the fork's). Each arm
is individually deterministic across runs, so the divergence is arm-to-arm numerics at a tie, not noise. The
layer-level tensor half of gate (d) was NOT run: an HF eager reference for a 156 GB checkpoint is impractical here, and
the CPU seam-equivalence test plus this end-to-end agreement on real weights are what stand in for it.

Generation quality is the load-bearing part of this result: a transposed expert matrix or a mis-scaled MXFP4 decode
produces fluent-looking garbage, not correct capitals and a valid Python guard clause. Single-stream decode measured
~3.6 tok/s (16 tokens in 4.43 s) against the plain 1Cat arm's published 30.79 tok/s — the loss the plan predicted,
not yet an equal-envelope A/B (that is Stage 5). No pack exists yet, so every boot pays the compile (Stage 4).

Three defects stood between the merged branch and a booting server. Both were invisible to every earlier gate
because each lives on a path only the real checkpoint's geometry and the engine's own scheduling reach, and each
killed all 16 workers:

- **Expert spelling assumed one weight layout.** `spell_mxfp4_inputs` was written for gpt-oss, whose experts trace
  as the `(in, out)` matrix applied with `x @ W`, so it closed the decode with a transpose. DeepSeek's experts are
  `F.linear` parameters — `(out, in)`, already the stored orientation — and `w_down` failed its shape check before
  a single weight was read. Each spec now declares its module's layout (`moe_expert_layout`), which is the only
  sound source: a square expert matrix (gpt-oss `down_proj`, DeepSeek `gate_up_proj`) reads correctly both ways,
  so a shape-sniffed guess would silently transpose the weights rather than fail.
- **The rider destination was sized from a q/k/v seam.** A chunk step carrying decode riders splits across two
  programs into one joint destination, sized from `(num_heads · head_dim, …)`. The fork-attention `pre` returns
  one hidden-width activation instead — 4096 against the 32768 that sizing computed. vLLM's profiling run executes
  at exactly the rider top (`max_num_batched_tokens` = prefill capacity + decode bucket = 4112), so this blocked
  every boot of this seam, not an edge case. The post path already read its widths off the program's output count;
  the pre path now does too.
- **The expert program was built one step too narrow.** `pre`/`post` split a rider-width step across their static
  twins, but the routed dispatch hands one expert program every row that chose it — and the profiling run's dummy
  rows are identical, so one expert takes all 4112 of them against a 4096-row buffer. The expert program now takes
  the rider allowance too (16 rows of arena on a 4096-row buffer). Pinned by a GPU test whose OLMoE router scores
  every expert alike, which reproduces the degenerate routing without depending on a profiling run.

**Found here — predicate half FIXED (#666), golden half still owed.** The twin lane and the serving lane disagreed
about this checkpoint's experts: `mxfp4_weight_profile` keyed on `quant_method == "mxfp4"`, and DeepSeek declares
`quant_method: fp8` with `expert_dtype: fp4`, so `capture_twin_graphs` recorded the expert twin as `@f8e4m3` while
serving deployed `@mxfp4`. #666 added the one shared predicate both lanes now read (`native_mxfp4_experts`), so the
recorded expert program is the one serving binds.

The golden re-record is NOT done: `golden/v100_sm70.yaml` is still the #558 recording from before any of this, so it
covers the routed-expert kernels not at all and they resolve from reservoir/prior evidence instead. Correctness is
unaffected, but Stage 4 must not warm, bake or seal until the re-record happens on the host — which is itself blocked
behind Stage 0 round three's runtime gap.

## Stage 4 — image + release plumbing

1. Build the plugin image FROM the immutable 1Cat digest with `cupy-cuda12x`, with its own image identity — do not
   inherit the Makefile's default `v0.23.0` version/tag for a 1Cat 1.2.3 base. Label the 1Cat digest + source SHA,
   Emmy SHA, checkpoint revision, and CUDA/NVRTC versions. The stage −1 NVRTC probe decides whether the serve
   entrypoint needs the CUDA-12 `libnvrtc` preload for sm_70.
2. Env passthrough: the serve image/env-file plumbing gains the fork's `VLLM_SM70_*` variables and
   `--tensor-parallel-size 8 --pipeline-parallel-size 2 --distributed-executor-backend mp` in `SERVE_EXTRA_ARGS`.
3. Headroom sweep on the host seals `docker/vllm-emmy-serve/models/deepseek-v4-flash-0731.env` (the sweep creates
   it; do not author widths off-host); record the serving golden; warm → bake → verify per the release skill.

→ verify: `make serve-config / serve-goldens / serve-warm / serve-image / serve-verify` pass on the host; the baked
TP8×PP2 image cold-starts offline and EVERY one of the 16 workers reports its pack hit (today's verify accepts one
`pack hit` line — insufficient for 16 workers), the cubin set is unchanged, and no request-time Triton JIT occurs.
Build/bake/verify only; registry publication is a separate approval.

## Stage 5 — A/B

`emmy bench` two arms at the SAME envelope (same `--max-model-len`, mnbt, concurrency, KV dtype, warmups,
immutable image digests and checkpoint revision; the existing serving_v100_sxm3 protocol, shorter context; one
priming repeat + three steady repeats): the Emmy image vs plain 1Cat. Profile in a separate run — profiling the
fork's multi-stream execution perturbs the A/B. Report output tok/s, TTFT,
TPOT with repeat spread; archive per the experiment conventions (no credentials or VM identifiers). Publish the
honest conclusion even if (as expected) the Emmy arm loses; include a per-phase profile (expert dispatch vs
attention vs mHC) so stage 6's hypothesis is grounded.

## Stage 6 (performance, optional) — MXFP4 expert inputs

Largely landed upstream since this plan was drafted: main now spells native MXFP4 expert twins
(`spell_mxfp4_inputs`, `decode_mxfp4`, `…@mxfp4` twin names — uint8 nibble blocks + uint8 e8m0 scales as program
inputs). What remains for THIS checkpoint: its declaration is `quant_method: fp8` + `expert_dtype: fp4` (not
`quant_method: mxfp4`), and its packing is `w1.weight I8 [out, in/2]` + `.scale [out, in/32]` (not the gpt-oss
`_blocks`/`_scales` layout the profile recognizes) — so a DeepSeek declaration/orientation mapping onto the existing
spelling, plus the loader keeping the fp4 store, plus tuning. NOTE: with the fp8 declaration, the expert twin already
spells `@f8e4m3` today, which is exactly the stage-1 cast lane's deployed form (test:
`test_deepseek_fp8_declaration_spells_the_expert_twin_for_the_cast_lane`). Only worth building if stage 5's profile
shows expert weight streaming dominates and the fused-unpack GEMM can plausibly beat TurboMind's on Volta.

## Risks

- Stage 0 is open-ended compiler work; nothing below it ships without it. It has now reopened once: the twins sit on
  the fusion/tile-lowering path, so any rewrite there (#648 was one) can re-block serving without touching this
  model's code. Treat a green gate (c) as revision-scoped evidence, not a permanent one.
- The fork's attention inside a foreign model class is the largest integration unknown (cache registration,
  metadata, capture breaks, `VLLM_MULTI_STREAM_GEMM` aux streams); mitigated by the stage-3 hybrid boot.
- Per-hit-expert dispatch at top-6 × 43 layers is a known latency wall (~0.23 ms/launch framing); the fixed-slot
  tier only covers T=1 and needs per-rank tables under sharding.
- Replicated mHC/norm/shared-expert compute across 8 TP ranks wastes ~7/8 of that compute; acceptable at first
  (it is small next to experts), but it caps the ceiling.
- vLLM/fork version drift: everything pins to the one 1Cat image; a fork bump reopens the weight-mapper and
  attention-API assumptions.

## Effort

Stage −1: DONE (~2 h). Stage 0 round one: DONE (fixed upstream by #602). Stage 1: DONE (#651). Stage 2: DONE (#656).
Stage 3 in-repo: DONE (#662); gate (c) passed once at `ab1ad4592`, gate (d)'s token-ID half with it.

**Stage 0 round three remains the critical path — now as one compiler gap, not election.** Partitioning
(#693/#694), the compiling composed cut (#700), the serial-work stamp (#702), the composed route rows (#739) and
the route-row pricing (#741) all landed, and the working golden's own strict election measures 0.245 s per
`post4096` forward, 75 % of it in one contraction the compiler cannot tile (two duplicate A cones on a
two-channel fold). What holds everything now, in order: make that contraction a tile site (merge the duplicate
cones), fix the same-scope accumulator redeclaration nvcc refuses at `post` m1, give `run --golden` finite
inputs and an independent reference for a correctness verdict, stop concurrent recorders losing rows, then pin
routes for the 28 realizations still without one (the second `post` kernel family at m1 / m32 / dynamic, the
`pre` twin at m32 / dynamic). Two of those are closed: the concurrent-recorder loss (#751 — the recorder
reloads the file under an exclusive lock and adds its rows to what is on disk, so parallel devices stop
dropping each other's rows) and the tile site (#752, above), which replaces item one with re-measuring the
route under the split. Nothing on that list reproduces from the checked-in
`recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.yaml` — its whole-model programs cat the sibling gate/up
linears into one weight, and its `k_div_*_reduce` family emits no redeclaration under any knob set tried — so
every one of them needs the V100 host. Then Stage 4: 2–4 days on-host (re-run gate (c), re-record the golden,
warm/bake/verify). Stage 5: 1–2 days.
Adding stage 6 (MXFP4 + tuning) is a further 1–3 weeks. The compiler, not the fork ABI, remains the dominant
uncertainty.
