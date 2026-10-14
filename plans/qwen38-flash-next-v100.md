# Qwen3.8-Flash-Next on V100 nodes through Emmy

Goal: serve `Qwen/Qwen3.8-Flash-Next` on CloudRift's 16× V100 SXM3 nodes through Emmy's own kernels, the way
DeepSeek V4 is served: first running, then optimized until the Emmy serving image is at least on par with the plain
1Cat-vLLM fork at an equal serving envelope. The result ships as an Emmy recipe that `emmy deploy` and Relay run.

Emmy's kernels run inside the fork's runtime, which supplies the serving shell and whatever Emmy does not take. So
the fork has to serve the model first: steps 1 to 3 set up the runtime and record the baseline to match, step 4 gets
Emmy serving the model correctly, step 5 closes the speed gap, and step 6 releases.

Written 2026-10-13, revised the same day after a review and after the owner fixed the goal. Nothing below has run on
our hardware yet. Numbers marked "upstream" are 1Cat-vLLM's own reports.

## What we know

### The model

- 180B parameters: a 125B language model with 6B active per token, a 51B n-gram embedding read at layer 2, a 4B
  one-layer MTP head for speculative decoding, and a vision encoder.
- 48 layers: 12 × (3 × (Gated DeltaNet → MoE) + 1 × (Qwen Sparse Attention → MoE)). The MoE has 512 experts, 10
  routed plus 1 shared. Every layer has a gated residual with 4 branches.
- Context is 262,144 tokens. It thinks by default. The model card's sampling: temperature 1.0, top-p 0.95, top-k 20
  when thinking; temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 when not.
- The Hugging Face architecture is `qwen4_exp`. Stock vLLM supports it since v0.29.0. Our Transformers pin (5.14.1)
  has no class for it.
- License is `qwen-community-1.0`, not Apache. Nobody here has read it for paid serving or for redistributing weights
  in an image.
- The hosted "Qwen3.8-Flash" is a different, production build of this model. It is not a reference for the open
  weights.

### Checkpoints

| Checkpoint | Size | Note |
| --- | ---: | --- |
| `Qwen/Qwen3.8-Flash-Next` (BF16) | 360 GB | the reference |
| `Qwen/Qwen3.8-Flash-Next-FP8` | 186 GB | official |
| `RadixArk/Qwen3.8-Flash-Next-NVFP4` | 135 GB | the one the Volta fork measured |
| `nvidia/Qwen3.8-Flash-Next-NVFP4` | 133 GB | |
| `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF`, IQ3_S | 84 GB | the fork's deployment example |
| `unsloth/Qwen3.8-Flash-Next-GGUF`, Q4 / Q8 | 111 / 188 GB | |

### The Volta fork already serves it

`1CatAI/1Cat-vLLM` (Apache-2.0) is the project our fork comes from. Upstream reports, all on **four** V100 SXM2 32GB
cards at TP4 with direct NVLink, one request:

| Checkpoint | Speculative decoding | Decode | Build | Other |
| --- | --- | ---: | --- | --- |
| NVFP4 | off | 80.7 tok/s | release v1.5.1 | GSM8K 15 of 16 |
| NVFP4 | MTP, 4 draft tokens | 138 to 150 tok/s | release v1.5.1 | |
| GGUF IQ3_S | MTP, 4 draft tokens | 106 to 170 tok/s | development build | first token 0.6 s at 1K, 4.5 s at 8K, 17.3 s at 29K |

- The release wheel does not contain the GGUF work. Upstream says it needs a development build.
- The GGUF example serves 32K context. History can live in host memory, with a cache on the cards that is allocated
  after the engine has sized its key/value pool. The n-gram tables are read from disk and are not held on the cards.
- Upstream's fast exchange between ranks is admitted only on a four-card ring of direct NVLink connections; its notes
  say full-mesh groups keep the ordinary path. Our cards sit behind NVSwitches, so expect the ordinary path.
- Upstream memory notes: a prefill chunk of 8,192 runs out of memory on 32 GB cards; packed weight copies add about
  1.25 GiB per rank at load; a manual key/value budget does not shrink to leave room for temporary peaks.
- Open upstream: MTP decode deadlocks with two or more concurrent requests on NVFP4 (issue 865, open since
  2026-10-03). First requests at a new batch size compile Triton kernels. Several long prompts queue behind chunked
  prefill. A feature request (1100) asks for the fast FP16 lanes when the model is served with vision.

### Our copy of the fork is two months old

`cloudrift-ai/1Cat-vLLM` was last pushed on 2026-08-13. It is 3,398 commits behind upstream and carries 13 commits of
our own: the DeepSeek V4 long-context path and KV pool, a Laguna FP8 precision fix, Poolside parser fixes and warm-up
changes. It has no `qwen4_exp`. Every V100 recipe pins an image built from it.

### Emmy today

- `recipes/Qwen3.8-Flash-Next` is an untested shell for H200, H100 and B200.
- Emmy's own kernels have nothing for this architecture. Transformers main has a `qwen4_exp` class, but our pin
  (5.14) predates it. Missing in Emmy: tracer coverage for the layer's parts, Gated DeltaNet together with MoE, the
  MoE split for Qwen's shared-expert gate, Qwen Sparse Attention, the gated residual, the n-gram embedding, MTP, and
  vision.
- The plugin's own Gated DeltaNet layers run eager and refuse prefix caching, speculative decoding and CUDA graphs.
  Only Emmy's own Gated DeltaNet state layer declares the hybrid cache vLLM needs; a fork layer hosted inside the Emmy
  model has no such declaration yet.
- The plugin refuses speculative decoding under pipeline parallelism and for a checkpoint whose embedding is not tied
  to the output layer. This checkpoint's is not tied.
- The plugin's tensor parallelism slices only the routed experts. Everything else Emmy runs is replicated per rank.
- Emmy's loader reads FP8, packed int4, MXFP4 and NVFP4 weights. DeepSeek's FP8 and MXFP4 run in Emmy's kernels on
  V100; NVFP4 has run only on the RTX 5090. There is no GGUF reader.
- Where Emmy's kernels serve this family, the evidence is thin and not in Emmy's favor. Qwen3.8-27B-FP8 on four
  V100s: at four concurrent requests Emmy gave 5.4 tok/s in total against 40.8 for the fork. Emmy ran PP4 × TP1 and
  the fork TP4, and the Emmy rows are single probes. On one RTX 5090 with BF16 weights and one request, Emmy gave 26.3
  tok/s running eager against 63 for stock vLLM with CUDA graphs. On the 27B, one request spends about 2.85 ms per
  Gated DeltaNet layer per token in Emmy's kernels, and the layer serves the requests of a step one after another.
- Native serving handles one dense Qwen3 request on one GPU. It has no MoE, no second GPU and no batching.

### The node

16× V100 SXM3 32GB (512 GiB, about 550 GB), 80 CPUs, 1,338 GB of RAM, CUDA 12.9, driver 580. By card count one node
holds four TP4 replicas; whether four run well together is not known. The test node has NVLink switched off by a
driver option; production nodes have it on.

## What DeepSeek taught, applied here

1. **Check the model does the job before any port.** DeepSeek took six weeks of serving work and then failed at exact
   details in long prompts, and we never learned whether the checkpoint or the Volta runtime was at fault. Step 1 comes
   first and includes a reference.
2. **The fork moves fast.** Our DeepSeek baseline was the August fork. Pin the exact upstream build and measure it on
   our node before comparing anything against it.
3. **One config pins every width.** Twins captured at one decode bucket and deployed at another cost six hours on 16
   cards.
4. **Size memory from the worst case, on a cold start and on a restart.** Load and short prompts passed at a 0.90
   share; a long prompt killed the engine. On the 27B the same share gave a cache 1.5 GiB larger after a warm restart,
   and that backend died five days later. Pin the key/value pool in bytes.
5. **Make two boots agree before any numeric comparison.** The fork tuned kernels at warm-up and every boot gave
   different text until that was pinned.
6. **Plan for the image size.** The first pull of a 164 GB image took 41 minutes and hit a 30-minute limit.

## Steps

Each step ends at a gate. A failed gate stops the plan there, with the evidence written down.

### 0. Before any work

- Read the license for paid serving and for baking weights into a published image. A no stops the plan.
- The owner confirms the minimums in step 2, and turns NVLink on at the test node.
- Create the experiment directory first, with the grid of step 2 as its matrix, so every run below lands in one place.
- Decide the fork binary: the published wheel or our source build. Upstream warns that source builds can take slower
  fallback paths. Steps 1 to 3 measure one binary, and step 3 builds the image from it.

### 1. Does the model do our jobs?

Pin the build before anything runs: the upstream commit, the wheel and its checksum, the serve script, and the Torch
and CUDA versions. NVFP4 runs from the v1.5.1 release wheel.

Run the RadixArk NVFP4 checkpoint on four cards of the test node, MTP off and then on.

- **Topology.** Confirm from the engine's start-up report which exchange path our cards get. Expect the ordinary one.
- **Reference.** The BF16 checkpoint on stock vLLM at a pinned version, on rented H200s through the existing recipe
  shell. The same prompts, the same sampling, the same chat settings.
- **Review set.** Ten merged pull requests, from 10,000 tokens up to the review tool's 128,000-token limit, three runs
  each, for three models: the current reviewer (Qwen3.8-27B-FP8), the candidate, and the reference. A review counts as
  usable when the tool accepts the answer, it is not cut off or empty, every file it names is in the change, every
  line it cites is inside a changed hunk, and no finding is false when checked against the change. The review tool's
  time limit is raised for these runs, and an answer that misses it is counted apart, so speed does not decide
  quality here.
- **Accuracy sets.** The 28 acceptance cases run against the DeepSeek offering, the long-document probes from the
  DeepSeek results at their recorded sizes, and 128 GSM8K questions. GSM8K runs at temperature 0 so the 3-point
  gate is not lost in sampling noise; the rest use the model card's sampling for each mode, three runs each.
- Gate, all of:
  - usable reviews: the candidate is at most 2 of 30 behind the current reviewer;
  - against the reference: GSM8K within 3 points, acceptance cases equal or better, long-document probes within 2
    correct answers per size;
  - no run ends by hitting the output limit.

  If NVFP4 fails where the reference passes, try FP8 on eight cards before stopping.

### 2. Pick the serving shape and record the baseline

One grid on the test node: checkpoint (both NVFP4 builds, FP8) × layout (TP4, PP2 × TP4) × MTP (off, on). Only
weights Emmy can read are in it, because step 5 compares Emmy against the fork on the same checkpoint. GGUF is out:
Emmy has no GGUF reader. FP8 is what Emmy already runs on V100 for DeepSeek; NVFP4 on V100 is new for Emmy, close to
the MXFP4 experts it runs there.

Proposed minimums, for the owner to confirm in step 0:

| Measure | Minimum | Why this number |
| --- | --- | --- |
| Context | 131,072 tokens | the review tool sends up to 128,000 and needs room for the answer |
| Decode, one request | 40 tok/s | five times what the node serves today, half of upstream's report |
| First token, 8K prompt | 10 s | upstream reports 4.5 s |
| First token, 128K prompt | 120 s | leaves time for the answer within 280 s, the review tool's limit raised to just under Relay's 290 s |
| Four requests, one of them a 128K prefill | no stall for 30 minutes; each stream decodes at 20 tok/s or more | long prompts queue behind chunked prefill upstream |

Per cell:

- Memory per rank at its peak, with the key/value pool pinned in bytes. Measure on a cold start and on a warm restart,
  with a full-context prompt, with generation that runs well past prefill, and with four long prompts at once. The
  cache for host-memory history, the n-gram temporary and the packed weight copies are outside the engine's own
  estimate and have to be counted.
- Hold 3 GiB per card back from the fork's pool. Emmy's prefill needed about 2 GiB per card more than the fork's on
  DeepSeek, and the step 5 comparison runs both at the same pool. A shape that only fits without that margin is not a
  candidate.
- Time to first token and decode speed at 1, 4 and 8 concurrent requests, including long prefill mixed with decode;
  the step 1 accuracy sets; boot time; whether two boots give the same text.
- All replicas of the layout started together on the node, not one in isolation: shared disk reads of the n-gram
  tables, pinned host memory, CPU placement and start-up traffic can change the result.

Decide: replicas per node, context, MTP on or off, n-gram tables on disk or in RAM, history on the cards or in host
memory. MTP stays off unless issue 865 is fixed in the pinned build or does not reproduce in a 30-minute run at eight
concurrent requests. This decision sets the production configuration, and step 5 measures parity against it: if
MTP is on here, Emmy has to serve with MTP before the plan is done.

Then record the baseline Emmy has to match, on the chosen shape, with the method of the DeepSeek A/B: first token at
1K, 8K and 32K prompt tokens and at the full context, per-token decode at one request, total output at the serving
concurrency, three repeats each. Add a per-phase profile of one decode step and one prefill chunk: GPU time by part
of the layer, host time, and time in exchanges between ranks.

Gate: one shape meets every confirmed minimum and the step 1 accuracy gate with all its replicas running together,
and its baseline and profile are in the results.

### 3. Bring our fork up to date

- Move `cloudrift-ai/1Cat-vLLM` to the build steps 1 and 2 used. Sort each of our 13 commits: already upstream, still
  needed, or dropped. For the ones that fixed a model we serve, run a targeted check on the new build before deciding:
  a full-context DeepSeek V4 prompt, Laguna FP8 output precision, Poolside single-line tool calls, and the warm-up
  memory peak.
- Build the image from `docker/1cat-vllm-sm70/` under the canonical name
  `cloudriftai/1cat-vllm-<model-slug>:<runtime-version>-<source-sha>`. Review the Dockerfile's defaults against the new
  fork. Warm the Triton and GEMM caches into the image, so the first requests compile nothing. Decide whether the
  weights are baked in or downloaded.
- Existing V100 recipes keep their pinned images. Moving one of them to the new fork is a requalification of that
  recipe, outside this plan.
- Publication goes through `emmy publish`: the dry run, the labels and the digest check, then a separate approval for
  the registry write.
- Gate: step 2's numbers reproduce on our image within the spread of repeated runs, and it boots offline with no new
  compile.

### 4. Emmy serves the model, correctly

Speed does not matter in this step. The split follows DeepSeek: the fork hosts what Emmy does not take, and Emmy
takes more as later rounds need it.

- **First split.** Emmy takes the gated residual, the norms, the router, the shared expert and the routed experts.
  The fork hosts Qwen Sparse Attention and the Gated DeltaNet layers, reads the n-gram tables, and keeps the MTP head
  and vision. This is the closest shape to what already works for DeepSeek, where Emmy took the stream mixing and the
  experts and the fork hosted attention.
- **The work, in order**, each its own pull request with its own gate:
  1. a Transformers pin that has `qwen4_exp`, and tracer coverage for the layer's parts, including layer 2, which
     also reads the n-gram embedding;
  2. trace seams for those parts, so the fork-hosted pieces sit between Emmy's programs;
  3. the loading contract: which checkpoint and quantization Emmy reads, who owns the n-gram tables, and how the MTP
     draft shares weights with the target;
  4. the MoE split for Qwen's shared-expert gate, with every routed expert sliced across tensor-parallel ranks;
  5. hosting the fork's sparse attention inside the Emmy model, as DeepSeek hosts its attention;
  6. hosting the fork's Gated DeltaNet layers, which needs vLLM's hybrid cache declared for a layer Emmy did not
     write;
  7. decode under capture with both hosted layers present, with prefix caching on. The fork's speed depends on both;
  8. the serving twins, one per layer class: the Gated DeltaNet layers, the sparse-attention layers and layer 2, and
     one serving config that pins every width;
  9. a golden recorded on the V100 for every twin at every width, then a boot under strict evidence.
- **Precision.** Volta has no BF16, so Emmy runs FP16 and keeps FP32 where range or accumulation needs it: the
  router (DeepSeek's router bias rounded to FP16 flipped expert picks), the gated residual's partial sums, the Gated
  DeltaNet state (the checkpoint asks for FP32) and the residual stream. Each is checked, not assumed.
- **Correctness against the fork**, on the step 2 shape, with the limits upstream uses for its own builds: KL
  divergence of the output distributions at mean 0.001, p99 0.01 and max 0.05 under teacher forcing, top-1 agreement
  of 99% or more, and a largest logit error of 0.5. Measure over long outputs, four concurrent requests, prefix reuse
  and the carried Gated DeltaNet state across steps, then run the step 1 accuracy sets. First-token parity alone does
  not qualify this model.
- Gate: the Emmy image boots strict on the test node, two boots give the same text, and it passes the correctness
  checks and the accuracy sets within the step 1 tolerances.

### 5. Close the gap to the fork

Rounds, as with DeepSeek. Each round starts from a profile and ends with a boot and an A/B.

- **A round.** Profile one decode step and one prefill chunk of the Emmy image against the step 2 baseline. Take the
  phase with the largest gap. Fix it where it lives: a recording gap (a twin or a width with no measured row), a
  schedule (re-tile, re-record on the card), a cut, the host loop, or a compiler defect. Re-record, boot, measure.
- **What DeepSeek's rounds found**, to look for first: a program with one slow row and no alternative recorded; host
  time between kernels larger than the kernels (one runtime call per group of experts fixed it); decode running eager
  instead of captured; loop-invariant work recomputed inside a sweep; tensor-core tiles recorded at a poor chunk size.
- **Widening the split.** If the gap sits in a part the fork hosts, there is nothing for Emmy to tune there. If it
  sits in the seams around a hosted part, moving that part to Emmy is a measured decision, not a default. Gated
  DeltaNet is the obvious candidate and a poor one today: on the 27B Emmy spends about 2.85 ms per layer per token,
  against roughly 12 ms for the fork's whole token here, and the plugin replicates it on every rank instead of
  slicing it. Move it only when an isolated program, sliced across ranks and captured, beats the fork's layer.
- **Speculative decoding.** Parity with MTP off on both sides is a milestone, not the end. If step 2 set MTP on for
  production, Emmy has to serve with it before the gate below. That is new work: the plugin refuses speculative
  decoding under pipeline parallelism and for an untied embedding, both true here, and the MTP head is a full
  attention layer the fork owns, so the target has to verify several draft tokens per step through Emmy's programs.
- **Checkpoint with the owner** after every round: the remaining gap by phase, what the next round would attack, and
  the trend over the rounds so far.
- Gate, the definition of done. Both sides run step 2's production configuration: the same checkpoint, layout,
  context, key/value pool, concurrency, prefix-cache and MTP settings; fork, Emmy, Emmy, fork rounds; three repeats
  each:
  - first token: Emmy at or below the fork at 1K, 8K and 32K prompt tokens and at the full context;
  - decode at one request: Emmy at or below the fork per token;
  - total output at the serving concurrency: Emmy at or above the fork;
  - the correctness checks and accuracy sets of step 4 still pass.

### 6. Release

- The serving image through the release pipeline: the golden audit on the card, the correctness gate, warm, bake, the
  offline verify with zero new compiles, then `emmy publish` with its dry run and a separate approval.
- The recipe for the chosen checkpoint with its own Hugging Face id, the V100 SXM3 entry and the Emmy image, an
  experiment holding the A/B, and a `RESULTS.md`.
- Check each of these with `--dry-run` first, and change Emmy only where one fails: several replicas on one 16-card
  node (`--scale-out-strategy replica-parallelism` exists), a host path for the n-gram tables and caches, a pinned
  key/value pool, and the catalog fields Relay reads.
- Emmy version bump and publish, the Relay pin, the production rollout, the offering (price, context, description),
  then the acceptance cases through Relay.
- Relay limits that apply: 290 seconds to the first byte of a non-streaming answer, a two-hour image pull, and
  streaming that arrives in bursts.
- Rollback: the previous Relay pin and the offering set to unavailable. The DeepSeek offering stays until this one has
  passed its gate.
- Gate: the acceptance cases pass through Relay, the step 2 minimums hold through Relay, and the server stays active
  for 24 hours.
- Optional, after the gate: point the pull-request review workflow at this model through its own setting.

### An earlier release on the plain fork

After step 3 the model can ship on the fork image alone, with a recipe and an offering, while steps 4 and 5 run. It
costs the recipe and release work of step 6 once more. It is the owner's call and not part of the goal.

### Out of scope

Native serving. It would first need MoE, quantized weights, more than one GPU and batching.

## Risks

- **The bar is higher than it was for DeepSeek.** In August the fork left whole phases in generic code, and Emmy won
  about 20%. Upstream has since tuned this model's path hard, and the two comparisons we have on this family are both
  against Emmy by a wide margin. For scale: the fork's whole token here is about 12 ms without MTP (80.7 tok/s on four
  cards), while Emmy's DeepSeek image spends 79 to 149 ms per token on sixteen. Parity may take many rounds, or may
  stop at a part only the fork can run fast.
- **A moving target.** Upstream lands changes daily. The baseline is the pinned build of step 1. Re-pinning during
  step 5 resets the A/B.
- **Compile time.** On a four-core V100 host one layer program spent 15 to 75 minutes pricing its kernel sets, and on
  2026-09-18 one greedy resolve of a Gated DeltaNet chunk took 48 minutes; both need re-measuring on main. Twins are
  shared per layer class, so this is three classes at every width, not 48 layers.
- **Memory.** Emmy's prefill needed about 2 GiB per card beyond the fork's on DeepSeek, and upstream already runs
  this model on four cards at a 0.90 to 0.95 share. If step 2's 3 GiB margin does not fit at TP4, the shape is
  PP2 × TP4, where the plugin cannot serve with MTP.

## Side finding: the same update may speed up what we already serve

Upstream reports DeepSeek V4 Flash at 65 to 74 tok/s and Qwen3.8-27B at 50 to 62 tok/s. We serve
DeepSeek-V4-Flash-0731 at about 7 to 12 tok/s per request. The two are not comparable as they stand: upstream ran the
earlier checkpoint on eight cards at a short context with 1,024-token prompts, and we run the 0731 checkpoint on 16
cards at a 1M context. Emmy's 0.8× lead on DeepSeek was measured against the August fork. It is worth one day on the
test node after step 3, at our checkpoint and envelope, and its own plan if it holds.

## Not established

- The quality of NVFP4 and FP8 against full precision on our tasks.
- Whether MTP is safe under concurrency, and whether issue 865 is fixed in any build we would pin.
- The longest context each layout holds on 32 GB cards, and whether PP2 × TP4 works for this model on Volta.
- How much the ordinary exchange path between ranks costs on our NVSwitch topology, and speed without NVLink.
- Whether four replicas run well together on one node.
- How fast vision input runs on Volta.
- What the license allows.

## Effort

No total yet. Steps 1 and 2 are measurement and depend on three things outside our hands: the license answer, NVLink
on the test node, and rented cards for the reference. Step 4 has more new parts than DeepSeek had. DeepSeek took about
six weeks from its first boot to parity; expect step 4 and step 5 together to take longer, and estimate step 5 only
after the first strict boot and its profile.

## Operations

The test node is shared. Check that every card is free before a launch, never stop a process that is not ours, and
delete nothing when done. Its address stays out of the repository. Turning NVLink on there is the owner's change.
