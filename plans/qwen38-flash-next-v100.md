# Qwen3.8-Flash-Next on V100 nodes through Emmy

Goal: serve `Qwen/Qwen3.8-Flash-Next` on CloudRift's 16× V100 SXM3 nodes through Emmy's own kernels, the way
DeepSeek V4 is served: first running, then optimized until the Emmy serving image is at least on par with the plain
1Cat-vLLM fork at an equal serving envelope. The result ships as an Emmy recipe that `emmy deploy` and Relay run.

Emmy's kernels run inside the fork's runtime, which supplies the serving shell and whatever Emmy does not take. So
the fork has to serve the model first: steps 1 to 3 set up the runtime and record the baseline to match, step 4 gets
Emmy serving the model correctly, step 5 closes the speed gap, and step 6 releases.

Written 2026-10-13, revised the same day after a review and after the owner fixed the goal. Each result is written
under its step. Numbers marked "upstream" are 1Cat-vLLM's own reports.

**Where it stands (2026-11-21).** Steps 1 to 4 are done; step 5's done gate is not met. Step 4's gate (2026-11-02):
two boots give the same text and the accuracy sets pass, but the correctness limits are not met, and the fork does
not meet them against itself either. Since the done gate's first run (2026-11-03), step 5 brought Emmy under the
3 GiB per card margin, sliced the shared expert across ranks, split pre8's down rows across blocks, measured where
the first-token and eight-request gaps sit (2026-11-07), ran the eight requests' routed experts in one launch (15%
more output), and found the host time of a decode step equal on both sides. On 2026-11-11 the owner turned async
scheduling on, on both sides, and on 2026-11-12 a prefill step's wider experts began running 32 per launch, the
rest on 2026-11-14. Against the async fork, Emmy is now 1.41x slower to the first token at 1K and 1.44-1.51x at
4K to 32K (2026-11-19), 1.58x slower per token at one request, and gives 0.87x the fork's output at two requests,
0.79x at four and 0.78x at eight (2026-11-21). Async doubles the first token at the full context on both sides; the
owner keeps the 2.5 GiB key/value pool, so that is a known limit (step 5). Of the 2026-11-19 ranking, round 1
stopped at its gate, round 2 shipped and round 3 shipped short of its gate (2026-11-21); rounds 4 and 5 and the
compiler items round 3 left are for the owner to pick. The minimums in step 2 and Qwen's license are still open with
the owner, who plans as if the license were granted.

## What we know

### The model

- 180B parameters: a 125B language model with 6B active per token, a 51B n-gram embedding read at layer 2, a 4B
  one-layer MTP head for speculative decoding, and a vision encoder.
- 48 layers: 12 × (3 × (Gated DeltaNet → MoE) + 1 × (Qwen Sparse Attention → MoE)). The MoE has 512 experts, 10
  routed plus 1 shared. Every layer has a gated residual with 4 branches.
- Context is 262,144 tokens. It thinks by default. The model card's sampling: temperature 1.0, top-p 0.95, top-k 20
  when thinking; temperature 0.7, top-p 0.8, top-k 20, presence penalty 1.5 when not.
- The Hugging Face architecture is `qwen4_exp`. Stock vLLM supports it since v0.29.0. Transformers has it since 5.16;
  Emmy pins 5.19, the version the fork's image ships.
- License is `qwen-community-1.0`, not Apache. A Model-as-a-Service business needs a separate license from Qwen
  before any commercial use. The owner continues the evaluation internally while that is open, so no model output is
  published until it is settled.
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
- Upstream's notes admit its fast exchange between ranks only on a four-card ring of direct NVLink connections. On
  our NVSwitch cards v1.5.1 admitted it anyway (step 1).
- Upstream memory notes: a prefill chunk of 8,192 runs out of memory on 32 GB cards; packed weight copies add about
  1.25 GiB per rank at load; a manual key/value budget does not shrink to leave room for temporary peaks.
- Open upstream: MTP decode deadlocks with two or more concurrent requests on NVFP4 (issue 865, open since
  2026-10-03). First requests at a new batch size compile Triton kernels. Several long prompts queue behind chunked
  prefill. A feature request (1100) asks for the fast FP16 lanes when the model is served with vision.

### Our fork

Until step 3, `cloudrift-ai/1Cat-vLLM` stood at 2026-08-13, 3,398 commits behind upstream, with no `qwen4_exp`. This
model now runs on its branch `cloudrift/v1.5.1` (step 3). Every other V100 recipe still pins an image from the August
build.

### Emmy today

- `recipes/Qwen3.8-Flash-Next` is an untested shell for H200, H100 and B200. `recipes/Qwen3.8-Flash-Next-NVFP4` holds
  the V100 serving config and golden, in a locked shell that proposes no deployment.
- Emmy runs the model on four V100s under strict evidence (step 4). It hosts the fork's Qwen Sparse Attention, Gated
  DeltaNet layers and n-gram table, and runs the pre and post programs and the experts. MTP and vision are not served
  through Emmy.
- The plugin refuses speculative decoding under pipeline parallelism and for a checkpoint whose embedding is not tied
  to the output layer. This checkpoint's is not tied. Step 2 chose MTP off, so this does not block the done gate.
- The plugin's tensor parallelism slices only the routed experts. Everything else Emmy runs is replicated per rank.
- There is no GGUF reader.
- Native serving handles one dense Qwen3 request on one GPU. It has no MoE, no second GPU and no batching.

### The node

16× V100 SXM3 32GB (512 GiB, about 550 GB), 80 CPUs, 1,338 GB of RAM, CUDA 12.9, driver 580. By card count one node
holds four TP4 replicas; whether four run well together is not known. The test node has NVLink on since its rebuild
on 2026-10-14, like the production nodes.

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

- Done: the license is read (see "The model"), NVLink is on at the test node, and the fork binary is the v1.5.1
  release wheel. Upstream warns that source builds can take slower fallback paths; steps 1 to 3 measure this one
  binary, and step 3 builds the image from it.
- Open: the owner confirms the minimums in step 2.
- Done differently: step 2 screened its grid by hand, one boot per cell, and its experiment directory records the
  chosen shape's baseline. Step 1's raw results stayed outside the repository.

### 1. Does the model do our jobs?

Pin the build before anything runs: the upstream commit, the wheel and its checksum, the serve script, and the Torch
and CUDA versions. NVFP4 runs from the v1.5.1 release wheel.

Run the RadixArk NVFP4 checkpoint on four cards of the test node, MTP off and then on.

- **Topology.** Confirm from the engine's start-up report which exchange path our cards get.
- **Reference.** The BF16 checkpoint on stock vLLM at a pinned version, on rented H200s through the existing recipe
  shell. The same prompts, the same sampling, the same chat settings.
- **Review set.** Nice to have, not part of the gate (the owner's decision of 2026-10-14). Ten merged pull requests,
  from 10,000 tokens up to the review tool's 128,000-token limit, three runs each, for three models: the current
  reviewer (Qwen3.8-27B-FP8), the candidate, and the reference. A review counts as usable when the tool accepts the
  answer, it is not cut off or empty, every file it names is in the change, every line it cites is inside a changed
  hunk, and no finding is false when checked against the change. The review tool's time limit is raised for these
  runs, and an answer that misses it is counted apart, so speed does not decide quality here.
- **Accuracy sets.** The 28 acceptance cases run against the DeepSeek offering, the long-document probes from the
  DeepSeek results at their recorded sizes, and 128 GSM8K questions. GSM8K runs at temperature 0 so the 3-point
  gate is not lost in sampling noise; the rest use the model card's sampling for each mode, three runs each.
- Gate, all of:
  - against the reference: GSM8K within 3 points, acceptance cases equal or better, long-document probes within 2
    correct answers per size;
  - no run ends by hitting the output limit.

  If NVFP4 fails where the reference passes, try FP8 on eight cards before stopping.

**Result, 2026-10-14.** The NVFP4 checkpoint on the plain fork matches the full-precision reference on every quality
check. One point is open to the owner: three fork runs ended at their output limit.

- **Pins.** Fork release v1.5.1 (tag commit `589a5d1e1a9e`), wheel sha256 `757f2200ba6b…`, on
  `nvidia/cuda:12.8.1-devel-ubuntu24.04` plus `python3.12-dev`: Triton compiles a helper against `Python.h` at
  start-up, and without it the model class does not load. Torch 2.10.0+cu128, Triton 3.6.0, Transformers 5.19.0.
  RadixArk NVFP4 at `7b719225242a`. Launcher `serve_flash_next_nvfp4_v100.sh` as shipped, on cards 0 to 3, plus the
  served name and the `qwen3` reasoning and `qwen3_coder` tool parsers it does not set. MTP off is
  `--speculative-config null`. Reference: stock vLLM v0.31.0, BF16 at `de4b8e4d43b9`, 4× H200 rented for 43 minutes,
  context 163,840, thinking off by default like the fork, no MTP.
- **Start-up.** The fast exchange between ranks is admitted on our NVSwitch cards (custom all-reduce, NCCL as the
  fallback). Boot takes 8 min 47 s cold and 6 min warm. Weights take 22.9 GiB per card; the n-gram table stays in
  pinned host memory, read through 47.7 GiB of file mappings. The KV pool is 173,353 tokens with MTP and 272,511
  without, the same on cold and warm starts. A card holds 29.4 GB idle and 31.2 GB after a 32K prompt.
- **Thinking.** The fork turns thinking off for every request that does not ask for it (its server default sets
  `enable_thinking` to false); the model card says thinking is on by default. Every test set the switch explicitly.
- **Speed, one request** (random prompts, 512 output tokens, greedy):

  | Prompt | First token, MTP on | First token, MTP off | Per output token, MTP off |
  | ---: | ---: | ---: | ---: |
  | 1K | 0.30 s | 0.51 s | 13.6 ms |
  | 8K | 1.57 s | 1.50 s | 13.0 ms |
  | 32K | 6.26 s | 5.89 s | 13.5 ms |

  On natural prompts the fork decodes 78 tok/s without MTP (upstream 80.7) and 114 tok/s with MTP4 (upstream 138 to
  150; 2.9 tokens per step). Prefill runs at about 5,250 tok/s (upstream 7,056 at 8K); a 130K prompt takes 25.7 s.
- **Quality against the reference:**

  | Check | Fork, NVFP4 on 4× V100 | Reference, BF16 on 4× H200 |
  | --- | --- | --- |
  | GSM8K, 128 questions, temperature 0, thinking off | 125 (MTP off), 124 (MTP on) | 125 |
  | Acceptance cases, three runs, thinking off / on | 27, 27, 27 / 27, 27, 28 | 27, 27, 27 / 26, 27, 27 |
  | Long-document codes found, three runs, off / on | 387 of 387 / 383 of 384 | 387 of 387 / 383 of 384 |
  | Usable reviews of 30 (not gating) | 7 | 6 |

  The long-document probes plant one code in repository text at 6.9K to 130.7K tokens, eight documents at 15K and at
  30K one at a time and at once, three codes at 1.5K to 120K tokens in prose and in a synthetic log, and one code per
  question in nine of those documents. Unlike DeepSeek V4, the model finds the codes at every size. The one
  acceptance case both miss is a three-step sum asked with thinking off.
- **Output limit.** Three fork runs reached it, none on the reference. Two GSM8K answers reasoned at length without
  thinking mode and reached the 2,048-token cap (MTP off), and one thinking answer used its whole 8,192-token budget
  without an answer.
- **Reviews.** Of 89 findings from the three models, checked against the code, 2 were true. A review is usable
  almost only when it finds nothing on a small change; the current reviewer had 7 usable of the 19 runs Relay
  completed. The tool's 128,000 is its own token count: its largest prompts are 129K to 145K tokens for this model,
  so a 131,072 context refuses them, and 163,840 (a 178,339-token KV pool with MTP) fits them.

### 2. Pick the serving shape and record the baseline

One grid on the test node: checkpoint (both NVFP4 builds, FP8) × layout (TP4, PP2 × TP4) × MTP (off, on). Only
weights Emmy can read are in it, because step 5 compares Emmy against the fork on the same checkpoint. GGUF is out:
Emmy has no GGUF reader. FP8 is what Emmy already runs on V100 for DeepSeek; NVFP4 on V100 is new for Emmy, close to
the MXFP4 experts it runs there.

Proposed minimums, for the owner to confirm in step 0:

| Measure | Minimum | Why this number |
| --- | --- | --- |
| Context | 131,072 tokens | the launcher's default; serving the review tool would need about 160,000 (step 1), wanted but not required |
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

**Result, 2026-10-15.** The shape is the RadixArk NVFP4 checkpoint at TP4 with MTP off, four replicas per node, a
131,072 context, the KV pool pinned at 2.5 GiB per card (194,016 tokens), a 4,096-token prefill step, up to eight
sequences, prefix caching, the n-gram table in pinned host memory and history on the cards. It meets every proposed
minimum with all four replicas running and passes the step 1 accuracy sets. The minimums still wait for the owner's
confirmation.

**Async scheduling on, 2026-11-11.** The owner turned vLLM's async scheduling on in this shape, on both sides: the
host prepares the next step while the GPU runs the current one. The shape above ran it off, as the fork's release
example does; with MTP off it is vLLM's default, so the recipe and the experiments now just drop
`--no-async-scheduling`. The production baseline is the experiment
`experiments/Qwen3.8-Flash-Next/fork_baseline_v100_sxm3`, re-run with async on the step 3 image (result below the
synchronous rows). Its first run, the synchronous numbers below, stays as history.

- **The grid, one boot per cell by hand.** `emmy bench` deploys every row cold, about 9.5 minutes each, so the full
  grid would have cost about 56 boots; the screen took one boot per cell and the experiment records only the chosen
  shape.

  | Cell | Result |
  | --- | --- |
  | RadixArk TP4, MTP off / on | boots; 272,511 / 173,353 KV tokens at a 0.90 share; 78 / 114 tok/s on natural prompts |
  | nvidia NVFP4 TP4, MTP off / on | boots; same speed and memory as RadixArk (MTP on 209,292 KV tokens: its draft experts are FP8) |
  | RadixArk PP2 × TP4, MTP off | boots only with `--block-size 816` (the n-gram table placement sizes the cache before the attention block is enlarged); 70 tok/s, 3.7 s to the first token at 32K against 5.9 s at TP4 |
  | RadixArk PP2 × TP4, MTP on | boots, then dies on the first request with a device-side assert |
  | Qwen FP8 PP2 × TP4 | fails at load: the fork's SM70 FP8 preparation rejects the checkpoint's scale layout; at TP4 the weights do not fit |

  The nvidia PP2 cells were not run: that checkpoint matches RadixArk at TP4, and RadixArk is the one step 1
  qualified against the reference.
- **Issue 865 does not reproduce.** MTP4 at eight concurrent requests ran 900 requests over 29 minutes with no stall
  (264 tok/s in total), so the rule allows MTP on.
- **The 3 GiB margin, tested directly.** A process holds 3 GiB on every card, the server starts beside it, and the
  worst-case loads run: one 130K prompt, four 32K prompts at once, a 64K prompt with 4,096 output tokens, and a 128K
  prefill arriving among three decoding requests. TP4 with MTP off passes with about 0.6 GiB to spare; PP2 × TP4
  with MTP off and the n-gram table moved to host memory (`VLLM_QWEN4EXP_PLE_HOST_GIB=12`) passes with about 8.7 GiB
  to spare; TP4 with MTP on runs out of memory at four 32K prompts even with its pool cut to 131,072 tokens. The
  launcher's own 8,192-token step at a 0.95 share leaves the fork about 0.3 GiB before any margin. So MTP is off: at
  TP4 it cannot keep the margin, and under PP it crashes.
- **TP4 over PP2 × TP4.** Four TP4 replicas give a node about 790 tok/s at eight requests each (measured, through the
  load balancer), two PP2 replicas about 460 (from one replica's 228). TP4 decodes faster per request (13 ms against
  14 ms); PP2 reaches the first token sooner on long prompts and keeps far more memory free. PP2 × TP4 stays the
  fallback if Emmy needs more than the 3 GiB.
- **All four replicas together.** They boot in about 7 minutes side by side and do not slow each other: each ran the
  four-request case for 30 minutes (three decoding streams, 128K prompts arriving back to back) with no stall or
  failure, the streams averaging 25 tok/s (about 6 tok/s while a 128K prompt is being prefilled), and each 128K
  prompt reaching its first token in 49-50 s. Host RAM in use: 327 GB, mostly the pinned n-gram tables.
- **Two boots give the same text.** Six greedy prompts came back identical from the four replicas.
- **Accuracy on this shape.** GSM8K 124 of 128, none at the cap; acceptance cases 27 of 28 in each mode (the known
  arithmetic miss); every long-document code found, 129 of 129 without thinking and 128 of 128 with it.
- **Baseline, one replica, repeats 2-4 of 4, synchronous (the first run, history):**

  | Load | First token | Per output token | Total output |
  | --- | ---: | ---: | ---: |
  | 1 request, 1K / 8K / 32K / 130K prompt | 0.49-0.54 / 1.57 / 6.12-6.22 / 25.9-26.2 s | 13.0-13.5 / 13.0-13.1 / 13.3-13.4 / 14.1-14.2 ms | |
  | 4 requests, 4K / 1K | 2.5 s mean | 20.6 ms | 173-174 tok/s |
  | 8 requests, 4K / 1K | 3.5 s mean | 25.2-25.3 ms | 278-280 tok/s |
  | 4 replicas, 32 requests, 4K / 1K | 7.9-11.9 s mean, 2.8-4.0 s median | 26.3 ms | 775-811 tok/s |

- **Profile, one request, rank 0.** About 11 ms of GPU work per decode step against a 13 ms token, so about 2 ms is
  host time. Per decode step: dense FP16 projections about 3.7 ms, exchanges between ranks (the fast all-reduce and
  the fused hyper-connection gathers) 2.7 ms, MoE 1.9 ms, hyper-connection local work 0.9 ms, sparse attention
  0.8 ms, Gated DeltaNet 0.5 ms. A 3,920-token prefill chunk takes about 0.6 s, led by dense GEMMs and the NVFP4
  expert GEMMs; prefill exchanges go over the NCCL ring, not the fast path.
- **Baseline with async scheduling (2026-11-11), the production baseline.** The same experiment and protocol on the
  step 3 image; repeats 2-4 of 4:

  | Load | First token | Per output token | Total output |
  | --- | ---: | ---: | ---: |
  | 1 request, 1K / 8K / 32K / 130K prompt | 0.47-0.52 / 1.59-1.60 / 6.13-6.21 / 46.3-46.6 s | 9.59 / 9.72-9.74 / 10.0 / 10.9 ms | |
  | 4 requests, 4K / 1K | 2.5 s mean | 17.2-17.3 ms | 203 tok/s |
  | 8 requests, 4K / 1K | 3.7-3.9 s mean | 21.3-21.6 ms | 318 tok/s |
  | 4 replicas, 32 requests, 4K / 1K | 7.4-8.8 s mean, 3.3-3.4 s median | 22.1-22.4 ms | 897-1,008 tok/s |

  - Async takes 3.2-3.9 ms from every decode step and gives 14-17% more output at four and eight requests. First
    tokens up to 32K do not move; at the full context the first token doubles, because the engine preempts the
    prompt's prefill twice and starts it again (step 5, "Async scheduling on both sides").
  - The other checks of this step, on boots never profiled: GSM8K 124 of 128, none at the output limit; acceptance
    cases 27 of 28 in all six runs; long-document lookups 339 of 339 and 336 of 336. The margin test passes, with
    338-398 MiB spare per card (0.4-0.6 GiB with async off). Two boots agree on 3 of 6 greedy prompts: the fork's
    warm-up tune picks kernels anew at every boot, and with async the batches differ from run to run; every
    divergence is a near-tie (step 5).

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
- *Done, 2026-10-16:* the fork branch `cloudrift/v1.5.1` (`248d877c5`) is upstream v1.5.1 plus the Poolside tool-call
  parser and the Laguna FP8 precision fix. Our warm-up trio and the decode-workspace commit are dropped, the rest was
  upstream already, and the non-MTP Gated DeltaNet full-forward guard is reverted: it cost about 130 ms per first
  token, and upstream does not use it. The image reproduces step 2 within the spread (`fork_image_v100_sxm3`) and
  keeps the same 3 GiB per card free under the worst-case loads. With the fork's JIT caches baked in,
  `cloudriftai/1cat-vllm-qwen3.8-flash-next-nvfp4:1.5.1-248d877c5` boots in 303 s instead of 553 s and compiles
  nothing at boot or on any request path the warm reached (`fork_serving_v100_sxm3`); the weights are downloaded at
  deploy. Not met as written: the boot is online, because `HF_HUB_OFFLINE=1` moves vLLM's compile key to the host's
  model path. Not run: the full-context DeepSeek V4, Laguna FP8 and Poolside checks, which come with moving those
  recipes to this fork. Published 2026-10-16 to the private Docker Hub repository, digest
  `sha256:bf976828fc00…`, through `emmy publish`. Since async scheduling went on (2026-11-11) the baked caches miss:
  the flag is part of vLLM's compile key, so the image boots in 421-464 s again and needs a bake in the async shape
  before a release.

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
     *Done on 2026-10-15:* the pin is 5.19, and the tracer takes the Gated DeltaNet layer, the sparse-attention layer
     and layer 2 (`--layer 1`). Layer 2 takes the looked-up n-gram embeddings as an input; the table stays with the
     runtime. Sparse attention traces only up to 2,051 tokens, where its indexer keeps every visible token and the
     causal mask is exact. `emmy trace` writes goldens for `--layer 0` and `--layer 3` but does not finish on `--layer
     1`: the compiler did not lift that whole-layer program into Tile IR in 15 minutes, and a tiny config hangs the
     same way. Also in the compiler: at 8 tokens Loop IR normalization fails with an index error on the Gated DeltaNet
     layers. The first split hosts those layers in the fork, so neither blocks items 2 to 4;
  2. trace seams for those parts, so the fork-hosted pieces sit between Emmy's programs;
     *Done on 2026-10-15:* a Qwen4Exp layer takes DeepSeek V4's hyper-connection seam. The carrier is the gated
     residual's four streams flattened, 10,240 wide; `pre` returns the attention-site mixed input, and `post` injects
     the mixer output, runs the MLP-site gated residual and injects the gated shared expert, returning the streams,
     the MoE input and the four injection weights for the routed combine. The fork's own layer cuts at the same
     points: its Gated DeltaNet and sparse attention take that mixed input, and its n-gram PLE adds to the streams
     before `pre`. Every layer folds into one twin profile; the real config's twins (TP4, widths 1, 8, 4,096 and any
     width) lower in under 10 s;
  3. the loading contract: which checkpoint and quantization Emmy reads, who owns the n-gram tables, and how the MTP
     draft shares weights with the target;
     *Contract, 2026-10-16:* Emmy reads RadixArk's NVFP4 checkpoint (revision 7b719225), the one step 1 qualified. It
     codes only the routed experts, as modelopt NVFP4 trios; the gated residuals, the router and the gated shared
     expert stay unquantized. Emmy runs the experts weight-only: the fork's V100 NVFP4 experts take FP16 activations
     and delete the stored input scales, so Emmy does not read them either. The fork owns the Gated DeltaNet, the
     sparse attention and the PLE with its n-gram tables; MTP is off in the step 2 shape, so the draft's weight
     sharing waits until MTP is on, and vision is not loaded. Every one of the checkpoint's 296,475 weights has
     exactly one owner under that split. The NVFP4 expert twins (packed pairs, e4m3 block scales, one global scale per
     row) capture from the config and weight index alone and lower in under 10 s at TP4; the shards are never fetched.
     The serving loader keeps each rank's expert slice packed (gate and up keep their own global scales, the input
     scales are skipped) and the runner spells the expert program's inputs to match; on the V100 the compiled NVFP4
     expert program matches the eager expert on the dequantized weights. Found for item 5: the same loader still reads
     the fork-owned weights into Emmy's model, the n-gram table (about 51 GB) included, so the ownership split has to
     be enforced at load;
  4. the MoE split for Qwen's shared-expert gate, with every routed expert sliced across tensor-parallel ranks;
     *Done, 2026-10-16:* every rank runs `pre` and `post` whole, the gated shared expert included, as DeepSeek keeps its
     shared experts, plus its slice of every routed expert; only the routed output is all-reduced. The fork slices the
     shared expert across ranks and reduces its partial output together with the routed experts' (its allocations,
     step 5's "Next (2026-11-03)"; a reading of the 2026-10-27 profile had said it reads it whole), and it keeps the
     hyper-connection mixers' weights whole on every rank. The router runs as the checkpoint's module does
     (FP16 logits, FP32 softmax, top-10 renormalized), and the correctness step checks it against the fork. A CPU test
     at two ranks reproduces the eager layer. The split loader no longer renames Qwen's `mlp.shared_expert` to the
     plural Laguna and GLM use, which had left the shared expert unloaded;
  5. hosting the fork's sparse attention inside the Emmy model, as DeepSeek hosts its attention;
     *Done, 2026-10-17:* the plugin builds the fork's sparse-attention module for every sparse-attention layer, with
     one top-k buffer and cache names per absolute layer, and loads its weights through the ownership table, query,
     key and value as the three shards of its fused projection. The runner closes the streams with the checkpoint's
     last gated residual, which takes the final norm's place, and its serving load leaves the attention and the PLE
     unread. On the V100 a tiny sparse-attention checkpoint builds under a real vLLM config, every layer registers
     its caches, the weights land in the fork modules, and the final collapse matches the module. Serving waits for
     item 6: the fork lays out the sparse attention's caches together with the Gated DeltaNet state and refuses a
     model without it, so the fork-against-Emmy comparison on a tiny checkpoint moves there. Found: the first Qwen
     `post` program to run on the V100 (16 tokens, picks with no measured rows) did not finish in 30 minutes in the
     boot audit; and no item hosts the PLE's add before layer 2 (proposed: item 6);
  6. hosting the fork's Gated DeltaNet layers, which needs vLLM's hybrid cache declared for a layer Emmy did not
     write;
     *Done, 2026-10-22 (#1224):* every Gated DeltaNet layer gets vLLM's `QwenGatedDeltaNetAttention` and the PLE
     layer the fork's own PLE module, added to the streams before that layer's `pre`; the hybrid class declares the
     state as the fork's model class does and hands the runner the fork's model state, which feeds the PLE. On a tiny
     checkpoint in the published layer order, in the fork image on the V100, Emmy generates exactly the fork's own
     tokens, alone and in a batch. Serving needs `VLLM_USE_V2_MODEL_RUNNER=1` (the fork picks that runner by
     architecture name) and a fixed table host share (`VLLM_QWEN4EXP_PLE_HOST_GIB`): Emmy's class reads the table
     inside the worker, where the fork's own placement sizes the KV cache before the attention pages are widened and
     refuses them, as step 2's PP2 boot found. The fork runs a one-token first step as a decode step over a state
     block it never zeroed, its own model too. Emmy's loader no longer reads the hosted layers' weights;
  7. decode under capture with both hosted layers present, with prefix caching on. The fork's speed depends on both;
     *Done, 2026-10-22 (#1229):* no code change was needed. On a tiny checkpoint in the fork image on the V100, Emmy
     generates the fork's own tokens with single-token decode steps captured, and on a prefix-cache hit over 700
     shared tokens (`mamba_cache_mode` "align", which needs a step of at least one block) with Emmy capturing. The
     capture ran on the symbolic programs: with the decode bucket on, the cold pick of the 16-token `post` is one fused
     kernel (about 10^14 iterations per thread) that the boot audit stops after a minute, the item 5 hang, so the
     static decode tier waits for item 9's golden. `emmy serve` forces eager decode when the decode bucket is off;
  8. the serving twins, one per layer class: the Gated DeltaNet layers, the sparse-attention layers and layer 2, and
     one serving config that pins every width;
     *Done, 2026-10-22 (#1232, #1233):* every layer folds into one twin profile, so the twins are 12 programs: pre and
     post at 8 rows, 4,096 rows and any width, the expert program at 1, 8, 256, 4,096 and any width, and the
     fixed-slot experts. `recipes/Qwen3.8-Flash-Next-NVFP4/serving-v100.env` pins them with the fork's serving shape,
     the V2 runner, the n-gram table in host memory and strict evidence, in a locked onboarding shell that proposes no
     deployment. The fixed-slot twin did not lower (a value read inside a reduce loop had no scope), fixed by #1232; it
     runs 35 ms per call on the V100, about eager's. The working golden still matches main's lowering; a recipe golden
     must have a schedule on every row, so it joins the recipe with item 9;
  9. a golden recorded on the V100 for every twin at every width, then a boot under strict evidence.
     *Done, 2026-10-23:* `recipes/Qwen3.8-Flash-Next-NVFP4/golden/v100_sm70.json` holds rows for all 12 twins, and
     the release audit deploys every one from the golden alone. The prior's cold picks were unusable (post8 1.95 s,
     the experts 33 ms to 1.4 s); what fixed them: a cut between the expert's gate/up and down projections, post4096's
     route on the small post twins, and tunes on the slow pieces. Serving gives each weight one layout per twin
     family, settled by the narrowest twin, while a record of one twin compiled it alone, so pre4096 recorded 1.6 ms
     and deployed 34 ms under pre8's layouts; a run of one realization now compiles under its family's layouts, and
     the wide twins were recorded again that way (pre4096 1.45 ms, post4096 4.2 ms, the symbolic twins 0.51 and
     1.22 ms at 512 tokens). The real model boots strict on four cards through the release entrypoint, which renders
     the hybrid class from `SERVE_ARCHITECTURE`. On eight greedy prompts, five match the fork for all 64 tokens; three
     diverge at near-ties. Decode is 47 ms per token against the fork's 12.7, mostly the 8-row post program (about
     0.8 ms per layer), and the first token after 2,006 prompt tokens takes 7.4 s against 0.6 s, most of it outside
     pre and post: step 5 starts from those two.
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
- **Gate result, 2026-11-02.** Main `c497a0cc7`, strict boots of the recipe golden in step 2's shape, against the step
  3 fork image in the same shape. Boots, accuracy: pass. Correctness: not met as written, on either side.
  - Two boots give the same text: 16 of 16 greedy prompts identical, and prefill and decode log-probabilities
    bit-identical.
  - Accuracy sets, all within the step 1 tolerances: GSM8K 126 of 128 (reference 125), none at the output limit;
    acceptance cases 27, 27, 28 without thinking and 27, 27, 26 with it (reference 27, 27, 27 / 26, 27, 27);
    long-document codes 387 of 387 and 384 of 384 (reference 387 of 387 / 383 of 384).
  - Correctness, as KL mean / p99 / max and top-1 agreement:

    | Teacher-forced scenario | Emmy against the fork | A second fork boot against the fork |
    | --- | --- | --- |
    | Long outputs, 12 prompts of 1,024 tokens | 0.0064 / 0.14 / 1.10, 99.45% | 0.0073 / 0.16 / 3.50, 99.57% |
    | The same, before the first divergence | 0.00086 / 0.0086 / 0.43, 99.48% | 0.00048 / 0.0026 / 1.11, 99.70% |
    | Four requests at once | 0.00036 / 0.0051 / 0.024, 99.41% | 0.00030 / 0.0039 / 0.026, 99.61% |
    | Prefix reuse, six suffixes on a shared 6K document | 0.0019 / 0.026 / 0.14, 99.22% | 0.0016 / 0.022 / 0.060, 98.96% |
    | Carried state, prompts of 4.5K to 40K tokens | 0.031 / 0.57 / 3.55, 93.55% | 0.028 / 0.50 / 3.64, 94.34% |
    | Prompt log-probabilities over prompt and output | 0.0093 / 0.085 / 0.45, 95.59% | 0.0074 / 0.069 / 0.38, 95.77% |

  - Method: the engine's log-probabilities through the API, the top 20 per position. Where the two sides first pick
    different tokens, the run restarts from the prompt plus the reference's tokens so far; "before the first
    divergence" is the positions before any restart. KL over the shared top 20 plus a rest bucket is a lower bound:
    on 48 positions read over the whole vocabulary it caught 90% of the exact KL (Emmy: exact mean 0.016, max 0.27).
    The largest logit error cannot be measured this way, since log-probabilities leave one offset per position
    unknown; the spread of log-probability differences over shared tokens fails 0.5 on both sides (up to 13 for
    Emmy, 12 fork against fork).
  - Reading: the fork does not meet the limits against itself, so in this shape they do not separate a defect from
    the fork's own variation. Emmy differs from the fork by about what one fork boot differs from another. The fork
    tunes small-shape kernels at warm-up (`VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES`, as DeepSeek's FP8 tune), so its boots
    differ; with the tune off, two fork boots are bit-identical in prefill and equal the second boot above. Against
    that reproducible fork, Emmy's prompt log-probabilities read 0.0087 / 0.079 / 0.46, 95.57%. Whether to relax
    the limits, or to measure them with logits against a tune-off fork, is the owner's call.

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
- **Round notes (2026-10-24).** Decode is GPU busy time, not idle: per step Emmy runs 3,524 kernels for 28.9 ms busy,
  the fork 1,434 for 9.7 ms; the gaps between kernels and the ~70 long gaps of vLLM's step work are the same on both
  sides (~3 ms unprofiled). Emmy's three decode twins take 16.5 ms of it (post8 8.75, pre8 4.1, fixed slots 3.65) and
  read their ~43 MB per layer per rank at 90-300 GB/s. The warm-L2 bench under-reads those pieces by up to 2.2x (the
  cold-cache regime matches serving). Runner round (the hosted sublayers take the fork's fp16 GEMV, fused GDN input
  and fused gated norm; Qwen-family routers take vLLM's fused top-k): 2,708 kernels, 25.4 ms busy, single-request
  TPOT 34.1 -> 28.9 ms against the fork's 13.8 on the same measurement; the probe agrees with the fork as often as
  before. Cold round: the golden is now recorded cold (`EMMY_COLD_CACHE=1`); the fixed-slot experts re-tuned cold
  (39 -> 21.5 and 51 -> 25 us), TPOT 28.9 -> 26.5 ms, TTFT at 4K unchanged (7.1 s; fork 0.9). Traps: a `--kernel`
  tune without `--realization` runs in the default fast-math regime, so its rows are no evidence for the
  FAST_MATH=false golden (re-measure the winners as `--ab` rows on the named cold row); post8's best warm tile for
  linear_4 is an f16-accumulate one the precise regime does not offer; a 1 us tie at post8's linear_2 layout fork
  flips the post family's layouts and re-keys post4096 (settled by recording the source piece's tuned schedule).
  Cut round (2026-10-25): a matmul whose operand is computed inline (post8's SiLU(x/c), pre8's nested 320-long
  dot product) is offered one k4-staged tile; cutting the operand out (`PLACE@place_<tok>=cut`, child sites
  `PLACE@place_<tok>/<site>=cut`, and every earlier stage pinned with `PLACE@step.N/<site>=cut` or the root
  collapses) gives a plain matmul: post8 261 -> 216 us, pre8 111 -> 71 us cold, TPOT 26.5 -> 22.9 ms, TTFT
  unchanged. A cut that changes a weight's layout re-lays the family's wide twins out (linear_9: post-sym ran a
  53 ms kernel at prefill widths) — check the audit's layout lines against the last good file before booting;
  post8's linear_2 layout fork is a 1 us tie that a re-record flips (re-record the source piece alone).
  Fixed slots: the down kernel's activation cut out, 58 -> 42 us cold, TPOT 22.9 -> 22.7 ms. Prefill experts
  (2026-10-26): experts with 9-256 rows pad to the 256-row twin, more to the symbolic one, and the two took 5.1 s of
  the 7.1 s first token; each gate/up matmul decoded its NVFP4 weight inline and got no tensor-core tile (434 us).
  Four cuts per twin (the root's split, the gate/up decode `PLACE@place_<tok>=cut`, then the down decode
  `step.1/map.1/inner.2/map` BEFORE the activation `step.2/map.1/inner.1/map` — the other order offers no decode
  seam) give two plain mma matmuls: 482 -> 77 us (256 rows), 742 -> 100 us (symbolic); TTFT at 4K 7.1 -> 2.38 s,
  TPOT unchanged, layouts unchanged. Prefix caching (align) splits a 4K prompt into steps of 3,920, ~80 and 1
  tokens on both servers, so the 4096-row chunk twins never ran: the runner now pads a step filling 3/4 of a wide
  chunk into them (post-sym 13 ms/layer at 3,920 vs post4096 4.4), TTFT 4K random tokens 2.61 -> 2.14 s (fork
  0.93), words 2.38 -> 1.97. Measure TTFT on random token ids (`dbg/ttftr.py`): the 13-word prompt hits fewer
  experts. Per 3,920-token step and layer: 121 calls of the 256-row expert twin (mean ~206 rows: the pad costs only
  ~50 ms per prompt), 76 of expert8 (53 us each for <= 8 rows, 194 ms), 29 symbolic, 36 expert1. Host profile
  (`dbg/hostprof.py`, `dbg/gpuby.py` on the torch trace's python_function events): the 1-token tail step is eager
  (the fork's vLLM classes a step holding prompt tokens as prefill, so no FULL decode graph; the fork replays
  piecewise graphs, 75 ms vs Emmy 223); the ~80-token step is the routed dispatch (host waits per layer for the
  expert counts). Expert8: the activation cut alone, tuned, 76 -> 36 us (the four-cut route tuned to ~38 over five
  kernels), TTFT 2.14 -> 2.04 s. Combine: each row's picks sorted by expert first and the outputs reordered
  slot-major (no gather pass, contiguous weighting and adds, bit-identical sum): 4.5 -> 3.0 ms per layer at 3,920
  rows, TTFT 2.04 -> 1.95 s; what is left is near the traffic an exact sequential sum needs (an fp32 bmm would be
  ~3x less but changes the rounding). In-place rows (branch feature/expert-rows-in-place): the launcher gathers the
  rows into per-expert blocks padded to the tier width (symbolic blocks last, then that program's capacity spare: a
  launch clearing its output clears its whole capacity) and lends them to run_each with the weight slices; in-boot
  A/B with a file switch (`abrows.sh`, VM-only patch): all copied 1.96 s, symbolic copied 1.88, all in place 1.84.
  Boot-to-boot drift on this VM is ~3% (TPOT 22.6 -> 23.3 ms on the SAME tree hours apart): A/B inside one boot.
- **Round notes (2026-10-26 to 31).** NVFP4 decode lowering (#1298): the pair-table lookups are computed in CUDA and
  byte runs vectorized, slots 42.0 -> 35.8 us per layer, but in side-by-side boots TPOT stayed 22.67 -> 22.64 ms and
  TTFT at 4K 1.863 -> 1.844 s, so the table gather was not the limiter. Post8's linear_7/8 operand cut is a divide, not
  a lever. Serving profile of post8 (2026-10-27): 20 kernels, 143 us busy per layer, 6.75 of 18.5 ms GPU busy per step
  (the fork's step is 9.7 ms). The four big matmuls sit about 49 us per layer over their read floor (2.4 ms per step),
  and 13 small pieces take 62 us per layer (3 ms per step), against an empty kernel's 0.9 us under graphs. The fork
  reads its quarter of the shared expert with cuBLAS gemv (this note first said the whole of it, at 550-690 GB/s;
  for a quarter that is 130-180 GB/s) and splits the hyper-connection mixers' work across ranks with all-gathers. A
  hand-written 8-row kernel was also about 2x off the read floor, so a new GEMV schedule family is
  not the obvious win. The cold bench ranked post8's matmuls wrongly because a re-run kernel keeps its code in the
  instruction cache, while serving runs about 58 other kernels between; #1299 evicts the code too and the golden was
  re-measured and re-tuned under it: post8 143 -> 117.6 us per layer, GPU busy 19.32 -> 17.78 ms per step, TPOT
  23.45 -> 21.38 ms, TTFT at 4K 2.08 -> 1.98 s. On the merged tree: TPOT 21.3 ms, TTFT at 4K 1.82 s. Before blaming a
  golden for a boot difference, diff the boots' cubin directories: two boots read 3.5% apart across a card swap while
  compiling the same 80 kernels.
- **Next (2026-11-01).** Every number so far is one request at a short or 4K prompt, and the rounds' gains have
  shrunk to about 1 ms per token. So measure the whole gap before more tuning:
  1. Run step 4's gate: the correctness limits, the accuracy sets, two boots giving the same text. **Done
     2026-11-02**, result under step 4.
  2. Run this step's done gate once against the fork: first token at 1K, 8K, 32K and 130K, total output at 4 and 8
     requests, four replicas together, and memory against the 3 GiB per card margin, of which step 2 left the fork
     about 0.6 GiB to spare. **Done 2026-11-03**, result below.
  3. Profile current main again, then choose the round. Candidates, largest first, in figures from the 2026-10-27
     profile: fewer kernels for post8's small pieces (about 3 ms per token); splitting the hyper-connection mixers
     across ranks, which adds an exchange, so its gain is unmeasured; the one-token tail of a prefill, eager at 223 ms
     against the fork's 75, and the ~80-token step's host wait per layer. The done gate adds two facts to the choice:
     prefill is further off than decode (2.0-2.5x against 1.67x), and memory fails the margin. **Approved by the
     owner on 2026-11-03 as the memory fix, then one decode round; both done the same day**, results below.
- **Done gate, first run (2026-11-03).** Main `c497a0cc7`, the step 3 fork image, step 2's production shape on both
  sides. First tokens by `dbg/ttftr.py` (random token ids; the first of four runs primes), the rest by `vllm bench
  serve` (random prompts, greedy, `ignore_eos`, no warm-up request; one priming repeat, then three). Rounds in the
  order fork, Emmy, Emmy, fork; a range is the measured repeats of both rounds of a side. Every point fails:

  | Point | Fork | Emmy | Emmy against the fork |
  | --- | ---: | ---: | ---: |
  | First token, 1K | 0.43-0.47 s | 1.11-1.16 s | 2.5x slower |
  | First token, 8K | 1.38-1.53 s | 3.39-3.42 s | 2.3x slower |
  | First token, 32K | 5.99-6.08 s | 13.1-13.2 s | 2.2x slower |
  | First token, 130K | 26.3-26.5 s | 52.2-52.6 s | 2.0x slower |
  | One request, 1K / 512, per token | 12.6-12.8 ms | 21.1-21.3 ms | 1.67x slower |
  | 4 requests, 4K / 1K, total output | 174-176 tok/s (20.3-20.5 ms) | 110-113 tok/s (30.3-30.9 ms) | 0.64x |
  | 8 requests, 4K / 1K, total output | 278-282 tok/s (25.0-25.4 ms) | 162-165 tok/s (42.0-42.5 ms) | 0.58x |
  | Four replicas, 32 requests through the load balancer | 742-894 tok/s | 459-530 tok/s | 0.68x (medians) |
  | Step 2's worst-case loads with 3 GiB held per card | serves, 0.4-0.6 GiB spare | out of memory | fails |
  | Boot | 419-465 s | 2,091-2,173 s | about 5x |

  - The fork's rows reproduce step 2's baseline; its four-replica row is as noisy as there (775-811 and 808-863 tok/s
    before). Emmy's first-token and bench rounds ran on two boots that agree, so no cubin diff was needed.
  - Memory: with 3 GiB held per card, Emmy served the full-context prompt and then ran out of memory on the four 32K
    prompts, on a 218 MiB allocation in the fork's per-layer-embedding short convolution during prefill, with Emmy
    holding 28.3 GiB per card. With nothing held, Emmy serves every load; three cards peak at 29.6 GiB, and one sits at
    32.2 GiB (0.5 GiB spare) for the whole run. Why one rank holds 2.6 GiB more is not established.
  - Boot: about 36 minutes on every boot, cold or warm. Three greedy resolves of 400-530 s each run every time, and
    the 1.2 MB kernel cache does not shorten them.
  - Emmy's log reports decode-shaped steps of 10-16 tokens that miss the decode twin (width 8) and fall to the
    symbolic path, and post8 at 12-13x its roofline floor (0.31-0.36 ms).
  - For the next run: `vllm bench serve` with `HF_HUB_OFFLINE` needs the tokenizer's snapshot path (the cache has no
    `main` ref), and a boot's time is best read from the log's timestamps, not from when polling started.
- **Memory round (2026-11-03), item 3 part 1.** Measured per rank by PyTorch's allocation history with the Python
  call site of every block, beside each process's device memory, on boots of both sides without the margin running
  step 2's worst-case loads. At its peak (the four 32K prompts) Emmy held 26.5 GiB live per rank against the fork's
  25.5 (its peak is the 64K prefill). Memory outside PyTorch is 1.35 GiB on both sides.

  | Per rank, GiB | Fork | Emmy |
  | --- | ---: | ---: |
  | Routed experts | 17.64 | 16.13 |
  | Dense weights: pre and post, routers, hosted attention and Gated DeltaNet | 2.88 | 3.16 |
  | Token embedding and output head | 0.59 | 1.48 |
  | Emmy's program buffers | | 1.04 |
  | Other (rotary cache, small buffers) | 0.19 | 0.19 |
  | Key/value pool | 2.50 | 2.50 |
  | Prefill transients at the peak | 1.59 | 1.81 |

  - The fork keeps a quarter of the token-embedding table and of the output head per rank; Emmy held the whole
    table on every card (1.19 GiB). Of Emmy's buffers, 0.34 GiB was a dead copy of one layer's experts: the
    fixed-slot programs' arena kept the build-time backing after every layer pointed its program at its own
    tensors. Emmy's dense weights are 0.28 GiB more because every rank holds the shared expert whole (0.44 GiB),
    where the fork holds a quarter (0.11 GiB); both hold the mixers whole (this note first named the mixers). The
    fork's experts are 1.5 GiB more because it keeps packed copies.
  - The 2.6 GiB extra on one card in the done gate's run is PyTorch's cache of freed blocks, which it gives back
    under pressure: live memory is the same on all four ranks at every snapshot, and with the margin held all four
    ranks failed at the same 28.3 GiB.
  - Fix: the table stays in pinned host memory and every row is read in place over the link (`SERVE_EMBED_GPU=0`;
    its gather is the golden's `embed-sym` twin, 7.2 us at one row), and each layer's fixed-slot program is built
    over the layer's own expert tensors. Weights and buffers after load go from 21.99 to 20.51 GiB per rank, the
    peak from 26.55 to 25.07 GiB, below the fork's 25.47. Reading the table over the link costs 7 ms per 4,096-token
    prefill chunk (about 0.4% of a prefill) and 9 us per decode step.
  - Found on the first fixed boot: the table's device view was a tensor on device 0, so ranks 1 to 3 each opened a
    CUDA context on rank 0's card (0.06-0.3 GiB each). The view now sits on the rank's own device.
  - Gate: `margin.sh` with 3 GiB held on every card passes all of step 2's worst-case loads, with no out-of-memory
    and three to nine cache releases per rank. Two boots give the same text, 16 of 16, and the same text as the
    step 4 gate's boots, so the fix changes no numerics.
- **Decode round (2026-11-03), item 3 part 2.** Profile of one decode step at one request after part 1, rank 0, GPU
  time per step by phase in ms (`dbg/phases.py`, kernels sorted by the layer cycle they sit in); the fork's busy time
  is 10.2 ms, Emmy's 17.6 ms:

  | Phase | Fork | Emmy |
  | --- | ---: | ---: |
  | Pre and post: hyper-connection mixing, norms, shared expert | 2.8, of it 1.5 push exchanges | 8.0 (pre8 2.3, post8 5.7) |
  | Routers and their top-k | 0.9 | 1.0 |
  | Routed experts | 1.1 | 1.4 (fixed slots) |
  | Glue between programs: copies, casts, the routed combine and placement | 0.5 | 2.5 |
  | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 3.4 | 3.5 |
  | All-reduces of the layer outputs | 0.7 | 0.7 |

  - Per layer Emmy runs 315 us in 48 kernels, the fork 164 us in 25. Post8 is 74 us over the fork's matching work:
    its four big matmuls are 8-row tiles of 14-18 us each, where the fork reads its quarter of the shared expert
    with GEMVs in 6-10 us and splits the mixer's work across ranks (6 us for its 80 down rows, a 3-5 us push
    all-gather, 7 us for its 640 mixed columns with the second gather inside), and its 13 small pieces take 47 us.
    Pre8 is 29 us over, and its fused mixer costs what post8's cut one does (48 us per site either way). Neither
    has a lever that fits one round: the matmuls sit at the latency floor their tiles reached in the earlier tunes,
    and splitting the mixers puts two exchanges inside pre and post, which makes pre two programs and post three
    and re-records their decode twins.
  - The round took the glue: a hyper-connection layer's routed placement now writes into the next layer's decode
    input (also that layer's post residual) and a hosted Gated DeltaNet layer writes into its post twin's input,
    so 2.7 copies per layer drop out (128 kernels and 0.24 ms of copies per step). Each rank's own GPU work fell
    by 0.2 ms per step, but the all-reduces now wait about as much longer for the slowest rank. A/B of the two
    boots in A B B A order: one request 22.14 -> 22.03 ms per token (within noise), eight requests 150.0 -> 152.8
    tok/s total output (+1.8%, every pair the same way). Same text as before the round, 16 of 16; GSM8K 126 of 128
    with none at the cap, acceptance cases 27 of 28 in each mode.
  - What is left, largest first: the bytes pre and post read per rank, since the fork reads a quarter of the shared
    expert and of each mixer's weights (planned under "Next (2026-11-03)"); fewer post8 kernels, which needs the
    compiler to fold a split-K finalize into its consumer; and at eight requests the fixed-slot tier runs the
    experts one row at a time. A 1-row tier, first listed here because the fork's GEMVs seemed to read at about
    680 GB/s against the 8-row tiles' 380, has no support: that rate assumed the fork reads the whole shared expert,
    and per byte Emmy's tiles read faster.
  - Done gate's decode points re-measured after both parts, the final tree against a fresh fork boot, rounds fork,
    Emmy, Emmy, fork, three repeats each after a priming one. Both still fail:

    | Point | Fork | Emmy, first run | Emmy now |
    | --- | ---: | ---: | ---: |
    | One request, 1K / 512, per token | 12.66-12.84 ms | 21.1-21.3 ms | 20.99-21.12 ms (1.65x slower) |
    | 8 requests, 4K / 1K, total output | 279.8-282.3 tok/s | 162-165 tok/s | 162.8-164.9 tok/s (0.58x) |
    | Step 2's worst-case loads with 3 GiB held per card | passes | out of memory | passes |

  - For the next run: a server profiled once (`/start_profile`) stays about 5% slower (22.3 against 21.2 ms per
    token on the same tree, and the fork 14.1 against 12.8), so profile on one boot and measure on another; the
    round's A/B used two profiled boots, which keeps its comparison fair.
- **Next (2026-11-03): pre and post across ranks.** Pre and post are the largest decode gap (8.0 against 2.8 ms per
  step at one request), and the fork's lead there is that each rank reads a quarter of the bytes. Its allocations and
  source show how: it slices the shared expert four ways, as vLLM's Qwen3-Next block does (1.6 and 0.8 MB per layer
  per rank, against Emmy's 6.6 and 3.3), and adds that partial output into the routed experts' all-reduce; and it
  keeps the mixers' weights whole on every rank but splits their decode work: each rank computes 80 of the 320 down
  rows and 640 of the 2,560 mixed columns, with a push all-gather after each, 18-23 us per site against Emmy's 48.
  Two parts, each its own pull request, boot and A/B, in this order:
  1. **Slice the shared expert across ranks**, at every width, the way the routed experts are sliced. `post` returns
     the shared expert's partial output instead of placing it in the streams; the runner adds it into the routed
     partial before the all-reduce it already runs, and the placement injects the sum. No new exchange. Expected:
     about 1 ms per decode step at one and at eight requests (post8's two shared-expert matmuls take about 37 us per
     layer whole), 0.33 GiB less per rank, and a slightly faster prefill. The post twins are re-recorded at every
     width; pre is untouched. The sum order changes to the fork's (four partial sums, then the all-reduce), so the
     text may move at near-ties: two boots must agree, and the agreement with the fork and the accuracy sets are
     checked as in step 4.
  2. **Split the mixers' decode work across ranks**, in the decode twins only; prefill keeps the whole weights, as
     the fork does. Pre becomes two programs (the norm and this rank's 80 down rows; then this rank's 640 columns of
     the up projection and the mix) and post three (the attention site's injection, then the same two for the MLP
     site with its injection weights; then part 1's shared expert). Each of the four exchanges per layer is the group
     all-reduce over a zeroed buffer whose own slice the program writes: exact, as a gather is, and captured as the
     routed reduce is. The down rows are a view of the whole weight; the up projection's rank slice is a copy
     (0.15 GiB per rank). Expected: a site from 48 to about 30 us if an exchange costs what Emmy's all-reduces cost
     now (6-7 us with the wait for the slowest rank), about 1.7 ms per step. **Gate before the loader, runner and
     golden work:** trace the split decode twins, bench them cold on the card, add the measured exchange cost, and go
     on only if a site comes out at least 10 us faster (about 1 ms per step); otherwise stop and report.
  - Both together would take one request from about 21 to about 18.5 ms per token (the fork 12.7) and give about 5%
    more total output at eight requests. After each part: in-boot or side-by-side A/B on boots never profiled, the
    margin test, two boots with the same text, GSM8K and the acceptance cases; one profiled boot for the phase table.
  - Risks: four more barriers per layer, each waiting for the slowest rank (the fork's exchanges take 1.5 ms of its
    step); moving pre8's and post8's weights to other programs can re-settle the wide twins' layouts, so check the
    audit's layout lines, re-record the wide twins if they move, and re-measure first tokens then.
  - What it leaves: post8's 13 small pieces (about 2.3 ms per step), and at eight requests the fixed-slot tier's one
    row at a time, which no profile at eight requests has measured yet.
- **Part 1, the shared expert across ranks (2026-11-04).** Qwen4Exp's `post` returns the gated shared expert's output
  apart, each rank holds a quarter of its intermediate axis (160 of 640), and the runner adds it to the routed partial
  ahead of the all-reduce it already ran. The three post twins trace a new program, so the golden's post routes were
  moved onto the new kernels where a piece's content did not change (a one-off migration matching each seam by the
  piece it cuts off), and the shared expert's new pieces were tuned and recorded cold on the V100 SXM3. The release
  audit deploys all 13 twins from the golden alone.
  - Isolated cold totals, old to new: post8 184.3 to 165.9 us, post4096 4,504 to 3,986 us, post-sym at 512 tokens
    1,369 to 907 us. The wide twins' down projection needed tensor-core rows the tune's prior did not offer (14.1 ms
    to 125 us at 4096 tokens), named with `--ab` from the card's offered schedules.
  - A B B A on two boots never profiled, main against the branch: one request 1K / 512, 21.17 to 20.61 ms per token
    (every pair the same way); eight requests 4K / 1K, 162.2 to 166.8 tok/s total output, 42.6 to 41.3 ms per token;
    first tokens unchanged. The plan expected about 1 ms per step at one request.
  - One profiled boot, GPU time per decode step at one request (the 2026-11-03 profile, then this one): busy 17.5 to
    16.6 ms; post8 5.66 to 4.86 ms; exchanges 0.88 to 0.64 ms; elementwise 2.41 to 2.59 ms. The rise is the shared
    add ahead of the all-reduce, one launch per layer (0.17 ms); folding it into the routed combine's matmul would
    take it back, at one more rounding change.
  - Weights 0.33 GiB less per rank (from the shapes). The margin test passes: with 3 GiB held per card every
    worst-case load of step 2 completes and the server keeps serving.
  - The sum order changed as planned: two branch boots give the same 16 of 16 free texts, and the branch keeps 7 of
    16 of the 2026-11-03 boots; where it differs from main, both had already left the fork at the same token. GSM8K
    124 of 128 (2 at the output limit; 126 before, the reference 125), acceptance cases 27 to 28 of 28 in each mode,
    long-document codes 387 of 387 and 383 of 384 (the reference).
- **Part 2, the mixers' decode work split (2026-11-05): stopped at its gate.** One rank's split pre was traced in the
  deployed pre8's cuts (the norm statistics, the rank's 80 down rows, its 2,560 up rows, its 640 mixed columns) and
  benched cold on the V100 SXM3: 39.0 us, plus two exchanges at 6.65 us each (their serving average, the wait for the
  slowest rank included), 52.3 against pre8's 62.5 us. That is 10.2 us faster cold, at the bar; scaled piece by piece
  to the serving times of the 2026-11-04 profile (pre8 48.6 us there), about 6.7 us, below it. The loader, runner and
  golden work did not start. The weak piece was the down rows: 80 rows at about 13.5 us in serving terms against the
  fork's 6, under schedules that read each weight row once per token.
- **Pre8's down rows split across blocks (2026-11-05).** Post8's down projection already ran as a cross-CTA split
  with tensor-core tiles over the eight tokens (17.2 us plus a 2.8 us finalize in that profile), while pre8's split
  had never been measured and its down rows ran unsplit at 28.9 us. The golden now records the split for pre8: the
  10,240-long sum in 64 slices, a tensor-core partial per slice (19.5 us cold) and a finalize (8.2 us), against 37.9 us
  unsplit. Pre8 goes from 68.6 to 56.3 us cold, a strict compile from the golden alone picks it, and the release audit
  deploys all 13 twins. No compiler change.
  - A B B A on two boots never profiled, main against the branch: one request 1K / 512, 20.86 to 20.07 ms per token
    (every pair the same way); eight requests 4K / 1K, 165.6 to 168.6 tok/s total output, 41.7 to 40.9 ms per token;
    first tokens unchanged.
  - The sum order changed: 4 of 6 free texts match main, the other two part after about 370 characters. GSM8K 124
    of 128 (none at the output limit), acceptance cases 27 to 28 of 28 in each mode, long-document codes 387 of 387
    and 384 of 384.
  - Not run: the margin test (no weight moved; the split's state buffer is under 1 MB), a second branch boot (the
    finalize sums the 64 states in a fixed order), a profiled boot.
- **Measurement round (2026-11-07): first token and eight requests.** Main `cc38c02d2` against the step 3 fork image,
  step 2's shape on both sides. One profiled boot per side gives the phase splits: rank 0's GPU time per phase, where
  an all-reduce counts only the fastest rank's kernel per call and the rest of rank 0's time in it is waiting. One
  boot per side that was never profiled gives the numbers: first token at 4K random token ids 1.84-1.86 s against
  the fork's 0.85-0.87 s; eight requests of 4K / 1K 166.7-168.4 tok/s against 269.5-278.9 tok/s (median gap between
  tokens 36.2-36.5 against 23.1-24.0 ms). Emmy matches the #1326 A/B, so main serves as that branch did.
  - First token at 4K, summed over its three steps (3,920, 175 and 1 tokens), ms:

    | Phase | Fork | Emmy |
    | --- | ---: | ---: |
    | Routed experts, with their row gather, weighting and combine | 171 | 831 |
    | Pre and post: hyper-connection mixers, norms, shared expert | 156 | 297 |
    | Glue between programs: copies, casts, the routed placement, eager norms | 30 | 147 |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 207 | 195 |
    | Router and its top-k | 14 | 15 |
    | All-reduces | 33 | 34 |
    | Token embedding and the n-gram embedding | 1 | 12 |
    | GPU work | 612 | 1,530 |
    | Host time, idle, and waiting for other ranks | about 245 | about 320 |

  - Per step, from first tokens at 2, 176, 3,921 and 4,096 tokens: the 3,920-token step takes about 1.34 s on Emmy
    and 0.64 s on the fork, and is GPU work on both. The 175-token step (0.36 against 0.17 s) and the 1-token step
    (0.12 against 0.03 s) are host time on both: the GPU sits in all-reduces waiting for the slowest rank's host.
  - In the 3,920-token step Emmy launches about 270 expert programs per layer, about 1,000 kernels, most of them at
    the 256-row twin (about 206 real rows), each kernel under one full wave of the card. The fork runs three grouped
    GEMMs and one finalize per layer. Emmy's expert kernels take 549 ms there, and their row gather, weighting and
    combine another 109 ms, against the fork's 147 ms for all of it. The fork's small steps count their mixer,
    shared-expert and router GEMMs under the projections (at most about 25 ms).
  - Eight requests, one decode step, ms:

    | Phase | Fork | Emmy |
    | --- | ---: | ---: |
    | Routed experts | 4.37 | 10.37 |
    | Fixed slots, per row: selector copy, input upload, output copy, combine | in the row above | 5.03 |
    | Pre and post | 5.76 | 6.99 |
    | Router and its top-k | 1.17 | 1.21 |
    | Glue between programs | 1.66 | 2.35 |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 6.83 | 7.32 |
    | All-reduces | 0.70 | 0.97 |
    | Embeddings and sampling | 0.19 | 0.20 |
    | GPU busy | 20.67 | 34.45 |
    | Host and idle: the never-profiled median gap minus GPU busy | 2.4-3.3 | 1.8-2.0 |

  - The fixed-slot tier runs the experts one row at a time, as suspected: per layer it repeats the selector copy, the
    input upload, the program's three kernels and the row's combine eight times, about 40 us per row and 320 us per
    layer against the fork's 91. Its slot kernels cost 1.30 ms per row and step, against 1.41 ms at one request. At
    eight requests the fork no longer splits the mixers' work across ranks, so its pre and post grow from 2.8 to
    5.8 ms, while Emmy's stay at the 8-row twins' cost.
  - The eight-request run: Emmy's 195 s are about 148 s of decode steps and 47 s of steps that carry a prompt chunk;
    the fork's 118-122 s are about 95-98 and 23-24 (a gap above twice the median counts as a prompt step). Decode is two
    thirds of the gap.
  - The largest lever for eight requests: one launch set of the fixed-slot program over all eight rows' slots (80
    expert slices) and one combine, in place of eight row passes; a step of one request keeps the 1-row program.
    Expected 8.5-10 ms per decode step if that launch reads the slices at 400-550 GB/s (the fork reads its about 74
    distinct experts at about 550; Emmy's row pass reads at about 250): total output from 168 to about 204-212 tok/s
    (the fork 270-279).
  - The largest lever for the first token: the prefill experts as one launch per tier over all its experts, with the
    weighting and the sum in the same pass. Removing the per-kernel cost and the 256-row padding alone is worth about
    0.2 s per 4,096-token chunk; the fork's rate would give about 0.65 s. First token at 4K from 1.85 s to 1.2-1.65 s,
    the same per chunk at every length (6-16 s of the 26 s gap at 130K), and 3-9% more total output at eight requests.
  - Proposed next round: the eight-row fixed slots. Its cost is measured exactly (the runner loops over the rows), it
    is 11 of the 13.8 ms GPU gap per decode step at eight requests and helps four requests too, and it is a new width
    of an existing program, not a new kernel shape. Gate it before the runner and golden work, as part 2 did: trace
    the 8-row program, bench it cold on the card, and go on only if the experts and their dispatch come to 8 ms per
    step or less (from 15.4). The prefill experts come after it.
  - Not measured: the rate one launch over 80 slices reaches (the gate measures it); how often eight rows pick the
    same expert in real routing (about 7% of picks if routing were random); the wait for the slowest rank under the
    wider program; four requests; the mixed steps at eight requests (estimated from the gaps, not profiled). The
    fork's 4K first token has three samples: the two runs after the 3,921-token prompts hit the prefix cache.
  - Boot: Emmy took 23 minutes on this main (1,387-1,417 s), the fork 7.
- **Eight rows through the fixed slots at once (2026-11-10).** Main `8dfe595a3`. On experts sliced across ranks the
  runner now also builds the fixed-slot program at the decode bucket's width (the `expert-slots8` twin) and runs a
  decode step of two to eight rows through it once, with one combine; a step of one row keeps the one-row program.
  - Main could not boot this model first. #1365's restamp kept the timings of only 63 of the golden's 139 rows, so
    12 of its 13 programs refused a strict compile, while `emmy golden check` still called the file current. The
    fixed-slot program also no longer compiled at all: a cut piece placed each expert's global scale load above the
    loop that loads the expert. #1377 fixes that placement and re-measures the 76 rows cold on the card (median
    against their old timings 1.00); main with it serves eight requests at 168.1-168.3 tok/s, as on 2026-11-07.
  - Gate: the eight-row program, cut like the one-row program (the gate/up product, then the activation) under the
    one-row program's tuned schedules, took 141 us per layer cold, about 7.4 ms per step with one dispatch per layer
    against the 8 ms bar. Recorded at 89.6 + 8.2 + 45.1 us; a 40-row tune of each matmul piece and a sweep of their
    neighbours found nothing faster. The pieces read the 80 slices at about 410 GB/s cold.
  - A B B A on boots never profiled, main against the branch: eight requests 4K / 1K, 168.1 and 168.3 against 192.3
    and 193.6 tok/s total output (+15%), median gap between tokens 36.2-36.3 against 30.0 ms. One request 1K / 512:
    the first pair read 20.02-20.12 against 20.23-20.74 ms per token, a second pair on other boots 20.20-20.39
    against 20.15-20.29; the boots compiled the same kernels but the three new ones, so one request is unchanged.
    The done bar for this round, 200 tok/s, is not reached.
  - One profiled boot, rank 0, one decode step at eight requests, against the 2026-11-07 profile: the routed experts
    5.93 ms (10.37), their dispatch 0.48 ms (5.03), GPU busy 25.5 ms (34.5), 2,961 kernels (5,361). The experts and
    their dispatch came to 6.4 ms, under the gate's 8. The step lost only 6.2 of the 9 ms because the profiles
    over-read the saving: the profiled busy time fell 8.95 ms, the GPU time per step measured without the profiler
    6.8 ms (the host time note below). Profiled step times flatter a step with fewer kernels, so read gains from
    unprofiled boots.
  - Correctness: two branch boots give the same text one request at a time, 6 of 6, and the same as main. Six
    requests at once give the same text on two branch boots for 5 of 6; batching changes the text against the same
    prompt alone on main too (4 of 6 identical alone and batched, the branch 3 of 6), and whether batched text varies
    between two main boots is not measured. GSM8K 124 of 128 with none at the output limit, acceptance cases 26 to 28
    of 28 in each mode, long-document lookups 339 of 339 and 335 of 336. The margin test passes: with 3 GiB held per
    card every worst-case load of step 2 completes, the busiest card peaking at 32,178 of 32,768 MiB. The release
    audit deploys all 14 twins from the golden alone.
  - Not measured: how often eight rows share an expert in real routing (each row reads its own ten slices); the wait
    for the slowest rank under the wider program (rank 0 waited about 0.2 ms per step in its all-reduces); four
    requests.
- **Host time of a decode step (2026-11-11).** #1378's tree, main with #1377 and the step 3 fork image, on boots never
  profiled. py-spy samples the engine core and all four workers without pausing them; the sampled runs read within
  1% of unsampled ones. The torch profiler cannot measure this: it slows the host's Python 2-3x and Emmy's CUDA graph
  launch call 4x. Per decode step, ms, the range over the four ranks:

  | Per decode step, ms | Fork, 8 requests | Emmy main, 8 | Emmy #1378, 8 | Fork, 1 request | Emmy #1378, 1 |
  | --- | ---: | ---: | ---: | ---: | ---: |
  | Step, mean gap between tokens | 24.0 | 36.5 | 29.7 | 12.9 | 20.2 |
  | GPU time: from the graph launch call to the sampled tokens on the host | 20.2-20.6 | 32.6-33.1 | 26.0-26.3 | 9.8-10.0 | 16.7-17.0 |
  | of it, the graph launch call itself | 0.03-0.09 | 3.6-3.8 | 1.5-2.0 | 0.01 or less | 1.3-1.9 |
  | Host time with nothing queued on the GPU | 3.4-3.8 | 3.4-3.9 | 3.4-3.7 | 3.0-3.1 | 3.2-3.5 |
  | of it, attention metadata (Gated DeltaNet 0.8-1.0) | 1.7-2.0 | 1.7-1.8 | 1.6-1.9 | 1.7-2.0 | 1.7-1.9 |
  | of it, input preparation with the n-gram context | 0.6-0.8 | 0.6-0.9 | 0.5-0.9 | 0.5-0.6 | 0.7-0.8 |
  | of it, the engine core's scheduling and the step's messages | 0.5-0.7 | 0.4-0.9 | 0.5-0.9 | 0.1-0.5 | 0.2-0.7 |

  - The host time with nothing queued on the GPU is 3.0-3.9 ms on every server, and it is the same code: the fork's
    vLLM builds the metadata of the Gated DeltaNet, short-convolution, Mamba and sparse-attention layers, prepares the
    inputs, and the engine core schedules the next step. Emmy adds nothing there. Its one extra host cost is the
    graph launch call, and the GPU works during it: inside Emmy's graph the GPU idles under 0.01 ms per step, even in
    the profiled trace where the call took 8.5 ms. Why Emmy's graph costs about 0.6 us per kernel to launch and the
    fork's almost nothing is not established.
  - Main and #1378 spend the same host time, so the eight-row round did not leave the GPU waiting on the host. The
    unprofiled gap between tokens minus the profiled GPU busy time is not host time: the profiler changes both.
  - So the gap left is GPU work: 26.0-26.3 against 20.2-20.6 ms per step at eight requests, 16.7-17.0 against
    9.8-10.0 at one. The profiled split at eight requests, #1378 against the fork: routed experts and their dispatch
    6.4 against 4.4 ms, pre and post 7.0 against 5.8, glue 2.4 against 1.7, the hosted dense matmuls 4.3 against 3.6.
  - Async scheduling hides the host time behind the previous step's GPU work. Both sides run with it off, as the fork's
    release-notes example for this model does; that example serves MTP to one sequence, and with MTP off the fork's vLLM
    turns it on by default. Measured with it on, boots never profiled; the fork's synchronous rows from the same day,
    Emmy's from the A/B above:

    | Point | Fork | Fork, async | Emmy #1378 | Emmy #1378, async |
    | --- | ---: | ---: | ---: | ---: |
    | One request 1K / 512, per token | 12.8-13.1 ms | 9.61-9.62 ms | 20.3-20.7 ms | 16.68-16.69 ms |
    | 8 requests 4K / 1K, total output | 275-277 tok/s | 315-317 tok/s | 192-194 tok/s | 212-216 tok/s |
    | 8 requests, median gap between tokens | 23.6 ms | 19.9 ms | 30.0 ms | 25.9 ms |
    | 8 requests, mean first token | 3.4-3.5 s | 3.8-4.0 s | 6.7-6.8 s | 7.6-8.1 s |

  - It takes 3.3-4.1 ms from every step on both sides, so it does not close the gap: at eight requests Emmy goes from
    0.70 to 0.68 of the fork's output, at one request from 1.6x to 1.7x its time per token. The first token at eight
    requests gets slower on both. Emmy's text does not change with it: one request at a time 6 of 6 against both #1378
    boots, six at once 4 of 6 (two synchronous boots agree on 5). The fork's does: 4 of 6 one at a time and 3 of 6 six
    at once, against two synchronous boots that agree 6 of 6.
  - Proposed next round: async scheduling in step 2's production shape, on both sides. It is one flag, measured above at
    10-12% more output at eight requests and 18% less time per token at one for Emmy, 14% and 25% for the fork, and it
    would also speed up an earlier release on the plain fork. It is the owner's call, because it moves the done gate's
    fork baseline to the faster rows, and it needs step 2's checks on the fork (accuracy sets, four replicas, the margin
    test, why its text changes) and step 4's on Emmy. The parity rounds do not change: Emmy has no host lever left, and
    the next one stays the prefill experts, then the routed experts at eight requests (80 slices at about 410 GB/s,
    against the fork's about 74 experts at about 550).
  - Not measured: first tokens at 8K, 32K and the full context with async scheduling; four requests.
- **Async scheduling on both sides (2026-11-11).** The owner took the proposal above. Main `488f9f0da` with
  `--no-async-scheduling` dropped from the recipe, against the step 3 fork image in the same shape; every number from
  boots never profiled. On this main every program of the recipe golden compiles strict from the golden alone,
  GPU-free, with no row missing its timings, and the release audit on the card deploys all 14 twins from it.
  - Why the fork's text changed: two causes, neither a defect. One at a time, async does not change the fork's
    numerics: with its warm-up tune off (`VLLM_SM70_NVFP4_TUNE_SMALL_SHAPES=0`) two async boots give the synchronous
    boots' text on 6 of 6 prompts. With the tune on, as the fork serves, the tune picks small-shape kernels by timing
    at every boot, so the text varies by boot: two async boots agree on 3 of 6, and one of them gives the synchronous
    text on all 6. Six requests at once, async lets the steps the requests share vary from run to run, so the batched
    text varies even between two runs on one tune-off boot (3 of 6). At every divergence, by boot or by batch, the
    two candidate tokens sit 0.016-0.11 apart in log-probability, near-ties.
  - Fork checks: GSM8K 124 of 128, none at the output limit; acceptance cases 27 of 28 in all six runs; long-document
    lookups 339 of 339 without thinking and 336 of 336 with it. The margin test passes: with 3 GiB held per card every
    worst-case load completes and the server keeps serving, with 338-398 MiB spare per card (0.4-0.6 GiB
    synchronous). Four replicas: step 2's baseline with async.
  - Emmy checks: two boots give the same text, 6 of 6 one at a time and 6 of 6 six at once, as the earlier async boot
    above; GSM8K 125 of 128 (2 at the output limit); acceptance cases 27, 27, 27 without thinking and 28, 27, 27 with
    it. The margin test passes, 538-540 MiB spare per card. Long-document lookups were not run.
  - Async doubles the first token at the full context, on both sides. Its prefill is preempted twice and recomputed
    from the start: the engine's preemption count grows by 2 per 130K prompt on both sides and by none on any other
    row, and the prefix cache serves none of it again. At 64K and 96K tokens the two modes take the same time (12.4-12.8
    and 19.2-19.9 s on the fork) with no preemption; at 130K the synchronous fork takes 26.1 s and the async one
    47.9-51.1 s. With the key/value pool raised from 2.5 to 3 GiB per card (194,016 to 232,523 tokens) the async fork
    takes 26.1 s with no preemption, so async needs a larger pool than synchronous scheduling for a full-context
    prompt; why is not established. The larger pool would come out of the 3 GiB margin, where the async fork keeps
    only 338-398 MiB. **The owner kept the 2.5 GiB pool (2026-11-13), so this is a known limit:** it slows both sides
    alike, full-context prompts are rare, and Emmy's 83 s first token at 130K is under Relay's 290 s.
  - The step 3 image's baked caches miss under async: the flag is part of vLLM's compile key, so every fork boot
    compiles its graphs again, 421-464 s against 303 s. Before a release the image needs a bake in the async shape.
    Emmy boots in 1,048-1,055 s.
  - The done gate against the async fork. Fresh boots of both sides, rounds fork, Emmy, Emmy, fork; first tokens on
    random token ids (the first of four runs primes), the rest by `vllm bench serve` (one priming run, then three); a
    range is both rounds' repeats. Every point still fails:

    | Point | Fork | Emmy | Emmy against the fork |
    | --- | ---: | ---: | ---: |
    | First token, 1K | 0.43-0.44 s | 1.05-1.10 s | 2.5x slower |
    | First token, 8K | 1.46-1.51 s | 3.28-3.33 s | 2.2x slower |
    | First token, 32K | 5.94-6.01 s | 12.69-12.87 s | 2.1x slower |
    | First token, 130K | 50.9-53.0 s | 93.4-96.4 s | 1.8x slower |
    | One request, 1K / 512, per token | 9.60-9.62 ms | 16.69-16.71 ms | 1.74x slower |
    | 4 requests, 4K / 1K, total output | 201.6-203.3 tok/s (17.2-17.3 ms) | 129.8-131.3 tok/s (25.3-25.5 ms) | 0.64x |
    | 8 requests, 4K / 1K, total output | 311.3-317.3 tok/s (21.3-21.7 ms) | 214.6-215.6 tok/s (29.6-29.7 ms) | 0.68x |
    | Step 2's worst-case loads with 3 GiB held per card | passes | passes | |
    | Boot | 421-464 s | 1,048-1,055 s | 2.4x |

  - Against the synchronous rows: the fork from 12.7-13.1 to 9.6 ms per token at one request and from 275-282 to
    311-317 tok/s at eight; Emmy (#1378) from 20.0-20.7 to 16.7 ms and from 192-194 to 215 tok/s. First tokens up
    to 32K are unchanged on both sides.
  - Method change: every bench row now has its own seeds, and the first-token prompts differ by length. `vllm bench`'s
    random prompts are runs of consecutive token ids from a seeded offset, so with one seed for every row the first
    4K prompts began with the 1K row's tokens and the eight-request row repeated the four-request row's prompts.
  - Next: the prefill experts in one launch per group, the round proposed on 2026-11-07, then the routed experts at
    eight requests.
- **Prefill experts 32 at a time (2026-11-12).** On top of async scheduling. A 3,920-token step ran about 180
  routed experts per layer, each its own launch set of five kernels well under one wave of the card: 512 ms of the
  step on rank 0. On experts sliced across ranks the runner now runs every expert routed more rows than the decode
  bucket through one grouped program, 32 experts per launch, each against blocks of 256 of its rows (the twin
  `expert-group32x256@nvfp4`); the experts with fewer rows keep their tiers.
  - The program takes the per-expert route's four cuts: each launch decodes its 32 experts' NVFP4 weights once
    (78 MB of fp16 per rank, the launch's transient), then runs the gate/up and down products as tensor-core tiles
    over all 32 experts. Gate: 32 experts of 256 rows took 710 us cold under the prior's schedules, against about
    2.7 ms one expert at a time; with the two products tuned (40 rows each: gate/up 406 to 252 us, down 222 to
    210 us) it is 646 us, and that is what the golden records.
  - A B B A on boots never profiled, #1385 against the branch, rounds' repeats 1-3:

    | Point | #1385 | This round |
    | --- | ---: | ---: |
    | First token, 1K | 1.09-1.15 s | 0.91-0.93 s |
    | First token, 4K | 1.82-1.84 s | 1.59-1.63 s |
    | First token, 8K | 3.30-3.33 s | 2.78-2.80 s |
    | First token, 32K | 12.84-12.90 s | 10.58-11.33 s |
    | First token, 130K | 93.9-96.9 s | 82.0-84.9 s |
    | One request, 1K / 512, per token | 16.70 ms | 16.70 ms |
    | 8 requests, 4K / 1K, total output | 214.0, 214.3 tok/s | 225.9, 223.9 tok/s |
    | 8 requests, mean first token | 7.8 s | 6.4-6.6 s |

  - About 0.25 s less per 4,096-token chunk, as the gate estimated. Against the async fork the first token is now
    1.6-2.1x slower (2.0-2.5x before) and eight requests give 0.71x its output (0.68x). The branch's second 32K
    round rose run by run (10.89 to 11.33 s); the cause is not established.
  - The grouped kernels sum each product in the same order as the per-expert ones: first-token log-probabilities at
    2K, 4K, 8K and 32K tokens are identical to #1385's, and so is the text, 6 of 6 one at a time and six at once.
    GSM8K 125 of 128 (2 at the output limit, as #1385), acceptance cases 27, 28, 27 without thinking and 27, 27, 27
    with it, long-document lookups 339 of 339 without thinking (the run with thinking was stopped: the numerics
    are #1385's). The margin test passes with 586 MiB spare per card. The release audit deploys all 15 twins from
    the golden alone. Boot 1,088-1,111 s.
  - Next: the experts at eight requests (80 slices at about 410 GB/s, against the fork's about 74 experts at about
    550), and the experts of up to eight rows in a prefill step, which still run one launch set each.
- **Measurement round (2026-11-13): one, four and eight requests, and a 4K first token, against the async fork.**
  Main `496b915d5` (#1388 and #1394 in) against the step 3 fork image, both in step 2's shape with async scheduling.
  Before any boot every program of the recipe golden compiled strict from the golden alone, GPU-free, with no row
  missing its timings. One boot per side that was never profiled gives the numbers, rounds fork, Emmy, Emmy, fork, a
  fresh seed per run; one profiled boot per side gives the phase splits: rank 0's GPU time per phase.

  | Point | Fork | Emmy | Emmy against the fork |
  | --- | ---: | ---: | --- |
  | First token, 1K | 0.45-0.46 s | 0.96-1.00 s | 2.2x slower |
  | First token, 4K | 0.85-0.87 s | 1.57-1.62 s | 1.9x slower |
  | One request, 1K / 512, per token | 9.59-9.60 ms | 16.69 ms | 1.74x slower |
  | 4 requests, 4K / 1K, total output | 201.7-203.2 tok/s (17.3 ms) | 135.0-136.0 tok/s (25.0-25.2 ms) | 0.67x |
  | 8 requests, 4K / 1K, total output | 315.6-317.9 tok/s (21.5-21.7 ms) | 227.5-228.7 tok/s (28.7-28.8 ms) | 0.72x |

  - Four requests: 135.0-136.0 tok/s in four runs against 129.8-131.3 in #1385's gate, before the prefill round:
    +4%, as expected from #1388, which changed only the steps that carry a prompt. The fork reproduces its
    201.6-203.3.
  - Both sides are GPU-bound at every decode point (py-spy, nonblocking, 250 Hz): per step a worker does 3.3-4.1 ms
    of host work on the fork and 5.2-7.0 ms on Emmy (its graph launch call 1.8-2.3 ms of it) and waits the rest, and
    rank 0's output thread waits on the GPU for 98-99% of the step. So the gap between tokens is GPU time.
  - Decode steps are three quarters of the four-request run's gap (99 of Emmy's 121 s against 69 of the fork's
    81 s) and 56% of the eight-request run's (104 of 144 s against 81 of 104 s); the rest is steps that carry a
    prompt chunk and the wait for first tokens.
  - Decode, GPU time per step by phase, profiled, rank 0, ms; the step is the median gap between tokens of the
    never-profiled boots:

    | Per decode step, ms | Fork, 1 | Emmy, 1 | Fork, 4 | Emmy, 4 | Fork, 8 | Emmy, 8 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | Pre and post: hyper-connection mixers, norms, shared expert | 3.41 | 7.02 | 5.50 | 7.05 | 5.89 | 7.35 |
    | Router and its top-k | 0.80 | 0.97 | 1.16 | 1.20 | 1.19 | 1.25 |
    | Routed experts, with their dispatch | 1.15 | 2.33 | 2.52 | 6.37 | 4.46 | 6.61 |
    | Glue between programs | 0.62 | 1.81 | 1.59 | 2.28 | 1.69 | 2.56 |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 3.39 | 3.66 | 5.90 | 6.43 | 6.98 | 7.63 |
    | All-reduces (of it, the wait at the step's first) | 0.63 (0.12) | 2.11 (1.56) | 1.00 (0.13) | 0.72 (0.06) | 0.69 (0.01) | 0.76 (0.05) |
    | Embeddings and sampling | 0.04 | 0.04 | 0.15 | 0.16 | 0.19 | 0.22 |
    | GPU busy | 10.04 | 17.94 | 17.81 | 24.19 | 21.09 | 26.37 |
    | Kernels | 1,434 | 2,725 | 2,054 | 2,961 | 2,054 | 2,961 |
    | Step, never profiled | 9.6 | 16.7 | 16.9 | 24.4 | 20.0 | 25.5 |

  - The split adds up. Busy time less the wait at the step's first all-reduce is 16.4, 24.1 and 26.3 ms on Emmy
    against its unprofiled 16.7, 24.4 and 25.5; the fork's 9.9, 17.7 and 21.1 against 9.6, 16.9 and 20.0 (its two
    streams overlap, so its summed busy time reads high). The profiled gap per step, 6.5, 6.4 and 5.3 ms, is a little
    under the unprofiled 7.1, 7.5 and 5.5.
  - The 1.56 ms wait at one request is the profiler: on Emmy's profiled boot three ranks wait for the fourth at the
    step's first all-reduce, and the fourth runs every kernel as fast as they do and idles instead. The profiler
    slows the host 2-3x and Emmy's graph launch call 4x, which brings Emmy's host near its GPU time at one request
    only; unprofiled, Emmy's host is a third of the step on every rank.
  - One request, 7.1 ms: pre and post 3.6. Pre8 runs 44 us per layer and post8 102 us in 19 kernels, 13 of them
    pieces of 2-5 us (42 us per layer, 2.0 ms per step); the fork runs 19 us per mixer site, its work split across
    ranks, and 25 us of shared expert. Glue 1.2: about 0.75 ms is the sparse-attention layers' query and key norms
    and rope, about 22 eager kernels per layer (60-100 us) where the fork's compiled code runs three (8-9 us).
    Experts 1.2: the fixed-slot program is 28 us per layer against the fork's 24, and its dispatch around it
    (selector copy, row upload, cast, combine, shared add) about 18 us more. Router and hosted sublayers 0.4.
  - Four requests, 7.5 ms: experts 3.9. The eight-row fixed-slot program costs 115 us per layer for four rows and
    122 for eight, so a four-row step pays for eight; the fork's experts scale with the rows (53 us at four, 93 at
    eight). Pre and post 1.6, glue 0.7, hosted sublayers 0.5.
  - Eight requests, 5.5 ms: experts 2.2 (122 against 93 us per layer), pre and post 1.5, glue 0.9 (the
    sparse-attention layers' eager norms alone are about 1.1 ms), hosted sublayers 0.65 (the same matmuls 5-10% slower
    inside Emmy's step).
  - First token at 4K: prefix caching splits it into steps of 3,920, 175 and 1 tokens on both sides. Step times from
    the never-profiled first tokens at 2, 3,921 and 4,096 tokens:

    | Step of a 4K first token | Fork | Emmy |
    | --- | ---: | ---: |
    | 3,920 tokens (3,921 less 2) | 0.59 s | 0.98 s |
    | 175 tokens (4,096 less 3,921) | 0.18 s | 0.36 s |
    | 1 token and the request (2 tokens) | 0.09 s | 0.27 s |
    | First token at 4K | 0.86 s | 1.60 s |

  - The 3,920-token step, GPU time by phase, profiled, rank 0, ms:

    | Phase | Fork | Emmy |
    | --- | ---: | ---: |
    | Routed experts, with their row gather, weighting and sum | 152 | 458 |
    | of it, experts above eight rows | 130, every expert in three grouped matmuls per layer | 230, nine grouped launches per layer |
    | of it, experts of up to eight rows | (above) | 84, about 150 launch sets per layer |
    | of it, the row gather, sort, weighting and sum | 22 | 142 |
    | Pre and post: hyper-connection mixers, norms, shared expert | 152 | 264 |
    | Glue between programs | 18 | 101 |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 155 | 159 |
    | Router and its top-k | 13 | 13 |
    | All-reduces (of it, waiting for other ranks) | 31 (1) | 40 (10) |
    | Token embedding and the n-gram embedding | 1 | 13 |
    | GPU busy | 521 | 1,049 |
    | Kernels | 3,087 | 20,710 |

  - The grouped launches carry about 38,300 rows per layer in 288 blocks of 256: about half their width is
    padding, since an expert's rows fill whole blocks. Narrower blocks do not pay: each block decodes its expert's
    weights again, and on #1388's gate split (the decode is about 28% of a launch) a 128-row block comes out about
    even.
  - The small steps are host time. Emmy's worker spends about 220 ms of host time per 2-token request, the fork's
    about 2 ms: Emmy's step runs eager, about 120 ms in its pre, post and expert calls (of it 25 ms making buffer
    views and 19 ms output prefixes) and about 74 ms in the hosted Gated DeltaNet and sparse attention, which the
    fork replays in graphs. A 176-token request keeps Emmy's worker busy 444 ms against the fork's 232 ms (host work
    and the waits for the GPU inside the step): 209 ms of it is the 175-token step's routed combine, of which 145 ms
    launches about 105 expert runs per layer one at a time (86 ms in the runtime's per-expert calls). In the
    profiled small steps the GPU works 231 against 78 ms (175 tokens) and 16 against 13 ms (1 token); the rest is
    all-reduces waiting for the slowest rank's host.
  - So the 0.74 s gap at 4K is the 3,920-token step 0.39 s (GPU: experts 306 of its 528 ms profiled gap, pre and
    post 113, glue 83, embeddings 12), the 175-token step 0.18 s and the 1-token step 0.18 s. At 1K the gap is
    0.52 s, the same 0.18 s of it in the 1-token step.
  - The next rounds, largest expected gain in the done gate first:
    1. **Prefill experts, second round** (twins and runner tiers, no compiler). Run the experts of up to eight rows
       in grouped launches like the wider ones (a grouped twin over blocks of eight rows), and the weighting and the
       sum in one pass over the slot-major outputs (a combine twin, the same sum order). Share: the routed experts are
       306 of the 3,920-token step's 528 ms profiled gap, of it 84 ms in the small experts' launch sets and 120 ms in
       the gather, weighting and sum beyond the fork's 22; and the 175-token step's host launches about 105 expert
       runs per layer. Expected: about 0.1 s less per 4K chunk and about 0.1 s less in the 175-token step, so first
       token 1K 0.98 to about 0.88 s, 4K 1.60 to about 1.4, 32K 10.9 to about 10.0, 130K 83 to about 75-80 (its
       prefill runs up to three times under preemption); one request unchanged; four and eight requests 3-5% more
       output, from the prompt steps. Unmeasured: the grouped small-expert program's time (at the eight-row
       fixed-slot program's 410 GB/s, about 10 ms per step against 84); the one-pass combine's (its read floor is
       about 0.25 ms per layer); how many small experts a 1K step has; how much of the host saving the 175-token step
       keeps, since it also waits per layer for the expert counts.
    2. **The fixed-slot program at the step's width** (twins and runner). Four- and two-row versions of the
       eight-row program, picked by the step's rows. Share: 2.9 of the 7.5 ms gap at four requests. Expected: four
       requests from 24.4 to about 21.8 ms per step, 135 to about 150 tok/s (0.67 to about 0.74x); one and eight
       requests unchanged, and the steps of two to four rows in real traffic, which pay for eight rows now, gain.
       Unmeasured: the narrower programs' cold times (expected about half, as the fork's experts scale).
    3. **The sparse-attention layers' norms and rope fused** (runner). Share: about 0.75 ms of the one-request gap
       (11%) and about 1.1 ms at eight requests. Expected: 0.7-1.0 ms less per decode step at every concurrency, one
       request 16.7 to about 16.0 ms, four and eight requests about 4% more output; first tokens about the same.
       Unmeasured: whether the runner can call the fork's compiled path as it is, and whether its rounding matches
       (the text checks decide).
    4. **The small prefill steps on the host** (runner, size not known). Share: 0.36 of the 0.74 s gap at 4K and at
       least 0.18 of the 0.52 s at 1K. Expected: up to 0.18 s less per first token at every length if the 1-token step
       replays a graph (1K 2.2x to about 1.8x), more if the 175-token step does too; steps that mix a prompt chunk
       with decode rows at four and eight requests run eager as well. Caching the runner's buffer views and output
       prefixes (about 45 ms per request) is the cheap part. Unmeasured: how vLLM can give a step holding a prompt
       token a graph (it classes such a step as prefill), and the mixed steps' share.
    5. **Post8's small pieces folded** (compiler: a split-K finalize folded into its consumer). Share: 2.0 ms per
       step at every concurrency, 28% of the one-request gap. Expected: up to 1.5 ms less per decode step, one request
       16.7 to about 15.2 ms, four and eight requests about 6% more output. Unmeasured: how much of the 42 us per
       layer a fold keeps.
    6. **The one-row expert dispatch** (runner). Share: about 0.85 ms of the one-request gap. Expected: about 0.5 ms
       less at one request. Unmeasured: which of its five kernels fold into the program.
  - Not a round: pre8 against the fork's split mixers (1.2 ms at one request) stopped at its gate on 2026-11-05; the
    hosted sublayers' 0.5-0.65 ms at four and eight requests is the same kernels running slower inside Emmy's step,
    with no lever measured.
  - Recommended: round 1. First token is the furthest gate metric (1.6-2.2x), the experts are the largest measured
    share of its GPU gap and carry the 175-token step's host launches, the round moves every first-token length and
    the prompt steps at four and eight requests, and it costs what #1388 cost. Gate it the same way: trace both
    twins, bench them cold on the card, and go on only if the 3,920-token step's expert work falls by at least
    100 ms. For a decode round instead, 2 is the cheapest measured gain (four requests) and 3 the cheapest one that
    reaches every concurrency.
  - Boot: Emmy about 1,085 s, the fork 475 s (its baked caches miss under async, as before).
- **Prefill experts, second round (2026-11-14).** Round 1 of the 2026-11-13 ranking. The experts a prefill step
  routes eight rows or fewer now run 32 at a time too, through a second grouped program over blocks of eight rows (the
  twin `expert-group32x8@nvfp4`), and each row's expert outputs are weighted and summed in one pass in float32
  (`embedding_bag`), in place of a gather, a weighting and ten adds in fp16. The round proposed a combine twin; torch's
  fused weighted gather does the same in one call, 0.92 ms per layer at 3,920 tokens against 1.9 (a sparse product
  took 0.52 ms, but sums in an order cuSPARSE picks).
  - The twin takes the 8-row expert's two cuts, and both large pieces were tuned cold on the card, 40 rows each, to
    tensor-core tiles: 191.5 us per launch of 32 experts, about 6 us per expert against about 35 one at a time. A
    16-way split of the gate/up sum ran slower under the prior's schedules and was not pursued.
  - The gate, profiled, rank 0, the 3,920-token step, main against the branch on the same base: expert programs 315 to
    266 ms (the small experts 83 to 33), the row gather, sort, weighting and sum 139 to 90 ms, so the expert work 454
    to 356 ms: 98 ms, just under the 100 ms bar. The step is 121 ms shorter (1,189 to 1,068 ms) and runs 7,855
    kernels against 18,912; the 175-token step takes 472 ms against 588 (5,380 kernels against 15,163).
  - A B B A on boots never profiled, main `0dfcf3e8c` against the branch; a range is both rounds' repeats:

    | Point | Main | This round |
    | --- | ---: | ---: |
    | First token, 1K | 0.93 s | 0.84 s |
    | First token, 4K | 1.59-1.63 s | 1.43-1.45 s |
    | First token, 8K | 2.80-2.81 s | 2.55 s |
    | First token, 32K | 10.66-10.70 s | 9.73-10.22 s |
    | First token, 130K | 78.8-79.0 s | 72.8-80.0 s |
    | One request, 1K / 512, per token | 16.68 ms | 16.71 ms |
    | 4 requests, 4K / 1K, total output | 136.1-136.2 tok/s | 137.4-138.4 tok/s |
    | 8 requests, 4K / 1K, total output | 227.6-227.9 tok/s | 232.9 tok/s |

  - The 4K first token is 0.17 s sooner: its 3,920-token step by 0.09 s, its 175-token step by 0.05 s and its 1-token
    step by 0.03 s (first tokens at 2, 3,921 and 4,096 tokens). A 176-token prompt from scratch: 0.53 to 0.39 s.
    Four and eight requests gained 1-2%, under the 3-5% expected.
  - At 130K the branch's prefill was preempted three times per prompt against main's two (the engine's counter: 18
    against 12 over six prompts each), which takes back what its faster chunks save. Why the count differs is not
    established. The branch's second 32K round rose run by run (9.94 to 10.43 s), as #1388's did; four more runs after
    the 130K rows took 9.69-10.12 s.
  - Against the async fork (2026-11-13's numbers): first token 1.9x slower at 1K (2.2x before) and 1.7x at 4K (1.9x);
    four requests 0.68x its output (0.67x), eight 0.73x (0.72x). The 4K gap is now 0.58 s: the 3,920-token step
    0.29 s, the 1-token step 0.18 s, the 175-token step 0.11 s.
  - Checks on the branch: GSM8K 124 of 128 (2 at the output limit; 125 before); acceptance cases 28, 28, 27 without
    thinking and 27, 27, 27 with it; long-document lookups 339 of 339 without thinking (the run with thinking was
    stopped for time). The margin test passes with 582-584 MiB spare per card, and the release audit deploys all 16
    twins from the golden alone. The float32 sum rounds differently: the top first token at 2K, 4K, 8K and 32K random
    tokens is main's and its log-probabilities move by up to 0.3, and 3 of 6 greedy texts are main's, the other three
    parting after 650-1,460 characters. Boot 1,116 s, as main's 1,117.
  - Next: from the 2026-11-13 ranking, for the owner to pick. In the 4K first token the 3,920-token step is still the
    largest part of the gap; its 256-row blocks are about half padding, and gathering the rows into them takes 33 ms.
- **The fixed-slot program at the step's width (2026-11-14).** Round 2 of the 2026-11-13 ranking, main `7cfbdf9db`.
  A fixed-slot program reads every slot's expert whatever the step's rows, so a step of two or four rows through the
  eight-row program paid for eight. The program is now built at every slot width, the decode bucket and its halvings
  down to two rows (the twins `expert-slots2@nvfp4` and `expert-slots4@nvfp4` beside `expert-slots8`), and a step
  runs the narrowest that holds it; the widths share the eight-row selector and partials.
  - Gate: both new programs take the eight-row program's two cuts and its schedules, cold on the card, per layer:
    four rows 102 us and two rows 59 us, against 176 us for eight in the same run: 0.58x and 0.34x, the "about half"
    the ranking expected. They were recorded as measured; the eight-row round's tunes found nothing faster for these
    pieces, so none was run.
  - A B B A on boots never profiled, main against the branch:

    | Point | Main | This round |
    | --- | ---: | ---: |
    | One request, 1K / 512, per token | 17.20 ms | 17.18 ms |
    | 2 requests, 4K / 1K, total output | 70.7-70.8 tok/s (25.6 ms) | 84.9-85.1 tok/s (20.8 ms) |
    | 4 requests, 4K / 1K, total output | 128.8-128.9 tok/s (26.3 ms) | 143.4-144.2 tok/s (23.1 ms) |
    | 8 requests, 4K / 1K, total output | 220.7-221.1 tok/s (27.4 ms) | 219.6 tok/s (27.4 ms) |
    | First token, 1K | 0.80 s | 0.80 s |
    | First token, 4K | 1.42 s | 1.42-1.43 s |

  - Two requests gain 20% and four requests 12%, a step 4.8 and 3.2 ms shorter (expected at four: 2.6 ms, 135 to
    about 150 tok/s). One and eight requests and the first tokens are unchanged, as expected.
  - Main itself is slower than on 2026-11-14 morning: four requests 128.9 against 137.4-138.4 tok/s, eight 221
    against 233, one request 17.2 against 16.7 ms. The cause is #1399, the Rust PTX compiler in place of nvcc: the
    same kernels under the same schedules, cold, before and at that commit: the eight-row slot program 128 to 175 us,
    the small-expert group program 193 to 225, the wide group program 629 to 677, the one-row expert programs 36-37
    to 41-42; the other programs within 2%. Not fixed here. The new programs were recorded on the slower code, so
    they gain with the eight-row one when it is fixed.
  - Against the async fork (2026-11-13's 201.7-203.2 tok/s): four requests 0.71x its output, from 0.64x on today's
    main and 0.68x before #1399.
  - Checks on the branch: all 18 programs compile strict from the golden alone and the release audit deploys all 18
    twins. GSM8K 124 of 128 with none at the output limit; acceptance cases 28, 27, 27 without thinking and 27, 28, 27
    with it; long-document lookups 339 of 339 and 336 of 336. The margin test passes with 566-568 MiB spare per card
    (582-584 before: the two programs' instances per layer). Six greedy texts, one request at a time, are main's, 6
    of 6. Boot 698 s, main's 689 (1,116 before #1399).
  - Next: the PTX compiler's slowdown of the gathered expert programs first, since it costs four and eight requests
    more than this round gained at eight; then the 2026-11-13 ranking, for the owner to pick.
- **The slot programs re-measured on the fixed PTX compiler (2026-11-16).** Main `8f544ccf7`, with #1419. The two-,
  four- and eight-row slot programs' rows recorded again cold, same route and schedules, no tune. Per layer, whole
  program cold: two rows 59 to 51 us, four rows 102 to 78, eight rows 176 to 130-131 (128 before #1399).
  - Two boots of main with these rows, A B B A, against 2026-11-14's main and #1417's branch:

    | Point | Main 2026-11-14 | #1417 branch | Now |
    | --- | ---: | ---: | ---: |
    | One request, 1K / 512, per token | 17.20 ms | 17.18 ms | 16.90 ms |
    | 2 requests, 4K / 1K, total output | 70.7 tok/s | 85.0 tok/s | 86.9-87.2 tok/s |
    | 4 requests, 4K / 1K, total output | 128.9 tok/s | 143.8 tok/s | 147.9-149.3 tok/s |
    | 8 requests, 4K / 1K, total output | 221 tok/s | 219.6 tok/s | 231.1-233.5 tok/s |
    | First token, 1K | 0.80 s | 0.80 s | 0.78-0.82 s |
    | First token, 4K | 1.42 s | 1.42-1.43 s | 1.38-1.43 s |

  - Eight requests are back at the 233 tok/s of before #1399. Against the async fork (201.7-203.2 tok/s at four):
    four requests 0.73x its output.
  - Every program cold against the tree before #1399, in one run: all within 2% except program 9 (1,899.5 to 1,955.8
    us, 3.0%) and the 32-expert group program 15 (190.5 to 197.6 us, 3.7%); the eight-row slot program 13 is 1.6%
    above (128.0 to 130.0). Programs 2, 10 and 11 read 5-13% above in a run of 13 at once and equal in the repeat.
  - Checks: all 18 programs compile strict from the golden alone, and the release audit deploys all 18 twins. Boot
    698 and 738 s.
- **The last token of a prompt replays the decode graph (2026-11-17).** Round 4 of the 2026-11-13 ranking, main
  `f1b7171fc`. Prefix caching ends a 4K prompt as steps of 3,920, 175 and 1 tokens, and Model Runner V2 runs any step
  that still holds a prompt token eager. Its refusal guards speculative decoding, which is off here. With no
  speculative decoding, a step whose every row is one token now replays the decode graph. The hosted Gated DeltaNet
  already runs such a row as a decode row, and the sparse attention has no prefill path of its own. The change is
  one patch of the runner in `emmy/serving/`, with no compiler change and no new twin.
  - Where the host time went, on main, profiled: the 1-token step spans 236 ms against about 16 ms of its own GPU
    work. Emmy's runner framing takes about 130 ms of host time (buffer views 38 ms, output prefixes 28 ms, the slot
    program 50 ms). The hosted Gated DeltaNet takes 46 ms and the sparse attention 43 ms. The 175-token step spans
    449 ms, with the GPU busy for 279 ms. Its host time is mostly the routed combine (97 ms), the Gated DeltaNet
    (77 ms) and the sparse attention (47 ms).
  - Gate, stated before the build: the 4K first token falls by at least 0.10 s. It passes, at -0.12 s in the A B B A
    and -0.18 s in the step run. In the step run, a 2-token prompt's first token went from 0.265 to 0.047 s; the
    fork's 1-token step is 0.09 s. A 176-token prompt's went from 0.386 to 0.271 s.
  - A B B A on boots never profiled, two of main against two of the branch:

    | Point | Main | This round |
    | --- | ---: | ---: |
    | One request, 1K / 512, per token | 16.89-16.91 ms | 16.89-16.90 ms |
    | 2 requests, 4K / 1K, total output | 87.1-87.2 tok/s | 87.6-87.7 tok/s |
    | 4 requests, 4K / 1K, total output | 148.9 tok/s | 149.6-149.8 tok/s |
    | 8 requests, 4K / 1K, total output | 232.6-232.9 tok/s | 233.1-234.1 tok/s |
    | First token, 1K | 0.77-0.83 s | 0.67-0.68 s |
    | First token, 4K | 1.38-1.44 s | 1.28-1.30 s |
    | First token, 8K | 2.49-2.51 s | 2.39 s |
    | First token, 32K | 9.71-9.73 s | 9.65 s |

  - Against the async fork (2026-11-13's numbers), the first token is 1.5x slower at 1K (1.7x on main) and 1.5x
    slower at 4K (1.6x). Four and eight requests are 0.74x its output. The 4K gap is now 0.43 s, from the
    3,920-token and 175-token steps, which still run eager.
  - Checks on the branch: all 18 programs compile strict from the golden alone, and the release audit deploys all 18
    twins. GSM8K 126 of 128 (2 at the output limit). Acceptance cases: 27, 27, 27 without thinking and 26, 27, 27
    with it. Long-document lookups: 387 of 387 and 384 of 384. Six greedy texts are identical across the four
    engines. The margin test passes on cards 12-15 with 750-752 MiB spare per card, the same as main on those cards.
    A first run on cards 4-7 read 358 MiB, so the card set moves the number. Boot 719 and 728 s, main's 709 and
    720 s.
  - What it costs: 31 lines that patch a private method of the fork's runner. Drop the patch when the runner itself
    counts such a step as a decode.
  - Next: the 175-token step's host time (routed combine, Gated DeltaNet, sparse attention) and the 3,920-token step,
    from the 2026-11-13 ranking, for the owner to pick.
- **The short prefill step's pre and post programs (2026-11-17).** Main `8809bfc9e`. A 4K prompt ends with a
  175-token step and a 1K prompt with a 239-token one. On main such a step took 0.238 s against the fork's 0.181, and
  its GPU work was about 200 ms against the fork's 71. The symbolic pre and post programs took 54 ms of it against the
  fork's 4: one pre kernel ran 499 us per layer on 12 blocks. The round re-tunes their rows on the card and makes the
  expert launcher's index uploads non-blocking. It changes no compiler code and adds no twin.
  - Rows: each piece tuned on the card with `--tune 40` on a fresh scratch DB, then recorded with
    `--record-greedy`. The slow pre kernel takes a cross-CTA split (`g8k`), with its two pieces recorded under the
    route. Six rows are new or faster at 512 tokens; the sixteen rows of unchanged kernels keep their old numbers,
    so no row was re-recorded slower. Pre plus post at 512 tokens, cold: 1,788 to 1,053 us per layer.
  - Uploads: the launcher sent its counts, offsets and picks with a blocking copy, 0.8-1.6 ms each on a busy card.
    A pinned non-blocking copy takes about 70 us.
  - Gates, stated before the build: pre plus post cold at 512 tokens -700 us per layer, passed at -735. The 1K
    first token -50 ms, passed at -62 in the step run and -55 in the A B B A. The 4K first token -25 ms, **missed**:
    -21 ms in the step run (5 pairs), with the 175-token step only -12 ms. The A B B A reads -33 ms.
  - Why 4K moves less: after the tune the short step is host-bound. Rank 0's host time per 176-token prompt is the
    routed experts about 74 ms (launch 44, combine 30), the pre and post program calls about 43 ms (run 13, output
    clones 8, output views 5, symbolic binds 4), the hosted Gated DeltaNet about 49 ms and attention about 22 ms.
    Profiled GPU time of pre plus post per step: 175 tokens 50 to 26 ms, 239 tokens 54.5 to 30.9 ms, 784 tokens
    97.5 to 73.2 ms.
  - A B B A on boots never profiled, two of main against two of the branch:

    | Point | Main | This round |
    | --- | ---: | ---: |
    | One request, 1K / 512, per token | 16.88-16.89 ms | 16.89-16.90 ms |
    | 2 requests, 4K / 1K, total output | 87.6 tok/s | 87.4-87.6 tok/s |
    | 4 requests, 4K / 1K, total output | 149.3 tok/s | 150.1-150.3 tok/s |
    | 8 requests, 4K / 1K, total output | 233.5-234.0 tok/s | 232.6-234.0 tok/s |
    | First token, 1K | 0.670-0.674 s | 0.608-0.625 s |
    | First token, 4K | 1.291-1.297 s | 1.254-1.268 s |
    | First token, 8K | 2.391-2.397 s | 2.333-2.355 s |
    | First token, 32K | 9.62-9.67 s | 9.44-9.53 s |

  - Against the async fork (2026-11-13's numbers), the first token is 1.45x slower at 1K (1.6x on main) and 1.5x at
    4K (1.6x).
  - Checks on the branch: all 18 programs compile strict from the golden alone, and the release audit deploys all 18
    twins. GSM8K 125 of 128, none at the output limit. Acceptance cases 27 of 28 in all six runs, with and without
    thinking. Long-document lookups 387 of 387 and 384 of 384. Greedy texts: the two main boots agree with each
    other and the two branch boots agree with each other; main and the branch part on three of six prompts after
    783-1,220 characters, from the new sum order. The margin test on cards 12-15 leaves 360-366 MiB spare per card
    on both sides. Boot 760-761 s, main's 759-761 s.
  - Next: the short step's host time, in the order above: the routed experts' launch and combine, then the pre and
    post program calls.
- **The sparse-attention layers' query and key norms and rope as an Emmy twin (2026-11-18).** Round 3 of the
  2026-11-13 ranking, main `ad56f5d38`. Each of the 12 sparse-attention layers splits its query, key and gate,
  RMS-norms the query and the key and applies the rope. On main the fork runs this eager: 25 kernels per layer on a
  decode step. The fork's compiled model runs it as 3. The round adds one symbolic twin, `qk-norm-rope-sym`, at a
  rank's heads, lent the layer's two norm weights and the first 262,144 rows of the rotary cache. A step of up to
  the decode bucket's rows runs it; a wider step keeps the fork's path. No compiler change.
  - Measured first, profiled, rank 0, per decode step at one, four and eight requests: main 300 kernels and a span of
    1.16, 1.26 and 1.26 ms (96-105 us per layer); the async fork 36 kernels and 0.15 ms (12 us per layer). The
    decode step is a graph replay, so the work costs no host time per kernel.
  - Rows: the twin tuned on the card on a fresh scratch DB, then recorded with `--record-greedy`. The prior's pick
    cuts it into three kernels, 18.4 us per layer cold at 8 rows; the fused root runs on one block (about 450 us).
    The seed root row was dropped. On the branch, profiled: 60 kernels per step and a span of 0.17-0.18 ms (15 us
    per layer, with the two input uploads).
  - Gates, stated before the build: the twin cold at most 25 us per layer at 8 rows, passed at 18.4. One request's
    step at least 0.5 ms shorter on boots never profiled, passed at -0.84 ms.
  - A B B A on boots never profiled, two of main against two of the branch:

    | Point | Main | This round |
    | --- | ---: | ---: |
    | One request, 1K / 512, per token | 16.90-16.91 ms | 16.05-16.07 ms |
    | 2 requests, 4K / 1K, total output | 87.4-87.7 tok/s | 91.0-91.3 tok/s |
    | 4 requests, 4K / 1K, total output | 148.4-149.6 tok/s | 153.9-155.3 tok/s |
    | 8 requests, 4K / 1K, total output | 233.0-234.1 tok/s | 239.5-241.4 tok/s |
    | First token, 1K | 0.60-0.65 s | 0.62-0.63 s |
    | First token, 4K | 1.27-1.29 s | 1.26-1.29 s |

  - Against the async fork (2026-11-13's numbers): one request 1.67x slower per token (1.76x on main); four and
    eight requests 0.76x its output (0.74x on main).
  - Checks on the branch: all 19 programs compile strict from the golden alone, and the release audit deploys all 19
    twins. `emmy golden check` reads current, and the prior reproduction gate passes. GSM8K 124 of 128 (2 at the
    output limit). Acceptance cases: 27, 27, 27 of 28 without thinking and 27, 26, 27 with it. Long-document lookups:
    387 of 387 and 384 of 384. Greedy texts: the two main boots agree and the two branch boots agree; main and the
    branch part on two of six prompts, after 318 and 1,118 characters, from the twin's rounding. Boot 730 and 739 s,
    main's 721 and 740 s.
  - Margin test: it passes on both sides. On cards 12-15 the branch left 370-372 MiB spare per card in three runs
    and main 750-754 in two. That gap is not memory the change holds. In one run with the memory probe on both sides
    at once (main on cards 8-11), torch's allocated peak is 25,510 MiB on both, its reserved peak 27,598-27,604 MiB,
    and the memory outside torch about 4,795 MiB; main left 362-364 MiB spare there. The number moves with when the
    allocator reaches the cap and frees its cache, which the #1428 and #1435 rounds also saw.
  - Next: from the 2026-11-13 ranking, for the owner to pick.
- **The one-row expert dispatch (2026-11-18): stopped at its gate.** Round 6 of the 2026-11-13 ranking, main
  `8256c51ce`. Measured first, on #1437's profiled boot (main's code) against the async fork's, rank 0, per decode
  step at one request, from the router's top-k to the next all-reduce, 48 layers:

  | Per layer, profiled | Fork | Emmy |
  | --- | ---: | ---: |
  | Routed expert kernels | 2 kernels, 23.7 us (the down projection weights and sums the picks) | 3 kernels, 29.4 us |
  | Dispatch around them | none | 5 kernels, 17.0 us |
  | Top-k to all-reduce | 28.8 us | 50.3 us |

  - The five kernels: the selector copy (int32 to int64, 3.9 us), the row upload (2.0), the scores' cast to fp16 and
    the shared expert's add (6.9 together), and the weighted sum (4.2). Per step 0.82 ms busy, and the span from top-k
    to all-reduce is 1.03 ms longer than the fork's. The fork adds its shared expert inside the all-reduce.
  - The same five kernels run at four and eight requests: 15.9 and 16.4 us per layer, 0.76 and 0.79 ms per step. The
    ranking's "nothing at two to eight requests" was wrong for the dispatch; it holds for the program swap below.
  - The two-row slot program with one row filled does not pay: cold per layer 54.3 us (28.7, 8.2, 17.4) against the
    one-row program's 36.9 (18.4, 6.1, 12.3), with the same dispatch around it.
  - Gate, stated before the build: one request's step at least 0.3 ms shorter on boots never profiled. With the row
    upload and the shared expert's output lent to the program without a copy and the scores uploaded, a program that
    weights, sums and adds the shared expert removes at most 9 us of dispatch per layer, so its cold time may grow by
    at most about 3 us: 39.8 us per layer.
  - Tried: the one-row twin traced with the router's scores and the shared expert's output as inputs, its output the
    layer's routed sum, cold on the card, the prior's cuts and about forty hand-pinned cut and schedule variants. Best
    59.4 us per layer (gate and up 17.4, the down projection fused with the weighted sum 35.8, the activation 6.1).
    The fused down-and-sum runs 34-36 us under every schedule tried, cross-CTA splits included, where the program's
    own down projection takes 12.3 and the fork's down-and-sum 9.2. Kept as four pieces (the program's three and a
    sum-and-add piece, 6.1 us measured) it would be about 42.9 us. Both miss the 39.8 us bar, so nothing was built.
  - Without a program change, the runner could lend the row upload (2.0 us) and fold the shared add into the
    weighted sum's matmul (3.6): at most 5.6 us per layer, 0.27 ms per step profiled, at every width up to eight
    rows, and less unprofiled, since the profiler adds about 1.6 us to each small kernel. Under the bar.
  - Where the gap is: Emmy's gathered down-and-sum kernel, 3-4x the fork's. A schedule that splits the picks across
    blocks and sums them after would be compiler work (the schedule space), like post8's small pieces.
  - Next: from the 2026-11-13 ranking, for the owner to pick; the down-and-sum schedule could join the post8 round.
- **Measurement round (2026-11-19): every serving point against the async fork, and a new ranking.** Main
  `5096d52a4` (every round since 2026-11-13 in, through #1441) against the step 3 fork image, both in step 2's shape
  with async scheduling. All 19 programs of the recipe golden compile strict from the golden alone, GPU-free. Two
  boots a side never profiled give the numbers, fork, Emmy, Emmy, fork per point, one server working at a time, a
  fresh seed per run; one profiled boot a side gives the phase splits (rank 0's GPU time), and py-spy gives host time.

  | Point | Fork | Emmy | Emmy against the fork | On 2026-11-13 |
  | --- | ---: | ---: | --- | --- |
  | One request, 1K / 512, per token | 9.59-9.60 ms | 16.05 ms | 1.67x slower | 1.74x |
  | 2 requests, 4K / 1K, total output | 110.6 tok/s (16.5 ms) | 91.2-91.3 tok/s (19.5 ms) | 0.82x | not measured |
  | 4 requests, 4K / 1K, total output | 203.2-203.5 tok/s (16.9 ms) | 154.6-155.5 tok/s (21.3 ms) | 0.76x | 0.67x |
  | 8 requests, 4K / 1K, total output | 316.4-320.3 tok/s (19.8-20.0 ms) | 238.9-240.7 tok/s (24.8 ms) | 0.75x | 0.72x |
  | First token, 1K | 0.43 s | 0.61 s | 1.41x slower | 2.2x |
  | First token, 4K | 0.82-0.83 s | 1.25 s | 1.51x slower | 1.9x |
  | First token, 8K | 1.46-1.47 s | 2.32-2.33 s | 1.59x slower | not measured |
  | First token, 32K | 5.93-5.94 s | 9.44-9.49 s | 1.59x slower | not measured |

  - In brackets: the median gap between tokens. Each side's two boots agree within 1.3%. Emmy repeats #1437's main at
    every point. The fork repeats 2026-11-13 at one, four and eight requests. Its first tokens are 4-5% faster than
    that day's (1K 0.45, 4K 0.86 s) and equal to its quiet runs of 2026-11-11 and 2026-11-18 (0.42-0.44 and 0.82 s).
    The fork's short prefill steps wait on its host (profiled, three ranks wait 230-270 ms at the 175-token step's
    all-reduces), so its first token moves with the host's load: 0.51 s at 1K on 2026-11-18, with four servers
    working at once. Every ratio here is against today's fork.
  - Correction to the 2026-11-13 step table. A prompt runs as whole cache blocks, then the rest less one token, then
    one token, so a 2-token prompt is two 1-token steps. Before #1428 both ran eager, about 0.12 s each on Emmy, and
    "3,921 less 2" took one of them too many off the 3,920-token step: that step was about 1.10 s, not 0.98, and
    "1 token and the request" about 0.15 s, not 0.27. The profiled spans agree: 1,186 ms then, 1,072 now. This round
    subtracts a 1-token prompt instead (fork 0.045-0.050 s, Emmy 0.029-0.031).

    | Step of a first token | Fork | Emmy | Emmy, 2026-11-13 |
    | --- | ---: | ---: | ---: |
    | 3,920 tokens of 4K (3,921 less 1) | 0.59 s | 0.99 s | about 1.10 s |
    | 175 tokens of 4K (4,096 less 3,921) | 0.18-0.19 s | 0.22-0.23 s | 0.36 s |
    | 784 tokens of 1K (785 less 1) | 0.22 s | 0.36 s | |
    | 239 tokens of 1K (1,024 less 785) | 0.21 s | 0.25 s | |
    | 1 token and the request (a 1-token prompt) | 0.05 s | 0.03 s | about 0.15 s |

    The 1K split comes from the profiled boots outside a capture, which read 5-9% above the others at 1K.
  - Decode, GPU time per step by phase, profiled, rank 0, ms; Emmy's 2026-11-13 value in brackets; the fork moved by
    0.25 ms or less in every row:

    | Per decode step, ms | Fork, 1 | Emmy, 1 | Fork, 4 | Emmy, 4 | Fork, 8 | Emmy, 8 |
    | --- | ---: | ---: | ---: | ---: | ---: | ---: |
    | Pre and post: hyper-connection mixers, norms, shared expert | 3.16 | 7.23 (7.02) | 5.47 | 7.35 (7.05) | 5.91 | 7.24 (7.35) |
    | Router and its top-k | 0.78 | 0.97 (0.97) | 1.15 | 1.21 (1.20) | 1.19 | 1.20 (1.25) |
    | Routed experts, with their dispatch | 1.14 | 2.12 (2.33) | 2.54 | 3.90 (6.37) | 4.23 | 6.27 (6.61) |
    | Glue between programs | 0.61 | 1.16 (1.81) | 1.60 | 1.55 (2.28) | 1.69 | 1.58 (2.56) |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 3.33 | 3.68 (3.66) | 5.90 | 6.51 (6.43) | 7.02 | 7.34 (7.63) |
    | All-reduces (of it, the wait at the step's first) | 0.50 (0.01) | 2.09 (1.50) | 0.69 (0.01) | 0.64 (0.01) | 0.65 (0.01) | 0.80 (0.01) |
    | Embeddings and sampling | 0.04 | 0.04 | 0.15 | 0.16 | 0.19 | 0.20 |
    | GPU busy | 9.55 | 17.29 (17.94) | 17.49 | 21.31 (24.19) | 20.87 | 24.63 (26.37) |
    | Kernels | 1,434 | 2,485 (2,725) | 2,054 | 2,721 (2,961) | 2,054 | 2,721 (2,961) |
    | Step, never profiled | 9.6 | 16.05 (16.7) | 16.9 | 21.3 (24.4) | 19.9 | 24.8 (25.5) |

  - Both sides are still GPU-bound at every decode point (py-spy): per step a worker does 3.4-4.0 ms of host work on
    the fork and 5.2-6.3 ms on Emmy (its graph launch call 1.5-2.6 ms), and rank 0's output thread waits on the GPU
    for 99% of the step. The 1.5 ms wait at one request is the profiler again (three ranks wait for the fourth).
  - What moved since 2026-11-13: the routed experts at four requests (6.37 to 3.90 ms, the slot programs at the
    step's width) and the glue at every width (0.6-1.0 ms less, the norms and rope twin). What did not: pre and post,
    7.2-7.35 ms, now the largest decode gap: 4.07 ms of one request's 6.45, 1.88 of four requests' 4.4 and 1.33 of
    eight requests' 5.5 (profiled span gap at eight). Post8 is 101 us per layer in 18 kernels, as on 2026-11-13; 13 of
    them are pieces of 2.2-5.5 us, 43.7 us per layer, 2.1 ms per step at every width.
  - The routed experts per layer, profiled, at one, four and eight requests: the fork's gate and up 14.5, 35.4 and
    58.4 us, its down projection with the weighted sum 9.2, 17.5 and 29.6; Emmy's gate and up 17.0, 41.5 and 73.2,
    its down projection 9.9, 24.2 and 41.6, and its dispatch around the program (selector copy, row upload, cast and
    shared add, weighted sum) 16.6, 16.3 and 16.5. From the router's top-k to the next all-reduce: fork 28.7, 57.8
    and 93.1 us, Emmy 49.5, 88.8 and 138.0.
  - The 3,920-token step, GPU time by phase, profiled, rank 0, ms; Emmy's 2026-11-13 value in brackets:

    | Phase | Fork | Emmy |
    | --- | ---: | ---: |
    | Routed experts, with their row gather, weighting and sum | 152 | 361 (458) |
    | of it, experts above eight rows | 130 on 2026-11-13, every expert in three grouped matmuls per layer | 234 (230), 8.75 launches of 32 per layer |
    | of it, experts of up to eight rows | (above) | 33 (84), 12 launches of 32 per layer |
    | of it, the row gather, sort, weighting and sum | 22 on 2026-11-13 | 94 (142): weighted sum 42, row gather 33, sorts 9 |
    | Pre and post: hyper-connection mixers, norms, shared expert | 152 | 266 (264): post4096 180, pre4096 86 |
    | Glue between programs | 18 | 101 (101): copies 43, adds 17 |
    | Hosted sublayers: projections, Gated DeltaNet, sparse attention | 154 | 170 (169) |
    | Router, all-reduces, embeddings | 45 | 46 (56) |
    | GPU busy | 521 | 944 (1,049) |
    | Kernels | 3,087 | 9,643 (20,710) |
    | Span | 614 | 1,072 (1,186) |

  - The steps of a 1K prompt, GPU time, profiled, rank 0, ms. The fork's dense matmuls are split into mixers and
    projections by their grids, which hold only at 3,920 tokens, so this table adds the two:

    | Phase | Fork, 784 | Emmy, 784 | Fork, 239 | Emmy, 239 |
    | --- | ---: | ---: | ---: | ---: |
    | Routed experts, with their row gather, weighting and sum | 50 | 155 | 27 | 99 |
    | Pre and post, and the hosted projections | 60 | 96 | 27 | 42 |
    | Glue between programs | 6 | 23 | 4 | 11 |
    | Hosted Gated DeltaNet and sparse attention | 17 | 18 | 13 | 14 |
    | GPU busy, less the all-reduces | 134 | 295 | 72 | 167 |

  - The wide group program costs the same per launch whatever the step: 0.56 ms per 32 experts at 3,920 tokens and
    0.57 at 175, about 17.5 us per expert whatever its rows; the 8-row group program costs about 5.5 us per expert.
    At 3,920 tokens the wide experts hold about 136 rows each in blocks of 256; at 175 and 239 tokens they hold 9 to
    about 30, and the wide program still takes 43-50 ms of the step, against the fork's 25-27 ms for all its experts.
    2026-11-13's "narrower blocks do not pay" held for splitting a large expert across more blocks, each decoding its
    weights again; it does not hold for an expert whose rows fit a narrower block.
  - The short prefill steps' host time, py-spy, rank 0, per 176-token prompt: Emmy's routed experts 51 ms (launch 30,
    route 21), its pre and post calls 27, the hosted Gated DeltaNet 37 and the sparse attention 17, and 46 ms waiting
    for the GPU's expert counts; the fork's model forward 60, Gated DeltaNet 26 and attention 16. The 175-token step
    is 0.04 s slower than the fork's, so a host round can gain at most that much there.
  - Sizes of the candidates, profiled GPU time unless stated, and what each would close if Emmy matched the fork:
    - Post8's small pieces (compiler): 2.1 ms per decode step at every width; nothing in the first token but the
      1-token step's 2 ms. Folded into their consumers they would close a third to a half of each decode gap (6.45 ms at
      one request, 4.4 at four, 5.5 at eight), less unprofiled: the norms twin's 1.0-1.1 ms profiled gave 0.84.
    - The gathered down-and-sum with its dispatch (compiler schedule, then runner): Emmy's down projection plus the
      dispatch against the fork's down-and-sum, 17.3, 23.0 and 28.5 us per layer, 0.83, 1.10 and 1.37 ms per step at
      one, four and eight requests: 13%, 25% and 25% of the decode gaps. Nothing in the first token. The gate and up
      pieces at eight rows add 0.71 ms more (73.2 against 58.4 us); the slot programs' rows were recorded again on the
      PTX compiler with one try each.
    - The 3,920-token step's half-empty blocks: with every wide expert in a block that fits its rows, 40-80 ms per 4K
      chunk; matching the fork's grouped matmuls, 137. The same fix takes 25-35 ms from the 175- and 239-token steps
      and 25-40 ms from the 784-token step. Per first token: 4K about 0.07-0.11 s, 1K about 0.05-0.07, 32K
      0.3-0.65 (eight chunks). Nothing in decode.
    - The 3,920-token step's row gather, sort, weighting and sum: 94 ms against the fork's 22, 72 ms per chunk if
      matched; the row gather alone is 33 (moved into the group program's row reads, if the twin takes the row index).
    - The short step's host time: at most the 175-token step's 0.04 s at 4K and the 239-token step's 0.05 s at 1K.
  - The next rounds, largest expected gain in the done gate per cost first:
    1. **The expert group program at the expert's width** (twins and runner, no compiler, the shape of #1417). Twins
       over 32- and 128-row blocks beside the 256-row one; an expert of 9 to 32 rows takes a 32-row block, of up to 128
       a 128-row block. Expected: first token 1K 0.61 to about 0.55 s (1.41x to about 1.28x), 4K 1.25 to about 1.16,
       32K 9.47 to about 8.9; decode unchanged. Cost: what #1388 cost. Gate, stated before serving: both new twins
       traced and recorded cold on the card, and the 3,920-token step's expert programs at least 40 ms less and the
       175-token step's at least 20 ms less, profiled. Unmeasured: how many wide experts fit 32 and 128 rows.
    2. **Tune the 4,096-row pre and post programs** (rows only, no code, as #1435 did at 512 rows). Pre4096's rows
       were recorded with one try each; post4096 had up to 39. Share: 114 ms per 4K chunk against the fork.
       Expected: 40-100 ms per chunk if they fall as #1435's did (pre-sym 60%, post-sym 27%), the low end likelier
       since post4096 was tuned once; 4K 0.04-0.1 s and 32K 0.3-0.8 s less. Cost: card time. Gate: pre4096 plus
       post4096 cold at 4,096 rows at least 1.0 ms per layer faster before any boot. Unmeasured: whether faster
       schedules exist at that width.
    3. **Post8's small pieces and the gathered down-and-sum, one compiler round.** Both are a sum split across blocks
       whose finish costs its own small kernels; the 2026-11-18 round found the fused down-and-sum at 34-36 us under
       every schedule tried, so the schedule space has to change. Expected, if both reach the fork's shape: one
       request 16.05 to about 14 ms (1.67x to about 1.45x), eight requests 240 to about 255-265 tok/s (0.75x to about
       0.8x), first tokens unchanged. Cost: compiler work of unknown size, the largest of the five. Gate: post8 cold
       at 8 rows at most 70 us per layer (101 now), and the one-row expert program with its down-and-sum at most
       30 us per layer (29.0 plus 16.6 of dispatch now).
    4. **The 3,920-token step's row gather and combine** (twin input and runner): the group programs read their rows
       through an index instead of a gathered copy. Expected 25-35 ms per 4K chunk. Gate: the step's dispatch at most
       60 ms profiled (94 now).
    5. **The short step's host time** (runner): at most 0.04-0.05 s per first token, nothing in decode. Last, since
       #1428 and #1435 took most of it: the 1-token step is now faster than the fork's.
  - Where the 2026-11-13 ranking no longer holds: its rounds 1 to 4 are done and gained what they expected or more,
    except round 4's 175-token step, now 0.04 s from the fork; its rounds 5 and 6 stand, but round 6's lever is the
    down-and-sum kernel, not the dispatch, so they merge into round 3 above; its "narrower blocks do not pay" is wrong
    for experts that fit a narrower block, which makes round 1 above a round at all; and its 3,920-token step was
    about 0.1 s slower than it read.
  - Recommended: round 1 first, then round 2. Both move every first-token length, which is where the done gate is
    furthest per unit of work, and both cost what earlier twin and tune rounds cost. Round 3 is the largest decode
    lever but the most uncertain; its gate should run before any serving work.
  - Boot: Emmy 758-771 s, the fork 415-468 s.
- **The expert group program at the expert's width (2026-11-19): stopped at its gate, nothing to ship.** Round 1 of
  the 2026-11-19 ranking, measured before building, on main `793647dc9`. Its gate: the 3,920-token step's expert
  programs at least 40 ms less and the 175-token step's at least 20 ms less, profiled. Neither can be met by narrower
  blocks in launches of 32 experts, nor by any mix measured.
  - Rows per expert, counted per layer on a serving boot of main (rank 0, every MoE layer). With random-token prompts,
    as the first-token runs use, an expert of more than 8 rows holds: at 175 tokens 9-16 rows (14 per layer), 17-32
    (10), 33-64 (6), 65-128 (4), 129-256 (3); at 3,920 tokens 31, 30, 25, 22, 17, and 29 experts past 256. Real text
    spreads its rows wider: a 3,136-token step of a document has 39, 43, 47, 52, 53 and 29 per layer, and a 628- or
    697-token step 41-60 experts of 9-32 rows, against 23-29 for a random 784-token step.
  - Cold cost of one launch of 32 experts, each piece tuned or pinned by hand, us: 8 rows 191 (recorded), 16 rows 206,
    32 rows 228, 64 rows 303, 128 rows 484, 256 rows 541 (recorded). Per expert: 6.0, 6.4, 7.1, 9.5, 15.1, 16.9 us. The
    16- and 32-row programs take the 8-row program's route (two cuts, weights decoded inside the matmuls), the 64- and
    128-row ones the same route with the gate and up weights decoded apart. A launch of 8 experts costs about twice
    as much per expert: 102, 107, 113 and 185 us at 8, 32, 64 and 256 rows, its gate and up piece 76 us at 8 and 32
    rows alike.
  - Every launch runs its 32 blocks whatever experts it holds, so a width adds a part-empty launch. At 175 tokens a
    layer has about 37 wide experts, which today fill one or two launches of 256 rows; split across widths, they fill
    more. Expert program time per step estimated from the counts and the costs above, ms (main's estimate against the
    profile: 262 against 267 at 3,920 tokens, its wide program 46 against 43-50 at 175):

    | Step | Main | + 32 rows | + 32 and 128 rows (the ranking's design) | + 32 and 64 rows | + 32 and 64, last launch of each width in 8s |
    | --- | ---: | ---: | ---: | ---: | ---: |
    | 3,920 tokens, random | 262 | 240 | 244 | 240 | 227 |
    | 175 tokens, random | 73 | 71 | 91 | 84 | 61 |
    | 784 tokens, random | 128 | 112 | 118 | 122 | 101 |
    | 239 tokens, random | 83 | 75 | 96 | 89 | 68 |
    | 3,136 tokens, document | 298 | 264 | 265 | 254 | 243 |
    | 628-697 tokens, document | 180-205 | 145-153 | 158-168 | 133-150 | 123-138 |

  - So the ranking's design (32- and 128-row twins) makes the 175- and 239-token steps 13-18 ms slower, and the best
    mix gains 35 ms at 3,920 tokens and 12 ms at 175: the gate is missed at both. The 2026-11-19 expectation assumed
    a wide expert would cost about what the 8-row program costs per expert at every width; the padding of the last
    launch of each width, and the cost of the down projection past 32 rows, take that back.
  - What is still worth having, for the owner to weigh: the 32-row twin alone, no other change, takes about 23 ms from
    a 4K first token and 24 ms from a 1K one on random prompts (2-4%), and 34-52 ms per step of real text. It costs a
    twin, a recorded route and the runner's pick of the narrowest program; the change is about 20 lines. Not built.
  - The tuner missed the mma schedules of the 16- and 32-row down projections: 40 tries found 203 and 369 us, all
    scalar, while the 8-row program's mma schedule, pinned by hand, ran them in 55 and 71 us. A record of these pieces
    has to try the mma rows by hand, or the prior's ranking of them has to change.
  - The first-token runs use random token ids, which route far more unevenly than text: they understate what any
    expert-width change gains on real prompts.
- **The 4,096-row pre and post programs, tuned on the card (2026-11-19).** Round 2 of the 2026-11-19 ranking, main
  `793647dc9`, rows only: no compiler or runner change. Gate, stated before tuning: pre4096 plus post4096 cold at
  4,096 rows at least 1.0 ms per layer faster, strict from the golden alone.
  - Measured first, cold per layer: pre4096 1,957 us in four kernels, the 10,240-deep mixer product 941 us on 320
    blocks and its 10,240-wide output product 635; post4096 3,744 us in fifteen kernels, none above 601.
  - Rows: every kernel above 100 us tuned on the card with `--tune 40`, the six largest movers again with
    `--tune 200`, each on a scratch tune DB; the pre mixer product, which no schedule moved below 920 us, takes a
    cross-CTA split (`g4k`): its partial swept by hand over ten schedules, 510-575 us plus a 33 us finish. The split
    did not pay for post's mixer product. The tune DBs merged into one and the greedy pick of each program recorded
    with `--record-greedy`; rows of unchanged kernels keep their old numbers.
  - Post serves the mixer product in the weight layout its narrower twins settle (kernel `k_82e6a5aca057`), not the
    one a plain `--program` bench picks; a realization-scoped bench shows the served route. Tuned 513 to 358 us alone.
  - Two faster schedules are recorded but not picked: an old row is stored faster than its own schedule measures
    today (pre's output product 577 stored, 635 today, the new row 616; post's output product's old row 501 stored,
    584 today, the new row 499 is picked by 1.5 us). No old row was re-recorded.
  - Gate **passed**: strict cold per layer, pre4096 1,957 to 1,571 us, post4096 3,744 to 2,863 us, -1,267 us.
  - A B B A on boots never profiled, two of main against two of the branch, one server working at a time:

    | Point | Main | This round | Fork, 2026-11-19 |
    | --- | ---: | ---: | ---: |
    | One request, 1K / 512, per token | 16.05-16.06 ms | 16.07 ms | 9.59-9.60 ms |
    | 8 requests, 4K / 1K, total output | 240.0-240.1 tok/s | 243.0-244.6 tok/s | 316.4-320.3 tok/s |
    | First token, 1K (control) | 0.602-0.611 s | 0.612-0.639 s | 0.43 s |
    | First token, 4K | 1.255-1.260 s | 1.185-1.196 s | 0.82-0.83 s |
    | First token, 8K | 2.340 s | 2.210-2.213 s | 1.46-1.47 s |
    | First token, 32K | 9.44-9.45 s | 8.93-8.95 s | 5.93-5.94 s |

  - About 65 ms per 4K chunk (4K -65 ms, 32K -0.51 s over eight chunks), inside the expected 40-100. Against the
    fork the first token is 1.44x slower at 4K (1.51x on main), 1.51x at 8K (1.59x) and 1.51x at 32K (1.59x).
  - Checks: every one of the 19 programs compiles under strict evidence GPU-free, the other 17 to the same listing as
    the 2026-11-19 round; `emmy golden check` says current; the release audit on the card passes. The prior gate does
    not hold this recipe. GSM8K 0.977 (main rounds 0.969-0.984); acceptance 27, 26, 27, 27, 27, 27 of 28 over three
    rounds, the same two known think-mode failures as main (an empty answer, and the reasoning-off case);
    long-document lookups 141 of 141 without thinking and 137 of 138 with it, the miss an empty answer in think mode,
    the same kind as the acceptance one. Greedy texts: none of 6 part across the four boots. Memory, same probe on
    both sides: torch peaks equal to the byte (26.75 GB allocated, 28.94 GB reserved), 368-370 MiB spare on both.
  - Regressions: none measured. The 1K control is within noise. The one cost is pre's output product, whose stale
    old row keeps a 19 us faster schedule from being picked.
- **The decode step's post program and slot programs, by routes nobody had measured (2026-11-21).** Round 3 of the
  2026-11-19 ranking, main `de841a597`. It set out as a compiler round on two things: post8's 13 small pieces (2.1 ms
  per step) and the gathered down-and-sum with its dispatch (0.8-1.4 ms per step behind the fork). Measured first,
  both turned out to be routes the compiler already offers and no row had measured, so the round ships golden rows,
  a twin form and a runner change, and no compiler change.
  - Why the search never got there. An unpinned cut fork takes one seam at a time, and a piece no row has measured
    is priced by the prior's schedule: for the merged pieces here a 512-thread cooperative reduce at 60-500 us, where
    a hand-pinned schedule runs 9-17 us. A merged route loses before anyone tunes its pieces. The 2026-11-18 round's
    "fused down-and-sum runs 34-36 us under every schedule" was the same gap: its sweep never pinned a cooperative
    reduce on the kernel that holds both sums, which runs in 16.4 us (`WORK=t16x8`, `REDUCE=coop`).
  - Gate, stated before building, cold per layer on one card: post8 at most 142 us in at most 14 kernels (164 in
    19); the one-row program with its sum inside at most 34 us with two small kernels of dispatch around it (35.8,
    with five); in serving, one request at least 1.0 ms per token faster and eight requests at least 252 tok/s.
  - Post8, cold per layer: 163.8 us in 19 kernels on main, 139.3 us in 15 here. Ten cuts are pinned on the root
    kernel and each piece keeps what main cut off it: the first mixer weight's product carries its norm's weights
    (14.3 us in two kernels to 9.2 in one), the second mixer's up product its sigmoid (21.5 to 16.4), the last gate's
    product its sigmoid (13.3 to 10.2), and the down product's cross-CTA finish its SiLU (11.2 to 6.1). The
    normalized mixer input stays a kernel of its own: folded into the down product's fill it saved a launch cold
    (140.3 us in 14 kernels) and cost 5.7 us per layer warm, 23.8 us against 18.1. Profiled, rank 0: 105.7 to 92.6
    us per layer, 0.63 ms per step, a third of the 43.7 us the 13 pieces took. One kernel still runs twice, the
    4-stream mean, once into scratch for the shared expert and once into the output (3.4 us per layer): a cut value
    that is also an output is stored twice.
  - The slot programs (`ExpertFFN.slots_sum`): the program takes the router's scores and the shared expert's output
    and returns the layer's routed sum. Around it the runner keeps the selector's write and one write of the scores;
    the row upload, the shared output's add and the weighted sum are gone (five kernels and 16.6 us per layer to two
    and 6.8, profiled). The program reads its rows and the shared output where the post twin wrote them. Recorded in
    three pieces: gate and up with the activation (the activation's own kernel is gone), the down projection as on
    main, and a sum-and-add piece. Cold per layer: one row 35.8 to 33.8 us, two rows 47.1 to 46.1.
  - Two forms were measured and dropped. The fused down-and-sum in one kernel: 15.0 us warm at one row and 54.6 at
    eight, against 9.9 and 41.5 for the plain down projection, because a thread walks an expert's 160 columns alone.
    And the summed program at four and eight rows: cut from the sum, the down projection writes its picks fastest,
    so a block reads the rows of 32 experts; 71.7 us warm at eight rows against main's 41.5, though both bench 42.0
    cold (the bench's selector does not spread its picks). The cut orders a workspace by a weight's layout only when
    the weight is read straight from memory, not through the selector. So the four- and eight-row programs stay
    main's, with main's rows.
  - A B B A on boots never profiled, two of main against two of the branch, one server working at a time:

    | Point | Main | This round | Fork, 2026-11-19 |
    | --- | ---: | ---: | ---: |
    | One request, 1K / 512, per token | 16.07-16.08 ms | 15.16-15.19 ms | 9.59-9.60 ms |
    | 2 requests, 4K / 1K, total output | 91.8-92.0 tok/s | 95.6-95.8 tok/s | 110.6 tok/s |
    | 4 requests, 4K / 1K, total output | 156.7-156.9 tok/s | 159.7-160.3 tok/s | 203.2-203.5 tok/s |
    | 8 requests, 4K / 1K, total output | 244.3-244.5 tok/s | 248.2-248.9 tok/s | 316.4-320.3 tok/s |
    | First token, 1K (control) | 0.609 s | 0.602-0.619 s | 0.43 s |
    | First token, 4K (control) | 1.189-1.200 s | 1.205-1.227 s | 0.82-0.83 s |

  - The steps between tokens fell by 0.90 ms at one request (16.07 to 15.18), 0.88 at two (19.49 to 18.61), 0.56 at
    four (21.37 to 20.80) and 0.55 at eight (24.82 to 24.28). Against the fork: one request 1.58x slower (1.67x), two
    requests 0.87x (0.83x), four 0.79x (0.77x), eight 0.78x (0.77x). The 4K first token read 1.191-1.199 s on an
    earlier pair of branch boots against 1.199-1.221 s on main: no change.
  - Gate **missed, closely at one request**: 0.90 ms per token against the 1.0 stated, and 248.2-248.9 tok/s at
    eight requests against 252. Behind it: post8 met its time (139.3 us against 142) with 15 kernels where 14 were
    stated; the one-row program met its bar (33.8 us, two kernels around it); the eight-row program is unchanged, so
    eight requests gain only what post8 gains.
  - Checks: every one of the 19 programs compiles under strict evidence GPU-free; `emmy golden check` says current;
    the release audit on the card passes (19 twins deploy from the golden rows alone); program by program, every
    program but post8 and the two summed twins keeps main's kernels, routes and rows. A numeric check on the card,
    in-domain inputs against the NumPy reference of post8's traced program: the new route and main's both within one
    fp16 step of it. GSM8K 0.969 (main rounds 0.969-0.984); acceptance 28, 27, 28, 26, 27, 28 of 28, the same two
    known failures as main (an empty answer in think mode, and the reasoning-off case); long-document lookups
    LONGDOC. Greedy texts: main's two boots agree on 6 of 6, the branch's two agree on 6 of 6, and the branch equals
    main on 3 of 6: the routed sum and post8's products add in another order. Memory, same probe on both sides:
    torch peaks 26.75 GB allocated on both, 28.94 GB reserved on main and 28.61 GB here; 768 MiB spare per card here
    against 346-370 on main, the allocator-timing flip of earlier rounds. CI's full suite passes. The prior gate
    does not hold this recipe.
  - Regressions: none measured in serving. Three of six greedy texts part from main's. The record re-timed two rows
    of kernels pre8 shares with post8; both are back at main's values.
  - Left for a compiler round, each measured here: the workspace order of a product cut from a sum over a gathered
    weight (it decides whether the summed program pays at four and eight rows, 9.5 us of dispatch per layer); the
    mean stored twice (3.4 us per layer); and a search that tunes the pieces of a route before it prices the route.
    A load-vectorizer change that widens a packed weight's byte loads in a serial loop sped the fused down-and-sum
    from 16.4 to 14.3 us at one row and from 63.5 to 51.2 at eight; no route here uses that kernel and the change
    moves code of other models' goldens, so it was left out.
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
  against Emmy by a wide margin. For scale: the fork's whole token here is 12.8 ms without MTP on our node (78 tok/s
  on four cards, step 1), while Emmy's DeepSeek image spends 79 to 149 ms per token on sixteen. Parity may take many
  rounds, or may stop at a part only the fork can run fast.
- **A moving target.** Upstream lands changes daily. The baseline is the pinned build of step 1. Re-pinning during
  step 5 resets the A/B.
- **Compile time.** On a four-core V100 host one layer program spent 15 to 75 minutes pricing its kernel sets, and on
  2026-09-18 one greedy resolve of a Gated DeltaNet chunk took 48 minutes; both need re-measuring on main. Twins are
  shared per layer class, so this is three classes at every width, not 48 layers.
- **Memory.** Emmy's prefill needed about 2 GiB per card beyond the fork's on DeepSeek. Step 2's TP4 shape keeps the
  3 GiB margin with only about 0.6 GiB more to spare. If Emmy needs more than that, the shape is PP2 × TP4, where the
  fork keeps about 8.7 GiB free beyond the margin but decodes about 10% slower and halves the replicas per node.

## Side finding: the same update may speed up what we already serve

Upstream reports DeepSeek V4 Flash at 65 to 74 tok/s and Qwen3.8-27B at 50 to 62 tok/s. We serve
DeepSeek-V4-Flash-0731 at about 7 to 12 tok/s per request. The two are not comparable as they stand: upstream ran the
earlier checkpoint on eight cards at a short context with 1,024-token prompts, and we run the 0731 checkpoint on 16
cards at a 1M context. Emmy's 0.8× lead on DeepSeek was measured against the August fork. It is worth one day on the
test node after step 3, at our checkpoint and envelope, and its own plan if it holds.

## Not established

- The quality of FP8 against full precision on our tasks. NVFP4 matches it (step 1).
- Whether MTP stays safe under concurrency with real prompts and sampling; issue 865 did not reproduce in step 2's
  30 minutes of random prompts at eight requests.
- The longest context PP2 × TP4 holds with the margin; TP4 holds 131,072 (step 2).
- How fast vision input runs on Volta.
- Whether Qwen grants the separate license commercial serving needs.

## Effort

No total yet. Step 1 took one day, with 43 minutes of rented H200s for the reference; step 2 is measurement on the
test node. Step 4 has more new parts than DeepSeek had. DeepSeek took about six weeks from its first boot to parity;
expect step 4 and step 5 together to take longer, and estimate step 5 only after the first strict boot and its
profile.

## Operations

The test node is shared. Check that every card is free before a launch, never stop a process that is not ours, and
delete nothing when done. Its address stays out of the repository.
