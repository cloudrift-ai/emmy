# Qwen3.8-Flash-Next on V100 nodes through Emmy

Goal: serve `Qwen/Qwen3.8-Flash-Next` on CloudRift's 16× V100 SXM3 nodes through Emmy's own kernels, the way
DeepSeek V4 is served: first running, then optimized until the Emmy serving image is at least on par with the plain
1Cat-vLLM fork at an equal serving envelope. The result ships as an Emmy recipe that `emmy deploy` and Relay run.

Emmy's kernels run inside the fork's runtime, which supplies the serving shell and whatever Emmy does not take. So
the fork has to serve the model first: steps 1 to 3 set up the runtime and record the baseline to match, step 4 gets
Emmy serving the model correctly, step 5 closes the speed gap, and step 6 releases.

Written 2026-10-13, revised the same day after a review and after the owner fixed the goal. Steps 1 and 2 ran on
2026-10-14 and 15; each result is written under its step. Numbers marked "upstream" are 1Cat-vLLM's own reports.

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

### Our copy of the fork is two months old

`cloudrift-ai/1Cat-vLLM` was last pushed on 2026-08-13. It is 3,398 commits behind upstream and carries 13 commits of
our own: the DeepSeek V4 long-context path and KV pool, a Laguna FP8 precision fix, Poolside parser fixes and warm-up
changes. It has no `qwen4_exp`. Every V100 recipe pins an image built from it.

### Emmy today

- `recipes/Qwen3.8-Flash-Next` is an untested shell for H200, H100 and B200.
- Emmy's own kernels have nothing for this architecture yet. The tracer covers every layer kind (step 4 item 1).
  Missing in Emmy: the trace seams, the MoE split for Qwen's shared-expert gate, hosting Qwen Sparse Attention and the
  Gated DeltaNet layers, MTP, and vision.
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
minimum with all four replicas running and passes the step 1 accuracy sets. The baseline is the experiment
`experiments/Qwen3.8-Flash-Next/fork_baseline_v100_sxm3`. The minimums still wait for the owner's confirmation.

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
- **Baseline, one replica, repeats 2-4 of 4:**

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
