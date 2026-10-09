# Qwen3.8-27B-FP8 on V100

The recommended platform is four V100 SXM2 16GB at TP4, **on a host where NVLink is up**. On the same four cards
with NVLink turned off, time to first token is 2.5-3.5x longer and prompts past about 190K tokens exceed a 300 s client
timeout. Two V100 SXM3 32GB cards were also qualified; they are faster at decode but cannot get NVLink on CloudRift,
and they are kept below as evidence rather than as a recommended platform.

**Volta has no FP8 arithmetic.** The checkpoint is served through the 1Cat-vLLM fork's TurboMind dequantization path,
the same route the DeepSeek-V4-Flash lane uses on this engine. Every number below is a property of that path, not of
hardware FP8.

**The H200 entry is not measured here.** The recipe also carries an H200 x1 entry at 0.55 of the card for a two-model
deploy plan beside Ornith-1.5-35B-A3B-FP8; that lane keeps the vision tower and accepts up to four images per request.
It has not been run, with or without an image. The Volta lanes below are text-only, exactly as qualified.

## Four V100 SXM2 16GB (TP4)

Re-qualified 2026-09-15 on a host with NVLink; first qualified 2026-09-05 on a host without it.

### Host requirement: NVLink

CloudRift's default GPU VM image (`ubuntu-noble-server-gpup-580-129-20260430-084759`) sets `NvLinkDisable=1` in the
NVIDIA driver, so a rental from the catalog comes up with every tensor-parallel all-reduce copied through host memory.
CloudRift fixed the image build on 2026-08-08, but the catalog still points at the April image. This lane was measured
on a VM booted with `emmy vm create cloudrift --image-url` from `ubuntu-noble-server-gpup-580-129-20260810-232733`:
driver 580.173.02, three active 25.8 GB/s links per GPU, pairs 0-1 and 2-3 joined by one link, pairs 0-2 and 1-3 by
two, and the diagonals 0-3 and 1-2 on PCIe. Check a host with `nvidia-smi topo -m`: linked pairs show `NV1`/`NV2`.

Because two of the six pairs are not linked, vLLM's own custom all-reduce stays off and tensor-parallel traffic goes
through NCCL, which can ring over linked pairs only.

### What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a` |
| GPUs | 4 x NVIDIA Tesla V100 SXM2 16GB, compute capability 7.0, driver 580.173.02, NVLink as above |
| Engine image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM `1.2.3.dev87+gd76126608.d20260810`) |
| Serving shape | TP4, context 262,144, `gpu_memory_utilization` 0.88 (the recipe now also pins the key/value cache at 4.5 GiB per card, see Memory below), text-only, concurrency cap 4 — the cap this table was measured at; the recipe now ships 16, see Concurrency below |
| Backends | FLASH_ATTN_V100 attention, Triton Gated DeltaNet prefill, TurboMind FP8 dequantization |
| Workload | 16 prompts, 1,000 input / 1,000 output tokens, client concurrency 4, temperature 0, ignored EOS, 2 warm-ups, three repeats on one server with seeds 0, 1, 2 |

| Metric | Repeat 1 | Repeat 2 | Repeat 3 |
| --- | ---: | ---: | ---: |
| Successful / failed requests | 16 / 0 | 16 / 0 | 16 / 0 |
| Benchmark duration | 386.98 s | 381.88 s | 417.64 s |
| Output token throughput | 41.35 tok/s | 41.90 tok/s | 38.31 tok/s |
| Total token throughput | 82.69 tok/s | 83.79 tok/s | 76.62 tok/s |
| Median TTFT | 1,036.83 ms | 1,307.74 ms | 1,309.00 ms |
| P99 TTFT | 2,758.08 ms | 1,327.88 ms | 1,329.39 ms |
| Median TPOT | 95.39 ms | 94.38 ms | 103.40 ms |
| P99 TPOT | 97.92 ms | 95.23 ms | 104.48 ms |
| Median inter-token latency | 95.10 ms | 94.39 ms | 103.25 ms |

The third repeat decoded about 9% slower than the first two, and the server's own generation-throughput log drops at
the same time, so the spread is the host's, not the client's. Per stream that is roughly 10 tokens/s. The KV pool
holds 288,281 tokens, 1.10x concurrency at the full window. Model load to health took 287.8 s: 7.2 s of weights,
107.2 s of `torch.compile` and 111.0 s of CUDA graph capture.

### Time to first token

One streamed request at a time, each with a passphrase planted at 40% depth; every row below returned it exactly. The
first column is an earlier test of this lane through Relay on the NVLink-disabled image, on a different host and
client path, so ratios against it are directional.

| Prompt | SXM2 x4, NVLink off (Relay) | SXM3 x2, no NVLink | **SXM2 x4, NVLink** |
| --- | ---: | ---: | ---: |
| ~4K | 5.0 s (4,067 tokens) | 2.9 s (4,038) | **1.6 s** (4,038; 2,506 tok/s) |
| ~16K | 19.2 s (16,096) | 11.4 s (16,002) | **5.5 s** (16,002; 2,926 tok/s) |
| ~32K | 39.9 s (32,198) | 24.8 s (32,012) | **11.9 s** (32,012; 2,689 tok/s) |
| ~64K | 81.7 s (65,147) | 57.6 s (63,894) | **28.0 s** (63,894; 2,279 tok/s) |
| ~128K | 179.9 s (130,530) | 147.9 s (127,892) | **73.2 s** (127,892; 1,748 tok/s) |
| ~200K | failed at 302 s | 287.5 s (199,747) | **144.0 s** (199,747; 1,387 tok/s) |
| ~250K | failed at 302 s | 407.8 s (249,581) | **205.9 s** (249,581; 1,212 tok/s) |

NVLink matters most for prefill, where every layer sends a full chunk of activations across the cards. Prefill speed
still falls as the prompt grows, because attention over a longer context costs more per token. These are lone
requests: a long prompt that arrives behind another still waits for it.

Against the 2026-09-05 benchmark of this lane without NVLink — 32.46 tok/s output, 100.98 ms median TPOT, 23,072 ms
median TTFT, one repeat — output throughput is about a quarter higher and TTFT drops from seconds to about one second.
Decode itself barely moves, since a decode step's all-reduce is small and the per-card dequantization dominates.

### Capability checks

Run against this lane's deployed recipe on the NVLink host:

| Gate | Result |
| --- | --- |
| Coherent answers | Pass — `144` for a multiplication with tools offered, a coherent answer from a returned tool result |
| Reasoning separation | Pass — with thinking on, the `reasoning` field is populated and the call is separate |
| Context fill | Pass — a planted passphrase was retrieved from a 249,581-token prompt |
| Tool calling | Pass — every probe in the table below |

Reasoning requires `chat_template_kwargs: {"enable_thinking": true}`; Qwen3.8 defaults thinking off and otherwise
reasons inline in `content`.

| Probe | Result |
| --- | --- |
| `auto`, three tools offered | `get_local_time{"city": "Tokyo"}` for a time question |
| `auto`, two cities | Two parallel calls, `get_weather` for Paris and for Rome |
| `auto`, typed arguments | `book_flight` with an ISO date and integer `passengers: 2` |
| `auto`, no tool needed | Plain answer `144`, no call |
| `required`, a relevant question | Structured calls, `finish_reason: tool_calls` |
| `required`, unrelated questions, thinking off | Structured calls, 3/3 |
| Named tool | The named tool is called; vLLM reports a forced call with `finish_reason: stop` |
| Tool result fed back | A coherent answer that uses the returned temperature and condition |
| Streaming | Tool-call deltas reassemble into `get_weather{"city": "Oslo"}` |
| Thinking on | Reasoning separated, and the call respects the argument enum |
| `none`, tool-shaped questions | Plain answers, no markup, no call, 3/3 |

**`tool_choice: "none"` needed a flag.** Without it the tool list stays in the prompt, the model still writes its
`<tool_call><function=get_weather>…` markup, and the `none` path returns that markup as the answer text — 3/3 before
the change. The recipe now sets `--exclude-tools-when-tool-choice-none`, which this image supports.

A combination matrix over thinking on/off x `auto`/`required`/`none`/named x streamed/plain x five prompt kinds (one
tool fits, two parallel calls, typed arguments, no tool fits, a tool result fed back) passed all 80 cases on this lane:
the expected call or no call, a known tool with parseable arguments carrying every required field, no markup in
`content`, reasoning present only with thinking on, and nothing stopped at the token limit. Streamed and plain replies
agreed case by case.

Two behaviours follow from the API rather than from this deployment, and a client has to plan for them:

- **`required` and a named tool invent a call when no tool fits.** Asked for a haiku under `required`, the model
  returned `get_weather{"city": "Tokyo"}`; with thinking on it wrote the haiku inside its reasoning first, where it is
  then discarded. Use `auto` unless a call is genuinely mandatory.
- **`required` after a tool result calls again instead of answering.** Fed `{"temperature_c": 18}` for Paris, it
  re-issued `get_weather{"city": "Paris"}`. A client that keeps `required` on every turn loops; switch to `auto` once a
  tool result is in the conversation.

**`tool_choice: "required"` can also run to the output limit when thinking is on and no tool fits.** `required` forces
a JSON array of calls, but only after the reasoning block ends. On the SXM3 lane with two tools offered, a haiku
request with thinking on reasoned past `finish_reason: length` at both 256 and 1,024 tokens, with empty `content` and
no call. The same request with three tools offered closed its reasoning after ~320 tokens and produced a call, so the
failure depends on the prompt rather than on the platform, and it is not fixed. Clients forcing a call should keep
thinking off for that request, or cap `max_tokens`.

### Concurrency

The cap was raised from 4 to 16 after measuring it. Both tables below are one run per point on one four-card SXM2
host, so treat differences under about 10% as noise — the qualification lane's own repeats spread that far.

Throughput against the cap, at 1,000-token prompts with client concurrency matched to the server cap:

| Cap | Output throughput | Median TTFT | P99 TTFT | Median TPOT |
| ---: | ---: | ---: | ---: | ---: |
| 4 | 40.8 tok/s | 1,024 ms | 2,500 ms | 92 ms |
| 8 | 76.8 tok/s | 1,686 ms | 4,176 ms | 96 ms |
| 16 | 133.8 tok/s | 3,310 ms | 5,828 ms | 105 ms |
| 32 | 235.5 tok/s | 4,778 ms | 9,917 ms | 116 ms |

Throughput scales almost linearly while per-token decode latency barely moves, so the cost of concurrency is paid
almost entirely in waiting for a prefill slot. 16 was chosen over 32 because it keeps P99 time to first token under
6 s.

The question that decides whether 16 is safe is what it does to long prompts. Measured at cap 16 against one request
at a time on the same server:

| Prompt | One request | 16 concurrent, median | 16 concurrent, P99 |
| ---: | ---: | ---: | ---: |
| 4K | 1.4 s | 6.2 s | 22.3 s |
| 16K | 5.6 s | 22.8 s | 112.5 s |
| 32K | 12.2 s | 132.2 s | 266.0 s |
| 128K | 75.0 s | 208.5 s | 337.6 s |
| 262K | 219.2 s | 438.4 s | 653.1 s |

Nothing fails and nothing thrashes. At the full window the behaviour is plain serialization: one request takes
219 s, and three in flight put the median at 438 s, which is twice that. The engine admits what the pool holds and
queues the remainder rather than preempting. The knee is at 32K, and it is the pool rather than the cap: sixteen
32K prompts need 524,288 tokens of key/value cache against a pool of 288,281, so half the batch waits a full cycle.
Sixteen 16K prompts need 262,144, which still fits.

Two limits here are properties of context length, not of this cap. A full-window prompt costs 219 s of prefill with
one request and an idle server, which already exceeds a 300 s client timeout. And the engine reports its own ceiling
at startup: maximum concurrency for 262,144 tokens per request is 1.10x, so one full-window request owns the pool
whatever the cap says. A client sending many long prompts needs its own concurrency limit or a longer timeout; a
smaller server cap does not help it.

Raising the cap does not slow a request that arrives alone. Per-token prefill rate at concurrency 1 on the cap-16
server, against the single-request ladder above that was measured with the cap at 4:

| Prompt | TTFT at concurrency 1 | Prefill rate | Against cap 4 |
| ---: | ---: | ---: | ---: |
| 4K | 1.37 s | 2,998 tok/s | see note |
| 16K | 5.60 s | 2,924 tok/s | +0.5% |
| 32K | 12.21 s | 2,683 tok/s | -0.3% |
| 128K | 75.03 s | 1,747 tok/s | 0.0% |
| 262K | 219.24 s | 1,195 tok/s | -1.4% |

The cap governs how many requests are admitted, not how fast one is processed, and the measurement agrees from 16K
up. The 4K row reads 18.8% faster on the cap-16 server, which is protocol rather than platform: the published ladder
planted a passphrase in each prompt and streamed one request, while this lane sent random prompts through the bench
client, and at 1.4 s the fixed per-request overhead is a large share of the total. Where prefill dominates the two
methods agree within 1.4%.

The same rows show what context length alone costs. Prefill runs at about 3,000 tokens/s up to 16K, holds 90% of that
at 32K, and falls to 1,747 at 128K and 1,195 at the full window - 2.5x slower per token across a 64x longer prompt,
which is 160x the wall-clock time. The degradation is that mild because 48 of the 64 layers are Gated DeltaNet, whose
cost is linear in sequence length; only the remaining 16 pay the quadratic attention cost. The floor this sets is
219 s to first token for a full-window prompt on an idle server, which no cap setting changes.

`--max-num-batched-tokens` was tested at 8192 against the shipped 4096 and rejected. It raised the KV pool from
288,281 to 355,162 tokens, but no row improved beyond noise, and 4K prompts at 16 concurrent got 69% slower
(6.2 s to 10.4 s median) because a request arriving mid-step waits longer for a larger step to finish.
FP8 key/value cache was not tested, so no claim is made about it. Memory sizing is covered next.

### Memory: the key/value cache is pinned

The lane shipped with the cache sized from `gpu_memory_utilization` 0.88, and in production that ran out of memory
twice, five days apart, mid-request: an 80 MiB activation buffer of a 4,096-token prefill chunk found under 75 MiB
free on every card. The engine core died and the server stopped answering until it was restarted by hand.

The cause is how vLLM sizes the cache. It profiles memory at start-up and gives the cache whatever the fraction leaves,
and that profile is about 1.5 GiB larger on a cold start than on a restart with a warm compile cache. The same fraction
therefore gives a different cache, and a different margin, depending on how the container last started:

| Shape, 2026-10-06, one SXM2 x4 host with NVLink | Cache per card | KV pool | Full-window concurrency |
| --- | ---: | ---: | ---: |
| 0.88, warm restart — the shape that ran out of memory | 6.02 GiB | 378,994 | 1.45x |
| 0.85, cold start | 4.04 GiB | fails to start: 40 MiB short of one full-window request | — |
| 0.85, warm restart | 5.54 GiB | 349,012 | 1.33x |
| **pinned 4.5 GiB, cold start and warm restart** | **4.5 GiB** | **289,050** | **1.10x** |

The production backend's first, cold start ran for eleven days; its restart got the warm 6.0 GiB cache and died five
days later. A lower fraction cannot fix this, because at 0.85 it already cannot start cold at full context, and it still
gives most of any saving back to the cache on a warm start. `--kv-cache-memory-bytes` pins the cache at 4.5 GiB, the
cold size, so every start leaves 1.5 GiB more per card outside the cache than the shape that failed, and the pool is
the 288,281 tokens the lane was qualified with.

The pinned shape passed every check on the host above: a cold start without a restart, a warm restart to the same
pool, a planted passphrase retrieved from a 240,046-token prompt in 194 s, and two loads at 16 concurrent requests with
512 output tokens and ignored EOS:

| Load, 16 concurrent | Shape | OK / failed | Output | Median TTFT | P99 TTFT | Median TPOT |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 48 x 16K prompts | pinned 4.5 GiB | 48 / 0 | 39.8 tok/s | 22.9 s | 142.9 s | 315 ms |
| 48 x 16K prompts | 0.88 warm | 48 / 0 | 42.1 tok/s | 17.0 s | 113.7 s | 334 ms |
| 64 x 4K prompts | pinned 4.5 GiB | 64 / 0 | 115.1 tok/s | 6.0 s | 22.7 s | 128 ms |
| 64 x 4K prompts | 0.88 warm | 64 / 0 | 116.5 tok/s | 6.0 s | 22.7 s | 126 ms |

These loads did **not** reproduce the failure: the 0.88 warm shape survived them too, so they show that the pinned
shape serves, not that it removes the failure. The production requests that failed ran with 93% prefix-cache hits
over several days, which random prompts in a 15-minute run do not recreate. The margin argument above is what the fix
rests on. Its cost is the smaller pool against a warm 0.88 start: 16K prompts at 16 concurrent wait about a third
longer for their first token, and 4K prompts are unaffected. One run per row.

### Fit

30.9 GB of FP8 weights give a min-to-serve near 40.2 GB against 4 x 16,384 MiB = 68.7 GB of platform capacity. Two
cards hold only 34.4 GB and cannot take the weights, so four is the floor. TP4 divides cleanly: 24 attention heads
/ 4 = 6, 4 key/value heads / 4 = 1.

## Two V100 SXM3 32GB (TP2): measured, not recommended

Qualified 2026-09-15 and removed from the recipe the same day. On V100 SXM3 nodes NVLink runs through NVSwitches, and
CloudRift passes those switches only to a rental that takes every GPU on the node, so a two-card rental gets the GPUs
alone: `nvidia-smi topo -m` reports PHB, `nvidia-smi topo -p2p r` reports no peer-to-peer, and every all-reduce is
copied through host memory. That rental's VM image also had `NvLinkDisable=1`.

| Metric | Repeat 1 | Repeat 2 | Repeat 3 |
| --- | ---: | ---: | ---: |
| Successful / failed requests | 16 / 0 | 16 / 0 | 16 / 0 |
| Output token throughput | 56.28 tok/s | 55.49 tok/s | 56.86 tok/s |
| Median TTFT | 1,826.25 ms | 2,205.16 ms | 1,828.56 ms |
| Median TPOT | 69.18 ms | 70.06 ms | 68.55 ms |

Same engine image, flags and workload as the SXM2 lane, at TP2 with a 365,925-token KV pool (1.40x at the full
window), driver 580.126.20. With fewer cards to synchronize it decodes faster than four SXM2 cards, but its prefill is
the slowest of the three measured setups (table above), and it cannot hold a 250K prompt under a 300 s timeout.

## Limitations

- **The NVLink requirement is outside the recipe.** A recipe cannot choose the VM image; until CloudRift's catalog
  serves an NVLink-enabled image, rent with `--image-url` or check `nvidia-smi topo -m` before deploying.
- **FP8 here is emulation.** Throughput reflects TurboMind dequantization and should not be read as an FP8 hardware
  result, nor compared against FP8 numbers from Hopper or Blackwell parts.
- **The engine image cannot build its own Volta Gated DeltaNet kernel.** 48 of the 64 layers are Gated DeltaNet.
  `flash_qla_sm70_gdn_strided` is a JIT torch extension whose build fails at `fatal error: cusparse.h: No such file
  or directory`, because the runtime layer ships the sources without the CUDA development headers. Selecting the
  Triton GDN *prefill* backend is not enough — the GDN decode path reaches for flash_qla independently — so the
  recipe sets `VLLM_SM70_GDN_DECODE_FLASHQLA=0`. A purpose-built sm_70 image that prebuilds it is the most likely
  source of a further speed-up.
- **Only Volta platforms were qualified.** The discovery shell this recipe replaces proposed one RTX PRO 6000
  Blackwell Max-Q and one H200 141GB; both remain unmeasured, not unsuitable, and both have native FP8 hardware.
- **Multimodal input is out of scope.** The recipe passes `--language-model-only`; the vision tower is not loaded.

## Emmy

Measured 2026-10-07 on the same four SXM2 16 GB cards, CUDA 12.9, torch 2.14.0+cu126.

Emmy now serves this checkpoint end to end on the same four SXM2 16 GB cards, one pipeline stage per card. It is
correct but much slower than the stock lane, so the recipe keeps stock 1Cat-vLLM. The FP8 weights stay coded on the
card and decode inside each matrix multiply under 16-bit activations; decoded to FP16 the trunk alone would need
13.5 GB of every card. `serving-v100.env` names the shape, and the golden now holds its nine serving programs.

| Setting | Value |
| --- | --- |
| Image | vllm-emmy built on `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` at repository `6e223e6ad`, not published |
| Shape | PP4 × TP1, FP16 trunk, standard lane (fast math off), decode width 16, prefill width 64, 64 batched tokens, 4 sequences, 8,192 context, eager, no prefix caching |
| Boot | 13 min from container start to ready, every kernel from a golden row under `EMMY_STRICT_EVIDENCE=1` |
| Key/value pool | 33.1× concurrency at 8,192 tokens |

| Workload | Emmy | Stock 1Cat-vLLM (TP4) |
| --- | ---: | ---: |
| One request, 96 output tokens: decode | 196 ms/token | — |
| One request, ~25-token prompt: time to first token | 4.6 s | — |
| Four concurrent requests, 100 output tokens each: per-stream decode | 557–638 ms/token | 92 ms/token (1,000-token prompts) |
| Four concurrent requests: output throughput | 5.4 tok/s | 40.8 tok/s |

The Emmy rows are single probes, not a benchmark run. Correctness: layers 0–3 (three gated DeltaNet layers and one
full-attention layer) on the real weights match transformers on decoded fp32 weights to a relative L2 error of 4e-4 to
8e-4 at 1, 16 and 64 tokens. The served model answers factual and explanatory prompts correctly, returns a structured
`tool_calls` entry with the `qwen3_coder` parser, and puts its thinking in the reasoning field with the `qwen3` parser.

Where the time goes. A decode token spends about 137 ms in the 48 gated DeltaNet layers (2.85 ms each) and about
39 ms in the 16 full-attention layers. Known gaps, largest first:

- Gated DeltaNet layers serve the requests of a step one after another, so decode slows with every request in flight.
- At one token no tensor-core tile is offered (one row has no output-axis pair for a fragment), so the width-1
  projections run thread schedules at 5–6× their weight-streaming floor; the width-1 state update recomputes its
  convolution and norm per output cell and takes about 1 ms.
- Prefill is slow: 55 ms per gated DeltaNet layer for a 16-token step and 120 ms for a 64-token one, and the any-width
  attention program runs scalar (1.7 s at 512 tokens). Long prompts are impractical.
- The serving programs carry no `torch.compile` comparison: on the bench's random FP8 weights the eager reference
  overflows FP16 and returns NaN.

Each program was cut by hand (pinned `PLACE` routes recorded with `--record-greedy`), and the slowest pieces were
tuned with `emmy run --kernel … --tune`. The gated DeltaNet routes take the full-projection cut and then cut the
multi-output pieces it leaves serial; the triangular state update reuses the shared-memory schedule recorded above.

### The golden

`golden/v100_sm70.json` holds the nine programs `serving-v100.env` compiles, traced from the real checkpoint with
`emmy trace --serving-twins`: the full-attention layer before and after the attention call at 16, 64 and any number of
tokens, and the gated DeltaNet layer at 1, 16 and 64 tokens, all in the standard lane. It has 151 kernels, 31 routing
rows and 133 measured rows, and a strict-evidence boot compiles every kernel from it. Replaying a program from the
file alone (`emmy run --golden … --realization … --strict-evidence`) takes, per layer: gated DeltaNet 2.8 ms at one
token, 55 ms at 16 and 120 ms at 64; attention 0.67 + 1.79 ms at 16 tokens and 6.6 + 43 ms at 64.

It replaces the four compile-path programs the 2026-10-02 passes recorded (one gated DeltaNet layer traced through
`emmy compile`, the output head and the recurrence), which were partial coverage and not what serving runs. Their
best schedules carried over where the kernels are the same: the shared-memory triangular update (200 us at 16 tokens,
against 42 ms unpinned), the 64-step scan and the cut of the QK product. Those passes measured, against
`torch.compile`: the scans 2.2–3.5x faster, the QK and triangular mask 1.1–1.5x, the recurrent update 1.45x and the
output head 1.85x.

### Reproduce

```bash
# capture the serving programs (needs the checkpoint), then replay one from the golden alone
emmy trace /path/to/Qwen3.8-27B-FP8 --serving-twins --serving-config recipes/Qwen3.8-27B-FP8/serving-v100.env -o twins.json
EMMY_FAST_MATH=0 emmy run --golden recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json \
  --realization gdn1-dense-linear@fp8 --strict-evidence --bench --bench-backends emmy
# serve, one pipeline stage per card, in a vllm-emmy image built on the 1Cat base
EMMY_FAST_MATH=0 EMMY_GEN_DECODE_BUCKET=16 EMMY_GEN_PREFILL_BUCKET=64 EMMY_GEN_M1_TIER=0 EMMY_STRICT_EVIDENCE=1 \
emmy serve Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a --runner generate \
  --golden recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json --dtype float16 --pipeline-parallel-size 4 \
  --max-model-len 8192 --max-num-seqs 4 --max-num-batched-tokens 64 --no-enable-prefix-caching \
  --language-model-only --gpu-memory-utilization 0.92
```

## Reproduce

```bash
emmy vm create cloudrift --instance-type v100-6-52-400-generic.4 --ssh-key ~/.ssh/id_ed25519.pub \
  --image-url https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpup-580-129-20260810-232733.img
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
