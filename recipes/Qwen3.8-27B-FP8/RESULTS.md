# Qwen3.8-27B-FP8 on V100

The recommended platform is four V100 SXM2 16GB at TP4, **on a host where NVLink is up**. On the same four cards
with NVLink turned off, time to first token is 2.5-3.5x longer and prompts past about 190K tokens exceed a 300 s client
timeout. Two V100 SXM3 32GB cards were also qualified; they are faster at decode but cannot get NVLink on CloudRift,
and they are kept below as evidence rather than as a recommended platform.

**Volta has no FP8 arithmetic.** The checkpoint is served through the 1Cat-vLLM fork's TurboMind dequantization path,
the same route the DeepSeek-V4-Flash lane uses on this engine. Every number below is a property of that path, not of
hardware FP8.

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
| Serving shape | TP4, context 262,144, `gpu_memory_utilization` 0.88, text-only, concurrency cap 4 — the cap this table was measured at; the recipe now ships 16, see Concurrency below |
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

`--max-num-batched-tokens` was tested at 8192 against the shipped 4096 and rejected. It raised the KV pool from
288,281 to 355,162 tokens, but no row improved beyond noise, and 4K prompts at 16 concurrent got 69% slower
(6.2 s to 10.4 s median) because a request arriving mid-step waits longer for a larger step to finish.
`gpu_memory_utilization` and FP8 key/value cache were not tested, so no claim is made about them.

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

There is no Emmy lane. The recipe serves the stock 1Cat-vLLM fork, and every figure here is that engine.

## Reproduce

```bash
emmy vm create cloudrift --instance-type v100-6-52-400-generic.4 --ssh-key ~/.ssh/id_ed25519.pub \
  --image-url https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpup-580-129-20260810-232733.img
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
