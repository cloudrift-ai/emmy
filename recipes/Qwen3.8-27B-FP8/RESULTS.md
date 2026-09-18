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
| Serving shape | TP4, context 262,144, max 4 concurrent requests, `gpu_memory_utilization` 0.88, text-only |
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

There is still no Emmy serving lane: the recipe serves the stock 1Cat-vLLM fork, and every serving figure above is
that engine. What follows is compiler evidence — measured CUDA kernels for this checkpoint's decoder path on one
V100 — and it does not imply that Emmy can serve the model.

Compiler-qualified 2026-09-18 against `962378ed4` on four Tesla V100-SXM2-16GB (one card per walk), CUDA 12.9,
torch 2.13.0+cu126, at nvcc's deployable `-O3`.

### Coverage

| Item | Value |
| --- | --- |
| Archetypes traced | layer 0 (Gated DeltaNet, 116 kernels) and layer 3 (full attention, 14), covering all 64 layers |
| Distinct targets | 129 |
| In the golden | 120 targets, 125 measured rows, every row strict-decoding against this compiler |
| Not measured | 9 targets — every one a kernel that does not finish 60 s of GPU time (below) |
| Reference | the isolated re-bench of each target's own greedy pick (`same-input-greedy`), 10 warm-up / 100 measured |

The checkpoint's weights reach the kernels as stored e4m3 bytes with one scale per 128x128 block, and its declared
dynamic per-token activation scaling is spelled in the graph, so these kernels compute the W8A8 form the checkpoint
declares rather than a dequantized f16 stand-in.

Summed over the 120 measured targets, the recorded kernels take **27.6 ms** against **119.2 ms** eager and
**6.7 ms** `torch.compile`. Emmy is 4.3x eager and 4.1x slower than `torch.compile`, and the gap is one family
(below), not a spread.

### Where tuning moved a kernel

Four targets were tuned by hand; `emmy tune` was not used. Each row is a deployable `-O3` measurement on this card.

| target | greedy | tuned | knobs | `torch.compile` | eager |
| --- | ---: | ---: | --- | ---: | ---: |
| `k_matmul_pointwise_bbd2dd` (GDN matmul) | 1,901 us | **16.4 us** | `WORK=t32x16,TILE=f4x4,RASTER=gm8` | 24.6 us | 36.9 us |
| `k_reshape_e4801c` (activation quantize) | 3,706 us | **158.5 us** | `WORK=,REDUCE=` (serial) | — | 604 us |
| `k_linear_pointwise_927765` (qkv projection) | 3,055 us | **1,976 us** | `WORK=w8x1,TILE=mma_m8n8k4_f16_f32/f4x4/k8,STAGE=d1/smem` | 863 us | 5,130 us |
| `k_linear_pointwise_3410af` (MLP projection) | 4,039 us | **1,991 us** | the same schedule | 1,320 us | 8,369 us |

The two projections take the same schedule, which is the shape the DeepSeek-V4 and EXL3 V100 goldens already favour
for a Volta `mma.sync` fold: eight warps down M, a 4x4 output fragment, k8, single-buffered shared memory. The
quantizer's greedy pick recomputes its 128-wide group maximum once per output element; the serial schedule computes
it once per group.

### What is wrong

**Nine kernels never finish 60 s of GPU time**, and each is the largest fusion on its path: both full-attention
blocks (`k_sdpa_linear_mean_reduce_68bd08`, `91af59` — one fused kernel holding about thirty `mma` sites), the fused
gate/up/down MLP (`k_linear_reduce_624aa0`, also with each of the two projection winners pinned), one piece of
`k_conv1d_linear_mean_reduce_e6909b`'s cut set at grid 3,145,728, and five matmul reduces of the Gated DeltaNet and
attention paths. Emmy fuses
a whole block here and the result does not run. This is the largest gap in the inventory and the reason the golden
holds 120 targets rather than 129.

**The Gated DeltaNet chunk family is 25-55x `torch.compile`** and does not respond to scheduling.
`torch.export` unrolls the delta rule's chunk loop, so chunk *k* carries O(*k*) work that eager amortizes. The worst
row, `k_slice_unsqueeze_reduce_d1044a`, is 4,843 us against 88 us. It offers eight single cuts; every one of them,
and every combination measured (all eight together, three composed sets, with and without a serial reduce), is slower
than the fused greedy, because the cut peels off a small kernel and the remainder keeps the 4.8 ms schedule. This
family is a fusion and code-generation gap, not a search shortfall, and it dominates the 4.1x aggregate above.

**The quantize kernel's greedy schedule is non-deterministic.** A second row realizing that same configuration
disagrees with its output, so `WORK=t128,REDUCE=coop` on this kernel has a race. The recorded row is the serial
schedule, which is also 23x faster.

**Strict eager correctness cannot gate a W8A8 kernel here.** The eager twin dequantizes the weights and computes in
f16, so it does not carry the checkpoint's own activation-quantization error: a synthetic block-FP8 linear traced
with `--quantize fp8-block` and benched under `--strict` disagrees with it on 261,803 of 262,144 elements. The rows
here are therefore recorded against each target's own greedy pick, as the sibling V100 goldens are. One separate
disagreement is unresolved: the quantizer differs from eager on 346 of 2.6M elements by one e4m3 step (index 145255,
288 against 320). It is not a rounding-mode difference — this card's `__nv_fp8_e4m3` conversion rounds 9.5 to 10
correctly — and it is not normalization's `amax * (1/448)` rewrite either: the same elements fail with that rewrite
removed.

### Reproduce

```bash
emmy trace Qwen/Qwen3.8-27B-FP8 --layer 0 --target sm_70 -o layer0.yaml     # and --layer 3
emmy run --golden recipes/Qwen3.8-27B-FP8/golden/v100_sm70.yaml --bench --bench-backends eager,tcompile,emmy
```

On a Volta host, install a torch build that still ships `sm_70` kernels (`torch==2.13.0+cu126`) and preload a CUDA 12
NVRTC, per the README's pre-Turing notes.

## Reproduce

```bash
emmy vm create cloudrift --instance-type v100-6-52-400-generic.4 --ssh-key ~/.ssh/id_ed25519.pub \
  --image-url https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpup-580-129-20260810-232733.img
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
