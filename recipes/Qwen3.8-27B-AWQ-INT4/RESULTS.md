# Qwen3.8-27B W4A16 on one RTX 4090

Status: serving-qualified with stock vLLM on the text-only path at an 8,192-token context. Emmy remains **ineligible**
for this checkpoint: exact compiler coverage is complete, but strict correctness and attention performance still
block a canonical golden file, so this recipe has no Emmy lane and no comparison column.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `philbert440/Qwen3.8-27B-W4A16-AWQ` |
| Model revision | `7908d42a71077a5e4dc458f273682b12dfe384a0` |
| Base checkpoint | `Qwen/Qwen3.8-27B` (W4A16 `compressed-tensors`, pack-quantized, group size 128) |
| Hardware | 1x NVIDIA GeForce RTX 4090, compute capability 8.9 |
| Driver | 580.159.03 |
| Engine | stock vLLM `0.1.dev19754+g3a0914114` (2026-08-19 nightly) |
| Image | `vllm/vllm-openai@sha256:dae7af23ea9b66b4f15de3d5e4ddebfdafa7be636be91d400184c1666f1b1462` |
| Serving shape | TP1, context 8,192, text-only, prefix caching off |
| Repository revision | `a20b10790824a04d195c707d0dda3d8fa5e1cf68` |

Weights occupy 20.7 GiB of the 24 GiB card. That is high for a 4-bit 27B because this checkpoint leaves
`embed_tokens` and `lm_head` unquantized and untied, at 248,320 x 5,120 each — about 5.1 GiB before any transformer
weight is counted. The remaining KV pool measures 2.68 GiB, or 32,182 tokens, which is what caps the context.

## Serving performance

Measured on the recipe's exact engine and serving shape: greedy decoding, ignored EOS, seed 0, no prefix caching.
All rows completed with zero failed requests.

| Input / output tokens | Concurrency | Output throughput | Median TTFT | Median TPOT | Duration |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 128 / 128 | 4 | 172.1 tokens/s | 313 ms | 21.0 ms | 23.8 s |
| 1,024 / 256 | 4 | 150.3 tokens/s | 1,316 ms | 21.6 ms | 27.2 s |
| 4,096 / 512 | 2 | 76.4 tokens/s | 2,294 ms | 21.7 ms | 53.6 s |

Decode cost is essentially flat at ~21 ms per token across all three points, so on this card the model is
weight-streaming-bound in decode and the differences between points are prefill and batching effects. The recipe keeps
concurrency 4, which is the largest setting the 32,182-token KV pool supports at full 8,192-token sequences.

## Validated capabilities

- **Context 8,192** — verified by retrieving a planted marker from a 7,768-token prompt, not from startup alone. The
  native 262,144-token context does not fit this card at any concurrency.
- **Tool calling** — returns structured `tool_calls` with the correct function name and arguments, and
  `finish_reason: tool_calls`, using `--tool-call-parser qwen3_coder`.
- **Thinking toggle** — `enable_thinking: false` suppresses reasoning cleanly (`reasoning_tokens: 0`); the default
  thinking path produces correct answers.
- **Text-only** — qualified with `--language-model-only`. This is an `any-to-any` checkpoint, but the vision tower is
  unquantized in every available W4A16 build and does not fit alongside a usable KV pool.

### Known limitation: reasoning content is not surfaced

With `--reasoning-parser qwen3`, thinking is correctly stripped from `content` and counted in
`completion_tokens_details.reasoning_tokens`, but `reasoning_content` is returned empty on every request. The chat
template does pre-fill `<think>\n`, which is the layout the parser expects, and no request-level flag changes the
result. No Qwen3.8-specific reasoning parser exists in this image. Clients that need the reasoning text should not
rely on this field until an upstream parser lands.

## Emmy eligibility: coverage complete, qualification blocked

The compiler inventory was re-traced from repository revision
`94e2a74a2b9359699701847b28839f20b20a1de0` (tree
`9a0866098b896dfb82e32c521e1a120e5fc1e643`) on the qualified RTX 4090. It covers seven representative programs and
227 golden configurations, including both kinds of decoder layer and the distinct MTP paths:

| Program | Model coverage | Configurations | Working-golden SHA-256 |
| --- | --- | ---: | --- |
| decoder layer 0 — Gated DeltaNet | 48 `linear_attention` layers | 196 | `4bd3e5bfe476aa43304de335886101e114e7344ebb6436aa0e018117bacde0d4` |
| decoder layer 3 — full attention | 16 `full_attention` layers | 13 | `30f1b77534e87c55f134318632147087bc2a41254f2fb42f1664f2fb6e948929` |
| final `RMSNorm` | final normalization | 1 | `d8486bf41bc248fb7a8a4a9a34b3939979dfa880435152a9824d3ff707e9be37` |
| `embed_tokens` | input embedding | 1 | `bda686565e67ce156888c98bc64059971478ed31147504d7dbc3a2d6e2e7b488` |
| `lm_head` | output projection | 1 | `571ac07a9fef3f71a387bb40478197f9d86eef1461bb68cb7114eb165410d071` |
| MTP full block | full MTP path | 13 | `75d06475d3ea8d4b68ad7590130122a24319c58e217798761a803e292a8ccb9e` |
| MTP pre-FC2 block | distinct pre-FC2 path | 2 | `0d405f6bc4b17841b2258b6ceddd72f440c08a9b6674a683a88e9c35b452078b` |

These are working golden files, not deployment evidence. Two gates prevent promotion to a canonical golden file:

- The final normalization has 14 of 2,621,440 BF16 outputs outside the unchanged strict tolerance. Each differs by
  one BF16 ULP; the maximum absolute error is 0.0078125. This remains a correctness failure.
- For full-attention target `k_sdpa_reduce_fd92e2.e496506297b2`, the fastest strict schedule measured at deployable
  `-O3` is `WORK=w2x2`, `TILE=mma_m16n8k16_f16_f32/f4x8/k2`, `STAGE=d1/smem`, `RASTER=gm8`. Two repeated replays
  measured 191.283 and 191.898 us against 61.850 and 62.240 us for eager PyTorch: 3.09x slower by the medians, with
  maximum absolute error 0.0009765625 and CUDA source SHA-256
  `f22aba2cb34fc1c6fa267ff8f5837947ff974489c560a586c854f9ff27c7f10d`.

Under equal cold search budgets, hybrid tuning found a 1.93x faster complete attention candidate than MCTS-only in
the `-Xcicc -O1` tuning regime. Repeated `-O3` replay still exposes the deploy gap above, so that search improvement
is not a serving speedup. Profiling a strict k2 kernel attributes the gap to 14.9 million load/store instructions,
255 registers per thread, and 3.4% DRAM throughput; schedule-neighbor, staging, raster, and fast-exponential sweeps did
not close it.

Until strict correctness passes across the inventory and attention reaches parity, this recipe has no Emmy serving
lane on the RTX 4090.

## Tesla V100 (sm_70): 4-bit pays in decode and costs in prefill

`golden/v100_sm70.json` holds 74 measured rows over 16 targets of decoder layer 0, recorded on one Tesla
V100-SXM3-32GB. Every row carries a strict correctness proof against its reference; none is timed-only.

This golden is over a program that **carries the coded weights**. Emmy reads this checkpoint's
compressed-tensors `pack-quantized` int4 directly, so the traced graph holds the unpack and dequantize
algebra — 26 `right_shift` and 26 `bitwise_and` operations over `i32` weight tensors — and the six
quantized projections decode inside their own kernels rather than being handed dense FP16 weights.

### The crossover

The same logical GEMM, the Gated DeltaNet input projection at N=10,240 and K=5,120, measured as a coded
W4A16 program and as a dense FP16 one:

| | M=64 | M=512 |
| --- | ---: | ---: |
| Emmy W4A16 (coded) | **155.8 us** | 1,546 us |
| Emmy dense FP16 | 196.4 us | 520 us |
| cuBLAS FP16 (eager) | 276.5 us | 536 us |
| 4-bit against Emmy dense | **1.26x faster** | 2.97x slower |
| 4-bit against cuBLAS | **1.77x faster** | 2.86x slower |

At M=64 the dense kernel moves its 105 MiB of weights at 536 GB/s of the card's 900, so it is
weight-bandwidth-bound and streaming a quarter of the bytes wins even though the decode is not free. At
M=512 the same GEMM is compute-bound, the weight traffic is amortized over eight times the rows, and the
decode is pure overhead. The crossover therefore lies between M=64 and M=512.

Both coded rows are bit-exact: maximum absolute and maximum relative error 0.0 against eager.

For a deployer the statement is: route decode through Emmy's 4-bit path and prefill elsewhere. That
matches how this model is served, where per-token decode dominates the time.

### Three bounds on that statement

**The pair is best-available-Emmy at each shape, not a controlled tile sweep.** At M=512 the coded kernel
is `WORK=w2x2, TILE=mma_m8n8k4_f16_f32/f4x4/k8, STAGE=d2/smem`; at M=64 the winner is the unscheduled
per-cell tier. The comparison is what a deploy would actually pick at each shape, which is the useful
question, but it is not one tile measured twice.

**M=1 cannot be traced for this architecture at all.** `emmy trace --layer 0 --seq-len 1` fails with
`aten.pad supports only explicit zero-width padding, got [0, 0, 0, 63]`: the Gated DeltaNet pads its
sequence up to the chunk width of 64, and the tracer rejects non-zero-width padding. No shape below 64
is reachable for 48 of this model's 64 layers — precisely the regime where 4-bit pays most. M=64 is the
floor of the evidence here, and supporting that one `aten.pad` case would unlock M=1 for every Gated
DeltaNet checkpoint.

**A 17.0x figure appears in the raw records and must not be quoted.** Against the coded program eager
measures 26,217 us, because torch executes the 26-operation unpack chain as separate tensor operations
where Emmy fuses it into the GEMM. That is an upper bound on torch's eager cost and nothing more; no
kernel anyone would deploy is that slow. The deployment-relevant baseline is cuBLAS FP16 at the same
shape, which is what the table above uses.

### The Gated DeltaNet chunk recurrence

Fifteen `k_slice_unsqueeze_reduce_*` chunk targets carry the delta-rule recurrence. Each elects a
single-CTA arm — grid 1 for the whole reduce — and twelve of the fifteen then cannot finish inside the
launch watchdog at all. A placement cut at `PLACE@map.1/reduce` restores a grid and takes the set from
unrunnable to 57,051 us eager against 31,373 us, **1.82x in aggregate**, every row strict-verified.

| chunk | eager | with the cut | ratio |
| --- | ---: | ---: | ---: |
| first | 3,842 us | 186 us | 20.64x |
| median | 4,034 us | 993 us | 4.06x |
| last three | 4,557 / 4,629 / 3,265 us | 4,577 / 5,456 / 5,973 us | 1.00x / 0.85x / 0.55x |

Per-chunk cost rises monotonically with chunk index where eager stays flat, because the carried state is
re-derived per chunk rather than carried: `torch.export` unrolls the chunk loop and fusion pulls each
chunk's history into its target. The cut removes the serialization, not that recompute, which is why the
last three chunks still sit at or below eager. That residual is the traced program's volume and is
upstream of scheduling.

These targets are unaffected by the weight coding — 186 us coded against 188 us dense on the first chunk —
because they contract activations rather than weights. That localizes the entire decode cost to the six
quantized projections, which is what makes the 2.97x actionable rather than diffuse.

### What still blocks a serving lane

Neither full-attention target runs: both exceed a 60 s launch watchdog, so decoder layer 3 has no rows.
Their cost is a nested reduce of 805,306,368 serial trips per cell in which the attention score cone,
which reads only the head index and never the head-dim component, is re-derived once per each of the 256
components, twice over. Every free axis is distributed — grid equals free on both kernels — so this is a
cone nothing hoists out of the loops enclosing it, not a grid that failed to spread.

The final normalization's strict-correctness failure is unchanged: 14 of 2,621,440 BF16 outputs one ULP
outside tolerance.

## Reproduce

```bash
emmy bench experiments/Qwen3.8-27B-AWQ-INT4/serving --ssh USER@HOST
```

## Limits

- The image is a dated nightly, not a release tag. Re-resolve it before any new performance run, and re-check whether
  a release tag has superseded it.
- The 24 GB card holds 8,192 tokens of context at concurrency 4. Raising either requires a build that quantizes
  `embed_tokens`/`lm_head`, which none of the current community W4A16 repositories do.
- `reasoning_content` is unusable; see the limitation above.
- Multimodal input is out of scope for this platform and untested here.
