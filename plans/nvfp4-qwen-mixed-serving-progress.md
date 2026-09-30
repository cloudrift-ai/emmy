# Qwen3.8 NVFP4 mixed serving: implementation and qualification

Status: implementation in draft PR #993, September 30, 2026. The exact 27B endpoint is healthy with the explicit
scalar pins below and returned stock-matching deterministic completions. Broader correctness and performance
qualification remain open.

## Boundary and target

The first serving lane replaces only the 64 dense text MLPs of vLLM 0.23's `Qwen3_5ForConditionalGeneration` with Emmy programs. The stock model still constructs and executes attention, GDN and its state, projections outside the MLP, norms, residuals, rotary embedding, embeddings, output head, and scheduling. Its non-MLP weights keep the checkpoint's ModelOpt NVFP4 loader and quantization configuration. Emmy takes the exact 12 packed checkpoint tensors per MLP layer and binds per-layer constants to shared static M=1 and symbolic M<=64 plans. These lower-level compilation and binding interfaces remain usable by a larger Emmy region later.

The qualification checkpoint is `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` on one RTX 5090. The contract is BF16, text only, TP1/PP1, one active request, 4,096 context tokens, at most 64 scheduled tokens, with prefix caching, speculation and outer CUDA graphs disabled. The earlier proposed FP16 boundary was rejected; no FP16 fallback is part of this lane.

## Evidence and current status

| Gate | Evidence | Status |
| --- | --- | --- |
| Stock baseline | Pinned vLLM 0.23 container loaded the exact checkpoint in BF16 with stock ModelOpt NVFP4 and FLA GDN. `/health` returned 200; deterministic text and the paired streaming benchmark succeeded. | Functional and small warm latency baseline passed; peak-load memory pending. |
| Capture and inventory | Exact checkpoint header inspection found `model.language_model.layers.*`; all 64 MLPs form one structural profile. The shared trace path captures two 148-node BF16 W4A4 graphs, `mlp1@nvfp4` and `mlp-sym@nvfp4`, and trace/audit checks exact identity. | Passed focused CPU tests. |
| Shared BF16 boundaries | The input carrier, constant binding/cache, device output view, and NumPy graph interpreter had independent BF16 carrier bugs. Shared fixes and focused tests encode numerical BF16 as bits, decode it for arithmetic, and preserve raw bitcasts. | Focused CPU tests passed; broader regression gates pending. |
| Synthetic GPU MLP | Two tiny layers compiled with plan reuse. Widths 1, 2, 15, 16, 17, 63 and 64 returned finite, distinct BF16 outputs; a nondefault stream passed. | Execution smoke passed. Exact stock quantized MLP comparison currently fails. |
| Mixed full model | The stock constructor replaced all 64 MLP modules; all seven checkpoint shards loaded, then Emmy bound all 128 programs. Under explicit pins, vLLM reported 4.97 GiB available KV memory and a 59,392-token KV cache; `/health` returned 200. | Pinned endpoint boot passed. |
| Real hybrid preservation | A two-layer actual vLLM 0.23 Qwen3.5 model with GDN and full-attention layers retained non-MLP module and parameter identities and hybrid state interfaces after Emmy replaced its MLPs. | Pinned-container constructor regression passed. |
| Full-size static MLP | Isolated layer-0 M=1 default compile yielded four launches and 150,410,260 bound weight bytes. Launch 0 took 0.393 ms. Uncaptured launch 1, `k_linear_reduce_93e407`, exceeded a 10,000 ms kernel watchdog. The explicitly pinned scalar route completed all launches and capture. | Unpinned default route blocked; pinned execution passed a smoke probe. |
| Warm text completions | Both pinned stock and mixed engines returned ` Paris.\nThe capital of Germany is Berlin.\nThe capital of Italy is` for `The capital of France is`, temperature 0, max 16 tokens, with prompt 5/completion 16. After warmup, mixed Paris, Japan, then Paris requests returned stable, expected text in 7.35, 7.30, and 7.30 s respectively. | Deterministic text and simple cross-request state smoke passed. |
| Single-request latency | The same warm 5-input/16-output Paris completion took 1.933 s stock versus 7.300 s mixed. Standard `vllm bench serve` with identical seed-42 random 5-input/16-output requests, one warmup, five measured, concurrency one, and ignore-EOS produced stock/mixed mean TTFT 288.16/617.07 ms, TPOT 110.10/444.49 ms, and end-to-end 1939.60/7284.45 ms. All requests succeeded. | The explicitly pinned scalar route has about 4.04 times stock decode TPOT; further bounded schedule work is needed. |
| Teacher-forced prompt | Stock and mixed returned the same 14 echo tokens and the expected Tokyo continuation for one fixed 13-token prompt. Selected-token prompt log probabilities differed by up to 0.281 in the 13 comparable positions. | Token smoke passed; numerical logit parity remains open. |
| End-to-end correctness | Multiple prompts, teacher-forced logits, request lengths, 4K context, hybrid state and cache behavior. | Pending broader comparison. |

The stalled kernel source has grid 1 and block 256, but only one thread enters the body. It performs full-K scalar gate/up reductions for output codes and again for per-block scales, with no native FP4 MMA. At the actual layer width, nested loops amount to more than one billion serial scalar operations with repeated FP4 decode reads. The generated source and boot logs are retained in the RTX 5090 qualification workspace. This was an **unpinned default schedule**. The user has since clarified that prior tuning is broken and every Emmy compilation and serving run must use explicit knob pins through a golden or `EMMY_KNOBS`. The timeout is evidence about the default route; the pinned route boots and serves but is slow.

### Explicit knob experiments on the RTX 5090

The current simple scalar route uses this exact `EMMY_KNOBS` value for both capture shapes:

```text
FAST_MATH=false,PLACE@map.1/map=cut,PLACE@map.1/map.2/inner=cut,PLACE@map.1/map.3/reduce.1/inner=cut,PLACE@map.2/map=cut,PLACE@map.2/map.2/reduce.1/inner=cut,WORK=w1x4,TILE=,STAGE=,REDUCE=,RASTER=
```

| Experiment | Actual 27B layer-0 outcome | Decision |
| --- | --- | --- |
| No pins, static M=1 | Four kernels; gate/up encode selected a grid-1 scalar kernel that exceeded a 10,000 ms watchdog. | Invalid deployment route; prior tuning is known broken. |
| Five `PLACE` cuts above, `FAST_MATH=false`, `TILE=mma_m16n8k64_e2m1_f32/f1x2/k8`, global `STAGE=d3/smem-async` | Compile rejected `STAGE pin 'd3/smem-async' does not resolve for this contraction`. | The global stage fails for at least one contraction; its precise scope or lowering cause is unestablished. Use an explicit stage-off route meanwhile. |
| Same cuts and native `TILE`, no global `STAGE` | Compiled nine parallel kernels, but none used an FP4 MMA instruction. | The native tile pin did not yield the requested instruction at M=1; do not claim native FP4 execution from the pin. |
| Same cuts, native `TILE`, `WORK=w1x2`, `STAGE=` off | The static M=1 attempt emitted eight scalar Tile kernels and timed out after 160 s while the diagnostic was still running. Native FP4 requires an async stage. | Invalid native combination; no native speed or numerical result. |
| Same cuts, M=1 strict `WORK/TILE/STAGE@place_36b2b31bd6` with `w1x2` and `w1x1`, native `/f1x2/k4`, `d2/smem-async` | Both scoped pins correctly reached the named gate/up piece and raised `its kernel pins ... leave no schedule row this kernel offers`. Exact Tile IR has a packed pair and the FP4 atom, but `contracts=False`: both free axes are weight/output axes after the token M=1 axis collapsed. | No M=1 native tensor-core schedule for this cut piece; varying WORK or STAGE cannot restore the missing M axis. |
| Full simple pin string above, static M=1 | Compiled eight parallel scalar kernels in 12.57 s. Every launch completed under the 10 s watchdog, graph capture and replay succeeded, and BF16 output was finite. Per-launch GPU times summed to about 7.19 ms. | Correctness and endpoint tests needed; timing is a single-layer diagnostic, not serving latency. |
| Full simple pin string above, symbolic M=64 | Compiled eight parallel scalar kernels in 12.79 s. Every launch, capture and replay completed with finite BF16 output. Per-launch GPU times summed to about 56.89 ms. | Correctness and endpoint tests needed; this is a working route for the initial hand sweep. |
| Same simple pins, experimental static M=16 padding | An isolated full-size M=16 program compiled eight kernels. Its gate/up Tile IR retains the token/output axis pair, recognizes the packed FP4 operands, and lists the native FP4 atom. | Native schedule eligibility established; the adapter still runs static M=1. |
| Static M=16 with the same cuts, `FAST_MATH=false`, and scoped native FP4 pins below | Eight kernels compiled; four contraction kernels emitted `mma.sync`. The padded M=1 execution returned finite BF16 output. Against the exact stock MLP, relative RMS error was 2.36543%, essentially the scalar route's 2.371%. | Usable native schedule evidence, but no numerical qualification or measured endpoint speedup. Do not promote the padded route yet. |

The M=16 native probe added these scoped overrides to the explicit scalar baseline:

```text
WORK@place_35acacc227=w1x1,TILE@place_35acacc227=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_35acacc227=d2/smem-async,WORK@place_4f08570ba8=w1x1,TILE@place_4f08570ba8=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_4f08570ba8=d2/smem-async,WORK@place_159d5b6179=w1x2,TILE@place_159d5b6179=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_159d5b6179=d2/smem-async,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async
```

The five projection-cut candidates came from the earlier `fp4-encode-recomputes-producer.md` report. Its native
FP4 and async-stage suggestions are historical starting points, not measured 5090 wins. The current compiler
supports `@place_<token>` pins for an individual cut piece; that selector was not demonstrated by the older report's
`@n0` and full-name experiments. The M=1 refusal above is structural for these cuts, not an ignored pin. An
identity-keyed golden may route a parent M=1 cut followed by a child cut that retains the unit row; existing cut-fork
tests cover the mechanism, but exact MLP replay remains untested. An environment first-cut pin marks placement
decided and cannot express this sequence. The scalar route has repeated endpoint probes; numerical qualification
remains pending.

An independent layer-0 stock vLLM `Qwen3NextMLP`/ModelOpt NVFP4 oracle compared the same packed checkpoint tensors and BF16 inputs at M=1, 2 and 64. Both sides produced finite, nonzero outputs. Relative RMS errors were 2.37%, 2.32% and 2.49%; mean absolute errors were 0.0309, 0.0299 and 0.0321; p99 absolute errors were 0.100, 0.098 and 0.105. Maximum absolute differences were 0.5, 1.0 and 1.0, against stock maximum magnitudes 112, 119.5 and 140. Exact BF16 elementwise equality was about 2%, and over 84% of values differed by at least four BF16 ULPs. The zero-tolerance diagnostic fails; these errors are too substantial to dismiss as a rounding tolerance. Emmy's scalar decode rounds scaled FP4 operands through FP16 then BF16, while the stock CUTLASS path carries raw FP4 block scales and a global FP32 alpha; whether that fully explains the divergence is unestablished. The exact checkpoint has equal gate/up input and weight global scales within every one of its 64 layers.

A per-launch snapshot avoids scratch-buffer reuse and compares the same layer-0 M=1 activation before the gate/up
contractions. Emmy and stock differ in 9 of 320 E4M3 block-scale bytes and 36 of 2,560 packed FP4 code bytes.
The exact vLLM 0.23 CUDA quantizer computes FP32 approximate reciprocal scaling before FP4 conversion, whereas
Emmy's spelled activation path rounds the decoded E4M3 scale times the global scale through FP16 before dividing.
This establishes a real input-quantization difference; its contribution to the 2.37% MLP output error still needs
a corrected-spelling rerun. The earlier comparison of intermediate buffers after the entire program was invalid
because scratch storage can be reused; only producer-launch snapshots are reported here.

An earlier two-layer synthetic build exited 139 once during compilation. Later full redirected and real 27B builds completed, so the cause is unestablished. It remains a cold-build stability risk until repeated boots or a native trace resolve it. The synthetic NumPy graph and GPU outputs are finite after the BF16 fix but differ at some BF16/W4A4 boundaries; that graph interpretation alone is not the independent stock quantized MLP oracle.

## Next qualification work

1. Investigate the measured 2.3–2.5% same-checkpoint MLP relative RMS error under `FAST_MATH=false`; the stock quantized MLP remains the independent numerical oracle. Compare more teacher-forced logits and deterministic prompts, including hybrid state across requests and the 4K context envelope.
2. Localize the discrepancy in activation FP4 codes and scales, projection output, and the down-projection input. The all-native M=16 probe did not materially change error, so scheduling alone is insufficient. Keep the legal native pins for a speed test once output semantics agree. The 7.19 ms and 56.89 ms single-layer GPU launch sums above are diagnostic, not endpoint throughput.
3. Measure memory and meaningful latency on the exact RTX 5090 envelope; investigate repeated full contractions and encode work only when reproduced. Run the repository's final test and lint gates, record exact versions and commands, and update the draft PR with measured results and remaining limits.

## Relation to the earlier investigation

[`nvfp4-qwen-performance.md`](nvfp4-qwen-performance.md) is the September 28 investigation, not a completed implementation plan. Its [`fp4-encode-recomputes-producer.md`](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md) report is directly relevant: the actual M=1 gate/up encode kernel repeats full-K work. This PR's isolated 5090 watchdog and generated CUDA give new evidence for the mixed lane; the report's 5080 timings and proposed cut are not assumed to transfer. Emmy's GDN #973 support is not required by this MLP boundary because stock vLLM owns GDN, but that Emmy path remains available. The investigation's LUT, TMA and native GDN ideas remain separate performance work. The later review-only mixed plan proposed FP16, which the user rejected in favor of the checkpoint's BF16; that review branch is not part of this PR.
