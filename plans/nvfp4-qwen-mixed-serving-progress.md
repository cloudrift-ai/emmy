# Qwen3.8 NVFP4 mixed serving: implementation and qualification

Status: implementation in draft PR #993, September 30, 2026. The exact 27B endpoint booted with pinned native
M=16 decode and M=64 prefill programs on the RTX 5090. Deterministic short and 4K completions matched stock.
The independent quantized MLP and broader numerical gates remain open.

## Boundary and target

The first serving lane replaces only the 64 dense text MLPs of vLLM 0.23's `Qwen3_5ForConditionalGeneration` with
Emmy programs. The stock model still constructs and executes attention, GDN and its state, projections outside the
MLP, norms, residuals, rotary embedding, embeddings, output head, and scheduling. Its non-MLP weights keep the
checkpoint's ModelOpt NVFP4 loader and quantization configuration. Emmy binds each MLP's 12 packed checkpoint
tensors to static M=16 decode and static M=64 prefill plans. Each uploads the active rows into its input prefix and
returns only those output rows. A GPU stale-row probe verified that changing decode's other 15 rows leaves the real
output unchanged. The lower-level MLP program still supports symbolic prefill when requested, so these interfaces
remain usable by a larger Emmy region later.

The qualification checkpoint is `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` on one RTX 5090. The contract is BF16, text only, TP1/PP1, one active request, 4,096 context tokens, at most 64 scheduled tokens, with prefix caching, speculation and outer CUDA graphs disabled. The earlier proposed FP16 boundary was rejected; no FP16 fallback is part of this lane.

## Evidence and current status

| Gate | Evidence | Status |
| --- | --- | --- |
| Stock baseline | Pinned vLLM 0.23 container loaded the exact checkpoint in BF16 with stock ModelOpt NVFP4 and FLA GDN. `/health` returned 200; deterministic text and the paired streaming benchmark succeeded. | Functional and small warm latency baseline passed; peak-load memory pending. |
| Capture and inventory | Exact checkpoint header inspection found `model.language_model.layers.*`; all 64 MLPs form one structural profile. The shared trace path captures BF16 W4A4 `mlp16@nvfp4` and `mlp64@nvfp4`; the symbolic option remains available in the helper. A pinned exact-checkpoint trace saved 2 graphs and 6 distinct kernels. | Focused CPU tests and trace passed; final release audit pending. |
| Shared BF16 boundaries | The input carrier, constant binding/cache, device output view, and NumPy graph interpreter had independent BF16 carrier bugs. Shared fixes and focused tests encode numerical BF16 as bits, decode it for arithmetic, and preserve raw bitcasts. | Focused CPU tests passed; broader regression gates pending. |
| Synthetic GPU MLP | Two tiny layers compiled with plan reuse. Widths 1, 2, 15, 16, 17, 63 and 64 returned finite, distinct BF16 outputs; a nondefault stream passed. | Execution smoke passed. Exact stock quantized MLP comparison currently fails. |
| Mixed full model | The stock constructor replaced all 64 MLP modules; all seven checkpoint shards loaded, then Emmy bound all 128 programs. Both M=16 with symbolic prefill and M=16 with M=64 native prefill booted and returned `/health` 200. The M=64 boot reported 23.43 GiB model memory and about 31.57 GiB resident GPU memory. | Pinned endpoint boots passed. |
| Real hybrid preservation | A two-layer actual vLLM 0.23 Qwen3.5 model with GDN and full-attention layers retained non-MLP module and parameter identities and hybrid state interfaces after Emmy replaced its MLPs. | Pinned-container constructor regression passed. |
| Full-size static MLP | Isolated layer-0 M=1 default compile yielded four launches and 150,410,260 bound weight bytes. Launch 0 took 0.393 ms. Uncaptured launch 1, `k_linear_reduce_93e407`, exceeded a 10,000 ms kernel watchdog. A corrected-quant M=16 program with scoped pins emits four native FP4 MMA kernels and completes. | Unpinned route blocked; isolated pinned native execution passed, endpoint speed pending. |
| Warm text completions | Stock, scalar mixed and native mixed returned the same 16-token Paris continuation. Native mixed also returned expected Japan then Paris continuations with warm wall times 1.895 and 1.877 s; the initial native request took 152 s while stock vLLM/Triton kernels compiled on first use. | Deterministic text and simple cross-request state smoke passed; first-request latency needs warmed-cache handling. |
| Single-request latency | Standard `vllm bench serve` with seed-42 random 5-input/16-output requests, one warmup, five measured, concurrency one, and ignore-EOS produced stock/scalar/M=16+symbolic/M=16+M=64 mean TTFT 288.16/617.07/390.51/271.86 ms and TPOT 110.10/444.49/100.27/101.32 ms. All 20 measured requests succeeded. | Final native route reached 9.87 decode tokens/s versus stock 9.08; TTFT was 16.3 ms lower than stock in this small warm run. This is a single request shape, not general throughput. |
| Teacher-forced prompt | Stock and native mixed returned the same 14 token IDs and expected Tokyo continuation for one fixed 13-token prompt. Selected-token logprob differences across 13 positions had mean absolute 0.0700, RMS 0.0980, maximum 0.2431; earlier scalar maximum was 0.281. | Token smoke passed; numerical logit parity remains open. |
| 4K context | The same 4,005 prompt tokens plus 16 generated tokens completed with identical stock and mixed text, 4,021 total tokens, HTTP 200, and no OOM. After a short first-use request, stock took 12.451 s, mixed symbolic scalar prefill took 224.22 s, and mixed native M=64 prefill took 11.547 s wall. | Full requested length envelope and long-prefill latency probe passed for one deterministic request. |
| End-to-end correctness | Multiple prompts, teacher-forced logits, hybrid state and cache behavior beyond the tested short requests. | Pending broader comparison. |

The stalled kernel source has grid 1 and block 256, but only one thread enters the body. It performs full-K scalar gate/up reductions for output codes and again for per-block scales, with no native FP4 MMA. At the actual layer width, nested loops amount to more than one billion serial scalar operations with repeated FP4 decode reads. The generated source and boot logs are retained in the RTX 5090 qualification workspace. This was an **unpinned default schedule**. The user has since clarified that prior tuning is broken and every Emmy compilation and serving run must use explicit knob pins through a golden or `EMMY_KNOBS`. The timeout is evidence about the default route; the pinned route boots and serves but is slow.

### Explicit knob experiments on the RTX 5090

The earlier endpoint's simple scalar route used this exact `EMMY_KNOBS` value for both capture shapes. It remains
the explicit shared baseline for the M=16 decode and symbolic prefill programs:

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
| Same simple pins, experimental static M=16 padding | An isolated full-size M=16 program compiled eight kernels. Its gate/up Tile IR retains the token/output axis pair, recognizes the packed FP4 operands, and lists the native FP4 atom. | Native eligibility established; this first probe preceded the adapter change. |
| Static M=16 with the same cuts, `FAST_MATH=false`, and scoped native FP4 pins below | Eight kernels compiled; four contraction kernels emitted `mma.sync`. The padded M=1 execution returned finite BF16 output. Against the exact stock MLP, relative RMS error was 2.36543%, essentially the scalar route's 2.371%. | Usable native schedule evidence, but no numerical qualification or measured endpoint speedup. Do not promote the padded route yet. |
| Corrected quant graph, M=16, scoped `WORK/TILE/STAGE` pins in the checked-in recipe | Eight kernels compiled, with three gate/up and one down contraction emitting `mma.sync`. The exact layer-0 stock MLP comparison gave 1.1937% relative RMS error. Different contents in the other 15 padded rows left the real output exactly unchanged. | Native execution and padding independence passed; full endpoint speed and numerical qualification pending. |
| Same scoped native pins in a single global `EMMY_KNOBS` value for both programs | Full 27B boot failed while compiling symbolic prefill: `STAGE pin 'd2/smem-async' does not resolve for this contraction`. The two program shapes can share a kernel identity but offer different schedules. | Unusable global combination. Shared baseline pins remain in `EMMY_KNOBS`; native overrides are applied only around static compilation via `EMMY_MLP_STATIC_KNOBS`. |
| Corrected quant graph, static M=64 with the old M=16 place identities | Eight kernels compiled, but only the down contraction emitted `mma.sync`; the gate/up override identities no longer matched. | A pin's presence is not evidence it selected a native kernel; inspect emitted source for each shape. |
| Static M=64, same shared cuts and scoped current-piece pins in the recipe | Eight kernels compiled, with all three gate/up and down contractions emitting `mma.sync`. A full 64-row same-checkpoint stock MLP comparison gave 0.9946% relative RMS, mean absolute 0.0121, p99 absolute 0.0453. The full endpoint took 11.547 s on a 4,005+16 request versus stock 12.451 s. | Native execution and endpoint latency passed on measured shapes; zero-tolerance diagnostic still fails. |

The first M=16 native probe used these overrides before the quant graph changed. Its place identities are stale and
must not be used for the current model graph:

```text
WORK@place_35acacc227=w1x1,TILE@place_35acacc227=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_35acacc227=d2/smem-async,WORK@place_4f08570ba8=w1x1,TILE@place_4f08570ba8=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_4f08570ba8=d2/smem-async,WORK@place_159d5b6179=w1x2,TILE@place_159d5b6179=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_159d5b6179=d2/smem-async,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async
```

The current M=16 and M=64 native overrides are in
[`scripts/serve_qwen38_nvfp4_mixed_5090.sh`](../scripts/serve_qwen38_nvfp4_mixed_5090.sh):

```text
WORK@place_6b4be893d5=w1x2,TILE@place_6b4be893d5=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_6b4be893d5=d2/smem-async
WORK@place_743937bec0=w1x1,TILE@place_743937bec0=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_743937bec0=d2/smem-async
WORK@place_b5f468b49f=w1x1,TILE@place_b5f468b49f=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_b5f468b49f=d2/smem-async
WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async
```

The M=64 prefill pins use the same base `EMMY_KNOBS` and the following
`EMMY_MLP_PREFILL_KNOBS` values; each is scoped to this program's compile:

```text
WORK@place_66b5682eed=w1x2,TILE@place_66b5682eed=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_66b5682eed=d2/smem-async
WORK@place_2cedf62283=w1x1,TILE@place_2cedf62283=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_2cedf62283=d2/smem-async
WORK@place_2c71f28601=w1x1,TILE@place_2c71f28601=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_2c71f28601=d2/smem-async
WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async
```

The five projection-cut candidates came from the earlier `fp4-encode-recomputes-producer.md` report. Its native
FP4 and async-stage suggestions are historical starting points, not measured 5090 wins. The current compiler
supports `@place_<token>` pins for an individual cut piece; that selector was not demonstrated by the older report's
`@n0` and full-name experiments. The M=1 refusal above is structural for these cuts, not an ignored pin. An
identity-keyed golden may route a parent M=1 cut followed by a child cut that retains the unit row; existing cut-fork
tests cover the mechanism, but exact MLP replay remains untested. An environment first-cut pin marks placement
decided and cannot express this sequence. The scalar route has repeated endpoint probes; numerical qualification
remains pending.

An independent layer-0 stock vLLM `Qwen3NextMLP`/ModelOpt NVFP4 oracle compared the same packed checkpoint
tensors and BF16 inputs at M=1, 2 and 64. Before the quant spelling correction, relative RMS output errors were
2.37%, 2.32% and 2.49%. The original M=16 native probe remained near 2.37%, so native contraction alone did not
remove that error. The exact checkpoint has equal gate/up input and weight global scales within every layer.

A per-launch snapshot avoids scratch-buffer reuse and compares the same layer-0 activation before gate/up. The
old Emmy quantizer differed from stock in 9 of 320 E4M3 block-scale bytes and 36 of 2,560 packed FP4 bytes.
The exact vLLM 0.23 CUDA quantizer computes the block scale as `(amax * f32(1/6)) * f32(1/input_scale)` and
multiplies BF16 input in FP32 by an approximate reciprocal of the decoded block scale times the recovered global
scale. Emmy previously rounded a fused scale through FP16 before choosing FP4 codes. A generic bound reciprocal
load operation now derives the per-layer inverse without embedding layer values in the graph; the spelled divisor
stays FP32. With this correction the initial activation has equal 320/320 E4M3 bytes and 2/2560 different packed
code bytes. Raw reconstructed activation relative RMS fell to 0.3967%, gate/up projection relative RMS to
0.462%/0.458%, and full scalar MLP relative RMS to 1.1689%. The corrected M=16 all-native MLP has 1.1937%
relative RMS. These measured residuals remain under investigation; the zero-tolerance diagnostic still fails.

The corrected down-projection input differs from stock in 58/1088 E4M3 bytes and 373/8704 packed FP4 bytes,
with 4.1487% raw reconstructed relative RMS. This accumulates prior projection and activation rounding and is
amplified by another FP4 quantization. Stock-byte injection at the first activation producer lowers gate/up
projection RMS to 0.1025%/0.0984%, which localizes most of the initial projection difference to two FP4 threshold
flips. Stock uses approximate reciprocal instructions; Emmy currently uses exact FP32 divide at those thresholds.
Further correction should be justified by full-model quality evidence rather than bit identity alone.

An earlier two-layer synthetic build exited 139 once during compilation. Later full redirected and real 27B builds completed, so the cause is unestablished. It remains a cold-build stability risk until repeated boots or a native trace resolve it. The synthetic NumPy graph and GPU outputs are finite after the BF16 fix but differ at some BF16/W4A4 boundaries; that graph interpretation alone is not the independent stock quantized MLP oracle.

## Next qualification work

1. Verify per-request state under the chosen M<=64 schedule and run the final release audit and repository gates.
2. Decide whether the 0.99–1.19% same-checkpoint layer-0 MLP relative RMS residual affects end-to-end output.
   The stock quantized MLP remains the independent numerical oracle; do not relax tolerance to pass a test.
3. Measure memory and major latency blockers on the exact RTX 5090 envelope. Run final repository test and lint
   gates, record commands and versions, and update the draft PR with measured results and remaining limits.

## Relation to the earlier investigation

[`nvfp4-qwen-performance.md`](nvfp4-qwen-performance.md) is the September 28 investigation, not a completed implementation plan. Its [`fp4-encode-recomputes-producer.md`](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md) report is directly relevant: the actual M=1 gate/up encode kernel repeats full-K work. This PR's isolated 5090 watchdog and generated CUDA give new evidence for the mixed lane; the report's 5080 timings and proposed cut are not assumed to transfer. Emmy's GDN #973 support is not required by this MLP boundary because stock vLLM owns GDN, but that Emmy path remains available. The investigation's LUT, TMA and native GDN ideas remain separate performance work. The later review-only mixed plan proposed FP16, which the user rejected in favor of the checkpoint's BF16; that review branch is not part of this PR.
