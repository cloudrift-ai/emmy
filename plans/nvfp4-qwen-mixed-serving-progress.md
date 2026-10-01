# Qwen3.8 NVFP4 mixed serving: implementation and qualification

Status: implementation in draft PR #993, October 1, 2026. The exact 27B endpoint booted with pinned native
M=16 decode and M=64 prefill programs on the RTX 5090 before the October 1 main merge. Selected deterministic
short and 4K completions matched stock; four other 64-token prompts diverged. Post-merge graph fusion changed
the pin sites. New explicit cuts yield three kernels emitting native FP4 instructions per shape on the 5090.
Post-merge numerical, endpoint, bounded answer-quality, and paired warm latency checks passed; the broader
quality limits and historical divergent continuations are recorded below.

## Boundary and target

The first serving lane replaces only the 64 dense text MLPs of vLLM 0.23's `Qwen3_5ForConditionalGeneration` with
Emmy programs. The stock model still constructs and executes attention, GDN and its state, projections outside the
MLP, norms, residuals, rotary embedding, embeddings, output head, and scheduling. Its non-MLP weights keep the
checkpoint's ModelOpt NVFP4 loader and quantization configuration. Emmy binds each MLP's 12 packed checkpoint
tensors to static M=16 decode and static M=64 prefill plans. Each uploads the active rows into its input prefix and
returns only those output rows. A GPU stale-row probe verified that changing decode's other 15 rows leaves the real
output unchanged. The lower-level MLP program still supports symbolic prefill when requested, so these interfaces
remain usable by a larger Emmy region later.

The qualification checkpoint is
`Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` on one RTX 5090. The contract
is BF16, text only, TP1/PP1, one active request, 4,096 context tokens, at most 64 scheduled tokens, with prefix
caching, speculation and outer CUDA graphs disabled. The earlier proposed FP16 boundary was rejected; no FP16
fallback is part of this lane.

## Pre-merge evidence and current post-merge status

The warm endpoint and numerical measurements in the table below were taken before merging main and are historical.
The merged branch's portable GitHub CI passed all native, lint, package, and full-test jobs at `65b9591c` and
again at code commit `4b71026c` after the nested pin fix and refreshed recipe (full test job 24m11s).
Locally, the new child-site placement tests pass; the broader local Nix test run has environment failures, including
two exact CPU expert-sharding failures reproduced on clean `origin/main`.

| Gate | Pre-merge evidence | Status at that revision |
| --- | --- | --- |
| Stock baseline | Pinned vLLM 0.23 container loaded the exact checkpoint in BF16 with stock ModelOpt NVFP4 and FLA GDN. `/health` returned 200; deterministic text and the paired streaming benchmark succeeded. | Functional and small warm latency baseline passed; peak-load memory pending. |
| Capture and inventory | Exact checkpoint header inspection found `model.language_model.layers.*`; all 64 MLPs form one structural profile. The shared trace path captures BF16 W4A4 `mlp16@nvfp4` and `mlp64@nvfp4`; the symbolic option remains available in the helper. A pinned exact-checkpoint trace saved 2 graphs of 148 nodes each and 6 distinct kernels. | Focused CPU tests and trace passed; a measured golden release audit is not part of this environment-pinned recipe. |
| Shared BF16 boundaries | The input carrier, constant binding/cache, device output view, and NumPy graph interpreter had independent BF16 carrier bugs. Shared fixes and focused tests encode numerical BF16 as bits, decode it for arithmetic, and preserve raw bitcasts. | Pre-merge repository CI passed. Post-merge focused suite: 245 passed, 2 skipped; lint passed. Full post-merge gate pending. |
| Synthetic GPU MLP | Two tiny layers compiled with plan reuse. Widths 1, 2, 15, 16, 17, 63 and 64 returned finite, distinct BF16 outputs; a nondefault stream passed. | Execution smoke passed. Exact stock quantized MLP comparison currently fails. |
| Mixed full model | The stock constructor replaced all 64 MLP modules; all seven checkpoint shards loaded, then Emmy bound all 128 programs. Both M=16 with symbolic prefill and M=16 with M=64 native prefill booted and returned `/health` 200. The M=64 boot reported 23.43 GiB model memory and about 31.57 GiB resident GPU memory. | Pinned endpoint boots passed. |
| Real hybrid preservation | A two-layer actual vLLM 0.23 Qwen3.5 model with GDN and full-attention layers retained non-MLP module and parameter identities and hybrid state interfaces after Emmy replaced its MLPs. | Pinned-container constructor regression passed. |
| Full-size static MLP | Isolated layer-0 M=1 default compile yielded four launches and 150,410,260 bound weight bytes. Launch 0 took 0.393 ms. Uncaptured launch 1, `k_linear_reduce_93e407`, exceeded a 10,000 ms kernel watchdog. Fresh final-pin CUDA dumps for both M=16 and M=64 contain four native FP4 MMA contraction kernels each. | Unpinned route blocked; native pinned execution and measured endpoint speed passed on selected shapes. |
| Warm text completions | Stock, scalar mixed and native mixed returned the same 16-token Paris continuation. Native mixed also returned expected Japan then Paris continuations with warm wall times 1.895 and 1.877 s; the initial native request took 152 s while stock vLLM/Triton kernels compiled on first use. | Deterministic text and simple cross-request state smoke passed; first-request latency needs warmed-cache handling. |
| Single-request latency | Standard `vllm bench serve` with seed-42 random 5-input/16-output requests, one warmup, five measured, concurrency one, and ignore-EOS produced stock/scalar/M=16+symbolic/M=16+M=64 false/M=16+M=64 true mean TTFT 288.16/617.07/390.51/271.86/270.97 ms and TPOT 110.10/444.49/100.27/101.32/101.19 ms. All 25 measured requests succeeded. | Final pinned route reached 9.88 decode tokens/s versus stock 9.08; TTFT was 17.2 ms lower than stock in this small warm run. This is a single request shape, not general throughput. |
| Teacher-forced prompt | Stock and the earlier `FAST_MATH=false` native mixed route returned the same 14 token IDs and expected Tokyo continuation for one fixed 13-token prompt. Selected-token logprob differences across 13 positions had mean absolute 0.0700, RMS 0.0980, maximum 0.2431; earlier scalar maximum was 0.281. | Token smoke passed; numerical logit parity remains open. |
| 4K context | The same 4,005 prompt tokens plus 16 generated tokens completed with identical stock and `FAST_MATH=false` mixed text, 4,021 total tokens, HTTP 200, and no OOM. After a short first-use request, stock took 12.451 s, mixed symbolic scalar prefill took 224.22 s, and mixed native M=64 prefill took 11.547 s wall. The final `FAST_MATH=true` route completed a second 4,005+16 request in 12.348 s with no OOM; its prompt differed, so that is a shape check rather than a paired stock comparison. | Full requested length envelope passed; matched false-pin native long latency passed. |
| End-to-end correctness | Selected short and 4K text requests, four additional 64-token prompts, one teacher-forced prompt, and real hybrid topology/state were compared. The four additional prompts each diverged from stock. Cache cancellation and broader quality were not tested. | Functional serving passed on this envelope; exact numerical parity failed and quality acceptance remains open. |

### Post-merge cut and pin findings

Main's producer fusion changed the layer-0 cut inventory: the first M=16 or M=64 gate/up piece now owns both
contractions and their output branches. The pre-merge `@place_...` schedule identities no longer select those
contractions. A post-merge boot with those stale pins compiled scalar gate/up code and stalled during warmup, so
that boot is not a post-merge serving pass. The current recipe first pins five parent workspace cuts and then pins
`PLACE@place_<parent>/map.1/inner=cut` and `/map.2/inner=cut` in each shape's own knob context. This preserves
the parent workspace cuts and maximal fusion; the other children settle to fuse. Parent-only pins retain their
former terminal behavior. New focused regressions check the parent/child sequence, sibling isolation, termination,
and rejection of stale child keys.

| Shape | Exact post-merge child cut scopes | Final contraction pins | Compile-only result |
| --- | --- | --- | --- |
| M=16 | `place_643aecc968/map.1/inner`, `place_643aecc968/map.2/inner` | `place_532c520dc2` with `WORK=w1x2`; `place_4b5e95ec28` with `WORK=w1x1`; down `node_linear_2` with `WORK=w1x2` | Seven kernels; two gate/up-derived output pieces and down emit native FP4 MMA. |
| M=64 | `place_f688369f74/map.1/inner`, `place_f688369f74/map.2/inner` | `place_c0904cfc6e` and `place_8f9ed3f314` with `WORK=w1x1`; down `node_linear_2` with `WORK=w1x2` | Seven kernels; two gate/up-derived output pieces and down emit native FP4 MMA. |

For each contraction pin the exact `TILE` is `mma_m16n8k64_e2m1_f32/f1x2/k4` and `STAGE` is
`d2/smem-async`; the global fallback pins remain explicit. The M=64 first child refused `WORK=w1x2` with “its
kernel pins leave no schedule row”; `w1x1` compiled and emitted native FP4. Two output cuts leave two
gate/up-derived kernels with native instructions plus one native down kernel, rather than the pre-merge four native
kernels; the other four handle activation quantization or block scales. The gate/up child IR includes a
tuple-valued contraction, so these are kernel counts, not one kernel per logical projection. Instruction counts
in emitted M=64 CUDA were 5 and 9 in the two gate/up-derived pieces and 3 in down. These are
emitted-instruction checks, not a measured post-merge endpoint latency.

The first post-merge same-checkpoint numerical run used layer 0, BF16 inputs, stock
`FlashInferCutlassNvFp4LinearKernel`, and these pinned Emmy M=16/M=64 programs. M=16 with one active row had
0.466133% relative RMS error, mean absolute error 0.006127, and maximum absolute error 0.03125. M=64 had
0.546888% relative RMS, mean absolute error 0.006545, p99 absolute error 0.02295, and maximum absolute error
1.0 against a reference maximum of 140. Both had zero elements outside the existing `atol=0.05, rtol=0.05`
diagnostic threshold. These fresh values are close to pre-merge results but do not imply bit parity or full-model
quality equivalence. The full post-merge 27B endpoint then bound all 64 MLPs, loaded 23.43 GiB of model memory,
and returned `/health` 200 on host port 8080. The first response took 145.705 s with vLLM Triton first-use JIT;
subsequent short fixed tasks took 0.38–0.81 s. The mixed endpoint answered all five predeclared expected-answer
tasks correctly: arithmetic, Japan's capital, list extraction, exact output format, and a 3,530-token needle.
It served fixed-text echo/logprob requests and token-width 63/64/65/131 probes without errors. One deterministic
4,005-input/16-output request completed in 11.163 s, 4,021 total tokens, without OOM. A warm standard
`vllm bench serve` run of five measured 5-input/16-output requests, one warmup, seed 42, concurrency one, and
ignore-EOS returned five successes: mean TTFT 276.50 ms, mean TPOT 104.23 ms (9.59 decode tokens/s).
The matched stock vLLM 0.23 server used the same checkpoint, BF16, eager execution, no prefix cache, and
single-request envelope on the same rebooted driver 580.178.04. It returned the identical expected text on all
five tasks. Its first-use JIT request took 146.468 s; subsequent task times were 0.42–0.91 s and its 3,530-token
needle took 10.356 s. The same 4,005+16 request took 12.247 s and returned the identical 16-token output,
`KKKKKKKKKKKKKKKK`, with 4,021 total tokens. In the same standard warm five-request benchmark, stock TTFT was
303.14 ms and TPOT 114.08 ms (8.77 decode tokens/s), against mixed 276.50/104.23 ms (9.59). Thus mixed was
8.8% lower in TTFT and 8.6% lower in TPOT in this small measured run. No higher-concurrency or broader throughput
claim follows from five single-request samples.

Three predeclared fixed texts had 48 aligned selected-token echo logprobs. Stock versus mixed mean absolute
logprob difference was 0.3525, RMS 0.7341, and maximum 3.8014. The maximum was a space token with stock
probability about 0.0046; among 29 positions whose stock selected token had probability above 0.135, mean
absolute difference was 0.1011. The top logged candidate agreed at 41/48 positions. These differences remain a
material numerical limitation even though the five bounded expected-answer tasks and the paired long output
matched. They do not by themselves prove a broader task-quality regression or equivalence.

After the stock run, the mixed endpoint was restarted on host port 8080 and returned `/health` 200. A streaming
request was disconnected after a non-whitespace generated text token (`1`); the next deterministic request returned
`Assistant: Tokyo.`, and health remained 200. This checks cancellation and fresh state for one active request.

During this qualification the host's unattended upgrade installed NVIDIA userspace 580.178 over loaded kernel
580.95.05, preventing GPU containers from starting. A reboot loaded 580.178.04 and restored `nvidia-smi`.
With the user's explicit authorization, `unattended-upgrades.service`, `apt-daily.timer`, and
`apt-daily-upgrade.timer` were disabled on this dedicated test box; the interrupted `initramfs-tools` configuration
was completed and `dpkg --audit` is empty. The post-reboot driver version is part of subsequent measurements.

The stalled kernel source has grid 1 and block 256, but only one thread enters the body. It performs full-K scalar
gate/up reductions for output codes and again for per-block scales, with no native FP4 MMA. At the actual layer width,
nested loops amount to more than one billion serial scalar operations with repeated FP4 decode reads. The generated
source and boot logs are retained in the RTX 5090 qualification workspace. This was an **unpinned default schedule**.
The user has since clarified that prior tuning is broken and every Emmy compilation and serving run must use explicit
knob pins through a golden or `EMMY_KNOBS`. The timeout is evidence about that default route; the final pinned
M=16/M=64 route has measured latency near stock on the reported request shapes.

### Explicit knob experiments on the RTX 5090

The earlier endpoint's simple scalar route used this exact `EMMY_KNOBS` value for both capture shapes. The first
M=16/M=64 route retained its `FAST_MATH=false` precision pin and shared cuts. A later numerical probe changed only
that pin to `FAST_MATH=true`, retaining the native piece pins; the checked-in recipe uses the latter candidate:

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
| Same native cuts and piece pins, explicitly `FAST_MATH=true` | Isolated layer-0 RMS versus stock fell from 1.1937% to 0.4661% at M=1 and from 0.9946% to 0.5469% at M=64. The M=1 first activation matched all 320 E4M3 scale bytes and all 2,560 packed FP4 bytes. Pinned exact-checkpoint trace produced two 148-node graphs, six kernels, and `.fm` realizations. Full endpoint bench TTFT/TPOT was 270.97/101.19 ms; a 4,005+16 request completed in 12.348 s. | This precision pin improves the measured MLP oracle and retains speed. Broader stock token parity still fails. |

The first M=16 native probe used these overrides before the quant graph changed. Its place identities are stale and
must not be used for the current model graph:

```text
WORK@place_35acacc227=w1x1,TILE@place_35acacc227=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_35acacc227=d2/smem-async,WORK@place_4f08570ba8=w1x1,TILE@place_4f08570ba8=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_4f08570ba8=d2/smem-async,WORK@place_159d5b6179=w1x2,TILE@place_159d5b6179=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@place_159d5b6179=d2/smem-async,WORK@node_linear_2=w1x2,TILE@node_linear_2=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE@node_linear_2=d2/smem-async
```

The post-merge exact `EMMY_MLP_STATIC_KNOBS` and `EMMY_MLP_PREFILL_KNOBS` are in
[`scripts/serve_qwen38_nvfp4_mixed_5090.sh`](../scripts/serve_qwen38_nvfp4_mixed_5090.sh). The first table in
“Post-merge cut and pin findings” spells every child cut and contraction scope. Each shape's overrides apply only
around its own compile; the five parent cuts, `FAST_MATH=true`, and all baseline schedule families remain pinned
in the shared `EMMY_KNOBS` value. The pre-merge identities in the earlier experiment rows are retained as historical
evidence and must not be copied into a current serving command.

The five projection-cut candidates came from the earlier `fp4-encode-recomputes-producer.md` report. Its native
FP4 and async-stage suggestions are historical starting points, not measured 5090 wins. The current compiler
supports `@place_<token>` pins for an individual cut piece; that selector was not demonstrated by the older report's
`@n0` and full-name experiments. The M=1 refusal above is structural for these cuts, not an ignored pin. An
identity-keyed golden may route a parent M=1 cut followed by a child cut that retains the unit row; exact MLP replay
remains untested. The new explicit child-site pins solve the post-merge M=16/M=64 output cut sequence without a
golden route. The M=1 unit-row problem remains separate because that shape drops the tensor-core M axis. The scalar
route has repeated endpoint probes; the current padded native route has the fresh layer-0 numerical result above.

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

Under the earlier `FAST_MATH=false` pin, the corrected down-projection input differs from stock in 58/1088 E4M3
bytes and 373/8704 packed FP4 bytes, with 4.1487% raw reconstructed relative RMS. This accumulates prior
projection and activation rounding, then another FP4 quantization amplifies it. Stock-byte injection at the first
activation producer lowers gate/up projection RMS to 0.1025%/0.0984%. This localizes most of the initial projection
difference to two FP4 threshold flips. Stock uses approximate reciprocal instructions; the earlier pin used exact
FP32 divide at those thresholds.
Further correction should be justified by full-model quality evidence rather than bit identity alone.

That full-model evidence now includes four independent deterministic 64-token prompts. With the earlier
`FAST_MATH=false` native route, all four eventually diverged from stock at token positions 1, 15, 29 and 58.
Stock's top two tokens were exactly tied at the first position of the sky-explanation prompt; the other stock
winning margins at their first divergences were 0.125, 0.375 and 0.125 logprob. This is a real numerical drift
gate, even though the earlier Paris and 4K requests matched stock text. With `FAST_MATH=true`, the same prompts
first diverged at positions 1, 27, 29 and 58: improved isolated FP4 matching did not eliminate full-model
argmax drift. Do not infer exact stock quality from the matched smoke requests.

An earlier two-layer synthetic build exited 139 once during compilation. Later full redirected and real 27B builds
completed, so the cause is unestablished. It remains a cold-build stability risk until repeated boots or a native
trace resolve it. The synthetic NumPy graph and GPU outputs are finite after the BF16 fix but differ at some
BF16/W4A4 boundaries; that graph interpretation alone is not the independent stock quantized MLP oracle.

## Remaining qualification and limits

The post-merge exact 5090 native instruction check, same-checkpoint stock MLP oracle, full 27B boot, bounded
answer-quality corpus, long-context request, and paired warm benchmark are complete. The environment-pinned
recipe has an exact checkpoint trace and runtime check; it has no measured golden file for the separate golden
release gate. GitHub's full repository test, native, lint, and package jobs passed on code commit `4b71026c`.
The final streaming cancellation and restored-endpoint checks also passed as described above. This report and
README update follow that tested code commit without implementation changes; their own documentation-only CI
run may have a different status until it completes.

The 0.47–0.55% same-checkpoint layer-0 relative RMS residual and the fixed-text logprob differences can affect
token choice. Four longer greedy continuations diverged before this main merge. The bounded five-task rubric
passed on stock and mixed after the merge, but it does not establish general model-quality equivalence. Preserve
the stock quantized MLP as the independent numerical oracle, the explicit `FAST_MATH=true` pin, and the emitted
native-instruction checks when changing the source graph or piece schedule. A larger task suite, cancellations
under concurrent traffic, prefix caching, speculation, and broader throughput remain future work outside the
initial one-active-request envelope.

## Relation to the earlier investigation

[`nvfp4-qwen-performance.md`](nvfp4-qwen-performance.md) is the September 28 investigation, not a completed
implementation plan. Its
[`fp4-encode-recomputes-producer.md`](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md) report is directly
relevant: the actual M=1 gate/up encode kernel repeats full-K work. This PR's isolated 5090 watchdog and generated
CUDA give new evidence for the mixed lane; the report's 5080 timings and proposed cut are not assumed to transfer.
Emmy's GDN #973 support is not required by this MLP boundary because stock vLLM owns GDN, but that Emmy path remains
available. The investigation's LUT, TMA and native GDN ideas remain separate performance work. The later review-only
mixed plan proposed FP16, which the user rejected in favor of the checkpoint's BF16; that review branch is not part
of this PR.
