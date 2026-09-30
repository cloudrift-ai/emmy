# Qwen3.8 NVFP4 mixed serving: implementation and qualification

Status: implementation in draft PR #993, September 30, 2026. The exact 27B endpoint has not passed readiness or numerical qualification.

## Boundary and target

The first serving lane replaces only the 64 dense text MLPs of vLLM 0.23's `Qwen3_5ForConditionalGeneration` with Emmy programs. The stock model still constructs and executes attention, GDN and its state, projections outside the MLP, norms, residuals, rotary embedding, embeddings, output head, and scheduling. Its non-MLP weights keep the checkpoint's ModelOpt NVFP4 loader and quantization configuration. Emmy takes the exact 12 packed checkpoint tensors per MLP layer and binds per-layer constants to shared static M=1 and symbolic M<=64 plans. These lower-level compilation and binding interfaces remain usable by a larger Emmy region later.

The qualification checkpoint is `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462` on one RTX 5090. The contract is BF16, text only, TP1/PP1, one active request, 4,096 context tokens, at most 64 scheduled tokens, with prefix caching, speculation and outer CUDA graphs disabled. The earlier proposed FP16 boundary was rejected; no FP16 fallback is part of this lane.

## Evidence and current status

| Gate | Evidence | Status |
| --- | --- | --- |
| Stock baseline | Pinned vLLM 0.23 container loaded the exact checkpoint in BF16 with stock ModelOpt NVFP4 and FLA GDN. `/health` returned 200; a deterministic 16-token text completion succeeded. | Functional baseline passed; peak-load memory and latency comparison pending. |
| Capture and inventory | Exact checkpoint header inspection found `model.language_model.layers.*`; all 64 MLPs form one structural profile. The shared trace path captures two 148-node BF16 W4A4 graphs, `mlp1@nvfp4` and `mlp-sym@nvfp4`, and trace/audit checks exact identity. | Passed focused CPU tests. |
| Shared BF16 boundaries | The input carrier, constant binding/cache, device output view, and NumPy graph interpreter had independent BF16 carrier bugs. Shared fixes and focused tests encode numerical BF16 as bits, decode it for arithmetic, and preserve raw bitcasts. | Focused CPU tests passed; broader regression gates pending. |
| Synthetic GPU MLP | Two tiny layers compiled with plan reuse. Widths 1, 2, 15, 16, 17, 63 and 64 returned finite, distinct BF16 outputs; a nondefault stream passed. | Execution smoke passed. Exact stock quantized MLP parity remains pending. |
| Mixed full model | The stock constructor replaced all 64 MLP modules; all seven checkpoint shards loaded, then Emmy bound all 128 programs. vLLM reported 4.98 GiB available KV memory and a 59,684-token KV cache. | Boot reached first dummy forward, but the endpoint did not become healthy. |
| Full-size static MLP | Isolated layer-0 M=1 compile yielded four launches and 150,410,260 bound weight bytes. Launch 0 took 0.393 ms. Uncaptured launch 1, `k_linear_reduce_93e407`, exceeded a 10,000 ms kernel watchdog. | Blocking compiler schedule defect. |
| End-to-end correctness | Stock and mixed same-prompt output, logits, request lengths, hybrid state and cache behavior. | Pending until the full-size kernel is corrected. |

The stalled kernel source has grid 1 and block 256, but only one thread enters the body. It performs full-K scalar gate/up reductions for output codes and again for per-block scales, with no native FP4 MMA. At the actual layer width, nested loops amount to more than one billion serial scalar operations with repeated FP4 decode reads. The generated source and boot logs are retained in the RTX 5090 qualification workspace. This was an **unpinned default schedule**. The user has since clarified that prior tuning is broken and every Emmy compilation and serving run must use explicit knob pins through a golden or `EMMY_KNOBS`. The timeout is evidence about the default route, not yet about the required pinned deployment route.

An earlier two-layer synthetic build exited 139 once during compilation. Later full redirected and real 27B builds completed, so the cause is unestablished. It remains a cold-build stability risk until repeated boots or a native trace resolve it. The synthetic NumPy graph and GPU outputs are finite after the BF16 fix but differ at some BF16/W4A4 boundaries; that graph interpretation alone is not the independent stock quantized MLP oracle.

## Next qualification work

1. Pin the complete Emmy schedule and `FAST_MATH=false` through a golden or `EMMY_KNOBS`, then test a legal projection cut and native FP4 cell on the same actual-size graph. Record the exact pins, graph identity, selected kernels and 5090 timings. Fix compiler scheduling or producer sharing if the pinned route still blocks serving, without weakening its maximal fusion invariant. Rerun the per-launch watchdog and full endpoint boot under those same pins.
2. Compare Emmy's MLP output with the pinned stock vLLM ModelOpt quantized MLP on identical checkpoint tensors and BF16 inputs. Validate static M=1 and symbolic widths through 64 before assessing teacher-forced logits and deterministic text completions.
3. Measure memory and meaningful latency on the exact RTX 5090 envelope; investigate repeated full contractions and encode work only when reproduced. Run the repository's final test and lint gates, record exact versions and commands, and update the draft PR with measured results and remaining limits.

## Relation to the earlier investigation

[`nvfp4-qwen-performance.md`](nvfp4-qwen-performance.md) is the September 28 investigation, not a completed implementation plan. Its [`fp4-encode-recomputes-producer.md`](nvfp4-qwen-performance/fp4-encode-recomputes-producer.md) report is directly relevant: the actual M=1 gate/up encode kernel repeats full-K work. This PR's isolated 5090 watchdog and generated CUDA give new evidence for the mixed lane; the report's 5080 timings and proposed cut are not assumed to transfer. Emmy's GDN #973 support is not required by this MLP boundary because stock vLLM owns GDN, but that Emmy path remains available. The investigation's LUT, TMA and native GDN ideas remain separate performance work. The later review-only mixed plan proposed FP16, which the user rejected in favor of the checkpoint's BF16; that review branch is not part of this PR.
