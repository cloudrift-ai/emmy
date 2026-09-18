# Serving dispatch evidence

The completed stock run does not show a large exposed CPU-dispatch opportunity for short single-request decode.
Almost all the interval between captured transformer steps is GPU work. This is evidence about this profiled stock
workload, not a native-runtime gain or a conclusion about Emmy-vLLM, whose baseline did not become ready in time.

## Evidence and timing boundaries

Configuration and row outcomes are in [RESULTS.md](RESULTS.md) and [recipe.yaml](recipe.yaml). Raw evidence is retained
in [results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz), under `2026-09-16_08-00-39/`.
The stock row directory is `rtx4080x1_c1024_lstock_m1_f5b4395d2139`; its GPU timeline is
`traces/rank0.1789545817465949906.pt.trace.json.gz`. The separate API-process trace is not a GPU timeline.

The profiled workload has two requests, 32 input tokens and 32 output tokens, concurrency one, after two warmups.
It records 62 single-token decode annotations and two prefill annotations. The model step is captured: 64 CPU
`cudaGraphLaunch` calls match those graph executions. Median decode annotation duration is 1.870 ms, with 348 recorded
GPU operations inside a typical decode annotation. Their median interval union occupies 1.850 ms. The remaining
roughly 0.020 ms lies inside a CUDA graph and cannot be credited to removal of Python per-kernel submission.

For the 60 adjacent decode pairs within a request, excluding the inter-request interval:

| Trace quantity | Median |
| --- | ---: |
| Decode graph start-to-start interval | 2.365 ms |
| End of one graph to start of the next | 0.495 ms |
| Recorded GPU work in that between-graph interval, union across streams | 0.487 ms |
| No recorded GPU work in that between-graph interval | 0.0077 ms |

The dominant between-graph operation is a matrix-vector kernel. Its correlated `aten::mm` has FP16 input shapes
`[1, 1024]` and `[1024, 151936]`, identifying the output projection. Across those 60 intervals it averages 0.464 ms.
Sampling, indexing, status updates, and copies occupy most of the rest; the named Gumbel sampling kernel averages
0.0034 ms per interval. CPU request scheduling and output handling can overlap this GPU work.

Eliminating every observed 0.0077 ms between-graph interval with no recorded GPU activity would change the median
2.365 ms interval by only about 0.33%. That is an optimistic local ceiling for that interval alone, not a predicted
Rust speedup. The trace does not establish that all of this remainder is removable CPU work, and it does not cover
all request lengths or contention. Median components also need not sum exactly.

The trace contains 64 `cudaEventSynchronize` calls totaling 91.793 ms of CPU wall time. Those waits overlap GPU work;
adding them to a proposed CPU saving would be wrong. Likewise, 8.413 ms of accumulated CPU `cudaGraphLaunch` duration
is not 8.413 ms of exposed critical-path delay. Recorded transfers total 0.005 ms for pinned HtoD, 0.092 ms for pageable
HtoD, and 0.124 ms for pinned DtoH across the complete profile. These totals are not per-token latency savings.

## What remains unresolved

The unprofiled short concurrency-one median TPOT is 2.278–2.282 ms across three repeats. The profiled 2.365 ms interval
is a different measurement and must not replace it. Detailed CPU attribution to scheduler, metadata, text output, and
serialization is not complete; the GPU timeline already rules out treating all between-graph time as CPU overhead.

The Emmy configurations timed out during initialization, so there is no measured Emmy-vLLM graph coverage or
critical-path comparison. The immediate next serving investigation should address compilation/readiness before
claiming a benefit from owning the serving loop. Kernel fusion or output-head improvements could help independently
of replacing the HTTP frontend or scheduler, but this run does not measure such changes.
