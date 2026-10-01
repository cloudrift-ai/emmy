# Serving dispatch evidence

All four configurations use captured execution. Their short single-request profiles show little exposed time
between model steps without GPU work. Emmy's slower compiled programs dominate its latency; these measurements
do not support replacing Python dispatch as the main performance remedy.

## Evidence and timing boundaries

The protocol and row stems are in [RESULTS.md](RESULTS.md). Raw evidence is retained in
[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz), under `2026-09-19_06-52-47/<row>/traces/`.
The GPU trace for each configuration is:

| Configuration | Trace |
| --- | --- |
| Stock | `rank0.1789800935307275925.pt.trace.json.gz` |
| Width 16 | `rank0.1789801194287453145.pt.trace.json.gz` |
| Single-token, capacity 1024 | `rank0.1789801458799809213.pt.trace.json.gz` |
| Single-token, capacity 256 | `rank0.1789801716362309840.pt.trace.json.gz` |

Separate API-process traces are also retained; they are not GPU timelines. Each profile has two requests, 32 input
and 32 output tokens, concurrency one, after two warmups. Every GPU trace contains 62 decode and two prefill
annotations, with 64 CPU `cudaGraphLaunch` calls. The interval comparison uses 60 adjacent decode pairs within a
request, excluding the inter-request boundary. GPU busy time is the union of recorded kernel and copy intervals
across streams, clipped to the interval being measured.

| Median trace quantity | Stock | Width 16 | Single-token, 1024 | Single-token, 256 |
| --- | ---: | ---: | ---: | ---: |
| Decode annotation duration, ms | 1.864 | 8.511 | 8.245 | 8.242 |
| GPU interval union inside annotation, ms | 1.844 | 8.362 | 8.199 | 8.196 |
| GPU operations inside annotation | 348 | 762 | 762 | 762 |
| Decode start-to-start interval, ms | 2.362 | 8.524 | 8.258 | 8.255 |
| End-to-next-start gap, µs | 498.159 | 13.021 | 13.630 | 13.566 |
| GPU work in that gap, µs | 490.590 | 9.280 | 9.680 | 9.696 |
| No recorded GPU work in that gap, µs | 7.664 | 3.837 | 3.935 | 3.854 |

The annotations do not enclose identical operations across engines. Stock's between-step work includes a large
matrix-vector operation averaging 467.669 µs per interval; Emmy's between-step work is mainly argmax and copies.
Compare complete start-to-start intervals rather than treating annotation duration alone as identical model work.
The small uncovered interval inside an annotation is also not proof of removable Python overhead: these steps
are captured, and GPU scheduling and instrumentation contribute gaps.

Even removing every observed between-step interval without recorded GPU activity would remove only about 0.32%
of the stock median interval and about 0.05% of Emmy's. These are optimistic local bounds on that interval alone,
not predicted native-runtime speedups. CPU scheduling and output processing can overlap GPU work; total CPU wait
or launch duration cannot be added to an exposed-time saving.

## Interpretation and limits

The unprofiled short-request TPOT ranges are 2.273–2.281 ms for stock and 7.935–8.378 ms across the Emmy
configurations. Profiled times differ and must not replace those repeated request measurements. Both single-token
profiles are almost identical despite a small unprofiled capacity difference; do not infer a general timing gain
from that capacity change alone.

This comparison closes the earlier missing Emmy graph-coverage measurement. It points to GPU schedule quality,
including the expensive isolated matrix-vector programs, as the next performance target. Efficient scheduling can
be improved within the current integration. The profile does not establish a benefit from replacing HTTP handling
or request scheduling, and it does not qualify complete native model execution.

Detailed CPU attribution, longer-context profiles, and contention profiles remain outside this measurement.
The desktop shares the card, and each configuration has one profile trial. No timing claim is extrapolated to
other GPUs or workloads.
