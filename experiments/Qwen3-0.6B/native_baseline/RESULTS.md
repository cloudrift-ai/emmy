# Native-serving baseline

## RTX 4080 × 1 — 2026-09-16

Stock vLLM completed the matrix. All three Emmy-vLLM configurations missed the 900-second readiness deadline while
compiling and produced no serving measurements. This run does not establish a native serving performance or memory
advantage. The useful completed observation is that stock short decode is largely GPU-busy, including the work
between transformer graph replays; replacing CPU dispatch alone has little exposed time to remove in that trace.

Read the separate [dispatch report](DISPATCH.md), [shape/memory report](SHAPES_MEMORY.md), and
[execution/API investigation](INVESTIGATION.md). They distinguish measurements from missing evidence.

### Protocol and environment

[recipe.yaml](recipe.yaml) pins Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, FP16,
context 4,096, maximum four sequences, a 256-token batched prefill limit, Triton attention, full CUDA graphs with
explicit capture sizes, and prefix caching disabled. It compares stock, width-16 decode, the single-token tier, and
that tier with activation capacity reduced from 1,024 to 256. Each fixed workload has three repeats, eight requests,
two warmups, concurrency one or four, inputs 32/256/1,024, and exactly 64 output tokens. Mixed-length and profiled cases
are separate. Each Emmy lane starts with an empty tune DB and online-prior path; the hardware golden digest is saved.

Local system: Ubuntu 24.04.5, kernel 7.0.0-31-generic, Core i9-14900K, one RTX 4080 with 16,376 MiB, driver
595.91.07, NVCC 13.3.73, cuBLAS 13.6.0.2. Installed packages include PyTorch 2.11.0+cu130, vLLM 0.23.0, and Transformers
5.14.1; exact packages are in each row's `requirements.txt`. The desktop shares the GPU. No cloud server was rented.

### Outcomes and retained evidence

Run ID `20260916T080039Z`; timestamped directory `2026-09-16_08-00-39`.
[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz) contains that complete directory, four system-only experiment
records, raw command logs, declared results, and the partial serving packs. Record and row-directory stems are:

| Configuration | Stem | Status |
| --- | --- | --- |
| Stock | `rtx4080x1_c1024_lstock_m1_f5b4395d2139` | succeeded |
| Decode bucket 16, single-token tier off | `rtx4080x1_c1024_lbucket16_m0_242586f9983a` | failed |
| Single-token tier, capacity 1,024 | `rtx4080x1_c1024_lm1_m1_6f930adf2e27` | failed |
| Single-token tier, capacity 256 | `rtx4080x1_c256_lm1-s_m1_679bbdc79eb9` | failed |

The stock short concurrency-one median TPOT spans 2.278–2.282 ms across repeats. Increasing concurrency to four gives
1,513.9–1,521.0 output tokens/s for the same short input. The complete fixed-length table and the single mixed-length
observation are in the shape/memory report, with their raw JSON member names. Profiler timings are not substituted
for unprofiled request latencies.

The failed logs repeatedly reject projection schedules whose epilogues reference an unavailable contraction
accumulator and then rerank. The first Emmy row eventually saved a pack after approximately 16 minutes, beyond the
readiness limit. Each failed row's graceful shutdown stalled, so its owned process group was killed to let the harness
finish the row and continue. The records report command failure; they must not be read as successful empty results.
All task-owned serving processes were terminated. No failure-only rerun was performed.

### Limits and next decision

All records capture the group's initial clean revision `52a7491181f7f63f6b2c3a827dd3d396653e8477`. Local commands
execute the live checkout. Subsequent runtime/export and setup-timing work changed Python backend files after the
stock row, so the later failure logs do not represent an immutable per-row checkout. No numerical Emmy result is
claimed from them. A renewed comparison must freeze the source for its entire lifetime.

The baseline is incomplete. It has no successful comparison of useful/padded rows, scratch/activation allocation,
KV capacity, or Emmy-vLLM dispatch. Fix or bound initialization and repeat the full serving matrix before using this
experiment to justify a new model-serving loop. The independent [runtime comparison](../native_runtime/RESULTS.md)
qualifies small static artifacts only and does not supply the missing generation evidence.

Published evidence replaces the local hostname, username, home/workspace paths, LAN address, and GPU UUID with
anonymous placeholders and clears archive owner metadata. Hardware/software specifications, measurements, and
executable tensor/binary payloads are unchanged. Local paths in the records are descriptive, not portable inputs.
