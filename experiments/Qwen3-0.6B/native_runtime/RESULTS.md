# Standalone runtime comparison

## RTX 4080 × 1 — 2026-09-16

Rust executes these exported FP16 RMSNorm programs correctly without Python model execution. It lowers uncaptured
host submission cost and process startup time. With CUDA graphs, GPU event times are essentially equal for the
one-row and sixteen-row cases; the 256-row case differs by about 3%, which this small, ordered experiment does not
establish as a durable advantage. This is a standalone-program result, not a full-model serving speedup.

### Protocol

[recipe.yaml](recipe.yaml) compiles RMSNorm at hidden width 1,024 with 1, 16, or 256 rows. These are Qwen3-0.6B's hidden
width and relevant serving widths, but there is no checkpoint or transformer execution in this experiment. Each pack
contains one kernel, the exact cubin, and deterministic FP16 inputs generated with seed zero. Both runtimes read the
same pack. Each measurement has 20 warmups and 200 timed iterations; three repeats cover Python/Rust, captured/
uncaptured, and persistent/one-shot worker lifetimes. Persistent workers also reload in the same context.

All 72 measurement records report exactly equal outputs against the Python dispatcher. This proves dispatcher
agreement on the exported work, not independent RMSNorm-versus-eager accuracy. The separate native GPU tests cover
ordered multiple launches, constants, repeated zeroing, input updates under capture, and process recovery after a
real CUDA fault or a hard deadline. The worker also ran with Python and NVCC absent from PATH and its cubin cache
removed. The tests are correctness evidence; they are not benchmark measurements.

### Timings

Ranges below use the three persistent-worker repeats. GPU event windows exclude IPC, input/output files, and warmup;
uncaptured windows include exposed host submission gaps.

| Rows | Python uncaptured, µs | Rust uncaptured, µs | Python captured, µs | Rust captured, µs |
| --- | ---: | ---: | ---: | ---: |
| 1 | 3.557–3.680 | 2.780–2.789 | 2.601–2.606 | 2.601–2.610 |
| 16 | 3.548–3.867 | 2.917–2.922 | 2.732–2.734 | 2.729–2.733 |
| 256 | 3.799–3.890 | 3.169–3.174 | 3.057–3.063 | 2.975–2.980 |

Rust's uncaptured submission wall time is roughly 0.88–0.97 µs per iteration in these persistent runs, versus
3.54–3.88 µs for Python. Captured submission is roughly 0.73–0.80 µs versus 0.88–1.07 µs. That CPU work overlaps the
GPU: it is not extra time to add to the event window, and its reduction does not yield an equivalent captured GPU gain.
One-shot event times show the same general pattern; the complete values remain in the raw JSON rather than a second
table that could imply a different worker-lifetime baseline.

Process-cold load round trips span 279.9–315.9 ms for Python and 62.5–85.5 ms for Rust across the matrix. Both use newly
started processes for those observations. Repeated loads in retained contexts span 0.841–1.156 ms for Python and
0.793–1.025 ms for Rust. Python already supports persistent workers, so a serving or tuning session does not pay its
process startup for every program execution. No disk or driver caches were flushed.

The raw load records separate module API time and allocation-related work. Python's allocation/upload field measures
submission together; Rust measures synchronized allocation/zeroing and upload separately. These subdivisions have
different boundaries and cannot be compared as isolated allocator performance. For example, the first one-row Rust
load spends 58.061 ms creating its context and 3.524 ms allocating/zeroing, while the matching Python load reports
63.703 ms inside the reference loader and a 286.067 ms parent round trip. The difference between parent and child
measurements includes startup, imports, protocol work, and scheduling; serialization is not separately isolated.

Python control round trips have several 3.9–5.6 ms outliers while event times stay stable. Their cause is unresolved.
Output downloads and file writes are included in those round trips; they are not token-generation latencies.

### System, provenance, and raw members

All three rows succeeded. Run ID `20260916T085308Z`, timestamped directory `2026-09-16_08-53-08`, clean staged source
revision `ea58663f95f2af2a1eef6a68215c30dc7e906898`. That source remained unchanged during measurement.
Host and GPU match the [serving baseline](../native_baseline/RESULTS.md): Ubuntu 24.04.5, i9-14900K, RTX 4080,
driver 595.91.07, NVCC 13.3.73, cuBLAS 13.6.0.2. Rust and Cargo are 1.95.0; CuPy is 14.1.1. The release worker uses
cudarc 0.19.9. Per-row `toolchain.txt` contains the Cargo lockfile and binary SHA-256, and `requirements.txt` freezes
Python dependencies. Cubins compile at the default deployable optimization level, with no golden or mutable tune DB.

[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz) contains the complete run directory and system-only row records.
The row directories contain `runtime.json`, `toolchain.txt`, `requirements.txt`, and the complete executable `pack/`:

- `rtx4080x1_r1_0e9e60864bb3`
- `rtx4080x1_r16_8ab783662460`
- `rtx4080x1_r256_d7946d1bf1ea`

The artifact is retained so the comparison can be repeated without recompiling different kernels or regenerating
inputs. The desktop shares this GPU; three sequential repeats and fixed runtime ordering do not control clocks,
thermal drift, host scheduling, or rare outliers. No model memory saving, generation latency, batching, or fairness
claim follows from these measurements. The evidence supports retaining the small reusable executor while reviewing
whether full native generation is justified after the existing serving baseline works.

Published evidence replaces the local hostname, username, home/workspace paths, LAN address, and GPU UUID with
anonymous placeholders and clears archive owner metadata. Hardware/software specifications, measurements, and
executable tensor/binary payloads are unchanged. Local paths in the records are descriptive, not portable inputs.
