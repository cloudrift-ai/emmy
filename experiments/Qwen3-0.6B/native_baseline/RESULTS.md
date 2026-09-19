# Native-serving baseline

The later [schedule qualification](SCHEDULES.md) fixes three numerical issues and passes a bounded serving startup
check. The current recipe selects that new experimental golden. The full four-configuration result below remains
the previous run; the updated recipe has not yet been rerun as a complete matrix.

## RTX 4080 × 1 — 2026-09-18

Stock vLLM completed the matrix. All three Emmy configurations compiled their programs without the previous
projection rejections, but missed the 900-second readiness deadline during GPU initialization. There are no Emmy
serving measurements. Stock short decode is largely GPU-busy, including work between transformer graph replays;
replacing CPU dispatch alone has little exposed time to remove in that trace.

Read the separate [dispatch report](DISPATCH.md), [shape/memory report](SHAPES_MEMORY.md), and
[execution/API investigation](INVESTIGATION.md). They distinguish measurements from missing evidence.

### Protocol and environment

The recipe at `0d5ac561` pins Qwen/Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, FP16,
context 4,096, maximum four sequences, a 256-token batched prefill limit, Triton attention, full CUDA graphs with
explicit capture sizes, and prefix caching disabled. It compares stock, width-16 decode, the single-token tier, and
that tier with activation capacity reduced from 1,024 to 256. Each fixed workload has three repeats, eight requests,
two warmups, concurrency one or four, inputs 32/256/1,024, and exactly 64 output tokens. Mixed-length and profiled cases
are separate. Each Emmy lane starts with an empty tune DB, online-prior path, and pack directory; the hardware golden
digest is saved. The machine's compiled-kernel cache is shared and was not cleared. Startup is not a cold-cache measure.

Local system: Ubuntu 24.04.5, kernel 7.0.0-31-generic, Core i9-14900K, one RTX 4080 with 16,376 MiB, driver
595.91.07, NVCC 13.3.73, cuBLAS 13.6.0.2. Installed packages include PyTorch 2.11.0+cu130, vLLM 0.23.0, and Transformers
5.14.1; exact packages are in each row's `requirements.txt`. The desktop shares the GPU. No cloud server was rented.

### Outcomes and retained evidence

Run ID `20260918T040005Z`; timestamped directory `2026-09-18_04-00-05`.
[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz) contains that complete directory, four system-only experiment
records, raw command logs, declared results, and serving packs. Record and row-directory stems are:

| Configuration | Stem | Status |
| --- | --- | --- |
| Stock | `rtx4080x1_c1024_lstock_m1_f5b4395d2139` | succeeded |
| Decode bucket 16, single-token tier off | `rtx4080x1_c1024_lbucket16_m0_242586f9983a` | failed |
| Single-token tier, capacity 1,024 | `rtx4080x1_c1024_lm1_m1_6f930adf2e27` | failed |
| Single-token tier, capacity 256 | `rtx4080x1_c256_lm1-s_m1_679bbdc79eb9` | failed |

The stock short concurrency-one median TPOT spans 2.273–2.278 ms across repeats. Increasing concurrency to four gives
1,497.4–1,511.8 output tokens/s for the same short input. The complete fixed-length table and the single mixed-length
observation are in the shape/memory report, with their raw JSON member names. Profiler timings are not substituted
for unprofiled request latencies.

The scheduler now rejects fragment epilogues that contain nested sibling reductions or combine contraction roots
whose outputs cannot be partitioned. Neither reproduced rejection appears in any of the three Emmy server logs.
The bucket-16 configuration saved 168 programs; both single-token configurations saved 224. All remained unready
afterward, with GPU work active in the initialization path containing the boot audit. No Emmy request or trace JSON
was produced; the failed records also report these missing results. Bounded shutdown finished each row in about
911 seconds and released its GPU process. No failure-only rerun was performed.

### Isolated post-attention diagnostic

After the matrix, the existing run command replayed the width-16 post-attention program against eager PyTorch
using synthetic inputs. The default single-kernel schedule exceeded the 60-second kernel watchdog. Explicitly
cutting seven selected reduction boundaries produced eight kernels and passed the CLI's scaled numerical
check. This narrows the remaining investigation to executable schedule selection; it does not establish complete
model correctness or a deployable schedule. The diagnostic command, input and raw outputs are retained separately
under `diagnostic/` in the archive. Two fresh explicit-cut processes passed numerical checks and measured
509.94/510.46 µs, versus eager at
93.26/93.16 µs, using captured whole-forward timings. Both wrote JSON before their overall 115-second deadlines but
failed to exit normally; neither process exit is recorded as a successful qualification. No schedule was promoted
to the model goldens. The eight-kernel proposal is executable in isolation but substantially slower than eager.

### Limits and next decision

All records capture clean revision `0d5ac561a53511994a116ee83099932324e7ad13`. The live checkout stayed unchanged
throughout the matrix. Each row has a 900-second readiness window and a 1,200-second command deadline; shutdown
escalates after ten seconds. Stock completed before the Emmy rows, and no other task-owned GPU workload overlaps it.
CPU-only diagnostics and clean-main golden checks overlapped some Emmy initialization. No startup-time speedup is
claimed. All task-owned serving processes were terminated after the matrix.

The baseline is incomplete. It has no successful comparison of useful/padded rows, scratch/activation allocation,
KV capacity, or Emmy-vLLM dispatch. Fix the GPU execution problem and repeat the full serving matrix before using this
experiment to justify a new model-serving loop. The independent [runtime comparison](../native_runtime/RESULTS.md)
qualifies small static artifacts only and does not supply the missing generation evidence.

Published evidence replaces the local hostname, username, home/workspace paths, LAN address, and GPU UUID with
anonymous placeholders and clears archive owner metadata. Hardware/software specifications, measurements, and
executable tensor/binary payloads are unchanged. Local paths in the records are descriptive, not portable inputs.

### Previous run

The September 16 run in merged PR #820 completed stock and timed out in all three Emmy configurations during
compilation. Its short stock TPOT was 2.278–2.282 ms. Those records and reports remain in Git at `27e11a30`; the
named platform archive now holds only the latest run. The new comparison keeps source fixed and bounds shutdown,
addressing the prior run's provenance and cleanup limitations.
