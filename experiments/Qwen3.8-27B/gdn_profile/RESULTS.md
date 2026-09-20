# V100 GDN register-schedule profiles

## V100 SXM2 16 GB — 2026-09-20

Packed FP32 operand loads reduce global-load instructions by 24.2% in the 48-head, 128-token carry kernel.
Global/local load-issue throttling falls from 48.7% to 18.7% of warp cycles per issued instruction. Total instruction
count falls only 2.7%; reducing load-issue pressure matters more here than reducing the instruction body's size.
The separate [uninstrumented benchmark](../gdn_register/RESULTS.md) measures a 22.1% whole-forward latency reduction
for this case. These counters support the load-pressure diagnosis; their stall percentages are not wall-time shares.

Both new profiling runs succeed and pass strict correctness. They cover 48 heads with 128 and 512 tokens using
FP32 accumulation. Both source hashes match their corresponding uninstrumented benchmark kernels exactly.

### Instruction changes

Counts are dynamic warp instructions from `sass__inst_executed_per_opcode`. Static counts are unique instruction
addresses in the SASS export. The before column is the preceding profile of the same 48-head, 128-token operation
on this host with identical pins and profiler settings.

| Measure | 128 tokens before | 128 tokens now | 512 tokens now |
| --- | ---: | ---: | ---: |
| Executed instructions | 4,349,184 | 4,232,448 | 16,891,008 |
| SHFL | 786,816 | 786,816 | 3,146,112 |
| HMMA | 786,432 | 786,432 | 3,145,728 |
| Global loads, LDG + LD | 811,776 | 615,168 | 2,460,672 |
| Local loads, LDL | 0 | 0 | 95,232 |
| Local stores, STL | 0 | 0 | 43,776 |
| Static instructions | 5,712 | 5,560 | 5,608 |
| Instruction bytes | 91,392 | 88,960 | 89,728 |

The packed helper uses generic-address `LD` instructions for its global pointers. Counting only `LDG` would
exaggerate the improvement: the new short case has 418,560 LDG plus 196,608 LD instructions. Its matrix work,
shuffle count, and FP32-to-FP16 conversion count are unchanged. Added address instructions partly offset the
reduction in loads, which is why total instructions fall much less than global loads.

The compiler reads each aligned FP32 pair with one 64-bit load and immediately packs it to FP16 inside inline PTX.
This keeps temporary floats inside the load operation. The shared four-value loader also serves FP16 operands and
shared-memory drains. Alignment and K-mask checks retain scalar gathers where the vector load cannot be proved safe.
The chunk loop, register ownership, intermediate reuse, and two-warp schedule remain unchanged.

### Hardware counters and remaining costs

Occupancy is achieved active warps relative to the hardware maximum. Tensor is percent of peak sustained elapsed
throughput. Stall columns divide each `smsp__average_warps_issue_stalled_*_per_issue_active.ratio` by
`smsp__average_warp_latency_per_inst_issued.ratio`. They describe warp cycles per issued instruction, not wall time.
Fetch includes instruction-fetch waiting and instruction-cache misses without separating them. Long scoreboard
means waits for L1TEX memory dependencies.

| Case | Occupancy | Tensor | Load issue throttle | Long scoreboard | Fetch |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128 tokens before | 7.44% | 3.33% | 48.69% | 13.98% | 3.50% |
| 128 tokens now | 7.46% | 4.72% | 18.65% | 37.48% | 1.26% |
| 512 tokens now | 7.53% | 4.75% | 18.95% | 35.59% | 1.91% |

The largest remaining measured stall is memory-dependency waiting. Low occupancy leaves few independent warps
available while a load is pending. Both kernels launch 192 CTAs of 64 threads and use 255 registers per thread,
which caps theoretical occupancy at 12.5%. Instruction-fetch stalls are now small. Tensor throughput remains far
below peak, so faster MMA arithmetic alone would not address the dominant measured waits.

The 512-token FP32 kernel uses 136 bytes of local memory per thread in the ordinary benchmark. The profile confirms
executed local loads and stores. This is a regression from the previous zero-local-memory implementation, although
its measured whole-forward latency improves from 480.77 to 381.61 µs. The shorter kernel has no local loads or stores.
The existing four-chunk correctness tests still satisfy their zero-local-memory assertions in both accumulation modes.

The larger output-assembly kernel costs 90.6 µs in the ordinary 48-head, 512-token benchmark, alongside 287.1 µs for
the carry kernel. Register pressure and output assembly therefore remain worthwhile targets. These profiles cover
only the carry kernels; their instrumented durations are not used to compare whole-forward speed.

### Earlier profiles

The preceding change replaced C-fragment operand gathers with direct loads and shuffled packed FP16 pairs. At
12 heads/128 tokens it reduced dynamic shuffles by 75%, total instructions by 54%, and the instruction body from
195 KiB to 89 KiB. That solved the original instruction-fetch bottleneck while increasing global-load instructions.
The seven corresponding runs, including two eager-reference GEMMs, remain in the
[preceding report](https://github.com/cloudrift-ai/emmy/blob/99c5ad453a6d7381e4f1421ff088b9e293b8c79e/experiments/Qwen3.8-27B/gdn_profile/RESULTS.md)
and [preceding archive](https://github.com/cloudrift-ai/emmy/blob/99c5ad453a6d7381e4f1421ff088b9e293b8c79e/experiments/Qwen3.8-27B/gdn_profile/results_v100x1.tar.gz).
The [original profile archive](https://github.com/cloudrift-ai/emmy/blob/3b47f9f116f1e730afa63ddbc1b57a01404e720a/experiments/Qwen3.8-27B/gdn_profile/results_v100x1.tar.gz)
preserves the initial implementation. The previous report's DRAM percentage was the active-cycle counter; it should
not be read as the percentage of peak byte bandwidth.

### Protocol and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_profile --ssh USER@HOST --filter heads=48 --filter acc=f32`, adjusting
Python, CUDA, and Nsight paths. Omitting filters profiles both head counts and accumulation modes plus the reference.
The recipe now includes 48 heads at 512 tokens. These two selected rows both succeeded; neither was selectively rerun.

Nsight Compute 2025.2.1.0, build 35987062, collects SpeedOfLight, LaunchStats, Occupancy, SchedulerStats,
WarpStateStats, ComputeWorkloadAnalysis, MemoryWorkloadAnalysis, InstructionStats, and SourceCounters, using 22 replay
passes per kernel. Cache and clock control are disabled. Two matching launches are skipped and one is captured.
Profiling uses sudo for restricted counters and disables bytecode writes. The wrapped CLI performs strict correctness
with two warmups and three iterations; its instrumented JSON is retained for correctness and source identity only.

The supplied host has four V100-SXM2-16GB GPUs; each profile uses GPU 0. Software is Ubuntu 24.04.1,
driver 580.178.04, CUDA toolkit 12.9.86, Python 3.12.3, PyTorch 2.14.0+cu126, and Transformers 5.14.1.
The clean source revision is `59e4a07bdada4c42bb4fcc8ec3fe622e35e729a6`. Run ID is `20260920T032805Z`, directory
`2026-09-20_03-28-05/`, completed at 03:30:07 UTC. CUDA hashes match the earlier benchmark revision despite the
intervening Python formatting, test-duration, and recipe changes.

[`results_v100x1.tar.gz`](results_v100x1.tar.gz) contains two system-only records, two command-result archives,
and two logs under that directory. Each nested archive includes `profile.ncu-rep`, `counters.csv`,
`counter_instances.csv`, `sass.csv`, `details.txt`, `profiler.log`, `measurement.json`, `measurement.log`,
`status.txt`, `profiler_version.txt`, and `requirements.freeze.txt`, all under `artifacts/`.
All six outer files were byte-verified, and all 28 files and nested members were scanned for secrets.
The raw directory remains local, and the supplied host remains running.
