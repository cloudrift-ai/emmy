# V100 GDN register-schedule profiles

## V100 SXM2 16 GB — 2026-09-19

The optimization removes most of the shuffle and instruction-fetch overhead found in the first profile.
At 12 heads and 128 tokens, FP32 executes 54% fewer instructions and 75% fewer shuffles. Its instruction body
shrinks from 195 KiB to 89 KiB, while the instruction-fetch stall share falls from 44.1% to 3.5%.
The separate [uninstrumented benchmark](../gdn_register/RESULTS.md) measures 2.37–2.87× faster execution across
both accumulation modes. Nsight timings are not used for that comparison.

All seven new profiling runs succeed and pass strict correctness. They contain six Emmy carry profiles and two
eager-reference GEMMs. Every measured Emmy source hash matches the corresponding uninstrumented benchmark.

### Before and after

The compiler now loads materialized B operands directly into MMA fragments through the existing global-memory
loader. It retains C fragments for computed and non-unit-stride operands. Volta C→A repacking converts and packs
adjacent FP16 values before shuffling them, reducing eight shuffles to four. The chunk loop, state ownership,
intermediate-result reuse, accumulation modes, and schedule pins are unchanged.

The following counts are for the 12-head, 128-token carry kernel. Dynamic counts are warp instructions from
`sass__inst_executed_per_opcode`. Static counts are unique instruction addresses in the exported SASS.

| Measure | FP32 before | FP32 now | FP16 before | FP16 now |
| --- | ---: | ---: | ---: | ---: |
| Executed instructions | 2,381,952 | 1,087,296 | 2,633,376 | 1,330,656 |
| SHFL | 786,528 | 196,704 | 884,832 | 295,008 |
| HMMA | 196,608 | 196,608 | 98,304 | 98,304 |
| LDG | 104,640 | 202,944 | 104,640 | 202,944 |
| Static instructions | 12,464 | 5,712 | 13,760 | 6,968 |
| Instruction bytes | 199,424 | 91,392 | 220,160 | 111,488 |

Tensor instructions are unchanged within each accumulation mode. The improvement comes from operand preparation,
not less matrix work. Global-load instructions increase by 94% because the direct loader replaces the C-layout
gather and shuffle path. This is an effective trade here, but it leaves more load pressure to address next.

FP16 still executes 22% more instructions than FP32 in this case despite halving HMMA instructions. Promotion adds
98,304 shuffles, 98,304 FP32 additions, and 98,304 half-to-float conversion instructions (`HADD2.F32`). The carry and
shadow remain FP32; partial sums promote every four MMA steps, or sixteen scalar products. The uninstrumented
measurements still favor FP32 accumulation by 1.9–15.6% at these pins.

### Hardware counters

Percentages below use the same metrics and replay settings as the first profile. Occupancy is achieved active
warps relative to the hardware maximum. Tensor and DRAM columns are percent of peak sustained elapsed throughput.
Fetch is the no-instruction stall ratio divided by average warp cycles per issued instruction. It is a share of
warp cycles, not kernel wall time. The counter includes instruction-fetch waiting and instruction-cache misses;
it does not separate those causes.

| Heads | Tokens | Accumulation | Occupancy now | Fetch before | Fetch now | Tensor now | DRAM now |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 12 | 128 | FP32 | 3.12% | 44.11% | 3.54% | 1.79% | 11.70% |
| 12 | 128 | FP16 | 3.12% | 46.45% | 5.27% | 1.56% | 10.21% |
| 48 | 128 | FP32 | 7.44% | 60.95% | 3.50% | 3.33% | 24.21% |
| 48 | 128 | FP16 | 7.49% | 64.62% | 2.70% | 3.05% | 21.83% |
| 12 | 512 | FP32 | 3.12% | 59.26% | 2.44% | 1.76% | 11.77% |
| 12 | 512 | FP16 | 3.12% | 42.60% | 5.25% | 1.62% | 11.23% |

The smaller instruction bodies and lower fetch stalls support the original diagnosis. This measures the two
optimizations together; it is not an ablation assigning a separate speedup to each change. Occupancy does not
improve. Tensor utilization rises from 0.6–1.3% to 1.6–3.3%, but the kernel is still far from tensor throughput limits.

The largest remaining stall in the 48-head FP32 case is global/local memory instruction issue throttling:
48.7% of warp cycles per issued instruction, versus 3.5% for fetching instructions. At 12 heads it is 28.9% for
128 tokens; at 512 tokens, long-scoreboard waits rise to 30.9%. Fixed-latency dependency waits account for
12.5–25.6% across the six carry profiles. These are latency and instruction-issue costs; DRAM throughput peaks
at only 24.2%. The small case has nearly 100% L2 hits, and the larger cases are around 85%.

Nsight attributes 65% of theoretical global sectors in the smallest FP32 case to excessive sectors. This measures
access coalescing, not a claim that 65% of actual DRAM traffic is wasted. Direct loads have reduced instructions
substantially while increasing the number of global-load instructions. Coalescing and bounded operand reuse are
now more promising targets than further reductions in instruction-fetch stalls.

Every measured carry kernel has 64 threads per CTA and 254–255 registers per thread, with zero shared memory and
zero local-memory bytes in the ordinary benchmark. Registers cap theoretical occupancy at 12.5%. There are
48 CTAs for 12 heads and 192 for 48 heads on an 80-SM GPU. The schedule's full state slice and intermediate results
still require many live registers. Reducing that demand is necessary before smaller ownership tiles can reliably
increase occupancy without spilling.

### Eager-reference GEMMs

The reference row profiles the first two matching FP32 cuBLAS GEMMs from the same eager workload. They are
`volta_sgemm_128x32_nn` and `volta_sgemm_128x32_nt`: 24/48 CTAs, 256 threads per CTA, and 57 registers per thread.
Their achieved occupancy is about 12.5%; scheduler issue activity is 50.6% and 45.5%, versus 12.5–19.4% for Emmy.
Fetch shares are 1.37% and 1.30%. Their instruction bodies contain 632 and 624 unique addresses, about 10 KiB each.

These GEMMs use FP32 FMA rather than tensor cores. Each performs one product, while Emmy's kernel performs the
ordered chunk loop, correction, and state updates. Their kernel durations cannot be divided into Emmy's duration
to infer a speedup. The paired whole-forward benchmark remains the performance comparison.

### Protocol and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_profile --ssh USER@HOST`, adjusting the recipe's Nsight, CUDA, and Python
paths. The recipe profiles six carry combinations and one reference row with two captures. It requests
SpeedOfLight, LaunchStats, Occupancy, SchedulerStats, WarpStateStats, ComputeWorkloadAnalysis,
MemoryWorkloadAnalysis, InstructionStats, and SourceCounters. Each kernel uses 22 replay passes. Cache and clock
control are disabled. Two matching launches are skipped, then one carry launch or two reference launches are captured.

Nsight Compute is 2025.2.1.0, build 35987062, which supports Volta. Profiling runs with sudo because the driver
restricts counters to administrators. Bytecode writes are disabled so profiling does not change staged-source
ownership. The wrapped CLI performs strict correctness and uses two warmups and three iterations; its instrumented
benchmark JSON is retained for correctness and source identity, not latency comparison.

The supplied host has four V100-SXM2-16GB GPUs; each profile uses GPU 0. Software is Ubuntu 24.04.1,
driver 580.178.04, CUDA toolkit 12.9.86, Python 3.12.3, PyTorch 2.14.0+cu126, and Transformers 5.14.1.
The run uses clean source `af3e1e04b2b1737e66b890833fb2b4b276b7c953`; compiler code is unchanged from the benchmark
revision. Run ID is `20260919T212816Z`, directory `2026-09-19_21-28-16/`, completed at 21:34:24 UTC.

[`results_v100x1.tar.gz`](results_v100x1.tar.gz) contains seven system-only records, seven command-result archives,
and two logs under that raw directory. Each nested archive includes `profile.ncu-rep`, `counters.csv`,
`counter_instances.csv` with named instruction counts, `sass.csv`, `details.txt`, `profiler.log`, `measurement.json`,
`measurement.log`, `status.txt`, `profiler_version.txt`, and `requirements.freeze.txt`, all under `artifacts/`.
All sixteen outer files are byte-verified. The original profiles remain in the
[before archive](https://github.com/cloudrift-ai/emmy/blob/3b47f9f116f1e730afa63ddbc1b57a01404e720a/experiments/Qwen3.8-27B/gdn_profile/results_v100x1.tar.gz).
The raw directories remain local, and the supplied host remains running.
