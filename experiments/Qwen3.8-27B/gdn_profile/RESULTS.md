# Qwen3.8 GDN register-kernel profiles on V100

## V100 SXM2 16 GB — 2026-09-19

The carry kernel spends most of its instructions preparing matrix operands, while instruction delivery and low
occupancy keep the tensor cores mostly idle. The FP16 option reduces HMMA instructions but adds more promotion and
repacking work than it removes. These profiles explain why register storage alone did not improve the
[unprofiled benchmark](../gdn_register/RESULTS.md).

### Protocol

The recipe profiles the same inter-chunk correction and state update, with 64-token chunks, 128 key dimensions,
128 value dimensions, FP32 inputs and carry, two warps per CTA, and `d1/reg`. The atom is
`mma_m8n8k4_f16_f32/f1x8/k4` or `mma_m8n8k4_f16_f16/f1x8/k4`. FP16 partial sums promote every four MMA steps,
covering sixteen scalar products. Inputs, seeds, and compiler source match the ordinary timing experiment.

Nsight Compute 2025.2.1 profiles GPU 0 on the supplied four-V100 host. The driver restricts counters to administrative
processes, so the recipe uses an elevated profiling process without changing the driver configuration. It wraps the
existing `emmy run --bench --strict` command; there is no separate workload or timing implementation. Eager PyTorch
provides correctness. Nsight skips two matching launches and collects one carry-kernel launch, or two GEMM launches
for the reference probe. Kernel replay collects nine standard analysis sections, including scheduler, warp-state,
instruction, memory, and source counters. GPU clocks remain unchanged and caches are not deliberately flushed.

**Timings collected under Nsight are not benchmark results.** Instrumentation changes execution, particularly CUDA
graph replay. The unprofiled experiment remains the performance comparison. The percentages below describe Nsight
counters for the selected kernel, not a speedup estimate or a whole-model measurement.

### Hardware counters

| Heads | Tokens | Accumulation | Achieved occupancy | Tensor activity | DRAM throughput | Instruction-fetch stalls |
| ---: | ---: | --- | ---: | ---: | ---: | ---: |
| 12 | 128 | FP32 | 3.41% | 0.68% | 4.37% | 44.1% |
| 12 | 128 | FP16 + promotion | 3.12% | 0.64% | 4.22% | 46.5% |
| 48 | 128 | FP32 | 7.48% | 1.28% | 9.07% | 61.0% |
| 48 | 128 | FP16 + promotion | 7.48% | 1.17% | 8.47% | 64.6% |
| 12 | 512 | FP32 | 3.12% | 0.68% | 4.82% | 59.3% |
| 12 | 512 | FP16 + promotion | 3.12% | 0.61% | 4.29% | 42.6% |

Achieved occupancy is active warps as a percentage of the hardware limit. Tensor activity is the tensor-pipeline
active-cycle counter normalized over elapsed cycles. The instruction-fetch column divides the `no_instruction`
warp-state cycles per issued instruction by all warp cycles per issued instruction. It is not a percentage of
kernel wall time. Nsight's `no_instruction` category includes waiting to fetch an instruction and instruction-cache
misses; this counter alone does not separate those causes.

Schedulers have no eligible warp on 85–91% of active cycles. Fixed-latency dependency waits add another 10–23% of
warp cycles per issued instruction. Register counts are 168/195 at 128 tokens and 255/209 at 512 tokens, matching
the unprofiled measurements. The schedule avoids spills but leaves very few warps to hide these stalls.

The memory accesses also need improvement: Nsight attributes 64% of theoretical global sectors to uncoalesced
accesses in every carry profile. That is not 64% extra DRAM traffic. Cache reuse keeps DRAM throughput at 4–9%,
and the instruction-fetch stalls dominate the recorded warp-state breakdown.

### Reference GEMMs

The two eager-reference launches are cuBLAS FP32 SGEMMs from the same twelve-head, 128-token workload. They each
perform one matrix multiplication; they are not the entire chunk loop, so their times are not divided by the
carry-kernel time to claim a speedup. They illustrate a different instruction schedule:

| Reference kernel | Threads per CTA | Registers per thread | Achieved occupancy | Scheduler issue activity | Instruction-fetch stalls |
| --- | ---: | ---: | ---: | ---: | ---: |
| NN SGEMM | 256 | 57 | 12.48% | 50.37% | 1.19% |
| NT SGEMM | 256 | 57 | 12.48% | 45.27% | 1.36% |

These use FP32 arithmetic rather than tensor cores. Their grids also have only 24 or 48 CTAs, so the small grid
alone does not explain Emmy's loss. More warps per CTA, fewer registers per thread, and much lower instruction-fetch
stalls allow substantially more frequent instruction issue. Emmy's scheduler issue activity is only 8.6–14.7%.
Their SASS bodies contain 632 and 624 unique instruction addresses, spanning 10,112 and 9,984 bytes: about 10 KiB
each, compared with roughly 195–215 KiB for the fused carry kernels below. The operations differ in scope, but the
code-size and instruction-fetch contrast supports reducing the emitted instruction body before changing carry storage.

### Instruction and code size

At twelve heads and 128 tokens, the executed warp-instruction counts are:

| Operation | FP32 accumulation | FP16 + promotion |
| --- | ---: | ---: |
| All instructions | 2,381,952 | 2,633,376 |
| HMMA | 196,608 | 98,304 |
| Warp shuffle | 786,528 | 884,832 |
| FP32 add | 6,144 | 104,448 |
| HADD2, including promotion conversions | 0 | 98,304 |

FP16 halves the HMMA count but increases all instructions by 10.6%. Its extra shuffles and FP32 additions follow
from the layout-aware promotion, which must redistribute the FP16 accumulator before adding it into the FP32 shadow
layout. The instruction increase is consistent with its 7–12% slowdown in the unprofiled benchmark, although
instruction counts are not cycle costs.

The FP32 kernel has 12,464 SASS instructions spanning 199,424 bytes, about 195 KiB. FP16 has 13,760 instructions
spanning 220,160 bytes, 215 KiB. The emitter expands the K steps and output fragments into straight-line code within
the chunk loop. That large instruction body, together with the high `no_instruction` fraction, points toward
instruction-cache pressure. A smaller emitted loop is the experiment needed to establish the causal speedup.

For FP32, shuffles alone are 33.0% of executed instructions; HMMA is 8.3%. Shuffles, selects, conversions, permutes,
shifts, and bitwise logic together account for 85.3%. This is an instruction mix, not a claim that all those
instructions are redundant or that they consume 85.3% of runtime.

### What to change first

1. Reduce the emitted instruction body. Reuse packed operands across several independent accumulators and keep K
   traversal compact instead of spelling each fragment's entire contraction separately.
2. Load external operands directly into the Volta MMA operand layouts. The current generic register path first loads
   FP32 C-layout fragments, then converts and shuffles them into A/B layouts. Existing direct operand loaders can
   avoid that work for streamed inputs. Register repacking remains necessary for computed matrix results.
3. Reduce register pressure and retile. The twelve-head grid has only 48 two-warp CTAs for 80 SMs. More tokens extend
   each CTA's ordered loop without adding parallel work. More independent CTAs or more useful instruction-level
   parallelism are needed after reducing the register and instruction costs.
4. Revisit the FP16 promotion interval after those changes. The tested interval is too expensive to produce a win;
   increasing it needs another eager-reference accuracy check on the intended inputs.

The state can remain in FP32 registers during these changes. The counters do not establish that shared-memory carry
would be faster. Input staging or prefetch is a separate choice that may improve the remaining load latency.

### Reproduction and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_profile --ssh USER@HOST`, adjusting the existing Nsight, CUDA, and Python
paths to the supplied host. The machine and Python environment are the same as the ordinary V100 benchmark:
Tesla V100-SXM2-16GB, 80 SMs, driver 580.178.04, CUDA toolkit 12.9.86, Python 3.12.3, PyTorch 2.14.0+cu126,
and Transformers 5.14.1. Nsight reports version 2025.2.1.0, build 35987062.

The measured source is `6816a06042471b9754975e6c17e4ef588226462b`, with clean staged inputs. Its compiler source is
unchanged from the ordinary benchmark. The run started at 20:43:22 UTC and finished at 20:54:07 UTC, with run ID
`20260919T204322Z` and local directory `2026-09-19_20-43-22/`. All seven rows succeeded, and all seven command JSON
records pass strict correctness. The profiles contain six carry launches and two reference GEMM launches. No selected
row is missing or was selectively rerun. An initial twelve-head FP32 probe independently showed 43.6% instruction-fetch
stalls and 0.69% tensor activity; the tables above use only the complete recorded run.

[`results_v100x1.tar.gz`](results_v100x1.tar.gz) contains the complete raw directory under
`2026-09-19_20-43-22/`: seven `<variant>_<row_id>.experiment.yaml` system records, seven
`<variant>_<row_id>_artifacts.tar.gz` command results, and two run logs. Each nested archive includes
`artifacts/profile.ncu-rep`, `counters.csv`, `counter_instances.csv`, `details.txt`, `profiler.log`,
`measurement.json`, `measurement.log`, `status.txt`, `profiler_version.txt`, and `requirements.freeze.txt` under
`artifacts/`. All records and declared raw results are preserved unchanged. The supplied host is retained, with all
four GPUs idle and no device memory allocated after profiling.

Opcode names and SASS code size can be recovered from the saved report with Nsight's `--import`, using
`--page raw --csv --print-metric-instances details` for opcode counts and `--page source --print-source sass --csv`
for instructions. Saving the binary report is necessary because the benchmark worker consumes some profiler console
output. The recipe exports counters from that report after execution, instead of depending on console forwarding.
