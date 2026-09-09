# Neptune, modern PyTorch, and Emmy on A100

## Conclusion

Manual schedules make all 40 Emmy shapes correct and runnable on the same NVIDIA A100-SXM4-40GB used by the Neptune
paper. They do not close the performance gap. Emmy is 1.20--1.23x slower than Neptune on the three prefill families,
3.54x slower on causal decode, and 6.31x slower on GQA decode by geometric mean. Neptune wins every matched shape.

Relative to PyTorch 2.13, Emmy is 1.41--1.54x slower on prefill and 3.47x slower than Inductor on causal decode. GQA
decode is the exception against eager PyTorch: Emmy is 3.16x faster. It is still 2.55x slower than Inductor on that
family. The result is therefore a successful manual qualification and a clear compiler performance gap, not parity.

The main implementation result is the causal early stop on the chunk tier: the coordinate mask the frontend adds to
the score is read once where the chunk loop opens, and the loop stops at the CTA's diagonal, so masked chunks are
skipped instead of folded while the per-element mask still guards the boundary tile. This reduced the 32768
causal-prefill row from about 130.7 ms to 70.0 ms and the GQA-prefill row from about 258.5 ms to 137.7 ms. The remaining
prefill gap is schedule and generated-code quality after masked work has already been removed.

The Neptune schedules and current PyTorch baselines also reproduce on this exact 40GB card. The five published
library-relative family ratios differ from the paper by 1.3--5.1%, with the same direction in every family. There is
no hardware-difference qualification on the current comparison.

## Manually tuned Emmy on A100 40GB

The full run measured revision `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`. A second complete GQA-decode row at
`1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c` replaces its 2048 schedule with the qualified fused schedule. No result
uses `emmy tune`: each golden was selected manually, checked with five warmups and 20 measurements, then accepted only
after two fresh correct measurements.

All 40 shapes completed two deployable-O3 strict golden replays and two source-reference measurements. Every source
run passed eager correctness at `rtol=1e-3, atol=1e-3`. Each latency is the arithmetic mean of the two repetitions;
each repetition reports the minimum of 15 captured GPU measurements. Decode latency sums the realized golden kernels.

| Operator | Emmy / eager | Emmy / `torch.compile` | Emmy / Neptune | Emmy wins vs Neptune |
| --- | ---: | ---: | ---: | ---: |
| Prefill global | 1.54x | 1.54x | 1.23x | 0/8 |
| Prefill causal | 1.41x | 1.42x | 1.20x | 0/8 |
| Prefill GQA | 1.44x | 1.44x | 1.23x | 0/8 |
| Decode causal | 3.12x | 3.47x | 3.54x | 0/8 |
| Decode GQA | 0.32x | 2.55x | 6.31x | 0/8 |

Lower ratios favor Emmy. The corresponding per-shape Emmy latency is:

| Sequence | Prefill global (us) | Prefill causal (us) | Prefill GQA (us) | Decode causal (us) | Decode GQA (us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 26.435 | 28.032 | 38.183 | 24.680 | 23.988 |
| 512 | 70.473 | 66.731 | 93.510 | 38.841 | 40.917 |
| 1024 | 203.366 | 172.459 | 274.432 | 73.169 | 76.296 |
| 2048 | 728.576 | 462.592 | 844.800 | 148.541 | 47.080 |
| 4096 | 2670.592 | 1582.592 | 3007.488 | 294.468 | 304.677 |
| 8192 | 8174.592 | 5226.496 | 8929.280 | 607.300 | 654.037 |
| 16384 | 33263.617 | 17671.679 | 35146.751 | 1384.960 | 1457.664 |
| 32768 | 172432.899 | 70041.088 | 137703.423 | 2775.040 | 2891.264 |

After the early stop moved from the loop IR to the chunk tier (revision `4598f8f17`), the same goldens replayed on the
same card: causal prefill measured 28.0 us at 256 keys (28.032 above), 61.8 at 512 (66.731), 1233.9 at 4096 (1582.592)
and 4542.5 at 8192 (5226.496); GQA prefill measured 2337.8 at 4096 (3007.488). Repeated replays of the 256-key row spread
from 22.5 to 29.4 us and the 512-key row from 55.0 to 66.9, so the short rows carry a run-to-run spread of about 20% on
this VM. The tables above keep the full-run values.

`paper-emmy-a10040.csv` contains the exact 40 Emmy, eager, Inductor, and Neptune values behind both tables. The fused
2048 GQA-decode schedule is 47.080 us, 3.15x faster than the earlier 148.215 us split schedule. The other GQA-decode
shapes retain split schedules because the fused alternative was slower in direct trials.

## Historical tuned Emmy decode-causal follow-up

The follow-up measured the eight committed decode-causal goldens at revision
`5642d020259d0e09d49cbdab04e8e96408616b3e`. All eight shapes completed two deployable-O3 golden replays and two
strict source-reference invocations: 32/32 required invocations succeeded. Every replay realized exactly the two
expected golden schedules, every source comparison passed, and the largest Emmy absolute error was `2.59e-4` under
the experiment's `1e-3` tolerance.

Each latency below is the arithmetic mean of two independently launched repetitions; each repetition reports the
minimum of 15 captured GPU measurements. Replay latency is the sum of the two golden kernels. The final column compares
the same-input untuned greedy Emmy latency recorded beside each replay, so values above 1.00x favor the tuned schedules.

| Sequence | Tuned replay (us) | Eager (us) | `torch.compile` (us) | Untuned greedy Emmy (us) | Tuned vs greedy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.522 | 13.102 | 12.651 | 73.179 | 2.98x |
| 512 | 38.306 | 17.262 | 16.603 | 149.285 | 3.90x |
| 1024 | 59.048 | 24.064 | 22.357 | 311.296 | 5.27x |
| 2048 | 125.897 | 38.912 | 33.280 | 640.000 | 5.08x |
| 4096 | 274.970 | 57.856 | 53.248 | 1281.024 | 4.66x |
| 8192 | 535.962 | 93.696 | 88.576 | 2551.808 | 4.76x |
| 16384 | 1152.640 | 163.328 | 175.616 | 5081.600 | 4.41x |
| 32768 | 2334.208 | 323.072 | 316.416 | 9318.400 | 3.99x |

Across the eight shapes, tuned replay was 4.32x faster than untuned greedy Emmy by geometric mean. It remained 3.82x
slower than eager and 4.02x slower than `torch.compile`. This row qualifies decode-causal only; it does not qualify the
other four Emmy operator families or change the broader Neptune comparison.

## Historical starter comparison

This table retains the full starter sweep. Its Emmy column describes the original untuned run; the tuned decode-causal
follow-up above is reported separately so historical failures are not rewritten.

The Neptune columns report `Inductor latency / Neptune latency`, so values above 1.00x favor Neptune. "Best" selects
the fastest measured manual or tuned Neptune schedule; "manual" uses only the artifact's fixed manual schedules. Each
summary is the geometric mean over eight sequence lengths from 256 through 32768.

| Operator | Inductor vs eager | Neptune vs Inductor, best | Neptune vs Inductor, manual | Full tunes | Valid Emmy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.02x | 0.84x | 0.82x | 8/8 | 8/8 |
| Prefill causal | 1.05x | 0.90x | 0.89x | 5/8 | 7/8 |
| Prefill GQA | 1.05x | 0.86x | 0.84x | 6/8 | 7/8 |
| Decode causal | 1.01x | 1.07x | 0.99x | 7/8 | 0/8 |
| Decode GQA | 7.67x | 3.01x | 2.90x | 8/8 | 0/8 |

Best-available Neptune beat Inductor on 15 of the 40 individual shapes: two prefill, five decode-causal, and all eight
decode-GQA setups. The decode-GQA speedup increased with context, from 1.68x at sequence 256 to 4.54x at sequence
32768. By contrast, Neptune's best prefill results ranged from 0.76x to 1.19x Inductor across individual shapes.

## Alignment with the Neptune paper

The [Neptune paper](https://arxiv.org/abs/2510.08726) reports the geometric-mean speedup of Neptune over the fastest
manually optimized library in Table 4. To compare with that table, the replay values below use the arithmetic mean of
the 15 projected GPU times for each implementation and then the geometric mean across the eight sequence lengths.
This is the closest aggregation available in the durable Nsight exports to the paper's mean-of-15 kernel rule. Values
above 1.00x favor Neptune.

| Operator | Paper, A100 | A100 40GB replay | Difference |
| --- | ---: | ---: | ---: |
| Prefill global | 0.84x | 0.80x | -5.1% |
| Prefill causal | 0.81x | 0.78x | -4.0% |
| Prefill GQA | 0.80x | 0.79x | -1.3% |
| Decode causal | 0.99x | 1.03x | +3.7% |
| Decode GQA | 1.24x | 1.21x | -2.4% |

The five comparable library results agree within 1.3--5.1%. Both the paper and this exact-card replay find that
optimized libraries beat Neptune on the three common prefill operators, while Neptune is competitive on causal
decode and ahead on GQA decode.

### Reconstructing the normalized paper table

`paper-baselines.csv` extracts the 40 shared shape measurements used by the paper. For each shape, the Neptune value
is the arithmetic mean of 15 projected GPU ranges for each available manual or tuned schedule, followed by selection
of the lower schedule mean. The Inductor value is the minimum of 15 captured, whole-forward CUDA-event measurements.
The per-family reproduced value is the geometric mean of `neptune_mean_us / inductor_min_us` over the eight sequence
lengths. This gives 1.20, 1.14, 1.15, 0.99, and 0.34 for global prefill, causal prefill, GQA prefill, causal decode,
and GQA decode, respectively. Lower values favor Neptune because current Inductor is normalized to one.

`paper-table.csv` records the published library-relative ratios, their artifact replay, and the displayed values after
putting both results on the current Inductor scale. For each operator, the bridge is

```text
paper Neptune / current Inductor
  = replayed Neptune / current Inductor
  * replayed library / Neptune
  / published library / Neptune
```

The Neptune paper publishes only two-decimal family ratios. The final column in `paper-table.csv` therefore records
the displayed paper value directly; recomputing from the rounded intermediate columns can differ by 0.01.

To audit `paper-baselines.csv`, use the sibling `compiler_neptune_replay_a100/results_a10040.tar.gz` archive. Neptune
ranges are under `nsys-stats/<operator>-b1-s<sequence>.csv`; current PyTorch JSON rows are under
`evidence/emmy-tcompile/json/`. Warmup ranges are excluded.

The paper's Table 2 reports Neptune relative to Triton, FlexAttention, TVM, and Mirage, rather than to the manually
optimized libraries. The pinned artifact revision leaves its TVM runners disabled, so this experiment cannot claim a
complete reproduction of every Table 2 cell. Its PyTorch 2.6 runners also select specialized SDPA, cuDNN, or CUTLASS
paths; they are not equivalent to the full-graph PyTorch 2.13 lane added here.

The paper does not publish per-shape latency tables or raw plot data. Its absolute attention results are throughput
plots at sequence length 8192 over varying batch sizes. Representative measurements from the 40GB replay are below,
reported as `Neptune / torch.compile` in microseconds. Neptune is the mean of 15 projected GPU ranges; Inductor is the
mean of two independently launched minimum-of-15 measurements.

| Operator | Sequence 2048 | Sequence 32768 |
| --- | ---: | ---: |
| Prefill global | 551.7 / 453.6 | 92,427.5 / 84,806.1 |
| Prefill causal | 375.4 / 334.8 | 61,207.6 / 45,048.8 |
| Prefill GQA | 619.2 / 563.2 | 100,425.6 / 88,315.4 |
| Decode causal | 45.5 / 38.9 | 410.3 / 395.8 |
| Decode GQA | 19.8 / 57.3 | 143.2 / 615.9 |

The absolute trend is coherent with the ratios: prefill remains close but favors current PyTorch, causal decode
converges toward parity, and Neptune's decode-GQA advantage grows with context length.

## Published artifact coverage

All ten operator families and all eight sequence lengths produced Nsight profiles. Sixty-four of 80 tuning jobs
completed their 128-trial search; 16 reached the 30-minute per-setup limit. Timed-out rows still profile the available
manual and partial tuned schedules and are not described as fully tuned.

The speedup column compares the fastest available Neptune schedule with the fastest valid non-Neptune runner in the
published artifact. Values above 1.00x favor Neptune. The artifact uses PyTorch 2.6, so this table characterizes
Neptune's original comparison environment; the modern Inductor comparison above is the more relevant baseline.

| Published operator | Full tunes | Profiles | Neptune vs fastest valid artifact runner | Validity note |
| --- | ---: | ---: | ---: | --- |
| Prefill global | 8/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton rows at 256–1024 |
| Prefill causal | 5/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton runner |
| Prefill GQA | 6/8 | 8/8 | 0.76x | Excludes the mismatching Tri Dao Triton runner |
| Decode causal | 7/8 | 8/8 | 1.04x | No cross-runner mismatch |
| Decode GQA | 8/8 | 8/8 | 1.21x | No cross-runner mismatch |
| Prefill ALiBi | 6/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Decode ALiBi | 8/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Prefill softcap | 4/8 | 8/8 | 1.04x | Compared with Flex |
| Decode softcap | 8/8 | 8/8 | 4.96x | Compared with Flex |
| Prefill windowed | 4/8 | 8/8 | 0.68x | Compared with CUTLASS |

The harness treats a Neptune manual schedule as its correctness reference. Agreement from the other runners supports
the non-ALiBi rows, but it is not an independent oracle for Neptune. ALiBi is therefore excluded rather than assigning
the disagreement to either side.

SoftCap decode is the remaining performance-reproduction outlier. This replay reports 4.96x over Flex, while the
paper's A100 compiler table reports 1.86x over its best compiler baseline. The modern PyTorch lane does not implement
SoftCap, so this result should not be treated as reproduced until that difference is explained.

## Protocol and limitations

- Neptune ran revision `3aa55c12ac822337e630b809b0d9eabb11eee5d3` in the pinned image
  `evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`.
  The artifact environment recorded PyTorch 2.6.0 with CUDA 12.4 and Nsight Systems 2025.3.1.
- The current Emmy lane reconstructs global, causal, and GQA attention for prefill and decode through
  `emmy run -c ... --bench`. It uses PyTorch 2.13.0 with CUDA 13.0, deployable O3, full-graph Inductor in
  `max-autotune-no-cudagraphs` mode, one warmup, 15 measured iterations, and strict correctness.
- The old starter lane used untuned Emmy and retained its failures. The current table replaces only that Emmy result;
  it does not rewrite the historical run below.
- The paper comparison uses the arithmetic mean of 15 projected Neptune GPU ranges. The PyTorch/Emmy lane uses the
  mean of two independent minimum-of-15 CUDA-event measurements. Both are GPU-time measurements from separate
  processes and software environments, so their ratios are kernel-level evidence rather than an end-to-end result.
- The softcap, ALiBi, and windowed families have no current PyTorch/Emmy twin in this experiment. Their table only
  reproduces the runners shipped in Neptune's artifact.

## Run and system

- Status: 5/5 full rows succeeded; the corrected GQA-decode row also succeeded
- Full run: `20260909T081524Z`; corrected GQA-decode run: `20260909T093117Z`
- Full-run Git revision: `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`; dirty: false
- Corrected-row Git revision: `1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c`; dirty: false
- Host: `bench-codex-a100-0908-0933-d43f`; Ubuntu 24.04.4 LTS; kernel `6.17.0-1022-gcp`
- CPU: Intel Xeon at 2.20 GHz, x86_64, 12 logical CPUs; memory: 89616363520 bytes
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, UUID `GPU-be299b90-0ff5-e1b4-db53-28465b6f874b`
- NVIDIA driver: `580.173.02`; host NVCC: `12.9.41`; host cuBLAS: `12.9.0.13`
- Docker client/server: `29.8.0` / `29.8.0`

The host was supplied for this work and remains running.

## Historical starter run and system

- Status: succeeded
- Result timestamp: 2026-08-16T00:41:38Z; run ID: `20260816T004138Z`
- Experiment row: `compiler_neptune_emmy_pytorch_a100_recovery/a100x1`; row ID: `e246bb6279fd`
- Git revision: `2550211d9c93e522ea4f9eb81e39735f4ab64d07`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; host NVCC: `12.9.86`; host cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

The source run (`20260815T040818Z`) completed all Neptune work in 71393.66 seconds, then failed because the host lane
started outside the staged repository. The successful 2353.12-second recovery verified the immutable source archive's
SHA-256 (`775fb71d3eac78f0371c1014b9945b29d17f41202347f8115e8703db5a4c14ca`), retained its failed status, and ran
only the missing host lane. The durable `recipe.yaml` contains the corrected working directory for clean future runs.

## Durable files

- Exact A100 40GB comparison: `paper-emmy-a10040.csv`
- Neptune paper reconstruction: `paper-baselines.csv` and `paper-table.csv`
- Current system records and composite task artifacts: five rows under `2026-09-09_08-15-24/` and the corrected
  GQA-decode row under `2026-09-09_09-31-17/`, both retained in the raw-results archive
- Starter experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Tuned decode-causal experiment record: `a100x1_lemmy_od-c_9f8816b4a4fb.experiment.yaml`; SHA-256
  `07a4b79cf046bfb16b766bc830974dc42f5cc291c87874c3de7f25d7fb7b81d3`
- Raw-results archive: `results.tar.gz`; SHA-256
  `0871843d4d8eb9232c3260143f7da63322a2978318895a54a570a9f91f6dafc8`
- Archived roots: `2026-08-16_00-41-38/`, `2026-08-24_22-35-24/`, `2026-09-09_08-15-24/`, and
  `2026-09-09_09-31-17/`
- Starter composite task artifact: `a100x1_artifacts.tar.gz`; SHA-256
  `015951d7cccf187c69dd2712bcaf966f3de179b53508942312a0e8e6cc31e4b5`
- Tuned decode-causal composite task artifact: `a100x1_lemmy_od-c_9f8816b4a4fb_artifacts.tar.gz`; SHA-256
  `6c288facacb05cf46b20c4d7be8a6bf56c1495a96ffdbe58c79f41b744412d4b`
- Raw evidence includes 80 `.nsys-rep` profiles, 80 CSV exports, all tune/profile logs, 40 modern PyTorch JSON rows,
  Emmy dumps and logs, all current replay/reference JSON rows, environment freezes, runner hashes, source/recovery
  status files, and all run records/logs.
