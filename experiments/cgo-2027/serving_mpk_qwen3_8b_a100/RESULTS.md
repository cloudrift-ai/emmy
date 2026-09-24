# MPK Qwen3-8B on A100

## Conclusion

Within Mirage's paired demo harness, the persistent-kernel path materially outperformed Mirage's own
kernel-per-operator path. Across five fresh-process repeats, mean reported per-token latency fell from 33.136 ms to
9.804 ms: a 3.38x speedup, or a 70.4% latency reduction. The repeat ranges do not overlap, so this result is clear for
the exact single-request Mirage workload tested here.

The stock vLLM reference reported 10.752 ms/token. Mirage's persistent-kernel latency was 8.8% lower, but this is only
directional: the Mirage and vLLM lanes used different prompts, generation lengths, termination rules, client paths,
and metric implementations. This run does not establish that MPK outperforms vLLM in an equivalent serving workload.

The lane was rerun in full on a 40GB A100 on 2026-09-11 and reproduces this shape at that part's
lower bandwidth: 3.022x controlled speedup, 0.906 normalized against stock. See the 40GB section
below, which also records the two environment defects that rerun exposed and why the lane still has
no Emmy arm.

Correctness remains the main unresolved issue. The Mirage baseline produced the same 274-token response in every
repeat, while the persistent-kernel path produced 258–273 tokens with visible wording changes despite temperature
zero. The responses were coherent and on-topic on inspection, but the harness performed no token, logit, or numerical
equivalence check. The run therefore demonstrates a strong latency result, not correctness parity.

## Measurements

| Lane | Repeats | Reported latency, mean | Per-repeat range |
| --- | ---: | ---: | ---: |
| Mirage kernel-per-operator | 5 | 33.136 ms/token | 32.632–33.442 ms/token |
| Mirage persistent kernel | 5 | 9.804 ms/token | 9.785–9.865 ms/token |
| Stock vLLM | 5 | 10.752 ms/token | 10.74–10.76 ms/token |

All five stock vLLM repeats completed eight requests with zero failed requests. Its per-token latency range spans only
0.02 ms/token, while the persistent-kernel latency range spans 0.080 ms/token. The directly comparable Mirage baseline
and persistent-kernel lanes are therefore both repeatable enough that run-to-run variation cannot explain their gap.

## The 40GB A100 rerun

The lane was rerun end to end on an **NVIDIA A100-SXM4-40GB** on 2026-09-11, as its own matrix row.
That part has 1555 GB/s of bandwidth against the 80GB part's 2039, so these numbers are internally
consistent and are **not** comparable with the 80GB rows above; they are recorded beside them rather
than merged into them.

| Lane | Repeats | Reported latency, mean | Per-repeat range | Same lane on the 80GB part |
| --- | ---: | ---: | ---: | ---: |
| Mirage kernel-per-operator | 5 | 35.595 ms/token | 35.039-36.231 | 33.136 |
| Mirage persistent kernel | 5 | 11.780 ms/token | 11.723-11.891 | 9.804 |
| Stock vLLM 0.23.0 | 5 | 13.008 ms/token | 13.00-13.01 | 10.752 |

The controlled MPK speedup is `35.5948 / 11.7800 = 3.022x`, against 3.380x on the 80GB part. Every
arm is slower and the speedup smaller, which is what the bandwidth ratio predicts. All five stock
repeats completed eight requests with zero failures, and their TPOT spread is 0.01 ms/token.

Against stock, the megakernel's normalized latency is 0.906 here and 0.912 on the 80GB part. The
cross-system caveat from the 80GB run applies unchanged: the Mirage and vLLM lanes use different
prompts, generation lengths, termination rules and metric implementations, so that ratio is
directional and not a serving claim.

**Two environment defects the rerun exposed**, both fixed in the recipe and both of which would stop
this lane on any fresh host: vLLM's engine subprocess invokes `ninja` by name, so it must be
installed AND on `PATH`. Missing either, the engine aborts at KV-cache init with
`FileNotFoundError: 'ninja'` and every stock repeat fails. The first attempt at this rerun lost its
whole stock arm that way.

**There is still no Emmy arm.** One is written and was removed again: `emmy serve --generate` cannot
boot Qwen3-8B on sm_80. A hard `cp.async / TMA staging is single-fold` assert in the kernel
materializer is fixed on the branch that made this rerun, and two multi-channel projection refusals
remain behind it — survivable, but each retry recompiles for about ten minutes and the boot does not
reach a healthy server. Tracked as issue #785, whose acceptance test is that arm going back in.

### Durable files for this row

- Experiment record: `a100x1_1c0334106d66.experiment.yaml`
- Raw-results archive: `results_a10040.tar.gz`; SHA-256 `c6f6aa90bb5e75dacaa56fba823d877a9770bf65af24a8c0e8725afeb8d49866`
- Reconstruction: `paper-baselines-a10040.csv` (every retained per-repeat value) and
  `paper-table-a10040.csv` (the two comparisons)

## Reconstructing the normalized paper table

`paper-baselines.csv` records every value read from the 15 retained text outputs. The arithmetic means are 33.136
ms/token for MPK's kernel-per-operator path, 9.8038 ms/token for its megakernel, and 10.752 ms/token for vLLM 0.23.0.
The controlled MPK speedup is therefore `33.136 / 9.8038 = 3.3799x`.

`paper-table.csv` contains the two comparisons used by the paper. The [MPK paper](https://arxiv.org/abs/2512.22219)
reports 12.5 ms/token for MPK and 14.5 ms/token for its vLLM-or-SGLang baseline, giving 0.8621. The replayed row divides
the 9.8038 ms/token megakernel mean by the current vLLM mean, giving 0.9118. Lower normalized latency is better. The
file also retains the controlled comparison against MPK's own kernel-per-operator path, which is the source of the
3.38x result.

`paper-support.csv` freezes the two inputs to the paper's memory-bandwidth calculation: 15.14 GB of unique BF16
weights read per decode step and 1565 GB/s from a separate A100 device-to-device copy measurement. Their quotient is
9.674 ms/token. The MPK archive does not contain the separate copy-benchmark log, so this CSV preserves the reported
input rather than presenting it as part of the MPK replay.

To audit the repeat table, extract `results.tar.gz` and inspect `2026-08-14_20-24-20/`. MPK's two lanes report
`per-token latency` in `a100x1_mpk_{base,mega}_r*.txt`; vLLM reports `Mean TPOT` in `a100x1_stock_r*.txt`. The two
metrics come from different harnesses, which is why only the MPK kernel-per-operator versus megakernel ratio is a
controlled speedup.

The stock server log records `enforce_eager=False` and `cudagraph_mode=FULL_AND_PIECEWISE`, followed by completed
piecewise and full CUDA Graph capture. The current vLLM baseline therefore does not pay repeated Python launch cost.

## Protocol and limitations

- All lanes ran Qwen3-8B on the same NVIDIA A100-SXM4-80GB. Mirage was pinned to revision
  `5c28cc68dc621cc9448c5c9882ef9e21fdc85884`; stock vLLM used version 0.23.0 and model revision
  `b968826d9c46dd6066d109eabc6255188de91218`.
- The two Mirage lanes are the controlled comparison: one fixed 39-token prompt, one request, natural EOS, and the
  same demo with only `--use-mirage` changed. Each repeat used a fresh Python process.
- The stock lane used the HTTP serving path with eight sequential requests per repeat, fixed 128-token inputs, fixed
  512-token outputs, concurrency one, and EOS ignored. Each repeat started a fresh vLLM server.
- Mirage's model load did not explicitly pass the recorded model revision. Both Mirage lanes shared the same cached
  model, preserving their paired comparison, but exact model-revision parity with stock vLLM is not proven.
- The two harnesses do not define per-token latency identically, and the vLLM client/server path adds work absent from
  the in-process Mirage demo. Their cross-system comparison should not be treated as a serving claim.
- Every persistent-kernel repeat compiled successfully but emitted the same NVCC warnings, including a warning about
  dynamic initialization of function-scope static shared memory. No runtime failure followed, but the run contains no
  separate validation of that warning's effect.

## Run and system

- Status: succeeded
- Timestamp: 2026-08-14T20:24:20Z
- Run ID: `20260814T202420Z`
- Experiment row: `serving_mpk_qwen3_8b_a100/a100x1`
- Git revision: `9a485df4229e0529720a3b46e1d2fc482e97a394`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; NVCC: `12.9.86`; cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

## Durable files

- Paper reconstruction: `paper-baselines.csv`, `paper-table.csv`, and `paper-support.csv`
- Experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Raw-results archive: `results.tar.gz`
- Archived root: `2026-08-14_20-24-20/`
- Raw measurements: `a100x1_mpk_base_r0.txt` through `a100x1_mpk_base_r4.txt`,
  `a100x1_mpk_mega_r0.txt` through `a100x1_mpk_mega_r4.txt`, and `a100x1_stock_r0.txt` through
  `a100x1_stock_r4.txt`
- Supporting evidence: runner/server logs, environment freezes, model-cache inventory, and MPK installation log are
  included in the archive.
