# RTX 5090 Qwen3.8 NVFP4 MLP tuning log

Status: manual hand sweep in progress, October 1, 2026. This log belongs to the experimental mixed serving lane in
PR #993. It records measured choices separately from ideas borrowed from earlier plans and golden files. The exact
implementation and qualification boundary is in [the mixed serving progress report](nvfp4-qwen-mixed-serving-progress.md).

## Fixed target and method

The model is `Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`, vLLM 0.23, BF16,
TP1/PP1, one RTX 5090 with driver 580.178.04. Emmy owns only the 64 text MLPs; stock vLLM owns attention, GDN,
state, and scheduling. Decode uses a static M=16 bucket and prefill a static M=64 bucket. Both permit fewer active
rows. Current explicit pins are sourced by `scripts/serve_qwen38_nvfp4_mixed_5090.sh` from
`scripts/qwen38_nvfp4_mixed_5090_knobs.sh`: `FAST_MATH=true`, five parent
workspace cuts, two per-shape child output cuts, and scoped native FP4 schedules for three contraction kernels.
Every other decision family has an explicit fallback value. No candidate is accepted from an unpinned prior pick.

The user requested a bounded manual sweep because automatic prior tuning is broken. The first pass uses existing
`emmy run --ir --bench --bench-backends emmy --warmup 5 --iters 12` on the saved exact layer-0 loop-fusion IR,
with the sourced recipe's global and shape-specific `EMMY_KNOBS` combined. The command fails before compile if
either required variable is empty or a pin cannot be audited. Timings are warmed single-layer diagnostics,
not serving latency. For routine candidates, check every intended pin, native FP4 instruction, finite output,
and same-input agreement with the qualified baseline. Finalists get an independent stock quantized MLP oracle
before paired endpoint measurements. The isolated IR and logs are on the dedicated
host under `/root/emmy-nvfp4-serving/src-merged/postmerge-probe-{16,64}` and `logs/tune-*`.

Inside the pinned `emmy-vllm023-dev:local` container with that checkout mounted at `/workspace`, the audited
baseline command is:

```sh
set -euo pipefail
source scripts/qwen38_nvfp4_mixed_5090_knobs.sh
test -n "$EMMY_KNOBS" && test -n "$EMMY_MLP_STATIC_KNOBS"
export EMMY_KNOBS="$EMMY_KNOBS,$EMMY_MLP_STATIC_KNOBS"
emmy run --ir /workspace/postmerge-probe-16/04_loop_fusion.json \
  --bench --bench-backends emmy --warmup 4 --iters 8 \
  --json /workspace/tune-audit16-final.json
```

For M=64, substitute `EMMY_MLP_PREFILL_KNOBS`, `postmerge-probe-64`, and a distinct JSON path. The run log must
have no `unreproducible pin` line; inspect all seven JSON `greedy.kernels[].knobs` rows before accepting timing.
Manual trials change one scoped key in the composed string at a time and keep the remainder identical.

## Measured starting point

| Shape | Whole MLP CUDA graph | Native gate/up pieces | Native down | Other four kernels |
| --- | ---: | ---: | ---: | ---: |
| M=16 | 165.7 µs | 35.8 + 59.9 µs | 31.6 µs | 5.7 µs summed |
| M=64 | 398.3 µs | 159.2 + 152.7 µs | 75.7 µs | 9.4 µs summed |

The M=16 table came from `tune-run-ir-baseline16.json` with twelve iterations and five warmups; M=64 from
`tune-run-ir-baseline64.json` with the same settings. Kernel medians sum to 132.9 and 397.0 µs respectively; the
whole-program CUDA graph measurement is separately timed and can differ. The three contractions account for about
96% and 98% of summed kernel time. Tuning therefore starts with their WORK/TILE/STAGE choices. Scaling these
single-layer times to all 64 MLPs is only a rough inference: actual vLLM TPOT is 104.23 ms, so a kernel-only gain
may have a modest end-to-end effect.

Both baselines emit native `mma.m16n8k64` FP4 bodies. The M=16 first gate/up-derived kernel has 272 blocks,
64 threads, 45 KiB shared memory, 114 registers and 8% reported occupancy; the second has 1088 blocks,
32 threads, 15 KiB, 70 registers and 12% occupancy. The M=64 counterparts have 4352 and 2176 blocks,
32 threads, 15/25 KiB, 74/106 registers and 12%/8% occupancy. The down kernels have 160/640 blocks,
64 threads, 15 KiB, 54 registers and 25% occupancy. These observations suggest bounded WORK geometry and
stage-depth trials before tuning the 1–4 µs encode/scales kernels.

The first `emmy run --ir` pass flagged the current recipe as `unreproducible pin` even though scoped native rows
were realized. Two audit issues were identified: child-site `PLACE@place_<token>/...` receipts were stored under
their local site names, and a bare family fallback was compared against kernels with an overriding scoped pin.
The shared audit now records exact applied child source key/value receipts after choosing the cut and checks bare
schedule pins only on kernels without an overriding scoped pin. Focused tests reject stale/mismatched keys and
accept applied cuts and fuses. The audit then exposed a genuine unused `WORK=w1x4` fallback: all three WORK
choices were scoped to native contractions; the four encode/scales kernels stamp WORK OFF. The fallback is now
explicitly `WORK=`. Both shapes pass the fail-closed pin audit, and their seven schedules and resources match the
old baseline. New eight-iteration/four-warmup checks measured M=16 167.4 µs and M=64 399.9 µs whole-program
latency, consistent with the provisional baseline. One failed audit attempt accidentally sourced no pins due shell
quoting and ran a 1.19-second scalar route. It is excluded from all comparisons; the reusable knobs file now
prevents that failure. This audit fix is a measurement gate, not a numerical or performance fix.

## Candidate sources and bounded sweep

The older [Qwen performance plan](nvfp4-qwen-performance.md) and its `pins-cannot-target-one-piece.md` report
suggest native FP4 `f1x2/k4` and `/k8`, async depth 1–4, and WORK widths around `w1x1` to `w1x4` for decode.
Those are historical hypotheses, not results on this graph. The report's older scoping limitation is partly
addressed by this PR's generic child-site PLACE pin, but each final piece identity must still be rechecked after
IR changes. The hardware 5090 golden `emmy/compiler/pipeline/search/golden/records/rtx5090_sm120.json` has
FP16 MLP rows with larger N fragments and TMA depth 2–4; Gemma 4 and OLMoE 5090 model goldens also suggest
WORK layouts and `t128`–`t512` encode rows. Their FP16 MMA atoms cannot be copied into NVFP4 W4A4, so only
geometry and transport choices are candidates. The existing native FP4 realization cases provide legality
examples. TMA `/p2` is a bounded alternative to async, not an assumed improvement; old #971 notes it was even
or slower in related decode/prefill shapes.

| Trial | Exact knob change from recipe | M=16 result | M=64 result | Decision |
| --- | --- | --- | --- | --- |
| Baseline | Explicit source file, `WORK=` fallback; native `f1x2/k4`, `d2/smem-async` | 167.4 µs whole graph (8/4) | 399.9 µs whole graph (8/4) | Accepted audited comparison; same seven schedules as 12/5 baseline. |
| Wider first gate/up N fragment | Only M=64 `TILE@place_c0904cfc6e=mma_m16n8k64_e2m1_f32/f1x4/k4`, fixed `WORK=w1x1`, `STAGE=d2/smem-async` | Not tried | Compile refused: “its kernel pins … leave no schedule row this kernel offers.” | Reject this tuple; no timing or instruction claim. |

The sweep will stop weak or refused families instead of repairing the general prior. Rejections will be recorded
with the exact piece scope, pin tuple, compiler message, emitted instruction result, and observed time if runnable.
No candidate gets a serving claim from isolated timings alone.

## Serving and correctness gates still to run

- Compare the winning layer-0 M=16 and M=64 outputs with stock vLLM's quantized MLP on the same checkpoint/input;
  retain the existing independent numeric diagnostic and report RMS/max/finite results.
- Rebuild all 64 Emmy MLP programs and verify three intended native contraction kernels in both shape buckets,
  with no stale scoped pins or unpinned schedule choices.
- Repeat the matched warm 5-input/16-output `vllm bench serve` workload against stock and the current baseline.
  Add fixed realistic ~500–1000-token summarization (128-output budget) and ~2500–3500-token extraction
  (64–128-output budget) requests. Report actual token counts, repeated warm TTFT, TPOT/decode rate, total wall
  time, and raw task answers under identical settings. Check expected content, not greedy token identity.
- Retain a long near-4K context probe and cancellation followed by a fresh request. Restore one healthy mixed
  container on host port 8080 when measurements finish.

## Automated review disposition

The review claim that symbolic prefill is fixed at eight rows is incorrect: eight rows is a trace example.
`MLPPrograms` gives axis 0 the symbolic `num_tokens` with capacity 64, and `run_device_sym` binds the actual row
count and returns the actual prefix. The deployed vLLM adapter currently chooses the static M=64 bucket; the
symbolic path remains available for future integration. A strengthened CPU trace test checks the symbolic axis.
No >8-row symbolic GPU execution is claimed by that test.
