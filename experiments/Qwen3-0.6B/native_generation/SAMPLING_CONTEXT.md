# Seeded sampling and longer context checks

GPU temperature/top-p sampling passes independent distribution and request-reset tests. All seventeen checkpoint cases
pass across 10,585 positions, including two 4,096-position prompts. The fixes retain FP32 attention and rotary
intermediates and accumulate residuals in FP32 before FP16 projections. No model acceptance threshold was changed.
Native HTTP serving is not implemented by this change.

## Scope and controls

The implementation adds request-local temperature, top-p, and a uint64 seed to generation artifact version 2. Greedy
remains the default. Sampling stays on the GPU, and captured execution preserves the seeded sequence. Older generation
artifacts require re-export. The CLI exposes the existing operation deadline through `--timeout`; the default remains
120 seconds, which may be insufficient for long sequential prefills.

Qualification uses dense FP16 Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, an RTX 4080 with 16
GiB VRAM, and the same FP16/FP32 reference construction and error budgets described in [RESULTS.md](RESULTS.md). Both
references consume the native-selected prefix and use FP16-rounded checkpoint weights. TF32 and reduced-precision
reference reductions are disabled. The final artifact has capacity 4,096 and uses the implementation at `10d35c83`.
Weights, projections, logits, and KV storage remain FP16. Residual storage is FP32, using the existing attention-split
wrapper support. FP32 rotary tables add 2 MiB at this capacity. Attention already used FP32 shared storage, so
removing its intermediate rounding adds no attention scratch storage. The changed residual regime has fresh measured
schedules.

The independent sampler test uses 257 logits, including a partial final histogram block, and 1,024 seeds for each of
15 input/control combinations. It compares observed frequencies with a separately sorted NumPy nucleus distribution,
using six binomial standard deviations plus one sample as the bound. It covers signed-zero ties, extreme finite
logits, greedy selection, temperature extremes, invalid logits, and graph replay. The tiny-model test also checks
repeated seeds and changed seeds across consecutive captured and uncaptured requests.

## Numerical investigation

The first matrix passed eleven of fourteen cases. All 5,939 positions executed, but three exceeded the unchanged
absolute budgets. Matching a reference argmax everywhere did not excuse these failures:

| Case | Initial failing position | Initial metric | Fixed limit |
| --- | ---: | ---: | ---: |
| Explanation | 8 | probability TV 0.024942 | 0.02 |
| Held-out repeated context | 146 | relative L2 0.022207 | 0.02 |
| 4,096 positions | 1,355 | relative L2 0.034985 | 0.02 |

Attention rounded its dot products, scaled scores, and normalized probabilities to FP16, following eager's storage
boundaries. Same-input checks against float64 attention exposed larger local errors than the compiled layer fragments.
Keeping those intermediates in FP32 fixes the two short failures. An independent regression uses scores near 1,024
whose 0.125 difference disappears in FP16; the old implementation returns zero instead of 0.005524 in every output
element. The FP32 implementation passes the same `1e-3` tolerance. Random cache checks also pass through 4,096
positions.

This first repair was insufficient: a sixteen-case matrix passed fourteen cases and failed both long prompts. The
original outlier improved to relative L2 0.022751; the newly selected long prompt reached 0.026173. Aggregate errors
were smaller than the FP16 control, but the per-position limits still failed. These results are retained.

At the original outlier, supplying prior cache history from the FP32 reference, rounded to the native cache dtype,
reduces relative L2 from 0.022746 to 0.004672. This diagnostic uses a full-prefix FP32 reference, so its last digits
differ slightly from the cached qualification reference. It points to accumulated history error rather than a large
same-input error in the current token's compiled pre/post or head programs.

Rotary computation was another source of avoidable rounding: both tables, both products, and the sum used FP16. The
final implementation prepares tables in FP32 and computes rotation in FP32, rounding only the query/key outputs to
FP16. An independent float64 rotary calculation checks the final rounding. The isolated original outlier drops to
relative L2 0.006981 without changing weights or compiled schedules. Tiny-model logits still pass `rtol=atol=1e-3`.
This replaces the earlier choice to reproduce eager's FP16 rotary boundaries with a more accurate intermediate
calculation; the earlier experiment remains historical evidence, not the final arithmetic contract.

The rotary-only matrix was stopped after another position exceeded the same limit: position 98 of the original long
prompt reached relative L2 0.020092. Isolating this shorter prefix again showed cache-history amplification. Using the
existing FP32 residual wrappers prevents layer sums from repeatedly rounding to FP16. Normalized inputs cast back to
the projection weight dtype; weights and KV cache stay FP16. Fresh strict fragment checks cover this new regime. The
two isolated outliers fall to relative L2 0.002871 and 0.002362. This was another implementation repair, not an
acceptance-budget change.

## Final checkpoint qualification

All seventeen cases pass the unchanged acceptance contract. Maximum relative L2 error is 0.016238 and maximum
probability TV is 0.008453, both in the new 4,096-position prompt. Per-case native/reference RMS ratios are at most
0.685 for relative L2 and 0.779 for probability TV. Native argmax agrees with both references at 10,584 positions; at
the remaining position it agrees with FP16, as permitted where the references disagree.

Python/Rust logits are bit-identical at all 552 checked positions: France, arithmetic, and the new 512-position case.
Both 4,096-position requests and the 1,024-position request pass the subsequent short-request reset assertion. The
full matrix took 811.71 seconds. Its assertions passed; the duration registry initially rejected the new 56.23-second
test, whose measured duration is now recorded.

The original fourteen cases were reused after the repairs. Two additional cases were selected after the attention fix;
their long case exposed the remaining error before the rotary fix. A fresh 512-position case was selected after the
rotary fix and first measured after the residual fix. It additionally checks exact Python/Rust logits beyond the old
short-prefix coverage. These checks establish this checkpoint/GPU/configuration's tested contract, not identical
future completions or production model quality. Qualification durations include both reference models and diagnostic
transfers; they are not native generation latency measurements.

## Executor corrections

Longer Python diagnostic replay exposed a separate shared-memory bug. Python supplied dynamic shared memory only above
48 KiB, omitting attention's 16 KiB allocation. Rust supplied total storage as dynamic bytes, reserving static storage
twice. Both executors now subtract the cubin's static allocation once at load time and pass the remainder on every
launch. Static, dynamic, and mixed-storage regressions pass in Python and Rust, captured and uncaptured.

The shared supervisor also needed a protocol correction after main began forwarding compiler precision settings. Those
settings belong to the Python message encoder. Native commands no longer receive an unsupported `fast_math` field;
prepared Python reference commands retain it outside their nested command.

## Current compiler evidence

Main changed after the earlier qualification. PR #861 retired 194 native-baseline records in 20 kernel sets because
identity updates could not preserve their measurements. That golden now contains three of the original eight programs.
Its incomplete coverage cannot reproduce the earlier full-model qualification. Strict export detects a missing pre-
attention cut decision; unconstrained post-attention execution reaches its watchdog.

Fresh static pre-attention, post-attention, and final-normalization/head targets were captured with the existing trace
command. Existing explicit cut controls materialize their intermediates. The run command checks each fragment against
eager execution and records the selected kernel set in isolated working evidence. No compiler arithmetic change or
canonical golden replacement was made. This is diagnostic qualification, not a hybrid/MCTS search or an optimization
result.

The first valid schedules used atomic cross-CTA reductions. They passed fragment accuracy but failed Python/Rust bit
identity on the same full-model artifact. The final artifact uses `FAST_MATH=0` and an empty `REDUCE` pin for unsplit
reductions. Complete export passes strict evidence. These deterministic choices cost more: post-attention program time
was about 1.21 ms versus 0.25 ms with atomic reductions, and the head was about 2.13 ms versus 1.32 ms. These single-
run fragment observations do not establish serving performance or an optimized configuration. The final FP32-residual
inventory is measured separately: pre-attention 94.9 µs, post-attention 1,206.2 µs, and head 1,090.4 µs. The head also
selects a different schedule, so its change cannot be attributed to precision alone. Both working goldens cover native
one-token fragments, not the complete vLLM serving matrix; neither is promoted as optimized deployment evidence.

## Reproduction and evidence

[Initial measurements](evidence/sampling-context.json.gz), [attention-only measurements](evidence/attention-
fp32-context.json.gz), [interrupted rotary-only measurements](evidence/rotary-fp32-context.json.gz), [final
measurements](evidence/residual-fp32-context.json.gz), [cache isolation before rotary](evidence/attention-cache-
isolation.json.gz), and [cache isolation after rotary](evidence/rotary-cache-isolation.json.gz), and [residual
isolation](evidence/residual-long-isolation.json.gz) preserve both failures and repairs. [Initial fragment
checks](evidence/sampling-fragments.json.gz), [final fragment checks](evidence/residual-fragments.json.gz), the [final
working golden](evidence/sampling-residual-golden.yaml.gz), and [package provenance](evidence/sampling-
provenance.json) retain the preparation evidence. No hostname, account path, device UUID, or weights are published.

```bash
gzip -dc experiments/Qwen3-0.6B/native_generation/evidence/sampling-residual-golden.yaml.gz \
  > /tmp/native-sampling-golden.yaml
EMMY_FAST_MATH=0 EMMY_REDUCE= emmy generate Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --context-length 4096 \
  --golden /tmp/native-sampling-golden.yaml --strict-evidence --export-native /tmp/native-qwen3
PATH="$PWD/target/release:$PATH" ./venv/bin/pytest \
  tests/serving/native/test_generation_gpu.py::test_checkpoint_logits_and_completions \
  --native-checkpoint /path/to/checkpoint --native-artifact /tmp/native-qwen3 \
  -n 2 --dist=loadgroup --durations=0 --durations-min=0.5 -p no:randomly
```

## Repository validation

Focused checks pass for tiny-model logits and request sampling, independent attention and rotary calculations, and
static/dynamic/mixed shared memory in both executors. The repository suite, lint, Rust gates, and CI outcomes are
recorded in [PR #876](https://github.com/cloudrift-ai/emmy/pull/876).
