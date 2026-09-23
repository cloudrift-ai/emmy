# Seeded sampling and longer context checks

GPU temperature/top-p sampling passes its independent distribution and request-reset tests. Full-checkpoint
qualification remains blocked: 11 of 14 cases pass the existing contract, while three exceed its fixed numerical
limits. No acceptance threshold was changed. Native HTTP serving is not implemented by this change.

## Scope and controls

The implementation adds request-local temperature, top-p, and a uint64 seed to generation artifact version 2.
Greedy remains the default. Sampling stays on the GPU, and captured execution preserves the seeded sequence.
Older generation artifacts require re-export. The CLI exposes the existing operation deadline through `--timeout`;
the default remains 120 seconds, which may be insufficient for long sequential prefills.

Qualification uses dense FP16 Qwen3-0.6B at revision `c1899de289a04d12100db370d81485cdf75e47ca`, an RTX 4080
with 16 GiB VRAM, and the same FP16/FP32 reference construction and error budgets described in [RESULTS.md](RESULTS.md).
Both references consume the native-selected prefix and use FP16-rounded checkpoint weights. TF32 and reduced-precision
reference reductions are disabled. The prepared artifact has capacity 4,096; the runtime and tests run at revision
`a2a804b1`, whose GPU implementation is unchanged from `9eb89328`.

The independent sampler test uses 257 logits, including a partial final histogram block, and 1,024 seeds for each
of 15 input/control combinations. It compares observed frequencies with a separately sorted NumPy nucleus
distribution, using six binomial standard deviations plus one sample as the bound. It covers signed-zero ties,
extreme finite logits, greedy selection, temperature extremes, invalid logits, and graph replay. The tiny-model test
also checks repeated seeds and changed seeds across consecutive captured and uncaptured requests.

## Checkpoint results

The matrix checks 5,939 positions, including prompt positions rather than only emitted tokens. Every native argmax
matches at least one reference. Both references and native agree at 5,938 positions; at position 747 of the longest
case, native agrees with FP32 while FP16 chooses another token. This does not establish identical future completions
or production model quality.

| Case | Result | Failing position | Observed metric | Fixed limit |
| --- | --- | ---: | ---: | ---: |
| Explanation | Fail | 8 | probability TV 0.024942 | 0.02 |
| Held-out repeated context | Fail | 146 | relative L2 0.022207 | 0.02 |
| 4,096 positions | Fail | 1,355 | relative L2 0.034985 | 0.02 |
| Other eleven cases, including 1,024 positions | Pass | — | All required checks pass | unchanged |

All 4,096 positions execute and produce finite logits. The longest case's maximum probability TV is 0.013507, but
passing that metric does not excuse its L2 failure. The 1,024-position case passes, including a short request after
its long cache history. The 4,096-case reset assertion follows accuracy assertions and was not reached after failure.
Python and Rust logits are bit-identical at the 40 positions covered by the two dispatcher-comparison cases.

The original short case passes separately. The remaining matrix reports ten passes and three failures in 448.46
seconds. The 1,024- and 4,096-position cases take 70.82 and 304.27 seconds respectively, including both reference
models and diagnostic transfers; these are test durations, not native generation latency.

## Current compiler evidence

Main changed after the earlier qualification. PR #861 retired 194 native-baseline records in 20 kernel sets because
identity updates could not preserve their measurements. That golden now contains three of the original eight
programs. Its incomplete coverage cannot reproduce the earlier full-model qualification. Strict export detects a
missing pre-attention cut decision; unconstrained post-attention execution reaches its watchdog.

Fresh static pre-attention, post-attention, and final-normalization/head targets were captured with the existing
trace command. Existing explicit cut controls materialize their intermediates. The run command checks each fragment
against eager execution and records the selected kernel set in isolated working evidence. No compiler arithmetic
change or canonical golden replacement was made. This is diagnostic qualification, not a hybrid/MCTS search or an
optimization result.

The first valid schedules used atomic cross-CTA reductions. They passed fragment accuracy but failed the stronger
Python/Rust bit-identity check on the same full-model artifact. The final artifact uses `FAST_MATH=0` and an empty
`REDUCE` pin to select unsplit reductions. Its complete export passes strict evidence, and dispatcher identity passes.
These deterministic choices cost more: post-attention program time was about 1.21 ms versus 0.25 ms with
atomic reductions, and the output head was about 2.13 ms versus 1.32 ms. These single-run fragment observations do
not establish serving performance or an optimized configuration.

The working golden is retained only to reproduce this failed full-model qualification. It covers native static
one-token fragments, not the complete vLLM serving matrix, and is not promoted as deployment evidence.

## Reproduction and evidence

[Per-position measurements](evidence/sampling-context.json.gz), [fragment checks](evidence/sampling-fragments.json.gz),
the [working golden](evidence/sampling-deterministic-golden.yaml.gz), and
[package provenance](evidence/sampling-provenance.json) preserve the result. The fragment JSON keeps
both the atomic diagnostic and deterministic runs. No machine hostname, account path, device UUID, or weights are
included.

```bash
gzip -dc experiments/Qwen3-0.6B/native_generation/evidence/sampling-deterministic-golden.yaml.gz \
  > /tmp/native-sampling-golden.yaml
EMMY_FAST_MATH=0 EMMY_REDUCE= emmy generate Qwen/Qwen3-0.6B \
  --revision c1899de289a04d12100db370d81485cdf75e47ca --context-length 4096 \
  --golden /tmp/native-sampling-golden.yaml --strict-evidence --export-native /tmp/native-qwen3
PATH="$PWD/target/release:$PATH" ./venv/bin/pytest \
  tests/serving/native/test_generation_gpu.py::test_checkpoint_logits_and_completions \
  --native-checkpoint /path/to/checkpoint --native-artifact /tmp/native-qwen3 \
  -n 2 --dist=loadgroup --durations=0 --durations-min=0.5 -p no:randomly
```

The existing supervisor also needed a protocol correction after main began forwarding compiler precision settings.
Those settings now belong to the Python message encoder. Native commands no longer receive an unsupported
`fast_math` field; prepared Python reference commands retain it outside their nested command.

## Repository validation

The full suite reports 7,291 passes, 757 skips, and one failure in 819.32 seconds. The failure was an existing test
asserting that precision metadata entered the shared supervisor before encoding. Its assertion now checks the Python
wire message, where that metadata belongs; the corrected test and seven native protocol tests pass separately.
The full suite was not repeated after that test-only correction. Python lint/formatting, Rustfmt, Clippy, and all
eight Rust unit tests pass. Model-golden decoding is included in this main revision's full suite. The opt-in
checkpoint matrix above is separate and still fails three cases.
