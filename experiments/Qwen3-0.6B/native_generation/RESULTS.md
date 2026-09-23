# Native cached-generation qualification

The [sampling and longer-context follow-up](SAMPLING_CONTEXT.md) records current-compiler checks through 4,096
positions. Sampling passes, but three checkpoint cases fail the unchanged accuracy contract; full qualification
remains open. The results below describe the earlier artifact and compiler revision.

Dense FP16 Qwen3 now runs cached generation through Rust using the same compiled programs as Python. A rotary
rounding defect is fixed. Broader checks use an explicit FP32-based accuracy contract; the original pointwise gate
and an exploratory per-position reference comparison failed and remain recorded below. All four fresh held-out
cases pass the revised contract. Repository checks found 26 failures reproduced on main; the PR remains draft.
This is not a performance or production-serving qualification.

## Scope and acceptance contract

Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, FP16, one RTX 4080 (`sm_89`). The new artifact
has capacity 256 and uses the pre/post schedules from the sibling native-baseline experiment. Its kernel sources
match the original capacity-32 artifact. Final norm/head selection has no strict measured-evidence requirement.
No schedules are retuned here.

Hugging Face eager FP16 and FP32 references receive the same native-selected prefixes. FP32 uses the same
FP16-rounded weights, with TF32 and reduced-precision reductions disabled. This isolates arithmetic differences
from weight quantization. Every processed prompt and decode position is checked:

- Native relative L2 logit error against FP32 is at most 2%, and next-token probability total variation is at most
  0.02. Total variation bounds the probability assigned to any token set by two percentage points.
- Each prompt's root-mean-square error is at most twice the FP16 reference RMS, with one FP16 epsilon as a floor.
  This compares aggregate accuracy without dividing by nearly zero errors at individual positions. The separate
  per-position limits prohibit hiding large outliers in an average.
- Native argmax matches either reference. Where FP16 and FP32 agree, exact agreement is required. If they disagree,
  both choices and margins remain visible; accepting either does not guarantee identical future completions.

These are experimental engineering budgets, not mathematical FP16 error bounds. They supplement exact Python/Rust
artifact parity, strict tiny-model logits, and independent rotary and attention checks. They are not a general
claim about all Qwen3 prompts, model sizes, or context lengths.

## Calibration and held-out checks

The first matrix covers France, arithmetic, explanation, code, German translation, JSON, and repeated text with
127- and 240-token prompts. Each adds 15 or 16 decode positions, reaching the artifact's 256-position boundary.
Its 554 positions all match both references' argmax choices. Maximum native relative L2 error is 1.2781%; maximum
probability total variation is 0.011396. Per-prompt RMS ratios against the FP16 control stay below 1.18.

That matrix failed the initial comparative contract: two cases passed and six failed. The initial rule required
both implementations to stay within the absolute budgets and native error at every position to stay below twice
reference error. Near-exact reference positions made that ratio unstable. The FP16 control also exceeded the
probability budget once (0.021069), while native was closer to FP32 there (0.006573).

The revised contract above retains both native per-position budgets and token agreement, compares RMS per prompt,
and treats the reference as a control rather than requiring it to pass native's budget. The failed matrix is
calibration evidence, not an independently held-out validation of this revised contract. The revision was committed
before four fresh cases ran: Spanish translation, counting, code, and a 128-token context, each with 24 further
decode positions. All four pass, covering 265 positions. Maximum relative L2 error is 1.1355%, maximum probability
total variation is 0.011396, and the largest per-prompt RMS ratio is 1.151. All 265 argmax choices match both
references; together with calibration that is 819/819. The run's duration-record gate flagged the four new test IDs;
the measured durations were then recorded. No accuracy limit was changed after inspecting the held-out results.

## The repaired defect and earlier failed gate

CUDA contracted the rotary expression's FP16 multiply and add into a fused operation. Hugging Face rounds both
products before adding them. Explicit round-to-nearest half intrinsics preserve those boundaries. An independent
NumPy regression failed at 115 of 512 query elements before the fix and passes exactly afterward; the largest
original error was 0.015625. It also checks keys and unchanged values.

Before the fix, native logits tied tokens 9625 and 15344 at France position nine (zero-based), while Hugging Face
favored 15344 by 0.03125. After the fix all 40 original checkpoint argmax choices agree, and Python/Rust execution
of the same artifact is bit-identical at all 40 positions. CPU-prepared rotary constants also match CUDA constants.

The original `rtol=atol=2e-2` pointwise full-model gate still fails: only eight of those 40 vectors pass. Maximum
absolute error is 0.0703125 and maximum relative L2 error is 0.004594. None passes the tighter `1e-3` pointwise gate.
A Hugging Face cached/full-prefix control also fails: 14/40 pass, with maximum absolute error 0.1064453125 and
relative L2 error 0.009375, although all argmax choices agree. The new contract is an explicit change, not a claim
that these vectors now pass the old tolerance. Both pointwise comparisons remain in new numerical records.

## Isolation and runtime checks

- Tiny-Qwen3 logits use `rtol=atol=1e-3` against eager execution. The test checks request reset, EOS, zero output
  budget, context bounds, graph replay, and first capture during decode. Python and Rust run the identical artifact
  with bit-identical logits. Python and NVCC are hidden from the worker's PATH after preparation.
- An independent NumPy attention check reaches positions 1, 127, 128, 129, 4,095, and 4,096, then resets to two.
  Future cache entries contain NaNs. Graph replay reads only the written prefix and passes `1e-3` throughout.
  This isolates attention; it does not qualify a complete checkpoint at 4,096 positions.
- All 28 attention layers on identical inputs at the formerly divergent checkpoint position pass `1e-3`, with
  maximum absolute difference 0.0009765625. The final norm/head also passes on identical inputs.
- Compiler-produced pre/post programs show small differences too. Three outputs fail the strict pointwise check
  at the first token, with relative L2 errors between 0.0000121 and 0.000174. Emitted programs preserve the expected
  FP16 cast boundaries. Double scalar accumulators, disabling fast math, and changing SiLU division ordering did
  not resolve the old checkpoint gate; none justified a production compiler change.
- Additional same-input arithmetic/code checks compare all 28 layers with high-precision products and activations
  rounded at FP16 storage boundaries. Across 224 outputs, native relative L2 error against that rounded reference
  is at most 0.000188. This bounds the sampled local differences; it is not proof of global numerical equivalence.

The public generation client also reproduces all 15 checked France tokens from the diagnostic step path using the
complete Rust loop, CUDA graphs, and the default deadline.

## Reproduction and evidence

Build the worker with `cargo build --release --locked --bin emmy-runtime-worker`, then prepare the pinned checkpoint:

```bash
emmy generate Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --export-native /tmp/qwen3-native --context-length 256 \
  --golden experiments/Qwen3-0.6B/native_baseline/golden/rtx4080_sm89.yaml
PATH="$PWD/target/release:$PATH" ./venv/bin/pytest \
  tests/serving/native/test_generation_gpu.py::test_checkpoint_logits_and_completions \
  --native-checkpoint /path/to/local/checkpoint --native-artifact /tmp/qwen3-native \
  -n 2 --dist=loadgroup --durations=0 --durations-min=0.5 -p no:randomly
```

The opt-in test saves per-position `measurements.json` before final accuracy assertions. The normal suite uses a
hermetic tiny model and skips checkpoint cases. Original numerical records, compressed calibration, held-out, and
rounded-layer records, and sanitized provenance are in [evidence](evidence). No weights, private paths, machine names,
device identifiers, or exploratory programs are published.

PyTorch documents that sliced and batched computations can differ and low-precision intermediates accumulate error.
This motivates a higher-precision control; it does not endorse these budgets. See the
[PyTorch 2.11 numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).

Prefill is sequential and sampling is greedy. HTTP serving, concurrency, native text processing, optimized weight
storage, and performance qualification remain outside this implementation. The complete checkpoint is checked only
through 256 positions on this GPU.

## Repository validation

The full suite completed: 5,191 passed, 748 skipped, 21 xfailed, and 27 failed. All native-generation tests passed.
Twenty-six failures reproduce in an unchanged checkout of main at `22a5253b3`: unsupported TMA on this GPU,
a block-scaled kernel expectation, and existing serving tests whose strict evidence no longer covers their forks.
The remaining benchmark-worker test exceeded its deadline during overlapping GPU work; it passes alone on both
branches (1.60 seconds on this branch). No compiler changes or test exclusions were added to conceal these failures.
Exact failing test IDs and baseline results are in [validation evidence](evidence/validation.json).

Python lint, Rustfmt, Clippy, and all seven Rust unit tests pass. The new checkpoint durations are recorded with
xdist group suffixes. The existing virtual environment was retained with `make -o venv/.setup-complete test` and
`make -o venv/.setup-complete lint`; the test and lint recipes themselves are unchanged. No compiler source changed,
so the separate model-golden decode gate is not applicable. Wheel and sdist CUDA resource inclusion was verified.
That qualification run did not pass the full-suite gate. PR #859 subsequently merged with those baseline failures
recorded.
