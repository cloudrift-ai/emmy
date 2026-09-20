# Native cached-generation numerical investigation

The native rotary kernel had a real FP16 rounding defect. Fixing it restored all 40 checked greedy-token choices
against Hugging Face. Full-checkpoint logit qualification still fails the existing pointwise tolerance; this result
is not a qualified release or a performance claim. The native-generation PR remains a draft.

## Scope

Qwen/Qwen3-0.6B, revision `c1899de289a04d12100db370d81485cdf75e47ca`, FP16, one RTX 4080 (`sm_89`). The
artifact has capacity 32 and uses the qualified pre/post schedules from the sibling native-baseline experiment.
The final norm/head has no strict measured-evidence requirement. This investigation does not retune schedules.

The prompts are “The capital of France is” and “2 + 2 =”. Each has five prompt tokens. We check those five positions
and fifteen further decode positions, for 40 total. The first request is uncaptured; the second uses CUDA graphs.
Hugging Face uses eager attention and a one-token cache. During decode every reference receives the native-selected
token, so comparisons always use the same prefix. Greedy equality at every position also checks that these bounded
trajectories agree. This is not broad generation-quality coverage or a 4,096-token context qualification.

## The fixed defect

CUDA contracted the rotary expression's half-precision multiply and add into a fused operation. Hugging Face rounds
both products to FP16 before adding them. Using the explicit round-to-nearest half intrinsics preserves those
boundaries. An independent NumPy regression failed at 115 of 512 query elements before the fix and passes exactly
afterward; its largest original error was 0.015625. The same check covers keys and unchanged values.

Before the fix, the checkpoint differed at one greedy choice: the tenth processed position of the France prompt.
Native logits tied tokens 9625 and 15344, while the reference favored 15344 by 0.03125. After the fix all 40 argmax
choices agree. The rotary constants prepared on CPU also match the CUDA reference constants bit-for-bit.

## What remains

At `rtol=atol=2e-2`, only eight of 40 native/reference logit vectors pass. Maximum absolute error is 0.0703125;
maximum relative L2 error is about 0.004594 (0.46%). None passes the tighter isolated-program `1e-3` pointwise gate.
The optional checkpoint test retains the failing `2e-2` assertion and exact greedy agreement requirement.

A control compares Hugging Face's cached execution with its full-prefix execution, using FP16 eager
attention and the same strict accumulation settings. That control also fails the pointwise gate: only 14 of 40
positions pass, although every argmax agrees. Its maximum absolute error is 0.1064453125 and maximum relative L2
error is about 0.009375 (0.94%). Both paths consume the same native-selected tokens in this control.

This control shows that the pointwise threshold also rejects ordinary reference execution differences. It does
not prove every native discrepancy harmless, justify removing the gate, or establish an alternative threshold.
A defensible full-checkpoint acceptance criterion and broader prompts/context coverage remain open.

## Isolation checks

- The same standalone binaries run through Rust and the existing Python executor. The tiny-model test requires
  bit-identical logits at every position, including request reset and first graph capture during decode. All 40
  checkpoint positions also match bit-for-bit between the two dispatchers. Cache
  allocations are persistent in the shared plan, and custom launches declare their writes for scratch allocation.
- On identical native inputs, all 28 attention layers at the formerly divergent decode position pass `1e-3`.
  The largest absolute attention difference is 0.0009765625. Most layers match exactly.
- The final norm/head on identical inputs passes `1e-3`; maximum absolute difference is 0.00390625 and relative L2
  error is about 0.00002077. It does not explain the accumulated full-model discrepancy by itself.
- Same-input pre/post checks at the first token find small differences in compiler-produced programs too. Query
  outputs at layers 7 and 11, and the post output at layer 27, fail the strict pointwise gate. Their relative L2
  errors are 0.000174, 0.000149, and 0.0000121 respectively. Other checked pre/post outputs pass.
- Inspection of emitted programs found the expected FP16 cast boundaries. Temporary probes using double scalar
  accumulators, disabling fast math, and changing SiLU division ordering each still failed the checkpoint gate.
  None justified a production compiler change. Those exploratory variants are not shipped.

These checks separate a repaired native arithmetic bug from the remaining accumulated numerical differences.
They do not establish exact equivalence between independently reduced floating-point programs.

## Reproduction and evidence

Build the worker with `cargo build --release --locked --bin emmy-runtime-worker`. Prepare the pinned local checkpoint
with the existing exporter, then run the opt-in test:

```bash
emmy generate Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --export-native /tmp/qwen3-native --context-length 32 \
  --golden experiments/Qwen3-0.6B/native_baseline/golden/rtx4080_sm89.yaml
PATH="$PWD/target/release:$PATH" ./venv/bin/pytest \
  tests/serving/native/test_generation_gpu.py::test_checkpoint_logits_and_completions \
  --native-checkpoint /path/to/local/checkpoint --native-artifact /tmp/qwen3-native \
  -n 2 --dist=loadgroup --durations=0 --durations-min=0.5 -p no:randomly
```

The test writes per-position `measurements.json` before its final assertion. It compares Python/Rust execution
exactly and records a full-prefix Hugging Face control on the same native-selected prefixes. The completed run took
138.37 seconds in the test body and failed only the retained full-model assertion. Raw numerical rows and provenance are in
[evidence](evidence). No model weights, machine names, account paths, device identifiers, or exploratory programs
are published.

Development checks cover native Rust unit tests, Python protocol/CLI validation, tiny-model generation, and the
rotary regression. Full-suite finalization and checkpoint qualification remain outstanding. HTTP serving,
concurrency, and throughput optimization are outside this implementation.

## Qualification contract under evaluation

The following contract is fixed before evaluating the broader eight-case matrix. Its limits are engineering
acceptance budgets for this experimental FP16 path, not a mathematical error bound or a production quality claim.
The three exploratory prompts used to examine precision were France, a short French translation request, and an
addition function. The six new qualification cases use explanation, different code, German translation, JSON,
and repeated astronomy/library text at 127 and 240 prompt tokens. The original France and arithmetic cases remain
as regressions. Every case adds 15 or 16 decode positions; the longest reaches the artifact's 256-position boundary.

A second Hugging Face model evaluates the **same FP16-rounded weights** in FP32, with TF32 disabled. References see
exactly the same prefix as native execution. Both native and Hugging Face FP16 must stay within 2% relative L2 logit
error and 0.02 total variation distance from the FP32 next-token distribution. Total variation limits the probability
assigned to any token set to a two-percentage-point difference. Native error must also stay below twice the FP16
reference error, with one FP16 epsilon as the floor for nearly exact reference calculations. This comparative guard
prevents the absolute budget alone from accepting a disproportionately inaccurate native result.

Native argmax must equal one of the FP16/FP32 reference choices. If those references agree, exact agreement is
required. If they disagree, the row records both choices and their margins; disagreement is visible evidence of a
precision-sensitive decision, not proof of equivalent future completions. The former `2e-2` pointwise comparison
remains recorded for every position, but is no longer the full-model acceptance rule. This is an explicit change
of qualification contract, not a claim that its previously failing vectors now satisfy that tolerance.

This contract supplements exact Python/Rust dispatcher parity, strict tiny-model logits, the independent rotary
rounding regression, and independent attention/cache-boundary checks. Its test budgets will not be increased to
make a failing held-out case pass. Broader tests are still pending at this revision.

PyTorch documents that sliced and batched computations can differ with the same mathematical inputs, and that
low-precision intermediates can accumulate error. This motivates checking a higher-precision reference; it does
not supply or endorse the budgets above. See the
[PyTorch 2.11 numerical-accuracy notes](https://docs.pytorch.org/docs/2.11/notes/numerical_accuracy.html).
