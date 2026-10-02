# Golden-bench: unsplit decode and causal prefill

Start from main after PR #1019, commit `e53587a910d9e6372814800e22e88a06f742fc31`. This phase tests two specific
hypotheses from the matched NVIDIA profiles. It uses the existing schedule model before considering larger changes.

## Establish the current baseline

Main also includes scalar staging, cooperative reduction and prior changes from PR #1029. The previous profiles
remain historical evidence until fresh lowering, selected schedules and emitted sources are compared on each card.
The V100 decode and H100 prefill experiment goldens pass fresh-lowering checks without restamping.

Use the pinned Qwen3-0.6B revision, layer zero, FP16, O3 and fast math disabled. Start every measurement process with
a fresh tuning database. Baseline and qualification use strict evidence without recording. Model comparisons use
the existing scaled correctness check; golden qualification uses the existing strict check.

## Single V100: remove projection finalizers when it pays

The prior profiles found Q and K/V partial reductions close to their compiled-reference counterparts, followed by
additional finalizers. Test an unsplit reduction with enough output parallelism and suitable weight access. First
try the newly available cooperative and scalar staging mechanisms; add only a focused lowering extension if needed.
Preserve intermediate FP16 boundaries. The reference's FP32 buffers do not authorize changing Emmy's numerics.

Compare the complete layer, including any changed weight preparation or other kernel boundaries. Do not assume
the isolated finalizer durations are recoverable whole-layer savings. Keep the MLP product cut unless measurements
justify another realization; the reference's repeated product computation is additional work.

## H100: measure existing causal bounds

The current attention lowering suppresses a one-sided causal loop bound when every CTA fits in one wave. Test that
bound on the accepted schedule while retaining the complete mask and all numerical checks. Fewer instructions can
leave the longest CTA unchanged. The existing head-width-256 regression also needs a control before changing the
general policy; a win at one shape does not justify removing the policy everywhere.

Do not repeat rejected TMA and tile sweeps without new evidence. A larger warp-specialization implementation is
outside this phase's initial scope.

## Acceptance and publication

Keep exploratory CLI runs bounded to two minutes. Cold reference setup may use the established fixed ten-minute
cap when compilation exceeds that limit; retain the initial timeout. Qualify a promising candidate with six fixed
balanced baseline/candidate pairs and five strict golden replays, retaining every failure and loss. No new benchmark
script, precision relaxation, fusion gate, or replacement of old measurements is permitted.

Consolidate implementation, tests, cumulative results and raw evidence in one draft PR. Delete this plan when its
conclusions are recorded. A100 is stopped with its disk retained; use H100 and the single V100, never the 4xV100 VM.
Shared compiler changes require regression qualification on the available cards before acceptance.
