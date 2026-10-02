# Remaining golden-bench performance work

Status: planned follow-up to [PR #1011](https://github.com/cloudrift-ai/emmy/pull/1011). These are hypotheses to
test, not demonstrated speedups. The completed shared K/V decode work and its raw evidence are recorded in the
[experiment report](../experiments/golden-bench-2026/kernels/RESULTS.md).

## Baseline and scope

The last five-card qualification used Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`, layer 0,
O3, fast math disabled, 10 warmups and 100 iterations. These are same-input model comparisons in microseconds:

| Target | Emmy | torch.compile | Emmy launches |
| --- | ---: | ---: | ---: |
| H100 80GB, sequence length 512 | 87.347 | 81.992 | 12 |
| V100 SXM2 16GB, sequence length 1 | 68.335 | 62.498 | 14 |

Repeat these controls on the implementation's source before tuning. Subsequent main changes to schedule enumeration,
priors and runtime program layout are not covered by those GPU timings. Use the existing recipe and CLI, fresh
tune databases, exact-card goldens, strict evidence and no timing writes for validation. Record GPU UUID, clocks,
software versions, source revision and the emitted sources, schedules and launch settings.

Use the single-GPU V100 or another approved V100 machine. Do not use the 4xV100 VM. Keep the other four cards as
regression controls when a shared compiler change reaches their selected kernels. Persistent kernels and serving
integration remain outside this bounded follow-up.

## 1. H100 prefill: share K/V production

Decode already shares the K/V producer while keeping Q separate. Prefill needs a different coordinate mapping:
V sweeps `(1024, 512)` and K sweeps `(8, 512, 128)`. Their row coordinate occupies different positions. Flattening
both in stored order would mix rows and channels and cannot establish shared input loads.

- Derive the correspondence from the actual output index maps. Extend common-coordinate formation only where
  ownership, binding and the row/channel mapping prove it legal; do not add a model-specific recognizer.
- Check a small reproducer with distinct rows and channels, including an incompatible mapping that must retain
  its original form. Verify every output value and preserve the original coordinate parameter order.
- Inspect the full-layer lowering to confirm which input loads and reductions are shared. Keep ordinary cuts
  available and let measured evidence choose between shared and separate producers.
- Compare the candidate against the complete layer, including attention and MLP. Fewer launches or a faster
  isolated K/V kernel is not sufficient evidence of a win.

This is the first implementation candidate. It can proceed independently of the V100 profiling below.

## 2. V100 decode: locate the cost of the new 14-kernel route

Profile the accepted shared K/V route before choosing another transformation. Separate in-graph kernel duration,
gaps, weight traffic and partial-reduction/finalization work. Use exact-source Nsight Systems and Compute captures;
compare against the same-input reference, and retain unprofiled runs as the latency measure.

- If a projection fails to use available memory bandwidth, inspect coalescing, load width, redundant reads and
  occupancy before changing its schedule. Low arithmetic intensity motivates this check; it does not establish
  the measured bottleneck by itself.
- If partial reductions or their consumers dominate, compare existing split factors and legal cuts first.
  Consider combining finalization with a consumer only when it avoids repeated projection work and preserves
  rounding and output ownership.
- Test one identified cost at a time. The previous eight-output-lane K/V schedules lost in the whole layer.
  A 128-thread K/V partial also lost to the selected 256-thread partial. Removing projection cuts repeated too
  much work; the consumer-summed Q split tied, and adding the V split slowed the layer.

## 3. H100 attention: explain the remaining execution gap

The paired profile placed Emmy attention about 2.4 microseconds behind the vendor path. That is a diagnostic
profile difference, not a promised whole-layer gain. Reconfirm it on current source, then inspect synchronization,
register pressure and overlap between loads and computation. Choose a lowering or scheduling change only after
identifying avoidable work or a wait that matters in the complete layer.

The earlier attention tile/staging trials and gate/up TMA trials did not produce a reliable layer win. Another
broad sweep needs a new diagnosis. Removing unused asynchronous copies is lower priority: six paired layer runs
showed only a 0.284-microsecond median improvement, with execution-order sensitivity, so that prototype was reverted.

## Acceptance and golden handling

- Use the existing benchmark CLI and recipe; do not create a benchmark script. Keep exploratory runs bounded
  under the repository's development test budget.
- Predeclare a fixed number of interleaved baseline/candidate pairs with balanced execution order. Retain every
  sample, including losses, and compare complete-layer distributions rather than isolated kernel sums.
- Require correctness on identical inputs with unchanged tolerances. State whether each comparison uses strict
  or scaled correctness. Repeat the strict golden replay and reject a candidate that fails it.
- Audit changed kernel identities, CUDA bodies, schedules and launch settings. Check all repository and experiment
  goldens for freshness, and compare a fresh model trace as well: replaying a stored trace cannot detect a loader
  change that alters newly traced programs.
- Preserve unchanged measurements. A changed kernel needs a new measurement on its exact card before becoming
  evidence; a loader change may also require a fresh trace. Never re-record an unchanged slow row to hide a loss.
  If a maintained repository golden changes, refit both priors as required by the repository workflow.
- Record accepted results and rejected probes in the cumulative experiment report with raw archives. Run the full
  suite, including prior reproduction, and lint at finalization. Delete this plan when its work is completed.
