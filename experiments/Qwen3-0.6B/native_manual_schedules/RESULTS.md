# Manual schedules for native Qwen3

The native one-token fragments now have fresh, manually selected schedules for the RTX 4080. Selection uses explicit
pins and measured evidence with strict evidence enabled. No prior prediction is accepted. Full-model qualification
and the repeated serving comparison are in progress; fragment timings alone do not establish a serving speedup.

## Scope and reproduction

The checkpoint is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Compilation uses
`d81a1a0b`, standard math (`EMMY_FAST_MATH=0`), NVCC's deployable optimization setting (`EMMY_NVCC_FLAGS=`), FP16
weights, and the existing FP32 residual contract. Cross-CTA atomic reductions are excluded. The embedded programs in
[the golden](golden/rtx4080_sm89.yaml) define the pre-attention, post-attention, and final normalization/head fragments.
They cover native one-token execution, not the vLLM serving matrix.

The previous compressed diagnostic golden predates the current serialization format. The fragments were retraced
through the existing trace command, with the native attention-split wrappers and the same shapes and dtypes. The
compiler reports all three stored targets as fresh. Full export succeeds with this golden, strict evidence, and a
fresh tuning database; the local tuning cache is not needed to reproduce the artifact.

Initial manual candidates use 128-thread cooperative reductions, explicit intermediate cuts, and no tensor-core
or staging choice. A second candidate uses `w2x2`, `mma_m16n8k16_f16_f32/f1x2/k4`, and `d2/smem`. Its accepted
MLP gate/up projection is combined with the measured cooperative projections. The output-head tensor-core proposal
does not realize and is rejected; its replacement explicitly selects the direct schedule. Selection among these
measured choices uses strict evidence. No MCTS or prior-led search was run.

Prepare the serving bundle with the existing native launcher and this golden, or export a generation bundle:

```bash
EMMY_FAST_MATH=0 EMMY_NVCC_FLAGS= EMMY_TUNE_DB=/tmp/native-manual-export.db \
  emmy generate Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --golden experiments/Qwen3-0.6B/native_manual_schedules/golden/rtx4080_sm89.yaml \
  --strict-evidence --context-length 4096 --export-native /tmp/native-manual-generation
```

The comparison recipe accepts `BASELINE_ARTIFACT` and `TUNED_ARTIFACT`, both prepared serving bundles. It runs the
same freshly built server for both, with greedy sampling, CUDA graphs, concurrency one, 32 output tokens, input
lengths 32/256/1024, four requests per repeat, one warmup, and three seeds. Numerical acceptance precedes serving
measurement. Neither sampling nor sequential prefill is changed.

```bash
BASELINE_ARTIFACT=/path/to/previous TUNED_ARTIFACT=/path/to/manual \
  emmy bench experiments/Qwen3-0.6B/native_manual_schedules --local
```
