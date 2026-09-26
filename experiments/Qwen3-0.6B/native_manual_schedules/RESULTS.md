# Manual schedules for native Qwen3

Manually selected schedules improve native Qwen3 serving on one RTX 4080. Across paired repeats, first-token latency
is 5.8–7.1× lower, decode time per token is 3.5–4.3× lower, and output throughput is 5.3–6.4× higher than the previous
qualified artifact. All seventeen numerical qualification cases pass. Strict evidence prevents prior fallback;
selection uses explicit schedules and their measured timings.

## Scope and reproduction

The checkpoint is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Compilation uses
`d81a1a0b`, standard math (`EMMY_FAST_MATH=0`), NVCC's deployable optimization setting (`EMMY_NVCC_FLAGS=`), FP16
weights, and the existing FP32 residual contract. Cross-CTA atomic reductions are excluded. The embedded programs in
[the golden](golden/rtx4080_sm89.json) define the pre-attention, post-attention, and final normalization/head fragments.
They cover native one-token execution, not the vLLM serving matrix.

The previous compressed diagnostic golden predates the current serialization format. The fragments were retraced
through the existing trace command, with the native attention-split wrappers and the same shapes and dtypes. The
compiler reports all three stored targets as fresh. Full export succeeds with this golden, strict evidence, and a
fresh tuning database; the local tuning cache is not needed to reproduce the artifact.

Initial manual candidates use 128-thread cooperative reductions, explicit intermediate cuts, and no tensor-core
or staging choice. A second candidate uses `w2x2`, `mma_m16n8k16_f16_f32/f1x2/k4`, and `d2/smem`. Its accepted
MLP gate/up projection is combined with the measured cooperative projections. The output-head tensor-core proposal
does not realize and is rejected; its replacement explicitly selects the direct schedule. Selection among these
measured choices uses strict evidence. No MCTS or prior-led search was run. Global pins apply across the cut pieces,
so each realized row was inspected; a useful schedule for one projection can make the whole fragment slower when
applied broadly.

Prepare the serving bundle with the existing native launcher and this golden, or export a generation bundle:

```bash
EMMY_FAST_MATH=0 EMMY_NVCC_FLAGS= EMMY_TUNE_DB=/tmp/native-manual-export.db \
  emmy generate Qwen/Qwen3-0.6B --revision c1899de289a04d12100db370d81485cdf75e47ca \
  --golden experiments/Qwen3-0.6B/native_manual_schedules/golden/rtx4080_sm89.json \
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

## Serving measurements

Every repeat completes four requests with 32 output tokens each: 72 measured requests, zero failures. Ranges below
are the three repeats of each client's mean. Pairing the same seed across lanes gives the speedups above. The two
fixed text probes also produce identical sixteen-token completions; that smoke check is separate from qualification.

| Artifact | Input tokens | Mean TTFT, ms | Mean TPOT, ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Previous | 32 | 1,240.87–1,579.74 | 32.76–43.71 | 12.326–12.329 |
| Manual | 32 | 173.97–253.23 | 7.67–10.23 | 65.132–65.200 |
| Previous | 256 | 9,974.26–10,129.62 | 39.42–44.43 | 2.819–2.819 |
| Manual | 256 | 1,437.96–1,475.32 | 9.71–10.98 | 17.993–18.012 |
| Previous | 1,024 | 41,166.29–41,515.00 | 35.17–46.91 | 0.750–0.751 |
| Manual | 1,024 | 7,006.08–7,105.28 | 10.06–13.43 | 4.310–4.314 |

The improvement survives complete-model execution and HTTP serving. It does not remove the sequential prefill cost:
a 1,024-token prompt still takes about seven seconds before the first token. Sampling and attention are unchanged.
The earlier stock-vLLM measurements remain much faster; they were not rerun here and are not a fresh controlled
comparison. The next measured work should address native sampling and sequential prefill, not treat this result as
production serving parity.

## Fragment measurements

Fresh processes measure captured whole-fragment time with five warmups and twenty iterations at deployable NVCC
optimization. Strict correctness checks against eager pass. These timings are diagnostic; the serving experiment
below measures the complete model. The old fragment figures are historical single observations from the previous
qualification, not a controlled same-compiler A/B comparison.

| Fragment | Previous diagnostic (µs) | Selected, repeated (µs) |
| --- | ---: | ---: |
| Pre-attention | 94.9 | 66.0–69.4 |
| Post-attention | 1,206.2 | 77.7–77.8 |
| Final normalization/head | 1,090.4 | 494.6–506.4 |

The cooperative post-attention candidate alone measured 156.3–157.5 µs. Applying the tensor-core candidate globally
measured 786.4 µs: only gate/up accepted it, while other pieces used slow direct schedules. Combining the already
measured manual rows kept the 12.6 µs gate/up projection and the 32.3 µs cooperative down projection. The original
profile's down projection took about 1,054 µs per call. The unchanged native sampling kernel remains significant.

The cooperative head candidate was slower at 3,590.1 µs and was not selected. The head tensor-core proposal failed
exact-pin integrity and was not accepted as that schedule. Its fallback observation is retained as a failed trial;
the subsequent direct candidate explicitly pins the choices it executes and passes strict validation. The selected
head combines direct projection with the measured cooperative normalization. A fresh empty-cache replay using only
the combined golden passes, establishing that deployment does not depend on the tuning database.

## Full-model numerical qualification

All seventeen existing checkpoint cases pass across 10,585 positions, including both 4,096-position prompts,
captured and uncaptured execution, and shorter requests after long contexts. The largest relative L2 error against
FP32 is 0.018090; the largest probability total variation is 0.016688. Both remain below the unchanged 0.02 limits.
The largest per-case native/reference RMS ratios are 0.666 for relative L2 and 0.943 for total variation, within the
existing acceptance rule. These errors are not claimed to improve over the previous artifact.

Native argmax agrees with both references at 10,584 positions. At the remaining position it agrees with one reference,
as permitted by the existing contract. The two executor paths produce bit-identical logits at all 552 checked
positions. Qualification took 446.89 seconds; that includes reference execution and diagnostic transfers and is not
a generation benchmark. These reused cases establish the tested contract, not general model quality.

Run the existing qualifier with the matching checkpoint, the new artifact, and matching native binaries on `PATH`:

```bash
pytest tests/serving/native/test_generation_gpu.py::test_checkpoint_logits_and_completions \
  --native-checkpoint /path/to/checkpoint --native-artifact /path/to/artifact \
  -n 2 --dist=loadgroup --durations=0 --durations-min=0.5 -p no:randomly
```

## Comparison limits

The baseline is the previously qualified artifact used in [the native-serving baseline](../native_serving/RESULTS.md).
Its manifest SHA-256 is `a9e0e3aa48ba80b0d1f81ba3673f4989abf7a1aea53c3c152544eb325fc05d6c`; the new artifact's
is `6bad6b565ed25208455da1b35810b05cd99b573070fae45ad9722eec6df9ac22`. Both use the same checkpoint revision,
context capacity, native attention, sampling, and current server binary. The comparison measures replacing an older
compiled artifact with newly compiled manual schedules; it does not isolate every compiler change since the earlier
artifact was prepared. Neither artifact contains compiler provenance in its manifest, so these hashes and the retained
preparation evidence identify the tested bundles. Checkpoint weights are not published.

Runs are sequential by lane, with three repeats per input length, on a desktop GPU. There is no claim of exhaustive
schedule optimization or a performance advantage over stock vLLM. Batching, paged KV, chunked prefill, and faster
sampling remain separate work. This change adds no compiler or runtime implementation.

## Retained evidence and validation

Run `2026-09-26_04-39-55` starts at 04:39:55 UTC on September 26, 2026. Source revision is `c7f629e5`; compiler and
runtime implementation match `d81a1a0b`. Hardware is one NVIDIA GeForce RTX 4080, 16 GiB, driver 595.91.07, on Ubuntu
with an Intel Core i9-14900K. Software includes NVCC 13.3.73, cuBLAS 13.6.0.2, PyTorch 2.11.0, Transformers 5.14.1,
and vLLM 0.23.0 as the benchmark client. The shared desktop remains a source of timing variation. The recipe holds
the common GPU lock throughout each lane.

Both rows succeeded: `rtx4080x1_lbaseline_367f0faa3b35` and `rtx4080x1_ltuned_8eae2f4ed9a4`.
[The platform archive](results_rtx4080x1.tar.gz) contains 130 text members under the run directory:

- Each lane's `c1_i{32,256,1024}_r{1,2,3}.json` and logs contain the serving metrics above.
- The two `*.experiment.yaml` records retain terminal state, source revision, and generic system information.
- `fragment_checks/` retains successful candidates, repeats, and the rejected head tensor-core trial.
- `qualification/` retains all seventeen per-position numerical records; `qualification.log` records the passing run.
- `identity.txt` in each lane identifies the artifact, tokenizer, and matching server binary by SHA-256.
- `ANONYMIZATION.txt` documents removed machine identities, addresses, local paths, and normalized tar metadata.

The publication copy preserves numerical benchmark and qualification values. Original records remain local. No model
weights, GPU UUID, hostname, or account path is published.

The three golden targets are fresh and all 32 entries strictly decode. Complete export succeeds using only the
golden and an empty tuning cache. Fragment checks pass against eager; all seventeen checkpoint cases pass, including
executor parity and request reset. The 45 focused recipe/matrix tests pass, and the recipe dry run expands both lanes.
CI passes the full test suite, native checks, lint, and package dry run on the initial artifact commit. No compiler,
runtime, or test implementation changed; the full suite was not repeated locally for these data-only changes.

## JSON format migration

After merging main through `520f1bee`, the golden uses JSON and textual index expressions introduced by #912.
Conversion preserves all realization names, schedule knobs, identities, and measurements. All three targets remain
fresh, all 32 entries decode, and all three fragments compile with strict evidence and an isolated empty cache.
The 65 focused serialization and golden-command tests pass. The archived measurements retain their original source
revision; this format migration does not claim a new performance measurement.
