# Native Qwen3 output precision

FP16 output storage can create a tie between distinct projection scores and change greedy token selection.
The native output head now keeps FP32 logits through GPU sampling while retaining FP16 projection inputs, weights,
and KV storage. Generation format 4 requires re-exporting old artifacts. The numerical acceptance limits are unchanged.

## Reproduced cause

A fresh artifact from main `a98fd4f85` reproduces the sequential failure reported by the
[chunked-prefill investigation](../native_prefill/RESULTS.md). At position 1,064 of `heldout_prefill_long`, native
execution selects token 576 while both checkpoint references select 785. Both native FP16 logits equal 20.078125,
so the sampler correctly resolves the rounded tie to the lowest token ID.

Keeping the same native final hidden state and identical FP16 projection operands, an FP32 output gives token 785
20.08124542236328 and token 576 20.077966690063477. The FP32 checkpoint reference also prefers 785, at
20.094772338867188 versus 20.06569480895996. Widening the stored half scores cannot recover this ordering; the
projection must preserve it before storage. This is a head-output precision defect, not a greedy reduction defect.

The sampler now accepts the full FP32 range. Positive-temperature sampling computes FP64 weights and locates the
nucleus cutoff through a 32-pass binary search over ordered FP32 keys. A fixed block reduction tree preserves
determinism, and ties retain the lowest token IDs.
Independent NumPy tests cover close FP32 scores inside a single FP16 bin, extrema, ties, invalid inputs, 1,024 seeds,
and captured replay. No prior or MCTS search is used.

The head also exposed two eager-reference errors: a rank-two axis swap was interpreted as an identity permutation,
and matrix multiplication rounded to the operand dtype before widening to its declared output dtype. The reference
now swaps the axes and promotes its operands to preserve the output precision. A CPU regression checks both together.

## Protocol and reproduction

Both serving artifacts use `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`, context capacity
4,096, prefill width 16, standard math, deployable NVCC optimization, CUDA graphs, and one RTX 4080. The baseline
uses main `a98fd4f85`; the final candidate implementation is `c65fda9c`. The recorded golden and initial recipe
were committed in `1f90615c`. Each artifact runs with its matching native server binary because generation formats
differ.
Hashes of manifests, plans, tokenizer metadata, and binaries are retained with the measurements.

The existing [manual golden](../native_prefill/golden/rtx4080_sm89.json) now contains 50 rows across five programs.
Only the output-head program changes; its route cuts normalization from the projection, uses a measured cooperative
normalization and direct scalar projection, and passes strict eager comparison. All five targets are fresh and all
50 rows strictly decode. A fresh tuning database exports the candidate with strict evidence. The final isolated
head observation is 539.1 µs over twenty iterations after five warmups; it is not a whole-model latency result.

Prepare each bundle on its corresponding revision with standard math, a fresh tuning database, and the appropriate
revision of that golden. Set `BASELINE_ARTIFACT`, `BASELINE_SERVER`, `TUNED_ARTIFACT`, and `TUNED_SERVER` to the
resulting bundle directories and matching binaries, then run:

```bash
emmy bench experiments/Qwen3-0.6B/native_accuracy --local
```

The recipe holds the shared GPU lock. Each lane runs three repeats at temperatures 0 and 0.7, top-p 0.8, seeds 1–3,
32 input tokens, 32 output tokens, concurrency one, four measured requests, and one warmup per repeat. This is a
short-prompt cost check, not a throughput comparison with vLLM or a production-concurrency qualification.

## Numerical qualification

FP16 and FP32 eager references consume identical prefixes and FP16-rounded weights, with TF32 and reduced-precision
reductions disabled.

All nineteen sequential cases pass across 15,712 positions. Maximum relative L2 error against FP32 is 0.0180891
and maximum probability total variation is 0.0165485, below the unchanged 0.02 absolute limits. Whole-prompt RMS
and reference-token agreement checks also pass. At the original failing position, all three paths select token 785;
native relative L2 is 0.00179835 and probability TV is 0.00584500.

All nineteen chunked cases also pass across 369 emitted vectors. Maximum relative L2 is 0.00714913 and maximum
probability TV is 0.00397417. Every native argmax across both modes matches FP32; the FP16 reference differs at one
sequential position. The chunked RMS comparison against matched sequential prefixes remains unchanged. The real
checkpoint HTTP lifecycle test passes, including seeded streaming, cancellation recovery, and shutdown.

The combined run records 39 checkpoint/HTTP passes in 1,352.28 seconds, plus a tiny-export setup failure caused by
strict-evidence settings intended for the recorded checkpoint artifact. The separate tiny-model rerun passes with
its normal export settings and the deterministic reduction pin.

The tiny-model test passes with exact Python/native replay and the existing `rtol=atol=1e-3` reference comparison.
Its schedule pins exclude atomic reductions; the original equality and numerical assertions remain intact.

After the shared-memory sampler change, that artifact passes both long-prompt modes and HTTP again: three
checks in 316.78 seconds. All 4,096 sequential and seventeen chunked measurement rows exactly match the earlier
full-matrix run. The compiled model kernels and non-sampling launch contracts are also unchanged.

## Serving measurements on RTX 4080 x1

The accepted run is `20260928T061132Z`, started 2026-09-28 at 06:11:32 UTC. Both rows succeed. All 48 measured
requests finish with 32 output tokens and zero failures. All twelve paired greedy texts and the fixed France
completion match. Nine of twelve temperature-sampling texts match; preserving FP32 scores changes probabilities,
so cross-artifact seeded completions are not required to match.

Ranges below span the three repeats of each client's mean. Greedy output throughput changes by +0.1–1.0%, within
this short comparison's variation. Temperature-sampling throughput improves by 16.8–19.7%. These are complete
serving observations, not isolated sampler timings or a general throughput claim.

| Output precision | Temperature | Mean TTFT, ms | Mean TPOT, ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| FP16 baseline | 0 | 41.61–80.65 | 3.69–5.03 | 161.78–163.92 |
| FP32 candidate | 0 | 41.69–80.21 | 3.69–4.96 | 163.34–164.35 |
| FP16 baseline | 0.7 | 41.45–197.59 | 11.37–15.79 | 58.17–60.82 |
| FP32 candidate | 0.7 | 41.90–183.97 | 9.25–13.03 | 67.96–72.83 |

Seed three includes a completion with empty decoded text. Its first visible response arrives after generation,
affecting the client's TTFT/TPOT split even though 32 output tokens are counted. Throughput includes that request;
no repeat is discarded. Only this short, single-request matrix is measured. Longer prompts, other temperatures,
other GPUs, and production concurrency remain unqualified for performance.

The final parallel sampler passes all seven focused GPU checks in 80.05 seconds and the checkpoint HTTP lifecycle
again in 3.98 seconds. Its compiled model kernels, launch contracts, and resident layouts match the long-prompt
recheck artifact. Its fixed reduction tree replaces serial mass accumulation; the independent distribution,
seed-repeatability, and graph-replay checks validate that sampling change separately.

## Memory

The runtime layout accounts for 1,978,542,184 resident plan-buffer bytes on the baseline and 1,979,824,236 on the
candidate: an increase of 1,282,052 bytes (1.22 MiB). FP32 logits add 303,872 bytes. Private sampling workspaces use
1,215,488 bytes; the cutoff reduction uses 1 KiB of shared memory. Removing the old histogram reduces decode
scratch from 262,144 to 24,836 bytes. Prefill retains 721,216 private bytes in both artifacts. Shared constants are
identified by equal binding bytes, as in the runtime.
These figures exclude CUDA contexts, modules, graphs, allocator overhead, and transient duplicate allocations during
load. The output head retains FP16 weights; no wider weight copy is introduced.

## Retained failed and interrupted checks

The initial worker checks accidentally used a format-3 binary left in the shared Cargo target directory by the
baseline build. Loading format 4 failed before model execution. Rebuilding the candidate and using distinct binary
paths repaired that setup error. These logs remain in the archive.

The tiny-model exact-replay check exposed an atomic head reduction chosen without explicit schedule pins. Its
FP32 outputs differed by at most 2.98e-8 between executions; the cubin contains atomic floating-point additions.
The replay test now pins serial reductions so its bit-identity assertion has a deterministic schedule. Its numerical
limits and equality assertions are unchanged. The checkpoint artifact already uses the manually selected direct head.

The first serving run overlapped checkpoint qualification because its shell lock omitted the device UUID suffix
used by the compiler lock. It was interrupted and its measurements are excluded from all latency comparisons.
A second run derived the correct name but still overlapped: Python removes the lock file on release, leaving the
shell holding an unlinked inode. Both runs are excluded. The recipe now holds the compiler's actual lock context
for the child shell, and the accepted run starts after qualification exits. The affected records and outputs remain
under `earlier-runs/`; a complete new run repeats both lanes.

An isolated intermediate implementation kept the radix histogram in global memory. Greedy throughput was unchanged,
but temperature-0.7 throughput fell from 57.91–60.56 to 16.67–17.06 output tokens/s. Serial global-memory histogram
updates caused an unacceptable cost. Moving that 4 KiB workspace to shared memory removes its persistent buffer
and reduces the cost, but leaves
serial scans too slow. The final implementation instead searches the ordered FP32 key space with fixed parallel
reductions. The cutoff and tie rules, RNG, and token-ID sampling order are unchanged; mass accumulation now uses a
fixed reduction tree instead of serial token order.
The intermediate measurements remain under `earlier-runs/2026-09-28_05-53-59/` and
`earlier-runs/2026-09-28_06-04-58/`. Default-suite runs were interrupted to make these measured repairs, then restarted
on the final code.

## Evidence and environment

The [anonymized archive](results_rtx4080x1.tar.gz) has root `2026-09-28_06-11-32/` and retains both terminal
`*.experiment.yaml` row records. Its `*_c1_i32_t*_r*.json` files contain all twelve client measurements, including
per-request timing and generated text. Matching logs, fixed completions, binary/artifact hashes, software inventories,
and coarse GPU memory observations are retained beside them. Build matching runtime binaries in separate Cargo target
directories when reproducing the two revisions.

`checks/reproduce.log`, `checks/inspect-logits.log`, and `checks/head-rounding.log` establish the original failure
and lost ordering. Their native/reference numeric arrays are retained as `checks/*.npz` without modification.
`checks/head-*-result.json` and their logs retain manual head measurements.
`qualification-final/` contains the complete checkpoint matrix; `shared-qualification/` contains the long-prompt
recheck. `checks/shared-comparison.json` and `checks/parallel-comparison.json` record unchanged model kernels,
launch contracts, layouts, and repeated measurement rows. `checks/parallel-tests.log` and `checks/parallel-http.log`
validate the final sampler. `checks/memory-summary.json` and `checks/layouts.json` contain the resident-byte accounting.
The interrupted and intermediate serving runs are under `earlier-runs/`; their timings are not pooled with the
accepted comparison. Validation logs are under `checks/`.

Hardware is one NVIDIA GeForce RTX 4080, capability sm89, with 16,376 MiB reported VRAM. Driver is 595.91.07,
NVCC 13.3.73, cuBLAS 13.6.0.2, Python 3.12.3, PyTorch 2.11.0, Transformers 5.14.1, vLLM client 0.23.0, and
NumPy 2.3.5. Both lanes use the same environment. The native server performs model execution; vLLM is only the client.

The publication copy removes host/account identities, GPU UUIDs, PCI and network addresses, filesystem metadata,
and uptime. Software versions and numeric measurement values are preserved and checked against local originals.
Archive member bytes are verified after writing, with normalized tar ownership and timestamps. Model weights,
local caches, native binaries, and temporary diagnostic programs are omitted.

Final repository validation passes: 7,482 Python tests passed and 798 skipped in 844.72 seconds under the default
correctness lane. Locked Cargo workspace tests, Python lint/format checks, duration-file formatting, Rustfmt, and
Clippy all pass. No numerical threshold or expected-failure classification was relaxed.
