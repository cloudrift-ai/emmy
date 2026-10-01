# Shared NVFP4 compiler and runtime fixes

Status: extraction in progress on `feat/nvfp4-shared-compiler-fixes`, based on `origin/main`
`754d1afa` (October 2, 2026). The mixed Qwen3.8 MLP serving route in
[PR #993](https://github.com/cloudrift-ai/emmy/pull/993) is paused by user request. This branch contains
only reusable compiler, runtime, and correctness-check changes. It does not include the MLP adapter,
`--compile-scope mlp`, model-specific schedules or goldens, or prior weights. The older
[mixed serving report](https://github.com/cloudrift-ai/emmy/blob/5670caae/plans/nvfp4-qwen-mixed-serving-progress.md)
and [5090 tuning log](https://github.com/cloudrift-ai/emmy/blob/5670caae/plans/nvfp4-qwen-mlp-tuning-5090.md)
retain the full-model experiments. The older performance plan and its linked investigations remain
historical inputs, not proof that their proposed changes are implemented here.

## Verified failures and extracted fixes

| Area | Failure on unpatched main | Shared fix | Regression scope |
| --- | --- | --- | --- |
| BF16 NumPy evaluation | Numeric values were cast to `uint16` rather than encoded as BF16 bits; `1.0` became `0x0001`, and generated FP4 lookup tables could contain NaNs. | Keep the logical BF16 type through coercion, decode before arithmetic, and encode at each BF16 result boundary. | NumPy arithmetic and a computed BF16 constant. |
| BF16 program boundaries | A BF16 device output appeared as `torch.uint16`; a BF16 input was numerically cast to the `uint16` carrier; the same source used as BF16 and F32 plan constants could reuse the wrong upload. | Return a logical BF16 tensor view; encode plan inputs with the canonical helper; choose physical constant storage and cache keys from each plan buffer's dtype. | Buffer view, generation split input, BF16/F32 source rebinding. |
| NVFP4 activation scaling | Fusing the FP32 global scale into an FP16 block scale before FP4 code selection changed threshold nibbles from the stock operation order. | Keep the FP32 code divisor, derive a layer-specific global inverse through a reusable reciprocal plan load operation, and serialize/rebind that operation. | Independent threshold vector, saved-plan round trip and rebinding. |
| Packed FP4 correctness | The CLI compared raw packed bytes, so `0x88` and `0x00` negative/positive zero falsely failed. | Decode only logically `f4e2m1x2` `uint8` carriers for comparison, preserving ordinary U8 semantics and rejecting changed nonzero nibbles. | Signed-zero, both nibble positions, U8 comparison. |
| First whole-program TMA capture | The runtime encoded/uploaded a TMA descriptor and synchronized its stream inside CUDA graph capture. | Prepare descriptors before capture, as the existing per-launch capture path already does. | Real descriptor capture before any warm run, then replay with changed input on sm_90+. |
| Explicit nested placement pins | A hand-pinned parent cut marked children terminal, preventing a named child output cut; a scoped pin could be reported unused despite its local cut being taken. | Allow a `PLACE@place_<terminal-token>/<relative-site>` child choice and record the exact applied source key/value. Audit bare schedule pins only on kernels without an overriding scoped pin. | Parent/child cut, stale child key, sibling preservation, receipt/retry, bare/scoped precedence. |

The nested placement change is included as a generic mechanism, not a Qwen-specific fusion rule. It
preserves the maximal-fusion invariant and existing parent-only pin behavior. The cut fixture produces
two fused contractions after parent pins alone and separate single-contraction children after named
child pins. Its exact piece tokens are graph-specific; the MLP tokens and measured golden rows from
PR #993 cannot be copied into a future full-model graph. Full-model work will need fresh traces,
explicit choices, and measurements.

## Independent main-only regression gate

The disposable base worktree was checked out at `754d1afa`. Only the new regression test files were
copied into it; production code, mixed serving modules, fixtures, and goldens remained at main.
Running the same nine selected tests on base and this branch gave:

| Source | Result |
| --- | --- |
| Base `754d1afa` | 8 failed, 1 passed in 2.21 s. The failures cover BF16 arithmetic, computed constant, logical buffer view, BF16/F32 binder cache, reciprocal plan serialization, FP4 threshold vector, packed-FP4 comparison, and nested placement. Bare/scoped precedence already passed this selected case. |
| Extracted `9d349929` | 9 passed in 2.21 s. |

Additional focused tests on the extracted branch: 10 BF16 cases passed; 36 FP4/plan cases passed;
7 placement/pin/packed-comparison cases passed. These are component checks, not evidence that full
NVFP4 model serving or stock byte parity is established. The generation split feed and program
view are covered, while complete BF16 `EmmyGenRunner.from_loaded` remains outside this extraction:
that path still calls `np.dtype("bfloat16")` and needs a separate end-to-end qualification.

The TMA GPU test is pending on this new branch. Local `cargo check --offline -p emmy-runtime` passes.
The existing 5090 host has byte-identical runtime and test source files from PR #993 (both SHA256
digests match this branch), but its running server occupies about 31.8 GiB. In a separate read-only
GPU container, Emmy's device initialization returned `CUDA_ERROR_OUT_OF_MEMORY`; pytest skipped the
TMA case because the runtime could not report its SM version. Automatic approval review rejected
exporting new/base repository snapshots to that host, and separately rejected stopping the running
server, citing missing explicit authorization for each action. Neither rejected action was retried
through another route. The actual GPU capture/replay gate remains open.

## Findings to retain for full Emmy NVFP4 work

The RTX 5090 hand sweep in PR #993 used explicit pinned decisions because the prior was unreliable
for this workload. Useful candidate geometry included native FP4 M-axis expansion and async staging.
Those choices were measured only for the mixed MLP shapes. Isolated synthetic `emmy run --ir`
benchmarks generated all-zero packed FP4 weight bytes for some random-source fixtures, limiting their
performance and accuracy representativeness. The later real-checkpoint, matched serving comparisons
showed flat or modestly worse end-to-end latency despite isolated M=64 kernel gains. A future full-
Emmy implementation should first measure real-checkpoint MLP time inside its own serving path, then
pin and measure its newly traced pieces. No whole-model speedup is claimed here.

Follow-up issues remain separate from the fixes above:

- The general `_wrong_answer_flag` comparator can miss a mismatched NaN or infinity because
  `max(0, nan)` retains zero. A legitimate matching `-inf` mask must still be accepted.
- Synthetic packed-FP4 source generation can collapse to zero bytes, so correctness and timing
  fixtures need nontrivial codes and positive, realistic scales.
- Tiny-block reciprocal and approximate-instruction differences can still change a few FP4
  threshold nibbles. The bounded correction here matches the stock scaling order, but it does
  not promise byte identity with every hardware approximation.
- Some larger FP4 tile fragments and global stage pins were refused in the mixed MLP sweep.
  The observed N=16 blocked axis explains part of the fragment limit; other failures remain
  unclassified until reproduced on a standalone full-model trace.
- The TMA descriptor preparation fix addresses first capture only. A TMA schedule's performance
  and numerical behavior still require separate full-model evidence.

## Remaining qualification

Run the real first-capture TMA regression against a rebuilt native extension on RTX 5090. Run the
affected compiler, runtime, and existing generation tests on this branch; verify repository golden
freshness and realization cases without re-recording rows; then run required full tests and lint.
Publish the separate draft PR with exact test results and limits. PR #993 stays draft and paused.
