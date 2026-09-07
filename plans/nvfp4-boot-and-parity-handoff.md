# NVFP4 serving: boot time, generation speed, and the 27B parity track

Handoff after PR #695 (the W4A4 serving PR). The serving mechanism is done and verified there: serving
compiles the declared W4A4 program, GPU parity holds on the RTX 5090, and the output-owning cut gives
the fused NVFP4 GEMM kernels real grids. What remains is speed and scale. We measured all numbers below on nvidia/Qwen3-8B-NVFP4 on a rented RTX 5090 at #695's final heads; re-measure on the head
that picks this up.

## Measured state

Boot attribution, three consistent runs. The boot never reached a healthy server; the last run stopped
after seven hours still inside vLLM's warm-up passes:

| boot phase | measured |
| --- | --- |
| layer-0 kernel compiles (8 per-layer serving programs) | 2186 s — the post-attention programs resolve twice |
| weight bind | 45 s |
| boot roofline audit | 721 s — 4 of the 8 programs (the audit skips symbolic widths), each timed as one warmup plus three repetitions |
| vLLM warm-up passes | the remaining six-plus hours, stopped incomplete |

The cause: the output requant — re-quantizing a layer's result into packed 4-bit codes plus block
scales — is generated inside the layer's final kernel. Its packed-code output carries six contraction
roots (a contraction root: a reduction whose result the kernel's output reads), and the kernel binder
requires an output-tiled root to own each output specification independently. The binder refuses every tiled plan, the greedy retires its structural pick, re-resolves down the keep-fused branch, and lands on
near-serial scalar code at ten-thousand to a million times over the roofline floor. ir/tile/ops.py's own docstring states the escape: the same computation is unbindable as one region of one kernel and
ordinary as a kernel of its own.

Reference numbers worth keeping. The long-input post-attention program: 23.2 seconds per measured
forward at main's #703 election record, 114.4 seconds per audit repetition at #695's final heads — the
two timings use different harnesses, and reconciling them is part of the regression bisect below. The
decode-width post-attention program: 23.9 seconds per audit repetition; a generated token runs the
per-layer programs across all 36 layers, so generation stays at tens of seconds to minutes per token
until the requant split lands. The output-owning cut's controlled comparison: the post-attention kernel
failed every timing budget at 25.7 seconds per launch on the base commit and measures 21.7 milliseconds
— a thousand-fold shift — under the recorded route with the cut.

The realization matrix: 87 of 128 realizations verified as measured rows at #695's final head (emmy run
--record-greedy covered the splits with no route). The 41 others are the requant family exceeding their
timing budgets: 8 of the 16 GEMM kernels blocked at all four widths (32 rows) plus 9 further
width-specific rows. Those 8 kernels are the split's work queue.

## Work items, in order

1. Split the output requant into its own kernel. Extend the placement-cut machinery so the packed-code
   branch with its six contraction roots leaves the fused kernel; each resulting kernel then satisfies
   the binder's ownership rule and reaches the block-scaled tensor-core cell. Design before code, as
   #695 did for the output-owning cut; ir/tile/ops.py's output_regions and the cut's offer condition
   (it fires only where the piece gains a grid axis) are the starting material. Verification: the eight
   all-width-blocked GEMM kernels gain benchable rows; the boot roofline audit flags nothing (its
   warning bar is ten times the floor); warm-up drops from hours toward minutes; generation leaves the
   tens-of-seconds-per-token floor.
2. Bisect the post-attention slowdown: 23.2 seconds per forward at #703's election record against 114.4
   seconds per audit repetition at #695's final head. First establish the two harnesses measure the same
   thing; then walk the merges between the two records.
3. Boot-time quick wins, independent of item 1. Set EMMY_PACK_DIR for any repeated boot: the
   execution-plan pack (emmy/compiler/backend/pack.py) round-trips — it is unset outside the serving
   image, so every bare-host boot recompiles all eight programs; the win applies from the second boot
   onward and invalidates when the checkpoint's quant digest changes. Memoize path.sites on the root
   Fold: a prototype measured 1.8-5x on schedule resolution with byte-identical kernel sources;
   re-measure uncontended before landing. Give the boot roofline audit a repetition cap: the first timed
   repetition decides its verdict, the remaining two refine a number nobody reads, and its module
   docstring already claims once-at-boot.
4. Re-record the serving golden once item 1 lands, promote it canonically, and ship the serving env file
   with it in one commit. The release skill's golden realization audit (its GATE step, run with strict
   evidence) requires every realization verified, which requires the 41 blocked rows to bench.
5. The 27B bridge: hybrid serving for Qwen3.6/3.8-27B. Emmy cannot run the DeltaNet (linear-attention)
   layers as of #695's merge: the serving runner carves SDPA out of every layer uniformly, the
   trace-side carve for linear-attention layers is absent from the tree (removed before #499's squash;
   #695's history records that it re-applied cleanly when last tested), and a DeltaNet layer's pre-step
   returns four tensors where the consumer unpacks three. Needed: restore the carve, fix the unpack,
   give the runner per-layer-kind dispatch, and wire vLLM's DeltaNet cache for those layers while emmy
   programs serve the full-attention ones.
6. vLLM parity numbers belong to the 27B track. Stock vLLM 0.23 cannot load nvidia/Qwen3-8B-NVFP4 (its
   kv-cache-scale loader crashes on the repack's 72 kv-scale tensors), so the 8B carries no stock
   baseline; the 27B repacks carry none of those tensors. The golden-bench recipes under
   experiments/golden-bench-2026/ define the reference benchmark matrices, and the
   recipes/Qwen3.8-27B* recipes carry stock serving numbers to compare against.

## Gotchas that already cost time

The corpus case matmul/imap-transpose-a-tma_xfail_correct hard-faults the GPU when its correct stage
runs and poisons the CUDA context for the next test in its worker. A corpus case recording a multi-root
kernel with a real grid cannot finish the offered stage in practical time — the stage materializes the
kernel set's complete row set; pin the composed seams in the case, or keep the coverage in Python tests.
Tune logs no knob row for a failing candidate; reproducing one means patching the bench to dump the
kernel source and knobs on failure. tests/compiler/realization/helpers.py's regenerate restamps only the
first realization's identity in a case; later entries need hand-restamping. We measured the sites-memo
prototype number above under load on a shared GPU server — re-measure before quoting it further.
