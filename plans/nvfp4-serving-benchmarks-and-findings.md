# NVFP4 serving: measured findings from the W4A4 serving PR

Benchmark findings and numbers from PR #695, preserved while the follow-up work (the output-requant
split, the golden re-record, the 27B hybrid track) is in flight. We measured everything below on
nvidia/Qwen3-8B-NVFP4 on a rented RTX 5090 at #695's final heads; re-measure on the head that uses them.

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

## Gotchas that already cost time

The corpus case matmul/imap-transpose-a-tma_xfail_correct hard-faults the GPU when its correct stage
runs and poisons the CUDA context for the next test in its worker. A corpus case recording a multi-root
kernel with a real grid cannot finish the offered stage in practical time — the stage materializes the
kernel set's complete row set; pin the composed seams in the case, or keep the coverage in Python tests.
Tune logs no knob row for a failing candidate; reproducing one means patching the bench to dump the
kernel source and knobs on failure. tests/compiler/realization/helpers.py's regenerate restamps only the
first realization's identity in a case; later entries need hand-restamping. We measured the sites-memo
prototype number above under load on a shared GPU server — re-measure before quoting it further.
