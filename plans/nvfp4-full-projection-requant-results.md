# NVFP4 full-projection requant: plan and measured results

This document records the scope and evidence for the full-projection cut in PR #737. Measurements are
from `nvidia/Qwen3-8B-NVFP4` on an RTX 5090. They compare the new routed Emmy programs with Emmy's
previous fused choices; they are not PyTorch performance comparisons.

## Delivered compiler behavior

Some NVFP4 output-requant projections have multiple stores whose contraction roots require different
grid axes. Keeping the entire projection fused can therefore leave no legal tiled binding and collapse
to a nearly serial kernel.

This change:

1. Promotes an axis read by every reduction in a term to the grid. A reduce-free piece promotes its
   store sweep only when placement has no other free axis.
2. Offers a single recorded full-projection cut when the fused projection cannot bind its outputs. The
   cut separates contraction occurrences, hoisted reductions, and output-owning branches.
3. Records the routing rows used by a realization in `kernel_set`. Verification, replay, and named
   benchmarking read through that set, while old golden files remain compatible.

The cut is evidence-gated. Merely offering it does not change the greedy choice: measured compile time
without a pin stayed within 0.2 seconds of the base behavior. Taking the route adds about 24 seconds for
the gate/up program and 60 seconds for the post program, paid when producing the reusable plan pack.

## Observable results

Pinned route measurements at decode widths 1 and 16:

| Program | Prior Emmy path | Full-projection route | Resulting program |
| --- | ---: | ---: | --- |
| gate + up + requant | 199.4 ms | 201 us | 10 kernels with MMA rows |
| output + norm + requant | over 2 s / timeout | 119-123 us | 11 kernels with MMA rows |

The sweep found 27 applicable realizations. It recorded 24 of them after two raised-budget retries,
producing 213 measured routing/child rows. The cleaned working artifact attaches all 24 seeds to measured
routing rows through `kernel_set` and removes the route knobs from the three hand-authored, unmeasured
`ead322` seeds. Its SHA-256 is `a684ffeee8caedf02b3fafb69b4f181ca284ecfeb4ff495aa8b23d6be8ebc2c5`.

The other 27 formerly unmeasured realizations correctly do not offer this arm. Their projections bind,
and their remaining gap belongs to split-route recording rather than this cut.

## Serving correctness gate

`scripts/validate_serve.py` independently decodes the checkpoint's packed NVFP4 weights for the eager
reference. This avoids both the checkpoint framework's unrelated quantizer and the earlier weak check
that compared two Emmy executions sharing the same possible fault.

The measured bucket-16 lane returned the same first token as the decoded eager reference for all four
built-in prompts:

| Prompt suffix | Reference and Emmy token |
| --- | --- |
| `The capital of France is` | ` Paris` |
| `The three primary colors are` | ` red` |
| `Water is made of hydrogen and` | ` oxygen` |
| `Once upon a time, in a small village,` | ` there` |

A minimal reproducible gate is:

```bash
EMMY_PACK_DIR=/path/to/pack \
python scripts/validate_serve.py \
  --model nvidia/Qwen3-8B-NVFP4 \
  --golden /path/to/measured.yaml \
  --decode-bucket 16 \
  --enforce-eager \
  --prompt-count 1 \
  --max-tokens 1 \
  --max-model-len 4096 \
  --max-num-batched-tokens 256
```

Use a bucket represented by measured evidence. Bucket 32 is not covered by this sweep. `--enforce-eager`
also prevents vLLM graph-capture probes at widths 24 and 32 from leaving the bucket-16 twin; it removes
about 13 minutes of irrelevant first-boot capture from this correctness check.

The first bucket-16 validation completed in about 30 minutes: roughly 10 minutes of Emmy compilation/model
setup, 13 minutes of avoidable graph capture, and the serving checks. The final one-prompt validation used
the cleaned artifact, hit the 288-plan pack, rebuilt all 36 layers in about 17 seconds, loaded the model in
44 seconds, initialized the engine in 92 seconds, and returned the matching token about three minutes after
starting the server. Independent reference loading and decoding remains a separate few-minute phase.

## Verification status

- Focused validator tests: 2 passed; focused lint and format checks passed.
- Golden-recording CLI tests after fixing cross-regime row aliasing: 23 passed; focused lint and format
  checks passed.
- Broad golden/search/realization/cut subset: 1480 passed, 54 skipped, 17 xfailed. One CUDA worker died in
  the parallel run; its NVFP4 slab-loopify case passed when rerun serially. One RMS-norm reference case
  still fails serially because its NumPy reference lacks `p_weight`; that fixture and its execution path
  are unchanged from this PR's base.
- Cleaned-artifact serving gate: 1/1 exact first-token match, with routed Emmy programs visible in the
  server's compiled-program log.

Steady-state TPOT optimization, comparison against PyTorch, prefill improvements, value CSE across cut
seams, and the unrelated split-route gaps are follow-up work rather than acceptance criteria for this PR.
