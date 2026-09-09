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
producing 213 measured routing/child rows. The remaining three `ead322` rows were hand-authored and
unmeasured; they must not be published as evidence.

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

The observed fresh bucket-16 validation completed in about 30 minutes: roughly 10 minutes of Emmy
compilation/model setup, 13 minutes of avoidable graph capture, and the serving checks. With
`--enforce-eager`, a cold verifier should budget approximately 10-15 minutes; a matching plan pack makes
later boots faster.

## Remaining publication work

- Attach each measured realization to its routing rows through `kernel_set`.
- Remove the three unmeasured `ead322` authored rows.
- Decode, replay, and run the short serving gate against that cleaned artifact.
- Record the final commands and artifacts in the PR description.

Steady-state TPOT optimization, comparison against PyTorch, prefill improvements, value CSE across cut
seams, and the unrelated split-route gaps are follow-up work rather than acceptance criteria for this PR.
