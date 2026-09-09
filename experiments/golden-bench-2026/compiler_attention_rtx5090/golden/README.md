# RTX 5090 attention goldens

This directory receives the recorded goldens for the RTX 5090 attention comparison. The measurement recipe does not
tune and fails when any requested golden is absent.

For each operator, batch, and sequence length that `operators.sh` offers, print the exact source with:

```bash
experiments/golden-bench-2026/compiler_attention_rtx5090/operators.sh OPERATOR BATCH SEQUENCE_LENGTH
```

Trace that source, tune it on the RTX 5090, validate it, and record it as
`OPERATOR-bBATCH-sSEQUENCE_LENGTH.golden.yaml`. Commit every offered setup before running the Emmy lane. Keep the
tuning records outside the measurement recipe; the committed golden is the schedule input that the recipe replays.

## What used to block recording them

Nothing here has been recorded yet, but the structural blocker is gone. Until the batched-contraction orientation
fix, a freshly traced `F.scaled_dot_product_attention` reached a Fold tree no recorded row could name: the tree kept
its route keys under `@inner…` while every recorded attention row is spelled under `@map.1/twist…`, so the greedy
pick fell through to the prior and no pin matched a site either. Causal MHA at one batch, 8 heads, 512 keys and head
dimension 128 ran at **20015 us** against **14 us** eager.

The cause was that the same query-key product oriented one way in one of the kernel's contractions and the other way
in another, which stopped the twisted rewrite recognizing the two as one score and demoted flash attention to its
two-pass form. Which way it went depended on what the traced program's inputs were called. That is why the card's own
hardware golden — recorded from a source that spells its tensors `x0`, `x1`, `x2` — held a row for this exact shape
that no ordinary trace could reach.

Measured on an RTX 5090 after the fix: the same fresh trace deploys the recorded schedule at **18.6 us**, and
`emmy compile --strict-evidence` passes on it. The GQA compile that used to raise
`materialize: kernel … reads names it never binds: ['acc4']` on the greedy pick now emits CUDA. Causal prefill at
1024 keys no longer hangs; it builds the same fused structure, and what remains there is an ordinary evidence gap —
`--strict-evidence` names the kernel and its `030_cut` fork as having no measured row, which is what recording a
golden here is for.

## What still has to happen

Each offered setup still needs its schedule measured and recorded. The routes the recorded rows use are now the routes
a trace produces, so a hand pin reaches them (`EMMY_KNOBS="TILE@map.1/twist=…,WORK=…"`) and
`emmy run --golden PATH --bench --record-greedy` can write what it measures back. Compile time at the larger sequence
lengths is the practical obstacle, not reachability.
