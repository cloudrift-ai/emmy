# A causal SDPA's expectation never joins the twisted carrier

Status: measured on an RTX 5090, 2026-09-08, after the fused-attention work landed. This is item 3 of the retired
`fused-attention-performance-gap` memo, restated with its cause found and nothing else left of that memo open.

## What happens

`F.scaled_dot_product_attention(..., is_causal=True)` traces to a shape the chunk tier cannot fold. The softmax DOES
fuse — its running max and denominator become a `twist=softmax` carrier — but the value channel stays a separate
top-level contraction over its own copy of the key axis, with the whole softmax recomputed underneath it. So
`Fold.chunked` is false, the chunk tier never applies, and the greedy ships whatever the planar tiers offer.

    emmy compile --ir tile -c "F.scaled_dot_product_attention(q, k, v, is_causal=True)"

    Fold[a6 in 0..512] contraction                        <- the expectation, on its own key axis
    +- operand[...]: Fold free
       +- operand[acc1, acc3]: Fold[a2 in 0..512] reduce  <- the twist: max and denominator only
                               (twist=softmax)               and a second copy of it beside this one

Against the non-causal program at the same shape, whose carrier holds three states — max, denominator and the
expectation — over ONE key axis.

`rewrite_twisted` names the refusal under `compile -vv`:

    twisted rewrite declined (sibling cluster): 'acc0' keeps 2 same-axis sibling(s) no recipe fuses onto it

The two sweeps carry DIFFERENT axis names for the same extent, so `_click` finds no reduce reading a reduce and no
recipe clicks them together.

## What it is not

- **Not the head width.** (1, 8, 512, 64), (1, 16, 512, 64), (1, 8, 512, 128) and (1, 16, 512, 256) all decline the
  same way, and the non-causal form of each fuses.
- **Not the mask reaching the tier.** `tests/compiler/realization/cases/attention/sdpa-hd128-causal-mask-mma.yaml`
  is a causal chunked carrier that offers, realizes, builds and runs. Its target is stored Loop IR, so it enters
  below the pass that declines here.
- **Not a tile-schedule gap**, which is why it earns no corpus case: the chunk tier's site does not exist on this
  program at all, so a pinned schedule naming it is silently inapplicable rather than refused, and the case would
  assert nothing. It needs a Python test on the fused shape, or a fix.

## Why it matters

It is the whole cost of the gemma-4 shape. Causal (1, 16, 512, 256) is what the published 29.7 us against torch's
30.7 us was measured on; today the same call reaches no fused kernel. The non-causal form of that shape now runs
64 us against eager's 41 us on the same card, so the tier handles a 256-wide head — the causal decomposition is
what is missing.

## Where to start

`rewrite_twisted` / `_click` in `emmy/compiler/pipeline/passes/lowering/tile/_twist.py`, and above it whatever gives
the two key sweeps different axis names. The question is whether the causal program can be made to present ONE key
axis with the expectation nested under the twist, the way the non-causal program already does — a frontend or
loop-fusion question, not a scheduling one.
