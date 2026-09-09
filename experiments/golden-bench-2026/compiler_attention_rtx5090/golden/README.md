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

## What blocks recording them today

Nothing here has been recorded yet, and hand tuning does not currently reach a schedule worth recording. Measured on
an RTX 5090 at `366caa52d`:

- A freshly traced `F.scaled_dot_product_attention` compiles to a kernel identity that no recorded row matches, so the
  greedy pick falls through to the prior. Causal MHA at one batch, 8 heads, 512 keys and head dimension 128 runs at
  **20015 us** against **14 us** eager.
- That same program IS in this card's repository hardware golden, recorded at **14.91 us**, and replaying that
  realization reproduces it (**16.8 us** measured). The recorded row is reachable only through the golden's own
  embedded realization, which carries the routing that steers the structural decisions. Pinning `PLACE`, `WORK`,
  `TILE` or `STAGE` on the traced program does not reach it: the traced program keeps route keys under `@inner...`
  while the recorded row is spelled under `@map.1/twist...`, so the pin matches no site.
- Much of the worker inventory does not compile at all on these programs. Four of seven `WORK` values raise
  `materialize: kernel ... reads names it never binds: ['acc4']`, two fail in nvcc, and the greedy pick for causal
  prefill at 1024 keys does not finish a single launch within 60 s.

This is not a recent regression. The same fresh trace already measured 20015 us at `39d9f91ef`, the commit that
recorded this card's causal attention rows, and at every commit since.

So the gap is structural routing, not schedule search: until a fresh trace can be steered onto the fused structure
the recorded rows describe, there is no schedule to hand-pin and record here.
