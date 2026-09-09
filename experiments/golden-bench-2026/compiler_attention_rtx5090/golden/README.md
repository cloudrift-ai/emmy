# RTX 5090 attention goldens

This directory receives the recorded goldens for the RTX 5090 attention comparison. The measurement recipe does not
tune and fails when any requested golden is absent.

For each operator, batch, and sequence length in `operators.sh`, print the exact source with:

```bash
experiments/golden-bench-2026/compiler_attention_rtx5090/operators.sh OPERATOR BATCH SEQUENCE_LENGTH
```

Trace that source, tune it on the RTX 5090, validate it, and record it as
`OPERATOR-bBATCH-sSEQUENCE_LENGTH.golden.yaml`. Commit all 60 files before running the Emmy lane. Keep the tuning
records outside the measurement recipe; the committed golden is the schedule input that the recipe replays.
