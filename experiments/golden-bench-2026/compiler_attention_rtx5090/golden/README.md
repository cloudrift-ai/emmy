# RTX 5090 attention goldens

One working golden per setup `operators.sh` offers, named `OPERATOR-bBATCH-sSEQUENCE_LENGTH.golden.yaml`. Each holds
the setup's traced program, its seed realization, and the child-identity receipt of the fastest row measured on the
card by hand pin in the standard lane. The measurement recipe replays them as they are (`run_emmy.sh`), does not tune,
and fails when any offered setup's golden is absent.

## How a golden is recorded

Print the setup's exact source, trace it into a working golden, then bench a few pinned rows on the card and record
the fastest one under the seed's exact name:

```bash
E=experiments/golden-bench-2026/compiler_attention_rtx5090
code=$($E/operators.sh prefill_causal 1 1024)
emmy trace -c "$code" -o $E/golden/prefill_causal-b1-s1024.golden.yaml
emmy run --golden $E/golden/prefill_causal-b1-s1024.golden.yaml --realization k_sdpa --bench \
  --ab "TILE@map.1/twist=mma_m16n8k16_f16_f32/f1x16/k2,TILE@map.1/twist.1/inner=mma_m16n8k16_f16_f32/f1x4/k4,\
STAGE@map.1/twist=d2/smem-tma,STAGE@map.1/twist.1/inner=d2/smem-tma,WORK=w4x1" \
  --ab "…"    # the other rows of the family, see below
EMMY_KNOBS="<the fastest row>" emmy run --golden $E/golden/prefill_causal-b1-s1024.golden.yaml \
  --realization k_sdpa_d8a514.e29cd0cf61e1 --bench --record-greedy
```

The rows are FlashAttention-2's geometry at head width 128: the value expectation (`TILE@map.1/twist`) spans the
whole head in one warp column (`f1x16`) over 32-key chunks (`k2`), the score (`TILE@map.1/twist.1/inner`) tiles those
32 keys (`f1x4`) and chunks the head width at `k4` or `k8`, both operands ride a two- or three-slot TMA ring
(`d2/smem-tma`, `d3/smem-tma`), four warps per CTA (`w4x1`). The three land within a few percent of each other on
every setup measured; the receipt keeps whichever won. Once a receipt exists the seed must be named exactly, because a
substring then matches both.

A causal setup's kernel stops its key stream at the CTA's own diagonal (the chunk tier's early stop), which is what
puts the causal rows ahead of eager at 1024 keys and above; a global setup runs every chunk.

## Coverage

Prefill setups (`prefill_causal`, `prefill_global`, `prefill_gqa`) at batch 1 and 8 are recorded as they land; the
measured numbers are in `../RESULTS.md`. GQA decode (`decode_gqa`) is recorded through its broadcast form, whose
eight query heads per group become the row axis of the same fused kernel. Plain decode (`decode_causal`) is not
recordable today: with one query row the carrier offers no tensor-core tile at all — only `WORK`, `REDUCE` and
`RASTER` — so there is no schedule worth a golden until its output tile can be widened to a whole head per CTA.
