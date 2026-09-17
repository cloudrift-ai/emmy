# Gemma 4 12B kernel goldens

One working golden per kernel of a Gemma 4 12B decoder layer at sequence length 512 — the five projections and the
sliding layers' causal attention — per card, named `KERNEL-s512_CARD.golden.yaml`. Each holds the traced FP16
program, its seed realization, and the receipts of the fastest row measured on that card by hand pin, once in the
standard lane and once under `EMMY_FAST_MATH=1`. The recipe replays them as they are, does not tune, and fails when a
golden is absent. A schedule is a claim about one card: a row recorded on one is never replayed on another.

## RTX 5090

| Golden | Matmul (M x K @ K x N) | Standard lane | Fast-math lane |
| --- | --- | --- | --- |
| `q_proj` | 512 x 3840 @ 3840 x 4096 | `f16_f32/f2x4/k2`, `g4k` | `f16_f16/f4x8/k4`, `g2k` |
| `kv_proj` | 512 x 3840 @ 3840 x 2048 | `f16_f32/f2x4/k2`, `g8k` | `f16_f16/f4x8/k4`, `g4k` |
| `o_proj` | 512 x 4096 @ 4096 x 3840 | `f16_f32/f2x4/k2`, `g2k` | `f16_f16/f4x8/k4`, `g2k` |
| `mlp_gate_up` | 512 x 3840 @ 3840 x 30720 | `f16_f32/f4x8/k4`, unsplit | `f16_f16/f4x8/k4`, unsplit |
| `mlp_down` | 512 x 15360 @ 15360 x 3840 | `f16_f32/f2x4/k2`, `g4k` | `f16_f16/f4x8/k4`, `g2k` |

Every projection row runs `WORK=w4x2` over a two-slot TMA ring (`STAGE=d2/smem-tma`); the tile's atom prefix is
`mma_m16n8k16_`. `g<n>k` is the cross-CTA split of the contraction axis with a separate finalize kernel, which the
whole-program latency includes.

### Attention

`attention` is `scaled_dot_product_attention` over `(1, 16, 512, 256)` FP16 inputs with `is_causal=True`, one fused
kernel with two schedule sites: the value expectation `TILE@map.1/twist` spans the 256-wide head in one warp column
(`f1x32`) over 32-key chunks (`k2`), and the score `TILE@map.1/twist.1/inner` tiles those 32 keys (`f1x4`). Both
operands ride a TMA ring, four warps per CTA (`WORK=w4x1`).

| Lane | Value expectation | Score | Ring |
| --- | --- | --- | --- |
| Standard | `f16_f32/f1x32/k2` | `f16_f32/f1x4/k8`, `RASTER=gm8` | `d2/smem-tma` |
| Fast-math | `f16_f16/f1x32/k2` | `f16_f32/f1x4/k4` | `d3/smem-tma` |

The fast-math row accumulates the value product in FP16 and promotes it into the FP32 carrier once per chunk; the
score and the softmax statistics stay FP32. The rows swept were the two- and three-slot rings and the single-slab
64-key form (`f1x32/k4` over `f1x8`, `d1/smem-tma`), each with both accumulators. The rings land within 2% of each
other; the single slab is 3% to 8% behind them, and there FP16 accumulation is worth 5% (41.2 to 39.0 us).

## How a golden is recorded

Trace the matmul with FP16 inputs, bench a few pinned rows on the card in each lane, then record the fastest one
under the seed's exact name, once per lane:

```bash
E=experiments/golden-bench-2026/gemma4_kernels
code="torch.matmul(torch.randn(512,3840,dtype=torch.float16,device='cuda'), torch.randn(3840,4096,dtype=torch.float16,device='cuda'))"
emmy trace -c "$code" -o $E/golden/q_proj-s512_rtx5090.golden.yaml
emmy run --golden $E/golden/q_proj-s512_rtx5090.golden.yaml --realization k_matmul_843d4a --bench --bench-backends eager,emmy \
  --ab "WORK=w4x2,TILE=mma_m16n8k16_f16_f32/f2x4/k2,STAGE=d2/smem-tma,REDUCE=g4k" --ab "…"
EMMY_KNOBS="<the fastest row>" emmy run --golden $E/golden/q_proj-s512_rtx5090.golden.yaml \
  --realization k_matmul_843d4a --bench --record-greedy
EMMY_FAST_MATH=1 EMMY_KNOBS="<the fastest fast-math row>" emmy run --golden $E/golden/q_proj-s512_rtx5090.golden.yaml \
  --realization k_matmul_843d4a --bench --record-greedy
```

Build the inputs with `dtype=torch.float16`. A `.half()` on an FP32 `randn` traces an FP32 input and a cast, the
operands become computed values, and no TMA stage resolves for them.

The rows swept per kernel were the two tiles above crossed with the splits `g2k`, `g4k`, `g8k` and no split. The
split that wins moves with the shape: the narrow outputs take a wide split in the standard lane, the 30720-wide
gate/up output loses to any split, and the fast-math tile prefers `g2k` except on the narrowest output.

`--record-greedy` refuses a row whose `EMMY_KNOBS` pin did not realize, so read its exit status. Bench at the
harness defaults (10 warmups, 100 iterations): on this display-attached card a longer loop reads eager about 10%
slower, and only ratios within one run compare.
