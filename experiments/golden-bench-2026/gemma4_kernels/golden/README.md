# Gemma 4 12B kernel goldens

One working golden per kernel and card, named `KERNEL-s512_CARD.golden.yaml`. Each holds the traced FP16 program, its
seed realization, and one receipt per lane: standard and `EMMY_FAST_MATH=1`.

## RTX 5090

| Golden | Matmul (M x K @ K x N) | Standard lane | Fast-math lane |
| --- | --- | --- | --- |
| `q_proj` | 512 x 3840 @ 3840 x 4096 | `f16_f32/f2x4/k2`, `g4k` | `f16_f16/f4x8/k4`, `g2k` |
| `kv_proj` | 512 x 3840 @ 3840 x 2048 | `f16_f32/f2x4/k2`, `g8k` | `f16_f16/f4x8/k4`, `g4k` |
| `o_proj` | 512 x 4096 @ 4096 x 3840 | `f16_f32/f2x4/k2`, `g2k` | `f16_f16/f4x8/k4`, `g2k` |
| `mlp_gate_up` | 512 x 3840 @ 3840 x 30720 | `f16_f32/f4x8/k4`, unsplit | `f16_f16/f4x8/k4`, unsplit |
| `mlp_down` | 512 x 15360 @ 15360 x 3840 | `f16_f32/f2x4/k2`, `g4k` | `f16_f16/f4x8/k4`, `g2k` |

Every projection row runs `WORK=w4x2` over `STAGE=d2/smem-tma`; the tile's atom prefix is `mma_m16n8k16_`.

| Attention lane (`WORK=w4x1`) | Value expectation | Score | Ring |
| --- | --- | --- | --- |
| Standard | `f16_f32/f1x32/k2` | `f16_f32/f1x4/k8`, `RASTER=gm8` | `d2/smem-tma` |
| Fast-math | `f16_f16/f1x32/k2` | `f16_f32/f1x4/k4` | `d3/smem-tma` |

## RTX 4090

Every row stages through `STAGE=d2/smem-async`. Projections run `WORK=w2x2` unless noted; attention runs `WORK=w4x1`.

| Golden | Standard lane | Fast-math lane |
| --- | --- | --- |
| `q_proj` | `f16_f32/f2x8/k2`, `gm8` | `f16_f16/f4x8/k4`, `gm8` |
| `kv_proj` | `f16_f32/f4x4/k8` | `f16_f16/f2x8/k8` |
| `o_proj` | `f16_f32/f2x8/k2` | `f16_f16/f2x8/k4`, `w4x2` |
| `mlp_gate_up` | `f16_f32/f2x8/k2`, `gm8` | `f16_f16/f4x8/k4`, `gm8` |
| `mlp_down` | `f16_f32/f4x8/k2`, `g2k` | `f16_f16/f4x8/k4`, `gm8` |

| Attention lane | Value expectation | Score |
| --- | --- | --- |
| Standard | `f16_f32/f1x32/k2` | `f16_f32/f1x4/k4` |
| Fast-math | `f16_f16/f1x32/k2` | `f16_f32/f1x4/k2` |
