# GDN1 in_proj: each view of the projection gets its own workspace

Status: investigation done on CPU; the fix is not chosen yet. Prose check pending.

## Observed

On the RTX 5090 (main a951e6c0), the width-1 GDN program's input projections read 399 MB of DRAM per layer (ncu, cold
cache) where the BF16 weights are 168 MB: `in_proj_qkv` (104.9 MB) about 2.6 times and `in_proj_z` (62.9 MB) twice.
They cost about 14.5 ms per decode token; stock vLLM reads the same weights once in about 5 ms.

## Which decision produces the pieces

The golden's routing row on the fused GDN kernel `k_conv1d_linear_mean_reduce_970a92` takes the full-projection cut
(19 `PLACE@…=cut` keys; `full_projection_seams` on the fresh lowering returns exactly those 19). Its contraction
workspace producers fuse into one projection kernel (`…__place_b770740dbf`), and a second routing row cuts that
kernel's outputs apart (`PLACE@map.{1..5}/inner=cut`) into the six serving kernels.

The projection kernel's Loop IR (golden, `…__place_b770740dbf`) has one loop per workspace:

| Loop | Weight columns | Workspace | Reader | Serving kernel |
| --- | --- | --- | --- | --- |
| `a11 in 0..10240` | qkv, all | `b1a6f8ed42` f32 [10240] | conv state store | `__place_96abea9776` |
| `a4 in 0..2048` | qkv 0..4095 (q, k twin) | `87a4fde736` f32 [16,128] ×2 | conv, recurrence | `__place_08306696ae` |
| `a2 in 0..6144` | qkv 4096.. (v) | `c9b1165b32` f32 [48,128] | recurrence | `__place_9476a18885` |
| `a6, a7` 384×16 | z, and v again | `9a713bc2db` bf16 + f32 | gated norm | `__place_33499fd2ac` |
| `a6, a9` 384×8 | z as even/odd pairs | `11ac1b62be`, `e3048e4596` bf16 | FP4 scale of the norm | `__place_904cb19908` |
| `a0 in 0..48` | a, b | `c8ded4e83a` | gating | `__place_287b0f2ef0` |

These are one value read through different index views. The cut pass is meant to fold such views into one seam
(`_cluster_reindexed_contractions`), but here it folds none of them.

## Why the views stay separate

Reproducer (CPU, about two minutes): lift the golden's program 6 at sm_120 and print the contraction seams that enter
`_cluster_reindexed_contractions`, with their row candidates and layouts
(`tmp-claude/decode-gap/in-proj/cluster_diag.py GOLDEN 6`, a scratch script outside the repo).

Expected: the flat qkv seam `PLACE@map.4/map.1/inner` (row `a50`, 10240) absorbs every q, k and v view, and one z
seam absorbs every z view. Observed: the flat seam absorbs only the two v pair views; z has no representative. Three
gaps:

1. A seam that value clustering already gave siblings is skipped as a member. The q/k twin (14 value siblings) and
   the v seam (2) are skipped. With the siblings stripped, the flat seam absorbs the q/k twin, v and both v pair views.
2. `_row_layout` recognizes a plain axis or a two-axis row-major pair only. The z channel's row
   `128 * a1 + 16 * a11 + a12` (head × block × lane, bijective onto 6144) is not recognized, so no seam can stand
   for z and its pair views (`128 * a28 + 16 * a31 + 2 * a32` and `+ 1`) join nothing.
3. A member joins only when every channel maps to the same representative. The z/v twin
   `PLACE@map.1/map.1/reduce.1/inner` needs z from a z seam and v from the flat qkv seam.

No existing knob avoids it. The cut arms are built from these seams: a pin or a composed route chooses which seams to
cut, never how they cluster. An uncut view is computed in place by its reader, which reads the weight again.

## Overlap with the gdn16 follow-up

Different gap, same family. gdn16's taps are not merged because the token pad guards every row (`(a1 < 16) ? … : 0`),
so `_channel_copies` and `_row_layout` see no plain row. The gdn1 rows have no guard. Gap 1 also hits gdn64: with
siblings stripped its flat seam `PLACE@map.4/map.1/inner` absorbs `PLACE@map.1/map.1/map.2/reduce.1/inner`.
Goldens with GDN kernels that a clustering change can re-key: Qwen3.8 NVFP4 on the RTX 5090, Qwen3.8 FP8 on the
V100, and the RTX 4090 hardware golden.
