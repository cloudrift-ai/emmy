# The golden-bench kernel cluster — what to try next

Written 2026-09-24 after the score-statistics pass (#898). Numbers are the committed goldens replayed unpinned at
deployable `-O3`, `EMMY_FAST_MATH=0`, 10 warmups / 100 iterations, eager and Inductor in the same process. Qwen3-0.6B
layer 0, sequence length 512.

## Where things stand

| target | H100 | Inductor | A100 | Inductor | V100 | Inductor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| input RMSNorm | 2.6 | 5 | 3.9 | 6 | 4.3 | 7 |
| value projection | 6.2 | 5 | 13.9 | 13 | 33.0 | 41 |
| query projection | 7.6 | 7 | 19.2 | 21 | 47.3 | 42 |
| key projection | 6.0 | 6 | 13.7 | 13 | 32.0 | 46 |
| **softmax x V** | **64.6** | 18 | **163.2** | 50 | **349.5** | 300 |
| **SDPA + o_proj + residual** | **46.8** | 21 | **67.0** | 63 | **281.3** | 321 |
| post-attn norm + gate/up | 18.3 | 22 | 49.5 | 52 | 105.5 | 130 |
| down_proj + residual | 15.8 | 11 | 36.3 | 32 | 110.0 | 67 |
| q/k norm + RoPE + score statistics | 29.5 | 29 | 49.2 | 64 | 229.3 | 285 |
| total | **197.4** | 124 | **415.9** | 314 | **1192.2** | 1239 |

The V100 cluster is ahead of Inductor. The H100 is 1.6x behind and the A100 1.3x, and on both cards the whole gap is
`softmax x V` plus `SDPA + o_proj + residual`. Those two are 72 µs of the H100's 73 µs of total excess over Inductor,
and on the A100 they are 117 µs against a total excess of 102 — the other seven targets net 15 µs AHEAD.

No target that reads its operands is more than 1.4x behind Inductor, and five of the seven are level or ahead on at
least two cards. `softmax x V`, which computes an operand inside the attention sweep, is 3.6x behind on the H100 and
3.3x on the A100, and `SDPA + o_proj + residual` is 2.2x behind on the H100. That is the one pattern left in this
corpus; `down_proj + residual` is the only other consistent loss, and on Volta it is a correctness gate (item 5)
rather than a schedule question.

## 1. The value projection's workspace layout — the biggest single lever

`softmax x V` computes the value projection inside the attention sweep. Cutting it out
(`PLACE@map.1/twist.2/inner=cut`) materializes it, but into a workspace shaped **(head, dim, key)**, which makes the
consumer's P·V product a transposed B whose K is contiguous. Nothing stages it: the consumer reads one fragment at a
time straight from gmem through `mma_load_b_gmem_trans` at a 2 KB stride. 150 µs on the A100, against 33 for the same
kernel reading an ordinary `transpose_2`.

- The dtype is not the binding constraint. A workspace at the atom dtype instead of the f32 carrier was measured:
  151.8 µs, no change. (`_workspace_dtypes`, the reducing-operand exception.)
- The axis order is. `cut_sites` orders a seam's axes as the kernel's free axes then the site's reduction scope, so a
  contraction's reduction axis lands LAST. For an A operand that is the canonical `(m, k)`; for the P·V B operand it
  is `(n, k)` where `(k, n)` is wanted. The q/k cones of the score target come out right by accident — their scope is
  the twist's key then the score's dim, so the score's K is last and the layout is the natural one.
- What to try: order a seam's workspace axes by the ROLE the consuming contraction reads it in. The consumer is
  already known at the seam (`store_dtype_consumers`, once widened past `edge.axis is None`), and
  `ContractionView.axis` / `right_axes` name the pair. Reordering `CutSite.axes` reorders `shape`, `index` and the
  read in lockstep, so the change is local; the risk is identity drift on every golden holding an operand-cone cut.
- Acceptance: the A100's `softmax x V` consumer takes a staged transport and lands near the 33 µs its
  `transpose_2`-reading twin measures. Expected cluster effect: H100 197 → ~165, A100 416 → ~305, V100 1192 →
  ~1090.

## 2. `SDPA + o_proj + residual` has the same shape one level up

Its o_proj reads the attention result as a COMPUTED A operand, so the same compute fill applies. The seam is
`PLACE@map.1/inner`, which materializes the attention output and leaves the o_proj an ordinary GEMM. Untried — the
H100's three-kernel set (10.5 + 16.5 + 19.7) has never been swept, and the 19.7 µs piece runs at 40 registers and 12%
occupancy on a 512×1024×2048 GEMM that should cost about 5. Cheap: no compiler change, one pinned sweep per card.

## 3. The V100's flash kernel spills

205 µs at 255 registers, 8 bytes of local, 12% occupancy. Every tile, warp split and staging depth reached in the
2026-09-24 sweep lands between 197 and 636 µs, so the shape is not the lever — the carrier is. The `mma_m8n8k4` atom
holds four fragments where `m16n8k16` holds one, and the twist carries three states per row on top. Worth reading the
SASS before sweeping again: the measuring loop in `plans/attention-optimization-memo.md` (compile to CUDA, `nvcc
--cubin`, `cuobjdump -sass`, count the largest backward-branch loop) needs no V100.

## 4. Re-record the 28 rows the rank fix cost

`promoted_sweep` now decides a cut piece's grid, so a piece that was one cooperative block per output element is one
block per row with a sweep. 28 recorded rows across three goldens spell schedules composed against the old grid and no
longer decode: 19 in `DeepSeek-V4-Flash-0731/v100_sm70.yaml`, 6 in `gemma-4-12B-it/rtx5090_sm120.yaml`, 3 in
`Qwen3.8-27B-FP8/v100_sm70.yaml`. The pieces are a different kernel now and carry no rows at all, so those kernels
price from the prior until someone re-records on a V100 and a 5090. The one piece of this class that was measured went
317 µs → 6.6, so re-recording is expected to find the new shape faster — expected, not measured.

## 5. Housekeeping

- The recipe's V100 row still traces and searches (`golden: ""`) instead of replaying
  `golden/qwen3-06b-s512_v100.golden.yaml`, which has existed since #889. One zip entry in `recipe.yaml`.
- `down_proj + residual` on the V100 fails its strict eager gate on 7 of 524288 elements at flat index 413453, the
  same row #889 recorded on three different schedules. Not schedule-dependent; it is f16 arithmetic on sm_70.
