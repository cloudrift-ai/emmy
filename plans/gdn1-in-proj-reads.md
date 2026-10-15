# GDN input projections: one projection value, several cut workspaces

Status: Ivan chose fix direction 2 ("the contraction's own output is the workspace"). A prototype is on this branch,
merged with main 45e754bc; gdn1 parity passes on the CPU loop backend. Resume point at the end. Prose check pending.

Terms used below. A **seam** is a Fold edge where the cut pass may split a kernel. A **workspace** is the global
buffer a cut writes at a seam: the producer piece writes the value once, and readers load it. A **view** is one way
a reader addresses a projection's output columns: flat, per head, in blocks of 16, or as even/odd pairs.

## Observed (RTX 5090, main a951e6c0)

The width-1 GDN program (gdn1) reads 399 MB of DRAM per layer in its input projections (ncu, cold cache). Its BF16
weights are 168 MB: `in_proj_qkv` (104.9 MB) is read 2.6 times and `in_proj_z` (62.9 MB) twice. These projections
cost about 14.5 ms per decode token. Stock vLLM reads the same weights once in about 5 ms.

## Which decision produces the pieces

The golden's routing row on the fused GDN kernel `k_conv1d_linear_mean_reduce_970a92` takes the full-projection cut:
19 `PLACE@…=cut` keys, exactly the seams `full_projection_seams` returns on the fresh lowering. That arm cuts every
contraction seam, every row statistic computed ahead of an output sweep, and every branch that owns kernel outputs.
It picks among the seams after clustering, so it writes one workspace per seam, not one per weight.

The workspace producers fuse into one projection kernel (`…__place_b770740dbf`). A second routing row splits that
kernel's outputs apart (`PLACE@map.{1..5}/inner=cut`) into six serving kernels. Before any fix, the projection kernel
has one loop per workspace (simplified Loop IR; `x̂` is the normalized input row, `W` the qkv weight [5120, 10240]
with q in columns 0..2047, k in 2048..4095 and v in 4096..10239, `Z` the z weight [5120, 6144]):

```
for c in 0..10240:          ws_qkv[c]          = Σ_k x̂[k]·W[k, c]                  # f32 [10240]     conv state
for c in 0..2048:           ws_q[c/128, c%128] = Σ_k x̂[k]·W[k, c]                  # f32 [16,128]    conv, recurrence
                            ws_k[c/128, c%128] = Σ_k x̂[k]·W[k, c + 2048]           # f32 [16,128]
for c in 0..6144:           ws_v[c/128, c%128] = Σ_k x̂[k]·W[k, c + 4096]           # f32 [48,128]    recurrence
for b in 0..384, l in 0..16: ws_zb[b/8, b%8, l] = Σ_k x̂[k]·Z[k, 16b + l]           # bf16 [48,8,16]  gated norm
                            ws_vb[b/8, b%8, l] = Σ_k x̂[k]·W[k, 16b + l + 4096]    # f32 [48,8,16]   v again
for b in 0..384, p in 0..8:  ws_ze[b/8, b%8, p] = Σ_k x̂[k]·Z[k, 16b + 2p]          # bf16 [48,8,8]   z again
                            ws_zo[b/8, b%8, p] = Σ_k x̂[k]·Z[k, 16b + 2p + 1]      # bf16 [48,8,8]
for h in 0..48:             ws_a[h], ws_b[h]                                        # a, b
```

The expected shape has one workspace per weight. The readers address it through their own views:

```
for c in 0..10240:  ws_qkv[c] = Σ_k x̂[k]·W[k, c]        # W read once
for c in 0..6144:   ws_z[c]   = Σ_k x̂[k]·Z[k, c]        # Z read once
reads:  q = ws_qkv[(h/3)·128 + d]    k = ws_qkv[(h/3)·128 + d + 2048]    v = ws_qkv[128h + d + 4096]
        z block = ws_z[128h + 16b + l]    z even = ws_z[128h + 16b + 2p]    z odd = ws_z[128h + 16b + 2p + 1]
```

## Why the gdn1 views stay separate

The cut pass is meant to merge such views. It has three mechanisms, applied in this order on the seams of one
kernel:
- value clustering (`_cluster_value_seams`);
- channel copies (`_channel_copies`);
- reindexed clustering (`_cluster_reindexed_contractions`), which serves block and pair views from a flat-row
  workspace.

On gdn1, reindexed clustering merges only the two v pair views into the flat qkv seam `PLACE@map.4/map.1/inner`
(row `a50`). Three gaps stop the rest:

1. **A seam that value clustering already gave siblings is skipped as a member.**
   - The q/k seam has 14 siblings and the v seam has 2, so both are skipped.
   - With those siblings stripped (a diagnostic, not a fix), the flat seam absorbs q/k, v and both v pair views.
2. **`_row_layout` recognizes only a plain axis or a two-axis row-major pair.**
   - z's block view row `128*a1 + 16*a11 + a12` (head × block × lane, bijective onto 6144 columns) is not recognized.
   - So no seam stands for z, and z's pair views `128*a28 + 16*a31 + 2*a32` and `+1` join nothing.
3. **A multi-channel member joins only if every channel maps to the same representative.**
   - The z/v seam `PLACE@map.1/map.1/reduce.1/inner` needs z from a z seam and v from the flat qkv seam.

No existing knob avoids this. Pins and routing rows choose which seams to cut, never how views group into seams. A
view left uncut is computed in place by its reader, which reads the weight again.

Reproducer, CPU only, about two minutes:
1. Lift program 6 of `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` at sm_120 (`frontend` through
   `tile/lift`).
2. Call `cuttable_seams` on the fused GDN tile.
3. Print the row candidates and `_row_layout` of each contraction seam that enters
   `_cluster_reindexed_contractions`.

## Reach across the repository goldens

The survey covers all 20 golden files: 8 hardware goldens and 12 recipe goldens, 574 fused targets. For each one it
takes the kernel set that the recorded routing rows deploy and counts the distinct weight addresses those kernels'
Loop IR loads. "Read" is that count over the weight's size. "Extra" is the bytes above one read, per call.

### One value, several workspaces (this bug)

| Golden | Program | Fused kernel | Weight (size) | Workspaces | Read | Extra per call |
| --- | --- | --- | --- | --- | ---: | ---: |
| Qwen3.8 NVFP4, RTX 5090 | 6, gdn1 | `k_conv1d_linear_mean_reduce_970a92` | qkv (104.9 MB) | flat [10240]; q/k [16,128]×2; v [48,128]; v block [48,8,16] | 2.6× | 167.8 MB |
| Qwen3.8 NVFP4, RTX 5090 | 6, gdn1 | same | z (62.9 MB) | block [48,8,16]; pairs [48,8,8]×2 | 2.0× | 62.9 MB |
| Qwen3.8 NVFP4, RTX 5090 | 7, gdn16 | `k_linear_matmul_mean_reduce_db4f79` | z (62.9 MB) | block; pairs ×2 | 2.0× | 62.9 MB |
| Qwen3.8 NVFP4, RTX 5090 | 8, gdn64 | `k_linear_matmul_mean_reduce_02d4dd` | z (62.9 MB) | block; pairs ×2 | 2.0× | 62.9 MB |
| Qwen3.8 NVFP4, RTX 5090 | 8, gdn64 | `k_conv1d_linear_mean_reduce_5f1430` | qkv (104.9 MB) | flat [64,10240]; flat last 4 tokens [10240,4]; k only [16,64,128] | 2.2× | 125.8 MB |
| Qwen3.8 NVFP4, RTX 5090 | 7, gdn16 | `k_conv1d_linear_mean_reduce_0e5142` | qkv (104.9 MB) | flat ×2; the conv-tap piece reads q/k twice more under the pad guard | 4.0× | 314.6 MB |
| Qwen3.8 FP8, V100 | 1, gdn16 | `k_conv1d_linear_mean_reduce_13e5d6` | qkv FP8 (52.4 MB) | same shape as the NVFP4 gdn16 qkv row | 4.0× | 157.3 MB |
| Qwen3.8 FP8, V100 | 2, gdn64 | `k_conv1d_linear_mean_reduce_353a47` | qkv FP8 (52.4 MB) | flat ×2; k only | 2.2× | 62.9 MB |

- **gdn1** decodes one token per call, so the extra reads cost 230.7 MB per layer. Over 48 layers that is about
  11.1 GB per token, or about 6.6 ms at 1.69 TB/s. This matches the ncu counts above.
- **gdn16 and gdn64** are prefill pieces of 16 and 64 tokens, so the same bytes cost less per token there.

Which gap causes each duplicate:
- gdn1 qkv: gaps 1 and 3.
- gdn1 z, and gdn16 and gdn64 z: gap 2 (the three-axis block row). This corrects an earlier note that only gap 1
  reached gdn64.
- gdn64 qkv, the k-only workspace: gap 1.
- gdn16 qkv: the pad guard, the known follow-up in `plans/qwen38-5090-followups.md` ("recomputes the qkv
  projection once per conv tap").
- gdn64 qkv, the two flat workspaces: a fourth address shape (below).

**The fourth shape: a token subset.** In gdn64, the conv state store needs the projection only for tokens 60..63
(row `a14 + 60`). The conv taps need it for all 64 tokens. One flat seam computes [64, 10240] and another computes
[10240, 4] for the last four tokens. Neither grouping rule covers this case:
- value clustering needs equal axis extents (64 against 4);
- reindexed clustering abstracts the output row, not the token range.

### Related, not the same bug

- **gemma-4 12B, RTX 5090, prefill programs 22 (1024 tokens) and 24 (192 tokens).** Targets
  `k_linear_mean_reduce_7513ab` and `…76c8a5`.
  - The decision that cuts all six seams at once has unpriced pieces, so the evidence pick takes a staged chain of
    single cuts instead.
  - That chain cuts the k/v (or q) norm statistic, and its remainder computes the projection again for the kernel's
    own outputs.
  - Read 2.0×: +31.5 MB per call in program 22, +31.5 MB in program 24.
  - The views do cluster into one seam; the deployed chain just never cuts that seam.
- **Uncut targets with no measured row**, read 1.4–3.0× (a lower bound):
  - A100 records p77–79 (Qwen3.8 FP8 attention "pre": k_proj 2.0×, q_proj 1.5×);
  - RTX 4090 records p51 (o_proj 3.0×) and p58–59 (GDN with integer weights: qkv 1.4×, z 3.0×, out_proj 3.0×);
  - Qwen3.8 GPTQ and AWQ on the V100, p0 (z, out_proj 3.0×);
  - Qwen3.8 FP8 on the V100, p0 and p2.

  Their views already cluster into one seam. Once a recorded route cuts that seam, they read once.

### Kernels that read through several views and still read once

250 (target, weight) pairs read a weight through several views and still read it once. Four mechanisms account for
them:

- **Value clustering.**
  - gdn1's q/k seam `PLACE@map.1/map.1/reduce.2/inner.1/map.1/inner` reads q at `(a1/3)*128 + a14` and k at
    `+2048`.
  - It has 14 siblings. One is the q/k norm statistic `PLACE@map.1/map.1/reduce.2/inner.1/map.9/map.1/reduce.1/inner`,
    read at `(a1/3)*128 + a3` (`a14 → a3`).
  - Another is `PLACE@map.3/map.3/reduce.1/inner.1/map.1/inner` at `(a28/3)*128 + a38`.
  - All 15 copies share one f32 [16,128]×2 workspace.
- **Channel copies.**
  - The SDPA target `k_sdpa_linear_mean_reduce_0d9564` is in the A100, H100, RTX 4090, RTX 5090 and V100 SXM2
    records. Its seam `PLACE@map.1/inner.1/map.2/reduce.1/map.1/inner` holds the RoPE q projection three times.
  - Plain: `(a9/128)*128 + a11`. Rotate-half: `(a9/128)*128 + 64 + (a11 < 64 ? a11 : 0)` and
    `(a9/128)*128 + (a11 < 64 ? 0 : a11 - 64)`.
  - The two rotated channels map onto the plain one, so `linear_wt` is read 1.0×.
- **Reindexed clustering** (#1089, #1102).
  - gdn1's out_proj target `k_linear_mean_reduce_6fee7a`. Representative `PLACE@map.1/map.1/inner`, flat row `a2`.
    Its indexed siblings are the block statistic `PLACE@map.2/map.2/reduce.1/inner` at `16*a4 + a5`, the FP4 pairs
    `PLACE@map.3/map.2/inner` at `2*a7` and `2*a7 + 1`, and `PLACE@map.3/map.3/reduce.1/inner` at
    `(a7/8)*16 + a9`.
  - The gate/up target `k_linear_reduce_139b4a` does the same from the two-axis row `16*a4 + a5`.
- **The route cuts an enclosing value.**
  - RTX 5090 records `k_linear_mean_reduce_f4e8b4` (Ministral FP8 post, 4096 tokens).
  - The o_proj seam `PLACE@map.1/map.1/inner` (two value siblings) is not cut. The route instead cuts the cones of the
    residual sum, which contain the o_proj.
  - The piece `__place_bf8f8f803a` writes that sum once, f32 [4096,5120], and `p_o_proj_weight_bits` is read 1.0×.

Most other linears have a single view.

## Workarounds today

None is deliberate.
- **#1089 and #1102 are fixes of the same family, for the post programs.** #1089 added the computed-address grouping.
  #1102 extended it to row-major pairs; the gate/up corpus route's "five sites to three" is its result.
  - #1089 kept plain axis captures out of that grouping on purpose, after a wrong DeepSeek V100 re-key.
  - The skip of seams that already have value siblings (gap 1) came in with #1089. Its message does not discuss it.
- **#1133 tuned the gdn1 duplicates as if they were a schedule gap.** It moved the two duplicated z pieces (384×16 and
  384×8) to an MMA tile. The "Paired projections" note in `plans/qwen38-5090-followups.md` treats them the same way.
- **No route or pin in any golden avoids a duplicate on purpose.** Neither the gdn64 routes (#1082) nor the gemma-4
  staged chains do.

## Fix directions

**1. Patch the matcher.** Close the three gaps in `_cluster_reindexed_contractions`:
- a seam with value siblings may join, and its siblings follow it;
- row layouts of any number of row-major axes are recognized;
- a multi-channel seam may split its channels across representatives.

Behavior: gdn1's producer writes `ws_qkv` and `ws_z` as in the expected shape above. The gdn64 token-subset duplicate
would still need a separate patch, a fourth shape.

**2. The contraction's own output is the workspace.** Key each workspace by input × weight × reduction, and index it
by the weight's column (and token, where present). Every reader translates its own address into that index. One
workspace per projection would follow from the key itself, without matching view against view. That covers the
three gdn1 gaps, the z views, and the gdn64 token subset (a reader of tokens 60..63 reads rows 60..63). It replaces
more of the existing grouping and re-keys more goldens.

Goldens either direction can re-key: Qwen3.8 NVFP4 (RTX 5090) and Qwen3.8 FP8 (V100) for their measured GDN rows. The
RTX 4090 hardware golden and the GPTQ/AWQ V100 goldens hold GDN kernels with no measured row.

## Decision: direction 2

Chosen by Ivan on 2026-10-14. Deliverable: one rule in the cut pass, no condition that keeps existing goldens valid,
then restamp and re-record what moves.

Behavior, as prototyped (`_group_contractions`, `_contraction_reads`, `_contraction_producer` in
`emmy/compiler/pipeline/passes/tile/_cut.py`):
- Every channel of a pure contraction seam is lowered with its load index components that read only captured
  coordinates replaced by placeholders. The canonical identity of that body is the key: same input cone, weight and
  reduction.
- One workspace per key, laid out over the union of the placeholders' address ranges (interval analysis, or an
  exact walk of a small static domain for guarded indices). The producer is the contraction swept over that space;
  each channel reads the workspace at its own address minus the range's start.
- Contractions whose channels share a seam node are one decision (z and qkv in gdn1, gate and up).
- Seams the rule does not apply to keep the old clustering: twisted, observed or carried folds; an index that mixes a
  captured coordinate with the reduction (the GDN recurrence); a computed index over a symbolic dimension.

Example (gdn1, `fullproj`-style cut: full-projection arm on the fused kernel, pieces left whole):

```
for a0 in 0..48:    ws_a[a0], ws_b[a0]                          # bf16 [48] each
for a2 in 0..6144:  ws_z[a2]   = Σ_k x̂[k]·Z[k, a2]               # bf16 [6144], Z read once
for a4 in 0..10240: ws_qkv[a4] = Σ_k x̂[k]·W[k, a4]               # f32 [10240], W read once
readers: ws_qkv[128h + d] / [+2048] / [(h/3)·128 + d + 2048] / [128h + 16b + l + 4096] / [… + 2p + 4096 (+1)]
         ws_z[128h + 16b + l] / [128h + 16b + 2p] / [128h + 16b + 2p + 1]
```

gdn16 and gdn64 read qkv and z once each too: qkv is f32 [16,10240] and [64,10240]. The gdn16 pad-guarded tap reads
and the gdn64 state store (tokens 60..63) read that one workspace at their own addresses.

## Prototype state (2026-10-15)

Verified on the CPU loop backend, same random input, `origin/main` against this branch (FP4 encode kernels do not
render there and were skipped, so the comparison covers every buffer up to them):
- gdn1 (program 6): `copy` and `add_5` bitwise equal, and each old workspace equals the matching slice of the new
  ones (qkv, q/k, v, z block, z even and odd), re-run after the merge with main 45e754bc.
- gdn16 and gdn64 (programs 7, 8) state outputs `add_70` and `pad`: bitwise equal (before the merge).
- RTX 5090 records SDPA program 41 (RoPE q and k, previously `_channel_copies`): `add_7` bitwise equal (before the
  merge).

The goldens this moves are in "Re-record list" below.

Not verified yet: `tests/compiler/passes/test_cut_forks.py` (about a dozen failures, not triaged), the full suite,
the prior reproduction gate on restamped goldens, lint, any GPU run.

## Re-record list

Survey of 2026-10-15: every golden restamped under this branch (d84f29e1, main 45e754bc merged in) and, as a control,
under main 45e754bc. The restamp is `restamp()` from `emmy/compiler/pipeline/search/golden/restamp.py`, the rewrite
`emmy golden restamp` writes. What a restamp does to a golden:
- A kernel is **re-keyed** when the fresh lowering gives it another body. Its rows keep their schedule and lose their
  microseconds: they become proposals, which are no evidence.
- A **decision** (routing row) whose arm the fresh parent no longer spells is dropped. Every piece under it is dropped
  with its rows.

"Measured rows lost" counts both kinds: demoted rows plus dropped rows that had a measurement.

| Card | Golden | Programs (target kernel) | Re-keyed | Measured rows lost | Decisions dropped |
| --- | --- | --- | ---: | ---: | ---: |
| RTX 5090 | `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` | p0–p5 attention, p6 gdn1, p7 gdn16, p8 gdn64 | 12 | 300 of 474 | 30 |
| RTX 5090 | `recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` | p2, p3, p20–p39 (`k_linear_mean_reduce_*`) | 143 | 197 of 577 | 1 |
| RTX 5090 | `emmy/.../golden/records/rtx5090_sm120.json` | p41 SDPA `0d9564`, p42 SDPA `f91a74`, p79–p80 gate/up | 12 | 24 of 218 | 8 |
| A100 40GB | `emmy/.../golden/records/a100_sm80.json` | p0 SDPA `0d9564`, p1 SDPA `f91a74`, p54, p74–p76, p78 | 23 | 43 of 166 | 8 |
| H100 80GB | `emmy/.../golden/records/h100_sm90.json` | p37 SDPA `0d9564`, p38 SDPA `f91a74`, p56 | 18 | 28 of 226 | 6 |
| RTX 4090 | `emmy/.../golden/records/rtx4090_sm89.json` | p39 SDPA `0d9564`, p40 SDPA `f91a74`, p42 | 21 | 31 of 227 | 6 |
| V100 SXM2 16GB | `emmy/.../golden/records/v100_sxm2_sm70.json` | p0 SDPA `0d9564`, p1 SDPA `f91a74`, p38 | 18 | 34 of 151 | 7 |
| V100 SXM2 16GB | `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` | p1 gdn16, p2 gdn64, p3–p8 | 16 | 48 of 62 | 13 |
| V100 SXM3 32GB | `recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json` | p5, p9–p11 | 8 | 7 of 18 | 4 |
| V100 SXM3 32GB | `recipes/Meta-Llama-3.1-8B-Instruct/golden/v100_sxm3_sm70.json` | p1 `k_linear_mean_reduce_6615bc` | 6 | 7 of 32 | 0 |
| V100 SXM3 32GB | `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` | p0–p5, p8 | 6 | 6 of 39 | 0 |
| V100 SXM3 32GB | `experiments/Meta-Llama-3.1-8B-Instruct/lora_v100_sxm3_32gb/golden/v100_sxm3_sm70_lora8.json` | — | 15 | 12 | 0 |

Per-program counts for the Qwen3.8 NVFP4 golden: p6 loses 57 rows and 3 decisions, p7 73 rows and 8 decisions, p8 95
rows and 13 decisions; p0–p5 lose 8–22 rows each.

Main itself is already stale on five recipe goldens that are not maintained: Ministral 3 14B and OLMoE (RTX 5090),
Qwen3.8 AWQ and EXL3 (V100), and Voxtral (V100; 13 kernels re-keyed, 7 rows demoted). The branch changes the first four
exactly as main does, so they are not on the list. The Voxtral row above counts only what the branch adds.

### A golden that is restamped but not re-recorded

- **A strict-evidence compile fails.** `_require_evidence` in `search/policy/greedy.py` raises `EvidenceError` "when a
  fork must be decided and no measured row — tune DB or golden — vouches for any of its candidates". `import_rows`
  files no proposal: "an unmeasured row is a proposal, not evidence". Checked on the laptop with the GPU hidden and an
  empty tune DB, `emmy compile --golden <restamped file> --program N --target sm_120 --strict-evidence`:
  - Qwen3.8 NVFP4 p3 and p6, and gemma-4 p22, raise `EvidenceError: … has no measured evidence for its 030_cut fork
    (no measured row spells a kernel-set arm)`;
  - main compiles each of them from its own golden;
  - the golden left as it is (not restamped) fails the same way on the branch.

  The V100 probes (Llama p1, Qwen3.8 FP8 p1) and RTX 5090 records p41 fail on main too on the laptop, so they say
  nothing. Without `--strict-evidence` nothing fails: the priors decide the forks that lost their rows.
- **The freshness test passes after the restamp, and fails before it.** `test_a_repository_golden_is_the_fresh_lowering`
  in `tests/compiler/pipeline/search/test_golden.py` asserts `fresh == document` after a restamp. It covers the
  hardware goldens and the maintained recipes, so every row of the table except Voxtral and the LoRA file.
  `test_lora_serving_golden_replays_nested_cuts` makes the same check on the LoRA file. A restamped file is a fixed
  point: restamping the NVFP4, gemma-4 and A100 outputs again changes nothing.
- **The prior reproduction gate (AGENTS.md step 22) fails on one golden.** The gate imports only measured rows, so
  lost rows leave its pools. On the restamped goldens it gave 152 passed, 2 failed, 2 skipped. Both failures are
  Qwen3.8 NVFP4 placement slices: 9 of 11 and 14 of 16 reproduced, on the gdn64 kernels
  `k_conv1d_linear_mean_reduce_5f1430` and `k_linear_mean_reduce_9a501e`. Step 22 says: "A PR that changes a hardware
  golden refits both priors … when it does not [pass], either refit or tag the recipe `prior-pending`". So the five
  hardware goldens need a refit whatever the gate says, and the NVFP4 recipe needs a refit or the tag.
- **Recipe tags today.** Qwen3.8 NVFP4: maintained, lifecycle-locked, compiler-qualified. gemma-4 and Llama 3.1:
  maintained, lifecycle-locked. Qwen3.8 FP8 and DeepSeek: maintained. Voxtral: best-effort. None is `prior-pending`.

## Resume here

1. Done: the restamp survey on the merged branch ("Re-record list" above).
2. Triage `test_cut_forks.py`. Expected kinds: tests that assert value or reindexed clustering of contraction seams
   (now the contraction groups), and pinned routes whose keys changed.
3. Decide what to delete. `_channel_copies` and `_cluster_reindexed_contractions` still run for the seams the rule
   does not take. Check whether any golden still reaches them; if none does, remove them.
4. Open shapes:
   - A twin node shared by two contractions is one decision, so a pin cannot cut z without qkv.
   - A union range computes rows no reader asked for when readers' ranges have gaps.
   - A nested cut inside a channel whose workspace is laid out in that channel's coordinates needs a channel that
     covers those placeholders (`_contraction_producer` raises otherwise; no golden hits it now).
5. Restamp the goldens, commit, and hand the re-record list to the box run.
