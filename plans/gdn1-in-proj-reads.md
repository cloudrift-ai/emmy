# GDN input projections: one projection value, several cut workspaces

Status: shipped as PR #1171 (the rule, its fixes, every golden restamped; no tuning). Two follow-up PRs record and tune
the GDN recipes: Qwen3.8 NVFP4 on the RTX 5090 and Qwen3.8 FP8 on the V100. The sections through "Decision: direction 2"
describe the rule as shipped; "Prototype state" and "Re-record list" are history from the week of development; "State at
merge" says what each golden holds.

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

History (2026-10-15). The state at merge is in "State at merge" below.

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

History (2026-10-15). The state at merge is in "State at merge" below.

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

## State at merge (2026-10-16)

The rule is `_group_contractions` in `emmy/compiler/pipeline/passes/tile/_cut.py`: one workspace per contraction in the
contraction's own coordinates. Each load index component that reads only captured coordinates (the token and the output
column the cut fixes per workspace element) becomes a placeholder variable, and the body's canonical identity keys the
group. Tests in `tests/compiler/passes/test_cut_forks.py` cover five address shapes that abstraction must handle:
- a block scale beside its weight row: `n / 128` beside `n`; the head `h` beside the head-major row `128*h + d`; the q
  and gate halves of one fused projection, which spell the scale differently and must form one group over
  `[tokens, 12288]`; the same shape in the FP8 GDN in_proj_qkv producer;
- a flat-row address (`4*a1 + a2 + 8`) whose coordinates a nested cut reads back as digits of the placeholder;
- a nested cut read at the bare token beside a clamped address (`(t < 4) ? t : 0`, the Qwen3.5 GDN layer at width 4),
  which reads through the address's preimage;
- a producer that binds only the parent buffers its kernel reads, so an unread weight does not force its layout
  program-wide.

A survey over all 35 goldens (repository goldens, `tests/serving/goldens` fixtures, corpus goldens; workspace element
count against the union of its readers' addresses) finds no oversize workspace.

CPU parity against main: projection workspaces bitwise equal; downstream outputs within 1 to 2 f32 ulp because main
#1175 orders a folded sum's terms by how the body reads its buffers, which the flat workspace read changes (the conv1d
taps add in another order; the same kernel with main's order is bitwise equal).

Every repository golden was rebuilt from main's version with `emmy golden restamp` under this compiler, then the
measurements recorded on the #1171 branch (RTX 5090, V100 SXM2, V100 SXM3) were carried onto rows of the same exact
identity, bindings and schedule. Two restamp defects found during #1171 are fixed in #1194: a decision spelled at a seam
alias compared unequal to the primary spelling, and re-keyed siblings paired by emission order swapped. Per file
(unmeasured rows after the rebuild, main's count in parentheses):

| Golden | Unmeasured rows (main) | Decisions dropped | State |
| --- | ---: | ---: | --- |
| records/v100_sxm2_sm70 | 29 (24) | 4 (SDPA projection cut) | proposals await V100 SXM2 |
| records/a100_sm80 | 46 (29) | 2 (SDPA) | proposals await A100 |
| records/h100_sm90 | 50 (27) | 2 (SDPA) | proposals await H100 |
| records/rtx5090_sm120 | 21 (12) | 2 (SDPA) | proposals await RTX 5090 (follow-up 1) |
| records/rtx4090_sm89 | 58 (37) | 2 (SDPA) | proposals await RTX 4090 |
| Qwen3.8 NVFP4 rtx5090 | 137 (27) | 13 (GDN producers) | needs re-record (follow-up 1) |
| Qwen3.8 FP8 v100 | 25 (9) | 9 (gdn16 root cut, gdn64 12-seam arm, 3 gate-half k-splits) | needs re-record (follow-up 2) |
| gemma-4 rtx5090 | 116 (106) | 0 | proposals await RTX 5090 (tuning sweep) |
| DeepSeek-V4-Flash v100 | 72 (69) | 0 | proposals await V100 |
| Llama 3.1 8B, Llama LoRA experiment, Voxtral (V100 SXM3) | 16 (16), 20 (20), 9 (9) | 0 | refreshed |
| Qwen3.8 AWQ-INT4, EXL3 (V100) | 3 (1), 2 (0) | 0 | proposals await V100 |
| `tests/serving/goldens` sm120 | regenerated on the RTX 5090 | | current |
| `tests/serving/goldens` sm70 / sm80 / sm90 | restamped; rows demoted | 16 / 3 / 11 (GDN conv1d cuts) | regenerate on each card |

Dropped decisions, by cause:
- GDN producer shapes the rule changes on purpose: Qwen3.8 NVFP4 (gdn1, gdn16, gdn64 producers and their pieces) and
  Qwen3.8 FP8 (gdn16 root cut: 8 pieces instead of 9; gdn64 12-seam arm no longer offered; attention gate-half pieces
  have no own contraction, so no k-split fork).
- The SDPA projection piece of the hardware goldens: K and V projections share one fold when their layouts agree, so the
  piece offers two seams instead of three. The Q | K+V split is `PLACE@map.1/inner` on every card and needs a record
  run; the V100 SXM2 three-way Q | K | V split cannot be made any more. Accepted: follow-up 2 measures the two-way split
  against main's rows.

Strict serving: Qwen3.8 NVFP4 on the RTX 5090 deploys 0 of 9 twins under the strict audit (`emmy eval golden --golden
recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json --serving-config recipes/Qwen3.8-27B-NVFP4/serving-rtx5090.env`,
with and without fast math); every twin stops at its first cut fork, "no measured row spells a kernel-set arm". Main
deploys 9 of 9. Non-strict serving works and measures the prior's picks, not this rule. The kernel evidence: gdn1 333 ->
187 µs fast math, gdn16 5005 -> 169 µs, projection DRAM 400 -> 169 MB per layer, which predicts about 7 ms per decode
token.

The `tests/serving/goldens` fixtures: `serving_sm120.golden.json` regenerated on the RTX 5090 (`python -m
tests.serving.regen`, 209 kernels / 164 rows / 45 decisions); sm70, sm80, sm90 restamped, their GDN cut decisions
dropped and their rows demoted, awaiting a regeneration on each card. The regeneration needs the refit priors: the
priors fit on the goldens as restamped before the #1194 fixes priced a 26×14 register tile at near zero, and the
compile's greedy selection built a 6.5 MB kernel for the bf16 GDN post-layer program (57 minutes in nvcc).

The refit after the merge of main fits both priors on the final goldens. The prior reproduction test passes 168 of 169
cases; nothing carries a `prior-pending` tag. The red prior node is one slice of the H100 hardware golden: for the pool
`k_sum_1_reduce` at 512 tokens the schedule prior ranks the recorded schedule 30th of 59, one place outside the top
half. That pool's rows did not change; the fit lost the 23 H100 projection rows the restamp demoted and gained the V100
tune rows of the measurement freeze. The node turns green once a record run re-times the H100 golden's demoted rows and
follow-up 3 refits the priors. The `tests/serving` helper gives each of the lane's compiles a tune DB of its own: with
the session-wide DB, xdist workers importing different golden scopes (the set of golden files a compile reads evidence
from) deleted each other's rows mid-compile, and strict evidence refused forks the fixture decides.

## Follow-ups

1. **PR: Qwen3.8 NVFP4 RTX 5090 recording and tuning.** First settle why the strict audit names twin root kernels
   (`k_conv1d_linear_mean_reduce_987e6b` for gdn1, `e00f21` gdn16, `3133c6` gdn64,
   `k_linear_mean_reduce_150d01/7f4680/223cf6` for the attention pre-layer twins, `k_linear_reduce_15a4db/9a3f88/862829`
   for the post-layer twins) that no stored row names: a naming artifact of the error message, or a real re-key of the
   twins against the golden's traced programs. Then record the GDN forks: the gdn16 root `0e5142`, the gdn64 producers
   under `02d4dd` and `db4f79`, the gdn1 and `5f1430` children. gdn64 likely needs the same producer split recorded for
   the gdn16 root, whose routing row is in the NVFP4 golden. Measure the proposals the nine twins reach. Tune the
   producer pieces with and without fast math; the fast-math gdn16 piece `0e5142__place_b367ec15f4` was 159 µs against
   56 standard. Audit to 9 of 9, then one strict boot against main's 31.7 tok/s. Gotchas: `--record-greedy` implies
   strict evidence and stops at the fork it would record, so find a cut for a root kernel with no stored row first
   (`emmy compile --golden PATH --realization NAME --ir tile --passes dolfstp` under `EMMY_KNOBS`, the passes up to
   tile); `--realization` needs an existing row. Then refit the priors.
2. **PR: Qwen3.8 FP8 V100 recording and tuning.** Record gdn16 (`13e5d6`) and gdn64 (`353a47`) under the new producer
   shapes, measure the FP8 golden's proposals, record the SDPA Q | K+V split on the V100 SXM2 hardware golden and
   compare it with main's three-way rows, regenerate `serving_sm70.golden.json` on the card. The qkv weight is still
   read twice in the gdn64 arm main's golden takes (one piece computes the rms factor inline, the other reads it
   materialized: different input cones, so two groups).
3. A100, H100, RTX 4090: hardware goldens and serving fixtures are restamp-only; demoted rows await the cards. The
   H100 re-time (23 rows that keep their schedules) plus a refit also closes the red prior node above.
4. gemma-4 RTX 5090: re-keyed rows kept main's schedule untuned (tails up to 2 times slower on 16 µs kernels): the
   tuning sweep.
5. The qkv weight is read twice in gdn16 (the bounds check on padded tokens), main and branch alike.
6. Tooling: the evidence DB deletes every `golden:` row and imports the compiling scope's; two scopes compiling
   concurrently against one tune DB thrash (the `tests/serving` lane sidesteps it with its own DB; deploy workers with
   different golden scopes would not). The schedule prior extrapolates near-zero prices for register tiles beyond the
   measured range (`k_matmul_b86008` gets `f26x14`, a 2 MB source, on main too).
7. Open shapes of the rule. A union range computes rows no reader asked for when readers' ranges have gaps; no golden
   has such a gap since the q and gate halves of a fused projection form one group. A block scale read as a non-division
   function of the row still gets its own axis. The exact `/ B` check gives up above 2^20 points of a row's coordinates.
   A size guard comparing a group's space with its readers' address union would make the rule fall back to the old
   clustering on the next unhandled shape instead of building a product space.
