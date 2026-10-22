# gdn1: the GDN decode recurrence as one kernel

Status: design, 2026-10-22. Nothing below is implemented on this branch yet except where a measurement says
"prototype". Branch `perf/gdn1-fused-recurrence`, on main 250f983f (#1226). Card: RTX 5090, fast-math lane.

## Goal and deliverables

In the Qwen3.8-27B-NVFP4 decode program gdn1, three kernels read the 3 MB recurrence state per GDN layer. After this
work one kernel reads it, computes the new state and the recurrence output together, and runs in about 3 µs instead
of 9.85 µs.

Deliverables:

1. **Load pairing (compiler rule, every kernel).** When a kernel forms, an output region whose store sweeps a
   coordinate that another region keeps on the grid gets those two coordinates swapped, if that makes the two regions
   read a shared input at the same element. Section "Load pairing".
2. **The join arm (cut pass).** A new arm of the placement fork: the piece that produces a cut workspace value also
   owns an output region of the same kernel. Offered when the two share a cut value (a reduce both compute). Section
   "The join arm".
3. **A pin and route spelling for that arm.** Ivan picks one of the options in section "Spelling".
4. **Goldens.** gdn1 re-recorded and tuned on the RTX 5090 with the joined route. Every other repository golden
   per the cost table (section "Cost").
5. **Corpus cases.** The isolated recurrence in both state layouts, and a gdn1-shaped join case.

Not in scope: the in-place state read (follow-ups), the rewrite `o = g·(q·S) + u·(q·k)` (changes rounding).

## Today (main 250f983f)

The recorded gdn1 route cuts the fused kernel `k_conv1d_linear_mean_reduce_970a92` at 11 seams. Three of its pieces
read `state`:

| piece | seam (`PLACE@…`) | computes | tuned µs |
|---|---|---|---:|
| `60d3d82a9e` | `map.1/map.1/reduce.4/map.1/reduce.1/inner.2/map.10/map.2/inner` | `r = Σ_k g·S[k,v]·k̂[k]` into a workspace | 2.43 |
| `72ae56ea07` | `map.2/map` (output-owning) | new state `S' = g·S + k̂·β(v − r)`, stores `add_5` | 3.38 |
| `90adff3621` | `map.1/map.1/reduce.4/map.1/reduce.1/inner` | `o = Σ_k q̂[k]·S'[k,v]`, the shared output (#1226) | 4.04 |

Sum 9.85 µs per GDN layer, 48 layers: 0.47 ms per token. Stock vLLM's fused recurrent kernel does the same work in
4.19 µs at this shape. Decode today: 57.2 tok/s (#1226); stock vLLM 63.2.

## Load pairing

### The problem

Loop fusion merges loops of equal extent position by position. In the recurrence, `o`'s nest is `(h, v)` and the
state store's nest is `(h, k, v)` (the state layout is `[H, K, V]`). Fusion merges `o`'s `v` loop with the store's
`k` loop. The lift then gives the kernel the grid `(h, v-of-o = k-of-store)` and the store sweeps the real `v`. Each
cell of that sweep folds `r` again for its own column: 128 times the work.

Observed, isolated recurrence (`rec_code.py`: inputs `S[48,128,128]`, `q̂, k̂, v, g, β`), main 250f983f:

```
$ EMMY_KNOBS=PLACE=fuse EMMY_FAST_MATH=1 emmy compile -c "$(cat rec_code.py)" --target sm_120 --ir tile --passes dolfstp
place  free=(a0, a1)                                     # a1 is v for o, k for the store
├─ operand[acc1]: Fold[a2] contraction … load s[a0, a2, a1]          # o reads S[h, ·, a1]
├─ operand[v14]: Fold free
│  ├─ operand[acc2]: Fold[a5] contraction … load s[a0, a5, a4]        # r for column a4, inside the sweep
│  └─ lift … load s[a0, a1, a4]
outputs
├─ sum_2[a0, a1] = acc1
└─ sweep(a4) add[a0, a1, a4] = v14
```

Untuned 66.9 µs, one kernel. With the state laid out `[H, V, K]` the same program gets the right shape and tunes to
2.1 µs.

### The rule

In a kernel whose root has independent output regions (the ownership partition the output-owning cut already
uses): take a region whose stores sweep a coordinate `w`, and a grid coordinate `g` that region and another region
both read, with equal extents. Swap `w` and `g` inside that region (its term and its stores) when the swap makes more
of its loads of a shared input read the same element as another region's loads, and breaks no read that already
coincides. Among several such swaps take the one with the most coinciding reads. Loads compare coordinate by
coordinate; reduce coordinates match any equal-extent reduce coordinate on the other side. After a swap the kernel is
lowered and lifted again, so the grid, hoisting and sweep come out of the same lift every kernel takes.

The rule runs in `lift_kernel`, so it applies to every kernel a program forms: fused kernels, cut pieces and fused
producers. It decides nothing: no fork, no knob. It extends `_align_owned_sweeps`, which already aligns two regions'
equal sweeps through equal load indices.

Examples:

| kernel | regions | result |
|---|---|---|
| isolated recurrence, `[H, K, V]` | `o[h,v]` on the grid; store sweeps `v`, reads `S[h, k, v]` | swaps the store's `k` and `v`: `o` and the store read `S[h, ·, v]` in one cell |
| isolated recurrence, `[H, V, K]` | both already read `S[h, v, ·]` | no change |
| recurrence with K=128, V=64 | fusion paired no loops (extents differ); grid is `h` only, both stores sweep | no change (no equal-extent pair); a separate gap, see follow-ups |
| Ministral FP8 prefill QKV `k_linear_mean_reduce_fd000c` (4096 tokens) | q region and the k/v region read `hidden[t, ·]` in one cell; q column extent = token extent = 4096 | no change: swapping q's column with the token breaks the shared `hidden` read |
| gdn1 joint piece (after the join arm) | `o` workspace on grid `(h, v)`; `add_5` region reads `state[h, k, v]` | swaps the store's `k` and `v` (expected; the piece does not exist before the join arm) |

The first version of the prototype accepted any swap that made one pair of reads coincide. It fired on the Ministral
QKV kernel and swapped the token with the q column, because both are 4096 long. The "breaks no coinciding read"
condition is what keeps that kernel unchanged.

Observed with the prototype (isolated recurrence, `[H, K, V]`, RTX 5090):

```
place  free=(a0, _um, a1)                                # a1 = v for both regions
├─ operand[acc1]: Fold[a2] contraction … load s[a0, a2, a1]
├─ operand[v10]: Fold free
│  ├─ operand[in1, v3]: … Fold[a2] contraction … load s[a0, a2, a1]  # r once per (h, v)
│  └─ lift … load s[a0, a4, a1]
outputs
├─ sum_2[a0, a1] = acc1
└─ sweep(a4) add[a0, a4, a1] = v10                       # the store sweeps k
```

| command | main | prototype |
|---|---|---|
| `emmy run -c "$(cat rec_code.py)" --bench` (`PLACE=fuse`) | 66.9 µs | 15.3 µs |
| same, `--tune 48`, best row | not tuned | 3.0 µs (`WORK=t8x16`, both folds `coop`), grid 384 × 128 |

The 3.0 µs against 2.1 µs for `[H, V, K]` is the strided read: the cooperative `k` lanes read the state with a 512-byte
stride. Stock vLLM runs 4.19 µs.

## The join arm

### What it adds

Today the cut pass has workspace cuts (a piece computes a seam's value into a workspace) and output-owning cuts (a
piece writes some of the kernel's own outputs). Sibling workspace producers with the same workspace dependencies fuse
back into one piece without a decision. An output-owning piece and a workspace producer never do. So the state store
(`add_5`, an output region) and `o` (a workspace for the norm and quantize pieces) always land in different kernels.

The join arm is one more arm of the placement fork, offered per pair (workspace seam S, output-owning seam R) of one
kernel when:

- S's cone and R's cone share a cut value (a seam both contain; in gdn1, `r`);
- neither reads the other's result.

The arm cuts S and R as today, then splices the two pieces through the Loop IR splicer and lifts them as one piece,
the way `_fuse_sibling_producers` joins sibling producers. Load pairing then gives that piece its grid. The piece
writes S's workspace and R's outputs. Like every cut it is a kernel-set fork: priced by the sum of its pieces'
measured rows, else by the placement prior. The plain cuts of S and R stay arms beside it.

It composes with the other seams of a route as any arm does: a composed route lists it beside its other keys. For
gdn1 the route to record is today's 11 keys with two changes: the `o` key and the `add_5` key become one joined key,
and the `r` key is dropped, so `r` is computed inside the joint piece (twice in the term, once in the kernel, as the
isolated program shows). Keeping the `r` cut beside the join is also an arm: then two kernels read the state. The
record measures both.

### Expected behaviour

| command | today | after |
|---|---|---|
| `emmy compile --golden G --realization gdn1-…970a92.m1.fm --pin-route --ir kernel`, kernels reading `state` | 3 | 1, storing `add_5` and the `o` workspace |
| same, number of kernels in the first half of gdn1 | 12 | 10 |
| `emmy run --golden G --realization gdn1-…970a92.m1.fm --bench --strict --strict-evidence`, the three state pieces | 9.85 µs | about 3–4 µs (one piece) |
| strict serving boot, decode | 57.2 tok/s | about 58.2 tok/s (0.3 ms per token less) |
| unpinned compile of gdn1's fused kernel: arms at its placement fork | fuse, 32 single cuts, full projection, recorded routes | the same plus one join arm per qualifying pair |

The joint piece, expected (names illustrative):

```
place  free=(a0, a1, _um, a2)                           # (head group, head in group, v)
Fold free
├─ operand[o]: Fold[a3] contraction  q̂[k] · (g·S[h, k, v] + k̂[k]·u)   # u = β·(v − Σ_k g·S[h,k,v]·k̂[k])
├─ operand[S']: Fold free  g·S[h, k′, v] + k̂[k′]·u
outputs
├─ copy__place_<token>_0[(3·a0 + a1), a2] = o
└─ sweep(k′) add_5[0, (3·a0 + a1), k′, a2] = S'
```

Risk inside it: `k̂` and `q̂` are computed cones (conv, SiLU, L2 norm) in gdn1, recomputed per `(h, v, k)` as every
piece does today. The isolated program read them from memory. If the joint piece measures far above 3 µs, cutting
`k̂`/`q̂` into a small per-head producer is an existing arm, and the evidence decides.

## Spelling

A routing row and a pin name a cut by the `PLACE` keys of its seams. A join needs one more fact: which two seams share
a piece. Options:

1. **Joined key.** `PLACE@map.1/map.1/reduce.4/map.1/reduce.1/inner+map.2/map=cut`. One key names both sites, joined
   by `+`. `=fuse` on it refuses that join. Values stay `cut` and `fuse`. The piece token is already the digest of the
   `+`-joined seam spellings (that is how `_fuse_sibling_producers` names its piece today), so the piece name follows
   from the key. Cost: the key grammar and `resolve` learn `+`, and the restamp re-spells a joined key half by half.
2. **A value naming the partner.** `PLACE@map.2/map=join:map.1/map.1/reduce.4/map.1/reduce.1/inner`, with the seam's
   own key `=cut` beside it. The key grammar is unchanged. Cost: a site path inside a value; the restamp and every
   reader of `PLACE` values learn a third form, and the decision is asymmetric (which side carries it is a choice).
3. **A bare third value.** `PLACE@map.1/map.1/reduce.4/map.1/reduce.1/inner=join`: cut this seam and give its piece
   every output region that shares a cut value with it. Shortest, no grammar change. Cost: the partner set is derived
   by a rule, not recorded. A later change to that rule changes what a recorded route means while its spelling stays
   the same, and the restamp cannot see it.

Recommendation: 1. One key is one decision, the values keep their meaning, the piece name already works this way, and
a recorded route says exactly which regions share a piece.

## Cost

Measured with the load-pairing prototype on the box: `emmy golden check` over every repository golden (GPU-free). The
join arm adds an arm and changes no recorded route, so on its own it re-keys nothing; gdn1 moves only because its new
route is recorded.

Command, per file: `emmy golden check PATH` with the rule in `lift_kernel` (prototype v2, box, 2026-10-22). The
prototype logs every kernel the rule re-forms; over all 21 files it re-formed none.

| golden | kernels | load pairing: re-keyed / dropped / decisions dropped | join arm | card work |
|---|---:|---|---|---|
| `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` | 211 | 0 / 0 / 0 | gdn1 gets the joined route | RTX 5090 (the box): record gdn1's joined route, tune the joint piece, strict audit and boot |
| `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` | 192 | 0 / 0 / 0 | gdn1 keeps its recorded route | none required; the joined route there needs a V100 |
| `recipes/Qwen3.6-35B-A3B-FP8/golden/rtx5090_sm120.json` | 303 | 0 / 0 / 0 | gdn1 keeps its recorded route | none required; optional on the RTX 5090 |
| `recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` | 292 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json` | 115 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Meta-Llama-3.1-8B-Instruct/golden/v100_sxm3_sm70.json` | 48 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Ministral-3-14B-Instruct-2512/golden/rtx5090_sm120.json` | 47 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Qwen3.5-122B-A10B/golden/v100_sm70.json` | 46 | 0 / 0 / 0 | GDN projections only, no recurrence | none |
| `recipes/Voxtral-Mini-3B-2507/golden/v100_sxm3_sm70.json` | 45 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/OLMoE-1B-7B-0125-Instruct/golden/rtx5090_sm120.json` | 36 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Qwen3.8-27B-GPTQ-Int4/golden/v100_sm70.json` | 9 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Qwen3.8-27B-AWQ-INT4/golden/v100_sm70.json` | 6 | 0 / 0 / 0 | no GDN recurrence | none |
| `recipes/Qwen3.8-27B-EXL3/golden/v100_sm70.json` | 4 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `rtx5090_sm120.json` | 193 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `rtx4090_sm89.json` | 187 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `h100_sm90.json` | 179 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `a100_sm80.json` | 165 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `v100_sxm2_sm70.json` | 148 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `v100_sm70.json` | 64 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `rtxpro6000_sm120.json` | 10 | 0 / 0 / 0 | no GDN recurrence | none |
| hardware `rtx4080_sm89.json` | 8 | 0 / 0 / 0 | no GDN recurrence | none |

So neither change needs a restamp or a re-record to keep the goldens current. The only card work is the gdn1
re-record that delivers the speedup, on the RTX 5090. The rule fires where fusion has paired two regions' loops by
position wrongly; no stored kernel has that shape today. The first prototype (any swap that makes one read pair
coincide) dropped the Ministral QKV decision on `h100_sm90.json` (one routing decision, its 4 piece rows and the
target's whole-target latency), which is why the rule refuses a swap that breaks a coinciding read.

The check lowers the kernels the goldens store. A program compiled without a recorded route can form other kernels,
and the rule applies to those too; that is the intent.

## Correctness evidence plan

- Load pairing changes what a kernel computes per thread, not the values: parity on the isolated recurrence against
  eager PyTorch (`emmy run -c … --bench`, done for the prototype), and on every kernel the rule re-keys, its new body
  against its old one.
- gdn1: the joined route against today's route (`--ab` with the old route's pins). These are different kernel sets,
  so the comparison is not a run against itself. #1226 found earlier "strict parity passes" that compared a run with
  itself, because `EMMY_KNOBS` pinned the greedy reference too; the reference here comes from a golden row or an
  `--ab` row, never from the same pins.
- Per tuned piece: the tuned row against its untuned row, as in #1226.

## Risks

1. **The joint piece and the conv cones.** See above. Unmeasured until the arm exists.
2. **Arm count.** Pairs of a workspace seam and an output region that share a cut value: gdn1's fused kernel offers
   26 workspace seams, 2 contraction seams and 4 output regions. The shared-value condition should leave a handful.
   If it leaves many, the placement prior ranks them, but the unpinned search gets slower.
3. **Composed routes.** The deployed gdn1 route is one composed cut. The joined key must compose with it, restamp,
   and `--pin-route` must replay it, before a strict serving boot accepts it.
4. **`[H, K, V]` stride.** 3.0 µs isolated against 2.1 µs for `[H, V, K]`. Acceptable; a layout fork for the state
   is a later option.

## Follow-ups

- **In-place state read.** With one kernel the only reader of `state`, read the old state from the cache block and
  drop the gather (`index_select`): about 0.22 ms per token (Oct 20 profile, state and history together). Needs its
  own rule: each CTA reads exactly the cells it writes, all reads before the first write, no cross-CTA split of `k`.
  History stays gathered, because the conv shift reads neighbouring cells.
- **K ≠ V.** With unequal extents fusion pairs no loops, and the kernel's grid is the head only. Aligning a region's
  sweep with a coordinate another region's sweep projects onto is a different rule. Qwen3.8 does not need it.
- **gdn16/gdn64** compute the recurrence in the chunked form, with other contractions and other pieces. Both changes
  apply to every kernel, but nothing here is designed or measured for those programs.
