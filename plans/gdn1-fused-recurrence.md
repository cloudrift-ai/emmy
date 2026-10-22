# gdn1: the GDN decode recurrence as one kernel

Status: implemented on this branch, 2026-10-23. Branch `perf/gdn1-fused-recurrence`, on main 13eb76bd (#1223). Card:
RTX 5090, fast-math lane. Section "Implementation results" has the measured kernels, the golden and the serving boot.

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
3. **A pin and route spelling for that arm:** the joined key (section "Spelling").
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
uses): a region's outer coordinates (the grid coordinates it reads and the coordinates its stores sweep) are
re-spelled as the expressions another region reads a shared input at, when that makes more of the two regions' reads
of shared inputs name the same element and breaks no read that already does. Concretely, for one read in each region of
the same input, every position where the first reads one of its own outer coordinates and the second reads an
expression over its own outer coordinates covering the same range maps that coordinate to that expression; reduce
coordinates match any reduce coordinate of equal extent. An outer coordinate the mapping leaves alone keeps its name,
or takes a name the mapping freed when the mapped coordinates now use it. Among all candidate mappings the one with the
most coinciding reads wins. The region's stores sweep every coordinate it now reads that the grid does not bind. After a
re-spelling the kernel is lowered and lifted again, so the grid, hoisting and sweep come out of the same lift every
kernel takes.

The rule runs in `lift_kernel`, so it applies to every kernel a program forms: fused kernels, cut pieces and fused
producers. It decides nothing: no fork, no knob. It generalizes `_align_owned_sweeps`, which aligns two regions' equal
sweeps through equal load indices.

Examples:

| kernel | regions | result |
|---|---|---|
| isolated recurrence, `[H, K, V]` | `o[h,v]` on the grid; store sweeps `v`, reads `S[h, k, v]` | store re-spelled `k ↔ v`: `o` and the store read `S[h, ·, v]` in one cell |
| isolated recurrence, `[H, V, K]` | both already read `S[h, v, ·]` | no change |
| recurrence with K=128, V=64 | fusion paired no loops (extents differ); grid is `h` only, both stores sweep | store's `v` re-spelled as `o`'s `v`: grid `(h, v)`, store sweeps `k` |
| gdn1 joint piece (join arm) | `o` over `(g, j, v)` with head `3g + j`; state region over `(h, k, v)` | state region's `h` becomes `3g + j`, its `v` becomes `o`'s: grid `(g, j, v)`, store sweeps `k` |
| Gemma 4 QKV projection, fused (`k_linear_mean_reduce_93c57d`) | q, k, v regions each sweep their own token coordinate | the token coordinates get one name, so the regions share the token loop |
| Ministral FP8 prefill QKV `k_linear_mean_reduce_fd000c` | regions already read `hidden[t, ·]` alike; q column and token both 4096 long | no change: re-spelling q's column as the token breaks the shared `hidden` read |

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

Decided (Ivan, 2026-10-22): the joined key. `PLACE@map.1/map.1/reduce.4/map.1/reduce.1/inner+map.2/map=cut` names
both sites, joined by `+`, and cuts them into one piece. `=fuse` on it refuses that join. Values stay `cut` and
`fuse`. The piece token is the digest of the `+`-joined seam spellings, which is how `_fuse_sibling_producers` already
names its piece. The key grammar and `resolve` learn `+`, and the restamp re-spells a joined key half by half.

Rejected:

- A value naming the partner (`PLACE@map.2/map=join:<seam path>`): a site path inside a value, and an asymmetric
  decision.
- A bare third value (`PLACE@<seam>=join`): the partner is derived by a rule, so a recorded route's meaning can drift
  when that rule changes.

## Cost

The first prototype of the rule only swapped a swept coordinate with a grid coordinate. With it every repository golden
stayed current. That rule does not form the gdn1 joint piece: the `o` piece and the state piece share no loop at all
(head split `(16, 3)` against `48`), so the rule had to re-spell a region's coordinates as expressions of another's.
The general rule moves more kernels. `emmy golden check` per file, then `emmy golden restamp` on a copy (box,
2026-10-22):

| golden | kernels re-keyed / dropped | decisions dropped | rows demoted / dropped | card work |
|---|---|---|---|---|
| `recipes/gemma-4-12B-it/golden/rtx5090_sm120.json` | 59 / 1 of 292 | 0 | 96 / 0 | RTX 5090: measure 96 rows |
| `recipes/Qwen3.6-35B-A3B-FP8/golden/rtx5090_sm120.json` | 14 / 0 of 303 | 20 | 2 / 19 | RTX 5090: re-record the dropped routes |
| `recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json` | 0 / 0 of 211 | 15 (gdn16, gdn64 routes) | 0 / 97 | RTX 5090: re-record and tune gdn16, gdn64; gdn1 joined route |
| `recipes/Qwen3.8-27B-FP8/golden/v100_sm70.json` | 34 / 0 of 192 | 16 | 1 / 15 | V100: re-record the dropped GDN routes |
| `recipes/DeepSeek-V4-Flash-0731/golden/v100_sm70.json` | 2 / 0 of 115 | 0 | 0 / 0 | none (restamp) |
| hardware `a100_sm80.json` | 0 / 0 of 165 | 0 (1 seam re-spelled) | 0 / 0 | none (restamp) |
| the 15 other goldens | current | | | none |

What moves, by kind:

- Gemma 4 and Qwen3.6 fused QKV projections: the q, k and v regions each swept their own token coordinate (no
  shared grid). Re-spelled, they share one token coordinate.
- Qwen3.8 gdn16/gdn64 conv parents: the conv output region's sweep is re-spelled onto the history-copy region's, which
  reads the same channels. The cut routes recorded on the old spelling do not replay.
- gdn1's parent: the FP4 code and FP4 scale regions share their block index; the state region swaps `k` and `v`. The
  recorded route re-spells and replays.

A narrower variant (count only reads inside a reduce) re-keys the same 59 Gemma kernels, so it saves nothing.

## Implementation results (2026-10-23)

What runs: the rule in `lift_kernel`, the join arm, the joined key in pins, recorded routes and composed routes, and
the inline copies of a cut value (`_cut._inline_copies`). Three corpus cases: `fused/gdn-recurrence-state-kv.json` (the
one-kernel recurrence with cooperative rows on both folds), `fused/gdn-recurrence-join.json` (the joined key on a
recurrence whose `o` feeds a norm) and `fused/gdn-recurrence-join-shared-key.json` (the joined key with the normalized
key cut once and read in all three places).

### The normalized key, computed once

The joint piece needs `k̂` (four conv taps, SiLU, the norm scale: about nine loads) at every state element in three
places: the `r` fold, the new state inside `o`'s operand, and the stored new state. The first is an operand edge, so a
cut seam; the other two are statements inside a lift, where the edge-based seams do not reach. The value clustering
finds such inline copies by the same value forms it compares edges with, and a cut of the `k̂` seam replaces their
statements with a read of the one workspace. With `k̂` and `q̂` cut at the parent (`PLACE@…/reduce.2/inner.2/map.10/map.2/inner.2/map`
and `PLACE@…/reduce.2/inner.1/map`), one small piece computes both, and the joint piece reads them.

gdn1 (`emmy run --golden … --realization gdn1-…970a92.m1.fm --bench --strict`, one run, the tuned rows):

| route | kernels | state pieces | sum of kernels | end to end |
|---|---:|---|---:|---:|
| the #1226 route | 12 | 2.43 + 3.42 + 4.04 = 9.88 µs | 94.6 µs | 129.9 µs |
| joined | 10 | 5.81 µs (one piece, `WORK=t8x16`) | 89.8 µs | 123.7 µs |
| joined, `k̂` and `q̂` cut | 11 | 2.97 µs + 0.92 µs (`q̂`/`k̂` producer) | 88.0 µs | 122.6 µs |

The joint piece's best row is `WORK=t4x32` with `REDUCE=coop/r4` on both folds. Against the #1226 route as an `--ab`
row, the outputs differ by at most 1.5e-5.

### The golden

`golden/rtx5090_sm120.json` holds the gdn1 route above with its tuned rows. The coordinate rule permutes the root
regions of the gdn16/gdn64 conv parents and swaps two operands in the recurrence parents, so their recorded routes did
not replay; they are the same cuts spelled on the new trees, re-recorded, and the pieces with new identities tuned. Per
target, `--strict-evidence` with the golden alone:

| target | kernels | main 250f983f µs | this branch µs |
|---|---:|---:|---:|
| gdn1 (fused parent) | 12 → 11 | 98.0 | 92.4 |
| gdn16 conv parent | 12 | 148.1 | 149.2 |
| gdn16 recurrence | 15 | 96.9 | 79.4 |
| gdn64 conv parent | 12 | 162.0 | 160.6 |
| gdn64 recurrence | 15 | 120.8 | 122.8 |

`emmy golden check` reports it current. The strict serving audit deploys 9 of 9 fast-math twins; the standard lane,
which serving does not use, deploys 0 of 9, as on main.

### Serving

Setup of `recipes/Qwen3.8-27B-NVFP4/DISCOVERY.md` (2026-10-21): one RTX 5090, one request, decode bucket 16, prefill
chunk 64, M1 tier 0, fast math, vLLM 0.23.0, `--strict-evidence`, an empty tune DB and an empty cubin cache. The boot
took 1037 s.

| prompt tokens | output tokens | decode tok/s | TTFT s | #1226 decode tok/s | #1226 TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 | 58.5 (55.3–61.7) | 0.15 (warm) | 57.2 | 0.16 (warm) |
| 227 | 38 | 59.2 (59.2–61.0) | 0.22 | 57.2 | 0.22 |
| 1027 | 195 | 58.7 (58.7–58.7) | 0.64 | 57.2 | 0.64 |

The first value is the second probe round; ranges are min–max over both. All answers are correct, and the prompts
tokenize to the same ids as on 2026-10-21.

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
- **gdn16/gdn64** compute the recurrence in the chunked form, with other contractions and other pieces. Both changes
  apply to every kernel, but nothing here is designed or measured for those programs.
- **State copies in serving** (moved from the deleted Qwen3.8 RTX 5090 follow-ups plan, measured at #1120): each GDN
  layer copies its 3 MiB recurrent state and 80 KiB history into runtime backing and back, about 0.383 ms per step of
  GPU copies. The in-place state read above is one answer; persistent state aliases or grouped copy submission are
  others. Aliasing must keep per-request ownership and graph pointer lifetime.
- **Nested `TILE`/`STAGE` stamps dropped under an output-tiled root** (moved from the same plan): in the gdn16
  recurrence piece `k_linear_matmul_mean_reduce_…__place_84c2d13867__place_149fbfea8c__partial`, picks stamp `TILE` on
  the two nested contractions and `STAGE=d2/smem-tma` on one, yet the CUDA runs those folds as serial per-cell loops
  with no shared memory. The binder accepts the stamps silently. Likely fix: offer a nested tile only where the binder
  realizes it, refuse the rest at the offer.
