# Qwen3.8-27B-NVFP4 (NVIDIA) discovery

## The two 6144-wide FP8 products split across blocks: milestone 9 met (2026-11-18)

One rented RTX 5090 (vast.ai; 128 vCPUs, driver 595.71, CUDA 13.0 toolkit), this branch on main at `45e539c6a`.
The 256-row Gated DeltaNet layer's two 6144-wide FP8 products are the in-z projection (`[256, 5120] x [5120,
6144]`) and the out projection (`[256, 6144] x [6144, 5120]`), about 42 us each, about 390 TFLOP/s. The qkv
projection (`[256, 5120] x [5120, 10240]`) runs at about 515.

- **The gap is how many blocks the output gives, not how the product is lowered.** The qkv pick is a 128 x 128
  block (`w2x2`, `f4x8/k2`, `d2/smem-tma`): 160 blocks on the card's 170 SMs. The out projection under that
  schedule gives 80 blocks and took 56.3 us in the 150-schedule tune of 2026-11-18. Per block that is 3.6
  TFLOP/s against the qkv block's 3.3. Smaller blocks fill the card and run slower; the best was 64 x 128. The
  loop, the staging and the PTX are the qkv kernel's: same atom, same TMA stage, no local memory.
- **The K split fills the card, and no card had measured it.** `030_cut` offers the split (`REDUCE=g2k`, `g4k`,
  `g2a` and wider) on both products. A tune by name measures only the schedule rows of the kernel set the greedy
  compile takes, so no tune had tried a split, and the golden held no split row for any matrix product.
- **Measured hot, the split pieces under the qkv block** (a partial kernel and its finalize):

| | unsplit (recorded) | 2-way split | 4-way split | 2-way, atomic adds |
|---|---|---|---|---|
| out projection, K = 6144 | 43.4 us | 33.9 + 3.0 | 30.3 + 3.8 | 39.9 |
| in-z projection, K = 5120 | 43.5 | 41.9 + 3.5 | 36.3 + 4.4 | 47.2 |

- **Recorded.** The `--ab` rows put the pieces' rows in the tune DB; the readout root (`k_372b36c38762`) was then
  recorded alone and took the 4-way split for both products: 152.6 us against 158.8. A 120-schedule tune of each
  partial found nothing faster than the qkv block.
- **Served, the 256-row pack exported strictly from this golden, against main's golden exported on the same card,
  timed alternately twice.** The strict serving audit passes with the 256-row chunk (80 of 80 twins).

| first token | 256 tokens | 7,680 tokens | 197,632 tokens |
|---|---|---|---|
| main's golden | 0.047 / 0.047 s | 1.363 / 1.362 s | 78.0 / 78.2 s |
| the split | 0.047 / 0.046 s | 1.319 / 1.322 s | 76.8 / 77.2 s |

  The 7,680-token first token is under the 1.34 s bar by about 20 ms, 1.97x stock's 0.670 s. This card ran main's
  pack 18 ms slower than the card of 2026-11-18 did (1.363 against 1.345 s). The products' rows fell 12 us a layer
  and the root's 6 us, about 9 to 17 ms over the prefill; the first token fell 42 ms. Unlike the MLP tune of
  2026-11-18, these rows understate what the chunk got; why was not measured.
- **TPOT moved with the tokens, not the speed.** 8.5 against 8.0 ms at 7,680 tokens, 9.0 against 10.3 ms at
  197,632, the same at 256. The decode programs are unchanged; the split sums in another order, so greedy decoding
  takes other tokens and the draft is accepted at another rate. Every cell is inside the TPOT bar.

## The 256-row Gated DeltaNet layer tuned (2026-11-18)

One rented RTX 5090 (vast.ai; 61 vCPUs, driver 620.70, CUDA 13.0 toolkit), main at `41b940d75`. The goal was the
7,680-token first token, 1.61 s against a bar of 1.34 s, through the Gated DeltaNet layer at 256 rows (0.85 ms
served, needs about 0.66).

- **The matrix-product arm of #1425 is not offered here.** The recurrence across the four 64-token chunks is one
  kernel (24 us). Its step also writes a per-step output that reads the state at the output's own coordinates, so
  `Fold.affine` refuses the step and the carried state's fork has no split to offer. #1425's own measurements
  say a step this wide loses under composition anyway, so this was not pursued.
- **The layer was a tune problem.** #1429 recorded the layer's kernel sets again without a tune, so most pieces
  held the one or two schedules the record tried. Twenty pieces were tuned by name (60 schedules; 150 for the two
  6144-wide FP8 products, which found nothing better; 250 for the MLP's two NVFP4 products), and the nine roots
  that hold them were recorded alone. Two names from the golden's row names no longer exist (`k_4f62d9ae0676`,
  `k_894fb90b13d9`); their kernels were found under their current names in a bench table.

| a root of the 256-row layer, recorded | before | now |
|---|---|---|
| MLP gate-up (NVFP4) and its activation pieces | 185.6 us | 143.0 |
| the readout (two FP8 products, the norm, the solve's consumers) | 225.7 | 158.8 |
| MLP down (NVFP4) | 95.4 | 79.7 |
| the input projections (qkv FP8 product) | 94.4 | 90.5 |
| the chunk's state products | 53.0 | 35.2 |
| the recurrence across chunks | 25.0 | 24.0 |
| two small recurrence roots | 41.4 | 33.8 |
| the gate's cumsum | 10.3 | 0.7 |
| the layer, all twelve roots | 734 | 569 |

- **Served, one session, the 256-row pack exported strictly from this golden.** First token 0.044 / 1.344 / 79.9 s
  at 256 / 7,680 / 197,632 tokens (0.054 / 1.61 / 82.4 on 2026-11-17), the 7,680-token cell timed twice (1.345,
  1.344). TPOT 8.0 / 8.2 / 10.0 ms; memory at rest 28.6 GiB. The strict serving audit passes with the 256-row chunk
  (80 of 80 twins). The 7,680-token first token misses the bar by 4 ms, 0.3%.
- **What the rows promised and what the chunk got.** The rows fell 165 us a layer, about 238 ms over a
  7,680-token prefill; the first token fell 266 ms. But the MLP's wider tune alone was 58 us a layer in its rows
  and only 25 ms of first token. Its rows are benched with the weight already in the card's L2; a served chunk
  reads it cold.
- **No faster schedule is known for** the three FP8 products (51, 42 and 42 us; the qkv one runs about 515 TFLOP/s,
  the two 6144-wide ones about 390), the MLP products at their new picks, and the recurrence across chunks.

## The 256-row pack on W8A8, against stock in one session (2026-11-17)

One rented RTX 5090 (vast.ai; 46 vCPUs, driver 595.99, CUDA 13.0 toolkit), main at `f1b7171fc`, native and stock
vLLM 0.32.0 in one session with the settings of the 2026-10-25 section and prompts cut again from this tree. The
pack is exported with 256-row prefill chunks under strict evidence from this golden alone, and the strict serving
audit passes with the 256-row chunk (80 of 80 twins).

| | native, 256-row chunks | stock vLLM 0.32.0 | native on 2026-11-15, W8A16 |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 7.74 ms / 0.054 s | 7.34 ms / 0.160 s | 8.6 ms / 0.068 s |
| TPOT / TTFT, 7,680-token prompt | 8.89 ms / 1.61 s | 7.96 ms / 0.670 s | 8.7 ms / 2.44 s |
| TPOT / TTFT, 197,632-token prompt | 10.1 ms / 82.4 s | 10.05 ms / 68.3 s | 11.5 ms / 113 s |
| device memory at rest | 28.6 GiB | 30.6 GiB | 28.9 GiB |

- **Against the bar.** TPOT is inside it in every cell (1.05x, 1.12x and 1.01x stock; the bar 1.2x, 1.2x and 1.5x).
  The first token is ahead of stock at 256 tokens, inside the bar at 197,632 (1.21x, the bar 2x) and outside it at
  7,680: 2.4x stock, where 2x is 1.34 s. Every native cell was run twice, on two exports of the pack, and repeats to
  about 1%. GSM8K was not scored and draft acceptance was not read in this session.
- **The golden.** Main's file already held the 256-row Gated DeltaNet program as #1420 traces it, fully recorded,
  so this file is main's plus the records of the section below for the other programs. Four roots were open after
  that and are recorded here, each with its pieces tuned and the root recorded alone: the 16-row program's first
  and second root (49 and 93 us) and its small recurrence root (28 us), and the 64-row program's second root
  (125 us). A record compiles every root of its program, so the three 16-row roots had to be benched before any
  of them could be recorded. Two roots of an older 16-row program that no fresh trace gives are dropped with the
  pieces only they reached, and two rows of one alternative piece are proposals again.
- **Where a 256-row chunk goes.** A 7,680-token prefill is 30 chunks of about 53 ms. From `nsys
  --cuda-graph-trace=node` over ten 256-token and four 7,680-token requests (130 chunks), kernel time a chunk: the
  Gated DeltaNet layers 40.8 ms (0.85 ms a layer), the attention layers 11.1 ms (`post` 5.5, the fold 3.4, `pre`
  2.2), everything else under 1 ms. So the chunk is kernel time now: the 25 ms that no row explained on 2026-11-15
  is not there on this pack. Inside a Gated DeltaNet layer the two MLP kernels take 161 and 94 us and the largest
  recurrence and projection pieces 71, 58, 53 and 52 us.
- **Where a decode step goes.** The same trace holds 1,497 verify steps. The three-row Gated DeltaNet programs take
  13.2 ms of kernel time a step, more than everything else in the step together: the next are the 16-row `post`
  and `pre` programs (2.2 and 1.2 ms), the head (1.4 ms) and the drafter (1.2 ms). One three-row kernel alone,
  71 us a launch, is 3.4 ms a step.
- **What the 7,680-token cell still needs.** 1.34 s is about 44 ms a chunk, 9 ms less than now. Only the Gated
  DeltaNet layers hold that much.

## Nineteen W8A8 roots recorded (2026-11-16)

The W8A8 programs of the section below had no rows in this golden, so no strict export worked on main. Recorded on
one rented RTX 5090 (48 vCPUs, CUDA 13.3), main at `0d69a2def`: every root the fresh trace adds, 19 of them, each
with a cut route, its pieces tuned (`--tune 40`) and the root recorded alone under strict evidence. The golden
drops the four targets only the replaced W8A16 programs wrote. The rental ended before the export passed: no pack
was served and no cell was timed in this session.

Main then moved under these records: #1420 traces the 256-row Gated DeltaNet program another way and settles a
new normal form, which re-keys pieces of the first root at 3, 16 and 64 rows. The section above holds what was
recorded again; the 256-row column below is the program as it was traced before #1420.

| a layer's new roots, warm | 1 row | 2 rows | 3 rows | 16 rows | 64 rows | 256 rows |
|---|---|---|---|---|---|---|
| Gated DeltaNet, first / second root | 94 us (one root) | 59 / 34 | 61 / 38 | 45 / 100 | 58 / 143 | 114 / 204 |
| attention `pre` | | | | 35 | 52 | 128 |
| attention `post` (the out projection's root) | | | | 18 | 31 | 76 |

The any-width `pre` and `post` roots take 233 and 94 us at 512 rows.

- **What is left before a strict export.** Five small recurrence roots that #1411's restamp left as proposals: one
  of the 16-row program and four of the 256-row one. Routes were benched for all five (37 us; 6, 41, 0.7 and 229 us,
  the last untuned) and none was recorded: the strict record of each stops in the 16-row program's first root, at a
  piece (`type_as`, the FP8 codes) that one route cuts and another leaves whole, where the whole piece's measured
  row is not among the schedules the family compile offers. The export from this golden alone gets through the
  vision tower, the drafter and the decode and verify programs and stops at the 256-row prefill program's first
  such root.
- **What the rows say, with those five missing.** A 256-row Gated DeltaNet layer sums to 0.65 ms without the four
  unrecorded roots (0.86 ms before, all roots) and an attention layer to 0.69 ms (0.90 ms). A three-row Gated
  DeltaNet layer of the verify step sums to 0.14 ms. Sums of isolated rows, not a measurement of a chunk or a step.
- **The routes.** As the priors pick them, all 19 roots run as one kernel on one block and hang. What worked: cut
  every contraction, every reduction and every whole output, on the root and on each piece; inside a piece that
  encodes an FP8 activation, cut the activation too, so the products are pure e4m3 products; leave a
  contraction's other operands inside it. Cutting an operand that gathers the chunk state out of its product
  writes a 200 MB buffer (the 256-row second root: 2.4 ms against 0.20 ms). Cutting every seam makes copy kernels
  (the one-row root: 294 us against 94 us). The priors then give a state contraction a cooperative reduce on a
  grid of up to 393,216 blocks (175 us) where a tuned tile takes 3 us, so every piece over a few microseconds has
  to be tuned by name.
- **Two things that cost the session time.** The first 256-row pack was exported without strict evidence to
  profile it, and never returned its first token: the priors' picks for the unrecorded roots are that slow. And
  the host stopped the instance once; the tune DB was in memory and its unrecorded rows were lost.

## The FP8 layers as W8A8, measured root by root (2026-11-15)

The checkpoint declares its FP8 leaves' activations static: a modelopt `config_groups` entry with 8-bit float
weights and inputs that are not dynamic, and an `input_scale` beside each linear. The loader read static FP8 only
from `quant_method: fp8`, so these linears ran as W8A16. It now reads the group too, and every attention q/k/v/o and
every Gated DeltaNet in_proj_qkv, in_proj_z and out_proj is a W8A8 product on the e4m3 tensor-core instruction, the
checkpoint's own arithmetic and stock vLLM's. The MLPs and the head stay NVFP4. Measured on a rented RTX 5090
(CUDA 13.3, 14 vCPUs), each root recorded under strict evidence, before main's block-scaled FP8 instruction (#1418).

- **The product.** The 256-row in-projection qkv (256 x 5,120 x 10,240) takes 81 us as W8A8 untuned against 175 us
  as W8A16 (131 tuned); at 512 rows 148.5 against 323.
- **Attention halves, a layer, warm:** `pre` 41.7 / 75.2 / 162.4 / 252 us at 16 / 64 / 256 / 512 rows (W8A16
  `pre256` about 255), `post` 26.2 / 40.0 / 72.5 / 117.4.
- **Gated DeltaNet roots, a layer**, against the cheapest W8A16 route the golden held (warm):

  | root | W8A16 warm | W8A8 warm | W8A8 cold |
  |---|---|---|---|
  | 1 row | 82.3 us | 75.1 | 143.3 |
  | 2 rows, `pre` / `post` | 105.4 / 183.9 | 32.8 / 45.7 | 64.6 / 96.0 |
  | 3 rows, `pre` / `post` | 49.7 / 119.2 | 28.4 / 53.7 | 62.2 / 97.6 |
  | 16 rows, `pre` / `post` | 369.0 / 488.4 | 29.3 / 100.2 | 64.3 / 153.1 |
  | 64 rows, `pre` / `post` | 200.8 / 166.2 | 48.9 / 145.7 | 72.6 / 202.3 |
  | 256 rows, `pre` / `post` | 219.6 / 296.8 | 108.8 / 234.2 | 131.0 / 289.6 |

  The large drops at 2 to 16 rows are partly schedules the old routes never tried, not only the FP8 operand.
- **The route that works.** Cut the contraction seams first, so one piece holds every projection, then the FP8
  seams inside it: one kernel writes the codes, one piece is pure e4m3 products. Cutting the FP8 seams first
  recomputes the projections (480 us for `pre256` against the W8A16 route's 255). K chunks of 2 win at 256 rows and
  up (the `pre256` q product 229 -> 111 us; the prior picks 8). The chunk-output contractions want `t16x8` or
  `t16x16` tiles (3 to 28 us) where the prior picks a cooperative reduce (0.1 to 3.5 ms).
- **in_proj_z.** It shares in_proj_qkv's calibrated scale. One shared decoded value crossed the recurrence's kernel
  boundary as a 16-bit buffer, so z ran on the bf16 instruction. Each linear now decodes the shared codes itself:
  the 256-row z product 102.1 -> 57.3 us cold, the `post` root 290.1 -> 248.6.
- **The triangular inverse.** At 256 rows it is a 61-step rolled scan that stored every step's 64 x 64 matrices
  (192 MB a layer) while only the last was read. It now stores the last alone: 129 -> 39 us cold.
- **Not measured end to end.** The golden here still holds the W8A16 programs. A fresh trace now gives the W8A8 ones,
  so the next strict export of this recipe needs those roots recorded first. Summed from the rows, a 256-row chunk
  would drop from 67 ms to about 51 ms, about 1.5 s to the first token at 7,680 tokens (2.0 s measured with
  W8A16 on this card, the bar 1.51 s); an estimate from isolated rows, not a measurement.
- **Left open.** One FP8 activation is offered as one seam per reader, and those seams do not group, so the kernel
  that writes the codes writes two identical copies. Every `post` bench warns that its wrong-answer reference is
  unusable. A verify step's state copy into the program's output costs about 8 us a layer (0.38 ms a step); removing
  it re-forms the scan's routes in three goldens and is left for its own change.

## A 256-row prefill chunk, recorded and timed (2026-11-15)

Native generation can run a prompt in 256-row chunks (`--prefill-size 256`, `SERVE_NATIVE_PREFILL=256`), keeping the
64-row chunk and the decode width for a prompt's tail. This recipe still serves 64-row chunks; the golden now holds
every program of the wider chunk. Measured on one rented RTX 5090 (48 vCPUs, 108 GiB of RAM, PCIe Gen4 x16, driver
610.57, CUDA 13.3), one session, both packs exported from this golden alone under strict evidence and served by the
same build. The client and settings are the 2026-10-25 section's; the prompts are cut again from this tree's Markdown.
Stock vLLM was not measured in this session, so the bar below is the 2026-11-12 one.

| | 64-row chunks | 256-row chunks | bar |
|---|---|---|---|
| first token, 256-token prompt | 0.147 s | 0.068 s | 0.32 s |
| first token, 7,680-token prompt | 4.46 s | 2.44 s | 1.51 s |
| first token, 197,632-token prompt | 172 s | 113 s | 136 s |
| TPOT at 256 / 7,680 / 197,632 | 8.84 / 8.78 / 11.9 ms | 8.60 / 8.67 / 11.5 ms | 9.44 / 9.28 / 14.3 ms |
| device memory at rest | 28.6 GiB | 28.9 GiB | |
| GSM8K strict / flexible match | 0.690 / 0.735 | 0.675 / 0.695 | |
| twins that deploy from the golden alone | 67 of 67 | 80 of 80 | |

- **Width halves the first token and does not reach the 7,680-token bar.** A 256-row chunk takes about 81 ms where
  four 64-row chunks take about 149 ms. The bar needs about 45 ms a chunk. The golden's rows hold about
  55.5 ms of it: a Gated DeltaNet layer is 0.86 ms (FP8 projections 0.40, MLP 0.29, the recurrence 0.16) and an
  attention layer 0.90 ms at a 4,096 window (projections 0.40, MLP 0.28, the fold 0.21). Over the chunk the FP8
  projections are 25.5 ms, the MLPs 18.6 ms, the recurrence 7.4 ms and the fold 3.3 ms. About 25 ms is in no row;
  a served chunk was not profiled. (Corrected 2026-11-16: the split first written here counted the MLP twice.)
- **The 197,632-token first token is inside the bar.** Long-window attention at 256 rows takes 4.96 ms a layer at
  131,072 positions, 3.1x the 64-row fold for four times the rows.
- **GSM8K.** The two packs differ by 0.015 strict and 0.040 flexible, with a standard error of 0.033 on each score:
  inside one standard error on strict, 1.2 on flexible. The same prompts continue differently after 0 to 110
  characters on the two packs, and both continuations read well; the kernels differ at every width, so their
  rounding does.
- **What was recorded.** The last three 256-row targets: the MLP's gate and up projection kernel (184 us) and the
  attention fold at both window ranges (207 us at 4,096, a 2-way key split; 4.96 ms at 131,072, a 16-way one). As
  the priors pick it, that fold takes 3.8 ms and 117 ms. The drafter's two 256-row programs are recorded again with
  the weight layouts its narrower programs settle (0.36 and 0.97 ms, as before); recorded alone they were kernels an
  export never builds. The plugin's any-width `pre` program, the one twin the strict audit failed on, is recorded.
- **No native pack exported on main before this session's fixes.** Token selection carries the picked index, a
  64-bit integer, through a warp shuffle and through a 32-bit float slab, and the PTX compiler that replaced nvcc had
  a rule for neither.

## One cache read per verify step, and the golden measured again (2026-11-12)

A verify step's attention now folds its three rows beside the query group, so every row shares one read of the
key-value cache; before, each row read every slice. Measured on one rented RTX 5090 (CloudRift; 14 vCPUs of a Ryzen 7
7800X3D, 21.5 GiB of RAM with swap on, PCIe Gen5 x16, driver 580.159, CUDA 13.3), one session, native and stock
vLLM 0.32.0 both. The client and settings are the 2026-10-25 section's; the prompts are cut again from this tree's
Markdown, so acceptance differs from the earlier sessions'. The pack exports under strict evidence from the golden
alone.

**The golden first.** On main the file held 225 measured rows of 415: #1365's normal form re-keyed the rest, and a
strict export stopped at its first fork. This session measured the demoted schedules again on the card (the file now
holds 375 measured rows of 412) and recorded the new verify programs. Three things needed more than a re-measure:

- **A split attention's combine kernel is priced at its window.** An export prices each range of windows at its own
  window, and a record stores the combine kernel's row at the stored program's hint. With that row at another size the
  split's arm has no price, the unsplit kernel's row wins, and nothing fails: a pack exported that way ran its decode
  attention unsplit (389 µs a layer against 11) and gave 10.6 ms a token at 256 tokens and 6.1 s to the first token
  at 7,680. Every attention twin's split is recorded again at its window, and each combine row is stored under the
  window it was measured at.
- **One Gated DeltaNet piece of the 64-row program had no row.** The compile's own pick for it takes 336 µs a layer;
  the schedule its sibling pieces carry takes 7.7 µs and is the row recorded.
- **Two schedules no longer realize** (a 64-row Gated DeltaNet projection piece and one alternative of the three-row
  program). The first is recorded at the schedule the compiler realizes in its place, 49.5 µs; the second is left a
  proposal, beside the plugin's any-width `pre` piece, which native does not run.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 10.7 µs (11.1) | g64k, 183 µs (205) |
| `attend3` (verify), rows folded | g32k, 12.5 µs (22.5) | g32k, 183 µs (487) |
| `attend16` | g16k, 39.1 µs (41.7) | g32k, 852 µs (974) |
| `attend64` (chunk) | g8k, 78.6 µs (73.0) | g32k, 1,601 µs (1,590) |

The 2026-11-07 times, on another card, are in brackets. The verify step's other new programs take 2.3 µs (the rotary
store, which now hands the query out per key-value head) and 0.7 µs (the unfold that puts the rows back in order).

| | native, this branch | stock vLLM 0.32.0 | native on 2026-11-07, another card |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.84 ms / 0.143 s | 7.87 ms / 0.159 s | 8.68 ms / 0.151 s |
| first token, 257-token prompt | 0.165 s | 0.127 s | 0.172 s |
| TPOT / TTFT, 7,680-token prompt | 8.72 ms / 4.37 s | 7.73 ms / 0.753 s | 8.55 ms / 4.62 s |
| TPOT / TTFT, 197,632-token prompt | 11.4 ms / 172 s | 9.54 ms / 67.9 s | 14.7 ms / 188 s |
| TPOT / TTFT, 261,632-token prompt | 11.4 ms / 251 s | does not fit | 17.7 ms / 277 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 63 / 77 / 68 / 75% | not read: its stream spaces every token | 75 / 79 / 71 / 64% |
| device memory at rest | 27.9 GiB | 30.4 GiB | 27.9 GiB |
| GSM8K strict / flexible match | 0.725 / 0.755 | | 0.715 / 0.765 |

- **Against this session's stock column.** TPOT is inside the bar at 7,680 tokens (1.13x stock, the bar 1.2x) and
  at 197,632 (1.20x, the bar 1.5x) and outside it at 256 (1.25x). The first token is ahead of stock at 256 tokens and
  behind at 7,680 (5.8x, the bar 2x) and at 197,632 (2.5x). Native holds 2.5 GiB less of the device and serves the
  whole window, which stock does not fit.
- **The long-window decode gate, met.** The captured verify step takes 26.4 ms at 197,632 positions (the target's
  attention 4.5 ms, the drafter's 0.5) against 21.6 ms at a window of up to 8,192 (attention 0.35 ms): +4.8 ms, where
  the bound is 8 ms and the 2026-11-07 step was +14.4.
- **The short cells.** The step is 21.6 ms where 2026-11-07 captured 21.1; TPOT at 256 tokens reads higher here
  because these prompts accept fewer drafts (2.26 tokens a step).
- **GSM8K holds:** 0.725 strict and 0.755 flexible match on native (standard error 0.032), against 0.715 / 0.765 on
  2026-11-07. Stock was not scored in this session, and the probe tokens were not compared.

Where the time goes, from `nsys --cuda-graph-trace=node` (one request at 7,680 tokens and one at 197,632), each
launch priced at its kernel's median:

| launches | 64-row chunk | verify step, up to 8,192 | verify step, 197,632 |
|---|---|---|---|
| Gated DeltaNet layers (48) | 27.7 ms | 14.3 ms | 14.4 ms |
| attention layers: `post` (16) | 3.0 ms | 2.2 ms | 2.3 ms |
| attention layers: `pre` (16) | 2.8 ms | 1.7 ms | 1.7 ms |
| attention | 1.2 ms | 0.35 ms | 4.5 ms |
| output head (one row; a verify step runs it three times) | 0.45 ms | 1.3 ms | 1.3 ms |
| the drafter's layer and prologue | 0.44 ms | 1.4 ms | 1.9 ms |
| state carry, rotary store, unfold, the rest | 0.3 ms | 0.3 ms | 0.3 ms |
| sum | 35.9 ms | 21.6 ms | 26.4 ms |

The 197,632-token first token is 171 s of kernels: 86 s in the Gated DeltaNet layers, 62 s in attention (69.9 s on
2026-11-07), 18 s in the attention layers' halves. A 64-row chunk's Gated DeltaNet layers take 577 µs each; the largest
launches are the MLP gate and up (74 µs), the convolution piece (71 µs), `k_slice_unsqueeze_reduce` (59 µs) and the
MLP down (54 µs).

**A 256-row chunk does not load yet.** Traced at 256 rows the Gated DeltaNet layer has twelve root kernels where the
64-row one has four: the recurrence across its four 64-token chunks adds eight. No golden row covers them, and the
compile's own cuts leave four workspaces of `[4, 64, 64, 48, 48, 128]` FP32 a layer, 18 GiB each, so the pack runs out
of device memory at load. Under the 64-row kernel's recorded route one piece of the 256-row kernel does not finish in
80 s. A wide chunk therefore needs its own recorded route before it can be timed, and no time for one is in this
section.

## The FP8 cache staged into shared memory: milestone 5 met, 6 still missed (2026-11-07)

Native attention's FP8 key-value cache now copies into shared memory, 16 bytes per copy, before the tensor-core fold
reads it; it was read one byte per load, which kept every fold at 255 registers with spills (#1366, on #1352). Recorded
and gated on the card and host of the section below (vast.ai, x16 RTX 5090, EPYC 9655, a 46-CPU quota, driver
610.57). Every attention twin was swept over its neighbouring key splits and four tiles with the cache staged, and the
compile's pick recorded. The client, prompts and settings are the section below's. The pack exports under strict
evidence, every twin compiles strictly from the golden, and `emmy eval golden --serving-config` deploys every twin from
the golden rows alone.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 11.1 µs (27.7) | g64k, 205 µs (581) |
| `attend3` (verify) | g32k, 22.5 µs (54.9) | g64k, 487 µs (1,413) |
| `attend16` | g16k, 41.7 µs (141) | g32k, 974 µs (6,516) |
| `attend64` (chunk) | g8k, 73.0 µs (195) | g32k, 1,590 µs (5,126) |

The section below's times are in brackets. The one-row decode's split piece reads its 268 MB cache in 189 µs, about
1.4 TB/s; at one warp per CTA it took 170 µs staged against 502 µs read directly. g128k, which the compiler does not
offer, measured 2% faster than g64k at one row and 7% at three.

| | this branch | the section below |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 8.68 ms / 0.151 s | 9.44 ms / 0.154 s |
| first token, 257-token prompt | 0.172 s | 0.177 s |
| TPOT / TTFT, 7,680-token prompt | 8.55 ms / 4.62 s | 9.72 ms / 4.88 s |
| TPOT / TTFT, 197,632-token prompt | 14.7 ms / 188 s | 23.7 ms / 318 s |
| TPOT / TTFT, 261,632-token prompt | 17.7 ms / 277 s | 32.7 ms / 505 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 75 / 79 / 71 / 64% | 70 / 71 / 79 / 62% |
| device memory at rest | 27.9 GiB | 27.9 GiB |
| GSM8K strict / flexible match | 0.715 / 0.765 | |

- **Milestone 5, met.** The attention term of the 197,632-token first token is 69.9 s of GPU time (`nsys
  --cuda-graph-trace=node`, 3,092 chunks), against at most 117 s (200 s in the section below). The full window's first
  token is 277 s, against at most 450 s.
- **Milestone 6, missed.** The captured verify step takes 35.5 ms at 197,632 positions (attention 14.3 ms) against
  21.1 ms at the 4,096 window (attention 0.41 ms): +14.4 ms, where the bound is about 8 ms (+38.9 in the section
  below). The three-row verify reads the cache once per row, three reads where one would do: its twin at 131,072 takes
  487 µs against the one-row decode's 205.
- **The short cells did not get slower.** The 256- and 7,680-token cells are faster on both TPOT and first token.
- **GSM8K holds,** inside one standard error (0.032) of main's 0.730 / 0.750 on 2026-11-06; the section below did not
  run it.
- Drafts accepted is the share of drafted tokens a step kept, read off the gaps between streamed tokens.

## Attention priced per range of windows: milestones 5 and 6 again (2026-11-07)

Native attention now compiles once per range of windows, priced at 4,096 and at 131,072 cached tokens, and a step runs
its window's range's program; the long range starts at 16,384 (#1352, stacked on #1329's grouped decode attention).
Recorded and gated on one rented x16 RTX 5090 (vast.ai; Gen5 x16; a 46-CPU quota on a 96-core EPYC 9655 host, driver
610.57, CUDA 13.1). The client, prompts and settings are the 2026-11-06 section's, the prompts cut from that section's
tree. The pack exports under strict evidence.

Every attention twin was swept whole and under key splits g2k to g128k, the FP8 cache unstaged, no TMA, fast math on,
and the compile's pick recorded. The compiler offers splits up to g64k; g128k measured faster on the decode twins at
131,072 (one row 475 against 581 µs, three rows 1,296 against 1,413 µs), and no compile can deploy it.

| twin | at 4,096 | at 131,072 |
|---|---|---|
| `attend1` (decode) | g32k, 27.7 µs | g64k, 581 µs |
| `attend3` (verify) | g16k, 54.9 µs | g64k, 1,413 µs |
| `attend16` | g8k, 141 µs | g8k, 6,516 µs |
| `attend64` (chunk) | g16k, 195 µs | g32k, 5,126 µs |

| | this branch | 2026-11-06 (main) |
|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.44 ms / 0.154 s | 8.70 ms / 0.15 s |
| first token, 257-token prompt | 0.177 s | 0.17 s |
| TPOT / TTFT, 7,680-token prompt | 9.72 ms / 4.88 s | 9.76 ms / 4.81 s |
| TPOT / TTFT, 197,632-token prompt | 23.7 ms / 318 s | 42.9 ms / 429 s |
| TPOT / TTFT, 261,632-token prompt | 32.7 ms / 505 s | 50.6 ms / 687 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 70 / 71 / 79 / 62% | 76 / 75 / 61 / 67% |
| device memory at rest | 27.9 GiB | 27.9 GiB |

- **Milestone 5, missed.** The attention term of the 197,632-token first token is 200 s of GPU time (`nsys
  --cuda-graph-trace=node`, 3,088 chunks), against at most 117 s (300 s on 2026-11-06). The full window's first token is
  505 s, against at most 450 s (687 s).
- **Milestone 6, missed.** The captured verify step takes 60.7 ms at 197,632 positions (attention 39.6 ms) against
  21.9 ms at the 4,096 window (attention 1.1 ms): +38.9 ms, where the bound is about 8 ms (+73.7 on 2026-11-06).
- **The short cells did not move.** The drafted step at 256 tokens takes 21.9 ms on both trees; TPOT reads higher here
  because fewer drafts were accepted on these prompts, which the attention's changed numerics move.
- **What is left is the kernels, not the pricing.** Every attention fold here runs at 255 registers with local memory
  spills and 17% occupancy. The one-row decode at 131,072 reads 268 MB of cache in 581 µs, about 0.46 TB/s; the 64-row
  chunk takes 5.1 ms there, where reading the cache once per query head at the card's bandwidth takes under 1 ms.

## The gates of milestones 3, 4 and 5 (2026-11-06)

The gates of the parity plan's milestones 3 (the rotary store's grid), 4 (prefill ends at the prompt's last token) and 5
(the folded prefill attention), native only, on one rented x16 RTX 5090 (vast.ai; Gen5 x16; a container with a 46-CPU
quota on a 384-CPU EPYC 9B14 host, driver 615.71, CUDA 13.1). Main is at `f33a86522`, after #1324 recorded the rope,
row-pick, drafter-token and attention twins. The client, prompts and settings are the 2026-11-01 section's. Two
reference packs ran on the same card, from the tree before 4 (#1311, `504165b13`) and the tree before 5 (#1318,
`57fabd6e2`); neither is strict, since rows later PRs recorded are missing from their goldens. Main's column is a
second pass on an idle host. The first pass, while pack exports compiled on the host, gave the same numbers within 4%
(413 and 684 s at the two long prompts).

On the card, `emmy eval golden --serving-config` deploys all 62 twins from the golden rows alone, the pack exports
under strict evidence, and `emmy golden check` says current, 517 kernels.

| | #1311 tree | #1318 tree | main | 2026-11-01 |
|---|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.65 ms / 0.70 s | 9.31 ms / 0.17 s | 8.70 ms / 0.15 s | 9.8 ms / 0.71 s |
| first token, 257-token prompt | 0.24 s | 0.19 s | 0.17 s | 0.24 s |
| TPOT / TTFT, 7,680-token prompt | | | 9.76 ms / 4.81 s | 10.5 ms / 7.39 s |
| TPOT / TTFT, 197,632-token prompt | | | 42.9 ms / 429 s | 26.5 ms / 698 s |
| TPOT / TTFT, 261,632-token prompt | | | 50.6 ms / 687 s | 41.9 ms / 1,154 s |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 71% at 256 | 73% at 256 | 76 / 75 / 61 / 67% | 70 / 66 / 88 / 57% |
| device memory at rest | 30.7 GiB | 30.7 GiB | 27.9 GiB | 30.7 GiB |
| GSM8K strict / flexible match | | | 0.730 / 0.750 | 0.720 / 0.755 |

The gates:

- **Milestone 4, met.** At 256 tokens, four 64-row chunks, the first token comes in 0.15 s against the 257-token
  control's 0.17 s; the tree before it took 0.70 s, its control 0.24 s. The trees before and after #1313 give the same
  64 greedy tokens on all four probes (`scripts/validate_serve.py`'s), on both models.
- **Milestone 3, met.** In the captured step the rotary store takes 0.09 ms of a 64-row chunk, against 11.9 ms on
  2026-11-01, and the chunk takes 38.6 ms at up to 4,096 positions, against 54.1 ms. The verify step's three-row store
  takes 0.08 ms of the step, against about 0.6 ms.
- **Milestone 5, missed.** The attention term of the 197,632-token first token is 300 s of GPU time, against at most
  117 s (544 s on 2026-11-01), and the full window's first token is 687 s, against at most 450 s. A layer at 65,536
  positions takes 3.66 ms against 6.7 ms: 1.8x, where the budget needs 4.6x. The cells within 4,096 positions did not
  get slower. GSM8K holds, inside one standard error of the plugin lane's 0.720 / 0.770 on 2026-11-01.

Where the time goes, from `nsys --cuda-graph-trace=node` on main (7,680 and 197,632 tokens; first token within 7% of
the unprofiled runs), the median launch in milliseconds:

| | 64-row chunk, up to 4,096 | 64-row chunk, 4,096 to 7,680 | 64-row chunk, all of 197,632 (mean) | verify step, 7,680 | verify step, 197,632 |
|---|---|---|---|---|---|
| rotary store | 0.09 | 0.09 | 0.09 | 0.08 | 0.08 |
| attention | 3.07 | 5.98 | 97.2 | 2.90 | 75.1 |
| the rest | 35.9 | 35.9 | 36.0 | 20.8 | 20.8 |
| the launch, start to end | 38.6 | 41.5 | 132.9 | 23.4 | 95.6 |

- **The 64-row fold takes the same split at every window.** Its rows were measured at 512 cached tokens, where the
  4-way key split wins, so a chunk runs that split in 192 blocks however many positions it holds: 0.37 ms a layer at
  4,096 positions, 3.66 ms at 65,536 and 12.3 ms at 197,568, where each block streams about 49,000 keys. Whether a
  wider split wins at long windows was not measured. The fold also frees 2.8 GiB of device memory, the old route's
  scores and probabilities.
- **Decode attention got slower at long windows.** The verify step's three-row attention runs as a score cut and a
  16-way key split of the rest, also picked at 512 cached tokens: 0.17 ms a layer at 7,700 positions and 4.3 ms at
  197,632, where it is 75 ms of a 95.6 ms step. The drafted step at 197,632 took 73 ms on 2026-11-01, so TPOT there went
  from 26.5 to 42.9 ms, and at the window's longest prompt from 41.9 to 50.6 ms. Milestone 6 re-spells this program.
- **The host leaves 2.6 ms between a prompt's chunks** (median, nsys at 7,680 tokens on an idle host), about 8 s over
  the 3,088 chunks of a 197,632-token prompt, and 3.7 ms while exports compiled on the host. Not investigated.
- **Probe tokens.** Main and the #1318 tree part at tokens 46, 9, 5 and 4 of the four probes' 64; each pack repeats its
  own tokens. Between them differ the fold and #1322's W4A4 head, which alone parted one probe at its second token.
  Where they part, both continuations read well ("a young boy named Tom" against "a young man named Tom"). Native
  returns no log-probabilities, so the margin at each parting point is not measured.

## Native, plugin and stock vLLM again, with concurrency and the chunk's kernels (2026-11-01)

The protocol of the 2026-10-25 table below, on main at `ea17c0964`, one session on one rented x16 RTX 5090 (Gen5 x16, 57
GB/s under load; 16 cores, 60 GiB of RAM, no swap, driver 595.84, CUDA 13.1, a container with no Docker).

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 9.8 ms / 0.71 s | 10.3 ms / 0.19 s | 7.7 ms / 0.085 s |
| TPOT / TTFT, 7,680-token prompt | 10.5 ms / 7.39 s | 10.4 ms / 5.09 s | 7.6 ms / 0.63 s |
| TPOT / TTFT, 197,632-token prompt (stock's longest) | 26.5 ms / 698 s | 11.9 ms / 180 s | 8.0 ms / 67.4 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 41.9 ms / 1,154 s | 15.7 ms / 260 s | does not fit |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 70 / 66 / 88 / 57% | 71 / 71 / 78 / 54% | 68 / 71 / 91% / - |
| TPOT with the drafter off, 256 / 7,680 | 16.2 / 16.5 ms | | |
| first token at 256 / 257 / 268 tokens | 0.71 / 0.24 / 0.42 s | | |
| device memory at rest / peak | 30.7 / 31.2 GiB | 29.9 / 30.7 GiB | 30.5 / 30.6 GiB |
| KV cache | 262,144 tokens, the window | 305,585 tokens at 0.93 | 201,383 tokens at 0.93 |
| host memory of the server at rest, peak | 3.4 GiB, 5.1 GiB | 11.9 GiB, 12.1 GiB | 5.5 GiB, 5.7 GiB |
| GSM8K strict / flexible match | 0.720 / 0.755 | 0.720 / 0.770 | 0.585 / 0.640 |
| ready after | 5 s from its pack | 1,500 s, no pack | 47 s, 270 s at its first compile |

Stock and the plugin at higher concurrency, with `--max-num-seqs` raised and nothing else changed but stock's window,
which 198,400 tokens no longer fit at these row counts (176,000 tokens here):

| | stock, 8 | stock, 32 | plugin, 8 | plugin, 32 |
|---|---|---|---|---|
| 256-token prompts: per-request TPOT | 9.2 ms | 10.1 ms | 87.7 ms | 168.9 ms |
| total output tokens a second | 756 | 1,070 | 75 | 69 |
| first token | 0.14 s | 4.75 s | 4.26 s | 74.4 s |
| 7,680-token prompts: per-request TPOT | 19.3 ms | 24.2 ms | 143.8 ms | 147.9 ms |
| total output tokens a second | 276 | 281 | 6.9 | 6.6 |
| first token | 1.42 s | 22.0 s | 261 s | 1,201 s |
| device memory peak | 30.6 GiB | 30.6 GiB | 30.8 GiB | 30.8 GiB |

Each cell runs 4 x concurrency requests at 256 tokens and 2 x concurrency at 7,680, each request a different slice of
the same Markdown. First token is the median request's, queueing included. The plugin keeps the recipe's settings, which
name one request; what stops it batching was not investigated.

What differs from the 2026-10-25 session:

- **The lanes are pip environments, not images:** vLLM 0.25.1 and this source for the plugin, vLLM 0.32.0 with
  FlashInfer 0.7.0.post1 for stock, which compiles FlashInfer at first use. That compile on 16 cores ran out of the
  container's memory, so stock's first boot and one warming request of each size ran on four cores. Nothing measured
  did.
- **The native pack is not strict.** `emmy eval golden` fails on this recipe at `ea17c0964`: 57 twins deploy from the
  golden rows and `prologue16-native` does not (`k_linear_mean_reduce_89f977` has no measured row for its `030_cut`
  fork). The pack was exported without `--strict-evidence`, so a prior decides that one fork; every other program's rows
  are the golden's. #1302 landed that fork's row after this session.
- **The plugin has no pack here,** so its ready time is a lowering of every program and does not compare.
- Native's `/health` means loaded: at its 5.4 s answer the device already held the full 30.7 GiB, and a request sent at
  once got its first token in 0.76 s.

Where the time goes, from `nsys --cuda-graph-trace=node` on native (7,680 and 197,632 tokens, under 4% slower than the
unprofiled cells):

- **Decode.** The drafted step takes 23.5 ms on native, 25 ms on the plugin and 18.1 ms on stock (TPOT times tokens a
  step), and the one-row step 16.6 ms on native. Both Emmy lanes took 32-33 ms on 2026-10-25, before #1282.
- **The 64-row chunk.** 54.1 ms at up to 4,096 positions and 60.3 ms up to 8,192; its kernels sum to 54.6 and 60.8 ms.
  The table is that chunk against the one-row step, by what each launch reads. Weight bytes are the checkpoint's. A
  launch that reads weights at the one-row rate, about 1.43 TB/s, is bound by that read; one that grows with rows is
  bound by rows.

  | launches | weights | one row | 64 rows | bound by |
  |---|---|---|---|---|
  | rotary cache store (one CTA) | | 0.06 ms | 11.9 ms | rows |
  | attention, up to 4,096 positions | | 0.74 ms | 6.3 ms | rows x positions |
  | Gated DeltaNet in projections and input norm | 4.03 GB | 3.14 ms | 12.7 ms | rows |
  | Gated DeltaNet state and convolution pieces | | 1.35 ms | 4.5 ms | rows |
  | `k_slice_unsqueeze_reduce_f670fb` | | | 2.9 ms | neither: also 2.9 ms at 16 rows |
  | Gated DeltaNet out projection | 1.51 GB | 1.03 ms | 1.7 ms | weight read, then rows |
  | MLP gate and up, Gated DeltaNet layers | 4.81 GB | 3.36 ms | 3.6 ms | weight read |
  | MLP down, Gated DeltaNet layers | 2.41 GB | 1.68 ms | 2.7 ms | weight read, then rows |
  | attention layers: q | 1.01 GB | 1.10 ms | 2.1 ms | rows |
  | attention layers: MLP gate and up | 1.60 GB | 1.07 ms | 1.2 ms | weight read |
  | attention layers: MLP down | 0.80 GB | 0.55 ms | 1.0 ms | weight read, then rows |
  | attention layers: k, v, o | 0.67 GB | 0.98 ms | 1.2 ms | weight read |
  | the rest | | 1.8 ms | 2.8 ms | |

  One read of all 16.8 GB of the trunk's weights is 11.8 ms, whatever the width.
- **The floor at a wider chunk.** Without rotary and attention the trunk takes 36.4 ms at 64 rows and 15.6 ms at one row
  without the head: 15.3 ms a chunk plus 0.33 ms a row. Two points on different programs, so an estimate, and no program
  above 64 rows was measured; the verify step sits above that line (19.7 against 16.3 ms, its head left out).

  | rows | trunk | rotary at today's rate | attention, up to 4,096 positions | chunk | per token |
  |---|---|---|---|---|---|
  | 64 | 36.4 ms | 11.9 ms | 6.3 ms | 54.6 ms | 0.85 ms |
  | 256 | 100 ms | 48 ms | 25 ms | 173 ms | 0.68 ms |
  | 512 | 184 ms | 95 ms | 50 ms | 330 ms | 0.64 ms |

  The attention term does not shrink with width: 1.09 s of a 7,680-token prompt at any width. A 7,680-token prompt at
  512 rows with no rotary cost at all is 15 x 184 ms + 1.09 s = 3.85 s; stock's whole first token there is 0.63 s. Wider
  chunks stop paying near 256 rows, because the row term is what is left.
- **A prompt's tail.** A chunk takes only prompt tokens before the last, and the tail runs in 16-row chunks, then one
  row at a time. A 256-token prompt is 3 chunks of 64, 3 of 16 and 16 single steps; 257 is 4 chunks and one step; 268 is
  4 chunks and 12 steps. The 16-row chunk costs 121.6 ms at 7,600 positions against 54-60 ms for 64 rows, and 1.91 s at
  197,600: 45% of it is attention at 7,600 and 96% at 197,600.
- **Long prompts.** Native's first token at the two long prompts is 38-42% slower than on 2026-10-25, and the plugin's
  and stock's did not move. At 197,632 tokens the 64-row chunks take 692 s: attention 544 s, all else 149 s (48 ms a
  chunk, flat). Attention grows linearly from 6.3 ms a chunk to 344 ms, 6.7 ms a layer at 65,536 positions against the
  4.6 ms measured before. The kernel's rows are the golden's, so the non-strict fork is not the cause. The commit that
  moved it was not bisected.
- **Accuracy.** The Emmy lanes score within one standard error of each other; stock 0.12-0.14 lower, as before.

## The recipe's native and plugin lanes on one RTX 5090 (2026-10-31)

Both RTX 5090 lanes of `recipe.yaml`, deployed and benchmarked by `emmy deploy local` and `emmy bench --local` on one
rented RTX 5090 (instance type `rtx59-3-24-750-xam.1`: 3 cores, 23.5 GiB of RAM, swap on), one session. The plugin lane
runs `cloudriftai/vllm-emmy:0.25.1-3454735b`, built on the card with `make vllm-emmy-image VLLM_VERSION=v0.25.1`. The
native lane runs `cloudriftai/emmy-native-qwen3.8-27b-nvfp4:0.3.23-6732a701`, baked on the card by `make native-image`:
its pack was prepared at `b60e2b02` (main at `35100514c` and this branch's drafter rows) inside the plugin image, from
the recipe golden alone under strict evidence, and the image tag's commit changes only the script that splits the pack
into image layers. Neither image is published. The
benchmark is the recipe's: 10 random-token prompts of 256 tokens, 256 output tokens with `ignore_eos`, temperature 0,
one request at a time, vLLM 0.25.1's client. Random tokens draw repetitive answers, which the drafter predicts well,
so these TPOTs sit below the ones of the tables further down; compare the two columns, not this table with those.

| | native | plugin |
|---|---|---|
| TPOT, median / P99 | 8.66 / 10.6 ms | 10.30 / 12.6 ms |
| TTFT, median / P99 | 721 / 723 ms | 198 / 361 ms |
| output tokens a second | 86.9 | 87.7 |
| tokens in / out | 2,482 / 2,560 | 2,482 / 2,560 |
| drafts accepted | the client reports none for the native server | 78%, 2.56 tokens a step |
| device memory at rest | 30.7 GiB | 30.7 GiB |
| host memory of the server at rest | 3.5 GiB | 15.8 GiB after a cold boot, 18.4 GiB after a warm one |
| ready after | 24-27 s, nothing compiled | 3,030 s cold, 437 s from its cubin cache |

Notes:

- **No compile at the native boot.** Both native boots load the pack and run the smoke request (24 s and 27 s, 0.6-0.8
  s of it the request); the server log names no compile.
- **The plugin's caches.** The bench's plugin boot took 2,927 s, a cold boot: the lane kept its cubins and packs under
  `/mnt/models`, which only `emmy deploy local` mounts, and `emmy bench` mounts `/hf_models`. The recipe now keeps them
  under `/hf_models`.
- **Prefill.** Both lanes run 64-row prompt chunks; native's first token comes 0.5 s after the plugin's at 256 tokens.
- **Memory.** Device memory is the card's total after the boot (`nvidia-smi`), host memory the server container's
  (`docker stats`), both at rest; peaks under load were not taken.

## Native and NInfer on one RTX 5090 (2026-10-29)

[NInfer](https://github.com/Neroued/ninfer) at `1882c6ad` is a C++/CUDA engine for this card that ships its own artifact
of Qwen3.8-27B. Built from its Dockerfile (CUDA 13.1.2, `sm_120a`) and run on the rented x16 RTX 5090 that measured the
Qwen3.6 recipe's table (driver 580, CUDA 13.0: the container runs with `NVIDIA_DISABLE_REQUIRE=1` and the host's
`libcuda` ahead of the image's forward-compatibility one, which a GeForce card refuses). Every cell below comes from one
session on that card. Both columns serve one request at a time with the FP8 KV cache, the 262,144-token window, the MTP
drafter at two tokens with the checkpoint's own head, images on, thinking off and no prefix cache, and get the same
prompts from the same client as the table below, with three differences NInfer forces: it serves only
`/v1/chat/completions`, so every prompt is one user message (268, 7,692 and 261,644 tokens with the template; native
reads the same message rendered by the checkpoint template from `/v1/completions`, and both servers report the same
prompt length); it has no `ignore_eos`, so neither side gets it and an answer ends at the model's end of turn or at 256
tokens; and GSM8K goes through the chat endpoint, the few-shot examples as turns. Two places in vLLM 0.25.1's client
were patched for NInfer's stream, which opens with an empty role chunk and sends `: keep-alive` comments every 5 s of a
long prefill: the first token is the first chunk with content, and comment lines are dropped (kept, the first one hid
every later chunk). Native's completions stream has neither.

| | native | NInfer |
|---|---|---|
| weights | NVIDIA's NVFP4 checkpoint (NVFP4 and FP8 linears) | unsloth's NVFP4 checkpoint as NInfer's artifact: NVFP4 MLPs in layers 0-55; row-scaled FP8 attention, Gated DeltaNet projections, last eight MLPs, head and embedding |
| TPOT / TTFT, 256-token prompt | 13.2 ms / 0.45 s | 6.3 ms / 0.050 s |
| TPOT / TTFT, 7,680-token prompt | 13.6 ms / 6.98 s | 6.5 ms / 0.62 s |
| TPOT / TTFT, 261,632-token prompt | 35.7 ms / 825 s (20 tokens) | 8.9 ms / 62.5 s (52 tokens) |
| drafts accepted at 256 / 7,680 / 261,632 | 73 / 71 / 75% | 73 / 73 / 68% |
| NInfer's own setting (three drafts, its smaller draft head): TPOT / TTFT and acceptance at 256 / 7,680 | | 5.3 ms / 0.050 s, 5.4 ms / 0.63 s; 62 / 64% of three |
| device memory at rest / peak | 30.1 / 30.6 GiB | 30.8 / 30.8 GiB (19.7 GiB of weights) |
| KV cache | 262,144 tokens, the window | 278,788 tokens, what the card has left |
| host memory of the server at rest, peak | 3.4 GiB (3.1 GiB pinned: the token-embedding table and the tower), 4.9 GiB | 2.4 GiB, 2.4 GiB |
| GSM8K strict / flexible match, chat | 0.905 / 0.905 | 0.925 / 0.935 |
| ready after | 20 s from its pack | 9-10 s |

Acceptance is the client's rule of the table below for both columns; NInfer's own counters agree within 1 point at 256
and 7,680 tokens. GSM8K's standard error is 0.019. The native column is the pack of the table below (main at
`b5dddf26c`, before #1282), exported again on this card from the recipe golden under strict evidence; its prompts run in
64-row chunks, NInfer's in 1,024-token chunks.

Where NInfer's time goes that native's does not:

- **Decode.** Both sides accept the same share of drafts, 2.45 tokens a step, so the gap is the three-row verify step:
  15.5 ms on NInfer against 32.3 ms on native (TPOT times tokens a step). #1282's profile put two thirds of native's
  step in the width-3 Gated DeltaNet program and took the step to 24.8 ms with re-recorded rows and a fixed-width
  prologue; this pack predates them. NInfer's own setting drafts three tokens from its smaller head and gains another
  17%.
- **Decode at the longest prompt.** Native's step grows to 89 ms at 261,644 positions, NInfer's to 21 ms. Native's
  attention reads each key-value head's cache once per query head.
- **Prefill.** NInfer prefills 12,300 tokens a second at 7,692 tokens and 4,200 at 261,644; native 1,100 and 317.
- **Accuracy.** The two score within one standard error of each other on GSM8K through the chat template.

## Native, plugin and stock vLLM on one RTX 5090 (2026-10-25)

One rented x16 RTX 5090 (14 cores, 23.5 GiB of RAM, swap on), one session: every cell below comes from the same card on
the same day. All three columns serve the recipe's checkpoint at its pinned revision one request at a time, with the
FP8 KV cache, the MTP drafter proposing two tokens a step, images on (one per prompt, at the recipe's bounds) and no
prefix cache. Every column gets the same text: slices of this repository's Markdown files, cut to exact token counts
with the checkpoint's tokenizer and sent raw to `/v1/completions` (no chat template, so no thinking) by `vllm bench
serve` (vLLM 0.25.1's client, `--dataset-name custom --skip-chat-template`), 256 output tokens with `ignore_eos` at
temperature 0. Medians over 10 requests at 256 prompt tokens and 4 at 7,680, one request at each long prompt. The
random-token prompts `emmy serve --bench` sends were measured first and dropped: their answers fall into repetition,
which the drafters accept 80-98% of the time, or at 261,632 tokens into more random tokens, of which both Emmy lanes
accept 36%.

| | native | plugin | stock vLLM |
|---|---|---|---|
| TPOT / TTFT, 256-token prompt | 13.7 ms / 1.34 s | 13.5 ms / 0.23 s | 7.7 ms / 0.12 s |
| TPOT / TTFT, 7,680-token prompt | 14.3 ms / 7.78 s | 14.5 ms / 5.31 s | 8.2 ms / 0.64 s |
| TPOT / TTFT, 197,632-token prompt (stock's longest) | 32.3 ms / 505 s | 17.3 ms / 184 s | 8.7 ms / 67.9 s |
| TPOT / TTFT, 261,632-token prompt (the window's longest) | 41.7 ms / 815 s | 16.0 ms / 266 s | does not fit |
| drafts accepted at 256 / 7,680 / 197,632 / 261,632 | 68 / 66 / 76 / 66% | 73 / 62 / 62 / 77% | 65 / 60 / 77% / - |
| TPOT with the drafter off, 256 / 7,680 | 18.3 / 18.7 ms | | |
| device memory at rest / peak | 30.1 / 30.6 GiB | 29.9 / 30.7 GiB | 30.4 / 30.9 GiB |
| KV cache | 262,144 tokens, the window | 305,585 tokens at 0.93 | 199,891 tokens at 0.93 |
| host memory of the server at rest, peak | 3.4 GiB, 4.8 GiB | 10.8 GiB (1.3 swapped), engine peak 9.6 GiB | 5.1 GiB (0.7 swapped), engine peak 10.9 GiB at load |
| GSM8K strict / flexible match | 0.710 / 0.770 | 0.735 / 0.760 | 0.580 / 0.635 |
| ready after | 20 s from its pack | 236 s from its pack | 181 s |

Acceptance is what the client sees: a step's tokens arrive within a millisecond of each other, so a longer gap starts a
step. vLLM's own counters agree with it within 2.5 points on both vLLM columns. GSM8K is
`scripts/run_lmeval_gate.py --num-concurrent 1`: 200 questions, 5-shot, seed 0, raw completions, the protocol of
`recipes/Qwen3.8-27B-NVFP4`; its standard error is 0.03. Host memory is the server processes' resident set and swapped
pages; native's 3.4 GiB hold 3.1 GiB of pinned pages (the token-embedding table and the vision tower's blocks).

The columns:

- **native**: `emmy-server` from the main branch after #1275 (`b5dddf26c`) over milestone 12's generation-format-13
  pack, exported from `golden/rtx5090_sm120.json` alone under strict evidence with images, the drafter, the FP8 cache
  and the 262,144-token window. Emmy compiles every program, attention included; a prompt runs in 64-row chunks. The
  pack predates #1282, whose golden rows and fixed-width prologue programs take native drafted decode from 12.1 to
  9.2 ms a token on its probes; the plugin and stock columns do not depend on it.
- **plugin**: `emmy serve --runner generate` in `vllm/vllm-openai:v0.25.1` with the same source installed, the recipe's
  engine block (fast math, strict evidence from the golden, `EMMY_GEN_EMBED_GPU=0`, `EMMY_GEN_VISION_GPU=1`,
  `--max-num-batched-tokens 64`, full decode graphs at sizes 3, 6 and 15) plus `--max-num-seqs 1 --max-model-len
  262144 --gpu-memory-utilization 0.93`. Emmy runs every decoder layer, the MTP head, the coded output head and the
  tower; vLLM schedules, drafts and runs FlashInfer attention over its cache.
- **stock**: `vllm/vllm-openai:v0.32.0`, the newest release (`sha256:befccf0e…`, the image `recipes/Qwen3.8-27B-NVFP4`
  pins), with `--max-num-seqs 1 --max-model-len 198400 --kv-cache-dtype fp8 --no-enable-prefix-caching
  --speculative-config '{"method":"mtp","num_speculative_tokens":2}' --gpu-memory-utilization 0.93` and the recipe's
  `--limit-mm-per-prompt` and `--mm-processor-kwargs`; the rest is vLLM's default: `FlashInferCutlassNvFp4LinearKernel`
  for the NVFP4 linears, `FlashInferFP8ScaledMMLinearKernel` for the FP8 ones, FlashInfer's Gated DeltaNet prefill
  kernel, FlashInfer attention (XQA decode over the FP8 cache), its own `Qwen3_5MTP` drafter at two tokens,
  `FULL_AND_PIECEWISE` CUDA graphs under torch.compile, and chunked prefill at 2,048 tokens a step.

What could not match:

- **The window.** Stock vLLM needs 8.98 GiB of KV cache for one 262,144-token request and has 6.99 GiB at 0.93
  utilization: it counts 20.64 GiB of weights, the 2.37 GiB token-embedding table among them, and a 2.6-2.7 GiB peak in
  its profiling pass. Text only at 0.97 it has 6.96 GiB (7.07 GiB with 512-token chunks), so it refuses the window
  however it is set, and serves at most 198,400 tokens. The 197,632-token row compares all three; the 261,632-token row
  is the two Emmy lanes alone. Both Emmy lanes keep the table in host memory.
- **The kernels.** Each side runs its own. The plugin and native run the same trunk programs, the recipe golden's
  twins; native runs its own attention, rotary and selection programs too.
- **The prefill chunk.** 64 tokens on both Emmy lanes, the width their programs are recorded at; 2,048 on stock.

Where the time goes:

- **Decode.** With two drafts a step is three rows. That step takes 32-33 ms on both Emmy lanes at the short prompts
  (TPOT times tokens a step) against stock's 18 ms, which is what native's one-row step takes with the drafter off
  (18.3 ms). It is the same programs on both Emmy lanes. #1282's profile of native's step put two thirds of it in the
  width-3 Gated DeltaNet program, whose pieces held the prior's untuned picks, and 1.1 ms in the drafter's any-width
  prologue; its re-recorded rows and fixed-width prologue take the step to 24.8 ms and drafted TPOT on its probes to
  9.2 ms.
- **Decode at long windows.** Native's step grows from 33 ms at 7,680 positions to 81 ms at 197,632 and 97 ms at
  261,632; the plugin's, with vLLM's attention, from 33 to 39-41 ms and stock's from 18 to 22 ms. Native's attention
  reads each key-value head's cache once per query head, six times at this model's grouping.
- **Prefill.** Native takes 1.01 s per 1,000 prompt tokens at 7,680, the plugin 0.69 s and stock 0.084 s. Both Emmy
  lanes prefill in 64-row chunks, about 55 ms a chunk for native and 44 ms for the plugin here; milestone 11 found
  native's extra cost in the rotary program's cache store, which runs on one CTA, and in attention at 64 rows, which
  writes its scores and probabilities to memory and grows with the window (milestone 17 items). Stock prefills 2,048
  tokens a step. At 256 tokens native's first token comes after 1.34 s with the drafter on or off: with carried Gated
  DeltaNet state a chunk takes only prompt tokens before the last, so a 256-token prompt runs three chunks and its last
  64 tokens one at a time through the decode program. A 257-token prompt, whose tokens before the last fill four
  chunks, gets its first token after 246 ms (the plugin 230 ms at 256).
- **Accuracy.** The two Emmy lanes score within one standard error of each other on GSM8K; stock scores 0.13-0.16 lower
  on the same prompts and client.
