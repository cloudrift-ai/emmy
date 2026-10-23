# Qwen3.8 NVFP4 on the RTX 5090: tuning plan

Goal: strict serving of `recipes/Qwen3.8-27B-NVFP4` boots from the golden (`emmy eval golden` reports 9 of 9 twins
in the fast-math lane), then end-to-end decode faster than stock vLLM.

## State

- Done: the RTX 5090 golden deploys all 9 fast-math twins under strict evidence, and a strict boot serves. Final boot
  (2026-10-20): about 56 tok/s steady decode; TTFT 0.17 / 0.25 / 0.74 s on the short (warm) / medium / long prompts.
  Stock vLLM 0.23.0, same shape: 63.2 tok/s, TTFT 0.08 / 0.30 / 1.24 s. GSM8K (200 questions, 5-shot): emmy 0.450,
  stock 0.470, within one standard error (0.035).
- 2026-10-23 boot (the GDN decode recurrence as one kernel, `DISCOVERY.md`): 58.5-59.2 tok/s decode, TTFT
  0.15 / 0.22 / 0.64 s.
- Not done: decode is still slower than stock vLLM. The standard lane (`FAST_MATH=false`) deploys 1 of 9 twins
  (gdn1); serving runs the fast-math lane (`SERVE_ENV="EMMY_FAST_MATH=1"`), so no standard-lane rows were recorded.

## Open follow-ups

- Decode gap to stock vLLM (56 vs 63 tok/s). The largest known pieces:
  - Decode weight readers stream DRAM at 1.66-1.73 TB/s while all CTAs run (cuBLAS 1.74 TB/s); the loss is a ramp
    at launch start and an uneven tail at launch end, a few us per launch. No offered WORK/TILE/STAGE schedule
    closes it (gate_up 67.8-70.5 us across CTA counts and pipeline depths).
  - Measured, no gain (#1219, closed): loading each kernel's first weight stages before `griddepcontrol.wait` and
    releasing the next kernel at entry. Strict boot 55.4 tok/s against 55.9 on main; per captured program gdn1 271.4
    against 262.0 us, pre16 and post16 within 1 us. The weight loads gain nothing, and the early release costs about
    10 us per gdn1 program (also with the release moved after the kernel's own wait). A microbenchmark of cold
    back-to-back launches had predicted 0.4-0.5 ms per token. Handing out work tiles dynamically to shrink the tail
    (about 0.2 ms per token) is not prototyped.
  - Measured, no gain (#1220, closed): the GDN in_proj sibling weights (qkv, z, a, b) joined into one 5120x16480
    weight read by one launch. Its best tuned row is 112.9 us against 112.5 us for the three kernels in serving,
    because the joined producer gets no tensor-core schedule: its guarded store to each weight's own workspace
    leaves no MMA atom offered, and 16480 is not a multiple of 64 or 128, so cp.async/TMA fills refuse the masked
    N tile. cuBLAS bounds the gain at about 8-12 us per layer (104.4 us joined against 116.7 apart, cold). To
    revisit: an MMA tile for the joined producer, a padded constant or masked-N fills, and a route of its own
    (after #1206 the joined root forms `map.1/map.1/map.1/reduce.1`, so the apart route no longer spells it).
- Qwen3.8 FP8 on the V100: its GDN rows (gdn16 `13e5d6`, gdn64 `353a47`) were restamped but not re-recorded or
  tuned after #1171.
- Unrecorded GDN routes after #1230's restamp (no card measured; the coordinate re-spelling moved these kernels):
  - Qwen3.8 FP8 A100: 11 decisions dropped.
  - Qwen3.8 FP8 H100: 26 decisions dropped.
  - Qwen3.8 FP8 RTX 4090: 16 decisions dropped.
  - Qwen3.8 FP8 V100: 17 decisions dropped.
  - Qwen3.6 FP8 RTX 5090: 20 decisions dropped.
  - Gemma 4 RTX 5090: no decision dropped, 96 rows demoted to proposals.

  A strict boot of those recipes on those cards stops at a fork with no evidence until a record run on that card
  measures it.
- Rolling a recurrence misfires at small shapes: with 6 value heads and 8 or 12 tokens, `beta.transpose(1, 2).float()`
  (a bf16 transpose, then an f32 cast) is rolled by `005_roll_recurrence` into a 2-step "recurrence"
  (`_to_copy_5_cast__steps0`, shape `[2, 1, 6, T]`), which leaves 3 extra kernels beside the fused conv parent. At the
  real shape (48 heads, 64 tokens) the parent stays one kernel. Repro:
  `tmp-claude/gdn-recurrence/final/caseB/conv3.py` (traced: `conv3.json`); casting before the transpose (`conv5.py`)
  avoids it. Not investigated.
- In-place state read: with one kernel the only reader of the GDN `state`, read the old state from the cache block and
  drop the gather (`index_select`): about 0.22 ms per token (Oct 20 profile, state and history together). Needs its
  own legality rule: each CTA reads exactly the cells it writes, all reads before the first write, no cross-CTA split
  of `k`. History stays gathered, because the conv shift reads neighbouring cells.
- GDN serving copies: since #1215 the programs write the new state straight into the cache block, with no scatter
  back, and a step that runs one request alone fills every layer's page table with one launch and, under capture,
  passes hidden rows between layers without copies. What remains is the gather the in-place state read above
  removes, and on verify steps (speculative decoding) the scatter of `1+k` states back with `index_copy_`
  (`forward_layer_gdn_rows_device`). The per-layer `state_indices_tensor.tolist()` reads are gone since #1209.
- gdn64 recurrence target `02d4dd`: 122.8 us on #1230's golden against 120.8 on main (+2 us), after re-recording its
  route on the re-spelled tree and tuning the new pieces. Not investigated.
- The joined cut is measured on gdn1 only. gdn16/gdn64 compute the recurrence in the chunked form, with other
  contractions and pieces; nothing joins them yet. No joined cut puts the out_proj epilogue into its FP4 MMA piece
  (the epilogue region contains the contraction); the out_proj join of the statistic reduce and the residual region
  measured 16.6 us against 17.7 for the recorded route (not recorded).
- Nested `TILE`/`STAGE` stamps dropped under an output-tiled root: in the gdn16 recurrence piece
  `k_linear_matmul_mean_reduce_…__place_84c2d13867__place_149fbfea8c__partial`, picks stamp `TILE` on the two nested
  contractions and `STAGE=d2/smem-tma` on one, yet the CUDA runs those folds as serial per-cell loops with no shared
  memory. The binder accepts the stamps silently. Likely fix: offer a nested tile only where the binder realizes it,
  refuse the rest at the offer.
- Restamp does not re-spell a joined key (`PLACE@<a>+<b>`): `golden/restamp._respelled` treats a key as one site path
  and tries only the four nearest candidate keys. Observed on the corpus case `fused/gdn-conv-join-in-place-name.json`
  when #1243's coalescer reordered the parent's regions: both halves moved two path segments, four other keys ranked
  ahead, and the restamp dropped the route (the realization test reports the decision as not taken). The fork still
  offered the arm under the moved spelling, and the case was re-keyed by hand. The gdn1 route of this golden stayed
  in place because the coalescer moves no site on its parent.
- 10 serving GPU tests (`test_gen_pack_gpu`, `test_gen_runner_gpu`, `test_attention_split_gpu`) stop with
  `EvidenceError: strict evidence: kernel 'k_linear_mean_reduce_af84cd' (node 'view_3') has no measured evidence for
  its 030_cut fork`. One of them, re-run on main fd9633b9, fails the same way; the cause is not investigated.
- Standard lane: record the 8 missing twins if serving ever needs `FAST_MATH=false`.
- Search policy: the prior or the greedy fallback can pick a single-CTA `Tile[]` arm over a large contraction (gdn64
  `k_linear_matmul_mean_reduce_73983d__place_68111979ee`: one thread for a 64x6144x5120 linear plus the recurrence,
  38.9 s per call; gdn16 `d29945__place_54138af7d7`, same shape). A non-strict boot then takes hours. No guard
  rejects such arms or prices them by work (about 20-50 lines plus a test; not done).
- Tuning and recording never measure `SHARED_CARRY=1` for serial (carried-loop) kernels: `--tune` and
  `--record-greedy` left the 61-launch form of the GDN prefill triangular solve `3d261f` in the golden (572 us against
  46 us for one launch). The shared-carry rows are hand-recorded. Other carried loops may need the same check.
- Stale measured rows that greedy trusts: the `SHARED_CARRY=0` rows of `3d261f` say 218.5 / 217.2 us, but a re-bench
  on 2026-10-19 gave 572-574 us. Re-bench or re-record rows whose kernels changed launch shape since they were
  measured.
- Compiler bug (wrong answer, not fixed): the bf16 MMA schedule `WORK=w1x4, TILE=mma_m16n8k16_bf16_f32/f1x1/k8,
  STAGE=d4/smem-tma` (192 CTAs, 48 KiB shared memory) on gdn16 `db4f79` piece `place_6a3e623a98` gives one wrong FP4
  code (flat index 86480 of `view_16_static_fp4_bits`: -6 vs -3) and its block scale. Changing any single knob gives
  output bit-identical to the scalar schedule. The prior picks this schedule greedily; the golden pins `TILE=f1x2/k8`
  there instead (15.2 us). The non-strict wrong-answer check misses it: its quantized threshold is a mean error above
  0.5% of the peak, and one bad element in 98304 stays under it.
- Compiler bug (performance, worked around in the golden): `_cut._abstracted` numbers a contraction's free-index
  placeholders in operand order, by first appearance in the lowered body. Under the block amax the weight operand
  comes first, elsewhere the activation does, so `_group_contractions` splits one contraction into two groups at M>1
  and the amax occurrences keep their own copy of the contraction (the GDN out_proj + rmsnorm + FP4 quant kernels
  recomputed out_proj as a scalar loop, 9-10 ms per layer). The golden cuts the second contraction group
  (`PLACE@map.2/map.2/reduce.1/inner`) and splits the fused sibling producer, at the cost of a second out_proj MMA
  per layer (about 20 us gdn64, 11 us gdn16). Fix: group by an operand-order-free numbering and lay the workspace out
  in the first member's order (about 40-60 lines); it would save about 1.2 ms per 64-token chunk.
- Strict rule, known loose spot: a float output's rounding step comes from the coarsest float format its producing
  chain computes in, and the rule has no list of which ops round and which only move data. So a bf16 slice cast to
  f32 and followed by f32-only math still gets the bf16 step, looser than the f32 math needs. The mean-error rule
  still applies to such outputs.
- `roofline.py` `time_program_us` reports a single cold first run when that run exceeds 10x the floor. In the full
  64-layer boot pre16 and post16 showed 4.26 and 3.88 ms against 44-51 and 134 us steady state. The cause (first
  touch or placement in the full boot) is not investigated.
- Rows of neighbouring schedules that passed strict and are worth re-checking: gdn64 `...5f1430.m64.8105a48e7b21`
  (`w1x8`, bf16 `f1x1/k8`, `d4/smem-tma`), pre16 `...a7efbb.m16.fm.83c0b7f74b51` (FP4 tile, `w1x4`, `d4/smem-tma`),
  gdn16 `0e5142` `...8105a48e7b21` (`d3/smem-tma`).

- Serving test golden regen: `python -m tests.serving.regen` does not cover the float16 hybrid of
  `test_vllm_plugin_gdn_gpu.py`. #1243 recorded its three GDN programs (widths 1, 4, 16) in `serving_sm120` outside
  the regen, with `PLACE` pins; a full regen drops those rows and the prior alone picks the slow kernels again. Add
  the hybrid to `RUNNERS`, with its pins, or teach the regen to keep rows it does not recapture.
- Dead scratch at width 4: one pin combination on the tiny hybrid's width-4 GDN program made the cut pass create a
  piece whose output nothing reads, and the runtime refused that buffer ("dead scratch"). #1243 dropped that pin
  instead of fixing the cut pass.
- The prior skips the small GDN programs' cuts: since the placement prior's refit in #1217 it no longer offers the
  cuts the tiny hybrid's GDN programs need. Unpinned, each width keeps one kernel that runs as a single thread block
  and recomputes the chunk's triangular solve per output element.

## The whole model on one card

Goal (2026-10-23): serve NVIDIA's `Qwen3.8-27B-NVFP4` checkpoint on one RTX 5090 with the MTP drafter on, the vision
tower answering image requests, and the full context window, inside the card's 32 GB. #1241 serves it text-only with
MTP off at 31.2 GiB after graph capture; the input embedding and the output head take 2.37 GiB each on the device,
the head decoded to BF16 from 0.67 GiB of NVFP4 codes. The MTP drafter's tensors are 0.79 GiB and the vision tower's
0.86 GiB in the checkpoint. The milestones below free about 4 GiB and spend it on those two; each is one PR, stacked
on the one before it. The dev box (RTX 5090 on a PCIe x1 link, 60 GB RAM) is for the gates that need no link speed;
the numbers come from a rented x16 5090 host (23 GB RAM, so the plugin boots with few compile workers).

1. **The head stays coded.** The NVFP4 head runs compressed through the compiled head program the EXL3 head already
   has (`emmy/serving/coded_head.py`: the coded weight is the program's input, FP32 logits, the logits processor
   applies scale and softcap), instead of decoding into vLLM's BF16 parameter. Gate: logits equal to the decoded
   head's on the probe prompts within FP32 rounding, device usage 1.7 GiB lower after capture, TPOT unchanged.
   Needs #1241.

   State (2026-10-23, #1247): done, measured on the rented x16 RTX 5090. The head keeps 0.668 GiB of codes and
   scales on the card and runs as one compiled program in the `lm_head` slot, so the MTP drafter shares it. At the
   baseline's KV cache size, device usage after capture is 29.47 GiB against 31.18 GiB (1.71 GiB lower); at the
   default 0.97 utilization the KV cache holds 333,496 tokens against 279,206. The four probe prompts give the same
   256 greedy tokens. The program's logits equal the decoded BF16 weights' within f32 rounding on the real head
   (7.6e-5 at logits up to 11.8); vLLM's decoded head rounds its logits to BF16, which moves served logprobs by up to
   0.16. On #1241's trunk TPOT is 26.6 ms against 27.5 ms: the head program takes 0.70 ms where vLLM's BF16 matmul
   takes 1.63 ms. The head's measured row lives in `recipes/Qwen3.8-27B-NVFP4-NVIDIA` (the `head-sym` twin), so every
   RTX 5090 boot picks it; without a row the prior picks a bf16 MMA tile that takes 7.9 ms. Found on the way: `emmy
   db import` of one freeze file files its rows as `golden:`, and a compile drops every `golden:` row before it
   imports the golden scope, so #1241's prefill rows never reach a compile imported that way (TTFT 5.3 s instead of
   0.25 s for 256 tokens). Import the directory holding the freeze files instead.

2. **The embedding table in mapped host memory.** The runner's `EMMY_GEN_EMBED_HOST` keeps the table pinned on the
   host and gathers rows over the link inside the captured step; make the recipe use it and check the MTP drafter's
   shared table follows (the drafter gathers from the runner's table). Gate: the same tokens, 2.37 GiB lower, and
   the TPOT cost of the gather measured on the x16 host (about 10 KB per token, so it should not show).

   State (2026-10-23, #1251): done, measured on the rented x16 RTX 5090 on #1247's code before its review fixes
   (f4328389b); no code changed. NVIDIA's checkpoint has no recipe yet, so the recipe that milestone 4 adds sets the
   switch. With `EMMY_GEN_EMBED_HOST=1`, at the baseline's KV cache size, device usage after capture is 27.11 GiB
   against 29.47 GiB (2.36 GiB lower); at 0.97 utilization the KV cache holds 393,991 tokens against 333,496. The
   four probe prompts give the same 256 greedy tokens with the same logprobs. A captured gather of 1 or 16 rows takes
   about 4 us more than from device memory, one of 64 rows about 12 us more, and TPOT does not move: 27.17 and
   27.24 ms against 27.28 and 27.29 ms at the same KV size, 26.91 and 26.70 ms against 27.17 and 27.09 ms at 0.97.
   Boots of one setup differ by up to 0.6 ms of TPOT (#1247 measured 26.60 ms on the same code and pack), so compare
   TPOT inside one session. The MTP drafter follows the table: vLLM's proposer shares the target's `embed_tokens`,
   whose weight is the runner's table wherever it lives. A boot with the drafter waits for milestone 4, since
   vLLM 0.23 refuses speculative decoding on Emmy's GDN layers. Found on the way: torch's pinned allocator rounds the
   2.37 GiB table up to 4 GiB, so the host holds 1.63 GiB more than with a device table (9.4 GiB of the host's
   23 GiB in use after boot, against 7.7 GiB); the expert pager pins its pages through the same allocator.

3. **The vision tower on demand.** Its weights live in pinned, mapped host memory like the embedding table and are
   read over the link only when an image prefill runs, so the tower costs no device memory at rest; a text request
   never touches them. The mapped read is the first form because an image prefill reads the tower once, about
   0.86 GiB, which an x16 link moves in tens of milliseconds; the pager's device slots are the fallback if that read
   proves too slow per image, with the tower's weights as pages the image prefill touches. Gate: an image request
   answers correctly, device usage at rest unchanged, text TPOT unchanged.

   State (2026-10-23, #1252): done, measured on the rented x16 RTX 5090 on #1251's code before its rebase onto main.
   `emmy serve` boots a Qwen3.5 / Qwen3.8 vision-language checkpoint as `EmmyGenQwenVLModel` (vLLM's own model with
   the Emmy trunk as its language model) unless `--language-model-only` is passed. The mapped read is too slow: a
   matmul reading its weight in place reads it again for every tile of rows, and a 1,024-token image took 1.40 s of
   tower time (0.65 s for 196 tokens). The kept form copies each block's weights to the device right before the
   block runs: 55.0 and 26.1 ms, against 37.5 and 10.7 ms with the tower on the device, so the pager's slots are
   not needed. At the baseline's KV cache size, device usage after capture is 27.11 GiB, as for the text-only boot
   (27.10 GiB; 27.99 GiB with the tower on the device). Tokens and logprobs are the same in all three placements,
   and the text probes give the text-only class's tokens. Text TPOT is 23.48 and 23.49 ms against 23.40 and 23.44 ms
   for the text-only class in the same session; this host's pack decodes faster than #1251's. Against stock vLLM
   0.23 on the same images and prompts, both answers name the three shapes and their colours. On the text image
   stock reads "EMMY 4207" and Emmy "EMMY4907": stock's own logprobs tie '2' and '9' (-1.157 each), Emmy's put '9'
   0.67 above '2'. Text prompts differ from stock as much (175 of 256 probe tokens), so the gap is the trunk's
   numerics, not the image path. Found on the way: vLLM's encoder cache answers a repeated image without running
   the tower, so only an image's first request measures it; and this host's pack lacks #1241's prefill rows, so
   every prefill here is slow (4.2 s to the first of 256 tokens, against 0.25 s in #1251).

4. **The MTP drafter resident, the whole window.** With the memory the first three milestones free, boot with the
   drafter (`--speculative-config` MTP, two draft tokens as the Qwen3.6 recipe does), the vision tower and the
   recipe's context window, and record the verify programs' rows into the golden so the boot runs strict. The
   drafter runs every step, so it stays on the device: on demand it would move 0.79 GiB per token over the link.
   Gate: a healthy boot under 32 GiB after capture, greedy tokens equal to the no-MTP boot's, TPOT and acceptance
   rate against the no-MTP boot, a recipe for NVIDIA's checkpoint with the final flags, and the full-window request
   #1241 ran still completing. The coded head takes a verify step's rows in one launch, but its measured schedule
   gives each row its own blocks and reads the codes once per row: 2.12 ms for 3 rows against 1.53 ms for vLLM's BF16
   matmul (0.70 against 1.63 ms for one row, which each draft step samples). A schedule that keeps the rows of a
   vocab entry in one block would read the codes once.

Milestone 2 does not need 1; 3 needs 2's mapped-memory form; 4 needs 1, 2 and 3.

## Decisions

- Trace reference: transformers 5.14.1. `pyproject.toml` pins 5.19 for the compile extras, the serving venv
  (vLLM 0.23.0) runs 5.14.1, and the golden programs were traced under 5.14.1. Under 5.19 the GDN layer traces
  differently (q/k norm in f32 in another layout, a divide by sqrt(128)), so `emmy eval golden` from a 5.19 venv
  reports the GDN twins as broken although serving traces the stored programs. Serving is what has to beat vLLM, and
  a 5.19 re-trace re-keys about 260 GDN rows for no speed.
- Slower measured rows stay beside the faster ones, as in the canonical goldens (several measured rows for 89 of 253
  measured kernel+pins groups; every `.tuned-*` row keeps its base row). A strict compile picks the fastest matching
  row either way. Routing rows stay one per arm.
- Decode matmul rows carry only schedules that also win with a cold cache (see Gotchas: hot benches).
- NVFP4 decode tables stay as they are. On the tuned golden every NVFP4 linear of the 9 fast-math programs runs as
  W4A4 (4-bit weights and 4-bit activations) on the block-scaled fp4 MMA
  (`mma.sync...kind::mxf4nvf4.block_scale...e2m1.e2m1.f32.ue4m3`). TMA copies the packed weight and activation bytes
  into shared memory and the tensor core reads them unchanged; the activation encode uses ALU compares. No deployed
  kernel reads the 256x2 pair table (`*_f4_pairs`) or the `EMMY_F4_LUT_*` fragment table, and no scalar kernel reads
  NVFP4 data (the scalar weight readers in gdn1 read bf16 in_proj weights). A table-free decode (byte load, nibble
  extract, `cvt.rn.f16x2.e2m1x2`) therefore gains 0 ms/token. Every fp4-MMA kernel still takes the pair table as an
  unread parameter, a few bytes per launch. A table-free decode would matter only if a W4A16 schedule (4-bit weights,
  bf16 activations) wins a pick. CUDA 13.0, the recording toolchain, has only the f16x2 convert on sm_120a, so a bf16
  decode there takes an extra f16x2-to-bf16x2 step (exact for e2m1 values); CUDA 13.3 adds `cvt.rn.bf16x2.e2m1x2`.

## Gotchas

- A kernel launched early (programmatic dependent launch) must not read an earlier kernel's output through the
  read-only path: `griddepcontrol.wait` makes the earlier writes visible to coherent loads only, and inputs declared
  `const __restrict__` compile to `LDG.E.CONSTANT`. On #1219 that read a stale RMS-norm sum of squares (gdn1
  launches 16 and 17). Only program constants may keep `__restrict__` / `__ldg` under early release.
- `emmy golden check` never re-traces: "current" means the stored programs still lower to the stored kernels, not
  that a fresh trace of the model gives those programs.
- A strict record also benches the named realization's own stored row as a pinned config, and that row's schedule
  knobs replay as bare, unscoped pins on every kernel of the compile. Name a row whose knobs suit every piece of the
  route (the big piece's own row); a small piece's row can leave the big piece with no measured candidate or with a
  pick over the 10 s GPU bench budget.
- `--pin-route` on a target root's own row that has no latency pins nothing, so the fused kernel benches (a hung
  kernel shows up as `HungKernelError ... did not complete within 2000 ms`). Name one of the route's piece rows as
  `--realization`; the bench still compiles the whole target.
- `GoldenFile.path_to` takes the first route that reaches a piece. gdn16 `db4f79` shares pieces with gdn64 `02d4dd`,
  so a hand pin spelled along that path can name seams of the other root (`PLACE pin ... names no seam of any kernel
  in this compile`). Spell the pins along the target's own decisions.
- A piece with one CTA and a multi-ms or hung time has lost its split decision (a restamp can drop producer
  decisions). Cut its seams by hand with `PLACE@place_<token>/<seam>=cut`.
- `--record-greedy` spells every precision gate in the new rows' pins (`FAST_MATH`, `FAST_EXP`, `F16_MMA_F32_ACC`,
  `FP8_MMA`); the audit reads both spellings as one lane. The golden rows carry `{"FAST_MATH": lane}` alone.
- A whole-target bench now and then falls back from CUDA graph capture, and a strict record refuses it ("backend
  comparison did not use CUDA graph capture"). The capture that fails is the eager PyTorch backend's
  (`cudaErrorStreamCaptureInvalidated`); a rerun usually records.
- `run --kernel <cut FP4 piece>` fills the piece's FP4 decode tables (`*_f4_pairs`) with random values, so tile rows
  fail correctness there and `--tune` counts every tile row as a failure. Check candidates through the whole target
  (`--realization <target row> --pin-route --strict`).
- `benchmark run stage exceeded 10.0s of GPU time`: a multi-ms pick in the pinned or greedy set. Replace the pick
  with a kernel-scoped schedule pin, or for a record-only run use `--warmup 2 --iters 5`.
- `unreproducible pin: ... realized ...`: a hand schedule pin clashes with a pinned golden row of the same
  realization. Drop the hand pin before the strict record.
- `--cold-cache` bench times come in steps of about 2.05 us, too coarse to rank decode matmul schedules. Cold A/B
  runs go under nsys and read per-launch durations instead.
- Hot benches of weight readers keep the weights in the 96 MiB L2 of the 5090 (gate_up, 100 MB: 25 us hot, about
  4 TB/s; 68 us cold). A hot `--tune` can rank schedules that gain nothing cold (post16 gate_up hot winner
  `w1x2 f1x4/k4`: -10 us hot, 0 cold).
- After a main merge that re-keys pieces, a re-keyed piece can lose its only measured row, or keep a row whose
  schedule the re-formed piece no longer offers (`WORK=t512` then fails as "realized (off)"). Re-record it on an
  offered schedule.
