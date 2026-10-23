# Native serving of a MoE model with paged experts

Status: proposed 2026-10-22. Serve a MoE model whose routed experts do not fit the card through the native runtime
(`emmy serve MODEL --runner generate --native`): hot experts in device slots, every expert in pinned host memory,
the compiled paging programs moving them, as the vLLM plugin does today. The target is `Qwen/Qwen3.6-35B-A3B-FP8` on
one RTX 5090. This continues `plans/moe-host-experts.md`, whose milestone 7 built paging under the plugin; it adds
no IR, no fusion gate and no schedule rule.

## Where it stands

The plugin serves the target in hybrid mode. Measured 2026-10-22 on the dev box (RTX 5090 on a PCIe 5.0 x1 link),
128 of 256 experts per layer on the card, one request, plain one-token decode: median TPOT 49 ms, median TTFT 3.4 s
on 20-token prompts, boot roofline audit clean. The pieces the native path would reuse already exist:

- **The paging mechanism is in the runtime.** The pager (`crates/emmy-runtime/src/pager.rs`: declare, bind, table)
  owns pinned host pages, device slots and page tables; the artifact format carries a paged buffer's geometry and
  slot count; the backend splices the paging programs and each launch's inspector when a buffer is declared with
  fewer slots than pages. The plugin drives all of it through the Python bindings.
- **Native preparation already compiles paged buffers.** The KV cache is a paged buffer of one page; `_Step.compiled`
  takes the paging declaration, sets the hint before compile and refuses a program that reads a page through a TMA
  descriptor or takes runtime arguments.
- **The expert programs are traced and recorded.** The fixed-slot expert program (`ExpertFFN.slots`: one row's k
  routed experts gathered by index from the E-stacked coded weights, the decode and scale algebra spelled in-graph)
  is program 0 of the recipe golden, and its rows stage nothing through TMA, so they hold when the operands are
  paged. The GDN state wrapper's routed form returns the mixer's residual with the shared expert folded in and the
  normalized input the routed experts read.
- **The host-memory lesson is learned.** A paged boot pins each expert tensor in place of the loaded one (#1234);
  the host holds one copy.

What was missing is everything around those programs that the plugin still does in Python; milestone 0's state
says what is left.

## Design

A step program per width, as today. A MoE layer's section of it is five launches in sequence, every one static:

1. **Mixer.** The GDN wrapper's routed form, or the attention split's `post`, leaving `h` (residual with the shared
   expert folded in) and `xn` (the normalized row the routed experts read) in the FP32 seam.
2. **Router.** The gate's linear in the model dtype, the score function and the expert-selection bias in FP32 as
   Transformers keeps it, then top-k spelled as k successive masked max reductions (argmax already compiles as two
   max reductions), writing `sel` (k int64) and the normalized `scores` (1 × k). An alternative is the pager's own
   trick, a rank counted per element and inverted by a scatter; the first measured one wins.
3. **Experts.** The existing slot program over the layer's E-stacked inputs, declared paged `(name, 0, 1, slots)`
   when the budget is below the expert count. The backend splices the paging programs ahead of it, so the step
   carries them as ordinary launches; their tables and clock are persistent step buffers.
4. **Combine.** `scores @ partials`, in FP32 where the model asks for it, as one compiled program.
5. **Residual.** `h + combined`, folded into the next layer's input normalization as the dense path does.

Prefill chunks (16 rows) run the slot program once per row: sixteen launch sequences of one program, all static.
Whether a wider routed program pays at prefill is a measurement, milestone 4.

The pack holds each layer's expert inputs as paged buffers with their host bytes beside the trunk's, and the
budget stays `EMMY_GEN_EXPERTS_GPU`. The server reads those bytes straight into pinned host memory and declares them
to the pager with the device slot count; the trunk, the KV cache and the GDN state stay where they are. The pack key
gains the budget, since the slot count changes the plan.

## Milestones

0. **A tiny hybrid MoE in native preparation.** A random-weight Qwen3.5-MoE config with GDN and attention layers,
   experts resident (slots equal to the expert count): the router, slot and combine programs in the step, the
   model type admitted. Gate: greedy tokens equal to the plugin's on the same checkpoint, and the pack's step
   count and buffer list in the native ARCHITECTURE.

   State (2026-10-22, PR #1236): built. Native preparation admits Qwen3.5-MoE; the plugin's torch routing and
   combine (`_moe_combine_slots`: HF router, selector copy, slot launch, score matmul) are compiled programs in the
   step (router, per-row slot program, combine), and the GDN state wrapper's routed form gained its FP32-residual
   form. Resident experts are in the pack as step constants. Still missing: the E-stacked expert bits and scales of
   a paged layer (32 GB for the target) have no place in the pack, and the server's load path binds
   device-resident buffers only (milestone 1). The gate runs against eager Transformers instead of the plugin
   (booting the plugin needs an image): a tiny Qwen3.5-MoE with two full-attention layers matches every logit and
   greedy token through sequential and chunked prefill, capture and reset (80 to 115 s on the dev box). The same gate
   with one GDN and one attention layer passes too, by hand only: without evidence the GDN decode kernel takes
   minutes to compile, so the run took about ten minutes on the dev box and is not in the suite; the GDN MoE
   program's CPU trace test stands in for it there.

1. **Paged experts in the pack and the server.** Slots below the expert count on the tiny model: the exporter
   writes the expert bytes as paged buffers, the server pins them at load and declares them to the pager, the
   paging programs run inside the replayed step. Gate: the same tokens at every budget, and the host peak one copy
   of the experts.

   State (2026-10-22, PR #1238): built. Below `EMMY_GEN_EXPERTS_GPU` native preparation declares each layer's two
   E-stacked expert buffers paged, one expert per page, with the budget as slots; the backend splices each fixed-slot
   launch's inspector and paging programs into the step, and their tables, slots and clock are step inputs named after
   the buffer, shared by every row's launch in both programs. The pack keeps the expert bytes once, under a `pages`
   index; the runtime reads each file straight into pinned, mapped host memory, declares it to one pager and lends the
   pager's tables to decode as it loads; prefill loads with every buffer it shares with decode lent (an executor now
   takes any subset of its regions lent), so neither the host nor the device holds a second copy of the experts, and the
   device no second copy of the trunk. Generation artifact version 8. Verified on the dev box: the tiny
   two-attention-layer gate at every expert, four and one of eight (two picks) matches every Transformers logit and
   greedy token through sequential and chunked prefill, capture and reset, about 150 s per budget; with about 100 MB of
   wide experts the worker's heap keeps no copy of them and its peak resident set equals its resident set. By hand, the
   same gate with one Gated DeltaNet and one attention MoE layer at two of eight passed too (709 s export); it found the
   step joining kernels by name, where a paging program's kernel met the Gated DeltaNet program's under one name, so the
   step and the paging programs now join their plans through one splice that merges kernels. The pack key does not gain
   the budget: a native pack is never looked up by key, the slot count lives in the plans' paging declarations, and a
   reused pack keeps the slots it was prepared with. The budget rule (`config.gen_expert_slots`) is shared with the
   plugin, and a reused pack rejects the budget. Still missing: quantized MoE checkpoints do not export, and the export
   holds every expert in memory as the model's tensors rather than streaming them from the shards (milestone 2); each
   paged launch carries its own inspector and paging run, 14 launches per buffer, and the two inspectors took about 50
   of the 158 s of a tiny export; a joined plan with 240 paged buffers is unmeasured.

2. **Qwen3.6-35B-A3B-FP8 on the dev box.** The FP8 experts through the loader's spellers as the plugin already
   compiles them, 4,096 context, one request, the golden's rows for every program it covers. Gate: the plugin's
   greedy tokens at temperature 0 on the same prompts, and a TPOT at 128 slots no worse than the plugin's 49 ms on
   the same box (native carries no vLLM step overhead, so the difference is the paging cost alone).

   State (2026-10-22, PR #1240): built; the token gate holds, the TPOT gate does not. The twin of a quantized MoE
   checkpoint leaves its routed experts unread, and the export reads them a layer at a time through the loader (one
   layer's `layer_range`), as the E-stacked FP8 codes and block scales the plugin's fixed-slot program decodes
   in-graph; codes and scales are all pages below the budget, and the pack writes each page file from a function, so
   below the budget the host holds one layer's experts (at or above it, every program's bytes hold them all, which
   for the target would be twice 30 GB). `emmy generate --native-pack` logs the time per token, and `--json` keeps
   the token IDs and times. The target exported in 9 minutes (exporting process peak 12.4 GB, never less than 30 of
   60 GB available), into a 34 GB pack. Tokens, four raw prompts of 64: three match the plugin's, the fourth matches
   40 and then takes the plugin's second choice, where native's own top two are 0.004 nats apart. TPOT about 770 ms
   against the plugin's 42 ms (re-measured twice; the 49 ms above was an earlier reading). Per step, the router takes
   320 ms and attention 308 ms on the prior's picks, the page copies a median 43 ms once warm, the inspectors and
   table updates 8 ms; the launch count is not the problem. No recipe row covers a native program: the recipe
   golden's rows are BF16 kernels recorded with fast math off, so none imports, and native runs FP16. Missing
   evidence is the likely cause, not a proven one: a 1 MiB router at 8 ms and a greedy pick over 248k logits at 29 ms
   are about four orders of magnitude off their memory bounds, which a lowering fault could also do. Deviations: the
   pack prefills one token at a time, since the 16-row Gated DeltaNet program compiled for 47 minutes without
   evidence and was stopped; the evidence scope was the recipe golden plus the card's hardware golden, as the plugin
   boots, because the recipe golden alone imports nothing; the 256-slot measurement cannot run on this card, whose
   32 GB cannot hold the 30 GB of experts beside the trunk. Milestone 3 therefore moves native to the checkpoint's
   dtype, so the recipe golden's rows apply, rather than recording FP16 rows for every program.

3. **Native in the model dtype, then evidence.** The exporter traces, compiles and binds in the checkpoint's dtype
   (the recipe's `--dtype`, BF16 here) instead of FP16 everywhere, so the trunk, the mixers, the router, the head and
   the fixed-slot program are the plugin's kernels and the recipe golden's rows import. First, on the router program
   alone, a recorded row must remove its 8 ms, which rules out a lowering fault before anything else. Then the
   programs the golden does not cover are recorded into the recipe golden, so the boot runs strict: the router, the
   combine, the 16-row Gated DeltaNet prefill program, greedy selection, and anything else native builds that the
   plugin does not. The two GDN gaps milestone 8 of the parent plan left (the width-4 program and the two-row verify
   program) still do not block this: one-token decode never builds them. Gate: the same token match as milestone 2,
   and a TPOT at 128 slots no worse than the plugin's 42 ms on the same box.

   State (2026-10-23, PR #1245): built; the token gate is weaker than milestone 2's, and the TPOT gate holds against
   the plugin as measured today but not against milestone 2's 42 ms. Native preparation builds, traces, compiles and
   binds in one model dtype, the checkpoint's by default (BF16 here) or `--dtype float16`, which the FP16 checkpoint
   qualifications now need for a BF16 checkpoint such as Qwen3-0.6B. The BF16 premise bought only the fixed-slot
   program's rows: native keeps FP32 residual seams at width 1, where the plugin runs BF16 residuals at widths 4, 64
   and any, so native needs its own rows for everything else. `SERVE_NATIVE=1` in a serving env now adds native's
   programs to the twins (traced, not compiled, each with a layout scope of its own), and the recipe golden holds
   measured kernel sets for all 18 native twins, decode and 16-row prefill. Main's #1230 restamped the same file; the
   three native prefill targets it re-keyed and the four plugin targets it left with no priced cut were recorded again
   on the card (the plugin's 64-row GDN and any-width output projections under their stored routes at their old times,
   its 4- and 64-row query projections on the prior's picks, 37 and 60 us summed against 30 and 41 before #1230).
   `emmy eval golden` deploys 43 of the 44 twins from the rows alone; the one that does not is milestone 8's width-4
   GDN program. An export from `--golden` runs in the regime of the golden's rows, as the plugin's boot does; milestone 2
   exported in the default fast-math regime, which is why no recipe row imported then. The router's top k is now a rank
   counted per expert in one reduction (ties to the lowest ID, a nonfinite row to the last expert) and a scatter of the
   experts ranked below k with their probabilities, compared on the stored logits; the router went from 7.85 ms to
   0.35 ms per layer with its first rows (the masked maxima's scan offered only the serial fold) and to about 12 us per
   layer rewritten. Strict export at prefill 16 from the recipe golden alone: 25 minutes, 13 GB peak, a 36 GB pack;
   the pager's own kernels are the only forks it decides without a row. Served over HTTP, both servers warm, the same
   four raw prompts, the same client, 64 tokens, the second pass: native 46 ms a token, the plugin 51 (59 on its first
   pass; milestone 2's 42 ms reading did not reproduce at 128 slots). Per step from the 16th output token on, both move
   a median 96 MB of pages over the x1 link in 29 ms, so on this box the gate is the link's; besides the copies, native's
   GPU work is 15.8 ms against the plugin's 17.9, of which the pager's table programs are 7.8 in both. At x16 the copies
   would take about 2 ms. One `emmy generate` process starts with a cold pager, so its time per token over 64 tokens is
   about 62 ms. Tokens: two of four prompts match the plugin's 64; the others take its second choice at tokens 49 and
   2, where native's top two are 0.17 and 0.06 nats apart and the plugin's 0.38 and 0.13 (the FP16 pack's own, under the
   same teacher forcing, 0.05 and 0.002); milestone 2 matched three. TTFT about 0.7 s through the server. Findings, not
   fixed here: a reduction nested in a scan step is offered only the serial fold (#1249); a scatter fused with a
   softmax over another sweep at more than one row loses its store silently in the kernel lowering (#1248; avoided by
   scattering the probabilities too); the pager's decision kernel lowers to one CTA and builds no other schedule, about
   34 us per paged launch at 256 pages and 5.4 ms a step (#1250); a kernel benched alone from a golden is offered TMA
   staging that a reader of a paged buffer refuses, so a row for such a reader must be measured without it. Cost
   milestone 4 owns: the 16-row prefill program (38,263 launches, 120,152 buffers) takes the worker about 12 minutes
   to load. Open: move native's seams to the plugin's BF16 form, so the mixers share the plugin's rows; spread the
   pager's decision kernel past one CTA (#1250); then the page copies.

4. **Prefill width.** Per-row slot launches against one routed program over the chunk, measured on the target.

   State (2026-10-23, PR #1254): built; the one program wins. A prefill chunk runs each MoE layer's fixed-slot program
   once over all its rows: the router writes the chunk's picks as one `[rows, k]` selector, the program batches its
   products over the rows as well as the picks, and the combine reads one `[rows, k, hidden]` buffer; the plugin's
   one-row form, and so decode, is unchanged. A paged buffer therefore takes one inspector and one paging run per
   chunk, not per row, and the target's 16-row prefill program went from 38,353 launches and 120,242 buffers to 3,799
   and 11,978. The runtime now finds a plan's buffers by name through an index: the per-row plan took 347 s to lay out
   with a scan per name and takes 0.9 s. The recipe golden holds rows for the three new twins (the 16-row router's
   selector, the chunk's fixed-slot program, its combine); the fixed-slot program runs at 348 us a layer on the prior's
   picks, where torch.compile takes 133. Strict export from the recipe golden alone: 18 minutes, 15 GB peak, a 36 GB
   pack. Measured on the dev box at 128 of 256 experts, one `emmy generate --capture` process per prompt, so every
   request starts with a cold pager. Load: 25 s for the chunk pack, 45 s for milestone 3's (666 s before the index).
   First token after a 993-token prompt (62 chunks): 25.7 s against 41.3 s; after a chat prompt of one chunk: 3.5 s
   against 11.3 s, most of the difference the capture of the per-row program's graph. Per warm chunk (nsys) both move
   the same pages over the x1 link, 1.2 GB in about 345 ms; besides the copies the GPU works 62 ms against 151, and the
   difference is the pager's table programs, 7.7 ms against 97 (the decision kernel runs once per paged launch,
   #1250); the model's own work is 54 ms against 52, the chunk's fixed-slot program taking 18.7 ms where sixteen
   one-row launches took about 17. At x16 the copies would take about 27 ms, so a chunk would take about 90 ms against
   190 (estimate). Decode is unchanged, 27 ms a token after the long prompt. Tokens: a prompt under 17 tokens never
   runs the prefill program, so the four raw prompts match milestone 3 and the plugin as before (two of four); of four
   chat prompts three match milestone 3's 64 tokens, and one leaves it at token 28, where the chunk pack's top two
   are 0.06 nats apart and milestone 3's 0.10 the other way; the long prompt matches 56 tokens, then the chunk pack
   takes a choice 0.04 nats ahead where milestone 3's runs 1.0 nats the other way, and a prefill one token at a time
   through the decode program agrees with the chunk pack for all 57. Against the plugin's greedy tokens on the same
   five prompts the two packs agree: three chat prompts match all 64, the fourth leaves the plugin at token 13 (its
   top two 0.50 nats apart) and the long prompt at token 23, where the plugin's top two tie exactly. Open: the chunk's
   fixed-slot program on a faster recorded schedule (torch.compile runs it 2.6 times faster); the pager's decision
   kernel (#1250); the page copies.

Milestone 1 needs 0; 2 needs 1; 3 needs 2; 4 needs 3, since a chunked prefill compiles only once milestone 3 covers
its Gated DeltaNet program. The gates run on the dev box; the x16 numbers come when a 5090 host with 48 GB or more of
RAM can be rented (the 23 GB ones cannot pin the target's experts).

## Out of scope

The MTP drafter (the one-rank fixed-slot tier serves width 1; speculative verify steps are the parent plan's
milestone 8 problem under either runtime), more than one active request, context past 4,096 tokens, hash-routed
and DeepSeek-style routers, EXL3 mixed expert groups, the NVMe tier.

## Risks

- **Top-k ties.** The last of k picks flips on float16 rounding of O(1) scores; the router program keeps scores and
  the bias in FP32, and milestone 0's token gate is where a flip shows.
- **Pack size and load time.** The target's pack is about 37 GB. The exporter must stream the expert bytes and the
  server must read them straight into pinned memory; a staging copy repeats the double-copy boot #1234 removed.
- **Many paged buffers.** Forty layers times six expert inputs is 240 paged buffers, each with its tables and
  inspectors in the step; the plugin instantiates the same count per layer, so the cost is known, but the joined
  static plan has not held that many.
- **Compile time.** The plugin's boot compiles the target in about 12 minutes from a warm cubin cache with golden
  rows for every program; a native export with the two new programs on the prior adds their search once.
