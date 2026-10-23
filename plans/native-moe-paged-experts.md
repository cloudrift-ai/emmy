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

2. **Qwen3.6-35B-A3B-FP8 on the dev box.** The FP8 experts through the loader's spellers as the plugin already
   compiles them, 4,096 context, one request, the golden's rows for every program it covers. Gate: the plugin's
   greedy tokens at temperature 0 on the same prompts, and a TPOT at 128 slots no worse than the plugin's 49 ms on
   the same box (native carries no vLLM step overhead, so the difference is the paging cost alone).

3. **Evidence.** Record the router and combine programs into the recipe golden, so the boot runs strict. The two
   GDN gaps milestone 8 of the parent plan left (the width-4 program and the two-row verify program) do not block
   this: one-token decode never builds them.

4. **Prefill width.** Per-row slot launches against one routed program over the chunk, measured on the target.

Milestone 1 needs 0; 2 needs 1; 3 and 4 need 2. The gates run on the dev box; the x16 numbers come when a 5090
host with 48 GB or more of RAM can be rented (the 23 GB ones cannot pin the target's experts).

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
