# MoE experts in host memory

Status: proposed 2026-10-15. Prototype for hybrid CPU/GPU inference of MoE models whose routed experts do not fit in
GPU memory: cold experts live in pinned host memory, and the GPU reads them over PCIe, either directly from the kernel
(zero-copy) or after the runtime copies them into a GPU cache. The full paged-attention plan (retired in #972, last
full text at `d8218abf5^:plans/paged-attention.md`) paged the KV cache; this plan pages expert weights. It adds no IR,
no fusion gate and no schedule rule.

## Why the hook already exists

Decode MoE in the serving runner is a fixed-slot dispatch: each slot launches the shared expert program, and the
program reads its weights through an **indirect operand**, `table[sel[slot]]`, where the table holds one device
pointer per expert and the selector is the router's top-k indices, written on the device. A page table in all but
name. Any pointer the GPU can dereference may sit in that table, and pinned host memory mapped into the device address
space (`cudaHostAlloc(…, cudaHostAllocMapped)` or `cudaHostRegister` + `cudaHostGetDevicePointer`) is such a pointer.
So the first measurement needs no compiler change: bind the cold experts' table entries to host-mapped slices.

Prefill takes the other path (host-side `unique()` over the routed rows, one launch per hit expert), so it knows its
experts on the host before it launches them; decode knows them only on the device, one kernel after the router.

## What we need to learn

1. **Zero-copy cost.** How much slower is one expert launch reading host weights than GPU weights, at decode width?
   PCIe 5.0 is ~50–60 GB/s in practice against ~1.8 TB/s on the 5090, so a bandwidth-bound expert should be ~30×
   slower. Whether the kernel even reaches PCIe bandwidth through plain loads, and which schedules do, is the question.
2. **Skew.** How many experts cover most routed rows? The routing histogram (`EMMY_GEN_ROUTING_HISTOGRAM_INTERVAL`)
   already counts selections per layer and expert; it was added to answer exactly this.
3. **Copy versus zero-copy.** For a hit cold expert, is it cheaper to read it once in place, or to copy it into a GPU
   cache slot and read it there? One read in place moves the same bytes as one copy, so zero-copy should win for a
   single use, and a cache only pays when the same expert is hit again soon.

## Experiments

All on the 5090 with one model (milestone 0), one experiment recipe, a fresh tune DB per run, and the same workload:
decode at concurrency 1 and 8, and a 512-token prefill. Each reports TPOT, TTFT, bytes moved over PCIe (nsys), and,
where a GPU cache exists, its hit rate. Every row sits beside two fixed baselines: **all experts on GPU** (the ceiling)
and **all cold experts read in place** (strategy A, the floor any fetch must beat).

**Micro-measurements first**, since they bound every strategy below:

- **M1. PCIe read rate by transport.** One expert slab read from pinned host memory four ways: plain loads in the
  expert kernel, cp.async in the expert kernel, a copy-engine `cudaMemcpyAsync`, and an SM copy kernel. Expert-sized
  slabs (a few MB at low bit rates). Tells whether a kernel reading in place reaches the copy engine's rate.
- **M2. Host round trip.** The cost of one device→host read of the router's indices plus the host issuing a copy and
  the next launch, per layer. Prices every strategy that brings the host into the loop.
- **M3. Prediction accuracy.** Per layer, how often layer L+1's router applied to layer L's hidden state names the
  real top-k, offline over the prompt set (top-k overlap, and recall at a wider k' > k). The routing histogram counts
  selections, not matches, so it cannot answer this.
- **M4. CPU expert rate.** One expert at 1, 4, 16 and 64 rows through the loop backend's host build against llama.cpp's
  CPU kernel for the same expert and bit rate, all cores. Says how far the compiled CPU kernel is from a hand-tuned one
  and where its rate crosses PCIe's.
- **M5. NVMe read rate.** One expert's bytes from the local NVMe into pinned host memory: O_DIRECT reads through
  io_uring at queue depths 1–32, against a page-cache read through mmap. With M1 it prices a disk miss, which sets
  how rare an expert must be to live on disk.

**Strategies**, each one variant of the same expert table:

| | who decides what to fetch | who moves the bytes | when | graph replay |
| --- | --- | --- | --- | --- |
| A. zero-copy | nobody: cold entries point at host memory | the expert kernel, every use | during compute | kept |
| B. in-kernel fetch | the expert kernel, on a cold entry | the expert kernel, into a cache slot | first use, then cached | kept |
| C. inspector on CPU | the host, after reading the router's indices | copy engine | between router and experts | broken per layer |
| D. inspector on GPU | a small kernel after the router | an SM copy kernel, into cache slots | between router and experts | kept |
| E. predictor | layer L+1's router on layer L's hidden state | copy engine on a side stream | one layer ahead | kept |
| F. static hot set | the boot-time histogram | nobody at runtime | never | kept |
| G. CPU or GPU per expert | inspector (C or D), per expert | only activations, for a cold expert run on the CPU | between router and experts | broken per layer |

- **A** is milestone 1, no compiler change. Its question is whether one read in place is already as fast as a copy.
- **B** is the in-kernel fault fallback: a residency check in the kernel's preamble, and on a miss the CTAs copy the
  slab into a cache slot and flip the table entry. Needs cross-CTA coordination (one CTA copies, the others wait, or
  each copies its own tile) and a cache allocator on the device. Worth it only if A loses to a copy by a wide margin
  (M1) and the same experts recur within a few steps.
- **C** is prefill's path today, extended to decode. It is the only one that works for remote memory on a card
  without GPUDirect RDMA. M2 says what it costs per layer; on a 48-layer model even 20 µs per layer is about 1 ms per
  token.
- **D** keeps C's exact knowledge without the host: the copy kernel and the table update sit inside the graph. It
  serializes the copy before the experts, so it beats A only if a copy into GPU memory plus reads from it beats one
  read in place, which M1 decides.
- **E** is the only one that overlaps the copy with useful compute. Its value is M3's accuracy times the time it hides;
  misses fall back to A. Combine with a cache (E + D's cache slots), since a predicted expert copied in has to live
  somewhere.
- **F** is milestone 2; every dynamic strategy must beat it at the same GPU budget to earn its complexity.
- **G** runs a cold expert where its weights are. Once the inspector knows the experts and how many rows each got, a
  cold expert with few rows runs on the CPU, which reads its weights from RAM at memory speed (~80–100 GB/s on a
  desktop, 300+ on a server, against ~55 GB/s of PCIe), and only its rows' activations cross the bus; a cold expert
  with many rows is still cheaper on the GPU. The CPU kernel comes from the same compiler: the loop backend already
  builds Loop IR as C++ for the host. The rule (rows routed × bytes against the two rates) is the experiment's output,
  not its input: sweep the row threshold and plot TPOT and TTFT. CPU and GPU experts run concurrently, and the layer
  waits for both before the combine. Needs M4.

Expected order, to check rather than assume: E + A ≥ F > A ≥ D > C, with B close to D. If M1 shows reads in place far
below the copy engine's rate, D and B move up. G's place depends on M2 and M4: at decode width it should beat every
GPU-only strategy for experts that are cold most of the time, if the per-layer round trip stays small.

## Milestones

0. **Measure skew on OLMoE-1B-7B.** It fits the 5090 whole, so every run has an all-GPU baseline, and the serving
   runner's MoE seam already covers its token-choice top-k router. Run the histogram over a fixed prompt set. Gate:
   per-layer coverage curves (experts needed for 50/80/95% of rows) in an experiment RESULTS.
   **Done 2026-10-16:** mild skew once workloads mix (a median 40 of 64 experts per layer for 80% of rows, against 51
   with none), stronger within one workload (23 for code), and hot sets that do not carry across workloads: a chat
   hot set covers code no better than random experts. A boot-time hot set (F) is weak on this model; residency that
   follows the workload, or prediction, carries the win.

1. **Zero-copy one layer.** Allocate one layer's E-stacked expert weights in mapped pinned memory and point the
   indirect table at the device-mapped addresses; everything else unchanged. Bench decode TPOT with that layer's
   experts on host against all on GPU, through the serving runner and the vLLM benchmark client. Gate: tokens identical
   to the all-GPU run (same bytes, different memory), and the measured per-launch cost beside the PCIe floor.
   Expected trap: a schedule staged with TMA cannot take an indirect operand already; check that cp.async and plain
   loads both work on mapped host memory, and record which schedule the evidence picked.
   **Done 2026-10-16** (`EMMY_GEN_EXPERTS_HOST`): layer 8's experts in host memory gave 8 of 8 greedy answers token
   for token. On a 5090 at PCIe 5.0 x16 TPOT went from 2.84 to 4.99 ms: the added 2.15 ms is 8 experts of 12 MiB per
   token at the copy engine's rate (45.8 GB/s), so reads in place match a copy and B or D pay only on reuse. The dev
   box's 5090 trains to x1 (3.59 GB/s, +32.9 ms). The evidence picked cp.async staging for gate/up and plain loads for
   down; no TMA.

2. **Hot set on GPU, cold set on host.** Use milestone 0's counts to keep the top-N experts per layer on the GPU and
   the rest on host, with N set by a memory budget flag. Gate: TPOT against GPU fraction, a curve, for N from all to
   none.
   **Done 2026-10-16** (`EMMY_GEN_EXPERTS_GPU`, `EMMY_GEN_EXPERTS_RANK`): on the x16 5090, TPOT ≈ 2.8 ms + (host share
   of routed rows) × 35.5 ms, the link time of one token's 128 expert reads. Keeping 48 of 64 experts per layer (9 of
   12 GiB) left 15% of chat rows and 7.5% of code rows on host and gave 6.94 and 5.64 ms against 2.76 and 2.82; none
   on the GPU gave 35.4 ms. Tokens identical in every row. A static hot set saves only about a quarter of the expert
   memory within 2× of all-GPU, so milestone 3 has to cut the host share of rows, not the cost per read.

3. **Residency at runtime.** Decide whether a GPU cache of recently hit experts beats a static hot set. The runtime
   already rewrites device tables between graph replays for carried state; the same move swaps an expert's table entry
   from its host address to a cache slot after a copy on a side stream. Decode cannot copy before the launch that
   needs the expert (the router is one kernel earlier, on the device), so the first hit always reads in place and a
   copy only serves later steps — unless the copy starts a layer early. **Router prediction** does that: apply layer
   L+1's router to layer L's hidden state, which is usually right because the residual stream changes slowly between
   layers (Pre-gated MoE, HOBBIT). The predicted experts' copies run on the side stream while layer L+1's attention
   computes; a wrong prediction costs only the wasted copy, since the real router still writes the selector and a miss
   reads in place. The prediction is one extra router matmul per layer, on the device, so decode stays one graph.
   Measure first how often the prediction matches the real top-k, per layer, offline over milestone 0's prompt set
   (the routing histogram counts selections, not matches, so it cannot answer this).
   Gate: TPOT of the cache, with and without prediction, against milestone 2's static set at the same GPU budget.

4. **Design.** Milestones 1–3 are prototypes: they may hard-code one model, one layer kind and the serving runner's
   Python path. Once the experiments say which strategies win, design the mechanism before building it for real, and
   throw the prototype code away rather than grow it. The design answers:
   - **One pointer-table mechanism.** Indirect operands (`table[sel[slot]]`) and paged buffers (`pages[i / page]`) are
     two spellings of the same thing: an operand whose base address comes from a device table the runtime owns. Merge
     them into one kernel ABI and one plan field if the winning strategies need both, so residency is one concept.
   - **Who owns residency.** A tier manager in the Rust runtime that owns the pools (GPU cache slots, pinned host
     memory, later remote memory), the tables, and the moves between tiers, beside the page allocator the
     paged-attention work planned. Python serving paths call it; they do not keep their own copy.
   - **Where placement is decided.** Whether an expert runs on the CPU or the GPU (strategy G) is a runtime decision per
     step, not a compiler fork; the compiler only builds both kernels. Say what the runtime reads to decide, and whether
     that rule is measured evidence or a fixed formula.
   - **What the compiler knows.** Ideally nothing new: residency stays out of kernel identity, schedules and goldens, as
     paging does today. If milestone 1 shows the evidence picks wrong schedules for host-resident operands, say whether
     residency joins the evidence context, and what that costs every golden.
   - **What stays out.** Each strategy that lost in milestone 3 is named with its number and left unbuilt.

   The design lands as text in the runtime's and backend's `ARCHITECTURE.md` files, reviewed by the architect agent.
   Gate: the author approves the design before milestone 5 starts.

5. **Build the design.** Implement the mechanism milestone 4 settled, in the runtime, with tests, for every MoE model
   the serving runner serves, not one layer kind. Gate: milestone 3's best TPOT reproduced on the new code, and the
   prototype paths deleted.

6. **Speculative decoding.** Strata gets 1.6–1.8× from it alone, independent of expert placement, so the comparison
   needs it on both sides. Qwen3.8-Flash-Next ships a one-layer MTP head that drafts tokens, and the target verifies a
   draft of k tokens as one (k+1)-row step. Two things block it today:
   - **Gated DeltaNet state.** 36 of the 48 layers carry a recurrent state, and a rejected draft token has already
     advanced it, which is why the serving runner refuses speculative decoding on GDN models. The verify step must keep
     the state after each of its k+1 rows, and the runtime then carries the one after the last accepted token. This is
     carried state with k+1 write pages instead of one, the same page-table move the runtime makes today.
   - **The MTP head** has no compiled program yet; it serves no twin. It is a small layer with its own MoE, compiled
     like any other.

   It also changes the expert picture: a verify step routes k+1 rows per layer, so it touches more distinct experts
   than a decode step and the rows-per-expert count rises, which shifts strategy G's CPU/GPU threshold. Re-run M3 and
   the placement sweep at the verify width. Gate: identical tokens with speculation on and off at temperature 0, and
   the accepted tokens per step and the speedup, measured on Qwen3.8-Flash-Next on the 5090 with experts in the tiers
   milestone 5 built.

7. **The goal: `nvidia/Qwen3.8-Flash-Next-NVFP4` on the local 5090, hot experts in GPU memory, warm in host RAM,
   rare on NVMe.** The checkpoint is 133 GB; the box has 32 GB of GPU memory and 64 GB of RAM, so after the trunk, the
   KV cache and the OS, roughly 40–50 GB of experts stay on disk. Repeat the best configuration with that third tier.
   **The bar is [Strata](https://github.com/niko1221/strata)**, a llama.cpp-lineage engine built for exactly this
   model on consumer cards: hot experts on the GPU, every expert in RAM, the CPU computing the rest, the SSD for what
   RAM cannot hold, and a draft model for speculative decoding (1.6–1.8×). Its published numbers, RTX 5070 12 GB,
   Ryzen 5 7600, 64 GB RAM, 32K prompt and 4K answer: 94 tokens/s decode and 2,650 tokens/s prefill at Q2_0, 53 and
   1,620 at IQ3_S, and 7–8.5 tokens/s decode for UD-Q4_K_XL once it reads from the SSD. Its fast rows are 2–3-bit
   GGUF checkpoints of 45–60 GB that fit RAM whole, so they are not the comparison. **We compare at 4 bits**: our
   NVFP4 against Strata's UD-IQ4_XS and UD-Q4_K_XL, both run on our 5090 box at the same prompt and answer lengths,
   with speculative decoding on for both. Report a quality check beside the speed (the same short eval on all three
   checkpoints) so "4-bit" is a measured match, not a label. Steps:
   - **Skew on the real model.** Run milestone 0's histogram on Qwen3.8-Flash-Next itself; OLMoE's curves do not
     transfer. The tier only works if the coldest ~40% of experts take a small share of routed rows (M5 says how small).
   - **The disk tier.** The NVMe pool is the checkpoint's own expert bytes, read in place, never a second copy. A miss
     reads the expert into a pinned host slot (O_DIRECT, io_uring), evicting the least recently used warm expert, and
     then follows the winning host strategy. GPUDirect Storage, which would read straight into GPU memory, is not
     supported on GeForce cards, so the host hop stays.
   - **Prediction pays most here.** A disk read is milliseconds, so a miss on the critical path stalls the token. Start
     the read from the router prediction (strategy E) a layer ahead, or more layers if M3 shows accuracy holds.

   Gate: decode and prefill faster than Strata on the same box at matched quality, published as an experiment, with
   the disk-miss rate and the TPOT it costs reported beside it.

8. **H100 confirmation.** Re-run milestones 1–2 and the built design on a rented H100 (PCIe 5.0 host link as well).
   Gate: the 5090 conclusions hold.

Milestones 0 and 1 are independent. 2 needs both. 3 needs 2. 4 needs 3, 5 needs 4, 6 needs 5, 7 needs 6, and 8 needs
5. M1–M4 run before milestone 1 and M5 before milestone 7; strategies B, C, D and G run inside milestone 3, beside the
predictor, and only those that beat F at some budget reach the design. The design (milestone 4) covers the disk tier
too.

## Out of scope for the prototype

- **Remote memory over RDMA.** GPUDirect RDMA into GPU cache slots, or GPU-initiated reads (NVSHMEM / IBGDA). Same
  table, a third tier behind host memory. Needs two machines and suitable network cards.
- **Native path.** The native runtime does not serve MoE yet; the prototype runs through the serving runner.

## Risks

- **Evidence measured on GPU memory.** Every recorded row priced the expert program with weights in GPU memory. A host
  pointer changes the speed and not the kernel's identity, so the pick may be wrong for a host-resident expert, and the
  tune DB would mix both if a bench ran with host pointers. Bench host-resident runs with a fresh tune DB and do not
  record them into a golden. Whether a host-resident operand should be part of the context is a question for after
  milestone 1, not before.
- **Graph capture.** Mapped host addresses are stable, so captured decode graphs stay valid when only table contents
  change, as they do for carried state. A table rewrite must not race a replay in flight.
- **Pinned memory limits.** Pinning tens of GB can fail or starve the host; allocate once at boot and fail loud.
- **Prefill.** A long prompt hits most experts; reading every cold expert in place per chunk is PCIe-bound. Prefill may
  want to copy the chunk's experts in up front, since it knows them on the host. Measure before designing.
