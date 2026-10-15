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

## Milestones

0. **Pick the model and measure skew.** A MoE model that fits the 5090 whole, so every run has an all-GPU baseline
   (OLMoE-1B-7B or Qwen3-30B-A3B at a low-bit rate; whichever the serving runner already serves). Run the histogram over
   a fixed prompt set. Gate: per-layer coverage curves (experts needed for 50/80/95% of rows) in an experiment
   RESULTS.

1. **Zero-copy one layer.** Allocate one layer's E-stacked expert weights in mapped pinned memory and point the
   indirect table at the device-mapped addresses; everything else unchanged. Bench decode TPOT with that layer's
   experts on host against all on GPU, through the serving runner and the vLLM benchmark client. Gate: tokens identical
   to the all-GPU run (same bytes, different memory), and the measured per-launch cost beside the PCIe floor.
   Expected trap: a schedule staged with TMA cannot take an indirect operand already; check that cp.async and plain
   loads both work on mapped host memory, and record which schedule the evidence picked.

2. **Hot set on GPU, cold set on host.** Use milestone 0's counts to keep the top-N experts per layer on the GPU and
   the rest on host, with N set by a memory budget flag. Gate: TPOT against GPU fraction, a curve, for N from all to
   none.

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

4. **A model that does not fit.** Repeat milestone 2 or 3 on a MoE model larger than the 5090's memory, against
   llama.cpp's and KTransformers' CPU/GPU split on the same box. Gate: a published comparison.

5. **H100 confirmation.** Re-run milestones 1–2 on a rented H100 (PCIe 5.0 host link as well). Gate: the 5090
   conclusions hold.

Milestones 0 and 1 are independent. 2 needs both. 3 and 4 need 2.

## Out of scope for the prototype

- **Kernel-initiated fetch.** A kernel that finds a cold entry, copies the expert itself and waits. Zero-copy already
  is the in-kernel fetch for host memory; a separate fetch path only matters for remote memory.
- **Remote memory over RDMA.** GPUDirect RDMA into GPU cache slots, or GPU-initiated reads (NVSHMEM / IBGDA). Same
  table, a third tier behind host memory. Needs two machines and suitable network cards.
- **CPU compute of cold experts** (the llama.cpp / KTransformers split). A different design: the CPU runs the expert
  instead of the GPU reading it. Milestone 4 compares against it rather than building it.
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
