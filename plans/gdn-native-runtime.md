# Gated DeltaNet in the native Rust runtime

Status: open, written 2026-10-03 against main `3a2510857`. Nothing here is implemented or measured yet.

Goal: native generation serves one active request of a dense, unquantized Qwen3.5-family text model — Gated DeltaNet
layers beside full-attention layers — on one GPU, with no Python after preparation, under the accuracy contract native
Qwen3 already holds.

Not the goal: Qwen3.8-27B-FP8 on four V100s. That also needs quantized weights, more than one GPU and batching, which
the [native serving plan](paged-attention.md) defers. That plan also defers additional model families; this plan lifts
one family out of that deferral and leaves the rest of its order unchanged.

## What exists

- **The state wrapper.** `build_gdn_state_wrapper` exposes a block as `(x, state, history) -> (y, state, history)`.
  The state is the FP32 recurrent matrix, `[batch, value heads, key dim, value dim]`; the history is the last
  convolution-kernel-width projected inputs, `[batch, conv channels, kernel width]`, in the model dtype. Zero tensors
  start a request and batch rows are independent. It traces at any static width, and a tiny configuration compiled on
  CUDA already matches transformers across a prefill-to-decode handoff and a reset
  (`tests/compiler/trace/test_huggingface.py`), with the host carrying the state between launches.
- **Paged buffers.** A kernel reaches a paged buffer through a table of page pointers in device memory. The runtime
  owns zeroed pages, and the prefill program borrows the decode program's tables. Rebinding a table clears the
  captured graphs; rewriting a table's contents would not, and no such operation exists yet.
- **Native preparation.** It joins per-layer compiled plans into one static plan per width, with seams named as shared
  allocations. It rejects everything but dense Qwen3 with full rotary embedding.
- **The runtime.** It replays the plan and holds no model math. A new request resets the position and prompt length
  and clears nothing on the device.
- **Evidence.** The Qwen3.8-27B-FP8 V100 golden holds the chunk rule's kernels at 64 and 512 tokens. Those are not the
  native step's shapes, and a paged operand changes a kernel's identity, so no row transfers.

## Design

### The state is a paged buffer the step reads through one table and writes through another

The step cannot update the state in place. Every plain kernel parameter is `__restrict__`, so a launch's output never
shares memory with its inputs. The update of one cell reads a whole column of the old state, and the history shift
reads a neighbouring cell. Which kernels exist depends on the cut the evidence picks, so correctness must not depend
on it either.

A copy per step is too slow at the target scale: Qwen3.8-27B carries 48 heads x 128 x 128 FP32 per layer, 3.1 MB, and
about 150 MB over its 48 Gated DeltaNet layers, for every token.

So each layer's state and history are each two paged buffers in the step plan — the one the step reads and the one it
writes — paged on the batch axis with one page per batch row. The runtime keeps two pages per buffer and writes them
alternately: after a step, the read table points at the page just written and the write table at the other. All
tables sit in one contiguous device allocation, so the exchange is one small upload per step, beside the position
and token uploads the step already makes. Table addresses never change, so the captured graph replays.

- **Reset is a table write.** One zeroed page per distinct shape is never written. `start` points every read table at
  it. Nothing is cleared.
- **Prefill and decode share the tables,** as they share the KV cache today, so decode reads what prefill last wrote.
- **Batching falls out later.** A table row per request is the request axis the native serving plan's batched
  decoding adds; the state needs no second mechanism.

### Prefill never runs past the prompt

Today a prefill chunk may hold rows past the prompt's end, because decode overwrites those cache rows. Carried state
has no such repair: those rows would be folded into the request. For an artifact that declares carried state, the
runtime dispatches a chunk only where every row is a prompt token before the last, and the one-token program covers
the rest — up to one chunk width minus one extra steps, each computing a head nobody reads. Dense Qwen3 keeps today's
rule.

Masking the update by prompt length inside the traced block is the alternative. It changes the kernels, so it waits
for the measurement in milestone 4.

The prefill shortcut that skips the last layer's attention and post-attention fragment applies only when that layer
is an attention layer. A Gated DeltaNet layer always runs whole, since its state must advance.

### The residual stays FP32

Native keeps the hidden seam between layers in FP32 and casts to FP16 before each projection. The state wrapper runs
the block whole in the model dtype. Give it the same FP32-residual form the attention split has, so both layer types
meet at one seam dtype and the accuracy contract does not change.

### Full-attention layers of this family

Two things native rejects today and this family needs: partial rotary embedding (a quarter of each head) and the
attention output gate. The attention split already carries the gate as a fourth `pre` output and third `post` input.
The rotary module rotates only the leading part of each head. Preparation finds the text decoder with
`find_text_decoder`, since these checkpoints wrap it beside a vision tower that native does not load.

### Contract

The generation contract in the pack key lists each read/write pair of paged buffers. That is the one thing the
runtime must be told. It is a new generation artifact version; older artifacts are exported again.

## Milestones

0. **Prove the mechanism on the tiny configuration, before any runtime change.** Extend the existing CUDA
   handoff-and-reset test: state and history declared paged on both sides, tables bound by the test, pointers
   exchanged between launches. Gate: equal to transformers across prefill, decode and a reset at widths 1 and 16, and
   every lowering refusal named. Two things to expect: a paged output with no start operand and a one-page axis may
   not lower yet, and paged tensor-core staging is refused today. A page that spans the whole buffer always contains
   the tile, so its base resolves once — the simplest case of the native serving plan's per-tile lookup.
1. **Runtime.** Add the table-contents write, which leaves graphs alone. The generator allocates the zero page and
   the two pages per buffer, exchanges after every prefill chunk and decode step, resets at `start`, and applies the
   stricter chunk rule when the artifact declares carried state. Gate: Rust tests for exchange, reset and the chunk
   rule; a failed step still ends the request.
2. **Preparation.** A layer loop that takes either layer type, the FP32-residual state wrapper, partial rotary, the
   output gate, and validation that accepts the dense Qwen3.5 text family and still rejects quantization. Gate: on a
   tiny two-layer model (one of each type), logits match transformers within the native error limits, sequential and
   chunked, at prompt lengths 1, 16, 17 and 65; a second request after reset equals the first; capture on and off
   select the same tokens.
3. **A real checkpoint on one card.** Use the smallest dense checkpoint of the family that fits the card; the
   repository's only small recipe is Qwen3.5-9B, 18 GB in FP16. Record schedules by hand for the one-token step and
   the 16-row chunk, with a fresh tune DB and strict evidence. Compile time is the risk: on 2026-09-18 one greedy
   resolve of the chunk rule at real shapes took 48 minutes. Re-measure on main before planning around that number.
   Gate: the checkpoint harness native Qwen3 passes, sequential and chunked, plus the HTTP lifecycle check.
4. **Measure, then decide.** Decode step time by layer type, the cost of the table upload, the cost of the prefill
   tail, and a 64-row chunk (the rule's own chunk size) against 16. These decide the mask, the chunk width, and
   whether tensor-core tiers over paged state are worth building.

Milestone 0 comes first. Milestones 1 and 2 are independent of each other. Milestone 3 needs both.

## Open questions

- Do the carried-state schedules that won on the V100 — the state held in shared memory across ordered steps — still
  lower when the state's source and destination are paged?
- Is one upload per step measurable against a decode step? If it is, two captured graphs with the tables exchanged
  remove it, at the cost of a second capture per program.
- Native's 4,096-token context limit stays. This family's window is far larger; raising the limit is the native
  serving plan's page-lifetime work, not this plan's.
