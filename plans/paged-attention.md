# Full paged attention

Status: proposed 2026-09-26, revised 2026-09-27. Builds on PR #871, which made paging a property of a buffer: a
graph hint names the buffer, the axis and the page size, every read and write resolves its page before its offset,
the runtime binds the page table as an operand and owns pages that span a buffer's declared shape, and the native
Qwen3 path serves one request at a time over such a cache, every kernel it runs compiled from a traced module and
the token selected on the host. Its prompt is consumed in fixed-width chunks through a second program that borrows
the decode program's page tables. This plan takes that to many concurrent requests at a cost close to the unpaged
kernels. It adds no IR and no fusion gate: paging stays on the buffer, kernel boundaries stay with the cut evidence.

## Objective

Serve concurrent requests through the native Rust server over a paged KV cache: pages allocated per request and freed
at its end, shared prefixes reused, prefill in chunks through compiled attention, and decode attention within a small
measured margin of the same kernel over a contiguous cache. Measured against stock vLLM on the same card with the
vLLM benchmark client, not claimed.

## What exists and what is missing

| piece | today (#871) | missing |
| --- | --- | --- |
| addressing | per-element page lookup in `Load`/`Write`; `start` shifts a chunk write | per-tile lookup when the page divides the KV tile |
| kernel forms | scalar and warp-tile attention page; `mma` and TMA staging refuse a paged operand | tensor-core attention over pages |
| runtime | one table per paged buffer; pages spanning the declared shape at load, or a host-bound table | allocator with a free list, per-request page sets, reference counts |
| native path | one request; prefill in fixed-width chunks (16 rows) sharing the decode cache's page tables; the token selected on the host; page size fixed at export | batch of requests, admission and scheduling, a chunk width that follows the prompt |
| evidence | V100 rows for the one-row fragments; the 4080 rows unpaged; no rows for the 16-row chunk fragments | paged attention rows per card at every width; a corpus case per form |

## Design

**Block table.** The plan's paging declaration gains a second axis: the table row. A paged buffer of shape
`[batch, heads, tokens, d]` paged along `tokens` resolves `pages[b * max_pages + t / page][...]`, so one device table
holds every request's pages, one row each, and a request is a row of the table plus its length. The `Paged` memory
implementation grows that one index term; `Load` and `Write` stay unchanged. Lengths and the write position are
runtime arguments of the symbol environment, as `start` already is; per-request lengths reach the kernel as one
small input buffer, since a symbol is one value per launch. Graphs stay valid across requests: the table's address is
baked, its contents are not.

**Per-tile lookup.** When the page size is a multiple of the KV tile and the tile is page-aligned, the `Memory`
protocol answers a per-tile base instead of a per-element one, and the tile's staging (register, `cp.async`) reads
rows from that base. This is the perf lever; the scheduler learns one fact, "this tile lies inside one page", and
offers the same schedules as before. TMA bakes a base address per descriptor, so it does not page: on sm_90 the
attention over a paged cache stages with `cp.async` rows, and the refusal stays for TMA rather than becoming a fusion
or schedule gate.

**Allocator.** In `crates/emmy-runtime`, a `PagePool` per paged buffer: a free list of equal pages, a `Sequence`
holding a request's page ids and length, and a reference count per page for prefix sharing. The executor's table is a
region the pool rewrites per step from the active sequences' rows; freeing returns pages to the list. Prefix reuse is
a hash chain over full pages (token ids of the page and its predecessor's hash), looked up at admission; a hit shares
the pages and bumps their counts, a miss allocates. Eviction is least recently used over unreferenced pages.

**Scheduler.** In `emmy-server`: admission up to the pool's capacity, decode steps over every active request in one
launch of the batched fragments, new requests joining at step boundaries, and prefill in chunks of a fixed token
budget through the compiled attention program at `q_len` = chunk, `kv_len` = position, writing each chunk at its
`start`. The token is selected on the host today, one vocabulary download per step; a batch downloads one row of
logits per request, which is fine at eight rows and is the point to move argmax back onto the device as a compiled
reduction when it is not. The fragments compile at the serving widths (M = 1, 8, 16, 32) the export declares, and
the step picks the smallest width that fits the batch; a wider batch waits.

## Milestones

0. **Price paging alone.** Bench the compiled attention program paged against flat at the same schedule, page sizes
   16, 64, 256, `q_len` 1 and 256, on the 5090 and the 4080, through `emmy run --golden … --ab`, and put a paged case
   per form into the realization corpus so `make bench-kernels` tracks it. Gate: numbers in an experiment RESULTS; they
   set how much of milestone 2 is worth.
1. **Page lifetime.** The pool, sequences, free and reference counts; the generator frees at EOS. Gate: a thousand
   requests through the worker with flat device memory, and a test that a shared page outlives the request that
   allocated it.
2. **Per-tile lookup.** Gate: decode attention over 64-token pages within 5% of the flat kernel at the same schedule
   on the 4080, and the corpus case moves.
3. **Batched block table.** The second table axis in the hint, the memory protocol and the runtime; fragments at the
   serving widths; the server's decode loop over a batch. Gate: output tokens per second at concurrency 8 against
   concurrency 1 with the vLLM benchmark client, and per-request logits equal to the single-request path.
4. **Chunked prefill at width.** The fixed 16-row chunk exists; it needs recorded rows for its fragments on each
   card (a strict export refuses it today) and a width that follows the prompt through `q_len` rather than a
   static row count, so a long prompt is not 64 launches of 16. Gate: time to first token for a 1,024-token prompt
   on the 4080 under one second.
5. **Prefix reuse.** The hash chain and eviction. Gate: a repeated system prompt skips its prefill, measured as time
   to first token; reference counts hold under concurrent requests.
6. **Tensor-core forms.** `mma` attention staging rows from pages with `cp.async`; TMA stays refused. Gate: the flash
   form over pages within 10% of flat on an H100 at 4,096 keys.
7. **Serving comparison.** The 4080 experiment's recipe at input 32/256/1024 and concurrency 1/8/32, this path against
   stock vLLM on the same card, published as an experiment.

Milestones 1 and 2 are independent and can run in parallel; 3 needs both; 4 and 5 need 3; 6 needs 2.

## Decisions for the author

- Page size: fixed per export as now, defaulting to the KV tile (16 or 32 tokens), or chosen per model by the golden.
- Where the scheduler lives: this plan puts it in the Rust server, since the native path is Python-free after export;
  the vLLM plugin path keeps vLLM's scheduler and would use only the addressing and the allocator.
- Batch widths: reuse the serving-twin widths the recipes already declare, or compile per observed batch.
- TMA on pages: refuse, as planned, or encode one descriptor per page at a cost of one descriptor slot per page.

## Risks

- Volta's tensor-core staging has miscompiled silently; every recorded paged row on sm_70 needs `--strict`.
- Paged kernels are new identities on every card, so each card needs rows before a strict export; the V100 flow in
  the paged-cache experiment is the template.
- One graph per symbol environment: batch widths and chunk sizes multiply the graphs; lengths must be inputs, not
  symbols, or every request shape captures its own graph.
- Sixteen-row prefill dominates time to first token until milestone 4 lands; a serving comparison before it
  measures that, not paging.
- The tracer records `permute` as a plain reshape, which is right only where the permute is a memory no-op. Every
  multi-row module here uses `transpose`; a batched fragment that permutes a batch axis must too, until the tracer
  is fixed, and a one-row test cannot catch the mistake.
