# V100 decode: next steps

Context: PR #1078 put Qwen3-0.6B layer-0 FP16 decode on a Tesla V100 SXM2 16GB at 53.93 µs against 55.08 µs for
`torch.compile` (median of seven fresh-process pairs, nine saved Inductor choices). The win is an exact one-key
softmax fold in normalization, the fused attention + output cut (eight launches) and `coop/v<n>` rows that read
weights as vector loads. The lead is 1.1 µs, about 2%. Each item below names the measurement that motivates it.

## 1. A prior feature for `coop/v<n>`

`coop/v<n>` is taken only where a row or a pin names it. The prior featurizes a run exactly like its plain band, so
offering runs unpinned tied every band with its runs; the tie-pessimistic reproduction gate then dropped two Llama
V100 schedule nodes (`k_linear_mean_reduce_9616db__place_741ced04c7`: rank 65 of 112, top 50% required).

- Add `D_reduce_run = log2(columns)` beside `D_reduce_transposed` in `_reduce_features`, bump the featurizer version.
- Let nightly refit both priors on the new datasets; then drop the site rule that hides runs from the catalog.
- Evidence the prior should learn from: isolated V100 rows, K/V 4.4 → 4.0 µs (`t128 coop/v8`), attention with
  output 5.5 → 4.5 µs (`t128 coop/v4`), down 6.9 → 5.5 µs (`t256 coop/v2`), gate/up 17.0 → 16.7 µs
  (`t256 coop/v2`). `t256 coop/v8` lost on every kernel (7.0, 6.4, 10.4 µs): a run longer than `K / coop` masks.
- Record the same rows on the other s1 goldens (A100, H100, RTX 4090/5090), which still read weights two bytes per
  lane per load.

## 2. Gate/up is at the card's bandwidth

Gate/up streams 12.6 MB. Emmy takes 17.4 µs in the layer, Inductor 17.1 µs (nsys, graph replay); under Nsight
Compute with flushed caches and base clocks, 21.1 µs against 19.6 µs. No thread count, ILP chain or run moved it
by more than 0.3 µs. The remaining lever is the kernel boundary, not the loop:

- Fuse the post-attention RMS statistics (`7c0ee4456e`, one CTA, 2.4 µs) into gate/up as a recomputed per-CTA row
  statistic. The earlier attempt repeated the whole residual per output row and took 146.6 µs; a cut that keeps the
  statistic in one per-CTA prologue over the 1024-element row is the shape to try.
- Same for the input statistics (`5ede0cd567`, 2.3 µs) in front of Q and K/V.
- Both are 1-CTA kernels at 2.3–2.5 µs where Inductor's equivalents take 1.6–1.8 µs; vectorizing their scalar
  `__half` loads (`coop/v8` on a `t128` band) is a cheaper first try.

## 3. The nsys graph-tracing slowdown

With `--cuda-graph-trace=node`, Emmy's whole-layer replay read 66.6 µs per replay against 56 µs unprofiled, while
Inductor's read 60.2 against 55.4. Small Emmy kernels in the layer graph read 3.1–4.1 µs against 2.4–2.6 µs alone.
Unexplained. Check whether Emmy's driver-API graph nodes carry attributes Inductor's do not (cache config: one Emmy
kernel, `8da2228835`, runs with no shared memory and a different carveout from its neighbours), and whether
`--cuda-graph-trace=graph` shows the same gap. Until this is understood, per-kernel profiling of the Emmy layer
under nsys is not trustworthy.

## 4. Real KV-length decode

The article's benchmark decodes one token against one key, so the softmax has one element and the fold applies.
Real decode attends over the whole cache. Add a recipe row at KV lengths 512 and 4096 (one new token, cached keys
and values as inputs) on the V100, measure Emmy against `torch.compile`, and record the attention kernel there.
The vector reads carry over to the projections; the attention kernel is new work.

## 5. The bench harness's unequal windows

`_bench_interleaved` times each torch backend over the largest calibrated batch (97 graph replays per window on this
layer) and Emmy's whole program over `round(1 ms / iter)` replays (16). Under nsys neither showed a first-replay
penalty large enough to matter, but the windows should be equal by construction: time Emmy's whole-program window
over the same replay count the torch side uses.

## 6. `make test` on a 6-core V100

The suite stalled on serving generation tests: `test_attention_reads_only_the_written_cache_prefix` compiled for
85 minutes on main before it was killed (kernel-set pricing in `_route_candidates`). Find why route pricing for
that program takes that long on sm_70.
