# OLMoE-1B-7B-0125-Instruct with a GPU cache of recently routed experts, RTX 5090 at PCIe 5.0 x16

## Result

At the same number of GPU experts per layer, a cache of the most recently routed experts decodes faster than milestone
2's static hot set: by 14%, 33% and 29% on chat at 48, 32 and 16 experts, and by 9%, 12% and 5% on code. Adding router
prediction, which copies the experts each layer's router predicts into the cache on a side stream during the layer's
attention, makes every row slower than the cache alone. Decode is bound by the bytes it moves over the link, and the
prediction moves more of them. Every row gave the same 16 greedy answers as the all-GPU row.

| GPU experts per layer | Chat: static | Chat: cache | Chat: cache + prediction | Code: static | Code: cache | Code: cache + prediction |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 64 (all) | 2.77 ms | | | 2.83 ms | | |
| 48 | 7.00 ms | 5.99 ms | 7.27 ms | 5.64 ms | 5.15 ms | 6.28 ms |
| 32 | 15.51 ms | 10.36 ms | 12.61 ms | 9.36 ms | 8.25 ms | 10.48 ms |
| 16 | 26.62 ms | 18.77 ms | 22.41 ms | 15.35 ms | 14.60 ms | 18.19 ms |

TPOT is the median over 8 requests. The static rows repeat hot_set_rtx5090's within 0.1 ms on a different VM of the same
type.

**The offline replay predicts the cache.** expert_cache_sim_rtx5090 counts host expert reads per decode step on the
same prompts. With hot_set_rtx5090's fit, TPOT ≈ 2.8 ms + 0.277 ms per host read, the replay's LRU counts give 5.4,
10.3 and 19.4 ms on chat (measured 5.99, 10.36, 18.77) and 4.2, 8.0 and 14.9 ms on code (5.15, 8.25, 14.60). At 48
experts the measured rows run about 0.5–1.0 ms over the fit. The cache adds about 15 small device ops per layer per
step, which probably costs that; it is not profiled.

**Prediction loses on bytes.** In the replay, prediction cuts the reads a step waits for by half to four fifths but
adds more prefetch copies than that. Both share the one link, so the step waits for the prefetch traffic as well. It
would pay only where the link is idle for part of the step, which at one stream on this model it is not.

**What it costs.** The cache keeps a host copy of every expert (12 GiB pinned for OLMoE), and prefill reads every
expert from host memory, because the device slots change owner during decode. TTFT rose from 104–241 ms (static) to
about 300 ms on chat (36–41 ms all-GPU).

What this means for the plan:

- **A cache beats a static set, and a better eviction rule adds little.** The replay's Belady bound reads half to two
  thirds of what LRU reads at 32 and 16 experts, but the best simple rule in expert_cache_sim_rtx5090 (a decayed count)
  recovers only 3–11% of LRU's reads. Cutting host reads further means running cold experts where their weights are
  (strategy G), not choosing them better.
- **Prediction (strategy E) stays out at one stream.** It may return at batch widths or on a model whose expert compute
  covers the link time, where the link idles between reads.
- **The copy and the bookkeeping belong in the runtime.** The prototype issues the cache step as torch ops and a Triton
  copy kernel per layer; the design milestone moves them into the tier manager.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/expert_cache_rtx5090 --ssh …` served
`allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's
vLLM 0.23 plugin, from the image built at commit `51e953dbc` (image id `sha256:2fbb4de5fe48…`), which holds both the
cache and the prediction prefetch. Serving shape, strict evidence, `rank.json`, workloads and client are those of
hot_set_rtx5090: 8 chat and 8 code prompts at concurrency 1, 64 output tokens, `--ignore-eos`, temperature 0. Each row
is one server boot of the same image; only `EMMY_GEN_EXPERTS_GPU`, `EMMY_GEN_EXPERTS_CACHE` and
`EMMY_GEN_EXPERTS_PREFETCH` differ.

The cache starts from the static set. On a decode step, each routed expert it does not hold evicts the least recently
routed slot, is copied there from host memory by a Triton kernel on the decode stream, and its table entry moves to
the slot; the evicted expert's entry moves back to its host copy. All of it is fixed-shape device work, so vLLM's
decode graph records it. The prediction applies the layer's router, through its post-attention norm, to the hidden
state entering the layer, and runs the same cache step on a side stream that the layer's own step waits for.

## Limitations

- Decode at one stream only.
- The cache's hit rate is not logged; the routing histogram counts selections, not hits. The offline replay gives the
  expected counts.
- Two rows (cache with prediction at 48, static at 32) failed in the first pass when the dataset server returned 502
  and were re-run on the same VM and image; they are the second and third run roots.

## Run and platform

Runs `2026-10-16_23-06-11`, `2026-10-16_23-55-36` and `2026-10-17_00-01-00`, ten rows, all succeeded. One RTX 5090
(32 GB, driver 580.159.03, PCIe 5.0 x16) in a CloudRift VM on a Ryzen 7 7800X3D with 7 cores and 23 GB of RAM plus a
32 GB swap file, kernel 6.8.0. `results_rtx5090x1_x16.tar.gz` (Git LFS) holds the three run roots: experiment
records, benchmark logs, and per row `bench_chat.json`, `bench_code.json`, `image.json` and `server.log`. Node addresses
read `node`.
