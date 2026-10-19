# OLMoE-1B-7B-0125-Instruct served through generic paging

Milestone 7's gate of the MoE host-experts plan: does the generic paging mechanism — paged E-stacked expert weights,
an inspector and compiled paging programs spliced ahead of each expert launch, least recently used out — decode at
least as fast as milestone 3's hand-built expert cache (expert_cache_rtx5090), and does it change no token?

## RTX 5090 x1, PCIe 5.0 x16

### Result

Yes on both counts. At every budget the paged rows decode faster than the milestone 3 cache rows, on chat and on code,
and every paged row gave the same 16 greedy answers as the all-GPU row of the same run.

| GPU experts per layer | Chat: paging | Chat: M3 cache | Code: paging | Code: M3 cache |
| ---: | ---: | ---: | ---: | ---: |
| 64 (all) | 2.10 ms | 2.77 ms | 2.17 ms | 2.83 ms |
| 48 | 5.61 ms | 5.99 ms | 4.89 ms | 5.15 ms |
| 32 | 9.87 ms | 10.36 ms | 7.72 ms | 8.25 ms |
| 16 | 17.80 ms | 18.77 ms | 13.60 ms | 14.60 ms |

TPOT is the median over 8 requests of 64 output tokens at concurrency 1; the M3 column is expert_cache_rtx5090's cache
rows, measured on a VM of the same type.

**The comparison is directional, not a controlled A/B.** The all-GPU row is 0.67 ms faster here than in the M3 run
(2.10 against 2.77 ms on chat): the trunk got faster between the two builds. Subtracting each run's all-GPU row, the
time spent on host-resident experts is 3.51 / 7.77 / 15.70 ms on chat at 48 / 32 / 16 experts here against 3.22 /
7.59 / 16.00 ms for the M3 cache, and 2.72 / 5.55 / 11.43 ms on code against 2.32 / 5.42 / 11.77 ms. So paging
spends about the same on host experts as the cache did — 0.1–0.4 ms more at 48 and 32 experts, 0.3 ms less at 16 —
and the gate passes on the faster trunk it now runs with.

**Repeat variation.** Within each row the mean TPOT is within 0.5 ms of the median; the p99 reaches 7.17 ms at 48
experts on chat and 10.4–10.7 ms at 32 experts, the requests whose routing misses most. One run per row.

**Prefill.** TTFT is 275–312 ms on every paged row against 38–41 ms all-GPU: prefill reads every expert from host
memory, as with the M3 cache.

### What it took

Earlier gate runs on this VM, with intermediate builds of the same branch, measured the paging programs' own cost
(those runs are not archived; the numbers are from their records): 6.43 / 2430 / 63 ms on chat at 48 / 32 / 16 experts
before the paging programs had golden rows (a prior-picked single-thread decision kernel took 55 ms per step), then
11.67 ms at 32 experts with recorded rows, 6.90 / 11.19 / 19.03 ms once single-cell kernels spread over their block,
and these rows once the decision program lost its scans. Each paged expert launch runs about 20 µs of paging kernels
per step on this card when nothing misses.

### Limitations

- One run of 8 requests per workload and row; no repeated runs.
- The M3 rows come from another run and build (see above).
- The VM has 23 GB of RAM and the engine pins a host copy of every expert (12 GiB). Each paged row's server took
  about 30 minutes to boot while the compile swapped; decode itself showed no swap traffic.
- A first attempt at this run lost its 48-expert row to a transient HTTP 502 from the Hugging Face dataset server
  that the prompt fetch now retries; the archived run has every row.

### Run

- Run directory (archive root): `2026-10-18_10-53-10`, started 2026-10-18T10:53:10Z, source revision `726bed3cb`.
- Image `cloudriftai/vllm-emmy:0.23.0-726bed3cb`; OLMoE revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`, fp16,
  vLLM generate runner with full decode graph capture, strict evidence over the OLMoE recipe golden and the 5090
  hardware golden, fast math off.
- Machine: one NVIDIA GeForce RTX 5090 (32 GB, driver 580.159.03) at PCIe 5.0 x16, AMD Ryzen 7 7800X3D (7 vCPUs),
  23 GiB RAM, Ubuntu 24.04.1, cuBLAS 13.6.
- Rows: `experts_gpu` all (unset), 48, 32, 16 — four rows, all `succeeded`.
- Archive: `results_rtx5090x1_x16.tar.gz`, members under `2026-10-18_10-53-10/`: one `*.experiment.yaml` per row,
  each row's `bench_chat.json`, `bench_code.json`, `image.json` and `server.log`, and `benchmark.log` /
  `benchmark_rtx5090_x_1.log`. Host addresses are replaced with `node`.
