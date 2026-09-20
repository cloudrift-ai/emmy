# Shape and memory evidence

All four configurations complete the same workload matrix. The qualified Emmy schedules are slower than stock.
Reducing activation capacity from 1,024 to 256 recovers 1.14 GiB of reported KV-cache space, but the single-token
execution tier does not produce a consistent latency improvement with these schedules.

## Repeated measurements

The protocol, source revision, system, and row stems are in [RESULTS.md](RESULTS.md). Raw files are in
[results_rtx4080x1.tar.gz](results_rtx4080x1.tar.gz), under `2026-09-19_06-52-47/<row>/`.
Each row below uses `c<concurrency>_i<input>_r{1,2,3}.json`. All repeats completed eight requests with 64 generated
tokens per request. Values are ranges across three repeats. “M1” denotes the single-token tier.

| Configuration | Concurrency | Input | Median TPOT, ms | Median TTFT, ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Stock | 1 | 32 | 2.273–2.281 | 7.085–7.821 | 422.1–426.3 |
| Stock | 1 | 256 | 2.299–2.307 | 9.851–10.369 | 409.6–413.4 |
| Stock | 1 | 1024 | 2.468–2.476 | 27.911–28.636 | 346.3–348.0 |
| Stock | 4 | 32 | 2.489–2.500 | 9.887–12.279 | 1503.5–1512.2 |
| Stock | 4 | 256 | 2.657–2.682 | 21.359–23.026 | 1321.4–1324.6 |
| Stock | 4 | 1024 | 3.545–3.570 | 47.608–50.381 | 881.8–891.8 |
| Width 16 | 1 | 32 | 7.935–8.189 | 100.938–105.158 | 102.7–106.5 |
| Width 16 | 1 | 256 | 8.385–8.392 | 65.978–67.294 | 107.4–107.7 |
| Width 16 | 1 | 1024 | 8.564–8.578 | 251.976–252.939 | 80.7–80.8 |
| Width 16 | 4 | 32 | 9.224–9.265 | 157.320–157.793 | 343.1–344.7 |
| Width 16 | 4 | 256 | 9.484–9.509 | 191.089–191.329 | 320.2–320.7 |
| Width 16 | 4 | 1024 | 18.067–18.094 | 447.516–448.876 | 155.9–156.5 |
| M1, capacity 1024 | 1 | 32 | 8.368–8.378 | 106.579–107.671 | 100.7–101.0 |
| M1, capacity 1024 | 1 | 256 | 8.365–8.411 | 65.715–66.122 | 107.4–107.9 |
| M1, capacity 1024 | 1 | 1024 | 8.567–8.587 | 251.699–252.843 | 80.6–80.9 |
| M1, capacity 1024 | 4 | 32 | 8.419–9.281 | 158.729–221.292 | 339.9–342.8 |
| M1, capacity 1024 | 4 | 256 | 9.461–9.494 | 190.200–190.418 | 320.8–321.7 |
| M1, capacity 1024 | 4 | 1024 | 17.050–17.081 | 420.230–421.553 | 165.1–165.8 |
| M1, capacity 256 | 1 | 32 | 7.937–7.957 | 99.865–100.158 | 106.3–106.5 |
| M1, capacity 256 | 1 | 256 | 7.959–7.961 | 62.361–62.940 | 113.3–113.4 |
| M1, capacity 256 | 1 | 1024 | 8.119–8.126 | 236.763–237.276 | 85.4–85.5 |
| M1, capacity 256 | 4 | 32 | 8.043–9.266 | 113.548–150.423 | 361.1–384.1 |
| M1, capacity 256 | 4 | 256 | 8.992–9.040 | 178.629–179.002 | 338.6–339.2 |
| M1, capacity 256 | 4 | 1024 | 17.008–17.023 | 419.017–420.154 | 165.9–166.2 |

The static single-token pre/post schedules total about 478–480 µs in isolated checks, versus about 237 µs for
width 16. Less padding therefore does not imply less time with the recorded schedules. The small-capacity run has
lower single-request latency than the larger M1 run, but profiles do not separate allocator placement, GPU clock,
and cache effects. No universal speedup is attributed to capacity alone.

## Mixed lengths

Each `mixed.json` contains one eight-request trial with 3,434 total input tokens and 512 output tokens at
concurrency four. These are single observations, not repeat ranges.

| Configuration | Median TPOT, ms | Median TTFT, ms | Output tokens/s |
| --- | ---: | ---: | ---: |
| Stock | 2.860 | 26.802 | 1179.8 |
| Width 16 | 12.407 | 260.087 | 236.8 |
| M1, capacity 1024 | 11.753 | 242.887 | 250.1 |
| M1, capacity 256 | 11.742 | 242.021 | 250.4 |

Actual useful/padded rows and partial-prefill chunk occupancy were not instrumented. Request lengths alone do not
establish internal occupancy. The source dispatch policy and configured widths are not substitutes for that measure.

## Memory

`server.log` reports the KV budget and token capacity; `memory.txt` records whole-device use after the generation
probe. All runs use a 0.6 GPU-memory-utilization setting and an actual scheduler limit of four sequences.

| Configuration | KV budget, GiB | KV tokens | Graph capture, GiB | Whole-device used, MiB |
| --- | ---: | ---: | ---: | ---: |
| Stock | 8.12 | 76,032 | 0.08 | 10,441 |
| Width 16 | 6.17 | 57,776 | 0.05 | 10,277 |
| M1, capacity 1024 | 6.18 | 57,856 | 0.05 | 10,345 |
| M1, capacity 256 | 7.32 | 68,528 | 0.05 | 10,366 |

The smaller capacity recovers 10,672 KV tokens, about 18.4% over the larger M1 configuration. This recovery is
available within the existing integration; it does not require a native scheduler. Stock still has 7,504 more
cache tokens than the small-capacity configuration. Whole-device snapshots include desktop use and an expanded
KV allocation, so their difference is not a direct activation-memory measurement.

Activation and scratch allocation categories were not separately measured. Summing exported buffers would count
shared storage incorrectly. The experiment also does not measure admission beyond four sequences. These limits
prevent a native-runtime memory claim even though the KV-capacity improvement is directly observed.
