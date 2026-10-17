# OLMoE-1B-7B-0125-Instruct: one expert on the CPU backend, beside llama.cpp and the memory rate

## Result

With its weights cold in host memory, as a routed expert's are on a decode step, Emmy's CPU backend runs one OLMoE
FP16 expert within 1.07x of llama.cpp's CPU kernels at 1 and 4 rows on both machines, which passes milestone 4's gate
(1.5x). At 16 and 64 rows it is within 1.3x. Hot, at one row on the 16-core box, it is 2x behind: the weights then
come from cache, and the launch overhead of three kernels shows.

The same expert with NVFP4 weights and 16-bit activations (W4A16) runs at 1.6x the time its bytes take at the
measured read rate on the 16-core box and 3.6x on the 7-core VM, at one row. Its decode, not memory, bounds it: hot
and cold times are close. llama.cpp's NVFP4 kernel is not the same computation — it quantizes the activations to
8 bits and applies each block's scale once to an integer sum — so it is shown for reference, not as the gate.

Cold, the three projections (gate, up, down), microseconds; llama.cpp is the same three matrix products:

| Machine | Weights | Rows | Emmy | llama.cpp | Emmy / llama.cpp | Bytes / read rate |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Ryzen 9 9950X3D, 16 threads | FP16 | 1 | 210 | 198 | 1.06 | 198 |
| | FP16 | 4 | 222 | 211 | 1.05 | |
| | FP16 | 16 | 272 | 268 | 1.01 | |
| | FP16 | 64 | 914 | 718 | 1.27 | |
| | NVFP4 W4A16 | 1 | 91 | 53 | 1.71 | 56 |
| | NVFP4 W4A16 | 4 | 169 | 71 | 2.38 | |
| | NVFP4 W4A16 | 16 | 337 | 255 | 1.32 | |
| | NVFP4 W4A16 | 64 | 1457 | 1006 | 1.45 | |
| Ryzen 7 7800X3D VM, 7 threads | FP16 | 1 | 256 | 253 | 1.01 | 253 |
| | FP16 | 4 | 273 | 256 | 1.07 | |
| | FP16 | 16 | 742 | 623 | 1.19 | |
| | FP16 | 64 | 2368 | 2353 | 1.01 | |
| | NVFP4 W4A16 | 1 | 255 | 61 | 4.19 | 71 |
| | NVFP4 W4A16 | 4 | 449 | 175 | 2.57 | |
| | NVFP4 W4A16 | 16 | 992 | 660 | 1.50 | |
| | NVFP4 W4A16 | 64 | 3645 | 2634 | 1.38 | |

Emmy's time is the best `CPU_TILE` per row (the sweep the evidence pick reads); the SiLU-and-product kernel between
the projections adds 4-25 us more. "Bytes / read rate" is the expert's weights (12.6 MB in FP16, 3.54 MB in NVFP4
with its e4m3 scales) over the read rate llama.cpp's cold FP16 expert reaches at one row: 63 GB/s on the 9950X3D,
50 GB/s on the VM. STREAM, which counts the writes it also makes, gives 59 and 49 GB/s copy, 44 and 38 GB/s triad.

The 7800X3D VM at 6 threads (one core left to serving) is within 2% of 7 threads on FP16 at 1 and 4 rows, and up to
21% slower at 16 and 64.

What this means for the plan:

- **FP16 experts are ready for strategy G.** A cold expert costs about what streaming its weights costs, on both
  machines; nothing in the kernel is left to win at 1-4 rows.
- **NVFP4 experts cost 1.6-3.6x their bytes at one row.** The decode bounds them, more so on a machine with fewer
  cores. Strategy G on Qwen3.8-Flash-Next pays that price; closing it needs a cheaper decode (a byte shuffle for the
  nibbles), not more bandwidth.
- **The tile choice matters and differs by kernel.** Across the sweep the fastest tile ranged over 1x1, 2x2, 4x1, 8x1 and 2x4 by
  kernel, row count and machine; the evidence pick takes it from measured rows.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/cpu_expert --local --filter cpus=0-15` on the dev box and `--ssh`
`--filter 'cpus=0-[56]'` on the VM, from the serving image built at commit `ed351f639`. Everything ran in that image
with `--cpuset-cpus` set to the matrix's CPUs, so the CPU backend (one thread per physical core it may use), STREAM
(OpenMP on every CPU) and llama.cpp (patched to one thread per CPU) all used the same cores.

- STREAM at commit `6703f75`, 200M-element arrays, 20 repetitions.
- llama.cpp at commit `ea0c4c2`, `test-backend-ops perf -o MUL_MAT -b CPU`, with the expert's three projections
  (1024x2048 and 2048x1024, F32 activations) added to its perf cases at 1/4/16/64 rows, F16 and NVFP4: one weight
  repeated (hot), and 256 distinct weights per run, 1 GiB in F16 (cold); the cold time per projection is the run's
  over 256. With 64 distinct weights (256 MiB) llama.cpp read 94 GB/s on the 9950X3D: part of the loop stayed in its
  128 MiB of L3, so 64 is not cold there.
- Emmy: `emmy run --gpu-arch cpu --bench` on the expert as a module of three bias-free linears in FP16, its rows
  symbolic (`--dynamic rows@x:0`), and with `--quantize nvfp4-w4a16`; 10 warmup and 50 timed runs of each kernel
  alone, hot or `--cold-cache` (every thread first reads twice the last-level cache); `CPU_TILE` pinned to each of
  1x1, 2x1, 4x1, 8x1, 2x2, 4x2, 2x4, 4x4 in turn; `--json` per run. Every run passed the accuracy check: FP16 against
  eager torch, NVFP4 against the numpy backend on the same graph.

## Limitations

- One expert at a time, alone. In serving the CPU shares memory bandwidth and cores with the GPU-driving process and
  the other layers; milestone 5 measures that.
- Rows measured at one size; a symbolic kernel's rows at every size are pooled by the evidence pick.
- llama.cpp's NVFP4 kernel quantizes activations to 8 bits: a different computation from Emmy's W4A16.
- Two machines, both AMD Zen 4/5 with AVX-512.

## Run and platform

Runs `2026-10-17_02-53-20` (dev box: Ryzen 9 9950X3D, 16 cores on two CCDs, 128 MiB L3, DDR5, cpuset 0-15) and
`2026-10-17_02-53-22` (CloudRift VM: 7 vCPUs of a Ryzen 7 7800X3D, 96 MiB L3, cpusets 0-6 and 0-5), all rows
succeeded. `results_ryzen9_9950x3d.tar.gz` and `results_ryzen7_7800x3d.tar.gz` (Git LFS) hold the run roots: the
experiment records, `cpu.txt` (lscpu), `stream.txt`, `llama.txt`, `image.json`, and `emmy.tar.gz` with each run's
`--json` record and log. Node addresses read `node`.
