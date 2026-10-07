# V100 LoRA serving: next steps

Llama 3.1 8B Instruct, rank-8 `limo` adapter, one Tesla V100-SXM3-32GB, FP16, 511 input / 256 output tokens.
Golden: `experiments/Meta-Llama-3.1-8B-Instruct/lora_v100_sxm3_32gb/golden/v100_sxm3_sm70_lora8.json`.
State after PR #1080: adapter and mixed requests run 1.6–2.1 times faster than stock vLLM. Base requests match
stock at concurrency 1. With the 16-row output-projection split, base time per output token matches stock at
concurrency 8 and 16, and base throughput stays about 4% below stock. First-token time causes the rest.

## 1. 512-row post-attention program (first-token time)

Measured: 3040 µs Emmy against 2619 µs `torch.compile` (isolated, strict replay). Base first-token time at
concurrency 1 is 151–155 ms against stock's 126.5 ms. Of the 25–28 ms gap, about 13 ms is the 32 layers' 512-row
programs (post 3040 vs 2619, pre 473 vs 490). The rest is not yet attributed.

- The gate/up projection runs about 1540 µs in the layer: 120 GFLOP at about 78 TFLOP/s. The down projection runs
  about 814 µs and the output projection about 245 µs. All three use `mma_m8n8k4` with `STAGE=d2/smem`.
- Read the generated kernel: global loads go straight into a shared-memory store, so each warp waits for its loads
  before the barrier and the tensor-core work. Try a register-staged prefetch on Volta (load the next K slab into
  registers before the MMA loop, store it after), which the synchronous-copy staging does not do today.
  This is a lowering change in the staged K loop, not a new knob.
- Profile one gate/up launch with `sudo ncu` against the cuBLAS/CUTLASS kernel `torch.compile` calls, and compare
  achieved tensor-pipe utilization and shared-memory bank conflicts before changing the lowering.
- Attribute the other ~12 ms of first-token time: profile one 511-token prefill in the serving container with the
  PyTorch profiler (the earlier `profile-opt6` traces in the archive are from an older golden).

## 2. `--record-greedy` records the minimum sample

`_record_greedy_pick` writes `min(launch.samples)` for each kernel. For the 512-row gate/up kernel that recorded
about 1394–1397 µs while every median measured since is 1535–1558 µs. Golden rows are ranked against each other, so
an optimistic minimum can choose the wrong schedule.

- Record the median (what the bench tables report), or record both and rank by the median.
- The same 1395.712 µs appeared twice: as the recorded minimum for `WORK=w2x2, TILE=f4x2` (record run) and as the
  `--ab` total for `WORK=w4x4, TILE=f2x2` an hour later in another process. Fresh runs of both from empty tune
  databases and kernel caches measured 1535–1558 µs, so this is not a cache or keying bug. Bench timings come in
  32 ns steps, and an identical outlier 10% below the median in two processes suggests a timing artifact in
  single-launch event timing. Reproduce: bench the gate/up kernel 20 times with `--warmup 20 --iters 50`, collect
  the per-sample timings, and check whether the low tail clusters on one value.

## 3. Remaining base gap at concurrency 8 and 16

- Decode batches of 2–16 requests all run the 16-row program; 8 requests pay for 16 rows. Stock runs M=8 GEMMs.
  Try an 8-row decode program (another decode bucket) and measure base time per output token at concurrency 8.
- 16-row gate/up projection: about 277 µs (235 MB of weights, about 850 GB/s). The one-row form reaches
  940 GB/s with the transposed cooperative reduction (`REDUCE=…/coop-t/v4`), which is not offered at 16 rows.
- 16-row down projection: 134 µs (117 MB, about 875 GB/s).

## 4. Smaller items

- Small adapter kernels: splitting the rank-8 shrink kernels saved 1–2 µs each. Several small kernels per layer
  remain (rms-norm statistics, adapter expand, residual adds); fusing them is a placement question for evidence.
- `--tune` refuses a program with more than one scheduled kernel, so a split kernel (partial plus finalize) cannot
  be tuned with it. A finalize with a single schedule row could be ignored by the tuner.
- The source weight layout (`LAYOUT=source`) was not evaluated: its kernels had no rows, so the greedy pick ran
  untuned schedules (8 ms for post-attention). Tuning those kernels needs them stored in a working golden first.
