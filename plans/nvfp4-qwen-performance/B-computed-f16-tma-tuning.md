# Computed f16 activation beside TMA weight copies: quick timing (B-computed-f16-tma)

Measured on 2026-09-29 on an idle desktop RTX 5090 (sm_120, CUDA 13.0, 600 W limit, about 30 °C at start). Base is
`567a42ad`; the PR is branch `fix/B-computed-f16-tma` at `ce4c09a1`. Every row is hand-pinned; no default pick, prior
ranking or autotune result is used. Compiles use the deployable flags (`-O3`, no `EMMY_NVCC_FLAGS` override), with
fast math on and repository goldens off (`EMMY_GOLDEN_FILE=`). Each row is `emmy run -c "$PROG" --bench --warmup 50
--iters 500`, run three times; the numbers are the kernel's time from the bench table. The three repeats never
differed by more than 1.5 %, so each cell below shows the median.

## Program and pins

The program is the report's `PROG`: `x + 1` shared by two `nn.Linear(4096, 1024)` weights whose outputs multiply.
Decode uses `x` of 16 × 4096, prefill 2048 × 4096. Weights come from the program's own seeded initialization, the same
on both trees. Every row pins `PLACE=fuse,REDUCE=,TILE=<tile>,WORK=<work>,STAGE=<stage>` and realizes as one fused
kernel with the pinned stage. Correctness is established separately: the bitwise parity test between the compute
fill and TMA, the corpus `correct` stage, and the report's strict command under `EMMY_FAST_MATH=0`.

## Existing schedules before and after

`d1/smem` and `d2/smem` emit byte-identical CUDA on both trees, and time the same within 1 %:

| Shape | `TILE` / `WORK` | `d1/smem` base → PR (µs) | `d2/smem` base → PR (µs) |
| --- | --- | --- | --- |
| decode | `f1x4/k2` / `w2x1` | 51.7 → 51.5 | 48.1 → 48.2 |
| decode | `f2x2/k2` / `w1x2` | 51.8 → 51.8 | 48.3 → 48.3 |
| prefill | `f4x4/k2` / `w2x2` | 214.4 → 215.5 | 222.6 → 223.2 |
| prefill | `f4x4/k4` / `w2x2` | 213.7 → 213.3 | 216.5 → 216.2 |

All tiles are `mma_m16n8k16_f16_f32/…`. A packed W4A16 row (`--quantize nvfp4-w4a16`,
`TILE=mma_m16n8k16_f16_f16/f4x8/k2,STAGE=d2/smem-tma/p2`) also times the same, 551.3 → 550.3 µs. Its source differs
only in where the barrier array is declared.

## Available options before and after

Base offers only the compute fill with cp.async weight copies (`d1/smem`, `d2/smem`; not the `smem-async` transport).
The PR adds TMA weight copies at depths one and two, with and without `/p2`. A dash means not measured:

| Shape | `TILE` / `WORK` | `d1/smem` | `d2/smem` | `d1/smem-tma` | `d2/smem-tma` | `d2/smem-tma/p2` |
| --- | --- | --- | --- | --- | --- | --- |
| decode | `f1x4/k2` / `w2x1` | 51.5 | 48.2 | 57.1 | 58.7 | 58.8 |
| decode | `f1x4/k8` / `w2x1` | 37.4 | 37.1 | — | 38.2 | 38.1 |
| decode | `f2x1/k8` / `w1x4` | 23.4 | **20.6** | 24.8 | 24.8 | **24.8** |
| prefill | `f4x4/k2` / `w2x2` | 215.5 | 223.2 | 225.2* | 223.7 | 223.7 |
| prefill | `f4x4/k4` / `w2x2` | **213.3** | 216.2 | **221.1** | 223.2 | 222.6 |

\* Measured before the barrier-array fix below; registers, not shared memory, limit this row's residency.

Best tried: decode 20.6 µs on base, 24.8 µs with TMA (20 % slower). Prefill 213.3 µs on base, 221.1 µs with TMA (4 %
slower). The best PR option overall is still an existing compute fill with cp.async copies, so the PR adds no measured
speedup here.

## What limits TMA here

- **Short K chunks.** At a 32-element chunk the existing all-stored TMA ring is also slower than cp.async (decode
  gate/up without `+ 1`, `f2x2/k2`, `d2`: 29.3 µs with cp.async, 38.2 µs with TMA). Each chunk pays an elected-thread
  issue and a barrier wait, which dominates at small M.
- **Barrier stalls at decode.** On `f2x1/k8` Nsight Compute shows about three times the barrier stall time with TMA
  (`smsp__average_warp_latency_issue_stalled_barrier`: 10.7 k against 3.8 k). A likely cause is the thread that issues
  the box copies holding its warp back at the per-chunk barrier; this was not investigated further.
- **A slowdown fixed in this PR.** The first version declared the ring's barrier array between the copied and the
  filled slabs, and the next 1024-byte-aligned slab added about 1 KB of padding. At `f4x4/k4` prefill that dropped
  residency from two blocks per SM to one: `d2/smem-tma` took 290 µs against 216 µs for `d2/smem`. Declaring the
  barrier array last brought it to 223 µs.

## Cut workaround, for context

`PLACE@map.1/inner.1/map=cut,STAGE=d2/smem-tma` with the rest left to the default pick measures 0.8 µs for the
`x + 1` kernel plus 18.9 µs for the matmul (`f2x1/k8`, `w1x4`) at decode, 19.7 µs in all. At prefill it picks an
f16-accumulate atom (6.2 + 106 µs), which is a different precision and not comparable. These are single isolated
kernels with hot weights; they do not predict serving throughput.
