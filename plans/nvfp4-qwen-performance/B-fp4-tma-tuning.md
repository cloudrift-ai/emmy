# B-fp4-tma: hand-pinned timing of native fp4 TMA staging

The quick performance check of assignment B-fp4-tma ([report](fp4-cell-no-tma.md)): the native fp4 tensor-core atom
`mma_m16n8k64_e2m1_f32` with its stored codes and block scales copied by cp.async (`smem-async`) or by TMA
(`smem-tma`).

Device: desktop RTX 5090 (sm_120), CUDA 13.0, deployable `-O3` (no `EMMY_NVCC_FLAGS`). Base `567a42ad`, PR branch
`fix/B-fp4-tma` at `eb6515a4`. A fresh tune DB and online prior, not shared with other runs. Clocks were not locked; the
GPU was otherwise idle.

## Programs and method

The report's two programs at K = 4096 under `--quantize nvfp4`, each linear 4096 → 4096. `one`: a single linear over
a separately encoded activation. `two`: two linears reading `a = x + 1`, multiplied, which fuses into one contraction
with two weights. Decode shapes use 16 rows, prefill shapes 2048 rows. Weights are the random snapshot
`--quantize` writes per run; timing does not depend on their values.

```sh
EMMY_KNOBS="TILE=$TILE,WORK=$WORK,STAGE=$STAGE" \
  emmy run --quantize nvfp4 -c "$PROG" --bench --warmup 50 --iters 500
```

`STAGE=dN/<transport>[/p2]`: `dN` is an N-slot shared-memory ring, `/p2` double-buffers the mma fragments in
registers. Decode pins `TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,WORK=w1x2` (the report's tile). Prefill pins
`TILE=mma_m16n8k64_e2m1_f32/f4x4/k4,WORK=w2x2` (a 128 x 64 tile). The times below are the matmul kernel's own row of
the bench table, in µs. A second run of a row is shown after a slash.

Correctness is established separately: the TMA and cp.async kernels agree bit for bit on the same buffers at
d1, d2/p2, d3, d4/p2 and a one-chunk stream, for one and two channels, and both hold the atom's declared tolerance
against the numpy reference (`test_fp4_tma_matches_cp_async_bit_for_bit`).

The program total is not reported. Under any global pin the activation encode kernel runs on one CTA (about 610 µs
in `two` at 16 rows): a global pin also reaches the encode kernel
([pins report](pins-cannot-target-one-piece.md)). That is not a property of either transport.

## Existing schedule, base vs PR (cp.async)

The cp.async CUDA of both report programs is byte-identical on the two revisions. Timings agree within noise:

| Program | Shape | STAGE | Base | PR |
| --- | --- | --- | --- | --- |
| one | 16 rows | d3/smem-async | 6.9 | 6.8 / 7.5 |
| two | 16 rows | d2/smem-async | 9.7 | 9.1 |
| one | 2048 rows | d1/smem-async | 116.1 | 114.2 / 114.6 |
| two | 2048 rows | d2/smem-async/p2 | 158.7 | 160.2 / 161.8 |

## Available options, cp.async vs TMA (PR)

| Program | Shape, transport | d1 | d2 | d3 | d4 | d2/p2 |
| --- | --- | --- | --- | --- | --- | --- |
| one | 16 rows, cp.async | 7.8 | 7.8 | 6.8 / 7.5 | 7.6 | 7.2 |
| one | 16 rows, TMA | 9.5 | 7.4 | 7.7 | 7.4 / 7.4 | 7.2 / 7.3 |
| two | 16 rows, cp.async | 10.8 | 9.1 | 9.4 | 10.0 | 9.1 / 9.1 |
| two | 16 rows, TMA | 12.7 | 9.8 | 9.8 | 9.9 | 9.5 / 9.6 |
| one | 2048 rows, cp.async | 114.2 / 114.6 | 119.0 | 124.3 | — | 120.9 |
| one | 2048 rows, TMA | 141.4 / 141.3 | 150.8 | 151.7 | — | 151.7 |
| two | 2048 rows, cp.async | 165.6 | 164.0 | — | — | 160.2 / 161.8 |
| two | 2048 rows, TMA | 199.7 / 199.5 | 205.5 | — | — | 209.3 |

Best tried, cp.async vs TMA: `one` decode 6.8–7.5 vs 7.2–7.4 (even); `two` decode 9.1 vs 9.5 (TMA about 4% slower);
`one` prefill 114 vs 141 (TMA 24% slower); `two` prefill 160 vs 200 (TMA 25% slower). d3 does not fit the prefill
`two` tile; d4 was not tried at prefill.

## Why TMA loses at prefill

Nsight Compute on the `one` prefill kernel at d2: shared-memory load bank conflicts 1.05 M with cp.async and 45.1 M
with TMA; shared load wavefronts 9.4 M and 53.5 M. The cp.async fill pads each byte row by 16 B, which spreads the
per-lane byte loads that read the mma fragments out of shared memory across banks. A TMA box deposits dense
128-byte rows, so every row of a fragment starts on the same bank. A hardware swizzle on these byte buffers, with the
matching XOR in those loads, would likely remove this; it is not implemented.

These are isolated hot-cache kernel timings; they do not predict serving throughput.
