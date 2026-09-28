# NVFP4 decoding unnecessarily uses lookup tables

## Summary and priority

Emmy implements a fixed floating-point conversion as a data-dependent memory lookup. This is a compiler performance
bug: an E2M1 nibble already contains its sign, exponent and mantissa, so decoding it does not require a value table.
The scalar NVFP4 path reads a 256-by-2 byte-to-value-pair table from global memory. The staged W4A16 tensor-core path
extracts both nibbles but decodes them through a 16-entry CUDA constant-memory table before FP16/BF16 MMA.

Both forms introduce avoidable memory accesses and cache dependencies. Replace them with direct conversion or exact
bit/arithmetic decoding, preserving the packed operands and the existing scale-rounding semantics. Native W4A4 FP4
MMA already consumes packed codes and scales without reading a decode LUT; preserve that path.

**This is the last, lowest-priority report in the overview.** The preceding bugs are more pressing and are likely to
cause larger performance penalties or block deployment. **We have no measured evidence of how much speedup removing
these LUTs will deliver.** The current evidence is emitted IR and source inspection, not a timing comparison.

## Reproduce

Checked on 2026-09-29 in PR #961's existing `nvfp4-plans` worktree at `25a7aaa0`; its compiler has no differences from
`main` at `a98fd4f8`. All nine compilations below completed without GPU execution. The shape is M=128, N=1024, K=1024.
The inline seed makes the example repeatable; it does not resolve the separate inline strict-reference report.

From that worktree, enter `nix develop`, then run the following. `EMMY_PYTHON` borrows the working sibling venv;
`PYTHONPATH` selects this checkout's source. In a checkout with its own venv, use `./venv/bin/python` instead.

```sh
set -eu
export PYTHONPATH="$PWD"
EMMY_PYTHON=../main/venv/bin/python
report_dir=$(mktemp -d /tmp/emmy-nvfp4-lut-report.XXXXXX)
export EMMY_TUNE_DB="$report_dir/tune.db" EMMY_GOLDEN_FILE=
PROG='torch.manual_seed(0); nn.Linear(1024,1024,bias=False).half()(torch.randn(128,1024,dtype=torch.float16))'
SCALAR='TILE=,WORK=t256,REDUCE=,STAGE=,RASTER=,PLACE=fuse'
W4A16='TILE=mma_m16n8k16_f16_f32/f2x2/k2,WORK=w1x4,REDUCE=,STAGE=d2/smem-async,RASTER=,PLACE=fuse'
NATIVE='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,WORK=w2x4,REDUCE=,STAGE=d2/smem-async,RASTER=,PLACE=fuse'
dump() {
    local name=$1 scheme=$2 pins=$3 ir
    for ir in loop tile cuda; do
        EMMY_KNOBS="$pins" "$EMMY_PYTHON" -m emmy.emmy compile -c "$PROG" \
            --quantize "$scheme" --target sm_120 --ir "$ir" -o "$report_dir/$name.$ir.txt"
    done
}
dump scalar nvfp4 "$SCALAR"
dump w4a16 nvfp4-w4a16 "$W4A16"
dump native nvfp4 "$NATIVE"
printf '%s\n' "$report_dir"
```

Inspect the actual reads and instruction calls, not just helper definitions or unused parameters:

```sh
rg -n 'f4_pairs\[' "$report_dir/scalar.cuda.txt"
rg -n 'EMMY_F4_LUT|emmy_mma_m16n8k16_f16_f32\(' "$report_dir/w4a16.cuda.txt"
rg -n 'f4_pairs|emmy_mma_m16n8k64_e2m1_f32\(' "$report_dir/native.cuda.txt"
```

## Observed IR and CUDA

The W4A16 Loop IR contains this weight decode inside the contraction. This is an actual excerpt, with surrounding
statements omitted; the scalar W4A4 example has the same lookup for both weights and activations.

```text
in1 = load p_weight_bits[a1, (((a2 / 16) * 8) + ((a2 % 16) / 2))]
i32 v0 = copy(in1)
in2 = load p_weight_f4_pairs[(((int)v0 < 0) ? ((int)v0 + 256) : (int)v0), ((a2 % 16) % 2)]
in3 = load p_weight_scale_bits[a1, (a2 / 16)]
v1 = from_f8e4m3(in3)
v2 = multiply(in0, v1)
f16 v3 = copy(v2)
v4 = multiply(in2, v3)
```

In W4A16 Tile IR, that table load remains in the computed operand's `lift`; the selected contraction is:

```text
=== 0: k_linear_bd09f1 ===
    place  free=(a0, a1)  grid=(a0, a1)
    work   w1x4
    Fold[a2 in 0..1024] contraction   ⟨TILE=mma_m16n8k16_f16_f32/f2x2/k2 STAGE=d2/smem-async⟩
```

The scalar CUDA reads `input_static_fp4_f16_f4_pairs[...]` and `p_weight_f4_pairs[...]` before scaling and multiplying.
The W4A16 CUDA instead emits and reads this smaller table in its fragment loader:

```cuda
__constant__ unsigned short EMMY_F4_LUT_F16[16] = {
    0x0000, 0x3800, 0x3C00, 0x3E00, 0x4000, 0x4200, 0x4400, 0x4600,
    0x8000, 0xB800, 0xBC00, 0xBE00, 0xC000, 0xC200, 0xC400, 0xC600,
};
// Inside emmy_mma_load_b_smem_trans_f4s_f16:
unsigned char byte = g[grp * ldm + (k >> 1)];
__half lo = __ushort_as_half(EMMY_F4_LUT_F16[byte & 0xF]);
__half hi = __ushort_as_half(EMMY_F4_LUT_F16[byte >> 4]);
__half sc = s[grp * sldm + (k >> 4)];
__half2 v = __halves2half2(__hmul(lo, sc), __hmul(hi, sc));
r[i] = *reinterpret_cast<unsigned*>(&v);
```

The kernel calls that loader, then `emmy_mma_m16n8k16_f16_f32`; this is an executed decode path, not merely an unused
prelude. The native W4A4 CUDA calls `emmy_mma_m16n8k64_e2m1_f32` and has no table reads. It still carries unused
`*_f4_pairs` pointer parameters, which must not be counted as observed LUT traffic.

## Why this needs fixing

The dependency is `load packed code -> form lookup address -> load decoded value -> arithmetic`. A direct conversion
can eliminate the second load. A cold lookup also has to bring table data into cache; an eviction by unrelated
traffic can make later lookups refill it. Those fills, refills and dependent accesses are potential performance
penalties even when the table is small. A cache hit is not evidence that the load has zero cost. Conversely, extra
conversion instructions can also cost time: the net effect needs measurement.

Direct decoding also makes the computation clearer to NVCC. A load through a global table pointer supplies a value
whose relationship to the packed bits is generally opaque to the compiler. Explicit bit/arithmetic operations or a
conversion intrinsic expose that relationship and its numeric semantics, which may enable simplification,
constant propagation or instruction selection across the surrounding computation. This is an additional
optimization opportunity, not evidence that NVCC will exploit it or that it will produce a speedup. A statically
initialized constant-memory table exposes more information than an arbitrary pointer, but still expresses an indexed
memory lookup instead of the conversion itself.

The table stores only the fixed E2M1 values, not learned values or calibration results. Block and tensor scales are
separate. The current table footprints, excluding allocation overhead, are:

| Path | Decode table | Payload and memory |
| --- | --- | --- |
| Scalar/general FP16 or BF16 decode | 256 bytes mapped to two 16-bit values each | 1 KiB per global-memory table; the scalar W4A4 repro reads two, 2 KiB total |
| Scalar/general FP32 decode | 256 bytes mapped to two 32-bit values each | 2 KiB per global-memory table |
| Staged FP16/BF16 MMA | 16 nibble codes mapped to 16-bit value encodings | 32 bytes per emitted dtype, in CUDA constant memory |
| Native FP4 MMA | No software decode-table reads | Zero decode-LUT traffic in the verified kernel |

**RTX 5090 desktop is the primary cache example.** NVIDIA specifies 96 MiB L2 and a configurable 128 KiB
L1/shared-memory pool per SM in its
[RTX Blackwell architecture paper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
If the full 2 KiB global-memory working set is resident, its capacity fractions are:

| Cache capacity used as denominator | Fraction occupied by 2 KiB |
| --- | ---: |
| RTX 5090 desktop: 96 MiB L2 shared by the GPU | 0.0020% |
| RTX 5090 desktop: full 128 KiB L1/shared pool of one SM | 1.5625% |
| Illustrative 64 KiB available to L1 after partitioning the pool | 3.125% |
| Secondary example, investigated RTX 5080 Laptop: reported 48 MiB L2 | 0.0041% |

The 128 KiB denominator is the combined pool, not a promise that all of it is usable L1. The laptop capacity comes
from the existing investigation's [device report](../nvfp4-qwen-performance.md#measurement-caveats), not a new
hardware query. Its per-SM pool has the same nominal 128 KiB scale. These are capacity estimates, not measured
occupancy or miss rates. The same table addresses are shared by warps on an SM, not duplicated per warp; different
SMs may each cache them. Distinct table allocations can occupy distinct cache lines even when their contents agree.
Do not apply the global-memory L1 estimate to the 32-byte CUDA constant-memory table: it uses a different access path.

Other supported sub-byte formats demonstrate that a value LUT is not required:

| Format | Executable decode in Emmy | Same value-LUT cost? |
| --- | --- | --- |
| Native MXFP4 expert inputs | Extract sign/magnitude bits and compute E2M1 values arithmetically; apply E8M0 scales | No GPU value LUT, despite sharing E2M1 with NVFP4 |
| AWQ, GPTQ and compressed-tensors INT4 | Shift/mask packed words, subtract zero points, multiply scales | No value LUT |
| EXL3 | Extract trellis windows and compute codebook values with integer/FP arithmetic | No stored value-codebook LUT |

These paths still have weight/scale/metadata traffic. CPU reference decoders may use tables; they do not establish
GPU cache overhead. No data-dependent value table is needed merely because a representation uses fewer than 8 bits.

## Cause and implementation boundaries

`loader/quant.py::_f4_pair_table` constructs the 256-by-2 table, and the weight and static-activation spellers decode
through gathers from it. `ir/kernel/render.py::_F4_LOADER` independently emits the 16-entry table for FP16 and BF16
fragments. Both must be addressed; deleting only the global-memory table leaves the staged MMA lookup intact.

`ir/schedule/packing.py` currently recognizes the pair-table gather as part of the packed operand's algebra. Replacing
that gather must preserve recognition of the packed bytes and block scales, otherwise a purported fix can lose the
native FP4 instruction or expand the weights into a dense buffer. Use canonical dtypes and generic algebra at the
existing boundaries; do not add a checkpoint-specific op or a new fusion restriction.

## Required IR deliverables

The following are **composed expected forms, not current compiler output**. Names and printing may differ. They
specify observable properties of the fix, not a requirement for a particular conversion implementation.

### Loop IR: direct numeric decoding inside the operand

For the W4A16 reproducer, the relevant part of the K-loop should have this meaning and retain the existing scales:

```text
for a0 in 0..128
    for a1 in 0..1024
        for a2 in 0..1024
            bits = load p_weight_bits[a1, (a2 / 2)]
            i32 shift = multiply(modulo(a2, 2), 4)
            i32 code = bitwise_and(right_shift(copy(bits), shift), 15)
            f32 value32 = from_f4e2m1(code)
            f16 value = copy(value32)
            sbits = load p_weight_scale_bits[a1, (a2 / 16)]
            tensor_scale = load p_weight_scale_2[0, 0]
            f16 scale = copy(multiply(tensor_scale, from_f8e4m3(sbits)))
            f16 weight = multiply(value, scale)
            x = load input[a0, a2]
            f32 product = multiply(x, weight)
            acc0 <- add(acc0, product)
        linear[a0, a1] = acc0
```

Nested expressions abbreviate SSA temporaries. `from_f4e2m1` expresses the existing dtype conversion semantics;
equivalent generic bit/arithmetic expressions are acceptable. There must be no `*_f4_pairs` buffer or table load in
the executable decode. Apply the same property to both operands of scalar W4A4. Keep the existing rounding of the
fused scale and decoded product; exact E2M1 conversion is not permission to reassociate scaling.

### Tile IR: preserve the computed operand and tensor-core choices

For W4A16, retain the packed stage and contraction; replace only the lookup-dependent decode in the operand:

```text
place  free=(a0, a1)  grid=(a0, a1)
work   w1x4
Fold[a2 in 0..1024] contraction   ⟨TILE=mma_m16n8k16_f16_f32/f2x2/k2 STAGE=d2/smem-async⟩
├─ operand[x]: load input[a0, a2]   ‹materialized›
├─ operand[weight]: Fold  free   ‹computed›
│  ├─ operand[bits]: load p_weight_bits[a1, (a2 / 2)]   ‹materialized›
│  ├─ operand[sbits]: load p_weight_scale_bits[a1, (a2 / 16)]   ‹materialized›
│  └─ lift: λ(bits, sbits, a2) -> (weight)
│       i32 code = bitwise_and(right_shift(copy(bits), multiply(modulo(a2, 2), 4)), 15)
│       f16 value = copy(from_f4e2m1(code))
│       tensor_scale = load p_weight_scale_2[0, 0]
│       f16 scale = copy(multiply(tensor_scale, from_f8e4m3(sbits)))
│       weight = multiply(value, scale)
├─ init: (0)
├─ lift: λ(a2, x, weight) -> (product)
│    f32 product = multiply(x, weight)
└─ combine: λ(acc0, acc0__o) -> (acc0)
     acc0 = add(acc0, acc0__o)
outputs
└─ linear[a0, a1] = acc0
```

The native W4A4 reproducer must still select this contraction over packed codes and block scales, with no decode
table dependency in its operands:

```text
work   w2x4
Fold[a2 in 0..1024] contraction   ⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 STAGE=d2/smem-async⟩
```

### CUDA: conversion without a table, followed by the same MMA

An exact table-free FP16 conversion could have this shape; this is illustrative code, not an implemented fix:

```cuda
static __device__ __forceinline__ __half decode_e2m1(unsigned code) {
    unsigned mag = code & 7u;
    unsigned bits = mag < 2u ? mag * 0x3800u
                            : (((mag >> 1) + 14u) << 10) | ((mag & 1u) << 9);
    return __ushort_as_half((unsigned short)(((code & 8u) << 12) | bits));
}
// In the staged FP16 fragment loader:
unsigned char byte = g[grp * ldm + (k >> 1)];
__half lo = decode_e2m1(byte & 0xFu);
__half hi = decode_e2m1(byte >> 4);
__half sc = s[grp * sldm + (k >> 4)];
__half2 v = __halves2half2(__hmul(lo, sc), __hmul(hi, sc));
r[i] = *reinterpret_cast<unsigned*>(&v);
```

CUDA's `__nv_cvt_fp4_to_halfraw` / `__nv_cvt_fp4x2_to_halfraw2` are another implementation option where the target
and toolkit support them; verify the generated instructions rather than assuming a particular expansion. See
[NVIDIA's FP4 conversion API](https://docs.nvidia.com/cuda/archive/13.0.3/cuda-math-api/cuda_math_api/group__CUDA__MATH__FP4__MISC.html).
BF16 fragments need the corresponding exact conversion. Scalar CUDA should use direct conversion on the selected
nibble with no value-table pointer or read. CPU-only reference tables may remain as independent correctness oracles.

The W4A16 kernel must still call `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32`; the native W4A4 kernel must still
call `mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X.f32.e2m1.e2m1.f32.ue4m3`.
Remove `EMMY_F4_LUT_*` declarations and accesses, and dead `*_f4_pairs` kernel arguments and runtime bindings.

## Acceptance evidence

- Deliver before/after Loop, Tile and CUDA excerpts for all three reproducers, plus the BF16 staged variant. Preserve
  packed byte storage, selected MMA instructions and existing staging choices; no dense decoded weight workspace
  may be introduced to remove the LUT.
- Exhaustively compare all 16 codes and all 256 packed bytes against the existing reference for FP16, BF16 and FP32,
  including negative zero and both nibble positions. Check representative block/tensor scales at rounding boundaries
  and end-to-end products under each path's existing accuracy contract. The native FP4 path keeps its current
  tolerance; this report does not redefine its scaling semantics.
- Inspect optimized PTX or SASS for scalar and staged MMA examples: there must be no value-table loads hidden behind
  helpers or intrinsics. Bind-time inspection must find no device decode-table allocations required by these kernels.
- Measure before/after on RTX 5090 first, with identical inputs, pins, compiler flags and correctness checks. Include
  repeated reuse and competing-memory-traffic cases to distinguish resident-table behavior from cold/refill costs.
  Report timings and cache evidence, including neutral or negative results. Laptop measurements are secondary.
  Use the existing run/golden benchmark mechanisms and the declared quantized reference; the separate inline strict
  issue must not be mistaken for a decode failure. There is no justified minimum speedup threshold today.

This report requests removal of unnecessary decode LUTs only. No implementation change or measured performance
improvement is claimed by these reproductions.
