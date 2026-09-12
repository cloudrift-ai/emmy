#include <cuda_fp8.h>
#include <cuda_bf16.h>
static __device__ __forceinline__ void emmy_ldmatrix_x4(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x2_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// x4.trans: two col-adjacent canonical-B fragments in one ldmatrix — lanes 0-15
// address the 16 K rows at the first fragment's column, lanes 16-31 at col+8
// (the 096_pair_ldmatrix_loads fusion; r[0..1] / r[2..3] are the two fragments).
static __device__ __forceinline__ void emmy_ldmatrix_x4_trans(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// Paired x4 → two B fragments in one ldmatrix (096_pair_ldmatrix_loads): the four
// loaded registers land DIRECTLY as b0[0..1] (lanes 0-15) and b1[0..1] (lanes 16-31)
// — no staging temp / register shuffle. ``_pair`` is the plain x4 (transposed-B slab,
// N-adjacent pair); ``_trans_pair`` is x4.trans (canonical-B slab, col-adjacent pair).
static __device__ __forceinline__ void emmy_ldmatrix_x4_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

static __device__ __forceinline__ void emmy_ldmatrix_x4_trans_pair(unsigned* b0, unsigned* b1, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(b0[0]), "=r"(b0[1]), "=r"(b1[0]), "=r"(b1[1]) : "r"(addr));
}

// Plain (no .trans) x2: a transposed-B operand staged as its native N-major
// slab (Q@K^T's K rows) — each 8x8 matrix's rows ARE the mma B fragment's
// col-major columns, so no transpose is needed (cf. emmy_mma_load_b_gmem_trans).
static __device__ __forceinline__ void emmy_ldmatrix_x2(unsigned* r, const void* smem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(addr));
}

// C->A register repack: the m16n8k16 f32 C fragment is lane-map ALIGNED with the
// 16-bit A fragment's k-halves, so two k-adjacent C fragments (c0 low, c1 high)
// convert into one A operand fragment per lane — no shuffle, no smem round-trip.
// cvt.rn packs (hi, lo) with round-to-nearest-even — the same rounding the
// retired smem handoff's RegStore vec2 pack applied, so the repack is bit-identical.
static __device__ __forceinline__ void emmy_c_to_a_f16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.f16x2.f32 %0, %1, %2;\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

static __device__ __forceinline__ void emmy_c_to_a_bf16(unsigned* a, const float* c0, const float* c1) {
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[0]) : "f"(c0[1]), "f"(c0[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[1]) : "f"(c0[3]), "f"(c0[2]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[2]) : "f"(c1[1]), "f"(c1[0]));
    asm("cvt.rn.bf16x2.f32 %0, %1, %2;\n" : "=r"(a[3]) : "f"(c1[3]), "f"(c1[2]));
}

// gmem-direct fragment loads — the fallback when an mma.sync operand was NOT
// staged into shared memory (ldmatrix is smem-only, so we read the fragment
// straight from gmem instead, replicating the PTX m16n8k16 lane→element map).
// Slower than ldmatrix (no smem reuse) but correct; ``005_lower_atom_tile``
// emits these for an unstaged operand. ``ldm`` is the operand's gmem row stride
// (K for the row-major A[M,K]; N for the row-major B[K,N]); ``g`` points at the
// atom cell's base element, each lane adds its own (row,col) within the tile.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 1) + ((i & 2) ? 8 : 0);   // K: 2*threadID_in_group, +8 for the k16 half
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);                     // .f16x2: low half = col, high half = col+1
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);           // .f16x2: low half = k, high half = k+1
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Masked-tile (M9) variants of the gmem-direct fragment loads: a tile straddling a masked axis's
// bound would read rows / cols past the runtime-sized buffer, so the lane coordinate on the gated
// axis clamps INTO the live range — ``left - 1`` (``left`` = in-range elements from the tile base)
// where the fragment straddles the bound, the tile base where it OVERHANGS it entirely. The latter
// is a tile wider than its axis (M=1 under a 128-row tile): ``left`` goes <= 0, and a bare
// ``left - 1`` addresses tens of KB BELOW the buffer — an out-of-bounds read that faults the context
// wherever that memory is unmapped. Clamped lanes read a duplicate in-bounds value — harmless,
// their stores are masked by the RegStore guard (the tile path's ``clamp_last``, same contract).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp(unsigned* r, const T* g, int ldm, int rows_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = max(rows_left - 1, 0);   // M: clamp to the runtime extent
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed;
        ((F*)&packed)[0] = F(p[0]);
        ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[k * ldm + n]);
        ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q @ K^T): B stored N×K (``g[n][k]``, K contiguous) — the native
// ``mma.row.col`` col-major B, so no ldmatrix ``.trans`` is needed. The fragment
// lane→element map is the same (n = groupID, k = 2·threadID_in_group + k16 half),
// but each lane now reads a contiguous (k, k+1) pair from row ``n`` of B. ``ldm``
// is B's gmem row stride (the K extent). Mirrors ``emmy_mma_load_b_gmem`` with the
// (k, n) index roles swapped to (n, k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);           // .f16x2: contiguous (k, k+1) in row n
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-tile variant of the transposed-B gmem load: the gated N axis is the
// row (``n``) here, so clamp ``n`` to the runtime extent (cf. _b_gmem_nclamp,
// which clamps N as the column). Clamped lanes read a duplicate in-bounds row;
// their stores are masked by the RegStore guard.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp(unsigned* r, const T* g, int ldm, int cols_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed;
        ((F*)&packed)[0] = F(g[n * ldm + k]);
        ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Masked-K (symbolic reduce) variants of the gmem-direct fragment loads: when the
// operand's REDUCE (K) axis is mask-padded (SDPA P@V over ``seq_len``), the final
// K tile straddles the runtime extent. Unlike the M/N clamp above, a K element
// past the extent must be ZERO-FILLED, not clamped to a duplicate — K is summed
// by the mma, so a duplicate corrupts the reduction. ``k_left`` = in-range K
// elements from the tile base; a half past it reads as +0.0 and is never
// dereferenced. Mirrors the staged path's slab zero-fill (``_stage_expand``).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// A: masked-M (clamp rows) AND masked-K (zero-fill cols).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp_kzero(unsigned* r, const T* g, int ldm, int rows_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = max(rows_left - 1, 0);
        int col = (tig << 1) + ((i & 2) ? 8 : 0);
        const T* p = g + row * ldm + col;
        unsigned packed = 0;
        if (col < k_left) ((F*)&packed)[0] = F(p[0]);
        if (col + 1 < k_left) ((F*)&packed)[1] = F(p[1]);
        r[i] = packed;
    }
}

// B (row-major K×N, NOT transposed): K is the row, zero-fill past the extent.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// B: masked-N (clamp col) AND masked-K (zero-fill row).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp_kzero(unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[k * ldm + n]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[(k + 1) * ldm + n]);
        r[i] = packed;
    }
}

// Transposed-B (Q@K^T, ``g[n][k]`` with K contiguous) masked-K: zero-fill the K
// halves past the runtime extent. K is summed by the mma, so a past-extent element
// must read as +0.0 (never a duplicate). Mirrors ``emmy_mma_load_b_gmem_kzero`` with
// the (k, n) index roles swapped to (n, k) — cf. ``emmy_mma_load_b_gmem_trans``.
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_kzero(unsigned* r, const T* g, int ldm, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);       // .f16x2: contiguous (k, k+1) in row n
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

// Transposed-B: masked-N (clamp the ``n`` row) AND masked-K (zero-fill the contiguous k).
template <typename T, typename F = T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp_kzero(
    unsigned* r, const T* g, int ldm, int cols_left, int k_left) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 1) + (i ? 8 : 0);
        unsigned packed = 0;
        if (k < k_left) ((F*)&packed)[0] = F(g[n * ldm + k]);
        if (k + 1 < k_left) ((F*)&packed)[1] = F(g[n * ldm + k + 1]);
        r[i] = packed;
    }
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void emmy_mma_m16n8k16_bf16_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

// f16-accumulate HMMA (the ``mma_m16n8k16_f16_f16`` atom): on the consumer GeForce dies the .f32-accumulate
// form above runs at HALF the tensor-core rate, so this variant keeps the whole mma chain on the
// full-rate .f16 accumulator — ``c``/``d`` are 2 packed b32 regs (two halfs each, the same
// element map as the four f32 regs, pair-packed). Paired with emmy_mma_promote_f16acc below,
// which periodically folds the f16 partials into the f32 shadow accumulator.
static __device__ __forceinline__ void emmy_mma_m16n8k16_f16_f16(unsigned* d, const unsigned* a, const unsigned* b, const unsigned* c) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 "
                 "{%0, %1}, {%2, %3, %4, %5}, {%6, %7}, {%8, %9};\n"
                 : "=r"(d[0]), "=r"(d[1])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "r"(c[0]), "r"(c[1]));
}

// The chunk promote of the f16-accumulate scheme: fold the packed f16 C fragment into the f32
// shadow accumulator and rezero it, bounding the f16 accumulation error to one K chunk. Inline
// PTX cvt (not __half2 intrinsics) keeps this prelude header-free, like the wrappers above.
static __device__ __forceinline__ void emmy_mma_promote_f16acc(float* c, unsigned* h) {
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        float lo, hi;
        asm("{.reg .b16 lo, hi;\n\t"
            "mov.b32 {lo, hi}, %2;\n\t"
            "cvt.f32.f16 %0, lo;\n\t"
            "cvt.f32.f16 %1, hi;}\n"
            : "=f"(lo), "=f"(hi) : "r"(h[i]));
        c[2 * i] += lo;
        c[2 * i + 1] += hi;
        h[i] = 0u;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 2) + ((i & 2) ? 16 : 0);  // K: 4*threadID_in_group, +16 for the k32 half
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[row * ldm + col + j];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;                                 // N: groupID (0..7)
        int k = (tig << 2) + (i ? 16 : 0);           // K: 4*threadID_in_group, +16 for the k32 half
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[(k + j) * ldm + n];
        r[i] = packed;
    }
}

// Transposed-B (``g[n][k]``, K contiguous): same fragment map with the (k, n) index roles swapped —
// each lane reads a contiguous 4-byte K run from row ``n``. Cf. emmy_mma_load_b_gmem_trans.
template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_b8(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[n * ldm + k + j];
        r[i] = packed;
    }
}

// Masked-tile variants: clamp the lane coordinate on the gated output axis to the runtime extent
// (duplicate reads; the RegStore guard masks their stores) — the b8 mirror of the 16-bit
// ``_mclamp`` / ``_nclamp`` family above.
template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_gmem_mclamp_b8(unsigned* r, const T* g, int ldm, int rows_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);
        if (row >= rows_left) row = max(rows_left - 1, 0);   // M: clamp to the runtime extent
        int col = (tig << 2) + ((i & 2) ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[row * ldm + col + j];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_nclamp_b8(unsigned* r, const T* g, int ldm, int cols_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[(k + j) * ldm + n];
        r[i] = packed;
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_gmem_trans_nclamp_b8(unsigned* r, const T* g, int ldm, int cols_left) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int n = grp;
        if (n >= cols_left) n = max(cols_left - 1, 0);       // N: clamp to the runtime extent
        int k = (tig << 2) + (i ? 16 : 0);
        unsigned packed;
        #pragma unroll
        for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = p[n * ldm + k + j];
        r[i] = packed;
    }
}

static __device__ __forceinline__ void emmy_mma_m16n8k32_e4m3_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

static __device__ __forceinline__ void emmy_mma_m16n8k32_e5m2_f32(float* d, const unsigned* a, const unsigned* b, const float* c) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e5m2.e5m2.f32 "
                 "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%10, %11, %12, %13};\n"
                 : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
                   "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}

#define EMMY_F4_SF_TID_A 0
#define EMMY_F4_SF_TID_B 1

static __device__ __forceinline__ unsigned emmy_mma_load_sfa_f4(const unsigned char* s, int ldm) {
    int lane = threadIdx.x & 31, q = lane >> 2, r = (lane & 3) - 2 * EMMY_F4_SF_TID_A;
    if (r < 0 || r > 1) return 0u;              // this lane supplies no row; the mma ignores it
    int m = q + 8 * r;
    unsigned packed;
    #pragma unroll
    for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = s[m * ldm + j];
    return packed;
}

static __device__ __forceinline__ unsigned emmy_mma_load_sfb_f4(const unsigned char* s, int ldm) {
    int lane = threadIdx.x & 31, q = lane >> 2;
    if ((lane & 3) != EMMY_F4_SF_TID_B) return 0u;
    unsigned packed;
    #pragma unroll
    for (int j = 0; j < 4; ++j) ((unsigned char*)&packed)[j] = s[q * ldm + j];
    return packed;
}

static __device__ __forceinline__ void emmy_mma_m16n8k64_e2m1_f32(
        float* d, const unsigned* a, const unsigned* b, unsigned sfa, unsigned sfb) {
    asm volatile(
        "mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::4X"
        ".f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, "
        "{%10}, {%12, %13}, {%11}, {%12, %14};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]),
          "r"(sfa), "r"(sfb), "n"(0), "n"(EMMY_F4_SF_TID_A), "n"(EMMY_F4_SF_TID_B));
}

template <typename T, typename T2>
static __device__ __forceinline__ void emmy_mma_load_b_smem_trans_f8_f16(unsigned* r, const T* g, int ldm) {
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int k = (tig << 1) + (i ? 8 : 0);            // K: 2*threadID_in_group, +8 for the k16 half
        __half2 h2 = __half2(*reinterpret_cast<const T2*>(g + grp * ldm + k));
        r[i] = *reinterpret_cast<unsigned*>(&h2);
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_a_smem_b8v(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        int row = grp + ((i & 1) ? 8 : 0);          // M: groupID, +8 for the second row block
        int col = (tig << 2) + ((i & 2) ? 16 : 0);  // K: 4*threadID_in_group, +16 for the k32 half
        r[i] = *reinterpret_cast<const unsigned*>(p + row * ldm + col);
    }
}

template <typename T>
static __device__ __forceinline__ void emmy_mma_load_b_smem_trans_b8v(unsigned* r, const T* g, int ldm) {
    const unsigned char* p = reinterpret_cast<const unsigned char*>(g);
    int lane = threadIdx.x & 31, grp = lane >> 2, tig = lane & 3;
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        int k = (tig << 2) + (i ? 16 : 0);           // K: 4*threadID_in_group, +16 for the k32 half
        r[i] = *reinterpret_cast<const unsigned*>(p + grp * ldm + k);
    }
}

static __device__ __forceinline__ void emmy_cp_async_cg(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(addr), "l"(gmem) : "memory");
}

template <int Bytes>
static __device__ __forceinline__ void emmy_cp_async_ca(void* smem, const void* gmem) {
    unsigned addr = __cvta_generic_to_shared(smem);
    asm volatile("cp.async.ca.shared.global [%0], [%1], %2;\n" :: "r"(addr), "l"(gmem), "n"(Bytes) : "memory");
}

static __device__ __forceinline__ void emmy_cp_async_commit() {
    asm volatile("cp.async.commit_group;\n" ::: "memory");
}

template <int N>
static __device__ __forceinline__ void emmy_cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N) : "memory");
}

extern "C" __global__
__launch_bounds__(128) void k_linear_334cf2(const float* input_static_fp4_scale_2, const float* p_weight_scale_2, const __nv_fp8_e4m3* p_weight_scale_bits, const __nv_fp8_e4m3* input_static_fp4_scale_bits, const unsigned char* input_static_fp4_bits, const unsigned char* p_weight_bits, const __nv_bfloat16* input_static_fp4_r1_f4_pairs, const __nv_bfloat16* p_weight_f4_pairs, __nv_bfloat16* linear) {
    int _gid = blockIdx.x * blockDim.x + threadIdx.x;
        int a0_b = _gid / 8192;
        int a1_b = (_gid / 128) % 64;
        int a0_u = (_gid / 128) % 1;
        int a1_u = (_gid / 32) % 4;
        int _lane = (_gid) % 32;
        unsigned _a0[4];
        unsigned _b0[2];
        unsigned _b1[2];
        unsigned _sfa0;
        unsigned _sfb0;
        unsigned _sfb1;
        float _c0_0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float _c0_1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        __shared__ __align__(16) unsigned char _a_smem[8704];
        __shared__ __align__(16) unsigned char _b_smem[34816];
        __shared__ __align__(16) __nv_fp8_e4m3 _as_smem[1024];
        __shared__ __align__(16) __nv_fp8_e4m3 _bs_smem[4096];
        for (int _fa = (a0_u * 4 + a1_u) * 32 + _lane; _fa < 256; _fa += 128) {
            emmy_cp_async_cg(&_a_smem[_fa / 16 * 272 + 16 * (_fa % 16)], &input_static_fp4_bits[(a0_b * 16 + _fa / 16) * 2048 + _fa % 16 * 2 * 8]);
        }
        for (int _fb = (a0_u * 4 + a1_u) * 32 + _lane; _fb < 1024; _fb += 128) {
            emmy_cp_async_cg(&_b_smem[_fb / 16 * 272 + 16 * (_fb % 16)], &p_weight_bits[(a1_b * 64 + _fb / 16) * 2048 + _fb % 16 * 2 * 8]);
        }
        for (int _fas = (a0_u * 4 + a1_u) * 32 + _lane; _fas < 32; _fas += 128) {
            emmy_cp_async_cg(&_as_smem[_fas * 16], &input_static_fp4_scale_bits[(a0_b * 16 + _fas / 2) * 256 + _fas % 2 * 16]);
        }
        for (int _fbs = (a0_u * 4 + a1_u) * 32 + _lane; _fbs < 128; _fbs += 128) {
            emmy_cp_async_cg(&_bs_smem[_fbs * 16], &p_weight_scale_bits[(a1_b * 64 + _fbs / 2) * 256 + _fbs % 2 * 16]);
        }
        emmy_cp_async_commit();
        for (int _ks = 0; _ks < 4096; _ks += 512) {
            for (int _fa = (a0_u * 4 + a1_u) * 32 + _lane; _fa < 256; _fa += 128) {
                emmy_cp_async_cg(&_a_smem[((_ks / 512 + 1) % 2 * 16 + _fa / 16) * 272 + 16 * (_fa % 16)], &input_static_fp4_bits[(a0_b * 16 + _fa / 16) * 2048 + ((((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fa % 16) * 2) / 16 * 8 + (((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fa % 16) * 2) % 16 / 2)]);
            }
            for (int _fb = (a0_u * 4 + a1_u) * 32 + _lane; _fb < 1024; _fb += 128) {
                emmy_cp_async_cg(&_b_smem[((_ks / 512 + 1) % 2 * 64 + _fb / 16) * 272 + 16 * (_fb % 16)], &p_weight_bits[(a1_b * 64 + _fb / 16) * 2048 + ((((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fb % 16) * 2) / 16 * 8 + (((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fb % 16) * 2) % 16 / 2)]);
            }
            for (int _fas = (a0_u * 4 + a1_u) * 32 + _lane; _fas < 32; _fas += 128) {
                emmy_cp_async_cg(&_as_smem[((_ks / 512 + 1) % 2 * 16 + _fas / 2) * 32 + 16 * (_fas % 2)], &input_static_fp4_scale_bits[(a0_b * 16 + _fas / 2) * 256 + (((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fas % 2) * 16) / 16]);
            }
            for (int _fbs = (a0_u * 4 + a1_u) * 32 + _lane; _fbs < 128; _fbs += 128) {
                emmy_cp_async_cg(&_bs_smem[((_ks / 512 + 1) % 2 * 64 + _fbs / 2) * 32 + 16 * (_fbs % 2)], &p_weight_scale_bits[(a1_b * 64 + _fbs / 2) * 256 + (((_ks + 512 < 4096) ? (_ks + 512) : (3584)) + 16 * (_fbs % 2) * 16) / 16]);
            }
            emmy_cp_async_commit();
            emmy_cp_async_wait<1>();
            __syncthreads();
            #pragma unroll
            for (int _ki = 0; _ki < 512; _ki += 64) {
                _sfa0 = emmy_mma_load_sfa_f4(reinterpret_cast<const unsigned char*>(&_as_smem[(_ks / 512 % 2 * 16 + a0_u * 16) * 32 + _ki / 16]), 32);
                emmy_mma_load_a_smem_b8v<unsigned char>(_a0, &_a_smem[(_ks / 512 % 2 * 16 + a0_u * 16) * 272 + _ki / 2], 272);
                _sfb0 = emmy_mma_load_sfb_f4(reinterpret_cast<const unsigned char*>(&_bs_smem[(_ks / 512 % 2 * 64 + a1_u * 16) * 32 + _ki / 16]), 32);
                emmy_mma_load_b_smem_trans_b8v<unsigned char>(_b0, &_b_smem[(_ks / 512 % 2 * 64 + a1_u * 16) * 272 + _ki / 2], 272);
                _sfb1 = emmy_mma_load_sfb_f4(reinterpret_cast<const unsigned char*>(&_bs_smem[(_ks / 512 % 2 * 64 + (a1_u * 16 + 8)) * 32 + _ki / 16]), 32);
                emmy_mma_load_b_smem_trans_b8v<unsigned char>(_b1, &_b_smem[(_ks / 512 % 2 * 64 + (a1_u * 16 + 8)) * 272 + _ki / 2], 272);
                emmy_mma_m16n8k64_e2m1_f32(_c0_0, _a0, _b0, _sfa0, _sfb0);
                emmy_mma_m16n8k64_e2m1_f32(_c0_1, _a0, _b1, _sfa0, _sfb1);
            }
            __syncthreads();
        }
        float _bs_alpha_l0 = input_static_fp4_scale_2[0];
        float _bs_alpha0 = _bs_alpha_l0;
        float _bs_alpha_l1 = p_weight_scale_2[0];
        float _bs_alpha1 = _bs_alpha0 * _bs_alpha_l1;
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _c0_0[_e] = _c0_0[_e] * _bs_alpha1;
        }
        #pragma unroll
        for (int _e = 0; _e < 4; _e++) {
            _c0_1[_e] = _c0_1[_e] * _bs_alpha1;
        }
        const int _g = (threadIdx.x & 31) >> 2; const int _t = (threadIdx.x & 31) & 3;
        *reinterpret_cast<__nv_bfloat162*>(&linear[(a0_b * 16 + a0_u * 16) * 4096 + (a1_b * 64 + a1_u * 16) + _g * 4096 + _t * 2]) = __floats2bfloat162_rn(_c0_0[0], _c0_0[1]);
        *reinterpret_cast<__nv_bfloat162*>(&linear[(a0_b * 16 + a0_u * 16) * 4096 + (a1_b * 64 + a1_u * 16) + (_g + 8) * 4096 + _t * 2]) = __floats2bfloat162_rn(_c0_0[2], _c0_0[3]);
        *reinterpret_cast<__nv_bfloat162*>(&linear[(a0_b * 16 + a0_u * 16) * 4096 + (a1_b * 64 + a1_u * 16 + 8) + _g * 4096 + _t * 2]) = __floats2bfloat162_rn(_c0_1[0], _c0_1[1]);
        *reinterpret_cast<__nv_bfloat162*>(&linear[(a0_b * 16 + a0_u * 16) * 4096 + (a1_b * 64 + a1_u * 16 + 8) + (_g + 8) * 4096 + _t * 2]) = __floats2bfloat162_rn(_c0_1[2], _c0_1[3]);
}
