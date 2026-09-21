// Standalone study of the Qwen3-0.6B layer-0 fused attention target:
//   q/k RMSNorm -> RoPE -> Q@K^T scores -> softmax row statistics
//
// Shapes are the real ones recovered from the committed golden:
//   Q [HQ=16, S=512, D=128] f16, K [HKV=8, S=512, D=128] f16, GQA 2:1, RoPE over the 64-wide halves.
//
// The question this answers: Emmy deploys this target at 19,609 us against eager's 745 us. Its
// generated code recomputes each RMSNorm inside the score loops. Is that recompute the whole gap,
// or is the shape itself bad? Three variants isolate it.
//
//   A  recompute  - normalize + rotate both operands inside the (i,j) score loop. Emmy's structure.
//   B  hoisted    - normalize + rotate once into scratch, then a plain score pass. Same math.
//   C  tiled      - B, but blocked through shared memory with online row statistics, so the score
//                   matrix is never held. What a FlashAttention-shaped schedule buys on top of B.
//
// Build: nvcc -O3 -arch=sm_89 -o qk_scores qk_scores.cu

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <algorithm>
#include <cuda_fp16.h>
#include <mma.h>
using namespace nvcuda;

#define CHECK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
  printf("CUDA error %s at line %d\n", cudaGetErrorString(e), __LINE__); exit(1); } } while (0)

static const int HQ = 16, HKV = 8, S = 512, D = 128, DH = D / 2;

__device__ __forceinline__ float rope_pair(float lo, float hi, float cosv, float sinv, bool second) {
  return second ? (hi * cosv + lo * sinv) : (lo * cosv - hi * sinv);
}

// ---------------------------------------------------------------- A: recompute inside the loop
// One thread per (head, i, j) score. Each thread re-derives the RMSNorm scale of its query row and
// its key row, and re-applies RoPE to both, before the dot product. That is 2*D extra work per
// score, i.e. the norm of every row is recomputed S times.
__global__ void scores_recompute(const __half* __restrict__ Q, const __half* __restrict__ K,
                                 const float* __restrict__ gq, const float* __restrict__ gk,
                                 const float* __restrict__ cosv, const float* __restrict__ sinv,
                                 float* __restrict__ rowsq, float* __restrict__ rowsum, float eps) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y;
  int h = blockIdx.z;
  if (j >= S) return;
  const __half* qrow = Q + ((size_t)h * S + i) * D;
  const __half* krow = K + ((size_t)(h / 2) * S + j) * D;

  float qss = 0.f, kss = 0.f;
  for (int d = 0; d < D; ++d) { float a = __half2float(qrow[d]); qss += a * a; }
  for (int d = 0; d < D; ++d) { float a = __half2float(krow[d]); kss += a * a; }
  float qinv = rsqrtf(qss / D + eps), kinv = rsqrtf(kss / D + eps);

  float acc = 0.f;
  for (int d = 0; d < DH; ++d) {
    float q_lo = __half2float(qrow[d])      * qinv * gq[d];
    float q_hi = __half2float(qrow[d + DH]) * qinv * gq[d + DH];
    float k_lo = __half2float(krow[d])      * kinv * gk[d];
    float k_hi = __half2float(krow[d + DH]) * kinv * gk[d + DH];
    float ci = cosv[i * DH + d], si = sinv[i * DH + d];
    float cj = cosv[j * DH + d], sj = sinv[j * DH + d];
    acc += rope_pair(q_lo, q_hi, ci, si, false) * rope_pair(k_lo, k_hi, cj, sj, false);
    acc += rope_pair(q_lo, q_hi, ci, si, true)  * rope_pair(k_lo, k_hi, cj, sj, true);
  }
  // Checksums, not softmax stats: |acc| and acc^2 are positive, so summing them cannot cancel to
  // near zero the way signed scores do, and a wrong kernel cannot hide behind a tiny denominator.
  atomicAdd(&rowsq[(size_t)h * S + i], acc * acc);
  atomicAdd(&rowsum[(size_t)h * S + i], fabsf(acc));
}

// ---------------------------------------------------------------- B: hoist, then score
// Pass 1 normalizes and rotates each row exactly once into f16 scratch. Pass 2 is a plain dot.
__global__ void normalize_rope(const __half* __restrict__ X, const float* __restrict__ g,
                               const float* __restrict__ cosv, const float* __restrict__ sinv,
                               __half* __restrict__ out, int heads, float eps) {
  int row = blockIdx.x;                 // head * S + pos
  int pos = row % S;
  const __half* x = X + (size_t)row * D;
  __half* o = out + (size_t)row * D;

  __shared__ float red[128];
  float part = 0.f;
  for (int d = threadIdx.x; d < D; d += blockDim.x) { float a = __half2float(x[d]); part += a * a; }
  red[threadIdx.x] = part;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] += red[threadIdx.x + s];
    __syncthreads();
  }
  float inv = rsqrtf(red[0] / D + eps);

  for (int d = threadIdx.x; d < DH; d += blockDim.x) {
    float lo = __half2float(x[d])      * inv * g[d];
    float hi = __half2float(x[d + DH]) * inv * g[d + DH];
    float c = cosv[pos * DH + d], s2 = sinv[pos * DH + d];
    o[d]      = __float2half(rope_pair(lo, hi, c, s2, false));
    o[d + DH] = __float2half(rope_pair(lo, hi, c, s2, true));
  }
}

__global__ void scores_hoisted(const __half* __restrict__ Qn, const __half* __restrict__ Kn,
                               float* __restrict__ rowsq, float* __restrict__ rowsum) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  int i = blockIdx.y, h = blockIdx.z;
  if (j >= S) return;
  const __half* q = Qn + ((size_t)h * S + i) * D;
  const __half* k = Kn + ((size_t)(h / 2) * S + j) * D;
  float acc = 0.f;
  for (int d = 0; d < D; ++d) acc += __half2float(q[d]) * __half2float(k[d]);
  atomicAdd(&rowsq[(size_t)h * S + i], acc * acc);
  atomicAdd(&rowsum[(size_t)h * S + i], fabsf(acc));
}

// ---------------------------------------------------------------- C: tiled, online statistics
// One block owns BM query rows of one head, walks K in BN-wide tiles through shared memory, and
// keeps a running max and sum. The score matrix is never written to memory.
template <int BM, int BN>
__global__ void scores_tiled(const __half* __restrict__ Qn, const __half* __restrict__ Kn,
                             float* __restrict__ rowsq, float* __restrict__ rowsum) {
  int h = blockIdx.y;
  int i0 = blockIdx.x * BM;
  __shared__ __half qs[BM][D];
  __shared__ __half ks[BN][D];

  for (int t = threadIdx.x; t < BM * D; t += blockDim.x) {
    int r = t / D, d = t % D;
    qs[r][d] = Qn[((size_t)h * S + i0 + r) * D + d];
  }
  __syncthreads();

  float mx[BM], sm[BM];
  for (int r = 0; r < BM; ++r) { mx[r] = 0.f; sm[r] = 0.f; }

  for (int j0 = 0; j0 < S; j0 += BN) {
    for (int t = threadIdx.x; t < BN * D; t += blockDim.x) {
      int r = t / D, d = t % D;
      ks[r][d] = Kn[((size_t)(h / 2) * S + j0 + r) * D + d];
    }
    __syncthreads();
    for (int r = 0; r < BM; ++r) {
      for (int c = threadIdx.x; c < BN; c += blockDim.x) {
        float acc = 0.f;
        for (int d = 0; d < D; ++d) acc += __half2float(qs[r][d]) * __half2float(ks[c][d]);
        mx[r] += acc * acc;
        sm[r] += fabsf(acc);
      }
    }
    __syncthreads();
  }
  // One lane folds each row's partials.
  __shared__ float pm[BM][32], ps[BM][32];
  int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
  for (int r = 0; r < BM; ++r) {
    float m = mx[r], s2 = sm[r];
    for (int off = 16; off > 0; off >>= 1) {
      m += __shfl_down_sync(0xffffffff, m, off);
      s2 += __shfl_down_sync(0xffffffff, s2, off);
    }
    if (lane == 0) { pm[r][warp] = m; ps[r][warp] = s2; }
  }
  __syncthreads();
  if (threadIdx.x < BM) {
    int r = threadIdx.x, nw = blockDim.x / 32;
    float m = 0.f, s2 = 0.f;
    for (int w = 0; w < nw; ++w) { m += pm[r][w]; s2 += ps[r][w]; }
    rowsq[(size_t)h * S + i0 + r] = m;
    rowsum[(size_t)h * S + i0 + r] = s2;
  }
}

// ---------------------------------------------------------------- D: tiled on tensor cores
// C's structure, but the score tile is computed by wmma instead of scalar half-to-float dots.
// One block owns 16 query rows of one head; its four warps each take a 16-wide key tile. The
// accumulator lands in shared memory so the row statistics fold from a plain layout rather than
// from the fragment's thread mapping.
__global__ void scores_tc(const __half* __restrict__ Qn, const __half* __restrict__ Kn,
                          float* __restrict__ rowsq, float* __restrict__ rowsum) {
  const int WARPS = 4, TILE = 16, JSTEP = WARPS * TILE;
  int h = blockIdx.y, i0 = blockIdx.x * TILE;
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;

  __shared__ __half qs[TILE][D];
  __shared__ __half ks[JSTEP][D];
  __shared__ float  sc[WARPS][TILE][TILE];
  __shared__ float  acc_sq[TILE], acc_abs[TILE];

  if (threadIdx.x < TILE) { acc_sq[threadIdx.x] = 0.f; acc_abs[threadIdx.x] = 0.f; }
  for (int t = threadIdx.x; t < TILE * D; t += blockDim.x)
    qs[t / D][t % D] = Qn[((size_t)h * S + i0 + t / D) * D + t % D];
  __syncthreads();

  for (int j0 = 0; j0 < S; j0 += JSTEP) {
    for (int t = threadIdx.x; t < JSTEP * D; t += blockDim.x)
      ks[t / D][t % D] = Kn[((size_t)(h / 2) * S + j0 + t / D) * D + t % D];
    __syncthreads();

    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
    wmma::fill_fragment(acc, 0.f);
    for (int k0 = 0; k0 < D; k0 += 16) {
      wmma::fragment<wmma::matrix_a, 16, 16, 16, __half, wmma::row_major> af;
      wmma::fragment<wmma::matrix_b, 16, 16, 16, __half, wmma::col_major> bf;
      wmma::load_matrix_sync(af, &qs[0][k0], D);            // A[m][k] = qs[m][k0+k]
      wmma::load_matrix_sync(bf, &ks[warp * TILE][k0], D);  // B[k][n] = ks[warp*16+n][k0+k] = K^T
      wmma::mma_sync(acc, af, bf, acc);
    }
    wmma::store_matrix_sync(&sc[warp][0][0], acc, TILE, wmma::mem_row_major);
    __syncwarp();

    if (lane < TILE) {                       // one lane per query row of this warp's tile
      float sq = 0.f, ab = 0.f;
      for (int c = 0; c < TILE; ++c) { float v = sc[warp][lane][c]; sq += v * v; ab += fabsf(v); }
      atomicAdd(&acc_sq[lane], sq);
      atomicAdd(&acc_abs[lane], ab);
    }
    __syncthreads();
  }

  if (threadIdx.x < TILE) {
    rowsq[(size_t)h * S + i0 + threadIdx.x]  = acc_sq[threadIdx.x];
    rowsum[(size_t)h * S + i0 + threadIdx.x] = acc_abs[threadIdx.x];
  }
}

// ---------------------------------------------------------------------------- harness
static float time_ms(void (*launch)(void*), void* ctx, int iters) {
  cudaEvent_t a, b; CHECK(cudaEventCreate(&a)); CHECK(cudaEventCreate(&b));
  launch(ctx); CHECK(cudaDeviceSynchronize());          // warm
  std::vector<float> ts;
  for (int it = 0; it < iters; ++it) {
    CHECK(cudaEventRecord(a));
    launch(ctx);
    CHECK(cudaEventRecord(b)); CHECK(cudaEventSynchronize(b));
    float ms; CHECK(cudaEventElapsedTime(&ms, a, b)); ts.push_back(ms);
  }
  std::sort(ts.begin(), ts.end());
  return ts[ts.size() / 2];
}

struct Ctx {
  __half *Q, *K, *Qn, *Kn;
  float *gq, *gk, *cosv, *sinv, *rowsq, *rowsum;
};
static Ctx C;

static void clear_stats() {
  CHECK(cudaMemset(C.rowsq, 0, (size_t)HQ * S * sizeof(float)));
  CHECK(cudaMemset(C.rowsum, 0,    (size_t)HQ * S * sizeof(float)));
}
static void run_A(void*) {
  clear_stats();
  dim3 b(128), g((S + 127) / 128, S, HQ);
  scores_recompute<<<g, b>>>(C.Q, C.K, C.gq, C.gk, C.cosv, C.sinv, C.rowsq, C.rowsum, 1e-6f);
}
static void run_prep(void*) {
  normalize_rope<<<HQ * S, 128>>>(C.Q, C.gq, C.cosv, C.sinv, C.Qn, HQ, 1e-6f);
  normalize_rope<<<HKV * S, 128>>>(C.K, C.gk, C.cosv, C.sinv, C.Kn, HKV, 1e-6f);
}
static void run_B(void*) {
  clear_stats(); run_prep(nullptr);
  dim3 b(128), g((S + 127) / 128, S, HQ);
  scores_hoisted<<<g, b>>>(C.Qn, C.Kn, C.rowsq, C.rowsum);
}
static void run_C(void*) {
  clear_stats(); run_prep(nullptr);
  dim3 b(128), g(S / 16, HQ);
  scores_tiled<16, 32><<<g, b>>>(C.Qn, C.Kn, C.rowsq, C.rowsum);
}
static void run_D(void*) {
  clear_stats(); run_prep(nullptr);
  dim3 b(128), g(S / 16, HQ);
  scores_tc<<<g, b>>>(C.Qn, C.Kn, C.rowsq, C.rowsum);
}

int main() {
  cudaDeviceProp p; CHECK(cudaGetDeviceProperties(&p, 0));
  printf("device: %s  sm_%d%d\n", p.name, p.major, p.minor);
  printf("shape:  Q[%d,%d,%d] K[%d,%d,%d] f16, GQA %d:1, RoPE over %d\n\n", HQ, S, D, HKV, S, D, HQ / HKV, DH);

  size_t nq = (size_t)HQ * S * D, nk = (size_t)HKV * S * D;
  CHECK(cudaMalloc(&C.Q, nq * 2));  CHECK(cudaMalloc(&C.Qn, nq * 2));
  CHECK(cudaMalloc(&C.K, nk * 2));  CHECK(cudaMalloc(&C.Kn, nk * 2));
  CHECK(cudaMalloc(&C.gq, D * 4));  CHECK(cudaMalloc(&C.gk, D * 4));
  CHECK(cudaMalloc(&C.cosv, (size_t)S * DH * 4)); CHECK(cudaMalloc(&C.sinv, (size_t)S * DH * 4));
  CHECK(cudaMalloc(&C.rowsq, (size_t)HQ * S * 4)); CHECK(cudaMalloc(&C.rowsum, (size_t)HQ * S * 4));

  std::vector<__half> hq(nq), hk(nk);
  for (size_t i = 0; i < nq; ++i) hq[i] = __float2half((float)drand48() * 2.f - 1.f);
  for (size_t i = 0; i < nk; ++i) hk[i] = __float2half((float)drand48() * 2.f - 1.f);
  std::vector<float> g(D, 1.0f), cv((size_t)S * DH), sv((size_t)S * DH);
  for (int s = 0; s < S; ++s)
    for (int d = 0; d < DH; ++d) {
      float th = s / powf(10000.f, (2.f * d) / D);
      cv[s * DH + d] = cosf(th); sv[s * DH + d] = sinf(th);
    }
  CHECK(cudaMemcpy(C.Q, hq.data(), nq * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C.K, hk.data(), nk * 2, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C.gq, g.data(), D * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C.gk, g.data(), D * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C.cosv, cv.data(), cv.size() * 4, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(C.sinv, sv.data(), sv.size() * 4, cudaMemcpyHostToDevice));

  // Correctness: B is the reference; A and C must agree with it on the row sums.
  std::vector<float> sB(HQ * S), sX(HQ * S);
  run_B(nullptr); CHECK(cudaDeviceSynchronize());
  CHECK(cudaMemcpy(sB.data(), C.rowsum, sB.size() * 4, cudaMemcpyDeviceToHost));
  auto agree = [&](const char* nm) {
    CHECK(cudaMemcpy(sX.data(), C.rowsum, sX.size() * 4, cudaMemcpyDeviceToHost));
    double worst = 0; for (size_t i = 0; i < sB.size(); ++i) {
      double d = fabs(sX[i] - sB[i]) / (fabs(sB[i]) + 1e-3); worst = std::max(worst, d); }
    printf("  %s vs hoisted: max relative row-sum difference %.3e %s\n", nm, worst,
           worst < 2e-2 ? "(agree)" : "(DISAGREE)");
  };
  run_A(nullptr); CHECK(cudaDeviceSynchronize()); agree("recompute");
  run_C(nullptr); CHECK(cudaDeviceSynchronize()); agree("tiled    ");
  run_D(nullptr); CHECK(cudaDeviceSynchronize()); agree("tensorcore");

  printf("\n%-28s %10s  %s\n", "variant", "median ms", "note");
  float a = time_ms(run_A, nullptr, 11);
  float b = time_ms(run_B, nullptr, 11);
  float c = time_ms(run_C, nullptr, 11);
  float d = time_ms(run_D, nullptr, 11);
  printf("%-28s %10.3f  %s\n", "A recompute in loop",  a, "Emmy's structure");
  printf("%-28s %10.3f  %s\n", "B hoisted, then score", b, "same math, norms once");
  printf("%-28s %10.3f  %s\n", "C tiled, online stats", c, "score matrix never stored");
  printf("%-28s %10.3f  %s\n", "D tiled on tensor cores", d, "C + wmma for the score tile");
  printf("\nA->B hoist %.1fx | B->C no score matrix %.1fx | C->D tensor cores %.1fx | A->D %.1fx\n",
         a / b, b / c, c / d, a / d);
  return 0;
}
