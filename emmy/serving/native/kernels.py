"""CUDA operations surrounding compiler-produced dense Qwen3 pre/post programs."""

SOURCE = r'''
#include <cuda_fp16.h>
#include <math.h>
extern "C" __global__ void native_embed(const long long* prompt, const long long* length,
    const long long* position, const long long* next, const half* weight, half* hidden) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    long long token = *position < *length ? prompt[*position] : *next;
    if (d < HIDDEN) hidden[d] = weight[token * HIDDEN + d];
}
extern "C" __global__ void native_rope_cache(const half* q, const half* k, const half* v,
    const half* cosine, const half* sine, const long long* position,
    half* rotated_q, half* cache_k, half* cache_v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = i % HEAD_DIM;
    int paired = d < HEAD_DIM / 2 ? i + HEAD_DIM / 2 : i - HEAD_DIM / 2;
    half c = cosine[*position * HEAD_DIM + d], s = sine[*position * HEAD_DIM + d];
    if (i < HEADS * HEAD_DIM) {
        half r = d < HEAD_DIM / 2 ? __hneg(q[paired]) : q[paired];
        rotated_q[i] = __hadd_rn(__hmul_rn(q[i], c), __hmul_rn(r, s));
    }
    if (i < KV_HEADS * HEAD_DIM) {
        half r = d < HEAD_DIM / 2 ? __hneg(k[paired]) : k[paired];
        long long offset = *position * KV_HEADS * HEAD_DIM + i;
        cache_k[offset] = __hadd_rn(__hmul_rn(k[i], c), __hmul_rn(r, s));
        cache_v[offset] = v[i];
    }
}
extern "C" __global__ void native_attention(const half* q, const half* k, const half* v,
    const long long* position, half* output) {
    extern __shared__ float scores[];
    int head = blockIdx.x, kv = head / (HEADS / KV_HEADS), count = int(*position) + 1;
    for (int t = threadIdx.x; t < count; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d)
            dot += __half2float(q[head * HEAD_DIM + d]) * __half2float(k[(t * KV_HEADS + kv) * HEAD_DIM + d]);
        // HF eager stores QK and the scaled scores in the activation dtype before float softmax.
        scores[t] = __half2float(__float2half(__half2float(__float2half(dot)) * SCALE));
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float peak = -INFINITY, total = 0.0f;
        for (int t = 0; t < count; ++t) peak = fmaxf(peak, scores[t]);
        for (int t = 0; t < count; ++t) { scores[t] = expf(scores[t] - peak); total += scores[t]; }
        for (int t = 0; t < count; ++t) scores[t] = __half2float(__float2half(scores[t] / total));
    }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float value = 0.0f;
        for (int t = 0; t < count; ++t) value += scores[t] * __half2float(v[(t * KV_HEADS + kv) * HEAD_DIM + d]);
        output[head * HEAD_DIM + d] = __float2half(value);
    }
}
extern "C" __global__ void native_greedy(const half* logits, long long* next) {
    if (threadIdx.x == 0) {
        float peak = -INFINITY; long long best = 0;
        for (int i = 0; i < VOCAB; ++i) {
            float value = __half2float(logits[i]);
            if (!isfinite(value)) { *next = -1; return; }
            if (value > peak) { peak = value; best = i; }
        }
        *next = best;
    }
}
'''
