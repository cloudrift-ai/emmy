
#include <cuda_fp16.h>
#include <math.h>
extern "C" __global__ void native_embed(const long long* prompt, const long long* length,
    const long long* position, const long long* next, const half* weight, float* hidden) {
    int d = blockIdx.x * blockDim.x + threadIdx.x;
    long long token = *position < *length ? prompt[*position] : *next;
    if (d < HIDDEN) hidden[d] = __half2float(weight[token * HIDDEN + d]);
}
extern "C" __global__ void native_rope_cache(const half* q, const half* k, const half* v,
    const float* cosine, const float* sine, const long long* position,
    half* rotated_q, half* cache_k, half* cache_v) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int d = i % HEAD_DIM;
    int paired = d < HEAD_DIM / 2 ? i + HEAD_DIM / 2 : i - HEAD_DIM / 2;
    float c = cosine[*position * HEAD_DIM + d], s = sine[*position * HEAD_DIM + d];
    if (i < HEADS * HEAD_DIM) {
        half r = d < HEAD_DIM / 2 ? __hneg(q[paired]) : q[paired];
        rotated_q[i] = __float2half(__half2float(q[i]) * c + __half2float(r) * s);
    }
    if (i < KV_HEADS * HEAD_DIM) {
        half r = d < HEAD_DIM / 2 ? __hneg(k[paired]) : k[paired];
        long long offset = *position * KV_HEADS * HEAD_DIM + i;
        cache_k[offset] = __float2half(__half2float(k[i]) * c + __half2float(r) * s);
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
        // Keep attention intermediates in FP32; only the output rounds to the activation dtype.
        scores[t] = dot * SCALE;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        float peak = -INFINITY, total = 0.0f;
        for (int t = 0; t < count; ++t) peak = fmaxf(peak, scores[t]);
        for (int t = 0; t < count; ++t) { scores[t] = expf(scores[t] - peak); total += scores[t]; }
        for (int t = 0; t < count; ++t) scores[t] /= total;
    }
    __syncthreads();
    for (int d = threadIdx.x; d < HEAD_DIM; d += blockDim.x) {
        float value = 0.0f;
        for (int t = 0; t < count; ++t) value += scores[t] * __half2float(v[(t * KV_HEADS + kv) * HEAD_DIM + d]);
        output[head * HEAD_DIM + d] = __float2half(value);
    }
}
__device__ void native_greedy(const half* logits, long long* next) {
    {
        float peak = -INFINITY; long long best = 0;
        for (int i = 0; i < VOCAB; ++i) {
            float value = __half2float(logits[i]);
            if (!isfinite(value)) { *next = -1; return; }
            if (value > peak) { peak = value; best = i; }
        }
        *next = best;
    }
}

// Every finite FP16 logit has an exact ordered bin. Signed zeros share a bin.
// Bin zero is reserved for invalid logits; neither infinity nor NaN is sampled.
constexpr int SAMPLING_BINS = 65536;
__device__ unsigned int logit_bin(half value) {
    unsigned short bits = __half_as_ushort(value);
    if ((bits & 0x7fff) == 0) bits = 0;
    return bits & 0x8000 ? (~bits & 0xffff) : (bits ^ 0x8000);
}
__device__ double bin_value(unsigned int bin) {
    return __half2float(__ushort_as_half(bin & 0x8000 ? bin ^ 0x8000 : ~bin));
}
extern "C" __global__ void native_histogram(const half* logits, const double* sampling,
    const long long* position, const long long* length, unsigned int* histogram) {
    if (sampling[0] == 0.0 || *position + 1 < *length) return;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < VOCAB) atomicAdd(histogram + (isfinite(__half2float(logits[i])) ? logit_bin(logits[i]) : 0), 1u);
}
extern "C" __global__ void native_sample(const half* logits, const unsigned int* histogram,
    const double* sampling, const unsigned long long* seed, const long long* position,
    const long long* length, long long* next) {
    if (*position + 1 < *length) return;
    if (sampling[0] == 0.0) { native_greedy(logits, next); return; }
    if (histogram[0]) { *next = -1; return; }
    int peak = SAMPLING_BINS - 1;
    while (peak > 0 && !histogram[peak]) --peak;
    double maximum = bin_value(peak), total = 0.0;
    for (int bin = peak; bin > 0; --bin)
        if (histogram[bin]) total += histogram[bin] * exp((bin_value(bin) - maximum) / sampling[0]);
    // Include the smallest descending prefix reaching top-p. Equal logits are ordered by token ID.
    double mass = 0.0;
    int cutoff = peak;
    unsigned int ties = 0;
    for (; cutoff > 0; --cutoff) {
        if (!histogram[cutoff]) continue;
        double weight = exp((bin_value(cutoff) - maximum) / sampling[0]);
        double group = histogram[cutoff] * weight;
        if (weight > 0.0 && mass + group >= sampling[1] * total) {
            ties = min(histogram[cutoff], (unsigned int)fmax(1.0, ceil((sampling[1] * total - mass) / weight)));
            mass += ties * weight;
            break;
        }
        mass += group;
    }
    // SplitMix64 counter: request seed and generated-token index, independent of prefill and capture.
    unsigned long long random = *seed + 0x9e3779b97f4a7c15ULL * (unsigned long long)(*position - *length + 2);
    random = (random ^ (random >> 30)) * 0xbf58476d1ce4e5b9ULL;
    random = (random ^ (random >> 27)) * 0x94d049bb133111ebULL;
    random ^= random >> 31;
    double target = (random >> 11) * 0x1.0p-53 * mass, cumulative = 0.0;
    long long last = -1;
    for (int i = 0; i < VOCAB; ++i) {
        unsigned int bin = logit_bin(logits[i]);
        if (bin < (unsigned int)cutoff) continue;
        if (bin == (unsigned int)cutoff) { if (!ties) continue; --ties; }
        double weight = exp((__half2float(logits[i]) - maximum) / sampling[0]);
        if (weight == 0.0) continue;
        last = i;
        cumulative += weight;
        if (target < cumulative) { *next = i; return; }
    }
    *next = last; // Rounding at the final cumulative boundary.
}
