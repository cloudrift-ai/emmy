# Ornith-1.5-35B-A3B-FP8 on one H100 80GB

Qualified 2026-09-29 on a GCP a3-highgpu-1g (1x NVIDIA H100 80GB HBM3, driver 580.173.02, host CUDA 12.9.41),
repository revision `a98fd4f851be305a225ffce7bac2ad61b2892c8a`, model revision
`fab11c26e2325a42f4b32da0249c819a0bade1b1`, engine image `vllm/vllm-openai:v0.30.0`
(digest `sha256:8a69ffad015f138d7170c4ddc429e230a3bc1c1719f67e14324749df200a4b90`).

## The model and the fit

Ornith-1.5-35B-A3B is a Qwen3.5-MoE hybrid: 40 decoder layers, three gated-delta-rule linear-attention layers
for every full-attention layer, 256 routed experts with 8 active per token plus one shared expert, a vision
tower, and one multi-token-prediction layer. The FP8 release is compressed-tensors: per-channel FP8 weights with
dynamic per-token FP8 activations on the attention projections and every expert; the linear-attention
projections, router, embeddings, output head, vision tower and MTP layer stay BF16 (32.6B FP8 + 3.3B BF16
parameters).

Fit arithmetic behind the deliberate partial-card share: 32.6 GB of FP8 weights plus 6.7 GB of BF16 weights is
about 37 GiB, and 1.3x that is a 48 GiB minimum to serve. At `gpu_memory_utilization 0.75` on an 80 GB card the
engine budget is 59.4 GiB; measured: 34.4 GiB of weights, a 22.7 GiB KV pool of 1,157,545 tokens (4.42
full-context sequences), and about 20 GiB of the card untouched. The BF16 checkpoint could not have been served
at half a card (72 GB of weights), which is why this recipe pins the FP8 twin.

## Serving qualification

Every capability probe passed on the deployed configuration at the native 262,144-token context:

- chat with thinking off returns coherent text; with thinking on, the reasoning lands in the engine's
  reasoning field and the answer in `content` (qwen3 reasoning parser);
- a weather tool request returns a structured `tool_calls` entry with the right function and arguments
  (qwen3_xml tool parser, the model's `<function=...><parameter=...>` format);
- a 236,780-token prompt (90% of the window) returns the buried code verbatim in 10.7 s;
- the deployment smoke test (`2 + 2` through chat) passes on every boot.

Text-only serving is deliberate (`--language-model-only`); image and video inputs were not qualified. No
speculative decoding: the MTP layer is not used.

## Selected configuration and its measurement

The recipe pins vLLM v0.30.0, one H100, `gpu_memory_utilization 0.75`, `max-model-len 262144`,
`max-num-seqs 64`, the two parsers above and text-only mode. Its measured lane is the concurrency-64 row of
`experiments/Ornith-1.5-35B-A3B-FP8/serving` (run `20260929T192754Z`, row `04a0751b1020`): 320 random
1024-in/1024-out requests at client concurrency 64, seed 0, temperature 0, ignore-EOS, no prefix reuse.

| Metric | Value |
| --- | ---: |
| Successful / failed requests | 320 / 0 |
| Output token throughput | 3,825.7 tok/s |
| Request throughput | 3.74 req/s |
| Median / P99 TTFT | 369 ms / 1,273 ms |
| Median / P99 TPOT | 16.54 ms / 16.74 ms |
| Benchmark duration | 85.7 s |
| Peak KV pool use | 28.4% |

Why 64: the experiment's grid shows 5.0 ms per token single-stream (about 195 tok/s), 2.4K tok/s at 32
concurrent 4K/4K streams (13.3 ms TPOT), 3.8K tok/s at 64 streams of 1K/1K (16.5 ms), and 7.6K tok/s at 256
short streams but at 31 ms TPOT with the pool 94% full. Sixty-four keeps per-token latency at or under 17 ms on
every 1K-4K workload with the pool under a third full; it is the throughput point that still feels interactive.

Reproduce the selected lane on a supplied host:

```bash
emmy bench experiments/Ornith-1.5-35B-A3B-FP8/serving --ssh user@host \
  --filter "engine.llm.max_concurrent_requests=64"
```

## Emmy

Not qualified: compiler qualification and the Emmy serving lane were out of scope for this onboarding. The
recipe is a stock vLLM recipe and sets no Emmy knobs; there is no golden under this recipe.

## Limitations

- Multimodal inputs and the MTP layer were not qualified; the recipe serves text only without speculation.
- One measurement per multi-stream row; the single-stream row's first repeat is slower than the other two
  because the engine JIT-compiles its linear-attention prefill on the first requests.
- No prefix-cache lane: real agent traffic with shared system prompts will see lower TTFT than the no-reuse
  matrix reports.
- Measured on an H100; the H200 target the BF16 onboarding shell proposes remains unqualified.
