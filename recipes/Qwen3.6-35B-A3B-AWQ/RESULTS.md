# Qwen3.6-35B-A3B-AWQ on 1x NVIDIA Tesla V100 SXM3 32GB

Verified 2026-10-11 on an exact single V100 SXM3 32GB profile (sm_70, Volta). The recipe requalifies as
`maintained`: the pinned 1Cat/vLLM sm70 image serves the AWQ checkpoint, all capability gates pass, and the
selected lane measures clean under 4-way concurrency.

## Serving

| Item | Value |
| --- | --- |
| Engine / image | 1Cat/vLLM sm70, `cloudriftai/1cat-vllm-sm70:1.0.0`, vLLM 0.22.x-generation build, `VLLM_ATTENTION_BACKEND=FLASH_ATTN_V100` |
| Model | QuantTrio/Qwen3.6-35B-A3B-AWQ @ `119886a1072372348f73ef0df2d801cdcc0f455b`, qwen3_5_moe (MoE: 40 layers, 256 experts, 8 routed + 1 shared per token), AWQ 4-bit group-128 zero-point gemm, ~17.5 GB weights, mrope (native context 262,144), vision-capable, thinking |
| Deploy | TP=1, 1x V100 SXM3 32GB (32,768 MiB), driver 580.178.04, CUDA 12.9.86, gpu_memory_utilization 0.88, max_concurrent_requests 4, max-num-batched-tokens 4096, context 32,768, qwen3_coder tool parser + qwen3 reasoning parser |
| Repo revision | Emmy `bb7ee02e` (0.3.22) |

Fit: 35B total params at AWQ 4-bit (~0.5 bytes/eff param + FP16 unquantized seams) gives ~17.5 GB of weights
plus per-token 3B active experts; min-to-serve ~23 GB. The 32 GB V100 holds it at gpu_mem_util 0.88 with a
measured 3.46 GiB KV pool (38,016 tokens, server-reported 4.20x max 32K concurrency); weights load in 38 s.
No tensor parallelism is needed; the 16 GB profile does not fit (weights + KV exceed 16 GB).

Workload and result (run ID `20261011T101558Z`, the recipe's matrix row exactly: 32 prompts, 1,000/1,000
tokens, concurrency 4, greedy, forced output length):

| Metric | Value |
| --- | --- |
| Completed / failed | 32 / 0 |
| Output tok/s | 212.7 (peak 256.0), total tok/s 425.5 |
| TTFT mean / median / P99 (ms) | 2635 / 664 / 16785 |
| TPOT mean / median / P99 (ms) | 16.18 / 16.14 / 16.39 |
| ITL mean / median / P99 (ms) | 16.18 / 16.13 / 18.91 |
| Wall | 150.4 s bench; deploy+load 372 s cold |

TPOT is flat across the queue (P99 16.39 ms vs median 16.14) - the decode side is stable and 4-way batching
is the operating point; TTFT carries the queue (median 664 ms, P99 16.8 s at saturation). Single-stream decode
sits near 87 tok/s theoretical (1000 ms / 16.2 ms); the 212.7 tok/s point is 5.2x that baseline.

Capability validation, same deployment: coherent completion ("The capital of France is" -> " Paris, a city
renowned for its rich history..."); 11,016-token input answered coherently at the 30K probe cap in 2.5 s
wall, and the 32K claim rests on the server's own 38,016-token KV pool (never claimed from startup alone);
tool calling returns structured `tool_calls` (`get_weather`, finish `tool_calls`) once the completion budget
is large enough - with `max_tokens` 512 or less the model spends the budget in thinking prose and never emits
the tool format, so tool traffic needs `max_tokens` >= ~2,048 or thinking disabled; reasoning lands in the
`reasoning` field when the chat template opens the think block and in `content` otherwise (the checkpoint's
default template does neither for a bare ask, which is why `preserve_thinking` is set in the experiment).

## Emmy compiler status

Eligible: **no** - gate 4. V100 sm_70 is accepted by the Emmy CUDA backend and the Qwen3.5-MoE text path has a
trace/runner route, but this checkpoint's AWQ quantization has no Emmy loader/serving path that keeps the
stored 4-bit representation in the deployed graph (reference-only dequantization does not count), and no
complete golden therefore commits for this checkpoint. No `emmy-blocked` tag: the gap is quantization-specific
and card-independent, not a blocker of a named engine release; the same card does run the FP16/AWQ-INT4 dense
families, and an FP16 or FP8 AWQ-free variant of this model would re-qualify the golden. No golden, no corpus
cases, no bounded fix under `emmy/` this run.

## Reproduce

```bash
emmy bench --filter "deploy.gpu=NVIDIA Tesla V100 SXM3 32GB" \
  --ssh <target> experiments/Qwen3.6-35B-A3B-AWQ/serving
```

Raw run and records: `experiments/Qwen3.6-35B-A3B-AWQ/serving/results_v100x1.tar.gz` (run
`20261011T101558Z`, row `v100x1_5835c0e10702`).

## Limits

- One recorded row, one repeat: no repeat-variability evidence; peak concurrent requests reached 8 against a
  max of 4 (corpus oversampling).
- `cloudriftai/1cat-vllm-sm70:1.0.0` is the only public tag of that registry; 1Cat v1.2.x (vLLM 0.21.x base)
  is not published there, so this pin tracks the image as-is.
- The 32K served context is the recipe's cap, not the checkpoint's native 262,144; 8K+ 4K/4K rows of the
  consumer matrix are inapplicable at this capacity.
- Tool-calling and thinking-budget interplay is template/prompt sensitive; see the capability row above.
- Multimodal: text-only qualification; `model.input_modalities` unset.
