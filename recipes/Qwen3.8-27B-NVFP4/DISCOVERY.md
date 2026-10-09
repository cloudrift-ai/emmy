# Qwen3.8-27B-NVFP4 discovery

## Measured serving performance (2026-10-09)

The full model serves with the recipe's `golden/rtx5090_sm120.json` and `--strict-evidence`. The measured setup is
one RTX 5090, one request, a BF16 trunk, decode bucket 16, prefill chunk 64, M1 tier 0, fast math, no prefix cache,
and vLLM 0.23.0. Emmy uses eager execution; stock vLLM uses compiled CUDA graphs. Strict server boot took 523 s.

| Formatted prompt tokens | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|
| 25 | 26.4 | 0.49 (warm) | 63.5 | 0.69 |
| 227 | 26.3 | 0.71 | 63.0 | 0.29 |
| 1027 | 26.2 | 1.47 | 62.9 | 1.18 |

These are single streaming requests with natural EOS. Decode throughput is `(output tokens - 1) / (last - first)`;
time to first token (TTFT) is measured from request submission. Evidence: `serving-probe-stage2.md` (stock) and
`serving-probe-sweep.md` (Emmy) in
[PR #1120](https://github.com/cloudrift-ai/emmy/pull/1120). Earlier serving results and comparison limits are in
[the performance history](../../plans/qwen38-5090-followups.md#serving-and-the-decode-profile).

## Assessment on 2026-10-05

This historical assessment retained a serving-blocked label; full-model serving was enabled by
[PR #1023](https://github.com/cloudrift-ai/emmy/pull/1023). The measurements above supersede that label.

The recipe was assessed as best-effort (compiler-qualified, serving-blocked), heat 55. The ecosystem was strong:
the base Qwen/Qwen3.8-27B held 6,821,761 30-day downloads and 16,960 likes
(https://huggingface.co/Qwen/Qwen3.8-27B), and the verified NVFP4 nodes carry the bulk of the quant volume
(unsloth/Qwen3.8-27B-NVFP4 at 2,365,557 30-day downloads,
https://huggingface.co/unsloth/Qwen3.8-27B-NVFP4). The serving-blocked tag meant engine support for the NVFP4 path was
not yet qualified in that assessment; the specific Inferact checkpoint also did not resurface in its bounded
identity checks (bounded search, not deletion evidence).
