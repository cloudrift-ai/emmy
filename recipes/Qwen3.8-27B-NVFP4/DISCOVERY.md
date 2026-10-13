# Qwen3.8-27B-NVFP4 discovery

## Measured serving performance (2026-10-13)

The full model serves with the recipe's `golden/rtx5090_sm120.json` and `--strict-evidence`. The measured setup is
one RTX 5090, one request, a BF16 trunk, decode bucket 16, prefill chunk 64, M1 tier 0, fast math, no prefix cache,
and vLLM 0.23.0. Emmy uses eager execution; stock vLLM uses compiled CUDA graphs. Strict server boot took 646 s.
Stock vLLM was not run again; its columns are the 2026-10-09 measurement.

| Formatted prompt tokens | Output tokens | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|---:|
| 25 | 8 | 29.5 | 0.61 (warm) | 63.5 | 0.69 |
| 227 | 36 | 29.2 | 1.05 | 63.0 | 0.29 |
| 1027 | 229 | 29.2 | 1.19 | 62.9 | 1.18 |

These are single streaming requests with natural EOS. Decode throughput is `(output tokens - 1) / (last - first)`;
time to first token (TTFT) is measured from request submission. All three answers are correct. The short and
medium answers match the 2026-10-09 ones token for token; the long answer is worded differently and gives the same
facts. Evidence:
[the measurement report on PR #1160](https://github.com/cloudrift-ai/emmy/pull/1160#issuecomment-6150058296).

The gated DeltaNet layers take a prompt in chunks of 64, 16 and 1 tokens, widest first, so each prompt length
weighs the three widths differently. Against 2026-10-09, the width-16 program is about twice as slow, because its
conv kernel recomputes the qkv projection once per conv tap; the width-64 program is about 20% faster; and the
width-1 program, which also runs decode, is 7–10% faster. That makes decode faster at every length, the 227-token
prompt (two width-16 chunks) slower to its first token, and the 1027-token prompt (no width-16 chunk) faster.

### Previous measurement (2026-10-09)

Same setup with the golden of that date. Strict server boot took 523 s.

| Formatted prompt tokens | Emmy decode tok/s | Emmy TTFT s | Stock decode tok/s | Stock TTFT s |
|---:|---:|---:|---:|---:|
| 25 | 26.4 | 0.49 (warm) | 63.5 | 0.69 |
| 227 | 26.3 | 0.71 | 63.0 | 0.29 |
| 1027 | 26.2 | 1.47 | 62.9 | 1.18 |

Evidence: the measurement reports in
[PR #1120's discussion](https://github.com/cloudrift-ai/emmy/pull/1120#issuecomment-6078603008).
The earlier full-Emmy strict boot of the first golden decoded at about 0.13 s per token
([PR #1023](https://github.com/cloudrift-ai/emmy/pull/1023), 2026-10-04).
[PR #993](https://github.com/cloudrift-ai/emmy/pull/993) was closed; its mixed route ran only the MLPs in Emmy
and measured against eager vLLM, so that comparison does not apply to the compiled-graph baseline above.

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
