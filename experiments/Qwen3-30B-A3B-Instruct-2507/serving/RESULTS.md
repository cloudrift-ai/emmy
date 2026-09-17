# Qwen3-30B-A3B-Instruct-2507 serving benchmark

The H200 qualification used the stock-vLLM configuration selected by the canonical recipe: TP1, PP1, 262,144-token
context, maximum concurrency 1, `gpu_memory_utilization` 0.90, 4,096 maximum batched tokens, and the Hermes tool
parser. The restricted outbound path required the pinned model snapshot and image to be staged before the run.

## NVIDIA H200 141GB x1

Qualified 2026-09-17 with vLLM 0.17.0 and model revision
`0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe`.

Each repeat used 8 unique random prompts, 4,096 input and 4,096 output tokens, concurrency 1, two warm-ups,
temperature 0, ignored EOS, and seeds 0, 1, and 2.

| Metric | Seed 0 | Seed 1 | Seed 2 |
| --- | ---: | ---: | ---: |
| Successful / failed requests | 8 / 0 | 8 / 0 | 8 / 0 |
| Duration (s) | 152.48 | 152.48 | 152.57 |
| Output throughput (tok/s) | 214.90 | 214.90 | 214.77 |
| Total throughput (tok/s) | 429.81 | 429.81 | 429.54 |
| Mean TTFT (ms) | 127.73 | 142.05 | 137.56 |
| Median TTFT (ms) | 127.84 | 144.27 | 141.34 |
| P99 TTFT (ms) | 141.05 | 153.05 | 148.27 |
| Mean TPOT (ms) | 4.62 | 4.62 | 4.62 |
| Median ITL (ms) | 4.63 | 4.63 | 4.63 |

Raw benchmark logs and the environment record are retained in `results_h200x1.tar.gz`.

The standard Emmy benchmark orchestration could deploy and smoke-test the server, but its client container did not
mount the provider-local model directory. The recorded rows therefore use the same vLLM benchmark client from the
pinned image with `/mnt/models` mounted read-only. No performance number from the interrupted orchestration attempt
is included.
