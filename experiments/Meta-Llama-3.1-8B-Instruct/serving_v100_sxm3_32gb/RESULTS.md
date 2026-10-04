# Llama 3.1 8B Instruct FP16 serving on one 32 GB V100

## Question and protocol

Does the pullable, pinned SM70 image serve the recipe's FP16 Llama 3.1 8B Instruct checkpoint at the configured
65,536-token context? Emmy deployed a fresh server on one Tesla V100-SXM3-32GB (SM70), driver 580.178.04, Ubuntu
24.04.1, with the public NousResearch mirror at `d10aef7999a2b5ba950ab3974312feeedbfe0b77` and the separate
tool-aware tokenizer at `db1f81ad4b8c7e39777509fac66c652eb0a52f91`. The image was
`cloudriftai/1cat-vllm-sm70@sha256:6f34e0b247a78ca65f88f305b1f1cc52c9020ecb83a5ca21df0599676dc443d3`.
Volta attention used the paged-KV gather path and prefix caching was disabled.

The `vllm bench serve` row ran three repeats of 32 requests with 512 random input tokens, 256 forced output tokens,
concurrency 8, two warmup requests, and greedy decoding. Emmy tore the server down after the row. A second deployment
of the same canonical recipe checked a tool call and a long prompt, then was torn down.

## Results

All 96 benchmark requests succeeded. The three output-throughput repeats were 236.28, 237.35, and 237.40 tokens/s:
237.01 ± 0.63 tokens/s (sample standard deviation). Mean time to first token was 1201.5 ms and mean time per output
token was 29.16 ms across the repeat means. Model load and warmup took 87.87 seconds for the benchmark deployment.

The standalone server returned HTTP 200 with a parsed `get_weather` tool call containing `{"city": "Paris"}`. A
separate 59,999-input-token request generated three tokens with no failed request or OOM. Its time to first token was
113.00 seconds. These are correctness and capacity probes; the long-prompt number is one request, not a latency
distribution.

The 2026-08-13 run reported 280.80 ± 9.64 output tokens/s on a locally resolved image digest that cannot be pulled
from the registry. The present number is 15.6% lower, but the image, driver, and attention path changed together;
this is a historical comparison, not a measured effect of the new setting. A server with the adapter loaded used the
same current image and a similar 512/256 concurrency-8 workload, delivering 236.38 base tokens/s in the companion LoRA
experiment. That suggests enabling one adapter had little base-request throughput cost here, although the protocols
used 32 versus 40 requests and were run separately.

## Evidence and limits

Run time: 2026-10-03 19:01:56 UTC; run ID `20261003T190156Z`; experiment status `succeeded`. Docker Engine was
29.8.1 and host CUDA toolkit 12.9.86. The Git LFS archive `results_v100x1.tar.gz` contains the dated raw run with
`v100x1_5835c0e10702.experiment.yaml`, its benchmark and server logs, `benchmark.log`, and
`benchmark_v100_x_1.log`. Its `diagnostics/` members are `tool_call.request.json`, `tool_call.response.json`, and
`long_context.benchmark.log` from the standalone probes.

The maximum context was configured as 65,536 tokens; the material prompt tested 59,999 input tokens. The recipe's
reduced-memory 0.65 variant was not tested in this run. This is a stock 1Cat/vLLM deployment; Emmy compiled kernels
were not used. Adapter performance is recorded in the separate LoRA experiment.
