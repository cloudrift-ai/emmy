# Qwen3-30B-A3B-Instruct-2507 on one H200 141GB

Qualified 2026-09-17 against repository revision `22f842497ba8f10e78c826789929b1623618e323`.

## What was measured

| Item | Value |
| --- | --- |
| Model | `Qwen/Qwen3-30B-A3B-Instruct-2507@0d7cf23991f47feeb3a57ecb4c9cee8ea4a17bfe` |
| GPU | 1 x NVIDIA H200 141GB, compute capability 9.0, driver 595.58.03, CUDA 13.2 |
| Engine | Stock vLLM 0.17.0, image `vllm/vllm-openai:v0.17.0` |
| Validation image | Local offline mirror, image ID `sha256:db8f26287ccc5ec31e8e47cc1cf5149e8537f0bb41f228d2fefc77186a02d8ed` |
| Serving shape | TP1, PP1, context 262,144, max concurrency 1, `gpu_memory_utilization` 0.90 |
| Workload | 8 prompts, 4,096 input / 4,096 output tokens, concurrency 1, temperature 0, ignored EOS, 2 warm-ups, seeds 0-2 |

Three clean repeats were run for the selected configuration. The table reports the range across those complete runs.

| Metric | Result |
| --- | ---: |
| Successful / failed requests | 24 / 0 |
| Benchmark duration | 152.48-152.57 s |
| Output token throughput | 214.77-214.90 tok/s |
| Total token throughput | 429.54-429.81 tok/s |
| Mean TTFT | 127.73-142.05 ms |
| Median TTFT | 127.84-144.27 ms |
| P99 TTFT | 141.05-153.05 ms |
| Mean TPOT | 4.62 ms |
| Median inter-token latency | 4.63 ms |

The service reached health in 61.8 s after container start. Weight loading took 13.6 s, cache warm-up took 24.7 s,
and CUDA graph capture took 2.0 s. The engine allocated a 744,304-token KV cache and reported 2.84x capacity at the
full 262,144-token context. The running server used 129,582 MiB of the GPU's reported 143,771 MiB.

## Capability checks

| Gate | Result |
| --- | --- |
| Coherent chat | Pass — returned `Tokyo` for a capital-city question |
| Tool calling | Pass — returned a structured `get_weather({"city": "Paris"})` call |
| Context retrieval | Pass at 123,816 and 237,815 input tokens with exact marker recovery |
| Native context allocation | Pass — vLLM started with a 262,144-token context window |

The checkpoint is an instruct-only model. It uses Hermes-style tool calls and has no separate reasoning field, so the
recipe intentionally enables the Hermes tool parser without a reasoning parser.

## Fit and selected configuration

The BF16 checkpoint occupies about 61.1 GB. TP1 avoids communication overhead and leaves enough H200 memory for the
native context, runtime buffers, and a single full-length request. The selected 4,096-token batched-token limit bounds
prefill activation memory while vLLM's chunked prefill handles longer prompts. Concurrency 1 is intentional: it keeps
the entire advertised context available to one request and matches the measured interactive lane.

## Limitations

- FCBK's rental network could not fetch the model or image from their public origins during qualification. The exact
  model snapshot and a local mirror of the stock image were staged manually. The recipe uses the normal public image
  and Hugging Face model identifiers; fresh-VM deployment at FCBK still needs working outbound artifact access or a
  supported provider-local artifact path.
- The built-in Emmy benchmark client did not mount FCBK's manually staged `/mnt/models` directory. The same vLLM
  benchmark client and workload were therefore run from the pinned image with that directory mounted read-only.
- Emmy compiler coverage was not evaluated. This is a stock-vLLM serving qualification, so no compiler golden or
  Emmy comparison is claimed.
- A synthetic prompt made almost entirely of one repeated token produced repetitive output. Realistic numbered-record
  retrieval prompts passed at 123,816 and 237,815 tokens; the latter is the largest end-to-end context result here.

## Reproduce

On a host that can reach Hugging Face and the container registry:

```bash
emmy deploy ssh --recipe recipes/Qwen3-30B-A3B-Instruct-2507 --ssh USER@HOST
```

Run the retained serving experiment with the same platform to reproduce the benchmark workload:

```bash
emmy bench experiments/Qwen3-30B-A3B-Instruct-2507/serving --ssh USER@HOST
```
