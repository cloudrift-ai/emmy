# Native text server

`emmy-server` is a thin Axum adapter over `emmy-runtime`. One dedicated thread loads and owns the CUDA generator,
checkpoint tokenizer, and template. HTTP tasks only validate JSON, acquire admission, and transport bounded output.
The runtime has no HTTP dependency. Python prepares artifacts and launches the executable; it does not execute model
operations during requests. vLLM remains the default serving engine.

## Frontend and text reuse

The selected adapter uses Axum 0.8, Hugging Face tokenizers 0.22, and MiniJinja 2 with Python-method compatibility.
The full Dynamo frontend owns distributed discovery and execution services. Its standalone frontend crates separate
protocol, renderer, and tokenizer components, but their demonstration handler does not supply local GPU ownership,
admission, or cancellation. async-openai supplies client types rather than these server behaviors. A narrow Serde
request type also rejects unsupported options instead of accepting a larger protocol whose semantics are absent.

Direct tokenizers and MiniJinja reuse the underlying text mechanisms without the standalone frontend's additional
model parsers, hub clients, caches, and protocol families. The selected server resolves 155 unique dependency-tree
entries and produces an approximately 11 MiB release executable on Linux x86-64; the first local release build took
10.4 seconds with an already populated dependency cache. This is not a clean-build timing or a measured comparison
with Dynamo. The dependency scope, rather than an unmeasured build-speed claim, determines the choice. See the
[recorded reuse investigation](../../experiments/Qwen3-0.6B/native_baseline/INVESTIGATION.md).

Preparation bundles `tokenizer.json` and the checkpoint's actual `chat_template.jinja`. MiniJinja renders it with
`enable_thinking=false` and `add_generation_prompt=true`; no hand-written Qwen prompt format or fallback tokenizer
exists. Checkpoint tests compare rendered text and token IDs against Transformers for multilingual input, whitespace,
special tokens, and multi-turn assistant history. Incremental decoding retains incomplete Unicode bytes. Final decode
flushes incomplete output consistently with the checkpoint decoder, including genuine replacement characters.

## API contract

The API exposes `/health`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`. Completions accept one text
prompt; chat accepts system, user, and assistant text messages. Requests select the exact served model ID. Streaming
uses SSE with a final `[DONE]`; non-streaming returns the corresponding completion object. Output limits default to
128 tokens, temperature to zero, top-p to one, and seed to zero. Seeds are unsigned 64-bit integers. Prompt plus output
budget must fit the configured context; the server rejects overflow instead of silently truncating.

Stop accepts up to four nonempty strings, each at most 256 UTF-8 bytes. The decoder retains any suffix that could
complete a stop string, so stops split across tokens never leak into output. Usage counts actual token IDs, including
EOS or tokens consumed to complete a stop. Finish reasons are `stop` and `length`; zero output budget yields `length`.
Streaming usage is emitted when `stream_options.include_usage` is true. Neutral benchmark-client fields
`repetition_penalty=1` and `logprobs=null` are accepted; non-neutral values and all unknown fields are rejected.
The benchmark client may select `ignore_eos=true` for fixed output lengths; output/context limits and stop strings
still apply. Tools, multimodal content, batching, log probabilities, and additional sampling controls are outside this
API subset.

## Admission, cancellation, and failures

One owned semaphore permit travels with each worker job. Additional requests receive HTTP 429. The output channel
holds eight events; a slow client stops further GPU submission when it fills. Dropping either a streaming body or a
non-streaming handler closes the receiver. The worker checks that cancellation signal before each GPU step and while
waiting for output capacity. It stops future submissions, waits for the current step, and only then releases admission.
No unbounded request queue or background drain defeats backpressure.

The generator synchronizes each step at its CPU observation boundary. Execution or decoding failures make readiness
false, release device state while admission is still held, report an error, and retire the execution thread. There is
no automatic retry or context recreation. A failed or panicking worker requires process restart. Invalid requests
fail before GPU submission and do not retire a healthy runtime.

Readiness remains false during load. SIGINT or SIGTERM makes it false, cancels active generation, drains HTTP tasks,
and joins the execution thread. A 30-second shutdown deadline retires the process if GPU completion or transport
cannot finish. This does not promise recovery from a hung GPU inside the same process.

## Build and validation

`make native-dist` packages `emmy-server` and `emmy-runtime-worker` from the same locked workspace build in one
platform-specific archive named with the source revision. Install both in the launcher's PATH. Server startup never
invokes Cargo. A compatible NVIDIA driver is required; a prepared artifact needs neither Python nor NVCC on PATH.

Cargo tests cover HTTP validation, SSE, usage, failure visibility, admission, bounded output, and disconnect behavior.
Opt-in Python tests accept `--native-checkpoint` for tokenizer/template parity and `--native-artifact` for HTTP GPU
qualification. The latter starts the real server with Python and NVCC absent from PATH and checks seeded replay,
streaming parity, stop strings, limits, busy responses, cancellation recovery, and clean shutdown. These functionality
checks establish neither a serving speedup nor production throughput.
