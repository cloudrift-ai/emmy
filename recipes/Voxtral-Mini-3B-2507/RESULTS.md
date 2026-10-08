# Voxtral Mini 3B (2507) FP16 on one 32 GB V100

Status: serving-qualified with stock vLLM (the 1Cat sm_70 fork) for text and audio input on one Tesla
V100-SXM3-32GB. The Emmy compiler golden is complete on this GPU; Emmy serving is ineligible because its runner
serves text only.

## Qualified deployment

| Item | Value |
| --- | --- |
| Hardware | 1× Tesla V100-SXM3-32GB, SM70 |
| Driver / host CUDA | 580.178.04 / 12.9.86 |
| Model revision | `3060fe34b35ba5d44202ce9ff3c097642914f8f3` |
| Image | `cloudriftai/1cat-vllm-sm70-audio:1.2.3-d76126608` (local build, not yet published) |
| Base image | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` |
| Precision / KV cache | FP16 (Volta has no bfloat16) / FP16 |
| Context / maximum sequences | 32,768 tokens / 16 |
| KV capacity reported at boot | 132,608 tokens (4.05 full-length requests) |
| Input | text and audio (`input_audio` chat parts, `/v1/audio/transcriptions`) |

Voxtral Mini is Mistral's speech model: a 32-layer Whisper-style audio encoder and projector in front of a 30-layer
Ministral 3B decoder, 4.7B parameters, 8.7 GiB of weights in FP16. Stock vLLM has no sm_70 support, so the recipe uses
the 1Cat Volta fork. That image lacks vLLM's audio extra and predates upstream's Voxtral fix for transformers 5.10+
(vllm-project/vllm#44559): without it the engine fails its audio profiling run at boot, and without the audio extra
every audio request fails to decode. `docker/1cat-vllm-sm70/Dockerfile.audio` adds both. The image was built on the
test host and is not in a registry yet, so this recipe deploys only where that image exists.

## Serving performance

Measured 2026-10-08 with seed 0, greedy decoding, forced output lengths and one warmup request. Every measured
request succeeded.

| Workload | Streams | Requests | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 in / 4096 out | 1 | 3 × 4 | 63.0 | 594 ms | 15.7 ms |
| 4096 in / 4096 out | 16 | 128 | 256.4 | 3.52 s | 61.6 ms |
| 8192 in / 256 out | 16 | 80 | 68.3 (2,254 total) | 9.78 s | 192 ms |
| 1024 in / 1024 out | 64 | 320 | 803.3 | 2.49 s | 77.3 ms |
| 256 in / 256 out | 256 | 1,280 | 1,841.4 | 2.54 s | 129 ms |
| LibriSpeech test-clean transcription | 1 | 64 | 57.4 (2.17 clips/s) | 121 ms | 13.2 ms |
| LibriSpeech test-clean transcription | 16 | 320 | 200.6 (7.82 clips/s) | 424 ms | 64.1 ms |

The concurrency-1 repeats agree within 0.05 tok/s. Decode scales poorly with streams; the fork captures CUDA graphs
for batch sizes 1 and 2 only on this card. `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`, which the Llama 3.1 8B V100
recipe needs, made concurrent prefill 7.4x slower here and is not set. The raw run, records and the attention
comparison are in
[`experiments/Voxtral-Mini-3B-2507/serving`](../../experiments/Voxtral-Mini-3B-2507/serving/RESULTS.md) and
[`experiments/Voxtral-Mini-3B-2507/attention_v100`](../../experiments/Voxtral-Mini-3B-2507/attention_v100/RESULTS.md).

Reproduce the single-stream row:

```bash
emmy bench experiments/Voxtral-Mini-3B-2507/serving --ssh user@host --filter benchmark.max_concurrency=1 \
  --filter benchmark.random_input_len=4096
```

## Capabilities

- Transcription: an English speech clip was transcribed word for word; a second clip was transcribed correctly with
  no language given.
- Audio chat: a two-clip comparison prompt got a coherent answer matching the model card's example. The chat path
  rejects stereo audio; send mono, as the model card's client does.
- Tool calls: with `tool_choice: required` the server returned a structured `get_weather({"city": "Paris"})`. With
  `auto` the model answered in prose instead, for a text and for a spoken request.
- Context: a needle prompt was recalled up to 30,343 tokens. At 32,553 tokens, the full window, the answer was wrong
  in three tries and changed between them. Treat the last ~2K tokens of the window as unreliable.

## Emmy compiler qualification

Measured 2026-10-07 on the same V100 at deployable `-O3`, five warmups and 20 timed iterations per target. The golden
[`golden/v100_sxm3_sm70.json`](golden/v100_sxm3_sm70.json) holds nine traced programs: the pre- and post-attention
serving twins at any width, M=1 and M=8, which stand for all 30 identical decoder layers, plus embeddings, final
normalization and the output head. vLLM owns attention, the Whisper-style audio encoder and the audio projector. The
file has 45 fresh-lowered kernels, 39 measured kernel rows and six routing decisions. Every target passed strict
correctness against eager PyTorch and has positive Emmy, eager and `torch.compile` timings.

| Target | Emmy (µs) | Eager (µs) | `torch.compile` (µs) | Emmy / `torch.compile` |
| --- | ---: | ---: | ---: | ---: |
| Post-attention, any width (512 rows) | 2,609 | 1,539 | 1,336 | 1.95 |
| Post-attention, M1 | 376 | 287 | 199 | 1.89 |
| Post-attention, M8 | 407 | 367 | 295 | 1.38 |
| Pre-attention, any width (512 rows) | 831 | 507 | 313 | 2.65 |
| Pre-attention, M1 | 139 | 114 | 51 | 2.70 |
| Pre-attention, M8 | 156 | 153 | 93 | 1.69 |
| Output head, M1 | 896 | 1,173 | 837 | 1.07 |
| Final normalization, M1 | 3.7 | 44.5 | 5.3 | 0.70 |
| Embeddings, M1 | 1.6 | 6.1 | 2.6 | 0.59 |

The prior's own picks were correct but far off: up to 1,278x slower than `torch.compile` on the any-width
pre-attention path, and the any-width post-attention path tripped the 2 s kernel watchdog. Each twin was recorded
under the cut route the Llama 3.1 8B V100 golden records for the same structure, with cross-CTA splits pinned off (a
`g8k` split piece failed with a misaligned address on this card). Then 24 pieces were tuned with `emmy run --kernel
… --tune 30` (the output head with 40 rows) and recorded again with `--record-greedy`. That took the any-width
post-attention path from 14.1 ms to 2.6 ms and the output head from 1,227 to 896 µs. Seven targets still miss the
`torch.compile` bar by 1.07–2.70x; these are schedule and code-generation losses on the Volta tensor-core atom, not
missing coverage. Final normalization and embeddings beat it. `emmy golden check` confirms every program matches its
fresh lowering.

## Emmy eligibility

Ineligible. The first failed gate is the serving runner: Emmy's generation runner serves text only and refuses a
prompt with multimodal features, so it cannot serve an audio checkpoint. Compute capability, the trace and runner
path for the Llama-architecture decoder, the unquantized checkpoint and the golden gates pass. No Emmy kernel ran in
the serving measurements above, and no serving image was built.

## Limits

- The image must be published before this recipe deploys on another host; that needs approval.
- The 16-stream 4096/4096 row ran 34 minutes, past the 20-minute row cap; it was kept, not repeated. The 32-stream
  4096/4096 row needs twice the KV pool and was not run.
- The transcription rows measure throughput and latency, not word error rate.
- Requalify after changing the image, model revision, driver, context length or attention backend.
