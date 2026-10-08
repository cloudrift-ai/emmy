# Voxtral Mini 3B serving on one 32 GB V100

## v100x1 — Tesla V100-SXM3-32GB, 2026-10-08

### Question and protocol

How does stock vLLM serve Voxtral Mini 3B (`mistralai/Voxtral-Mini-3B-2507@3060fe34b35ba5d44202ce9ff3c097642914f8f3`)
on one V100, for text generation and for speech transcription? Emmy deployed each row on a fresh server: one Tesla
V100-SXM3-32GB (SM70), driver 580.178.04, host CUDA 12.9.86, image `cloudriftai/1cat-vllm-sm70-audio:1.2.3-d76126608`
(the 1Cat sm_70 fork plus vLLM's audio extra, built from `docker/1cat-vllm-sm70/Dockerfile.audio` on the host; not
yet in a registry). FP16 weights, FP16 KV cache, Mistral tokenizer/config/weight format, `FLASH_ATTN_V100`, 4,096
batched tokens, prefix caching off, 0.88 memory utilization, 32,768-token context. The KV pool holds 132,608 tokens.

The text rows follow the datacenter profile with seed 0, greedy decoding, forced output length and one warmup request.
The 4096/4096 row at 32 streams needs 262,144 KV tokens, twice the pool, so it was not run. The concurrency-1 row ran
three repeats of four requests (halved from eight after an earlier attempt passed the 20-minute row cap). The
transcription rows send LibriSpeech `test-clean` clips to `/v1/audio/transcriptions`; that client reports throughput
and latency, not word error rate.

### Results

Every measured request succeeded (run `20261008T011351Z` for text, `20261008T030329Z` for transcription).

| Workload | Streams | Requests | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4096 in / 4096 out | 1 | 3 × 4 | 63.0 | 594 ms | 15.7 ms |
| 4096 in / 4096 out | 16 | 128 | 256.4 | 3.52 s | 61.6 ms |
| 8192 in / 256 out | 16 | 80 | 68.3 (2,254 total) | 9.78 s | 192 ms |
| 1024 in / 1024 out | 64 | 320 | 803.3 | 2.49 s | 77.3 ms |
| 256 in / 256 out | 256 | 1,280 | 1,841.4 | 2.54 s | 129 ms |
| LibriSpeech clip transcription | 1 | 64 | 57.4 (2.17 clips/s) | 121 ms | 13.2 ms |
| LibriSpeech clip transcription | 16 | 320 | 200.6 (7.82 clips/s) | 424 ms | 64.1 ms |

The three concurrency-1 repeats agree to 0.05 tok/s (63.02–63.05) and 1 ms of TTFT. Decode scales poorly: 16 streams
give 4.1x the single-stream output rate at 3.9x the per-token time. The fork captures CUDA graphs for batch sizes 1
and 2 only on this card, so wider decode steps run eager; that is a likely cause, not a measured one. The 16-stream
4096/4096 row ran 34 minutes, past the 20-minute row cap; it was kept rather than repeated.

### Attention path

The first full run used `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1`, the setting the Llama 3.1 8B V100 recipe needs.
With it, concurrent prefill collapsed: 8192/256 at 16 streams moved 304 total tok/s (1.4 s per output token) and the
64-stream row stalled near 6 generated tok/s before it was stopped. The paged path ran 7.4x faster with no failed
request, so this lane drops the setting; see [`../attention_v100/RESULTS.md`](../attention_v100/RESULTS.md).

### Capability probes (final configuration)

A separate deploy of `recipes/Voxtral-Mini-3B-2507` passed the text and inline-audio smoke tests, then:

- transcription of a stereo MP3 speech clip with `language: en` was word-for-word correct, and a second clip was
  transcribed correctly with no language given;
- an audio chat with two clips answered which speaker was more inspiring, matching the model card's example;
- `tool_choice: required` returned a structured `get_weather({"city": "Paris"})`; with `auto` the model answered in
  prose instead of calling the tool, for both a text and a spoken request;
- a needle prompt was recalled at 3,513 through 30,343 tokens. At 32,553 tokens (the full window) the answer was wrong
  in all three tries and changed between them ("PALICAN", "PALACE", "PALM"); the gather path recalled it once.

The chat path rejects stereo audio (`audio_array.ndim=2`); clients must send mono, as the model card's client does.

### Archive

`results_v100x1.tar.gz` holds `2026-10-08_01-13-51/` (all text rows; its two transcription rows failed because the
client was sent `--temperature`, which the `openai-audio` backend refuses) and `2026-10-08_03-03-29/` (the two
transcription rows after that fix). Each holds the system-only experiment records, benchmark logs and server logs.
