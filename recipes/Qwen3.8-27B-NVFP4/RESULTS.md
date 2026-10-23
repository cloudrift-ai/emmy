# Qwen3.8-27B NVFP4 on one RTX 5090

`Inferact/Qwen3.8-27B-NVFP4` (revision `6128240ebaf4eaa7bad2b3d1c72c37d677c5f462`, ModelOpt NVFP4 W4A4, group
16) served by stock vLLM v0.32.0 on one NVIDIA GeForce RTX 5090 (sm_120, 32 GB). The recipe is the single-card
replacement for the 4x V100 SXM2 16GB production lane of `recipes/Qwen3.8-27B-FP8`, whose prefill under
concurrent requests is the problem this deployment answers. Measured 2026-10-21 on a rented CloudRift node
(driver 580.159.03); the raw run is `experiments/Qwen3.8-27B-NVFP4/serving/results_rtx5090x1.tar.gz`, run
`20261021T210320Z`, and the experiment report interprets it row by row.

## Qualified deployment

| Item | Value |
| --- | --- |
| Image | `vllm/vllm-openai@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34` (vLLM v0.32.0) |
| Shape | TP1, text-only, context 114,688, cap 16, `gpu_memory_utilization` 0.97, FP8 key/value cache, 4,096-token prefill chunk, no prefix caching |
| Backends | FlashInfer CUTLASS NVFP4 GEMM, FlashInfer attention (FP8 cache), FlashInfer Gated DeltaNet prefill |
| Parsers | `--tool-call-parser qwen3_coder`, `--reasoning-parser qwen3`, `--enable-auto-tool-choice` |
| Weights resident | 23.21 GiB: the MLPs, attention projections and Gated DeltaNet out_proj are NVFP4; embed_tokens, lm_head and the Gated DeltaNet input projections stay BF16 (14.7 GiB) |
| Key/value pool | 3.83 GiB = 117,666 FP8 tokens (1.03x concurrency at 114,688) |
| Load to health | 146 s: weights 18.7 s, torch.compile 33.7 s, CUDA graphs 11 s, warm-up 35 s |

## Serving performance

Random prompts through `vllm bench serve`, temperature 0, ignored EOS, two warm-ups, one run per row (three
repeats on the single-stream row, identical to the hundredth of a second), client on the node.

| Prompt / output | Concurrency | Requests | Output tok/s | Median TTFT | P99 TTFT | Median TPOT |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4,096 / 4,096 | 1 | 3 x 3 repeats | 60.9 | 375 ms | 376 ms | 16.3 ms |
| 4,096 / 4,096 | 4 | 32 | 205.2 | 1,090 ms | 1,498 ms | 19.2 ms |
| 4,096 / 4,096 | 8 | 64 | 405.4 | 1,344 ms | 2,956 ms | 19.4 ms |
| 8,192 / 256 | 4 | 20 | 129.2 | 1,920 ms | 3,033 ms | 23.4 ms |
| 1,000 / 1,000 | 4 | 16 | 213.9 | 340 ms | 344 ms | 18.4 ms |
| 1,000 / 1,000 | 8 | 32 | 429.9 | 559 ms | 678 ms | 18.1 ms |
| 1,000 / 1,000 | 16 | 64 | 630.1 | 846 ms | 1,994 ms | 20.1 ms |
| 4,096 / 512 | 16 | 64 | 446.6 | 2.2 s | 16.1 s | 26.7 ms |
| 16,384 / 512 | 16 | 48 | 138.5 | 40.2 s | 53.0 s | 30.9 ms |
| 32,768 / 512 | 16 | 32 | 68.1 | 97.4 s | 113.2 s | 34.7 ms |

One request at a time, 64-token outputs: 16K prompts prefill in 1.72 s (9,500 tok/s), 32K in 4.13 s (7,900), 64K in
11.07 s (5,900), 112K in 25.9 s (4,300). The 256-prompt row at cap 64 and the 1,000-token row at cap 32 failed, the
first because the engine did not boot with 64 slots (see Limits), the second on a dropped SSH session to the node.

## Against the production lane

The 4x V100 SXM2 lane's own report, same prompt shapes, cap equal to client concurrency:

| Row | RTX 5090 x1 | V100 x4 |
| --- | ---: | ---: |
| 1K prompts, cap 4: output tok/s, median / P99 TTFT | 213.9, 0.34 / 0.34 s | 40.8, 1.02 / 2.50 s |
| 1K prompts, cap 8 | 429.9, 0.56 / 0.68 s | 76.8, 1.69 / 4.18 s |
| 1K prompts, cap 16 | 630.1, 0.85 / 1.99 s | 133.8, 3.31 / 5.83 s |
| Per-token decode latency at cap 16 | 20 ms | 105 ms |
| 4K prompts at 16 concurrent: median / P99 TTFT | 2.2 / 16.1 s | 6.2 / 22.3 s |
| 16K prompts at 16 concurrent | 40.2 / 53.0 s | 22.8 / 112.5 s |
| 32K prompts at 16 concurrent | 97.4 / 113.2 s | 132.2 / 266.0 s |
| One 16K / 32K / 64K prompt | 1.7 / 4.1 / 11.1 s | 5.6 / 12.2 / 28.0 s |
| One 112K (5090) or 128K (V100) prompt | 25.9 s | 75.0 s |
| Largest context | 114,688 | 262,144 |
| Key/value pool | 117,666 tokens | 288,281 tokens |

At the production prompt sizes the single card gives about five times the output throughput, first tokens three
to four times sooner at the median and up to seven times sooner at P99, and five times faster decode. A lone long
prompt prefills 2.5 to 3.2 times faster at every length. The card loses one thing: the pool. Each request pins
147 MiB of float32 Gated DeltaNet state before any key/value token, so at 16 concurrent only about 13 requests of
4K tokens, 6 of 16K or 3 of 32K are resident and the rest wait for a slot. That is why the 16K row's median first
token is longer than on the four V100s while its P99 and its batch time (178 s against about 620 s) are far
shorter. A deployment whose traffic is mostly prompts above 16K at 16 concurrent should set the cap nearer the
resident count, or accept the queue.

## Validated capabilities

Run against the same configuration outside the harness, through the chat endpoint on the node:

- Coherent answer: "What is 12 times 12? Answer with the number only." returns `144` (thinking off).
- Tool calling: a `get_weather` function with `tool_choice: auto` returns `finish_reason: tool_calls` and a
  structured call with `{"city": "Oslo"}`.
- Reasoning: with thinking on, 45 reasoning tokens land in the response's `reasoning` field and the content holds
  only the answer.
- Context: a passphrase planted at 40% depth of a 16,019-, 32,019-, 65,012- and 112,020-token prompt is returned
  exactly, in 1.67, 3.95, 10.78 and 25.73 s to first token. 114,688 is the largest context one request can take;
  the native 262,144 window needs an 8.2 GiB FP8 pool and does not fit beside these weights.
- Accuracy: GSM8K, 1,000 questions through the chat endpoint with thinking off, seed 0
  (`scripts/run_lmeval_gate.py --chat --chat-template-kwargs '{"enable_thinking": false}' --limit 1000`), scores
  0.829 strict / 0.849 flexible with the FP8 cache against 0.845 / 0.859 with the BF16 cache, standard error 0.012:
  the same within noise. With thinking on, the gate's 256-token budget is spent on hidden reasoning and the score
  reads 0.395 on either cache, which is the protocol, not the model.

## How the configuration was chosen

Six deploys, each reading the engine's own pool line:

| Cache | Utilization | Prefill chunk | Attention backend | Pool | Outcome |
| --- | ---: | ---: | --- | ---: | --- |
| BF16 | 0.92 | 8,192 | FlashAttention 2 | 3.99 GiB | 262,144 refused; engine estimates 61,936 max |
| FP8 | 0.95 | 8,192 | FlashInfer | 3.12 GiB | 262,144 refused; 95,648 max. FlashInfer costs 1.4 GiB more workspace and 0.54 GiB of CUDA graphs than FlashAttention 2 |
| FP8 | 0.97 | 4,096 | FlashInfer | 3.83 GiB | 131,072 refused; 117,600 max |
| FP8 | 0.97 | 4,096 | FlashInfer | 3.83 GiB | **114,688 serves**: the recipe |
| BF16 | 0.97 | 4,096 | FlashAttention 2 | 4.23 GiB | 81,920 refused; 65,856 max |
| BF16 | 0.97 | 4,096 | FlashAttention 2 | 4.23 GiB | 65,536 serves: the comparison lane |

The FP8 cache buys 112K instead of 64K of context and 1.8 times the pool in tokens, costs 5 to 10% of
single-request prefill speed (FlashInfer against FlashAttention 2: 16K in 1.67 s against 1.63 s, 64K in 10.78 s
against 9.62 s), nothing in decode, and nothing measurable on GSM8K. Measured at 16 concurrent, the BF16 cache
matches FP8 at 1K prompts, loses 29% of throughput at 4K and 16K and takes the 4K median first token from 2.2 s to
13.9 s, because its pool holds ten requests instead of thirteen. The 4,096-token prefill chunk is kept over 8,192:
the larger chunk crashed the engine under load at this utilization (FlashInfer's FP4 GEMM asked for a 532 MiB
MLP intermediate with 357 MiB free, failing half of a row's requests), and where it survived it cost 11% of
throughput and 43% of the median first token at 4K prompts, with no gain at 16K. Cap 16 keeps every slot resident
at the production prompt sizes; 32 only adds queueing
(32 x 210 MiB against 3.8 GiB) and 64 did not boot. The Gated DeltaNet state stays float32 as the checkpoint
config asks; halving it with `--mamba-ssm-cache-dtype bfloat16` would fit about a third more requests but was not
tested, because no accuracy gate for the recurrent state's precision was run.

## Emmy

The recipe stays `compiler-qualified`: the golden under `golden/` and the serving twins in `serving-rtx5090.env`
are unchanged by this work, and `DISCOVERY.md` records Emmy's own serving measurements. The engine block in
`recipe.yaml` is the stock vLLM lane measured here, which is what the recipe deploys.

## Reproduce

```bash
emmy deploy ssh --recipe recipes/Qwen3.8-27B-NVFP4 --ssh USER@HOST:PORT
emmy bench experiments/Qwen3.8-27B-NVFP4/serving --ssh USER@HOST:PORT     # the full grid, about three hours
scripts/run_lmeval_gate.py --base-url http://HOST:8000/v1 --model Inferact/Qwen3.8-27B-NVFP4 --chat \
  --chat-template-kwargs '{"enable_thinking": false}' --limit 1000
```

## Limits

- One run per row; treat differences under about 5% as noise. The V100 figures come from that lane's report, not
  from a same-run comparison, and the two lanes' bench clients are different vLLM builds.
- The client ran on the node; a real client adds its network latency to every time to first token.
- Cap 64 does not boot at 0.97 utilization: after the pool is allocated, FlashInfer's warm-up asks for a 128 MiB
  workspace with 69 MiB free. It is not a useful cap on this card anyway (64 slots of recurrent state are 9.4 GiB).
- The prompts are random tokens with no shared prefix, so prefix caching is off and unmeasured; a deployment with
  repeated system prompts may turn it on at the cost of the cache's page alignment of the Gated DeltaNet state.
- GSM8K bounds the FP8 cache's cost to about 3 points at 95% confidence; long-context quality is checked only by
  the passphrase retrievals.
