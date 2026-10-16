# OLMoE-1B-7B-0125-Instruct routed expert histogram on one RTX 5090

## Result

OLMoE's routing is only mildly skewed once workloads mix, and its hot experts depend on the workload. Over all three
workloads, a layer needs a median of 19 of its 64 experts for 50% of routed rows, 40 for 80% and 56 for 95%; with no
skew the same coverage would take 32, 51 and 61. Within one workload the skew is stronger: code needs 23 experts for
80%, math 36.5 and chat 43. But a hot set chosen on one workload barely helps another. The 24 hottest experts of the
chat window cover only 33.6% of code's rows, below the 37.5% that 24 random experts would cover; chosen on code they
cover 38.8% of chat's.

For the host-memory plan this means a static hot set (strategy F) chosen at boot from a different workload is worth
little on this model, and a GPU cache that follows the current workload, or router prediction, carries the win. On the
5090 the question is moot for OLMoE itself: all 16 layers' experts take 12 GiB in FP16 (12 MiB per expert) and fit
whole. Keeping the 40 hottest experts per layer, enough for 80% of the mixed rows, would take 7.5 GiB.

| Hot experts per layer | Chat | Math | Code | All three | Chat set on code | Code set on chat | No skew |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 20.7% | 29.9% | 47.2% | 27.7% | 11.3% | 12.9% | 12.5% |
| 16 | 37.5% | 47.8% | 67.0% | 45.0% | 23.4% | 26.3% | 25.0% |
| 24 | 52.2% | 62.6% | 78.9% | 58.9% | 33.6% | 38.8% | 37.5% |
| 32 | 65.2% | 74.6% | 86.9% | 70.5% | 47.1% | 50.8% | 50.0% |
| 48 | 86.4% | 92.1% | 96.5% | 88.5% | 77.3% | 75.6% | 75.0% |

Each cell is the share of a window's routed rows that land on the given number of experts, averaged over the 16 layers.
The first four columns pick each layer's hot set from the same window they score, so they flatter the skew; the two
cross columns pick on one window and score on another, which is what a boot-time hot set faces.

## Experts needed per layer

| Window | Rows per layer | 50% (min / median / max) | 80% | 95% |
| --- | ---: | --- | --- | --- |
| Chat | 8,635 | 21 / 23 / 26 | 41 / 43 / 46 | 55 / 57 / 59 |
| Math | 6,549 | 13 / 17.5 / 22 | 30 / 36.5 / 43 | 46 / 52 / 57 |
| Code | 7,136 | 7 / 8.5 / 17 | 20 / 23 / 38 | 37 / 42.5 / 54 |
| All three | 22,320 | 16 / 19 / 23 | 36 / 40 / 45 | 53 / 56 / 58 |

Per layer over all three windows, the experts needed for 80% run 44, 45, 42, 39, 40, 38, 36, 36, 38, 39, 41, 40, 40,
40, 41, 43 from layer 0 to 15: the middle layers are the most skewed, the first and last the least. The single
hottest expert of a layer takes a median 41% of its tokens (each token picks 8 of 64, so with no skew it would be
12.5%), and the coldest a median 3.3%. No expert went unused in any layer.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/routing_histogram_rtx5090 --local` served
`allenai/OLMoE-1B-7B-0125-Instruct` at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's
vLLM 0.23 plugin (the generate runner), from the image built at commit `53bf508d8` (image id
`sha256:0cd07a18640b…`, in `image.json`). Every program deployed from the recipe's RTX 5090 serving golden under
`EMMY_STRICT_EVIDENCE=1`, so nothing ran on a predicted schedule. The server ran one sequence at a time with a
512-token prefill chunk and a 16-row decode bucket; the recipe holds the full command.

The client sent the first 32 rows of `HuggingFaceH4/ultrachat_200k` (chat), `openai/gsm8k` test (math) and
`openai/openai_humaneval` (code), sequentially, greedily, up to 128 output tokens each, between two one-word probes.
The server logged the cumulative per-layer, per-expert selection counter before each uncaptured forward, 106
snapshots in all; a window is the difference between the snapshots at its first request and at the next window's
first request (or the final probe). Each window's per-layer total is the same in all 16 layers, and its row count
matches the requests' prompt and completion tokens within 0.3% (the counter also sees a few tokens the usage omits,
so the boundaries were taken as the nearest snapshot; requests are at least 150 tokens apart, so the choice is
unambiguous). About half the rows are prefill and half decode; the counter does not separate them.

All 98 requests returned HTTP 200 with content: 11,156 completion tokens, finishing by length or stop. A spot check
of answers (a GSM8K solution, chat replies) read as sensible; this is not a quality evaluation.

## Limitations

- 32 prompts per workload at 128 output tokens: enough for a coverage curve, not for the tail of rarely hit experts.
- Frequency only. The counter records how often each expert is hit, not in what order, so it cannot give a cache hit
  rate or how often a hot set must change; milestone 3 measures those.
- One model. OLMoE has 64 experts with top-8 routing; Qwen3.8-Flash-Next has 512 with top-10, and its skew must be
  measured on its own.

## Run and platform

Run `20261016T000622Z`, started `2026-10-16T00:06:22Z`, one row (`rtx5090x1`), status succeeded in 186 s. One RTX
5090 (32 GB, driver 580.178.04) on Ubuntu 24.04 with kernel 7.0.0, shared with a desktop session that held about 5 GB
of the card; the server used a 0.75 memory fraction. Earlier attempts of this recipe failed during server boot (an
out-of-memory kill, the wrong vLLM version, and a boot that never finished because the golden lacked the serving
programs) and are not retained.

Archive: `results_rtx5090x1.tar.gz` (Git LFS), root `2026-10-16_00-06-22/`, holding the experiment record, the
benchmark logs, and per-row `image.json`, `request_summary.json`, `responses.jsonl`, `server.log` and
`workloads.json`. Node addresses in the raw files read `node`.
