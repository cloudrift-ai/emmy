# Voxtral Mini 3B attention path on one 32 GB V100

## v100x1 — Tesla V100-SXM3-32GB, 2026-10-08

### Question and protocol

Which attention configuration should the Voxtral V100 serving lane use? The first serving run used
`VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` with `FLASH_ATTN_V100`, the gather path the Llama 3.1 8B V100 recipe needs to
stay under Volta's shared-memory limit. Its concurrent rows were far slower than its single-stream row, so two
alternatives ran the long-prompt row: 32 requests of 8,192 random input tokens and 256 forced output tokens at 16
streams, seed 0, greedy decoding, one warmup. Same host, model revision, image and engine flags as the serving lane.

### Results

| Attention configuration | Requests | Total tok/s | Mean TTFT | Mean TPOT | Failed |
| --- | ---: | ---: | ---: | ---: | ---: |
| `FLASH_ATTN_V100`, gather path (80 requests, serving run) | 80 | 303.9 | 81.4 s | 1,418 ms | 0 |
| `FLASH_ATTN_V100`, paged path | 32 | 2,254.8 | 13.2 s | 174 ms | 0 |
| `TRITON_ATTN` | 32 | 955.4 | 35.7 s | 399 ms | 0 |

The paged `FLASH_ATTN_V100` path is 7.4x the gather path and 2.4x Triton, with no failed request and no
shared-memory error in its server log. The gather path also stalled the 64-stream 1024/1024 row near 6 generated
tok/s before it was stopped. The serving lane therefore drops `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL`. The gather
row ran 80 requests rather than 32, so its comparison is directional; the gap is far larger than that difference.

### Archive

`results_v100x1.tar.gz` holds `2026-10-07_22-54-44/` (the gather-path serving run: the 4096/4096 rows at 1 and 16
streams and the 8192/256 row finished; the remaining rows were stopped and their records still read `running`) and
`2026-10-08_01-00-16/` (the paged and Triton rows; the gather row of this recipe was not rerun).
