# Llama 3.1 8B Instruct with a selectable LoRA on V100

## Question and setup

Can one 32 GB V100 serve the pinned base checkpoint and a named LoRA adapter through the same vLLM endpoint, and
what does selecting the adapter cost? The base is the public `NousResearch/Meta-Llama-3.1-8B-Instruct` mirror at
`d10aef7999a2b5ba950ab3974312feeedbfe0b77`; its config and four weight shards were previously checked as
byte-identical to Meta's gated Llama 3.1 8B Instruct revision. The test adapter is
`t83714/llama-3.1-8b-instruct-limo-lora-adapter` at `cfccf812259ae6131253b623aaa386577c7fc791`, rank 8.

The server used one Tesla V100-SXM3-32GB (SM70), driver 580.178.04, FP16 weights, 65,536-token configured context,
eight maximum sequences, and the pinned `cloudriftai/1cat-vllm-sm70` image at
`sha256:6f34e0b247a78ca65f88f305b1f1cc52c9020ecb83a5ca21df0599676dc443d3`. The image's Volta attention
backend ran with `VLLM_FLASH_V100_DISABLE_PAGED_PREFILL=1` and prefix caching disabled. The server advertised both
the base model and `limo` in one `/v1/models` response. Both names passed the `2 + 2` chat smoke test.

## Protocol and results

Emmy deployed a fresh server for each row and ran `vllm bench serve` against it. Prompts had 512 random input tokens,
greedy decoding, two warmup requests, and a forced output length. Concurrency 1 used 16 prompts of 128 output tokens
in three repeats; concurrency 8 used 40 prompts of 256 output tokens in one repeat. Every row loaded the adapter,
then selected either the base model or `limo` by request name. The same image and serving settings were used throughout.

| Request | Concurrency | Success | Output tok/s | Mean TTFT | Mean TPOT |
| --- | ---: | ---: | ---: | ---: | ---: |
| Base | 1 | 48/48 | 42.78 (42.76–42.80) | 122.8 ms | 22.59 ms |
| `limo` | 1 | 48/48 | 20.06 (20.06–20.07) | 189.7 ms | 48.74 ms |
| Base | 8 | 40/40 | 236.38 | 1197.3 ms | 29.27 ms |
| `limo` | 8 | 40/40 | 126.05 | 1682.0 ms | 57.11 ms |

The adapter delivered 47% of base output throughput at concurrency 1 and 53% at concurrency 8. Its mean TPOT was
2.16× and 1.95× the base value, respectively. The three concurrency-1 repeats varied by at most 0.04 output tok/s
for base and 0.01 for the adapter. The concurrency-8 rows have one repeat each, so they do not establish run-to-run
variance. These figures compare request names on a server with the adapter loaded.

A separate run without an adapter used the same pinned image and V100 attention setting. It averaged 237.01 base
output tokens/s at concurrency 8, close to 236.38 on the server with the adapter loaded. It used 32 rather than 40
requests per repeat, so the comparison is directional. Load and warmup took about 137 seconds with the adapter loaded
versus 88 seconds without it; most of that difference was CUDA graph capture (40 versus 2 seconds in the raw timings).

## V100 attention failure and correction

An initial run on the same public image tag, without the paged-prefill setting, succeeded at concurrency 1 but
completed only 1/40 requests in each concurrency-8 row. Its server logs reported `RuntimeError: Shared memory exceeds
96KB: 114176 bytes` from the direct paged prefill kernel. Emmy marked those rows succeeded because the benchmark
client returned exit code zero even though 39 requests failed. The corrected run above disabled that kernel path and
all 176 measured requests succeeded. The archived `diagnostics/` directory retains the two failed client logs, their
server logs, and the initial recipe. Read client success and failure counts as well as Emmy's row status.

## Evidence and limits

Run time: 2026-10-03 18:22:46 UTC; run ID `20261003T182246Z`. All four system-only experiment records ended in
`succeeded`. Docker Engine was 29.8.1 on Ubuntu 24.04.1; the raw records carry the rest of the machine inventory.
The Git LFS archive `results_v100x1.tar.gz` contains the dated raw run, including:

- `v100x1_rnN-M-L-3.1-8B-I_7043c7f7d3c9.experiment.yaml` and its benchmark and server logs;
- `v100x1_rnlimo_07eef6eccf5f.experiment.yaml` and its benchmark and server logs;
- `v100x1_mc8_np40_rol256_r1_rnN-M-L-3.1-8B-I_c4cd74020277.experiment.yaml` and its logs;
- `v100x1_mc8_np40_rol256_r1_rnlimo_25237df6ee6a.experiment.yaml` and its logs;
- `benchmark.log`, `benchmark_v100_x_1.log`, and `diagnostics/` with the initial failed concurrency-8 evidence.

The rows were run sequentially on fresh servers, so this is a comparison of selectable request names rather than a
mixed base-and-adapter traffic test. The smoke test checks a simple answer; it does not establish adapter task quality,
tool-call behavior, or 65,536-token request success. Emmy compiler kernels were not used by this serving image.
