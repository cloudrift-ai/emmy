# OLMoE-1B-7B-0125-Instruct with a hot set of experts on the GPU and the rest in host memory, RTX 5090

## Result

A static hot set does not make host experts cheap. Decode time is the link time of the routed rows that miss the hot
set, and on OLMoE even a 3/4 hot set misses often enough to more than double TPOT. With the 48 most selected experts of
every layer in device memory (9 of 12 GiB) and the other 16 read in place over PCIe 5.0 x16, decode at one stream went
from 2.76 to 6.94 ms per token on chat and from 2.82 to 5.64 ms on code. With no expert on the GPU it is 35.4 ms. Every
row gave the same 16 greedy answers as the all-GPU row.

| GPU experts per layer | Expert bytes on GPU | Chat TPOT | Chat rows on GPU experts | Code TPOT | Code rows on GPU experts |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 64 (all) | 12 GiB | 2.76 ms | 100% | 2.82 ms | 100% |
| 48 | 9 GiB | 6.94 ms | 84.9% | 5.64 ms | 92.5% |
| 32 | 6 GiB | 15.56 ms | 61.6% | 9.37 ms | 80.7% |
| 16 | 3 GiB | 26.62 ms | 30.6% | 15.36 ms | 62.0% |
| 8 | 1.5 GiB | 31.92 ms | 13.9% | 21.02 ms | 43.5% |
| 0 | 0 | 35.42 ms | 0% | 35.51 ms | 0% |

TPOT is the median over 8 requests. "Rows on GPU experts" is the share of routed selections, prefill and decode
together, that landed on a device-resident expert, read from the routing histogram the server logged.

A simple model fits every row within 15%: TPOT ≈ 2.8 ms + (1 − GPU share) × 35.5 ms. 35.5 ms is the time for one
token's 128 expert reads (16 layers × top 8 × 12 MiB, 1.61 GB) at the 45.4 GB/s the copy engine measures on this
machine. The all-host row lands on that floor, so the GPU's own work hides under the link time. This is milestone 1's
result again, applied per expert: a host expert costs its bytes at the link rate on every hit, wherever it sits.

What this means for the plan:

- **The hot set has to cover almost every row.** To stay within 2× of all-GPU, the host share must stay under about
  8% of rows (0.08 × 35.5 ≈ 2.8 ms). On OLMoE that takes about 48 of 64 experts for code and more for chat. The memory
  saved is a quarter of the experts, not most of them.
- **Skew depends on the workload, as milestone 0 found.** The same hot set, chosen on mixed chat, math and code,
  covers code much better than chat at every size (80.7% against 61.6% at 32). Code routing is more concentrated.
- **Strategy F is the baseline to beat, and it is a weak one here.** A dynamic strategy that brings the host share of
  rows down at the same GPU budget wins directly in proportion. That means a GPU cache with router prediction
  (milestone 3), or running cold experts where their weights are (strategy G), so that their bytes never cross the
  link.
- **Bigger, more skewed models change the numbers, not the shape.** Qwen3.8-Flash-Next's NVFP4 experts are much
  smaller per expert, and its skew is not measured yet. The same model gives its TPOT from the host share of rows.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/hot_set_rtx5090 --ssh …` served `allenai/OLMoE-1B-7B-0125-Instruct`
at revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e` in FP16 through Emmy's vLLM 0.23 plugin (the generate runner).
The image was built from this branch's code on the VM (tag `0.23.0-23cf231ba`, image id `sha256:8b905931332a…`). The
serving shape and strict evidence from the recipe's RTX 5090 golden are those of zero_copy_experts_rtx5090. The six rows
are six server boots of the same image; only `EMMY_GEN_EXPERTS_GPU` differs.

`rank.json` is the last cumulative snapshot of routing_histogram_rtx5090, over the first 32 prompts of the chat, math
and code sets. Each layer keeps the experts with the most selections there. The benchmark prompts are the next 8 of
the chat (`HuggingFaceH4/ultrachat_200k`) and code (`openai/openai_humaneval`) sets, so the hot set is chosen on the
same kind of workload but not on the same prompts. Each workload ran through `vllm bench serve` with a custom dataset
at concurrency 1: 8 prompts, 64 output tokens, `--ignore-eos`, temperature 0, with the generated texts saved. The
texts of every row equal the all-GPU row's, all 16 of them.

The routing histogram logged before every prefill. A workload's window is the difference between the last snapshot
before its first request and the last snapshot before the next workload (or the run's last snapshot). The code window
therefore misses the last request's decode. The copy rate is five 256 MiB `copy_` calls from pinned memory, timed in
the same image on the same card.

## Limitations

- Decode at one stream only. At higher concurrency the rows of a step hit more experts, so more of them miss the hot
  set; the per-token cost should grow.
- The row shares count prefill and decode together. TPOT is decode only.
- The VM has 23 GB of RAM. A 32 GB swap file let the server load the weights, and the all-host row pinned 12 GiB; the
  pinned memory does not swap. The VM had no outbound DNS after boot, so the image build, the prompt download and the
  server reached the internet through an SSH-tunneled proxy. None of this touches the measured decode path.
- No nsys: bytes over PCIe are inferred from the routing and the expert size.

## Run and platform

Run `20261016T195653Z`, started `2026-10-16T19:56:53Z`, six rows, all succeeded. One RTX 5090 (32 GB, driver
580.159.03, PCIe 5.0 x16) in a CloudRift VM on a Ryzen 7 7800X3D with 7 cores and 23 GB of RAM, kernel 6.8.0. Archive
`results_rtx5090x1_x16.tar.gz` (Git LFS), root `2026-10-16_19-56-53/`, holds the experiment records, the benchmark
logs, and per row `bench_chat.json`, `bench_code.json`, `image.json` and `server.log`. Node addresses read `node`.
