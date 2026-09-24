# Gemma 4 12B llama.cpp serving lane — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

Does the llama.cpp column of the serving tables in "Outperforming vLLM and Llama.cpp on Gemma4-12B" (published
2026-08-01) still hold, and is stock vLLM still the stronger baseline? The article includes llama.cpp as a sanity
check that the vLLM baseline is not underperforming.

### Status

The one row succeeded (run `2026-09-19_07-28-37`, run ID `20260919T072837Z`). Five of its six points completed
every request. The 256/256 c=64 point failed as the recipe expects: llama-server could not allocate its 15 GiB KV
cache for 64 slots beside the FP16 weights (`cudaMalloc failed: out of memory`), which is the article's footnote 1.

Two earlier attempts the same morning died about two and a half minutes in, just after the GGUF conversion. The
server cleanup `pkill -f "[l]lama-server"` matched the lane's own ssh command line, which carries the whole script.
The recipe now matches the process name exactly (`pkill -x llama-server`). The 2026-08-06 attempt lost its results
the same way. This run also added the article's 8192/256 point, which the recipe did not run before.

### Protocol

`emmy bench experiments/gemma-4-12B/serving_llamacpp_rtx5090 --local` on a pre-allocated RTX 5090. The lane builds
llama.cpp from the head of its repository (build `b23701f`; the article used `0a50d99`), converts the checkpoint to
an F16 GGUF, and serves it with `llama-server -ngl 99 -fa on`. The single-stream points use one slot of 16384
tokens. The batched 4K/4K points and the 8192/256 point share one server with 8 slots of 8448 tokens, run in that
order. The client is `vllm bench serve` from the repository venv (vLLM 0.29.0) with random prompts, seed 0,
temperature 0 and `--ignore-eos`, the same flags as the vLLM lanes. Every point runs once.

### Measurements

| Tokens in | Tokens out | Concurrency | Prompts | Output tok/s | Median TTFT (ms) | Median TPOT (ms) | P99 TPOT (ms) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 256 | 1 | 16 | 59.31 | 173.2 | 16.24 | 16.36 |
| 4096 | 4096 | 1 | 8 | 58.94 | 1152.9 | 16.53 | 17.06 |
| 4096 | 4096 | 4 | 32 | 175.57 | 2938.8 | 20.65 | 111.86 |
| 4096 | 4096 | 8 | 64 | 271.53 | 2542.6 | 25.96 | 96.87 |
| 8192 | 256 | 4 | 16 | 68.38 | 3695.8 | 45.26 | 55.55 |
| 256 | 256 | 64 | 256 | out of memory | | | |

### Comparison with the article and with stock vLLM

Stock vLLM is the `cgo-2027/gemma4_serving` run of the same morning on the same box
(`2026-09-19_06-15-04`), which also reproduces the article's stock column to within 0.7%.

| Point | Article tok/s | This run | Stock vLLM | Article TTFT / TPOT (ms) | This run | Stock vLLM |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096/4096 c=1 | 56.4 | 58.94 | 57.23 | 1144 / 17.4 | 1153 / 16.5 | 566 / 17.3 |
| 4096/4096 c=4 | 153.1 | 175.57 | 216.52 | 2570 / 21.1 | 2939 / 20.7 | 1085 / 18.2 |
| 4096/4096 c=8 | 294.0 | 271.53 | 384.21 | 2653 / 26.1 | 2543 / 26.0 | 1099 / 20.5 |
| 8192/256 c=4 | 80.2 | 68.38 | 112.62 | 3655 / 35.2 | 3696 / 45.3 | 2028 / 27.3 |
| 256/256 c=64 | out of memory | out of memory | 1434.07 | | | |

### What the numbers say

The article's conclusion holds: stock vLLM is not the weak baseline. It leads llama.cpp by 1.23x at 4K/4K c=4,
1.41x at c=8 and 1.65x at 8192/256. It prefills a 4096-token prompt in half the time (566 against 1153 ms), and
llama.cpp still cannot fit the 64-slot point at all.

One cell flipped. At single-stream 4K/4K decode, llama.cpp is now 3% ahead of stock vLLM (58.94 against 57.23
tok/s, TPOT 16.5 against 17.3 ms), where the article had it 1.4% behind. Stock vLLM did not move, so the gain is
llama.cpp's own: its median TPOT dropped from 17.4 to 16.5 ms between builds `0a50d99` and `b23701f`. At 256/256 c=1
the two decode at the same speed (TPOT 16.2 against 16.3 ms), and vLLM keeps a 3x lead on first-token latency
(56 against 173 ms).

The batched points moved in both directions against the article: 15% more throughput at c=4 and 8% less at c=8,
with median TPOT within 0.5 ms of the article at both. Their P99 TPOT is 97 to 112 ms, four to five times the
median, the mark of decode steps that wait behind another slot's 4096-token prefill. How prefills and decodes
interleave is what sets these points' throughput, and the build can change it. The 8192/256 point is the weakest:
68.4 tok/s against the article's 80.2 and 86.3 in a hand run of 2026-08-06 on build `3db4ff8`. Here it runs on a
server that has just served the 4K/4K points; that hand run started a fresh one, so both the build and the server
state differ. Read the batched cells as directional.

### Limitations

- Every point runs once, so the size of the c=4, c=8 and 8192/256 shifts has no repeat to check against.
- The llama.cpp build is whatever the repository head was on 2026-09-19. The recipe does not pin it, so a rerun
  measures a different engine.
- The comparison with stock vLLM spans two runs on the same box on the same morning. It is direct in protocol but
  not interleaved. Stock vLLM's 4K/4K throughput agreed to within 0.3% across three runs a month apart, so the
  pairing is sound there; its 8192/256 latencies move between runs.
- Running `--local` points the lane at the repository venv. Its `pip install` added `gguf` 0.19.0 there, and the
  client is the venv's vLLM 0.29.0 while the vLLM lanes ship their own.

### System

| Item | Value |
| --- | --- |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic |
| Engine | llama.cpp `0.4.1-dev (build 1, commit b23701f)`, CUDA 13.0, `-ngl 99 -fa on` |
| Model | `google/gemma-4-12B-it`, converted to F16 GGUF (23 GB) by the build's `convert_hf_to_gguf.py` |
| Client | `vllm bench serve`, vLLM 0.29.0 |
| Harness revision | `12e05ed6a`, clean tree |
| Run window | 2026-09-19 07:28 to 08:20 UTC |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS) holds `2026-09-19_07-28-37/` as `emmy bench` wrote it, less the lane's working
folder (the llama.cpp source tree and the 23 GB GGUF, both rebuilt by every run). Its members are `benchmark.log`,
`benchmark_rtx5090_x_1.log`, the row record `rtx5090x1_995e0128f901.experiment.yaml`, and the row's collected files
`rtx5090x1_995e0128f901_<name>`: the point results `small_c1.txt`, `head_c1.txt`, `head_c4.txt`, `head_c8.txt` and
`rag_c4.txt`, the server logs `serve_p1.log`, `serve_p8.log` and `serve_p64.log` (the out-of-memory failure), and
`build.txt`, `cmake.log`, `convert.log` and `snap.txt`.
