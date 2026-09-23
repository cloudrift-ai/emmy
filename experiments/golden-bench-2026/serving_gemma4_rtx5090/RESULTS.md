# Gemma 4 12B, one published image, stock vLLM against Emmy — results

## NVIDIA GeForce RTX 5090 x1 (`rtx5090x1`)

### Question

On one immutable image, serving one checkpoint at one set of scheduler settings, what does routing the model through
Emmy's compiled kernels do to end-to-end serving against vLLM's own path? This is the suite's matched-system
comparison: the two arms differ in the model implementation and in nothing else the harness controls.

### Status

All 24 rows succeeded (4 points x 2 arms x 3 repeats), one run ID (`20260923T155835Z`, directory
`2026-09-23_15-58-35`). Every request of every repeat completed and no row logged an engine fatal.

This is the recipe's first run. Its stock arm could not have run before: Gemma 4 12B is a multimodal checkpoint, vLLM
sizes an encoder budget from it, and a stock server refuses to start whenever `--max-num-batched-tokens` is below
`max_tokens_per_mm_item` (2496 here) — which the 64-stream and 8-stream points pin at 2112 and 2056. The Emmy arm
never hit it because `EmmyGenModel` is text-only. Both arms now declare no image, video or audio items, which costs a
text benchmark nothing and keeps their argv identical.

### Protocol

`emmy bench experiments/golden-bench-2026/serving_gemma4_rtx5090 --local` on a pre-allocated RTX 5090, one freshly
deployed server per row. Both arms run
`cloudriftai/vllm-emmy-gemma-4-12b-it@sha256:3a690e9f7859d46b969dd9eaaed36f52f92c25c5595dc112aee2adb781d26e28`, the
per-model image published from this branch, which carries the model snapshot, the warmed sm_120 cubins and the
execution-plan packs. The stock arm overrides the image's entrypoint with `python3 -m
vllm.entrypoints.openai.api_server`; the Emmy arm selects `EmmyGenModel` through `--hf-overrides`. Every row runs
`--dtype float16 --no-enable-prefix-caching`, context 16384, `--gpu-memory-utilization 0.96`, and the point's own
`--max-num-batched-tokens`. The Emmy arm additionally sets the decode bucket (and, on the mixed points, the prefill
chunk) the golden records rows for. The client is `vllm bench serve` on random prompts with seed 0, temperature 0 and
`--ignore-eos`. Within each point the arms alternate which one boots first across the three repeats, so wall-clock
position and thermal drift fall on both.

The Emmy arm serves the standard lane — FP32 accumulation. There is no fast-math arm here by design; the precision
fork is the `gemma4_serving` recipe's subject.

### Measurements

Mean of three repeats. Throughput is output tokens per second; latencies in milliseconds.

| Point | Arm | Output tok/s | Median TTFT | Mean TTFT | Median TPOT | Median ITL |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 4096/4096 c=1 | stock | 57.4 | 549 | 557 | 17.30 | 17.31 |
| 4096/4096 c=1 | emmy | 53.6 | 590 | 616 | 18.52 | 18.53 |
| 4096/4096 c=8 | stock | 384.7 | 1106 | 1252 | 20.53 | 19.78 |
| 4096/4096 c=8 | emmy | 381.2 | 1194 | 1418 | 20.68 | 19.87 |
| 8192/256 c=4 | stock | 112.8 | 2257 | 2496 | 26.62 | 18.20 |
| 8192/256 c=4 | emmy | 105.7 | 2612 | 2825 | 27.11 | 18.78 |
| 256/256 c=64 | stock | 1301.4 | 1796 | 3497 | 27.10 | 21.73 |
| 256/256 c=64 | emmy | 1303.9 | 1701 | 2147 | 31.63 | 23.39 |

### Repeat variation

Throughput across the three repeats of a cell spans 0.0 to 1.5 tok/s — under 0.2% everywhere, and under 0.1% at the
three lower-concurrency points. The 64-stream point is the widest (1.5 tok/s of 1300) and is also the only point whose
mean and median first-token latencies disagree sharply, in both arms. Differences below about 1% in the table are
inside that spread and should not be read as an ordering.

### What the numbers say

**Throughput is level at high concurrency and behind at low.** At 64 streams the two arms are within 0.2% of each
other, which the repeat spread cannot separate. At 8 streams Emmy is 0.9% behind. The two points with a single long
stream or few of them are where the gap is real: 6.6% behind at 4096/4096 c=1 and 6.3% behind on the long-prompt RAG
point.

**Per-token latency tracks that.** Emmy's median TPOT is 7% above stock's at the single-stream point, 0.7% at c=8, and
1.8% on the RAG point. At 64 streams it is 17% above — the one place the two arms separate clearly on per-token cost,
and the point where Emmy nonetheless matches stock's throughput, because its first-token latency is better there.

**First-token latency is the one place Emmy wins, and only under load.** At 64 streams Emmy's median TTFT is 1701 ms
against stock's 1796, and its mean is 2147 against 3497 — a 39% lower mean. A mean far above the median in both arms
means a queue: with 256 prompts at 64 concurrency most requests wait. Emmy's tail is much shorter. At every other
point its first-token latency is 7 to 16% worse than stock's.

**Nothing here is a compiler-kernel isolation.** Stock runs vLLM's native Gemma implementation and Emmy runs
`EmmyGenModel`; the two differ in attention backend, in how the layers are dispatched and in what is captured in a
cudagraph, not only in the GEMM kernels. The delta is a system-level result for this image.

### Limitations

- One card, one host, three repeats per cell in a single session. Not a designed repeat protocol across days.
- The image was built at `806fef9a9`, before this branch was rebased onto `main`. It therefore predates four upstream
  commits, one of which (#877) changes how a gated MLP stages its operands — the same lift this branch had made for
  itself and dropped as superseded. The A/B is internally consistent, because both arms come from that one image, but
  it measures that image and not the branch's final compiler.
- The standard lane only. The `gemma4_serving` recipe measures the fast-math fork, where the first-token latencies
  are much lower; nothing here speaks to it.
- `vllm bench serve` with `--ignore-eos` does not compare the two arms' output text. Semantic equivalence is not
  established by this run; it is checked separately by the compiler's own correctness gates.
- The 64-stream point queues 256 prompts at concurrency 64, so its first-token latencies are queueing latencies and
  are not comparable with a single-wave protocol.
- Two of vLLM's own Triton kernels JIT-compile on the first request of every boot in this image; they are written to
  no on-disk cache, so the warm cannot bake them. That is a one-off latency spike per server, present in both arms.

### System

| Item | Value |
| --- | --- |
| Run | `2026-09-23_15-58-35` (run ID `20260923T155835Z`), 24 rows, all `succeeded` |
| Source revision | `12d1a98604d5`, clean tree |
| Image | `cloudriftai/vllm-emmy-gemma-4-12b-it:0.23.0-806fef9a9@sha256:3a690e9f…d26e28`, vLLM build `91df0fad4dc9` |
| GPU | NVIDIA GeForce RTX 5090, 32607 MiB, driver 580.173.02 |
| CPU / memory | AMD Ryzen 9 9950X3D (16 cores, 32 threads), 64.9 GB |
| OS | Ubuntu 24.04.2 LTS, kernel 7.0.0-28-generic, Docker 29.5.0 |
| Toolchain | CUDA 13.0 (nvcc 13.0.88), cuBLAS 13.1.1.3 |
| Model | `google/gemma-4-12B-it` at `707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7`, FP16 |

### Archive

`results_rtx5090x1.tar.gz` (Git LFS), root member `2026-09-23_15-58-35/`: the run logs `benchmark.log` and
`benchmark_rtx5090_x_1.log`, and three files per row — `<variant>_<row id>.experiment.yaml`, `.benchmark.log` and
`.server.log`. Variants are `rtx5090x1_a{stock,emmy}_mc{1,4,8,64}_…_r{0,1,2}`.
