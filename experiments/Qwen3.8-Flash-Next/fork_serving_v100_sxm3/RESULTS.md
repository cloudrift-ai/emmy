# Qwen3.8-Flash-Next on the 1Cat fork with baked kernel caches, V100 SXM3: no compile at boot or first request

**Question.** With the fork's JIT caches baked into CloudRift's image, does a fresh container boot and serve every
request path without compiling a kernel, at the speed `fork_image_v100_sxm3` measured without them?

**Image.** `cloudriftai/1cat-vllm-qwen3.8-flash-next-nvfp4:1.5.1-248d877c5` (ID `sha256:bf976828fc00…`, 33.6 GB): the
image `fork_image_v100_sxm3` measured (ID `sha256:09bf5dfc0b46…`, 31.8 GB, tagged as the bake's base
`cloudriftai/1cat-vllm-qwen3.8-flash-next-nvfp4-base:1.5.1-248d877c5`) plus one layer from
`docker/1cat-vllm-sm70/Dockerfile.triton-cache`: 15,357 cache files, 1.5 GB (vLLM's compile cache with
TorchInductor's, Triton, TileLang, FlashInfer), manifest digest `241aff6dc3f4…`. The weights are not in the image;
deploy downloads them into the host's model cache.

## How the caches were made

A cold boot of the base image with this recipe's flags, then a warm workload against it: one request at 1K, 8K, 32K
and 130K tokens; 2 to 8 concurrent requests at varied lengths around 4K, so every decode batch size up to the
eight-sequence cap meets prefill; prefix-cache hits; greedy, the checkpoint's own sampling defaults and explicit
top-p/top-k; tool calls with `tool_choice` auto, required and named, a JSON-schema response and logprobs. The boot
compiled the graphs, and 10 Triton kernels compiled at request time (sparse attention and the router's top-k). A second
workload with other seeds and lengths in the same container added no file, so one pass covered the shapes.

Two findings shaped the bake:

- **Triton's folder must be fixed.** While `TRITON_CACHE_DIR` is unset, TorchInductor points Triton at its own folder
  when its first kernel loads, so where a kernel lands depends on what the boot compiled before it. A boot from the
  cache loads one of its two graphs whole and re-traces the other (vLLM cannot save that one: "cannot pickle 'module'
  object"), so it looked in other folders than the cold boot wrote and compiled the fork's kernels again. The image
  fixes `TRITON_CACHE_DIR=/root/.triton/cache`, and the bake merges every Triton entry there (each is named by its
  content hash). TorchInductor's own cache keys did not change with it.
- **An offline boot changes vLLM's compile key.** With `HF_HUB_OFFLINE=1`, vLLM resolves the model to its local
  snapshot path, and the path is part of the key, so every baked graph missed (12,151 new files at boot). Online, the
  logged engine config holds the repo id and no host path, so one image should hit from any host's model folder; a
  second folder was not tried. The image is therefore warmed and served online; deploy has the weights on the host
  before the server starts.

## v100x4 (4x NVIDIA Tesla V100 SXM3 32GB, cards 0-3)

**Zero-recompile check.** A fresh container from the baked image, through `emmy deploy` with this recipe, then a fifth
workload of the same kinds with new seeds (14 benchmark runs, 5 chat requests, all succeeded). `docker diff` lists
every file a container writes outside its image: zero new files under `/root/.cache`, `/root/.triton` and
`/root/.tilelang` after the boot, and zero after the workload. The only files written there are one empty
`model.<pid>.tmp` per worker, left by the graph vLLM cannot save, which holds no code. vLLM's JIT monitor still warns
10 times, on each kernel's first launch in the process, as it does for a kernel loaded from the cache.

**No weights in the caches.** The largest cached file is 10.6 MB. A search of all cached files for 32-byte windows of
seven unquantized checkpoint tensors (the router, a norm, hyper-connection and shared-expert-gate weights), each as
stored, as FP16 and as FP32, found none.

**Protocol.** `emmy bench` with the recipe in this directory: one deployment, one request at a time, 8,192 random
input tokens and 512 output, greedy, `ignore_eos`, four repeats with fresh prompts (seed 380 + repeat), no warm-up
request. It is `fork_image_v100_sxm3`'s 8K row, so the two compare directly.

| | This image | Without baked caches (`fork_image_v100_sxm3`, 8K row) |
| --- | ---: | ---: |
| Deployment, start to healthy | 303 s | 553 s |
| Repeat 1, mean time to first token | 1.58 s | 2.09 s |
| Repeats 2-4, mean time to first token | 1.54-1.57 s | 1.56-1.57 s |
| Repeats 2-4, per output token | 12.9-13.0 ms | 12.9-13.0 ms |
| Repeats 2-4, output throughput | 62.2-62.9 tok/s | 62.4-62.7 tok/s |

**Conclusion.** The baked image boots in about five minutes instead of nine and compiles nothing, at boot or on any
request path the workloads reached. The first request is no longer slower than the rest, and steady-state speed is
unchanged.

**Limitations.** The caches hold for this recipe's shape only: TP4, one data-parallel rank, these flags, on a V100
SXM3. A different flag, or `emmy deploy` on a 16-card host without `--gpu-count 4` (which runs four data-parallel
copies in one container), compiles anew. The boot is online and was not checked with Hugging Face unreachable. Of the
remaining five minutes, loading the weights takes about 2.4; the graph vLLM cannot save is re-traced on every boot (15
s of Dynamo tracing) and its compiled code then loads from the cache. The warm and zero-recompile workloads ran from
scripts on the test node and are not in the archive; this record carries their method and counts. The image is
published to a private Docker Hub repository (2026-10-16, through `emmy publish`, registry digest
`sha256:bf976828fc00…`, equal to the local image ID). In the archived raw files the test node's address is replaced
with `test-node` and the local run directory with its repository path; nothing else in them was changed.

**Run.** 2026-10-16T18:34:37Z, run ID `20261016T183437Z`, Emmy revision `03ebe3c05`, 1 of 1 rows `succeeded`, no
failed requests. Host: the same VM as `fork_image_v100_sxm3`, 16x Tesla V100-SXM3-32GB, driver 580.178.04, kernel
6.8.0-139.

**Archive.** `results_v100x4.tar.gz`, root `2026-10-16_18-34-37/`: the `v100x4_mc1_np4_ril8192_rol512` record, its
`*.benchmark.log` and `*.server.log`, and the run logs.
