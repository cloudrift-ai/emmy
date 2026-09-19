# Qwen3.8 GDN kernels on V100

## V100 SXM2 16 GB — 2026-09-19

Emmy's current decode is 3.88× slower than torch.compile at 12 value heads and 5.67× slower at 48. The new register
schedule in [PR #853](https://github.com/cloudrift-ai/emmy/pull/853) cannot run on V100: it uses the m16n8k16 MMA
fragment layout. Volta needs its own atom and matching register repacking. These measurements establish the current
supported decode performance and prefill gaps; they do not measure a speedup from the new register schedule.

### Protocol

The dimensions come from [Qwen3.8-27B's pinned configuration](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json):
128 key/value dimensions, 16 Q/K heads, and 48 value heads. Twelve value heads represent one TP4 rank; 48 represent
an entire layer. Each command runs on GPU 0 of the supplied four-V100 host. These are single-request kernel tests
with synthetic inputs, without tensor-parallel communication or model weights.

The direct comparison runs the Transformers recurrent GDN function for one token. It includes Q/K normalization,
grouped-head expansion, the state update, output calculation, and the final state output. Q/K/V are FP16, the state
and gates are FP32, and the initial state is nonzero. All three backends receive the same seeded inputs. Each fresh
process requests ten warmups and 100 measurements; the whole forward is captured in a CUDA graph for every backend.
Three processes cover each head count. Emmy uses its greedy standard-precision schedule at O3, with repository
goldens disabled and an empty task-local tuning database. This experiment does not tune schedules or test FAST_MATH.

### Direct decode comparison

Median microseconds across three fresh processes; ranges are across those process measurements.

| Value heads | Eager PyTorch | torch.compile | Emmy |
| --- | ---: | ---: | ---: |
| 12 (TP4 rank) | 70.90 (70.48–71.14) | 13.85 (13.73–13.85) | 53.73 (52.71–54.21) |
| 48 (full layer) | 100.18 (96.43–100.35) | 26.97 (26.28–27.14) | 152.92 (146.77–153.09) |

All six rows pass the CLI's strict eager-reference check at rtol=atol=0.001. Emmy's maximum absolute error is
3.05e-5 at 12 heads and 1.53e-5 at 48 heads. The torch.compile comparison uses fullgraph Inductor with max-autotune
and the same external CUDA-graph capture as Emmy and eager. Emmy beats eager at 12 heads but loses at 48; it loses
to torch.compile in every repeat at both head counts.

### Prefill comparison

The external implementation is [1Cat-vLLM at b711d530](https://github.com/1CatAI/1Cat-vLLM/tree/b711d5304525dfc0cca6bc8a0bb005f33fe1bbf8).
Its existing prefill benchmark compares its FLA/Triton implementation and FlashQLA in the same CUDA-event harness,
without CUDA-graph capture. Each row requests ten warmups and 100 iterations in one process. Inputs use normalized
FP16 Q/K, FP16 values, FP32 gates, and a nonzero FP32 initial state. Normalization is outside the timed operation.
The comparison below uses the ordinary log-gate FlashQLA path and ordinary FLA path, without selecting a faster
precomputed-gate or preallocated-output variant. The upstream sm70 Triton schedule defaults remain enabled.

| Value heads | Tokens | FLA/Triton, µs | FlashQLA, µs | Faster implementation |
| --- | ---: | ---: | ---: | --- |
| 12 | 128 | 780.22 | 130.70 | FlashQLA, 5.97× |
| 12 | 512 | 795.17 | 509.50 | FlashQLA, 1.56× |
| 12 | 2,048 | 1,674.76 | 2,581.34 | FLA/Triton, 1.54× |
| 48 | 128 | — | — | Process timed out; no result |
| 48 | 512 | 1,317.73 | 697.74 | FlashQLA, 1.89× |
| 48 | 2,048 | 4,872.44 | 2,767.94 | FlashQLA, 1.76× |

FlashQLA's advantage depends on shape. The 12-head result crosses over by 2,048 tokens, while FlashQLA still wins
at 48 heads. This is one process per shape, and uncaptured timing can include gaps between launches. It does not
establish the same ratios under graph capture or inside serving.

Across the five completed shapes, FlashQLA differs from FLA by at most 1.22e-4 in output and 7.23e-4 in final state.
The upstream program records these differences without a strict accuracy assertion. They are a numerical
comparison between implementations, not a model-level quality check.

Emmy's isolated chunk-state workload takes the chunk-local transforms as FP32 inputs and computes the inter-chunk
correction and state update. It has 64-token chunks, 128 total tokens, and 12 heads. The compiler recognizes its
carried loop, but the supported fallback fails the kernel watchdog: the first launch waits about 11.9 seconds,
then a later launch exceeds the two-second limit. No valid timing or accuracy result is produced. The full
Transformers prefill workload, also 12 heads and 128 tokens, produces no result within the 110-second process
limit. Neither failure is evidence about the new register schedule, which is unavailable on this GPU.

### External decode diagnostic

The upstream FlashQLA decode program reports synchronized host means of 87.75 µs at 12 heads and 74.89 µs at
48 heads. This operation includes mixed-QKV processing, gate calculation, normalization, and state cloning. It
does not use CUDA-graph capture, and it is a different operation from the direct decode comparison. Its timings
cannot be divided by the direct-comparison timings to claim a kernel speedup. Outputs are archived, but this
invocation does not establish correctness against an independent reference.

### Reproduction and evidence

Run `emmy bench experiments/Qwen3.8-27B/gdn_kernels --ssh USER@HOST` after preparing the environment and reference
checkout named by the recipe. The host used Python 3.12, PyTorch 2.14.0+cu126, Transformers 5.14.1, CUDA toolkit
12.9.86, and NVIDIA driver 580.178.04 on Ubuntu 24.04.1. CPU: Xeon E5-2680 v4, 24 exposed logical CPUs. The recipe
preloads CUDA 12 NVRTC because the environment also contains CUDA 13 NVRTC, which cannot target sm_70. Installed
Python packages are recorded for every row; the system record separately identifies the toolkit's cuBLAS.

The reference checkout uses its `VLLM_TARGET_DEVICE=empty` Python-only build plus the FlashQLA CUDA extension,
built against the active PyTorch and CUDA toolkit. Its source-build package version is
`0.1.dev1+gb711d5304.empty`, not a published release. The only source patch makes the CUDA platform's import of
unrelated serving custom ops optional. The GDN kernels and benchmark programs are unchanged. Every external row
archives the exact patch and source revision. Extra dependencies live separately; the supplied Emmy environment
and existing source tree were not overwritten.

The reference extension was built before the run with `MAX_JOBS=4`, targeting only sm_70, with
`--split-compile=8`; the archived Ninja file records all compiler flags. The recipe loads that binary through
`FLASH_QLA_SM70_PREBUILT_EXTENSION_PATH`. Its SHA256 is
`7a52e3e6c612114715f35c5378ed093c00d408b87817ff530a36bfc222548ad6`.

Run ID: `20260919T173120Z`. Emmy source: `f04d579356273b1f09b1cc468f3725591b199ac0`, clean staged inputs.
All 16 selected rows have terminal records: 13 succeeded and three failed. The failed row IDs are
`3444e4adacee` (Emmy chunk-state watchdog), `9f3870a22b16` (Emmy prefill timeout), and `09103b3ae3f3` (external
48-head, 128-token prefill timeout). Each process has a 110-second limit and a 120-second command limit.
The supplied host is retained, with all four GPUs idle after the run. No VM was provisioned or deleted.

Archive: `results_v100x1.tar.gz`, rooted at `2026-09-19_17-31-20/`. Each expanded row has a system-only
`*.experiment.yaml` and an `*_artifacts.tar.gz`. The nested archive contains `artifacts/measurement.log`,
`status.txt`, `requirements.freeze.txt`, and `versions.txt`; completed timing records are in `measurement.json`.
External rows also retain `reference-revision.txt`, `reference.patch`, `reference-binary.sha256`, and their
benchmark output. External decode writes a Torch payload instead of JSON; its timing summary is in the log.
The root includes `reference-build.log` and `reference-build.ninja`. Failed rows remain in the archive.
Earlier setup runs and exploratory probes are not measurement evidence for these tables.

The next scheduling step is Volta atom and register-repacking support, followed by measurements against the same
captured decode baseline and the external prefill implementations. The current results do not justify choosing
Emmy over torch.compile for this decode workload.
