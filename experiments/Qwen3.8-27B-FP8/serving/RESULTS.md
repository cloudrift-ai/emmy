# Qwen3.8-27B-FP8 serving on V100

Factual artifact index for the shared serving protocol. Each platform section records what was executed and where the
raw evidence lives; it does not interpret the measurements. The recommended-configuration report lives beside the
recipe in `recipes/Qwen3.8-27B-FP8/RESULTS.md`.

## NVIDIA Tesla V100 SXM2 16GB x4

- Archive: `results_v100x4.tar.gz`, archived root `2026-10-15_10-52-51/`, byte-checked against the raw run.
- Run timestamp `2026-10-15T10:52:51Z`, run ID `20261015T105251Z`; repository revision
  `546f1fd3c59ad2fbe42f2451e03f650616a9c749` with uncommitted changes (this verification's report and archive edits).
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-139-generic, Intel Xeon E5-2680 v4, 24 logical CPUs, 204 GiB RAM.
- GPUs: four NVIDIA Tesla V100-SXM2-16GB, 16,384 MiB each, compute capability 7.0, driver 580.178.04. NVLink is up:
  pairs 0-1 and 2-3 share one link, pairs 0-2 and 1-3 two, and the diagonals 0-3 and 1-2 cross PCIe (PHB).
- Model: `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: temperature 0, ignored EOS, 2 warm-ups, three repeats against one server with seeds 0, 1 and 2, text-only
  path, context 262,144, TP4, Triton GDN prefill backend, FLASH_ATTN_V100 attention, TurboMind FP8 dequant
  (`VLLM_SM70_FP8_TURBOMIND=1`, `VLLM_SM70_QUANT_BACKEND=turbomind`, `VLLM_SM70_GDN_DECODE_FLASHQLA=0`),
  `--exclude-tools-when-tool-choice-none`, key/value cache sized from `gpu_memory_utilization` 0.88. vLLM's custom
  all-reduce stays off (not every GPU pair has NVLink), so tensor-parallel traffic goes through NCCL.
- Measured KV pool: 287,512 tokens, 1.10x maximum concurrency at full context.
- Rows executed: 1 of 1 succeeded; 0 failed requests in each of the three repeats.

| Row | Input / output | Concurrency | Prompts | Repeats |
| --- | ---: | ---: | ---: | ---: |
| `v100x4_r3` | 1,000 / 1,000 | 4 | 16 | 3 |

Startup cost recorded for this row: deploy 359.97 s of model load and warm-up (12.83 s weights, 104.43 s
`torch.compile`, 108.0 s CUDA graph capture). The benchmark took 1,297.57 s across the three repeats and the serving
workload was torn down afterwards.

This archive replaces the 2026-10-12 run of the same row (archived root `2026-10-12_11-26-36/`, driver 580.178.04),
which in turn replaced the 2026-09-15 run (archived root `2026-09-15_19-15-00/`, driver 580.173.02); both remain in
the git history.

The archive contains the per-row system-only experiment record, the client benchmark log with one stanza per repeat,
and the engine server log.

## NVIDIA Tesla V100 SXM3 32GB x2

This platform is kept as evidence and is not part of the recommended recipe: two-card SXM3 rentals cannot get NVLink.

- Archive: `results_v100x2.tar.gz`, archived root `2026-09-15_06-28-41/`, byte-checked against the raw run.
- Run timestamp `2026-09-15T06:28:41Z`, run ID `20260915T062841Z`; repository revision
  `825986afd21dfc745204a3b70755fd11356ac1a8` with uncommitted changes.
- Host: `riftvm`, Ubuntu 24.04.1, kernel 6.8.0-51-generic, Intel Xeon Platinum 8168, 10 logical CPUs, 167 GiB RAM.
- GPUs: two NVIDIA Tesla V100-SXM3-32GB, 32,768 MiB each, compute capability 7.0, driver 580.126.20, passed through
  to a VM on separate emulated PCIe ports. The host reports PHB between the pair and no peer-to-peer support, no
  NVSwitch is visible, and the VM image sets `NvLinkDisable=1`, so the tensor-parallel all-reduce goes through host
  memory.
- Model: `Qwen/Qwen3.8-27B-FP8@017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`.
- Engine image: `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608` (vLLM
  `1.2.3.dev87+gd76126608.d20260810`).
- Controls: temperature 0, ignored EOS, 2 warm-ups, three repeats against one server with seeds 0, 1 and 2, text-only
  path, context 262,144, TP2, Triton GDN prefill backend, FLASH_ATTN_V100 attention backend,
  `--exclude-tools-when-tool-choice-none`, `VLLM_SM70_FP8_TURBOMIND=1`, `VLLM_SM70_QUANT_BACKEND=turbomind`,
  `VLLM_SM70_GDN_DECODE_FLASHQLA=0`.
- Measured KV pool: 365,925 tokens, 1.40x maximum concurrency at full context.
- Rows executed: 1 of 1 succeeded; 0 failed requests in each of the three repeats.

| Row | Input / output | Concurrency | Prompts | Repeats |
| --- | ---: | ---: | ---: | ---: |
| `v100x2_r3_tps2` | 1,000 / 1,000 | 4 | 16 | 3 |

Each repeat draws its prompts from its own seed. An earlier run of this row replayed seed 0 three times, and because
the server keeps a prefix cache, the second and third repeats skipped most prefill (median TTFT 685 ms against 2,167 ms
in the first); that run was discarded and the harness fixed rather than reported.

Startup cost recorded for this row: deploy 270.9 s, of which 265.5 s was model load and warm-up. The benchmark took
1,012.8 s across the three repeats and the serving workload was torn down afterwards.

The archive contains the per-row system-only experiment record, the client benchmark log with one stanza per repeat,
and the engine server log. The SXM3 row was removed from `recipe.yaml` with the platform; its archive is kept.

## Reproduce

```bash
emmy vm create cloudrift --instance-type v100-6-52-400-generic.4 --ssh-key ~/.ssh/id_ed25519.pub \
  --image-url https://storage.googleapis.com/cloudrift-vm-disks/disks/github/ubuntu-noble-server-gpup-580-129-20260810-232733.img
emmy bench experiments/Qwen3.8-27B-FP8/serving --ssh USER@HOST
```
