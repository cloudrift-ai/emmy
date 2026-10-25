# Qwen3.6-35B-A3B-FP8 discovery

## Native, plugin and stock vLLM on one RTX 5090 (2026-10-25, in progress)

The setup of `recipes/Qwen3.8-27B-NVFP4-NVIDIA/DISCOVERY.md` (2026-10-25): one request at a time, the FP8 KV cache, the
MTP drafter at two tokens, no prefix cache, text only, the same real-text prompts and client. The rented x16 RTX 5090
these cells began on powered off at 20:34 during the native GSM8K run, so no column is complete and none of its cells
counts toward a table from one session; they are recorded here as a first look, and the table will come from one
session on another card. Not yet measured: every plugin cell, the native GSM8K and 261,632-token cells, and every
stock cell past the first.

- **Stock vLLM 0.32.0 does not load the checkpoint on its own.** Without offloading it runs out of device memory while
  it loads the weights (30.55 GiB allocated, 256 MiB more asked for): the routed experts alone are 30.0 GiB of the
  checkpoint's 34.9 GiB. With `--cpu-offload-gb 13 --cpu-offload-params experts` (vLLM's UVA offload: 13.25 GiB of
  expert weights in pinned host memory, read in place over the link) it boots with 20.98 GiB of weights on the card and
  a 528,230-token KV cache, twice the window, so 13 GiB offloads more than the window needs.
- **First cells**, 256-token prompts: native 20.7 ms TPOT, 5.4 s to the first token, 81% of drafts accepted, 24.5 GiB on
  the device; stock with 13 GiB offloaded 14.2 ms, 0.33 s, 82%, 29.8 GiB. Native at 7,680 tokens: 20.9 ms, 159 s to the
  first token (20.7 s per 1,000 tokens), 73%, with the kernel's memory compaction thread at 100% of a core beside the
  16 GiB of pinned expert pages.

## Assessment on 2026-10-05

Maintained, heat 35. The Qwen3.6-35B-A3B line's strongest direct quant signal (NVFP4 node 6.05M downloads at its
check); mid-size MoE with useful hardware coverage across RTX PRO 6000, A100, and H200 single cards at 262K context.
