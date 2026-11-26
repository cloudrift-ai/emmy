# GLM-5.2-FP8 onboarding results

## NVIDIA H200 141GB x8

Qualified 2026-10-23 on one GCP spot node, repository revision `13eb76bda` plus this change.

- **Model:** `zai-org/GLM-5.2-FP8` at `f33c6dc501ee5a2c7e35155653b1b1abbc320951`, block FP8 (e4m3, 128x128). Text
  only: the checkpoint ships no vision or audio config.
- **Fit:** about 750B parameters (256 routed experts of width 2048 in 75 MoE layers plus the MTP layer, about 735B,
  plus attention, dense layers and embeddings), about 40B active; 756 GB on disk at about one byte per parameter.
  Min-to-serve is 756 GB x 1.3 = 983 GB against 8 x 143,771 MiB = 1,206 GB: one replica fits, two do not.
- **Platform:** 8x NVIDIA H200 141GB, NVLink, driver 580.178.04, Ubuntu 24.04.
- **Image:** `vllm/vllm-openai:v0.32.0` (CUDA 13.0), the newest vLLM release on this date. `lmsysorg/sglang:v0.5.22`
  was the SGLang comparison.

**Serving configuration.** Data-parallel attention on 8 ranks with the 256 experts split across them (DP8+EP8),
FP8 KV cache, MTP with 3 draft tokens, 2,048 batched tokens per rank, `--gpu-memory-utilization 0.95`, 8 requests
per rank (64 for the node), 40,960-token context. Each rank holds about 106 GiB of weights and 320,128 tokens of cache.

**Throughput** on 32,768 random input tokens and 4,096 output tokens (seed `i` for repeat `i`, temperature 0, EOS
ignored, unique prompts), 192 requests at concurrency 64, three repeats:

| | Mean | Repeats |
| --- | ---: | --- |
| Output tokens/s | 1,316.7 | 1,320.8 / 1,329.9 / 1,299.6 |
| Total tokens/s | 11,850.7 | 11,886.9 / 11,968.6 / 11,696.8 |
| Requests/s | 0.32 | 0.32 / 0.32 / 0.32 |
| Mean TPOT | 39.96 ms | 40.14 / 38.85 / 40.89 ms |
| Failed requests | 0 | 0 / 0 / 0 |
| Duration | 597 s | 595.4 / 591.4 / 605.1 s |

TTFT was not the objective: at this concurrency it is mostly queueing behind 32,768-token prefills. MTP accepted 92.4%
of drafts here (3.77 tokens per step) because greedy continuations of random-token prompts repeat; on natural text
(GSM8K prompts on the same server) it accepted 77.5% (3.33 tokens per step).

**Why this configuration.** KV capacity caps concurrency at this length. TP8 copies the MLA cache to every rank and
holds about ten 36,864-token sequences (395.2 output tokens/s at its limit of 8). Data-parallel attention spreads the
cache (871.6 at 64 with a BF16 cache), the FP8 cache holds 1.66 times as many tokens (1,108.1 mean over three repeats
at 64), and MTP adds 18.8% on top. Without MTP, concurrency 80 and 96 gave less than 64. SGLang `v0.5.22` with the
same split and an FP8 cache gave 1,118.2 at 64, close enough to call a tie, so vLLM is kept. DeepEP low-latency
all-to-all and a 4,096-token prefill chunk were both slower. The full comparison is in
`experiments/GLM-5.2-FP8/serving/RESULTS.md`.

**Correctness.** Chat returns coherent answers; reasoning lands in `reasoning_content` (`glm47` parser); a tool
request returns structured `tool_calls` (`glm47` parser); a 40,384-token request recalls a code planted at its start.
The FP8 KV cache is lossy, so it was held to an accuracy gate declared before it ran: GSM8K 5-shot strict-match over
400 questions within 0.03 of the BF16-cache lane. This configuration scored 0.9475 ± 0.011 against 0.950 ± 0.011.

**Emmy.** Ineligible. Gate 1 passes: compute capability 9.0 is accepted. Gate 2 fails: Emmy has no trace or runner
path for `glm_moe_dsa`. A single-layer trace of a full-indexer layer stops in the sparse-attention indexer, which
needs a materialized causal mask, and its top-k and scatter mask have no Emmy lowering. A shared-indexer layer needs
the top-k indices of the previous full layer as an input. Transformers drops the MTP layer, so Emmy has no path for
it either. Coverage is partial, so no golden is committed. The blocker is the architecture, so it holds on every
declared deployment and the recipe carries `emmy-blocked`.

**Other workloads** (the datacenter rows of `prompts/onboard-model/benchmark.md` on this configuration, 0 failed):
4096/4096 at concurrency 1 gives 142-157 output tokens/s (6.2-6.9 ms TPOT); at 16, 1,238.2; at 32, 1,578.9. 8192/256
at 16 gives 254.1 output and 8,385.7 total tokens/s; 1024/1024 at 64 gives 2,299.5; 256/256 at 256 gives 3,961.2.

**Reproduce:**

```bash
emmy bench experiments/GLM-5.2-FP8/serving --ssh user@host --filter lane=vllm-dp8ep-fp8kv-mtp3 \
  --filter benchmark.repeats=3
```

**Limitations.** The context is qualified at 40,960 tokens; the native 1,048,576 cannot fit one data-parallel rank.
Accuracy was checked on GSM8K only. Random-token prompts overstate MTP's acceptance. Single runs without MTP varied
by up to 16% between repeats; with MTP by 2.3%.
