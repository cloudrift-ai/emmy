# OLMoE-1B-7B-0125-Instruct: router prediction and expert caches, replayed offline

## Result

A cache of recently routed experts cuts the expert reads that reach host memory to about half of what the static hot
set leaves at the same number of GPU experts per layer. Router prediction is accurate, but in a cache it moves more
bytes in total than it saves. Layer i's router applied to the hidden state entering layer i, one attention ahead, names
a median 89–90% of the experts the layer really picks (layers 3–15). Prefetching those experts leaves a fifth to a half
of the cache's waiting reads, but adds more copies than it removes. Decode at one stream on PCIe is bound by the
bytes it moves over the link, so prediction pays only where the link would otherwise sit idle.

Host expert reads per decode step, summed over the 16 layers (128 routed experts per step):

| GPU experts per layer | Workload | Static | LRU cache | LRU + prediction, d=1: waiting / prefetched | d=2: waiting / prefetched | Belady bound |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 48 | chat | 17.9 | 9.5 | 2.8 / 10.8 | 4.2 / 10.9 | 3.8 |
| 32 | chat | 49.4 | 27.0 | 6.9 / 29.8 | 10.3 / 30.0 | 13.4 |
| 16 | chat | 90.0 | 60.1 | 13.0 / 63.7 | 20.0 / 62.3 | 39.7 |
| 8 | chat | 111.5 | 83.1 | 20.6 / 87.2 | 31.9 / 83.4 | 83.1 |
| 48 | code | 10.8 | 5.1 | 2.3 / 6.8 | 3.0 / 7.0 | 2.1 |
| 32 | code | 28.1 | 18.7 | 5.8 / 22.1 | 8.5 / 22.3 | 9.0 |
| 16 | code | 53.5 | 43.6 | 11.6 / 48.5 | 18.0 / 47.9 | 28.9 |
| 8 | code | 74.5 | 69.7 | 21.7 / 75.4 | 33.9 / 72.4 | 69.7 |

"Waiting" reads are the ones a step still waits for, and "prefetched" are the copies the prediction started earlier.
With prediction, the total bytes over the link are the sum of the two, which is more than the LRU cache moves alone at
every budget. At 8 GPU experts the cache holds exactly one step's picks, so LRU equals the bound.

- **Accuracy by depth.** One attention ahead (d=1), the predicted top-8 holds 70% of the real picks at layer 1, 80–85%
  at layers 2–4, and 87–95% from layer 5 on; a top-16 prediction holds 94–100% of them from layer 3 on. One layer
  ahead (d=2) the top-8 holds 5–17 points less, most near the input. Layer 0 cannot be predicted from the embeddings (20% overlap). Chat
  and code agree within a few points.
- **LRU against the bound.** At 32 and 16 GPU experts, the bound reads half to two thirds of what LRU reads, and at 48
  about 2.5× less, on few reads. No simple eviction rule closes much of that gap (below).

## Eviction rules

The same demand cache with other rules for which expert to evict. The best is a decayed count with a 16-step half-life
(`ema16`): 3–7% fewer host reads than LRU on chat and 5–11% on code. Rules that remember longer do worse, and so do
seeding the counts with the boot-time histogram, refusing to cache an expert that scores below its victim, and pinning
half the budget to the static set. The bound's advantage comes from knowing the future, which no rule over past
routing has; the only signal of it here is the router prediction, and in a cache that costs more bytes than it saves.

Host expert reads per decode step:

| Workload | GPU experts | LRU | LFU | LFU, histogram prior | ema16 | ema64 | ema256 | ema64, admission | ema64, histogram prior | Half pinned, half LRU | Belady bound |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chat | 48 | 9.5 | 9.3 | 9.9 | 8.9 | 9.0 | 9.2 | 10.3 | 9.9 | 11.1 | 3.8 |
| chat | 32 | 27.0 | 27.8 | 29.5 | 26.3 | 27.1 | 27.7 | 32.4 | 29.4 | 35.0 | 13.4 |
| chat | 16 | 59.9 | 59.4 | 60.6 | 57.4 | 58.7 | 59.1 | 68.5 | 60.8 | 71.1 | 39.7 |
| code | 48 | 5.1 | 5.7 | 5.7 | 4.9 | 5.3 | 5.7 | 5.6 | 5.6 | 5.4 | 2.0 |
| code | 32 | 18.7 | 19.6 | 19.4 | 16.7 | 17.9 | 19.4 | 20.0 | 18.1 | 19.5 | 9.0 |
| code | 16 | 43.6 | 42.9 | 42.7 | 40.6 | 41.9 | 42.8 | 47.2 | 41.9 | 43.5 | 28.9 |

LFU counts every routing since the start; `ema<h>` decays each count by half every h decode steps; ties go to recency.
The histogram prior seeds the counts with milestone 0's selections, weighted as 64 steps. Admission reads a missing
expert in place without caching it when its score is not above the victim's. At 8 GPU experts every rule equals LRU,
because the cache holds exactly one step's picks.

## Protocol

`emmy bench experiments/OLMoE-1B-7B-0125-Instruct/expert_cache_sim_rtx5090 --local` on the dev box's RTX 5090. The HF
model (`allenai/OLMoE-1B-7B-0125-Instruct`, revision `b89a7c4bc24fb9e55ce2543c9458ce0ca5c4650e`, FP16, transformers
5.12.0 in the `0.23.0-53bf508d8` serving image; Emmy is not used) ran milestone 2's benchmark prompts: the 8 chat
(`HuggingFaceH4/ultrachat_200k`) and 8 code (`openai/openai_humaneval`) prompts from offset 32, with the chat template,
greedily, 64 output tokens without stopping at EOS. That is 504 decode steps per workload. Each step recorded every
layer's real top-8, and the top-16 of the layer's router applied to the hidden state entering the layer (d=1), and
entering the layer before it (d=2), through the layer's own post-attention norm. 11 of the 16 texts equal the served
run's in expert_cache_rtx5090; the other five part ways later, from FP16 rounding between HF and Emmy.

The replay counts, per layer, the routed experts not in GPU memory at each decode step. The static set is milestone
2's `rank.json`. The LRU cache starts from the same set, copies a missing expert in before the read, and evicts the
least recently routed. The prediction variant first inserts the step's predicted top-8. Belady's bound evicts the
expert whose next use is farthest. Caches persist across a workload's requests. Prefill is not replayed.

## Limitations

- Decode at one stream only. Each step's picks are 8 per layer; a batch picks more distinct experts per step.
- Read counts, not times. Expert reads cost the same bytes whether a copy or an in-place read moves them.
- Prediction is replayed on the HF model's own hidden states; the served model's rounding differs slightly.

## Run and platform

Run `2026-10-17_00-13-37`. One RTX 5090 (32 GB, PCIe 5.0 x1 in this box, which does not matter offline) on the dev box.
`results_rtx5090x1.tar.gz` (Git LFS) holds the experiment record, `trace_chat.npz` and `trace_code.npz` (real and
predicted experts per step and layer), `texts.json`, `accuracy.tsv`, `sim.tsv` and `evict.tsv`.
