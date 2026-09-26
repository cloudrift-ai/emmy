# Paged KV cache on the native Qwen3-0.6B path

## Status: serves on a V100 over a paged cache, from recorded schedules

On a Tesla V100-SXM2-16GB (sm_70, CUDA 12.9, driver 580.178) the native path exports, generates and serves over a
paged KV cache with no environment pin: every fork of the six compiled fragments — the projections, the head, and
the embedding, rotary and attention glue that used to be hand-written CUDA — is decided by a measured row of
[`golden/v100_sm70.json`](golden/v100_sm70.json) under `--strict-evidence`, with the repository goldens out of scope
and an empty tuning database. At 16-token pages the pack holds 56 paged buffers (28 layers × K, V) and 650 launches;
only the two sampling kernels are still hand-written.

```
emmy generate Qwen/Qwen3-0.6B --export-native pack --context-length 256 --page-tokens 16 \
  --golden golden/v100_sm70.json --strict-evidence          # 32-36 s
emmy generate Qwen/Qwen3-0.6B --native-pack pack --prompt "The capital of France is" --max-new-tokens 8
  ->  Paris. The capital of Italy is Rome
emmy serve Qwen/Qwen3-0.6B --generate --native --golden golden/v100_sm70.json --strict-evidence \
  --max-model-len 256 --page-tokens 16 --port 8000
```

Two requests against that server, each one shot, wall time on the client including HTTP. The first two columns are
the hand-written glue kernels with two head schedules; the last is the compiled glue with the direct head:

| request | steps (prompt + output) | cooperative head | direct head | compiled glue |
| --- | ---: | ---: | ---: | ---: |
| chat, "What is the capital of France? Answer in one word." → "Paris" | 24 + 2 | 0.44 s | 0.28 s | 0.31 s |
| streamed completion, "The capital of France is", 32 tokens | 5 + 32 | 0.81 s | 0.57 s | 0.61 s |

That is 15–16 ms per token step through the server, about 60 output tokens per second at concurrency one, and every
run produces the same text ("Paris. The capital of Italy is Rome. The capital of Spain is Madrid. …"). These are
single requests, not a benchmark: the vLLM benchmark client is not installed on this host, and the recipe's 256- and
64-token page rows were not run on the card. The paged addressing is not what these numbers measure; the schedule is.

## The schedule

Per layer, the best recorded row of every piece, in microseconds on the V100, beside the 4080 rows the cuts came from:

| fragment | pieces | V100, sum of best rows | 4080 (native_manual_schedules) |
| --- | ---: | ---: | ---: |
| pre-attention (q, k, v projections and norms) | 11 | 91 | 66–69 |
| rotary and cache write | 3 | 5 | — |
| attention over the cache | 4 | 45 | — |
| post-attention (o_proj, residual, MLP, norms) | 6 | 196 | 78 |
| final norm and output head | 3 | 406 | 495–506 |
| embedding (once per step) | 1 | 2 | — |

The pieces that matter and what they chose:

| piece | schedule | µs | the alternative |
| --- | --- | ---: | --- |
| q projection [1,2048] | `WORK=t128 REDUCE=coop` | 37 | — |
| k, v projections [1,1024] | `WORK=t128 REDUCE=coop` | 20 + 20 | — |
| gate/up projection [1,3072]×2 | `WORK=w2x4 TILE=mma_m8n8k4_f16_f32/f2x2/k8 STAGE=d1/smem` | 97 | 108 cooperative |
| down projection [1,1024] from 3072 | `WORK=t128 REDUCE=coop` | 57 | 1,543 direct |
| o_proj [1,1024] | `WORK=t128 REDUCE=coop` | 37 | 81 tensor-core, 40 `g2a` |
| output head [1,151936] | direct (no work split, no reduce) | 403 | 6,521 cooperative |
| attention, 16 heads over 256 keys | scores, softmax statistics and probabilities each cut into their own kernel | 45 | 1,269–1,434 fused, 1,628 direct |

The fused attention is the one place the maximal fusion has to be cut before it is usable: fused, every output
element recomputes all 256 scores twice, once for the softmax statistics and once for the weighted sum, which is
134 million multiply-adds per layer for half a million of useful work. Two cuts, at the score contraction and at the
statistics fold, make scores and statistics kernels of their own, and the probability-weighted sum reads them back.

The head is the whole difference between the first two serving columns above: 6.1 ms of the 17 ms step was one kernel
reducing over 1,024 elements per output with a 128-thread cooperative block for each of 151,936 outputs, which the
4080 experiment had already found slow there. Cutting the fragments is what makes every projection a kernel with a
grid over its outputs; the branch's earlier scalar pin ran the fused fragment in a single block, 69 ms for pre and
249 ms for post.

## How it was found

The cuts are the 4080 experiment's routing rows, applied unchanged: seam spellings address the Loop IR, which is the
same on every card, and `emmy golden check` confirms the three stored targets are the fresh lowering on the V100.
Each candidate is one `emmy run --golden working.json --realization <seed> --bench --record-greedy --strict` under
`EMMY_KNOBS` spelling those cuts plus a schedule, against a fresh tuning database, 30–60 s each:

| candidate | pin beside the cuts | outcome |
| --- | --- | --- |
| A1 | `TILE=` | scalar tier, greedy chooses the rest: `t128`/`coop` on every projection |
| A2 | `TILE= WORK=t128 REDUCE=coop STAGE= RASTER=` | the same rows as A1 |
| A3 (head) | `TILE= WORK= REDUCE= STAGE= RASTER=` | direct head, 16× faster than A1's |
| B (post) | `TILE=mma_m8n8k4_f16_f32/f2x2/k8 STAGE=d1/smem WORK=w2x4 REDUCE=` | gate/up accepted it and won by 10%; o_proj lost; the down projection fell to direct and took 1.5 ms |

A global pin reaches every piece of the set, so B is recorded for the one piece where it won and the pick discards
the rest. Every run passed the strict accuracy check against eager; Volta's tensor-core tile has miscompiled silently
before, so no row was recorded without it. The greedy pick was not tried unpinned on the projection fragments.

The three glue fragments were traced from the same modules the export compiles (`emmy trace` over a graph dump) and
recorded the same way. The embedding and rotary kernels take whatever the greedy picks, 1.5–1.8 µs each. Attention
needed its seams listed (`cuttable_seams`, from inside a compile) and three routes tried:

| candidate | pin | outcome |
| --- | --- | --- |
| fused, unpinned or `TILE=` | — | 1.27–1.43 ms: the scores recomputed per output element |
| fused, direct | `WORK= REDUCE=` | 1.63 ms at eight blocks |
| scores + statistics cut | `PLACE@map.1/inner.1/map=cut PLACE@map.1/inner.1/map.4/twist=cut TILE=` | 45 µs: 26 statistics, 8 scores, 11 weighted sum |
| the same plus the probabilities cut | `… PLACE@map.1/inner=cut` | 46 µs |

## Before the port

The first version of this experiment, on the branch's own runtime before the rebase onto #885, generated on this
card only under a scalar-tier environment pin, and slowly. Unpinned, the token step never completed: the fused
post-attention kernel took tens of seconds per launch with a Volta tensor-core tile the greedy reached for, with no
paging, no pack and no Rust runtime involved. Under `EMMY_TILE= EMMY_WORK=t128 EMMY_REDUCE=coop EMMY_STAGE=` the export
dropped from 10.3 s to 0.8 s and the pack generated, correct but at about three minutes per token: at one row the
fused fragment ran in one block on an 80-SM card. Tuning the unsplit layer targeted the wrong kernel set, tuning the
split wrappers did not finish its first target in 2.5 hours, and `UNROLL`, `VECTORIZE_*` and `LOOPIFY` pins did not
reach the schedule. The golden recorded then was in the YAML format #912 retired and did not survive the rebase; the
one above replaces it.

## Next

1. **Run the recipe's other page sizes** (256 and 64 tokens) on the card; only 16 has been served.
2. **Qualify numerically.** The seventeen checkpoint cases the 4080 experiment ran have not been run on this
   artifact; the strict check covers the three fragments in isolation, and the served text matches the 4080's.
3. **A benchmark, not two requests.** Install the vLLM benchmark client on the host and run the same 32/256/1024
   input lengths as the 4080 experiment.
4. **Measure what paging costs** against the same schedule at one page, on this card and on a 4090-class one.
