# RTX 5090 attention comparison — results

## NVIDIA GeForce RTX 5090 x1

### Status

The baseline lane is complete: 59 of 59 offered setups measured, every one with whole-forward CUDA graph capture and
every backend passing its correctness check against eager SDPA. The Emmy lane has goldens for the setups listed below
(`golden/README.md` says how each was recorded). For GQA decode the recipe's Emmy lane has been re-run over the
goldens — every recorded setup replays, both arms — and its table stands on its own below. For the prefill setups the
lane has not been re-run yet: those Emmy numbers are the recording runs themselves — the fastest hand-pinned row of
each setup, timed on the card beside eager in one process, whole forward captured, over 20 iterations after 3 warmups
where the recipe takes 10 after 1. Plain decode (`decode_causal`) has no golden: with one query row the placement
drops the query axis, so the fused kernel offers no tensor-core tile and nothing worth replaying can be recorded yet.

### GQA decode, replayed from the split-KV goldens

Each GQA decode golden pins FlashAttention-2's split-KV: the key range splits across `n` CTAs
(`REDUCE@map.1/twist=g<n>k`, about 256 keys per CTA), the partial keeps the fused kernel's tensor-core chunk tier
and writes its three carrier states to an f32 workspace, and a serial finalize merges them per cell. The kernel
sum is the lane's replay (`emmy run --golden`, one warmup and ten iterations, minimum of the ten, mean of the two
repetitions' minimum); the whole program adds the two launches' gaps. The baseline columns are the baseline
lane's records above.

| setup | Emmy replay, kernel sum (us) | Emmy whole (us) | FlashAttention-2 (us) | Inductor (us) | best baseline (us) | Emmy whole / FA-2 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| decode_gqa-b1-s1024 | 6.4 | 8.2 | 12.0 | 18.2 | 12.0 (FlashAttention-2) | 0.68 |
| decode_gqa-b1-s2048 | 9.9 | 12.3 | 15.3 | 25.5 | 15.3 (FlashAttention-2) | 0.80 |
| decode_gqa-b1-s4096 | 10.7 | 12.3 | 25.0 | 31.8 | 25.0 (FlashAttention-2) | 0.49 |
| decode_gqa-b1-s8192 | 17.9 | 20.2 | 34.6 | 55.6 | 34.6 (FlashAttention-2) | 0.58 |
| decode_gqa-b1-s16384 | 34.4 | 36.5 | 58.1 | 112.8 | 58.1 (FlashAttention-2) | 0.63 |
| decode_gqa-b1-s32768 | 89.5 | 90.1 | 101.2 | 207.9 | 101.2 (FlashAttention-2) | 0.89 |
| decode_gqa-b8-s1024 | 17.5 | 18.4 | 29.2 | 61.5 | 29.1 (FlexAttention) | 0.63 |
| decode_gqa-b8-s2048 | 34.2 | 35.1 | 52.1 | 107.8 | 52.1 (FlashAttention-2) | 0.67 |
| decode_gqa-b8-s4096 | 89.3 | 90.2 | 92.9 | 198.1 | 92.9 (FlashAttention-2) | 0.97 |
| decode_gqa-b8-s8192 | 190.2 | 192.2 | 171.8 | 401.7 | 171.8 (FlashAttention-2) | 1.12 |
| decode_gqa-b8-s16384 | 372.7 | 374.8 | 335.8 | 774.6 | 335.8 (FlashAttention-2) | 1.12 |

Batch 1 is ahead of FlashAttention-2 at every length, 0.49-0.89x of its time whole-program, and 2.2-3.1x ahead of
Inductor. Batch 8 is ahead through 4096 keys and 1.12x behind at 8192 and 16384, where the split's CTAs already fill
the card and the partial's per-chunk work decides. The batch-8 setup at 32768 keys has no golden: its record run's
greedy worker returned no outputs (the eager reference materializes 4.3 GB per broadcast operand), so the lane's
batch-8 row fails on the missing golden while its five recorded setups replay. The reference arm's own Emmy column
is not reported for decode: without golden evidence the greedy deploys the prior's pick, which ranges from a fused
single-wave row to a scalar loop.

The raw lane records — per-setup JSON, logs and status tables for both batches — are in
`results_rtx5090x1_emmy_decode_gqa.tar.gz`; `emmy_decode_gqa_lane.csv` is the same table as numbers.

### Emmy, recorded rows

| setup | eager (FA-2) µs | Emmy µs | Emmy vs eager |
| --- | ---: | ---: | ---: |
| prefill_causal-b1-s1024 | 94 | 76 | 1.24x |
| prefill_causal-b1-s2048 | 275 | 271 | 1.02x |
| prefill_causal-b1-s4096 | 894 | 878 | 1.02x |
| prefill_causal-b1-s8192 | 3150 | 3066 | 1.03x |
| prefill_causal-b1-s16384 | 10401 | 10722 | 0.97x |
| prefill_causal-b8-s1024 | 474 | 452 | 1.05x |
| prefill_causal-b8-s2048 | 1628 | 1533 | 1.06x |
| prefill_global-b1-s1024 | 126 | 127 | 0.99x |
| prefill_global-b1-s2048 | 484 | 429 | 1.13x |
| prefill_global-b1-s8192 | 6066 | 5642 | 1.08x |
| prefill_global-b8-s1024 | 808 | 748 | 1.08x |
| prefill_global-b8-s2048 | 2983 | 2800 | 1.07x |
| prefill_gqa-b1-s1024 | 148 | 140 | 1.06x |
| prefill_gqa-b1-s2048 | 465 | 450 | 1.03x |
| prefill_gqa-b1-s4096 | 1421 | 1420 | 1.00x |
| prefill_gqa-b1-s8192 | 5990 | 5778 | 1.04x |
| prefill_gqa-b8-s1024 | 892 | 827 | 1.08x |
| prefill_gqa-b8-s2048 | 3148 | 2942 | 1.07x |

### Protocol

One process per setup. Inputs are allocated in both layouts, every backend is warmed, each is checked elementwise
against eager SDPA at rtol/atol 1e-2, then all of them are captured into one shared CUDA graph pool and replayed in a
single interleaved window — one warmup, ten measurements, rotated so no backend keeps the same position in the window.
The reported number is the mean of those ten, following the Nautilus paper. Each record also keeps the minimum, the
median and every sample.

FP16, head dimension 128, batch 1 and 8, sequence lengths 1024 to 32768. MHA uses 32 query and KV heads; GQA uses 64
query and 8 KV heads. Decode is a single query position against the full key length.

### System

torch 2.14.0+cu130, FlashAttention 2.8.3 built from source for sm_120, TileLang 0.1.8 with apache-tvm-ffi 0.1.8.post2,
cuDNN 9.24.0, CUDA 13.0, driver 580.173.02. Full provenance in the archive under `evidence/`.

### Measurements

Mean latency in microseconds over ten repeats. `n/a` means the backend has no kernel for that setup and recorded why.

| setup | SDPA | PyTorch Inductor | FlexAttention | FlashAttention-2 | cuDNN | TileLang | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| decode_causal-b1-s1024 | 17.9 | 18.1 | 28.4 | 18.4 | 11.9 | n/a | cuDNN |
| decode_causal-b1-s2048 | 26.1 | 26.2 | 46.8 | 25.8 | 14.8 | n/a | cuDNN |
| decode_causal-b1-s4096 | 49.5 | 48.7 | 85.7 | 62.1 | 40.8 | n/a | cuDNN |
| decode_causal-b1-s8192 | 78.9 | 88.7 | 162.2 | 93.3 | 83.2 | n/a | SDPA |
| decode_causal-b1-s16384 | 177.8 | 178.5 | 314.4 | 174.3 | 165.7 | n/a | cuDNN |
| decode_causal-b1-s32768 | 335.8 | 335.3 | 618.8 | 333.4 | 325.3 | n/a | cuDNN |
| decode_causal-b8-s1024 | 122.9 | 125.6 | 49.4 | 123.9 | 82.0 | n/a | FlexAttention |
| decode_causal-b8-s2048 | 173.7 | 173.9 | 150.0 | 173.7 | 164.2 | n/a | FlexAttention |
| decode_causal-b8-s4096 | 361.9 | 361.3 | 382.9 | 378.2 | 394.2 | n/a | PyTorch Inductor |
| decode_causal-b8-s8192 | 649.7 | 650.0 | 638.2 | 649.6 | 637.5 | n/a | cuDNN |
| decode_causal-b8-s16384 | 1286.3 | 1285.7 | 1282.8 | 1286.0 | 1267.9 | n/a | cuDNN |
| decode_causal-b8-s32768 | 2561.1 | 2561.0 | 2548.0 | 2559.4 | 2529.8 | n/a | cuDNN |
| prefill_causal-b1-s1024 | 95.8 | 96.5 | 101.0 | 94.3 | 94.5 | 113.8 | FlashAttention-2 |
| prefill_causal-b1-s2048 | 281.3 | 275.7 | 285.8 | 280.3 | 274.9 | 334.3 | cuDNN |
| prefill_causal-b1-s4096 | 894.7 | 892.0 | 922.7 | 883.8 | 870.5 | 1075.3 | cuDNN |
| prefill_causal-b1-s8192 | 3110.8 | 3108.7 | 3203.7 | 3069.6 | 3026.8 | 3758.5 | cuDNN |
| prefill_causal-b1-s16384 | 10891.8 | 10840.6 | 11121.2 | 10748.3 | 10668.8 | 13247.6 | cuDNN |
| prefill_causal-b1-s32768 | 42146.1 | 42166.7 | 43350.0 | 41780.1 | 41919.9 | 51620.5 | FlashAttention-2 |
| prefill_causal-b8-s1024 | 474.9 | 475.0 | 504.3 | 471.4 | 469.6 | 578.7 | cuDNN |
| prefill_causal-b8-s2048 | 1626.6 | 1628.4 | 1699.7 | 1606.3 | 1587.0 | 1989.1 | cuDNN |
| prefill_causal-b8-s4096 | 5741.8 | 5731.4 | 5925.5 | 5694.2 | 5616.5 | 7018.1 | cuDNN |
| prefill_causal-b8-s8192 | 21050.7 | 21051.4 | 21810.2 | 20876.8 | 20805.8 | 25757.8 | cuDNN |
| prefill_causal-b8-s16384 | 83792.0 | 83763.5 | 86190.5 | 82929.5 | 83275.4 | 102396.5 | FlashAttention-2 |
| prefill_causal-b8-s32768 | 333874.4 | 333994.0 | 342797.0 | 330225.4 | 332515.8 | 406858.0 | FlashAttention-2 |
| prefill_global-b1-s1024 | 127.4 | 127.7 | 134.0 | 123.6 | 121.7 | 150.6 | cuDNN |
| prefill_global-b1-s2048 | 490.6 | 487.5 | 485.8 | 479.0 | 455.5 | 566.9 | cuDNN |
| prefill_global-b1-s4096 | 1650.7 | 1649.5 | 1684.7 | 1616.0 | 1532.9 | 1919.6 | cuDNN |
| prefill_global-b1-s8192 | 5800.6 | 5830.5 | 5985.5 | 5706.9 | 5446.6 | 6758.7 | cuDNN |
| prefill_global-b1-s16384 | 21282.9 | 21369.8 | 21869.7 | 20752.6 | 20450.8 | 25118.9 | cuDNN |
| prefill_global-b1-s32768 | 84916.3 | 84843.9 | 87760.1 | 82717.1 | 82222.9 | 100361.2 | cuDNN |
| prefill_global-b8-s1024 | 803.9 | 804.4 | 813.6 | 784.8 | 768.3 | 949.4 | cuDNN |
| prefill_global-b8-s2048 | 2992.2 | 2993.3 | 3040.5 | 2920.1 | 2780.5 | 3531.5 | cuDNN |
| prefill_global-b8-s4096 | 10662.0 | 10718.6 | 10877.4 | 10410.3 | 10097.0 | 12675.5 | cuDNN |
| prefill_global-b8-s8192 | 41885.1 | 41919.8 | 42900.8 | 40823.9 | 40271.3 | 50092.5 | cuDNN |
| prefill_global-b8-s16384 | 168401.7 | 168550.8 | 174113.7 | 164266.7 | 163098.1 | 200318.3 | cuDNN |
| prefill_global-b8-s32768 | 675063.5 | 675132.0 | 693415.0 | 657359.8 | 656591.7 | 802548.1 | cuDNN |
| prefill_gqa-b1-s1024 | 148.4 | 148.0 | 154.4 | 145.8 | 146.3 | 133.8 | TileLang |
| prefill_gqa-b1-s2048 | 467.1 | 466.2 | 484.4 | 464.3 | 460.4 | 462.1 | cuDNN |
| prefill_gqa-b1-s4096 | 1616.7 | 1622.8 | 1671.7 | 1597.9 | 1587.6 | 1647.5 | cuDNN |
| prefill_gqa-b1-s8192 | 5829.3 | 5830.5 | 5952.3 | 5735.6 | 5655.0 | 6082.7 | cuDNN |
| prefill_gqa-b1-s16384 | 21002.2 | 21064.5 | 21626.4 | 20802.9 | 20832.0 | 22832.0 | FlashAttention-2 |
| prefill_gqa-b1-s32768 | 83849.5 | 83827.0 | 86049.4 | 82888.1 | 83331.0 | 92575.6 | FlashAttention-2 |
| prefill_gqa-b8-s1024 | 886.7 | 886.3 | 937.0 | 877.6 | 881.8 | 914.2 | FlashAttention-2 |
| prefill_gqa-b8-s2048 | 3110.5 | 3122.3 | 3259.0 | 3082.3 | 3058.6 | 3258.5 | cuDNN |
| prefill_gqa-b8-s4096 | 11015.0 | 10958.5 | 11404.9 | 10849.0 | 10769.1 | 11808.1 | cuDNN |
| prefill_gqa-b8-s8192 | 42241.7 | 42244.7 | 43624.9 | 41815.9 | 41947.4 | 46658.9 | FlashAttention-2 |
| prefill_gqa-b8-s16384 | 168130.5 | 168118.5 | 172913.4 | 166264.0 | 166986.7 | 187009.0 | FlashAttention-2 |

### What the numbers say

cuDNN is the fastest backend in 33 of 59 setups and FlashAttention-2 in 20, so between them they own 53. That matches
the framing the Nautilus paper uses, where cuDNN is the manually written kernel a compiler has to beat.

Eager SDPA and Inductor coincide almost everywhere on prefill, usually within a few tenths of a percent. That is
expected rather than surprising: Inductor does not replace the ATen SDPA call, so both dispatch the same fused kernel,
and `torch.compile` buys nothing on a graph that is one operator. Where Inductor does separate from eager is GQA
decode, where the eager path falls back to a materialized broadcast: at batch 1 and 32768 keys eager takes 4834.9 us
against Inductor's 207.9 us and FlashAttention-2's 101.2 us, a 48x gap that is the fallback, not the hardware.

TileLang 0.1.8's unchanged prefill examples are consistently the slowest prefill backend, around 20 to 25 percent
behind cuDNN. Its one win is GQA prefill at batch 1 and 1024 keys. These are the repository's example kernels at their
example tile parameters, not a tuned TileLang, and should be read that way.

cuDNN has no kernel for GQA decode as this experiment expresses it. That operator reaches SDPA through a five
dimensional broadcast reshape rather than materialized KV heads, and cuDNN's backend refuses it with "No available
kernel"; the refusal is recorded per setup instead of a number.

### Repeat variation

Ten repeats after one warmup is the paper's protocol and it is thin at the fast end. Across all measured backends the
spread between the slowest and fastest repeat is 1.5 percent of the mean at the median, but 3.4 percent for setups
under 200 us against 0.9 percent above it, and the worst case is 108 percent — every backend on decode_causal at
batch 8 and 4096 keys, where the whole window is under half a millisecond. Sub-100-microsecond differences in the
decode rows are inside that noise and should not be read as rankings. The archive keeps every sample, so a stricter
estimator can be recomputed without re-measuring.

### Limitations

This is not a Nautilus reproduction, and the departures are listed in the experiment README. The largest is that six
of the paper's baselines are absent — Triton, Helion, Tawa, TVM, FlashInfer and ThunderKittens — so "best baseline"
here is the best of six, not of twelve. PyTorch is 2.14.0 rather than the paper's 2.11.0, deliberately, to measure
against the current production stack. Precision is FP16 only; the paper also measures FP8-E4M3.

GQA prefill at batch 8 and 32768 is not offered. Its query tensor is 4.3 GB, the comparison holds it in both layouts
beside a reference and a candidate output, and that does not fit the 32 GB this card has.

There are no fresh-process repeats, so this is an engineering comparison rather than a confidence-interval claim.

### Archive

`results_rtx5090x1.tar.gz` holds the per-setup JSON records, the per-setup logs, the setup status table, and the
system, version and input-digest provenance under `evidence/`. It was produced by driving the lane's own
`run_baselines.py` with the recipe's flags and environment, not by `emmy bench`, so it carries no per-row experiment
records; `evidence/provenance.txt` says so.
