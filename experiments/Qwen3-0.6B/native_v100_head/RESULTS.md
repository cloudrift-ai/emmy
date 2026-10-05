# Native output head on V100

## Result

On one NVIDIA Tesla V100 SXM2 16GB, cutting the native Qwen3-0.6B decode output head into its norm, scale, and
projection kernels reduced the captured fragment latency from 133,452 µs to 438 µs in the retained replay. The
reduction is 304× for this fragment. Three earlier strict tuned runs ranged from 441 to 447 µs. All runs passed the
same-input accuracy check with zero reported error. A fresh tuning database selected the committed route under
`--strict-evidence`.

This is a fragment result, not a serving result. The current attention fragment still costs about 9.8 ms unpinned,
and a sampled width-sixteen prefill fragment takes 18.8 ms against 0.22 ms for eager. No new native serving artifact
was qualified.

## Protocol

The workload is the decode-width-one output head traced from Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca` at a 4,096-token context. Both variants use the same traced
program, FP16 weights, standard math, and one V100. The host ran driver 580.178.04, CUDA 12.9, and PyTorch
2.13.0+cu126. The compiler source was main `d2c14aead` on October 5, 2026. Timings use five warmups and twenty
iterations of the captured whole forward, with `emmy run --bench --strict` and the same-input eager check.

The baseline golden has no measured placement route. The tuned golden records the two placement cuts at
`inner.1/map.3/map` and `inner.1/map.3/map.1/reduce`, followed by three measured kernel rows. Its projection uses
594 blocks and takes about 420 µs. The baseline fused kernel launches 4,861,952 blocks and takes about 133 ms.
Both goldens contain twelve traced native fragments (decode width one and prefill width sixteen); only the output
head route in the tuned golden has been measured and promoted. `emmy golden check` reports both files current.

The [recipe](recipe.yaml) replays the output head from both committed goldens on a prepared V100 checkout through
`emmy bench experiments/Qwen3-0.6B/native_v100_head --ssh USER@HOST`. It records fresh accuracy and timing output
in separate tuning databases. The tuned lane requires measured evidence. It does not trace a new program or search
for another schedule. The prepared checkout must have CUDA 12 and a PyTorch wheel with `sm_70` support.

| Run | Captured fragment latency (µs) | Accuracy |
| --- | ---: | --- |
| Retained unpinned baseline | 133,452 | pass |
| Retained strict-evidence route | 438 | pass |
| Earlier unpinned baseline | 133,433 | pass |
| Earlier cut candidate | 447 | pass |
| Earlier recorded route | 441 | pass |
| Earlier fresh strict-evidence replay | 442 | pass |

## Remaining gap

The current width-one attention fragment passed its accuracy check but took 10,397 µs in the backend timing
(9,781 µs for the greedy whole program). Its dominant split partial kernel took about 9,292 µs. An older 45 µs
attention route addressed a seam that no longer exists in the fresh lowering. Candidate cuts failed pin integrity,
exceeded the two-minute compile bound, or produced a kernel that exceeded the 2,000 ms watchdog. None was recorded.
Attention and the unmeasured prefill fragments must be resolved before this golden can qualify a serving artifact
or support a V100 throughput claim. The earlier native golden in `paged_cache` uses a retired format and is not a
current replay source.

A separate strict run of the width-sixteen pre-attention fragment passed its eager accuracy check. Its captured
latency was 18,793 µs versus 220 µs for eager, about 85× slower. One projection kernel accounts for 15,591 µs.
This single fragment does not measure full prefill latency. It is a diagnostic run outside the two-lane recipe.

The first unpinned export on an older main revision spent over ten hours compiling attention and produced no
artifact. After updating to `d2c14aead`, the same fragment completed, exposing the high runtime cost above.
The rental was therefore used for a bounded output-head optimization and diagnostic attention runs. There is no
stock vLLM comparison from this round.

## Evidence

The retained run began at 17:23:08 UTC on October 5, 2026. Both rows succeeded:
`v100x1_lbaseline_d7a9999f039f` and `v100x1_ltuned_fdb1c5ce363d`. The platform archive
`results_v100x1.tar.gz` contains the timestamped directory, both system-only experiment records, raw `result.json`
files and logs, package freezes, GPU and compiler versions, source revision, and golden hashes. Its two
`*_result.json` members supply the retained latencies and accuracy checks. It contains no checkpoint weights.
The `diagnostics/` members retain attention and prefill probes from the same card and source; they have no
experiment-row record and are not part of the two-lane comparison.

Results are specific to this V100 and source revision. The two retained rows ran sequentially, baseline first, and
each has one measured run of twenty iterations. The earlier 438–447 µs tuned range gives a small repeat check but
does not measure across hosts or driver versions. A full-model numerical and serving check remains required.
