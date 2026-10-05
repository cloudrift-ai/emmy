# Native output head on V100

## Result

On one NVIDIA Tesla V100 SXM2 16GB, cutting the native Qwen3-0.6B decode output head into its norm, scale, and
projection kernels reduced the captured fragment latency from 133,433 µs to 441–447 µs across three strict runs.
The reduction is about 300× for this fragment. All three tuned runs passed the same-input accuracy check with zero
reported error. A fresh tuning database selected the committed route under `--strict-evidence`.

This is a fragment result, not a serving result. The other native fragments are not fully measured in this golden,
and the current attention fragment still costs about 9.8 ms unpinned. No new native serving artifact was qualified.

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

The [recipe](recipe.yaml) replays the output head from both committed goldens on a matching V100 through
`emmy bench experiments/Qwen3-0.6B/native_v100_head --ssh USER@HOST`. It records fresh accuracy and timing output
in separate tuning databases. The tuned lane requires measured evidence. It does not trace a new program or search
for another schedule.

| Run | Captured fragment latency (µs) | Accuracy |
| --- | ---: | --- |
| Unpinned baseline | 133,433 | pass |
| Cut candidate | 447 | pass |
| Recorded route | 441 | pass |
| Fresh strict-evidence replay | 442 | pass |

## Remaining gap

The current width-one attention fragment passed its accuracy check but took 10,397 µs in the backend timing
(9,781 µs for the greedy whole program). Its dominant split partial kernel took about 9,292 µs. An older 45 µs
attention route addressed a seam that no longer exists in the fresh lowering. Two attempted replacements either
failed pin integrity or exceeded the bounded compile window, so none was recorded. Attention and the unmeasured
prefill fragments must be resolved before this golden can qualify a serving artifact or support a V100 throughput
claim. The earlier native golden in `paged_cache` uses a retired format and is not a current replay source.

The first unpinned export on an older main revision spent over ten hours compiling attention and produced no
artifact. After updating to `d2c14aead`, the same fragment completed, exposing the high runtime cost above.
The rental was therefore used for a bounded output-head optimization and diagnostic attention runs. There is no
stock vLLM comparison from this round.

## Evidence

The committed goldens are the replay inputs. The platform archive retains the machine-generated benchmark JSON,
accuracy status, logs, software versions, and experiment records. It contains no checkpoint weights. Results are
specific to this V100 and source revision; a full-model numerical and serving check remains required.
