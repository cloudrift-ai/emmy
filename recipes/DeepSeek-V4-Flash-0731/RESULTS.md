# DeepSeek V4 Flash 0731 on 16× V100 SXM3 32 GB

Status: serving-qualified with the Emmy serving image pinned by the recipe, at a 131,072-token context. Qualified
2026-10-07 on 16× V100 SXM3 at repository revision `4f9dba86d`. Until then the recipe pinned the plain 1Cat/vLLM image
at the checkpoint's full 1,048,576-token context. That configuration's measurement stays in
`experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3`, and `experiments/DeepSeek-V4-Flash-0731/emmy_ab_v100_sxm3`
compares the two images at one envelope.

## Qualified deployment

| Item | Value |
| --- | --- |
| Model | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| Model revision | `7872f01b1d1fe23eabc4c98b48bffcef5a386062` |
| Hardware | 16× Tesla V100-SXM3-32GB, compute capability 7.0, 12 NVSwitches |
| Driver / toolkit | 580.159.03 / CUDA 12.9.1, nvcc 12.9.86 in the image |
| Engine | 1Cat/vLLM `1.2.3.dev87+gd76126608.d20260810` with the Emmy plugin at `992de5c80` |
| Image | `cloudriftai/vllm-emmy-deepseek-v4-flash-0731:1.2.3-992de5c8` |
| Image digest | `sha256:52a11f6c297d200d12f9c2262b9684792797b71aa509dff5bdd4ca94b2b9424b` |
| Runtime base | `cloudriftai/1cat-vllm-deepseek-v4-flash-0731@sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1` |
| Serving shape | TP8, PP2, context 131,072, memory share 0.80, concurrency 8, FP8 KV cache, prefix cache on |

The Emmy plugin runs compiled kernels for the hyper-connection stream mixing, the norms and the shared and routed
experts, and hosts the fork's attention per layer. Decode steps of up to 8 rows are captured as CUDA graphs. Every
kernel is deployed from the recipe's golden under strict evidence, and the image starts from its baked execution-plan
packs with no compile.

The image was warmed and verified at a 4,096-token context. Its packs are keyed on the prefill step and the decode
bucket, not on the context length, so they hit at this context too: every boot of this envelope reports a pack hit on
all 16 workers and no new kernel compile.

Two switches keep greedy output stable from boot to boot. The recipe turns the fork's MXFP4 small-shape timing
selection off, and the image turns its fp8 one off.

### Host prerequisite: NVIDIA Fabric Manager

These 16 GPUs sit behind 12 NVSwitches. Until Fabric Manager trains that fabric, `nvidia-smi` lists all 16 GPUs while
every engine worker dies at `cudaGetDeviceCount()` with `error 802: system not yet initialized`, which reads like an
engine or model fault rather than a missing host service. A freshly provisioned host for this run had no Fabric Manager
installed and could not deploy at all. `emmy deploy` now installs and starts it automatically on NVSwitch hosts, pinned
to the running driver's exact version; no recipe change is required. Anyone deploying this recipe outside Emmy must
ensure `nvidia-fabricmanager` matching the driver is running first.

## Best recipe performance

Measured 2026-10-07 with the recipe's exact engine block (run `20261007T054136Z`, the experiment in "Reproduce").
Greedy decoding with ignored EOS, three client repeats per point against a fresh deployment, every request completed.
Spread is the sample standard deviation across the three repeats.

| Point | Metric | Three-repeat mean ± standard deviation |
| --- | --- | ---: |
| One request, 2,048 in / 128 out (24 requests) | Median TTFT | 3,040 ± 4 ms |
| | Mean TPOT | 120.25 ± 0.01 ms |
| | Output throughput | 7.10 ± 0.00 tokens/s |
| 8 concurrent, 1,024 in / 64 out (96 requests) | Output throughput | 21.86 ± 0.17 tokens/s |
| | Total token throughput | 371.7 ± 3.0 tokens/s |
| | Mean TTFT | 7,826 ± 478 ms |
| | Mean TPOT | 246.9 ± 10.2 ms |
| | Median ITL | 194.95 ± 0.09 ms |

Prompts are random and share nothing, so the prefix cache serves only the client's own warm-up prompt: one prompt in 8
at the first point, which is why that point reports its median, and one in 32 at the second. Model load and warm-up
takes 340 to 352 s, 214 s of it loading weights. Eight of the fork's Triton kernels compile once during the first
requests and then stay cached.

Against the plain fork at one envelope (context 4,096, prefix cache off, six repeats per arm): this image reaches the
first token of one 2,048-token request in 0.80× the fork's time and takes 0.81× its time per output token, and it
delivers the same throughput at 8 concurrent requests. It takes 2.2× as long to start.

With prompts that share a prefix the cache does what the shared part allows: with three quarters of each prompt
shared and cached, the time to first token falls to 0.37× and 8 concurrent requests deliver 56% more output tokens
per second (`experiments/DeepSeek-V4-Flash-0731/emmy_prefix_cache_v100_sxm3`).

## Context, memory and accuracy

Probes on the host on 2026-10-07, outside the benchmark archive. Unless a row says otherwise they ran on a deployment
of this recipe's envelope.

**Capacity.** The engine allocates KV capacity for 1,491,482 tokens on the first pipeline stage and 1,527,439 on the
second, 11.4× and 11.7× the full context. A 130,711-token prompt completes with HTTP 200 in 124 s. A prompt past the
limit is refused with HTTP 400 and the server stays up.

**Memory, and why the share is 0.80.** A card holds 27,724 to 27,832 MiB of its 32,510 when idle. During a long
prefill the allocator climbs to the top of the card and releases: 32,250 MiB at the peak of the full-length prompt,
31,130 MiB with eight concurrent 30,000-token prompts, no error. At the fork's share of 0.90 there is no such room. A
6,925-token prompt then fails with an out-of-memory error on the first-stage cards, in the routed-expert combine (194
MiB requested, 160 MiB free), and the engine dies. Emmy's prefill needs about 2 GiB of working memory per card that
vLLM's KV sizing does not account for.

**Why the context is 131,072.** The image also boots at the checkpoint's 1,048,576 with this memory share, with KV
capacity for 3.6M tokens. But memory use grows with prompt length: a 262,845-token prompt completes in 224 s with the
fullest card at 32,414 of 32,510 MiB.

**Recall over long prompts.** Each prompt is repository documentation with one sentence a third of the way in that
states a four-digit code; the question asks for the code. The fork ran from its former recipe, at a 1,048,576-token
context and a 0.90 share.

| Prompts | Emmy image | Plain fork |
| --- | --- | --- |
| 6,925 / 28,658 / 115,199 tokens, one at a time | exact; 10.7 / 22.7 / 85.7 s | exact; 8.6 / 23.9 / 91.6 s |
| 130,711 tokens, the full context | `7319` for 7391; 124 s | exact; 112 s |
| 262,845 tokens | wrong; 224 s | `7319` for 7391; 235 s |
| eight of about 15,000 tokens, one at a time | 4 of 8 exact | 4 of 8 exact, the same four |
| eight of about 15,000 tokens, at once | 3 of 8 and 5 of 8 in two runs | 4 of 8 |
| eight of about 30,000 tokens, at once | 6 of 8 exact | 7 of 8 exact |

The Emmy rows up to 115,199 tokens and at 262,845 ran on a 1,048,576-context boot of this image at share 0.80. A
miss is almost always one digit off (`4815` for 4814, `6042` for 6040). Run one at a time, the two images miss the
same four prompts, three of them with the same wrong digits. So exact recall of a number from far back is a limit of
the model, not of either runtime, and the two are level on it.

**Capabilities.** A weather question with a tool defined returns a structured call,
`get_weather({"city": "Paris", "unit": "celsius"})`, and a greeting with the same tool defined returns plain text.
Reasoning is separated only for a request that sends `chat_template_kwargs: {"thinking": true}`: a multiplication then
returns 74 characters of reasoning and `391`. Without it the image answers `323` for 17 × 19 and `Paris` for the
capital of France cleanly, and `153` for 17 × 23, which is wrong; the fork's answers to the same three carry stray
`</think>` markers.

**Quality.** GSM8K, 200 questions through the chat endpoint at seed 0, scores 0.955 by flexible extraction and 0.73
by strict match at this envelope, the image's scores at its warm envelope. The fork as shipped scores 0.975 and 0.91;
its higher strict match comes from an fp16 overflow in its prefill kernels, and with that fixed it scores 0.96 and
0.755 (`experiments/DeepSeek-V4-Flash-0731/emmy_ab_v100_sxm3`).

**Prefix cache.** A cache hit gives exactly the output of a prefill split at the cached boundary, on this runtime
and on the plain fork. It is not bit-identical to a prompt prefilled in one step, and the wording of an answer can
differ between the two.

## Compiler qualification

The committed golden is now the serving-twin inventory: the 152 kernels the TP8 × PP2 server compiles, at widths 1,
16 and 4096 and the dynamic width. The per-layer coverage, verification and tuning notes below describe the earlier
279-target per-layer file it replaced.

### Golden refresh after CSE (2026-09-30)

The V100 SXM3 golden was remeasured after #981 with strict correctness and 10 warmups / 100 timed iterations. These
are kernel-level route measurements, not serving throughput. Each selected route also replayed with strict evidence
from a fresh tune database. The previous column is the route row measured before #981, under its earlier compiler and
measurement conditions.

| Route | Previous golden | Current whole program | Result |
| --- | ---: | ---: | --- |
| Dynamic pre-attention mean/reduce | 4,590 µs | 312 µs | A four-kernel cut and `WORK=t32x4` recover the route. |
| Dynamic post-attention matmul/reduce | 442 µs | 461 µs | `WORK=w2x2` on the first MMA is best measured; 4% slower. |
| 4,096-token post-attention matmul/reduce | 2,999 µs | 3,359 µs | Prior schedule is best of the tested variants; 12% slower. |

The 4,096-token pre-attention mean/reduce route runs in 2,906 µs with three kernels after choosing its own two cuts
and changing the large child's tile to `f2x2`. A replay bug had combined cuts from sibling routing rows, producing a
five-kernel route with a 22.5 ms child and about 25 ms whole time. The target Loop body did not change across #981;
only its input order changed. Its old 2,719 µs nested route row is not a whole-program timing, so it is not used as a
whole-program comparison. The older unmeasured route rows remain in the golden as proposals.

The single-token post-attention matmul route remains unpromoted. Its old measured row was 180 µs. With the same current
compiler, card and tune database, its pre-#981 and current Loop input orders replay at 274 and 275 µs; an earlier
current-target run under different measurement conditions was about 297 µs. The final child is now about 187 µs,
against 64 µs in the old row. A different tile or staging choice was slower, one work choice and an added cut failed
strict correctness, and another cut pin did not realize. The old rows remain as proposals until a correct schedule
recovers their performance.

### Kernel reference numbers (2026-09-11)

Each recorded row of the golden carries a `latency` block for this card: Emmy, eager PyTorch where the kernel has a
PyTorch slice of the embedded program, and torch.compile where it agreed with eager. Measured on 8× Tesla
V100-SXM3-32GB (CUDA 12.9, torch 2.13.0+cu126) with `emmy run --golden <copy> --bench --record --bench-backends
eager,tcompile,emmy --warmup 3 --iters 10`, the file's own rows as evidence; 191 of 192 kernel-and-width rows record.
Eager here runs the golden's program through PyTorch op by op, so it is an upper bound on PyTorch time.

| Program | Width | Emmy, all kernels | Emmy / eager on kernels with a slice |
| --- | --- | ---: | ---: |
| Pre-attention | 1 / 16 / 4096 / dynamic | 29.7 s / 17.2 s / 1.94 s / 247 ms | 2,142× / 1,202× / 2.8× / 14× |
| Routed expert | 1 / 16 / 4096 / dynamic | 1.29–1.38 s / 1.2–4.8 ms / 108–200 ms / 7.8 ms | 51–55× / 0.05–0.19× / 3.7–6.9× / 0.31× |
| Post-attention | 1 / 16 / 4096 / dynamic | 76 ms / 68 ms / 692 ms / 286 ms | 1.0× / 0.03× / 1.2× / 0.09× (on 20 of 36 kernels) |

Pre-attention at the decode widths and the single-token experts are where serving time goes. The post-attention
normalization loop is 31× slower than torch.compile at one token. The large fused post-attention kernels, 99% of that
program's time at 4096 tokens, have no PyTorch slice because each computes part of an op, so they are not compared.

### Coverage

All 43 decoder layers reduce to three distinct traced graphs, set by `compress_ratios` in the model config. Tracing
seven layers and comparing Graph IR node counts closes the manifest empirically:

| Class | `compress_ratio` | Representative | Graph IR nodes | Verified identical |
| --- | ---: | --- | ---: | --- |
| layers 0–1 | 0 | layer 0 | 945 | layer 1 |
| even layers 2–42 | 4 | layer 2 | 1,156 | layers 4, 42 |
| odd layers 3–41 | 128 | layer 3 | 1,087 | layer 41 |

Layer 41 and layer 42 are `dspark_target_layer_ids`, and they trace identically to their ordinary siblings, so the
dspark specialization is not visible as a distinct architecture path. The committed golden's fourth representative
(layer 4) is redundant with layer 2. Non-layer seams are covered by the `model-seam` targets.

One path is **not** covered: the model declares `num_nextn_predict_layers: 1`, but the MTP head is not exposed as a
decoder layer — `emmy trace --layer 43` fails with "layer 43 not found (model has 43 layers)" — so it cannot be
traced through this interface. The committed golden does not contain it either.

A whole-model architecture trace is not bounded on this checkpoint: it grew past 830 GiB of host RAM, climbing about
45 GiB/minute, before it was stopped. Per-layer tracing is bounded (about 1 GiB), which is why the inventory is built
from representatives. Per-layer tracing is nonetheless dominated by merge-region dependency resolution in the Loop
splicer (`ir/loop/splicer.py`, `_ensure_dep` under `build_merged_region`); three representative layers ran 4h45m in
that phase without emitting a post-fusion inventory, confirming the 2026-08-11 report's attribution. The tuning below
therefore ran against the committed inventory rather than a freshly re-derived one.

### Verification on the target GPU

| Gate | Checked against | Result |
| --- | --- | --- |
| Repository-level validation | committed file | passes |
| Strict decode of every realization | committed file | 279 / 279 |
| Reconstruct and lower every target | Tesla V100-SXM3-32GB (sm_70) | 279 / 279, exit 0, 3 min 16 s |

### `REDUCE=g2a`: a validator false negative, now fixed and re-measured

The nine realizations the golden pinned to `REDUCE=g2a` all recorded worse-than-greedy numbers, and an exact `--ab`
pin reported `unreproducible pin: REDUCE=g2a realized (off)`, so the row would not bench at all. Neither symptom
meant what it looked like.

`g2a` decodes as a cross-CTA split of width 2 with atomic finalize, and it does realize: pinning it halves the
emitted K loop from 16,384 to 8,192, adds the partition axis and closes with `atomicAdd`. What changed is where the
receipt lives. #539 made a split mint brand-new kernels and had `knob.consume_kernel_row` strip their schedule row,
so no piece may carry the `g<n>` it came from — `test_split_fresh_kernels` asserts that outright. The partition is
recorded by the piece's sliced reduce axis, not by a stamp, exactly as a realized `PLACE` cut is. The
realized-vs-pinned gate reads only stamps, so it saw the pieces' `REDUCE=(off)` and called a realized pin dropped.
The golden was recorded 2026-08-11, seven days before #539, which is why its rows predate the problem.

The gate now skips the `g<n>` stage the way it already skips `PLACE`, and still gates the rest of the value. With
that, all eight surviving `g2a` rows bench cleanly (16/16 runs, no unreproducible-pin error) and were re-measured on
the target GPU. `g2a` and greedy are indistinguishable on every one of them:

| Realization | `g2a` (µs) | greedy (µs) |
| --- | ---: | ---: |
| `layer0.i003` | 913.4 | 914.4 |
| `layer0.i052` | 914.4 | 916.5 |
| `layer2.i003` | 914.4 | 914.4 |
| `layer2.i063` | 916.5 | 916.5 |
| `layer3.i003` | 914.4 | 916.5 |
| `layer3.i064` | 915.5 | 915.5 |
| `layer4.i003` | 915.5 | 916.5 |
| `layer4.i063` | 912.4 | 912.4 |

Each row keeps its `g2a` knobs — the configuration is real and ties with greedy — and carries the slowest of its two
runs on each side. Across the file, realizations materially slower than greedy fall from 41 to 11, and the remaining
eleven are sub-microsecond pointwise kernels.

### Tuning: equal-budget hybrid versus MCTS-only (2026-08-21, whole model)

The previous round had to scope its A/B to nine targets because `emmy tune` re-validated and rewrote the whole
document on every incremental persist. With that fixed upstream, both arms now sweep the **entire 279-target
inventory** at roughly 15 measured rows per minute against about 3.4 before — 1 h 18 m per arm instead of an
estimated ten hours.

Both arms started from the same inventory-only base (no knobs, timings, or ranking), an empty tune DB and online
prior, separate empty cubin caches, `--max-candidates 8`, `--patience 4`, `--seed 731`, all 16 GPUs, same compiler
revision, MCTS arm first. The hybrid arm added 18 knob proposals across 8 realizations, each reserving a candidate
slot before MCTS. The arms came out closely matched, which is what makes the comparison fair:

| Arm | Wall clock | Benches | Measured rows | ok / bench_fail | Prior calibration |
| --- | ---: | ---: | ---: | ---: | ---: |
| MCTS-only | 1:17:57 | 9,156 | 3,312 | 2,279 / 1,033 | +0.96 |
| Hybrid | 1:18:56 | 9,200 | 3,343 | 2,296 / 1,047 | +0.93 |

**Outcome: no winner, and nothing promoted.** Across 279 targets the O1 ranking lane put hybrid ahead on 44, MCTS
ahead on 44, and tied the remaining 169. The golden is unchanged by this round.

That result is less interesting than why. After the previous round resolved the `g2a` cluster, this golden loses
19.2 ms to greedy in total, and **19.2 ms of it sits in one realization** — `model-seam.k_linear_d74dc7` at 0.876×
(154.2 ms against greedy's 135.0 ms). Every other slower-than-greedy row is under 3 µs. It is also the last
`model-seam` row still on the Volta MMA warp tile, while its winning siblings all use a cooperative reduce over a
thread work inventory — the same swap that won 2.82× on `k_linear_db1eb0` last round. So the hybrid arm proposed
exactly that.

**Every candidate for that target was killed by a watchdog rather than measured.** In the tune lane the proposals hit
the 2.0 s GPU-time budget; only the alternative MMA work shape survived, at 176.2 ms, worse than the incumbent. The
O3 verification lane cannot settle it either: there the *greedy baseline itself* exceeds the 100 s wall budget and is
SIGKILL'd, so the run produces no comparable output at all. This kernel is simply too large for the budgets the
benchmark harness applies, and its 19.2 ms of headroom is currently unreachable by tuning — not refuted, unmeasurable.

**Bench failures.** About 31% of rows in each arm failed to bench, near-identically in both, so they do not bias the
comparison:

| Class | MCTS | Hybrid |
| --- | ---: | ---: |
| nvcc compile failed | 451 | 462 |
| bench worker exceeded the 16 s wall budget | 434 | 460 |
| benchmark run exceeded the 2.0 s GPU-time budget | 137 | 129 |
| hung kernel | 7 | 3 |

Every one of the nvcc failures is the same defect: generated CUDA for a `k_div_*` kernel emits `float v0 = in0 + in2;`
where `in0` was never declared, and nvcc rejects it (2,173 occurrences in the hybrid log, one kernel family, no other
compiler message). It reaches only searched variants — the golden's own recorded configurations compile and replay
cleanly — so it costs search coverage rather than deployed correctness. Knob values do not separate failing from
passing rows and all 462 failing rows are distinct ops, so the trigger is structural to that kernel family.

### Reproducing the compiler work

Emmy needs `nvcc` on `PATH`, and the CUDA **12.9** toolkit specifically: CUDA 13 dropped Volta. Two further host
conditions cost hours before they were found. PyYAML silently falls back to its pure-Python loader without `libyaml`,
which alone cost more than 13 minutes per parse of this 3.4 MB golden. And `torch 2.13` installs
`nvidia-cuda-nvrtc 13.0.88`, which `cupy-cuda12x` then resolves in preference to any CUDA 12 NVRTC; because CUDA 13
has no Volta support, every CuPy JIT path dies with `invalid value for --gpu-architecture` and all benchmarking fails.
Running with `LD_PRELOAD=/usr/local/cuda-12.9/lib64/libnvrtc.so.12` restores it.

## Reproduce

```bash
emmy bench experiments/DeepSeek-V4-Flash-0731/emmy_serving_v100_sxm3 --ssh <user>@<16x-v100-host>
```

The experiment's engine block is this recipe's. It runs two points, three client repeats each. Use `$run-experiment`
to retain the latest raw results, system-only experiment records, and factual artifact index.

## Limitations

The performance table covers two short-context shapes; long prompts are checked for capacity, memory and recall, not
for throughput. The recall, capability and quality probes ran once each on one boot, outside the benchmark archive.
The context stops at 131,072 tokens because of memory, so the checkpoint's 1M context is not served by this recipe;
the plain fork image serves it, and its record is `experiments/DeepSeek-V4-Flash-0731/serving_v100_sxm3`. A memory
share above 0.80 is unsafe: at 0.90 one long prompt kills the engine.
