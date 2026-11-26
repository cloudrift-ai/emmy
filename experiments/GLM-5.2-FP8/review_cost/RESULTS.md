# GLM-5.2 FP8 synthetic review costs on eight H200 GPUs

The confirmed GPU-only configuration serves **106,406 total input tokens/s**, split into **97,023 cached** and
**9,383 uncached**, across 384 complete median-shaped reviews. At the regional public node price, that is
**997.55 reviews/hour and $0.058066 per review**, including cold fill and drain. Its wider verified full-batch window
serves 104,904 input tokens/s. The final measured run is `2026-10-24_02-50-15`; all 4,608 calls succeeded.

The selected configuration is TP1/DP8 with EP8, three-token MTP, a 2,048-token scheduling budget, 0.95 memory budget,
and 64 concurrent reviews. This is the confirmed choice from a bounded search, not a global hardware optimum.
Native RAM caching preserves reuse when concurrency exceeds GPU cache capacity, but does not demonstrate a sustained
throughput advantage over this setting. RAM c96's complete-window rate is only 0.27% higher, while its planned
full-batch input rate is lower. That difference does not justify claiming a new winner from the shorter four-wave run.

## Question and scope

Can a fully utilized eight-GPU node serve growing code-review sessions economically compared with the current
DeepInfra route through OpenRouter? The user prioritizes completed throughput, with no TTFT constraint. Report total
input tokens per second first, then cached and uncached components, completed reviews per hour, and cost per review.

The optimized workload is median-shaped: each review makes twelve sequential calls, with a mean prompt length of
32,000 tokens and exactly 330 generated tokens per call. This is 384,000 input and 3,960 output tokens per review.
It has an input/output ratio of about 97:1. The separately reported 130:1 aggregate ratio cannot simultaneously be
represented by those median inputs and outputs. Earlier p90 and 130:1 scenarios remain in the evidence, but later
optimization follows the user's explicit median-only scope.

Prompts are synthetic tokenizer IDs, not real source code. Reviews share 256 prefix tokens. Each new turn appends
512 fresh input tokens to the previous prompt and its exact generated tokens. Initial length centers the growing
context at the stated mean. Sessions have no tool wait and stay on one data-parallel rank. Healthy points achieve
about 91.18% cache reuse; the user's 92% traffic observation is a comparison target, not an injected hit rate.

Cold first calls and the final drain are included in the complete-window rates. Fresh seeds prevent a later point
from inheriting an earlier point's prefix cache. Random prompts do not reproduce code-review quality or real MTP
acceptance, so this experiment cannot establish recall preservation for either FP8 or NVFP4.

## Platform and execution

The GCP Spot node is an a3-ultragpu-8g in asia-south1-b. Inspection identifies exactly eight NVIDIA H200 GPUs, each
with 143,771 MiB visible memory and a 700 W power limit. The host has 224 logical CPUs, two NUMA nodes, and about
2.84 TiB of OS-visible RAM. The driver is 580.178.04. The system record reports Ubuntu 24.04.5 LTS and the host CUDA
compiler version 12.9.41; host toolchain versions are distinct from libraries inside the serving container.

The checkpoint is `zai-org/GLM-5.2-FP8` at revision `f33c6dc501ee5a2c7e35155653b1b1abbc320951`. Serving uses
`vllm/vllm-openai:v0.32.0`, FP8 KV, prefix caching, and initially three-token MTP. The baseline has TP1/DP8 with eight
expert-parallel ranks, a 2,048-token scheduling budget, and 0.95 GPU-memory utilization. The image digest is
`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`. A live package probe records
vLLM 0.32.0, PyTorch 2.13.0+cu130, Triton 3.7.1, FlashInfer 0.7.0.post1 and Transformers 5.19.0. These probes are
retained separately from the generic host system record.

The rental controller owns one VM throughout same-node phase transfers. Its parent is held while the benchmark child
finishes, then ownership is transferred before another phase starts. VM startup and experimental tuning time are not
charged to the full-utilization steady serving rate, but remain part of experimental spend.

## Completed exploratory measurements

The following points each complete 128 reviews. These are screening results; the six-wave confirmation is separate.
Rates are rounded; raw client usage and elapsed time determine the exact values.

| Configuration | Sessions | Total input/s | Cached input/s | Uncached input/s | Cache reuse |
| --- | ---: | ---: | ---: | ---: | ---: |
| TP1/DP8, MTP3, 2,048 tokens, corrected admission | 64 | 112,429 | 102,504 | 9,925 | 91.17% |
| TP1/DP8, MTP3, 2,048 tokens, corrected admission | 72 | 104,141 | 91,365 | 12,776 | 87.73% |
| TP1/DP8, MTP off, 2,048 tokens, corrected admission | 80 | 100,633 | 91,958 | 8,675 | 91.38% |
| TP2/DP4, MTP3, 2,048 tokens, corrected admission | 48 | 94,953 | 86,580 | 8,373 | 91.18% |
| TP2/DP4, MTP3, 4,096 tokens, corrected admission | 32 | 73,129 | 66,680 | 6,449 | 91.18% |
| TP2/DP4, MTP3, 4,096 tokens, corrected admission | 48 | 84,602 | 77,141 | 7,460 | 91.18% |
| TP1/DP8, MTP3, 0.96 memory budget, earlier admission | 72 | 117,306 | 106,317 | 10,989 | 90.63% |
| TP1/DP8, MTP3, 0.96 memory budget, earlier admission | 80 | 87,990 | 72,768 | 15,222 | 82.70% |
| TP1/DP8, MTP off, 2,048 tokens, earlier admission | 64 | 102,851 | 93,985 | 8,867 | 91.38% |

The earlier refinement's c64 point is faster than the corrected baseline: 417.410 seconds versus 437.183 seconds for
the same review count. It serves 117,755 total input tokens/s, with 90.96% reuse. Other independent MTP3 c64 runs take
441.408 and 443.655 seconds. The approximately 6% range across these short runs includes seed and admission-policy
differences; it is not a confidence interval. Do not select only the fastest run for the final cost claim.

The initial concurrency sweep reaches a clear cache-retention limit. At c96 and c128, cache reuse falls to 23.65%
and 0.89%, with complete review durations of 1,430.137 and 1,737.979 seconds. Higher fresh-token throughput caused by
recomputing evicted prefixes does not mean more useful review throughput. The selected configuration preserves reuse
instead of maximizing recomputation.

### Session admission correction

Earlier phases chose the rank by session index before admission. Uneven completion could concentrate newly admitted
reviews on one rank. During an earlier global-c64 point, one rank reached twelve active or queued requests, and 26 of
87 metric scrapes showed a rank above eight. From the tuning phase, admission chooses the least-active rank and rotates
ties, while each review remains sticky. The repeated c64 point keeps every rank at or below eight in all 86 scrapes.

The corrected baseline is close to the earlier prefill baseline with matching prompts. Balanced admission is verified,
but these measurements do not show a large throughput gain from routing alone. Each phase's retained client source
documents the policy change.

### Memory and failed controls

At TP1/DP8, MTP3 with an 8,192-token scheduling budget cannot allocate KV blocks. The 16,384-token attempt also fails
engine initialization, but its truncated native stack does not establish the exact cause. Disabling MTP does not make
the 8,192-token deployment viable at the configured context length: initialization reports 2.67 GiB available KV,
while a 98,304-token context needs 4.94 GiB. Its estimated maximum length is 53,056 tokens. These are deployment
failures, not zero-throughput points.

TP2/DP4 at 2,048 tokens reduces model allocation to 100.08 GiB per GPU and leaves 26.59 GiB for KV. Logical capacity is
521,600 tokens per DP engine, or 2.0864 million across four engines. This holds the late contexts at c48 but not c64.
The c64 point is deliberately stopped after 472.54 seconds, with 704 of 1,536 requests in the last snapshot and 65.85%
reuse. It already exceeds the corrected TP1/DP8 c64 complete duration, so it cannot beat that point's complete-window
throughput. The partial log and stop note are retained; this is not a completed c64 measurement.

The earlier MTP-off c96 point stops at 1,408 of 1,536 requests after 778.99 seconds, with 66.85% reuse. Its c128 point
is unmeasured. Before the scope narrowed, the initial p90 sweep completed 128 reviews at c16 and c32 in 2,545.26 and
1,949.45 seconds; the 130:1 sweep completed the same review count in 1,025.25 and 706.07 seconds. Both c64 points
were stopped with incomplete calls and no completed reviews. These workloads are retained evidence outside the
median optimization claim. The later p90 and 130:1 follow-ups stopped after the user's scope change. Their partial
work, completed lower points, failed row status, and explicit reasons remain in the archive.

## Utilization evidence

Continuous one-second GPU queries and timestamped GPM samples align with client start/end timestamps. Actual GPM
cadence is approximately 1.3–2 seconds. Missing counters remain missing, including the initial short probes;
deployment and idle samples do not belong to measured-serving averages.

The earlier MTP3 c64 baseline averages 96.8–97.8% headline GPU activity but only 45.7–49.1% SM activity, 9.4–10.3% SM
occupancy, and 14.6–16.6% tensor activity. Power is 447–484 W per GPU. GPU activity can include communication kernels;
these counters neither establish achieved FLOPS nor demonstrate saturation of useful compute.

Request histogram deltas cover 1,532 of 1,536 calls in that window: mean prefill is 1.020 seconds, decode 15.022 seconds
and queue 1.438 seconds. These concurrent request times overlap, so they are not an exclusive division of GPU time.
Together with measured cache reuse, they argue against assuming that the large input/output ratio alone makes this
cached workload prefill-bound.

Turning MTP off at c64 raises SM activity to 54–55%, DRAM activity to about 34%, and power to 536–566 W, while completed
throughput falls. That comparison uses different seeds and the earlier admission policy, so it is directional.
Utilization itself is not the objective: completing more reviews at a lower node cost is.

TP2 c48 averages 49.5–52.1% SM activity, 11.3–12.3% occupancy, 16.3–17.6% tensor activity, and 468–505 W. Its additional
device activity likewise does not yield higher review throughput. Increasing the TP2 token budget to 4,096 retains
cache at c48 but reduces throughput from 94,953 to 84,602 input tokens/s with matched prompts. Its model allocation
is 100.51 GiB per GPU, with 22.75 GiB of available KV and 446,336 logical tokens per DP engine. The one-token MTP
control completed c64 in 495.47 seconds at 99,203 input tokens/s, below the matched-seed MTP3 reference. Its c80 point
was stopped after 456.46 seconds with 861 of 1,536 calls complete and 81.02% cache reuse to prioritize the requested
RAM comparison. The partial row and stop reason are retained without assigning it a completed throughput rate.

## RAM prefix caching

Native OffloadingConnector passed smoke and ran 384 reviews at c64 with 64 GiB of CPU cache per DP engine. The eight
shared mappings have a physical PSS total of 511.986 GiB. This is resident host storage; a zero Mlocked reading does
not establish whether CUDA registration pins those pages. Generated-token blocks are retained for the next call.

The completed six-wave point contains 4,608 successful calls, all returning 330 tokens, with 147,456,000 input and
1,520,640 output tokens. API cached usage is 134,748,160 tokens. Aligned server counters independently give
134,453,760 local cache hits plus 294,400 external cache hits, exactly reproducing the API value. RAM supplied only
0.200% of total input at c64. There were zero preemptions, 16.112 GB of reported loads and 770.784 GB of stores.

| Measurement window | Input tokens/s | Cached input/s | Uncached input/s | Review arrivals/hour | USD/review arrival |
|---|---:|---:|---:|---:|---:|
| All 384 reviews, 1,442.70 seconds | 102,208 | 93,400 | 8,808 | 958.20 | 0.060451 |
| 64th to 192nd completed review, 518.73 seconds | 103,902 | 95,474 | 8,428 | 888.33 | 0.065206 |
| 64th to 320th completed review, 1,016.74 seconds | 101,055 | 92,596 | 8,459 | 906.42 | 0.063904 |

The planned first-to-third-wave interior window contains 1,691 actual request completions and 128 review arrivals;
these are different populations because some work crosses its boundaries. Its mean active session count is 63.99971
of 64, with 99.9709% of elapsed time at the cap. The wider first-to-fifth-wave sensitivity window likewise stays at
the cap for 99.9714% of elapsed time. Both windows preserve measured handoff gaps, actual token completions and actual
review arrivals. Finite-window cost estimates differ from nominal token-work allocation; neither replaces the full
run's cost. The wider window was added after observing this boundary effect, and the original window is retained.

Whole-run GPM averages range across GPUs from 42.1–43.2% SM activity, 8.73–8.94% occupancy and 12.9–13.4% tensor
activity. During the wider interior window they are 40.7–41.7%, 8.47–8.67% and 12.1–12.6%. Server telemetry averages
59.8 running and 3.0 waiting requests in that interior. These samples do not demonstrate compute saturation.
The counters do not isolate prefill from decode.

This RAM c64 result does not demonstrate an improvement over the six-wave GPU-only confirmation below. Ordinary
prefix offload preserves reusable contexts but does not remove the GPU KV working-set requirement for simultaneously
decoding calls.

Native RAM c96 also completed all 384 reviews and 4,608 calls. This four-wave point finished in 1,382.10 seconds at
106,690 input tokens/s: 97,495 cached and 9,195 uncached. It delivered 1,000.22 reviews/hour at $0.057911 per review.
The cache fraction stayed at 91.38194%, but the source changed: 24,068,672 local hits plus 110,679,488 RAM-restored
tokens exactly match 134,748,160 API cached tokens. RAM supplied 75.06% of all input. There were 41 preemptions,
6.058 TB of reported loads and 768.882 GB of stores during the measured window.

Its first-to-third completed-wave interval lasted 765.60 seconds, with 103,860 input tokens/s: 95,214 cached and
8,646 uncached. The 192 actual review arrivals imply 902.82 reviews/hour and $0.064159 per arrival. Client sessions
averaged 95.99970 of 96, with 99.9701% of the interval at the cap. Server telemetry instead averaged 68.48 running
and 26.07 waiting requests. GPM SM activity was 38.7–39.9%, occupancy 8.05–8.30% and tensor activity 11.5–12.1%.
The full-batch input rate is essentially unchanged from RAM c64's planned interior window, despite many more host
cache restorations. This supports a limited conclusion: RAM preserved reuse beyond the GPU-only cache capacity,
but raising offered concurrency from 64 to 96 did not materially increase sustained throughput in these trials.
The GPU-only c96 control was stopped after exceeding the completed RAM point's duration. The final six-wave GPU-only
c64 confirmation has a slightly higher planned full-batch input rate than RAM c96. RAM c96 is retained as a four-wave
comparison, not a separately confirmed winner; the bounded search ends with GPU-only c64.

The GPU-only c96 control was stopped after 1,419.41 observed seconds. The final pre-stop capture contains 1,212
of 4,608 calls and 94 of 384 completed reviews, with 393,536 cached tokens out of 38,490,984 input tokens (1.02%).
It had already exceeded the RAM c96 point's complete 1,382.10-second duration, so this point could no longer beat
that complete-window throughput for the same review count. The partial log and explicit stop note are retained
without assigning a completed GPU-only throughput or cost. Different seeds limit a causal RAM-only speedup claim.

The measured c96 comparison has different seeds: increasing run length changed the RAM ladder's second seed to
8,007,873, while its already expanded GPU-only row retained 8,004,801. Both original records are preserved. The
durable recipe now aligns the seeds for future runs. RAM c64 overlaps the earlier baseline's first 128 reviews;
the final GPU-only c64 confirmation uses a new seed and 384 reviews.

## Six-wave GPU-only confirmation

The final run uses fresh seed 28,001,729. It completes 384 reviews in 1,385.791910 seconds: all 4,608 calls return
exactly 330 tokens, every review has twelve sequential turns and 384,000 total input tokens, and each review keeps
one DP rank. Total usage is 147,456,000 input, 134,453,760 cached input and 1,520,640 output tokens, with 91.18229%
cache reuse. The server's local-cache and fresh-input counters exactly reproduce the API totals. Scrapes cover
4,607 of 4,608 response completions and show no preemptions; the client records cover every completion.

| Window | Seconds | Input/s | Cached/s | Uncached/s | Review arrivals/hour | USD/review arrival |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Complete 384 reviews | 1,385.79 | 106,406 | 97,023 | 9,383 | 997.55 | 0.058066 |
| 64th to 192nd completed review | 457.04 | 105,166 | 95,908 | 9,258 | 1,008.22 | 0.057451 |
| 64th to 320th completed review | 975.99 | 104,904 | 95,943 | 8,961 | 944.27 | 0.061343 |

The planned interior contains 1,499 actual request completions and 128 review arrivals. The wider interior contains
3,202 calls and 256 arrivals. Both keep mean active sessions at 63.99970 of 64, with about 99.970% of elapsed time at
the cap. These counts use the first request start and last response end as observed session boundaries, retaining
logging and semaphore handoff gaps. Token completions and review arrivals cross the window boundaries differently;
the change in arrival cost despite nearly unchanged input throughput demonstrates this finite-window effect.
The complete-window cost is the primary aggregate cost estimate.

During the wider interior, server telemetry averages 61.80 running and 0.95 waiting requests. Per-GPU GPM averages
are 40.8–42.1% SM activity, 8.60–8.85% occupancy and 12.1–12.6% tensor activity. Complete-window means are
42.3–43.7%, 8.87–9.12% and 12.8–13.4%, respectively. These counters do not demonstrate compute saturation or isolate
prefill from decode. The completed run is about 5.4% slower than the shorter corrected c64 screen, so the final
cost uses this long result rather than the fastest exploratory point.

The exact live client command and source were captured before its temporary container disappeared. Source SHA256 is
`cd4f7618957fdd0c792169dc1db6f8daf25805f1d533f706dfed333aba9cbc1b`. Although the command's base configuration allows
128 sessions, its actual point start verifies the concurrency ladder selects 64. Historical revision snapshots and
phase dry-run logs supplement this provenance; three historical records were dirty, so their revision bytes alone
are not proof of the exact executed client bytes.

## Cost model

The retained regional public-price snapshot totals $57.923968402 per hour: $57.481855562 for the Spot node, $0.227398
for 2,000 GiB Hyperdisk Balanced capacity, $0.082188 for extra IOPS, $0.13002684 for extra throughput, and $0.0025 for
the external IPv4 address. The node component includes the eight GPUs, CPU, RAM, and bundled local SSD. This is a
regional public-price model, not an invoice. It excludes egress, taxes, contract discounts, and operational overhead.
Official snapshots were refreshed at 01:37 UTC on October 24 with unchanged prices. Sources are the GCP Spot VM,
disk, and external-IP pricing pages. The disk selection excludes similarly named replication-protection charges.

The retained OpenRouter endpoint snapshot lists regular DeepInfra FP4 rates of $0.5625/M uncached input, $0.105/M
cached input, and $1.80/M output. The flex route is separate. Do not apply another advertised discount to already
discounted endpoint rates, or assume the hosted checkpoint and review quality match either local model.

The relay accounting rule caps cached input at total input, subtracts cached input from total input, then adds the
three charges. A mixed serving measurement gives one total-cost constraint; it cannot identify three independent
marginal token prices. One explicit allocation is to preserve the provider's relative prices and multiply all three
by the same factor until the observed token mix pays exactly one node hour. Those are inferred allocated rates.

For the confirmed run, the actual mean review contains 350,140 cached input tokens, 33,860 uncached input tokens,
and 3,960 output tokens. The provider reference charges $0.06293895 for that mix, versus the node's measured-workload
cost of $0.05806609. Multiplying all three provider rates by 0.922578 allocates exactly one node hour across the
observed token mix; it does not measure each token category's independent marginal cost.

| USD per million tokens | Uncached input | Cached input | Output |
| --- | ---: | ---: | ---: |
| Regular DeepInfra FP4 reference | 0.562500 | 0.105000 | 1.800000 |
| Inferred H200 allocation, provider relative prices preserved | 0.518950 | 0.096871 | 1.660640 |

At 260 median-shaped reviews per day, the confirmed full-utilization allocation costs $15.10 and uses about 0.261
node-hours. The provider reference costs $16.36 at the measured reuse fraction. A continuously rented node instead
costs $1,390.18/day, or $5.35 per review if these 260 reviews are its only traffic. Startup and operational overhead
make a short daily rental different from allocating a share of an already busy node. Medians are not traffic averages.

At the user's separate 92% cache target, the provider reference costs $0.0615024 per review, or $15.99/day. Applying
the same inferred node rates gives $0.0567408 per review, or $14.75/day. This is a counterfactual accounting mix, not
measured performance or a prediction that raising cache reuse changes node time by that amount. The always-on node
price is unchanged. Synthetic serving measurements cannot validate recall or equate the hosted FP4 and local FP8
models' review quality.

Prices were verified against the [GCP Spot pricing page](https://cloud.google.com/spot-vms/pricing),
[Hyperdisk pricing](https://cloud.google.com/compute/disks-image-pricing),
[external IPv4 pricing](https://cloud.google.com/vpc/network-pricing), and the
[OpenRouter endpoint response](https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints).
The archived source pages and endpoint response retain the October 24, 01:37 UTC snapshot.

## Retained evidence

The named Git LFS archive is `results_h200x8.tar.gz`, rooted at the latest run `2026-10-24_02-50-15/`. Its final
record and client log are `h200x8_np4608_s28001729_sc64_ccgpde0_pconfirmation_smedian_846aaa8bb760.experiment.yaml`
and the matching `.benchmark.log`; the matching `.server.log` retains deployment and serving output. Earlier raw
invocations are nested under `earlier-runs/<timestamp>/`. All original system records are retained except address
redaction. Their statuses include deliberate partial stops, not just engine failures.

| Invocation | Original row status | Coverage |
| --- | --- | --- |
| 2026-10-23_16-39-56 | 3 interrupted `running` records | Model download interrupted; no client measurements |
| 2026-10-23_16-45-31 | 1 succeeded, 2 failed | Initial median and earlier p90/ratio sweeps, including partial points |
| 2026-10-23_20-50-40 | 1 succeeded, 2 failed | Median refinement; later p90/ratio work stopped after scope change |
| 2026-10-23_21-37-07 | 1 succeeded, 2 failed | Prefill budget comparison and initialization failures |
| 2026-10-23_22-06-45 | 1 succeeded, 2 failed | Memory/MTP controls with partial and unmeasured higher points |
| 2026-10-23_23-12-43 | 3 succeeded, 2 failed | Corrected admission, TP2 and MTP1 comparisons |
| 2026-10-24_01-18-26 | 1 succeeded, 1 failed | RAM c64/c96 complete; GPU c96 intentionally stopped |
| 2026-10-24_02-50-15 | 1 succeeded | Fresh six-wave GPU c64 confirmation |

The interrupted 16:39:56 invocation preserves its original `running` records: median reached deployment, while p90
and ratio130 remained in staging. The rental log records cancellation and KeyboardInterrupt before the next
invocation on the same node. These are historical incomplete records, not live jobs. The separate interruption
note explains the missing terminal transitions; no terminal records were manufactured retrospectively.

The archive also retains timestamped `telemetry/gpu.csv`, `telemetry/gpm.csv`, `telemetry/server-metrics.jsonl`,
`telemetry/host-memory.jsonl` and complete per-container server logs. `supporting/` contains completed-window JSON
summaries, stop notes, live source capture, software probes, host mappings and rental output, including
`rental-initial-run.log`. `client-source-snapshots/` holds revision sources and their provenance index. `pricing/`
holds original and refreshed public-price snapshots. The integrity list records the sanitized archived bytes;
original local raw files remain unchanged. Missing counters remain missing rather than zero.

The owned H200 VM was deleted at 03:27 UTC on October 24. The lease is marked deleted, a fresh provider query returns
no matching instance, and all five telemetry collectors and the rental controller have exited. Archive verification
at 03:30 UTC checks every archived file against its sanitized source and all 23 system records. The archive is
58,732,246 bytes with SHA256 `cc214e9ef86c8ace6e198611d2de6af5684d8029c7b1c530837a8585531f572e` and uses Git LFS.

The separate NVFP4 experiment retains its own measurements, failures, archive and interpretation. Hardware and
checkpoint precision differ, so cross-platform results are serving-cost comparisons rather than controlled
quantization-quality tests.
