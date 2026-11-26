# GLM-5.2 NVFP4 cached review serving on 8×RTX PRO 6000

The strongest measured configuration is TP1/PP8 with MTP3, a 12/11/11/10/10/10/10/4 target-layer partition, and 104
concurrent median-shaped reviews. Its six-wave run completes 624 reviews in 4,498.24 seconds and serves 53,269 input
tokens/s: 48,572 cached and 4,697 uncached. At the public regional all-in node price of $13.73570/hour, that is 499.40
reviews/hour and $0.02750 per review. The wider fully supplied interior serves 57,929 input tokens/s and 536.10 actual
review arrivals/hour, or $0.02562 per arrival. The complete-window cost includes cold fill and drain.

This is a bounded search result, not a proof of a global optimum. The adjacent c96 six-wave run is 7.0% lower in
complete-window throughput. Both retain about 91.18% cached input. GPU activity reaches nearly 100%, but measured SM
activity, occupancy, and tensor activity do not establish compute saturation. The owned VM was deleted after the last
run, and provider cleanup was independently verified at 05:10 UTC on 2026-10-24.

Native RAM prefix caching works on TP4/DP2. Its six-wave c24 result serves 30,927 input tokens/s for $0.04737 per
review. Its c48 interval spends most additional offered concurrency waiting and is slower. RAM restores inactive
prefixes; it does not expand the resident active decode batch. The separate HiSparse active-KV experiment failed
during initialization and has no serving result.

## Question and protocol

The question is the serving cost of GLM-5.2 on eight GCP RTX PRO 6000 Blackwell Server Edition GPUs with prefix
caching and enough concurrent reviews to use the node efficiently. The user narrowed optimization to the median
scenario: 12 sequential calls per review, averaging 32,000 input and 330 generated tokens per call. Earlier p90 and
130:1 attempts are retained as historical evidence, not promoted into estimates of mean traffic.

Each synthetic review starts with 27,369 tokens and grows to 36,631. Every call appends the actual 330 generated token
IDs and 512 fresh random input tokens. Only 256 initial tokens are common across reviews. The model runs greedy
completion requests with EOS ignored to make output length exactly 330. Calls within one review are sequential, and
there is no tool delay. Requests must return generated token IDs and actual cached-token usage. The initial cold
requests and final drain are included in the measured elapsed time.

Healthy median points achieve about 91.18% cache reuse. The user's 92% traffic figure is a target, not an injected
benchmark parameter. Every concurrency point uses a fresh random seed to avoid warming it with a prior point.
Different seeds and generated trajectories make comparisons empirical rather than bitwise identical.

Report total input throughput first, with its cached and uncached components. Cached input is useful served traffic,
but it does not represent repeated prefill computation. For complete fixed-shape runs, ranking by total input tokens/s
is identical to ranking by completed reviews/hour. Interior windows count actual request completions and review
arrivals separately, so their rankings can differ. More uncached work caused by eviction is not an improvement by
itself. There is no TTFT constraint.

## Hardware and serving configuration

The owned GCP spot VM is a g4-standard-384 in us-central1-b, with eight 97,887 MiB GPUs, 384 vCPUs, and 1,440 GiB host
RAM. The GPUs have a 600 W power limit and driver 580.178.04. GPUs 0–3 belong to one NUMA group and GPUs 4–7 to the
other. The node uses PCIe peer-to-peer communication and has no NVLink. Storage is a 2,000 GiB Hyperdisk Balanced disk
provisioned for 15,000 IOPS and 2,400 MiB/s.

The checkpoint is [NVIDIA's GLM-5.2 NVFP4](https://huggingface.co/nvidia/GLM-5.2-NVFP4), pinned to
`53e0691e21895a3863a606dfd12910c69eba94ab`. Its safetensors files total 464,823,042,096 bytes. Routed experts use
NVFP4; other tensors retain FP8 or BF16 precision. The MTP layer alone occupies 18.54 GiB in the checkpoint.

The serving image is `vllm/vllm-openai:v0.32.0`, with image digest
`sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34`. It uses native FlashInfer CUTLASS NVFP4 MoE
kernels and FlashInfer sparse MLA attention. KV cache is FP8. The baseline enables prefix caching and MTP with three
speculative tokens. The image contains Torch 2.13.0+cu130, FlashInfer 0.7.0.post1, Triton 3.7.1, Transformers 5.19.0,
and compressed-tensors 0.17.0.

The final PP8 runs use TP1, PP8, DP1, the V2 model runner, a 98,304-token context limit, 0.95 GPU memory utilization,
128 maximum sequences, and a 2,048-token prefill budget. MTP uses three draft tokens. The explicit target-layer
partition is 12/11/11/10/10/10/10/4. These settings and actual image/source identity are retained with each final row.

FlashInfer autotuning is disabled after an initialization stall. Worker samples showed one rank inside a router GEMM
while another waited in a tuning collective. That is evidence of a stall, not proof of a particular deadlock. Setting
OMP_NUM_THREADS=1 resolved a subsequent weight-loading stall in an OpenMP tensor copy. Default kernel tactics may
leave performance below a deployment with a working autotuner.

## Exploratory observations

The original TP8 trial used 64 reviews per point. Completed points improved from 168.39 output tokens/s at concurrency
four to 253.66 at concurrency sixteen, while cache reuse stayed at 91.18%. Concurrency twenty-four lost nearly all
prefix reuse and was deliberately stopped. Four is a low-concurrency baseline, not a tuned result.

TP4/DP2 uses eight expert-parallel ranks with the allgather/reduce-scatter backend. Device inspection confirmed the
two four-GPU tensor groups stay within their NUMA groups. Its completed c8 and c16 points use 64 reviews each:

| Topology | Concurrency | Total input/s | Cached input/s | Uncached input/s | Reviews/hour | Cost/review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| TP4/DP2+EP | 8 | 22,152 | 20,199 | 1,953 | 207.7 | $0.06614 |
| TP4/DP2+EP | 16 | 28,362 | 25,861 | 2,501 | 265.9 | $0.05166 |
| TP4/PP2 | 8 | 19,319 | 17,615 | 1,703 | 181.1 | $0.07584 |
| TP1/PP8, MTP3 | 64 | 29,004 | 26,447 | 2,557 | 271.9 | $0.05051 |
| TP1/PP8, custom partition, MTP3 | 64 | 27,595 | 25,162 | 2,433 | 258.7 | $0.05309 |
| TP1/PP8, custom partition, MTP3 | 96 | 30,460 | 27,774 | 2,686 | 285.6 | $0.04810 |

The custom-partition c64 row uses 128 reviews; the preceding rows use 64. It completes 1,536 calls over 1,781.17
seconds with 91.1829% cached input. It does not establish a throughput gain from repartitioning. Its seed differs, and
sampled MTP draft-token acceptance is 77.80%, versus 84.55% in the earlier default-partition c64 window. These
different generated trajectories limit a causal comparison. The custom c96 point also completes 128 reviews with
91.1829% cache reuse, over 1,613.68 seconds. Its full-window input rate is 10.4% above custom c64. The c96 test admits
its last review at 706.33 seconds and then drains for 907.34 seconds, averaging only 61.18 client-active reviews over
the whole window. Custom c64 averages 55.37 active reviews and drains for 404.22 seconds. The concurrency cap alone
does not describe average utilization in these finite trials. The final confirmation therefore uses six complete
review waves and reports separately labeled interior windows alongside the complete-window result.

At TP4/DP2 c24, only 553 of 768 requests had finished after 898.43 seconds, longer than the complete c16 trial's
866.51 seconds. Its partial cache-hit rate was 79.65%. It could no longer beat c16 for the same work, so the client
was stopped. The failed row and partial evidence are preserved.

The original DP2 dispatcher assigned sessions by index modulo rank. Later code selects the least-active rank at
session admission, with rotating ties, while preserving affinity for the whole review. This prevents uneven session
completion from overloading one rank while another has room. The earlier DP2 cache-loss result therefore does not
establish the final concurrency bound under the corrected dispatcher. PP8 has only one DP rank and is unaffected.

TP4/PP2 passed the semantic smoke test and measured a complete c8 point. Higher PP2 concurrency was deferred to
prioritize the user's PP8 request. This is not evidence that higher PP2 concurrency is slower. The containing ladder
has a terminal failed status with the completed c8 result retained separately.

## Memory and utilization

TP8 allocated 57.15 GiB of model memory per GPU and exposed 578,432 logical KV tokens. TP4/DP2 allocated 66.09 GiB per
GPU and exposed 401,984 logical tokens per DP rank, or 803,968 across the node. These capacities explain why adding
concurrent long reviews can destroy cache reuse before it improves useful throughput.

TP4/PP2 exposes 1,045,888 logical KV tokens. Its first stage allocates 51.39 GiB of model memory per GPU, while the
last stage allocates 60.12 GiB because it carries the MTP draft. Available KV memory is 36.01 versus 26.85 GiB per
GPU. The global cache limit leaves usable memory on the first stage unfilled.

For the complete PP2 c8 window, GPUs on the first stage average about 38.7% SM activity, 6.1% SM occupancy, 10.3%
tensor activity, and 238–243 W. The last stage averages 46.7% SM activity, 7.7% occupancy, 11.6% tensor activity, and
264–272 W. GPU activity itself is approximately 99–100%, so that headline counter conceals the stage imbalance. GPU
activity can include communication kernels; none of these counters alone is achieved FLOPS or a measure of useful SM
efficiency.

Aligned server scrape deltas cover 765 of the PP2 point's 768 requests. Mean prefill time is 0.700 seconds, decode
time 11.951 seconds, and queue time 0.199 seconds. Concurrent request durations overlap and cannot be summed into
exclusive GPU time. They do support testing decode batching and MTP despite the large input-token count.

PP8 maps GPU 0–7 directly to stages 0–7. Its default target partition is 9/10/10/10/10/10/10/9 layers. Actual model
memory is 36.86 GiB on stage zero, about 54.6 GiB on stages one through six, and 71.29 GiB on stage seven. Its initial
MTP3 deployment exposes 2,348,224 logical KV tokens and passes the semantic smoke test. Its completed c64 point
contains 64 reviews, 768 successful calls, and 24,576,000 prompt tokens over 847.32 seconds. It reports 91.1826%
cached input. The initial c32 point was stopped after its first review wave to prioritize higher concurrency; that
partial point is not a completed throughput result or an upper bound.

Across the complete PP8 c64 window, stage zero averages 27.6% SM activity, stages one through six 34.2–35.4%, and the
last stage 56.7%. DRAM activity is about 20.1%, 26.0–27.3%, and 47.3%, respectively. The last stage carries the
unsharded draft as well as target layers. Aligned scrape deltas cover 765 calls: mean prefill 0.928 seconds, decode
51.828 seconds, and queue 4.518 seconds. These overlapping request durations are not exclusive GPU time.

The MTP-off control reduces last-stage model allocation to 50.95 GiB and raises logical KV capacity to 5,138,816
tokens. The freed memory is measured, but it does not guarantee higher throughput. An additional MTP3 comparison moves
target layers to a 12/11/11/10/10/10/10/4 partition to test the measured last-stage imbalance.

The larger 8,192-token MTP-off deployment exposes 4,934,272 logical KV tokens. At c128, all 128 requests run without a
waiting queue after the cold prefill wave. The bounded screen captures 256 completed calls, one cold and one warm call
per review. Warm calls reuse 27,648 of 28,211 prompt tokens. Server intervals show about 155–166 output tokens/s
during warm decoding, with roughly 20–21% SM activity on stages one through seven. The screen stops after 745.69
seconds to prioritize MTP3 partition and RAM comparisons. No review has completed at that stop, so this is not a
completed-review throughput result or a proven bound on every MTP-off configuration. The earlier c64 controls are
partial as well.

The custom MTP3 partition passes live semantic smoke and exposes 3,835,840 logical KV tokens. Model allocations are
53.21 GiB on stage zero, 60.08–60.10 GiB on stages one and two, about 54.6 GiB on stages three through six, and 44.05
GiB on the final stage. Its actual environment and engine partition calculation give target-layer ranges 0–12, 12–23,
23–34, 34–44, 44–54, 54–64, 64–74, and 74–78. Across its complete c64 window, stages zero through six average about
34–37% SM activity and the final stage 39%. This narrows the measured stage imbalance without demonstrating a gain in
completed-review throughput at c64. Aligned scrape deltas cover 1,535 calls: mean prefill 1.01 seconds, decode 60.75
seconds, and queue 2.39 seconds. No preemption is recorded in that aligned scrape interval.

No isolated prefill-only benchmark was run. In a cold-prefill-heavy window 20–100 seconds after the custom c64 point
starts, stages zero through six show 66–76% SM activity, 12–14% occupancy, and 33–40% tensor activity. The final stage
shows about 49%, 8%, and 18%, respectively. Server intervals during this cold fill process about 13.6–16.3 thousand
new prompt tokens per second, while decode overlaps. In a later warm-turn window at 350–550 seconds, stages zero
through six show about 38–41% SM activity, 6% occupancy, and 12–13% tensor activity; the final stage shows about 47%,
6%, and 8%. That window also contains short new-turn prefills. These are selected portions of a mixed session run, not
isolated GPU kernel measurements. They do not demonstrate prefill compute saturation.

## RAM controls

The pinned engine's [native prefix offload](https://docs.vllm.ai/en/v0.32.0/features/kv_offloading_usage/) restores
inactive prefix blocks from host memory. It does not directly enlarge the active GPU decode batch. PP8/MTP3
configuration validation passes, but this alone does not establish live connector correctness or throughput. Actual
request accounting sums local and external cache hits; server prompt-source counters and load/store byte counters
distinguish restored prefixes from fresh prefill.

[HiSparse](https://docs.vllm.ai/en/v0.32.0/design/hisparse/) is a different mechanism. It keeps active sparse-MLA KV
in RAM and fetches selected rows into a GPU hot buffer. The installed SM120 backend implements this path and handles
MTP verification rows. The pinned configuration explicitly rejects pipeline parallelism greater than one. The bounded
RAM trial therefore uses TP4/DP2 with eight expert-parallel ranks with the same serving settings as the GPU-only
control. Its longer workloads use different seeds. It also composes ordinary prefix offload for the indexer cache.
Both attempted sustained HiSparse rows failed before semantic smoke during memory profiling. The SM120 backend entered
prefill staging without a host staging plan and raised an assertion. The second attempt's continuous full server log
preserves the root cause. Host allocation itself succeeded: each DP replica created one shared 64 GiB mmap pool opened
by its four TP ranks. This establishes allocation behavior, not serving compatibility or performance. There is no
measured HiSparse cache-hit rate, throughput, or cost per completed review.

The replacement native-prefix-only comparison retains TP4/DP2, MTP3, and a 24-request server limit per DP rank. It
reserves 64 GiB of host prefix storage per DP engine and uses 144 reviews at c24 and 288 at c48, six whole waves per
point. Summing proportional shared memory across all eight worker mappings verifies 127.996 GiB of physical host
storage. The live server passes semantic smoke and exposes 388,608 logical GPU KV tokens per DP rank. Native prefix
storage does not expand active GPU decoding capacity.

The c24 point completes all 144 reviews and 1,728 calls. Every call emits 330 tokens with the expected length finish,
and every review contains 384,000 prompt tokens. The complete window takes 1,787.95 seconds: 30,927 input tokens/s,
including 28,262 cached and 2,665 uncached. It serves 289.94 reviews/hour for $0.047374 per review. Actual cached
input is 91.3824%. Native offload consistently reports one additional 64-token cached block on warm calls compared
with the previous GPU-only control, whose measured hit rate is about 91.18%. The report uses each engine's actual
usage.

The first-to-third completed concurrency-wave interior contains 590 actual calls and 48 review completions over 608.80
seconds. It serves 30,963 input tokens/s and 283.84 actual review arrivals/hour, or $0.048393 per arrival. The wider
first-to-fifth interior contains 1,191 calls and 96 review completions over 1,227.81 seconds. It serves 30,963 input
tokens/s: 28,337 cached and 2,626 uncached, with 281.48 review arrivals/hour and $0.048799 per arrival. Time-weighted
client occupancy is 23.99991 of 24, with 99.9910% of the wider interval at the cap. Its 96 handoff gaps total 0.110
seconds. Review arrivals and completed request tokens are counted independently because requests can belong to reviews
that started or finish outside the interior interval. The full-run cost remains the primary result.

Server prompt counters account for every input token: 8,990,464 restored from RAM plus 41,540,352 local cache hits
equal all 50,530,816 API cached tokens. Adding 4,765,184 computed tokens gives all 55,296,000 input tokens. The final
scrape precedes the last two response completions, so response counters cover 1,726 of 1,728 calls. Five request
preemptions are recorded. Engine-reported load bytes total 1.969 TB and store bytes 284.46 GB. These counters can
aggregate worker views; they are not asserted to be unique PCIe traffic. The wider interior averages 21.2 running
requests and 2.4 waiting, while the client keeps nearly 24 reviews active. Whole-window SM activity is 37.1–37.3% and
tensor activity 11.0–11.2%. This does not establish compute saturation. At c48, a closed interval from the 48th to
96th completed review contains 604 calls over 659.25 seconds. It serves 29,295 input tokens/s: 26,732 cached and 2,563
uncached, with 262.12 actual review arrivals/hour and $0.052403 per arrival. The client spends 99.9931% of this
interval at 48 active reviews, but the server averages only 21.26 running requests and 26.18 waiting. SM activity is
34–35%. Scraped counters show 12.776 million external prompt tokens, 2.792 TB of reported loads, and four preemptions
within the narrower scrape interval. Offered concurrency increases queueing and transfers without a useful gain over
c24 in this bounded comparison. The remaining c48 work was stopped to prioritize PP8 confirmation. Its terminal failed
row and partial calls are retained. This is a measured interior comparison with different seeds, not a full-run result
or a global throughput upper bound.

The preceding GPU-only control, with corrected session admission and the same 24-request server limit, completed c16
with 64 reviews in 944.50 seconds: 26,020 total input tokens/s, 23,726 cached, and 2,295 uncached. Its logical cache
is 388,608 tokens per DP rank, smaller than the earlier 16-request-limit deployment. The c24 partial point was
deliberately stopped to prioritize the requested sustained RAM comparison after 254 calls and four completed reviews.
Late calls had already lost most reuse, but this is not a completed throughput point or a proved upper bound. Its last
observed completion was at 546.61 seconds. The queued short HiSparse row was stopped before measurement and explicitly
marked superseded; it is distinct from the later two genuine initialization failures.

GPU query samples use a native one-second nvidia-smi loop. A complementary privileged dmon stream collects SM
activity, SM occupancy, tensor activity, DRAM activity, and PCIe traffic. Its actual timestamps define the cadence.
Full server metrics are captured about every 15–17 seconds, and client start/completion clocks define the measurement
windows.

## Sustained PP8 confirmation

The fresh c96 confirmation completes 576 reviews and all 6,912 calls, six complete concurrency waves. Every review has
384,000 input tokens and 3,960 output tokens. The complete cold-fill/drain-inclusive window lasts 4,465.73 seconds,
serving 49,529 input tokens/s: 45,162 cached and 4,367 uncached. It completes 464.34 reviews/hour for $0.0295814 per
review, with 91.1832% cached input. Mean client occupancy is 90.02 of 96. The last review is admitted at 3,948.13
seconds; the final drain takes 517.60 seconds.

The closed first-to-third wave interior runs from the 96th to 288th completed review, over 1,349.86 seconds. It
contains 2,169 actual completed calls and 192 completed reviews. Throughput is 51,282 input tokens/s: 46,527 cached
and 4,754 uncached, with 512.05 actual review arrivals/hour and $0.026825 per arrival. The wider first-to-fifth window
spans 2,701.40 seconds, with 4,438 calls and 384 review arrivals. It serves 52,514 input tokens/s: 47,740 cached and
4,774 uncached. Its 511.74 review arrivals/hour and $0.026841 per arrival closely match the shorter interior. These
windows count actual request completions independently of review arrivals; boundary work is not replaced with nominal
tokens. The complete-window cost remains the primary finite-run result.

Client occupancy averages 95.99987 of 96 and stays at the cap for 99.9874% of the interval. The server averages 95.36
running requests and 0.38 waiting. SM activity is 49.1–54.0%, occupancy 7.9–8.9%, and tensor activity 19.5–22.6% on
stages zero through six versus 11.9% on the last stage. GPU activity is approximately 100%, while mean power is
282–348 W per GPU against the 600 W limit. DRAM activity is 36.4–40.0% on stages zero through six and 45.4% on the
last. There are zero preemptions in the narrower aligned scrape interval. Its draft-token acceptance is about 82.46%.
These mixed prefill/decode counters show better useful throughput with a supplied pipeline; they do not demonstrate
compute saturation or isolate a single bottleneck.

The sustained PP8 interior exceeds the native-RAM c24 wider interior's 30,963 input tokens/s. The earlier custom PP8
c96 full window served only 30,460 input tokens/s while averaging 61.18 active reviews and draining for 907 seconds.
That finite short-window rate must not be treated as a steady-state limit. Seeds and review counts differ from earlier
screens.

The c104 upper bracket completes all 624 reviews and 7,488 calls in 4,498.24 seconds. It serves 53,269 input tokens/s:
48,572 cached and 4,697 uncached. It completes 499.40 reviews/hour for $0.0275047 per review, with 91.1833% cached
input. The 7.55% complete-window throughput gain over c96 selects c104 for this bounded search. Mean active reviews
are 97.54 of 104. The last review is admitted at 3,988.17 seconds, followed by 510.07 seconds of drain. All calls emit
exactly 330 tokens, and every review contains the expected 384,000 prompt tokens.

The c104 point's planned interior is bounded by the 104th and 312th completed reviews. Across 1,431.38 seconds, it
contains 2,603 actual calls and 208 review arrivals. It serves 58,314 input tokens/s: 53,349 cached and 4,966
uncached, with 523.13 review arrivals/hour and $0.026257 per arrival. Client occupancy averages 103.99987 of 104 and
remains at the cap for 99.9872% of the interval. The server averages 103.32 running requests and 0.33 waiting. SM
activity is 52.9–58.5%, occupancy 8.7–9.6%, and tensor activity 21.1–24.3% on stages zero through six versus 12.9% on
the last. This window has a higher input rate than c96 but only 2.2% more review arrivals, because the number of
completed calls crossing the review boundaries differs. The wider 104th-to-520th interval spans 2,793.51 seconds and
contains 5,051 actual calls and 416 review arrivals. It serves 57,929 input tokens/s: 52,874 cached and 5,055
uncached. Its 536.10 arrivals/hour cost $0.0256215 each. The input rate is within 0.7% of the shorter interval. Mean
client occupancy is 103.99986 of 104, and the batch is at its cap for 99.9863% of the interval. The server averages
103.31 running requests and 0.37 waiting. Wider-window SM activity is 53.1–58.6%, occupancy 8.7–9.7%, tensor activity
21.2–24.4% on stages zero through six and 13.0% on the last stage, and power 298–368 W per GPU. There are no request
preemptions; sampled draft-token acceptance is 82.23%. The larger batch improves useful throughput without
demonstrating prefill or overall compute saturation.

The final c104 server counters match every API token and completion after subtracting the one semantic smoke request:
218,489,664 local cache hits plus 21,126,336 locally computed tokens equal all 239,616,000 input tokens. There are
zero external cache hits, 2,471,040 generated tokens, 7,488 length completions, and zero preemptions. The earlier c96
final scrape misses the last request; its complete API usage remains authoritative. This accounting distinction is
preserved rather than forcing approximate server windows to equal the full client run.

| Final six-wave result | Reviews | Total input/s | Cached input/s | Uncached input/s | Reviews/hour | Full cost/review |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| PP8 c96 | 576 | 49,529 | 45,162 | 4,367 | 464.34 | $0.0295814 |
| PP8 c104, selected | 624 | 53,269 | 48,572 | 4,697 | 499.40 | $0.0275047 |

| Wider supplied interior | Actual calls | Review arrivals | Total input/s | Arrivals/hour | Cost/arrival |
| --- | ---: | ---: | ---: | ---: | ---: |
| PP8 c96, 96th→480th review | 4,438 | 384 | 52,514 | 511.74 | $0.0268414 |
| PP8 c104, 104th→520th review | 5,051 | 416 | 57,929 | 536.10 | $0.0256215 |

The wider input rate rises 10.3%, while actual review arrivals rise 4.8%. These are different counts over overlapping
reviews, so neither is silently converted into the other. The complete fixed-shape results are the cleanest aggregate
comparison. Six waves and the two interior lengths reduce the earlier underfilled-window problem, but each final
concurrency still has only one fresh-seed run. Seed effects and run-to-run variation limit the precision of the
margin. The c104 maximum final prompt population is about 3.81 million tokens against 3.84 million logical KV tokens,
before block and speculation overhead and shared-prefix savings. The maximum late-context population approaches the
measured cache capacity; this does not prove that every larger asynchronous batch must be slower.

## Cost accounting

The public regional price estimate is $13.308/hour for the G4 node, $0.219178/hour for disk capacity, $0.082188/hour
for IOPS above the included 3,000, $0.1238367/hour for throughput above the included 140 MiB/s, and $0.0025/hour for
the spot VM's external IPv4 address. Total: $13.7357027/hour. This is a public-price estimate, not an invoice; it
excludes tax, egress, and contract discounts. Deployment time, failed attempts, and idle time belong to experiment
spend rather than the full-utilization serving rate.

OpenRouter's current regular DeepInfra endpoint lists $0.5625 per million uncached input tokens, $0.105 per million
cached input tokens, and $1.80 per million output tokens. Its separate flex endpoint lists lower prices and is not
silently substituted for the regular route. The endpoint metadata labels DeepInfra FP4, which does not establish that
it serves this exact checkpoint or that its quality matches this deployment.

The relay accounting formula charges uncached input, cached input, and output separately. Cached input is a subset of
the prompt count. A mixed workload measures one aggregate infrastructure-cost constraint; it cannot uniquely identify
three independent token prices. One possible allocation scales all three DeepInfra reference rates by the same factor
so their charge on the observed token mix equals the measured node cost. That allocation is a pricing assumption, not
a measurement of isolated prefill, cache-read, or decode costs. The node charge is counted once.

For the selected PP8 c104 confirmation, the regular reference route charges $0.0629373 per review on the actual token
mix. Scaling all three rates by 0.43701704 allocates the measured $0.0275047 node cost as $0.245822/M uncached input,
$0.0458868/M cached input, and $0.786631/M output. This is an inferred allocation for this workload and utilization,
not an independently measured cost for each token class.

At 260 selected PP8 median-shaped reviews per day, the full-utilization allocation is $7.1512/day and requires 0.5206
busy node-hours/day. A node rented continuously costs $329.6569/day, or $1.26791 per review if those 260 reviews are
its only traffic. Near-full utilization requires additional traffic or concentrating the work into a shorter operating
window. These median-shaped scenarios are not estimates of the actual mean review distribution. Their input-to-output
ratio is about 97:1, separate from the user's aggregate 130:1 estimate.

For the selected c104 configuration, the daily accounting comparison is:

| Median-shaped scenario, 260 reviews/day | Allocated node cost/day | Regular DeepInfra/day |
| --- | ---: | ---: |
| Observed 91.1833% cached input | $7.15121 | $16.36369 |
| 92% cache, holding the inferred token prices fixed | $6.98818 | $15.99062 |

The 92% row is a billing-mix counterfactual, not a measured runtime or throughput result. It holds all three inferred
rates fixed while changing cached versus uncached input. It gives $0.02687760 per review for the node allocation,
compared with $0.0615024 for regular DeepInfra. The always-on node still costs $329.6569/day regardless of this token
allocation. The relay model clamps cached tokens to the prompt count, then charges uncached input, cached input, and
generated output once each.

Pricing was refreshed on 2026-10-24 from the [GCP spot table](https://cloud.google.com/spot-vms/pricing), [disk
pricing](https://cloud.google.com/compute/disks-image-pricing), [IPv4
pricing](https://cloud.google.com/vpc/network-pricing), and the [OpenRouter endpoint
API](https://openrouter.ai/api/v1/models/z-ai/glm-5.2/endpoints). The archive retains selected regional tables,
retrieval times and hashes, the endpoint snapshot, actual VM disk settings, and the relay accounting revision.

## Limits

Synthetic random-token sessions establish throughput and cache behavior under this workload. They do not validate
code-review recall, real-code MTP acceptance, quantization quality, or behavior during tool delays. The semantic smoke
test only establishes a basic serving path. Production requires a review-benchmark A/B test and session routing that
keeps prefixes warm. Each final concurrency has one six-wave run with a distinct fresh seed. Interior-window
sensitivity checks measure stability within these runs; they are not independent repeated trials or confidence
intervals. The selected configuration is the strongest measured setting in this bounded search, not a proof of a
global optimum. Spot interruption, reload time, fallback behavior, and production utilization affect real cost.

## Retained attempts

The archive keeps the latest terminal run at its root and these earlier attempts under `historical-runs/`. Original
interrupted records retain their emitted status; accompanying notes distinguish them from completed experiments.
Deliberate early stops remain failed system rows with partial measurements.

| UTC run | System row status | Interpretation |
| --- | --- | --- |
| 2026-10-23 18:47:08 | 3 interrupted `running` records | Autotuning startup stall; no benchmark result |
| 2026-10-23 19:23:14 | 3 interrupted `running` records | OpenMP loading stall; no benchmark result |
| 2026-10-23 19:34:13 | 3 failed | TP8 completed median bounds; higher and non-median work deliberately stopped |
| 2026-10-23 21:11:11 | Missing records | Artifact filename exceeded the filesystem limit before topology deployment |
| 2026-10-23 21:17:09 | 2 failed | TP4/DP2 and TP4/PP2 useful points retained; remaining ladders deliberately stopped |
| 2026-10-23 22:40:40 | 1 failed | Partial PP8 c32; search prioritization |
| 2026-10-23 23:01:40 | 1 succeeded | Default PP8 c64 single-wave screen |
| 2026-10-23 23:20:29 | 2 failed | Partial MTP-off controls; search prioritization |
| 2026-10-23 23:35:36 | 1 failed | MTP-off c128 cold/warm screen; no completed review |
| 2026-10-23 23:50:49 | 1 succeeded | Custom PP8 c64/c96 finite-window exploration |
| 2026-10-24 00:51:27 | 2 failed | GPU c16 retained; c24 stopped, queued short RAM row superseded |
| 2026-10-24 01:20:18 | 2 failed | HiSparse initialization failure; no serving measurement |
| 2026-10-24 01:26:18 | 1 succeeded, 1 failed | Native RAM c24 six-wave result; c48 bounded and deliberately stopped |
| 2026-10-24 02:33:57 | 2 succeeded | Final PP8 c96 and c104 six-wave confirmation |

The earlier PCI-detection preflight failure is retained separately with its controller log and lease deletion record.
The bounded filename and hardware-detection fixes are generic harness changes. The node from the filename failure was
automatically deleted; its replacement is the only node used for subsequent topology and RAM work.

## Reproducibility and retained evidence

The latest run is `2026-10-24_02-33-57`, with run ID `20261024T023357Z`. It covers both expected filtered confirmation
rows, c96 and c104, and both have terminal `succeeded` status. Its recorded source revision is
`fc8a937787ff0104b43108369a5d62ea92fe6055` with a clean worktree at run initialization. The actual command submitted
to each client container is retained independently. Both extracted client sources have SHA-256
`cd4f7618957fdd0c792169dc1db6f8daf25805f1d533f706dfed333aba9cbc1b`. This matters because an evolving shared checkout
can change a later row's client source after the run's revision was recorded. Historical dry runs and versioned source
snapshots preserve earlier admission behavior; their recorded revision alone is not asserted to prove every live
command.

The platform archive is `results_pro6000x8.tar.gz`, tracked with Git LFS. Its root is `2026-10-24_02-33-57/`. It
includes 25 system-only YAML records: five succeeded, fourteen failed, and six original historical `running` records
from interrupted startup attempts. Notes identify those interruptions; their emitted statuses are not rewritten. The
filename failure has no record and is explicitly retained as a missing-record failure. No successful result is
invented for it. Every record reports the exact eight-GPU platform. The latest records are:

| Row | Record member beneath the archive root |
| --- | --- |
| c96 | `pro6000x8_np6912_s65537_sc96_mcr128_pps8_tps1_eevllm-U-V2-M-R-1-vllm-P-L-P-12-11-11-10-10-10-10-4_p12-11-11-10-10-10-10-4_pconfirmation_smedian_ttp1pp8_88dcbe895f55.experiment.yaml` |
| c104 | `pro6000x8_np7488_s131101_sc104_mcr128_pps8_tps1_eevllm-U-V2-M-R-1-vllm-P-L-P-12-11-11-10-10-10-10-4_p12-11-11-10-10-10-10-4_pconfirmation_smedian_ttp1pp8_5c8157662d72.experiment.yaml` |

Each record's same-stem `.benchmark.log` contains the authoritative complete request records, and `.server.log`
retains server output. `confirmation-c104-full.json`, `confirmation-c104-interior.json`, and
`confirmation-c104-interior-wide.json` contain the structural token, arrival, and occupancy totals.
`confirmation-c104-telemetry.json`, `confirmation-c104-cache-accounting.json`, and
`confirmation-c104-cost-allocation.json` retain the aligned counters and arithmetic. Equivalent c96 files are present.
Raw streams are `gpu-telemetry-1s.csv`, `gpu-gpm-counters.csv`, and `gpu-and-server-telemetry.jsonl`.
`confirmation-c104-client-command.json` and `confirmation-c104-client-source.py` retain actual submitted source.
`image-identity.txt` records the actual image ID and repository digest.

The thirteen earlier attempts are nested under `historical-runs/`, including each failed row, partial client output,
startup evidence, and deliberate-stop note. `support/` retains hardware, software, model, offload-support, pricing,
relay-accounting, source, and dry-run evidence. Only copied node addresses and hostnames are redacted; original system
records otherwise remain unchanged. Public SSH metadata is removed from copied provider inspection data. Archive
members are byte-verified against the sanitized staging tree, and every system record is compared with its original
under the documented address-only substitutions. Secret and address scans and client-source hashes are checked.

`rental-controller.log`, `rental-lease.json`, and `owned-vm-cleanup.json` retain cleanup evidence. The controller
reported deletion at 05:09:43 UTC on 2026-10-24, and an independent GCP query at 05:10:33 UTC found no matching owned
instances. All six identified rental, benchmark, and telemetry processes exited. The system records retain their
original `external` infrastructure state because Emmy received an SSH host; the separate lease and provider query
prove cloud deletion. No infrastructure is retained.
