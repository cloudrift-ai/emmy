# GLM-5.2 NVFP4 cached review sessions

Measure serving cost for the user's synthetic code-review workload on exactly eight NVIDIA RTX PRO 6000 Blackwell
Server Edition GPUs rented as one GCP G4 spot VM. NVIDIA's official checkpoint is pinned to
`53e0691e21895a3863a606dfd12910c69eba94ab`; its safetensors total 464,823,042,096 bytes. Eight 96 GB cards provide
768 GB of nominal GPU memory. Runtime allocations, replicated attention state, and KV cache still require measurement.
Report total input tokens per second first, split into cached and uncached input from actual usage over each complete
measurement window. Cached tokens count toward served traffic, but do not represent repeated prefill computation.
For complete fixed-shape median runs, total input throughput and completed-review throughput rank configurations
identically. Interior windows count actual completed calls and review arrivals separately.
Select configurations by that throughput and cost. There is no time-to-first-token constraint. The user subsequently
narrowed optimization to the median scenario. Retain earlier p90 and 130:1 evidence, but do not continue
those searches. Record per-GPU activity, memory, power, and clocks every second, alongside timestamped server metrics.

The initial deployment uses vLLM 0.32.0, TP8, one attention replica, prefix caching, FP8 KV cache, and MTP3. G4 supports
PCIe peer-to-peer communication. TP8 avoids replicating the higher-precision attention and shared-expert weights across
eight attention replicas; later topology comparisons must be measured on this interconnect. Do not assume the H200
DP8/EP configuration is optimal here. Record the selected NVFP4 kernel backend from the server log.

FlashInfer autotuning is disabled after an initialization hang in the first deployment. Repeated worker-stack samples
showed one rank blocked in the router GEMM while another waited in FlashInfer's tuning collective. Preserve that
interrupted attempt and the stack samples. Native NVFP4 kernels remain enabled; their default tactic selection may
leave performance below a working autotuned deployment, which must be stated when interpreting throughput.
`OMP_NUM_THREADS=1` also bounds CPU threading during weight loading after a subsequent attempt stalled in an
OpenMP tensor copy. vLLM already reduces Torch to one thread for serving when this override is absent.

The three scenarios center growing prompts on 32,000 input / 330 output / 12 calls, 60,000 / 590 / 28, and
42,900 / 330 / 12. These are separate synthetic scenarios, not a fitted distribution or estimates of mean traffic.
Each session begins with an independent random token prefix plus 256 common tokens, then appends actual generated
token IDs and 512 new random input tokens after each call. Those two prefix sizes are assumptions. Calls within a
session are sequential and have no tool wait time. The same vocabulary sampling and seed scheme as the H200
experiment is used. Synthetic text does not validate code-review recall or MTP acceptance on actual code.

The initial search measures 64 sessions at each of 4, 8, 12, 16, and 24 concurrent sessions. TP8 duplicates MLA KV
state across ranks, so its useful concurrency may be lower than the H200 DP8 run. Inspect actual cache-hit usage and
the throughput peak before extending the ladder, comparing topology, and repeating the best settings. Confirmed
comparisons should use 128 sessions to match the H200 trial size; report different trial sizes explicitly.

Select `phase=initial` for the TP8 search or `phase=topology` for the topology comparisons. TP4/DP2 uses eight
expert-parallel ranks, the general-purpose `allgather_reducescatter` backend, and session affinity. Search
8/16/24 concurrent sessions for median and 130:1, and 8/12/16 for p90. The TP8 search already provides a low-load
reference at four sessions. Extend a winning endpoint and repeat the selected settings; a completed search point
does not by itself establish repeat stability. Preserve partial measurements and failed rows when ending an observed
cache-thrashing point early. Exploration uses 32 reviews for p90, giving four waves at concurrency eight and two at
sixteen; confirm the chosen setting with a longer trial. Do not spend hours completing a low-concurrency baseline
after it stops helping the topology decision.

The median topology comparison also includes TP4/PP2, without expert parallelism, at 8/16/24/32 concurrent sessions.
After its first completed point, defer the remaining ladder to prioritize the user-requested PP8 comparison. A
deferred point is not evidence that the configuration is slower; preserve the terminal failed row and partial logs.
The intended mapping keeps each four-GPU tensor-parallel group within one NUMA group and transfers activations
between pipeline stages. Verify actual device grouping after deployment. The pinned vLLM image passed config-only
validation with MTP3, the V2 model runner, async scheduling, and PP1 for the draft. That establishes configuration
support, not runtime correctness or performance. Smoke-test the live pipeline before accepting its results. Compare
median throughput first, then refine and repeat the median workload on the useful topology. Test larger practical
prefill token budgets after selecting a useful topology; record their KV capacity and cache reuse.
Enough concurrent sessions must be present to keep pipeline stages busy, even when that increases latency.

The median-only `pipeline8` phase tests TP1/PP8 with a 2,048-token prefill budget and 32/64/96/128 concurrent
sessions, using 128 reviews per point. The pinned image passed config-only validation with MTP3 and async scheduling.
Its default target partition is 9/10/10/10/10/10/10/9 layers. Checkpoint headers show 18.54 GiB for the MTP layer; the
V2 runner loads it only on the last pipeline stage. This puts roughly 71 GiB of checkpoint tensors on that stage,
before runtime buffers, and leaves room for an initial live trial on 96 GB cards. Verify actual allocations and smoke
results, then increase the prefill budget where measured memory permits. Examine per-stage activity before changing
the layer partition or MTP; repeated decoding on the last stage may limit throughput even when the target layers are
evenly distributed. Extend a winning endpoint only if cache capacity and measurements justify it, and repeat the
selected setting with enough sessions to assess stability.

The initial PP8 c32 point was stopped after its first review wave because its observed rate and stage imbalance made
continued low-concurrency exploration a poor use of search time. This is a deliberate search-prioritization stop,
not a completed throughput result or a proven upper bound. The `pipeline8_bound` phase screens c64 with 64 reviews
before the MTP-off controls. One wave is acceptable for screening; confirm useful winners with multiple waves.

The `pipeline8_control` phase removes MTP and compares 2,048- and 8,192-token prefill budgets at 64/96 concurrent
sessions. The live MTP3 deployment placed 71.29 GiB of model allocations on stage seven, versus about 54.6 GiB on
stages one through six, and its decode samples showed greater SM activity on the last stage. Removing the unsharded
MTP layer tests that measured imbalance and releases cache memory. These are throughput controls, not quality tests.
Use 128 reviews per point and matching seeds across budgets; the first seed also matches the MTP3 c64 point. Verify
actual KV capacity before continuing the higher concurrency, then confirm the useful winner with fresh seeds.

The `pipeline8_batch` phase screens MTP-off with the 8,192-token prefill budget at 128 concurrent reviews. The
2,048-token MTP-off deployment exposes 5,138,816 logical KV tokens, enough to motivate a larger batch despite slow
c64 decoding. Verify the larger budget's actual capacity before accepting this point. Its 128 reviews form one wave;
a useful winner needs a longer confirmation. Partial low-concurrency controls do not establish its throughput.

The `pipeline8_partition` phase keeps MTP3 and moves five target layers off its last stage. The partition is
12/11/11/10/10/10/10/4. The completed default-partition c64 screen showed about 57% SM activity on the last stage,
versus 28–35% elsewhere. This motivates the comparison but does not establish the cause of all pipeline overhead.
The MTP-off control releases memory while reducing steady decode activity; retain its partial records if ending that
control to prioritize the partition comparison. Such stops are search decisions, not completed throughput results.
Use a 2,048-token prefill budget, 128 reviews, and c64/c96 initially. Verify live partition ranges, model allocations,
cache capacity, and semantic smoke before interpreting throughput. Extend a winning endpoint only when actual cache
capacity supports it; compare larger prefill budgets and repeat useful winners with fresh seeds.

The `confirmation` phase returns to the useful custom PP8 partition with fresh prompts and six complete waves:
576 reviews at c96 and 624 at c104. The latter is the one authorized upper bracket, close to the measured 3,835,840
logical KV-token capacity: last-call prompts consume 3,809,624 tokens before output generation, shared-prefix
deduplication, block rounding, and draft overhead. Confirm actual late-context capacity, prefix reuse, preemption,
stage activity, and full-window throughput. Compare interior windows from the first to third and first to fifth
completed concurrency waves. Verify actual batch occupancy from logged admissions and completions. Preserve any partial
point if this boundary causes sustained cache loss. These rows retain the prior kernel settings; an optional autotune
startup check is lower priority than completing the requested RAM and sustained throughput measurements.

The `host_cache` phase tests the user's RAM-cache proposal on TP4/DP2 with corrected least-active session admission.
The pinned engine rejects HiSparse when pipeline parallelism exceeds one. Its SM120 sparse MLA backend does implement
host-backed active KV fetches, and the TP4/DP2/MTP3/V2 configuration passes validation. Compare GPU-only c16/c24 with
HiSparse c24/c48, initially 64 reviews per point and a 24-request server limit per DP rank. The overlapping c24 point
uses the same effective seed: each preceding 64-review point advances the seed by 768 requests. Repeat useful winners
with more waves and fresh seeds before selecting a serving configuration.

The user subsequently requested sustained measurements with enough reviews to keep the batch full. Preserve the
initial GPU-only points and any deliberately stopped short rows as historical evidence. The `host_cache_sustained`
phase replaces the short RAM screens with 144 reviews at c24 and 288 reviews at c48: six whole waves per point, with
fresh prompts. Verify actual live allocations and transfer counters before interpreting either result. Use the full
cold-fill/drain-inclusive window for cost per completed review. Also report an interior window from the first to the
third completed concurrency wave, plus the first to fifth as a wider sensitivity check. Integrate recorded admissions
and completions to report time-weighted active reviews, time at the concurrency cap, and any handoff gaps. Count
actual request-token completions and completed-review arrivals separately; boundary requests can belong to reviews
that finish outside the window. A short warm output interval is not a completed-review throughput measurement.

HiSparse reserves 64 GiB of logical host KV per DP replica. A composed native OffloadingConnector reserves another
16 GiB per DP engine for indexer prefix blocks and includes decode-created blocks. These are configuration budgets;
verify actual physical host allocations, GPU hot-buffer capacity, transfer activity, and smoke correctness. Host KV
capacity can have replicated physical views. The available host memory before this phase is approximately 1.469 TB.
The ordinary OffloadingConnector also passes PP8/MTP3 configuration validation, but it extends inactive prefix storage
and does not itself increase resident active decode capacity. A live connector test is required to establish support.

The live HiSparse combination failed before smoke during memory profiling: its SM120 prefill path reached an absent
host staging plan. Preserve both failed sustained rows and the full traceback. The `host_prefix` phase therefore
compares native prefix offload alone on the same TP4/DP2/MTP3 topology, with 64 GiB of host prefix storage per DP
engine and the same six-wave c24/c48 workloads. It stores inactive prefixes, including generated tokens. It does not
expand active GPU decode capacity; if preemption or transfers make a higher point unproductive, retain the partial
point and its explicit stop reason. Its two rows have separate fixed seeds, so changing the first row length does not
shift the second seed. Comparisons with earlier GPU-only points use different seeds and review counts and are
directional, not matched controls. Do not patch upstream serving code for this bounded compatibility experiment.

The pinned request accounting sums local and external cache hits. Validate it against actual response usage and the
server's prompt-token source counters. Retain host-cache utilization, load/store byte and time counters, preemptions,
and failed transfers. Treat host hits as cached input in billing; do not call restored prefixes fresh prefill. HiSparse
is experimental, and synthetic throughput does not validate review recall or numerical parity on real review inputs.

Checkpoint tensor headers total 397.69 GiB for routed experts and 35.19 GiB for all other tensors. Relative to TP8,
TP4/DP2 can add up to 4.40 GiB per GPU of non-expert weight shards; existing replicated tensors need no extra copy.
The initial TP8 server measured 57.15 GiB of model allocations and 29.48 GiB of KV cache per GPU, with 578,432 logical
KV tokens across its one attention replica. TP4/DP2 measured 66.09 GiB of model allocations and 20.49 GiB of KV cache
per GPU, with 401,984 logical KV tokens per attention replica, or 803,968 across the node. Its actual runtime allocation
exceeds the weight-only estimate. Device inspection confirmed GPUs 0–3 form DP rank zero and GPUs 4–7 form DP rank one.
Both deployments selected native FlashInfer CUTLASS NVFP4 MoE kernels and FlashInfer sparse MLA attention.

Every measured call must return exactly the requested output length, token IDs, and cached-token usage. Count the
initial cold calls and the final drain in elapsed time. A 92% cache-hit rate is the user's traffic target, not a value
inserted into the synthetic run. Fresh seeds at each point prevent prior points from warming the next point's prefixes.
Session affinity is configured; one DP rank makes it immaterial for the initial deployment.

Use uncached-input, cached-input, and output token accounting. One mixed workload measures one aggregate cost
constraint, so any three component prices require an explicitly stated allocation assumption. Do not allocate the
full VM charge independently to every token class. Verify current regional GPU, CPU, RAM, disk, and public-IP prices
after rental. Preserve every attempted row, including deployment or benchmark failures, with the raw logs.

Sources checked on 2026-10-23:

- [NVIDIA model card](https://huggingface.co/nvidia/GLM-5.2-NVFP4/tree/53e0691e21895a3863a606dfd12910c69eba94ab)
- [vLLM GLM-5.2 recipe](https://recipes.vllm.ai/zai-org/GLM-5.2)
- [GCP GPU machine types](https://docs.cloud.google.com/compute/docs/gpus/)
- [vLLM expert parallel deployment](https://docs.vllm.ai/en/latest/serving/expert_parallel_deployment/)
- [vLLM data parallel deployment](https://docs.vllm.ai/en/latest/serving/data_parallel_deployment/)
- [vLLM native KV offload](https://docs.vllm.ai/en/v0.32.0/features/kv_offloading_usage/)
- [vLLM HiSparse architecture](https://docs.vllm.ai/en/v0.32.0/design/hisparse/)
