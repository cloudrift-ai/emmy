# GLM-5.2 NVFP4 cached review sessions

Measure serving cost for the user's synthetic code-review workload on exactly eight NVIDIA RTX PRO 6000 Blackwell
Server Edition GPUs rented as one GCP G4 spot VM. NVIDIA's official checkpoint is pinned to
`53e0691e21895a3863a606dfd12910c69eba94ab`; its safetensors total 464,823,042,096 bytes. Eight 96 GB cards provide
768 GB of nominal GPU memory. Runtime allocations, replicated attention state, and KV cache still require measurement.

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
