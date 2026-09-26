# Native Qwen3 serving on RTX 4080

This experiment compares the checkpoint-qualified native artifact with stock vLLM before choosing milestone 4 work.
It measures the current implementations, not the isolated benefit of Rust: their kernels, precision policies,
prefill algorithms, and cache allocations differ.

## Protocol

One local NVIDIA GeForce RTX 4080, 16 GiB, driver 595.91.07, serves Qwen/Qwen3-0.6B revision
`c1899de289a04d12100db370d81485cdf75e47ca` with FP16 weights and a 4,096-token context. The desktop shares the GPU.
The recipe takes the common GPU lock; no other compute process was present during the native measurements.

Native uses the existing artifact qualified by the
[sampling and context investigation](../native_generation/SAMPLING_CONTEXT.md). Its conservative deterministic
schedules and FP32 residual/attention intermediates were chosen for correctness, not serving speed. It processes
prefill sequentially and captures the token step. The recipe records hashes of its manifest, serving metadata,
tokenizer, and server binary. The artifact contains checkpoint weights and is not published with the results.

Stock uses FP16, Triton attention, full CUDA graphs, no prefix caching, a four-sequence limit, a 256-token prefill
budget, and a 0.6 device-memory budget. Native has one active request and rejects additional requests as busy.
Consequently, the concurrency-four trial measures native overload behavior, not native batching throughput.

For each engine, the existing vLLM client sends four requests at concurrency one with 32, 256, and 1,024 input tokens
and exactly 32 output tokens. Each length has three repeats, seeds 1–3, and one warmup before each repeat. A separate
mixed-length trial sends eight requests at concurrency four, seed zero, input length 512 with a 0.5 range ratio,
and 32 output tokens. Greedy decoding and ignore-EOS keep output work fixed. A fixed text completion is retained
for inspection; it is not a replacement for model qualification.

## Reproduction

Build the matching native binaries and provide the qualified serving bundle with `NATIVE_ARTIFACT`. The recipe uses
the checkout's virtual environment and local cached checkpoint. Run through `emmy bench` with `--local`; no cloud
capacity is provisioned. Native artifact preparation and numerical qualification are described in the linked
investigation. New compiler schedules require a fresh qualified artifact; replaying this bundle does not measure
improvements in the current compiler.

## Measurements

The table gives ranges across the three repeats of each client's mean. All single-request trials complete all four
requests with zero failures and 32 output tokens per request.

| Engine | Input tokens | Mean TTFT, ms | Mean TPOT, ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Native | 32 | 1,317.96–1,673.33 | 34.96–46.97 | 11.48–11.60 |
| Stock | 32 | 7.12–7.95 | 2.34–2.35 | 395.30–399.50 |
| Native | 256 | 9,980.80–10,179.58 | 39.46–44.60 | 2.81–2.82 |
| Stock | 256 | 10.07–10.94 | 2.36–2.40 | 376.43–380.02 |
| Native | 1,024 | 41,222.02–41,761.74 | 35.24–46.99 | 0.747–0.750 |
| Stock | 1,024 | 29.02–30.69 | 2.547–2.549 | 291.41–295.69 |

Native output-token timings vary substantially between repeats; the experiment does not isolate that variation.
Its first-token cost scales with the sequential prefill, and dominates long requests. Both fixed-prompt responses
are exactly ` Paris. The capital of Italy is Rome. The capital of Spain is Madrid.` with sixteen output tokens.
That agreement supports the smoke check only, not general cross-engine equivalence.

The mixed-length trial completes one native request and rejects seven with HTTP 429, as the current admission
contract requires. Stock completes all eight without failures. The successful native request has TTFT 27,485 ms and
TPOT 45.97 ms; stock means are 32.08 ms and 2.99 ms. Output throughput is 1.11 versus 996.60 tokens/s, but the native
number excludes seven rejected requests and is not a like-for-like throughput comparison.

Whole-device memory snapshots after readiness are 3,356 MiB native and 10,989 MiB stock. These include desktop use
and different preallocated KV capacities. These figures do not establish equal-capacity
memory savings, allocator efficiency, or admission capacity.

## Interpretation and next decision

This artifact has no measured latency advantage over stock. The companion
[GPU profile](../native_profile/RESULTS.md) attributes 69.2% of short-request GPU kernel time to one repeated linear
kernel and 9.3% to greedy token selection. Improve those measured kernels and the sequential prefill before treating
continuous batching or paged KV as the primary performance remedy. Preserve the existing numerical contract when
qualifying new schedules. A faster kernel must be checked in the full model, not promoted on isolated timing alone.

Milestone 4 is not implemented by this experiment. Batching, paged KV, fairness, and multi-request cancellation
remain open. The evidence gate is complete; choosing schedule tuning versus expanding serving features remains an
explicit scope decision. No serving speedup or Rust-versus-Python speedup is claimed.

## Retained run

Run ID `2026-09-25_07-43-44`, source `9b77e1d8`, Ubuntu 24.04.5 LTS, Intel Core i9-14900K, driver 595.91.07.
Both experiment rows succeeded: `rtx4080x1_lnative_755c7a682bb6` and `rtx4080x1_lstock_44ae6b2bbfec`. Their successful
command status includes the deliberately observed native overload responses; it does not mean every HTTP request
succeeded. Package versions are retained in each row's `requirements.txt`; stock reports vLLM 0.23.0.

`results_rtx4080x1.tar.gz` contains that timestamped tree, both system-only experiment records, logs, identities,
memory snapshots, and raw JSON. Each row's `c1_i{32,256,1024}_r{1,2,3}.json` supplies the table; `mixed.json` supplies
the overload result and `generation.json` the fixed text response. No cloud infrastructure remains. The local
servers were terminated after measurement.

## Publication privacy

The shared archive is an anonymized copy. Usernames, hostnames, GPU UUIDs, PCI addresses, network addresses, local
paths, filesystem details, and uptime were removed. Archive ownership and timestamps were normalized. Numeric
benchmark measurements are unchanged. Original records and binary traces remain local. `ANONYMIZATION.txt` records
these transformations; the shared records are not the unmodified originals emitted by the harness.

The execution revisions above predate consolidation of unpublished commits to exclude the private archives from
Git history. Runtime and compiler source match merged PR #890 at `b4b3774c`; this work changes only experiment
recipes, reports, and the plan.
