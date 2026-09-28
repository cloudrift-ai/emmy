# Chunked native Qwen3 prefill

Fixed-width prefill reduces time to first token by 2.5–7.9× and raises output throughput by 1.67–6.30× against
sequential prefill on one RTX 4080. All 36 paired benchmark completions match. Both artifacts use
identical one-token decode plans, standard math, FP16 weights and KV storage, and FP32 residuals and attention
intermediates. The native server still admits one active request. This is not a comparison with stock vLLM.

## Later output-precision update

The [output-precision investigation](../native_accuracy/RESULTS.md) replaces the FP16 output-head program in the
shared golden with an FP32-output program. The file now contains 50 rows across five programs, and preparation
writes generation format 4. The measurements below remain the historical format-3 comparison; reproducing them
requires the source and golden revisions recorded here. Current preparation uses the updated head and sampler.

## Update after merging main

Main commit `6556d75e2` (#914) changes cut workspace dtypes and kernel identities. The targets remain fresh, but
35 receipts need new identities, covering 24 distinct kernels. Comparing the generated CUDA shows that the cut
stores now apply the FP16 conversion previously performed by readers. The output-head kernels are unchanged.

Four selected routes were remeasured on the same RTX 4080 with standard math, five warmups, twenty iterations,
CUDA graphs, and strict eager checks. M=1 pre/post measure 66.2/78.5 µs; M=16 pre/post measure 89.0/129.0 µs.
The latter are close to the original 89.4/128.6 µs fragment observations. All four accuracy checks pass. The
current golden keeps 52 rows, including the freshly recorded routes and unchanged output-head evidence. Thirteen
unselected alternatives with stale timings are retired from the deployable file; the original archive preserves
their measurements. No prior-led search is used.

All 52 rows strictly decode and all five targets remain fresh. A new width-16 artifact exports from an empty tune
DB with strict evidence. The short-prompt and both held-out prefill checkpoint cases pass (three checks in
180.58 seconds). The full suite after the merge passes 7,473 tests and skips 795.
Python lint, Rust formatting, and Clippy pass.

The serving latency, full checkpoint qualification, and memory figures below describe the original revision and
artifacts named in the evidence section. They were not remeasured after this merge. Additional fragment records,
identity mappings, retired alternatives, and validation logs are retained under the archive's `checks/` directory;
new checkpoint measurements are under `merge-qualification/`.

## Implementation and reproduction

The checkpoint is `Qwen/Qwen3-0.6B` at revision `c1899de289a04d12100db370d81485cdf75e47ca`. Prefill processes all but
the final prompt token in chunks. Each query attends only to its causal prefix. Padding never writes the cache.
The final prompt token uses decode to compute logits and sample. Intermediate chunks omit the output head and
sampler; the last layer also omits attention and its post-attention fragment. Decode and prefill have independent
CUDA graphs and scratch slabs, with shared weights, request inputs, and KV allocations.

Generation artifact format 3 requires re-exporting older bundles. `--prefill-size 1` selects sequential execution;
16 is the default. The historical
[golden](https://github.com/cloudrift-ai/emmy/blob/a98fd4f851be305a225ffce7bac2ad61b2892c8a/experiments/Qwen3-0.6B/native_prefill/golden/rtx4080_sm89.json) includes the three existing one-token programs and two new
width-16 programs. Its five targets are fresh, all 52 rows strictly decode, and a fresh tuning database suffices for
strict-evidence export. All schedules are manually selected from explicit measured candidates. No MCTS or prior-led
search is used.

Prepare two serving bundles with the existing library, on the target card, with the matching native binaries built:

```python
from pathlib import Path
from emmy.serving.native.launch import prepare

for label, width in (("baseline", 1), ("chunked", 16)):
    prepare(
        "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca", Path("/tmp/native-prefill") / label,
        4096, "experiments/Qwen3-0.6B/native_prefill/golden/rtx4080_sm89.json", True, prefill_size=width,
    )
```

Use `EMMY_FAST_MATH=0`, `EMMY_NVCC_FLAGS=` and a fresh `EMMY_TUNE_DB` for that preparation. The serving recipe takes
`BASELINE_ARTIFACT` and `TUNED_ARTIFACT` and runs through `emmy bench experiments/Qwen3-0.6B/native_prefill --local`.
It holds the shared GPU lock for each row. Both lanes use the same server binary, greedy sampling, CUDA graphs,
concurrency one, 32 output tokens, four measured requests and one warmup, input lengths 32/256/1024, and seeds 1–3.
Each lane also has a separate short CUDA profile with three fixed text requests. Binary profiler reports stay local;
the archive retains their text summaries.

## Serving measurements

All 72 measured requests complete with 32 output tokens each and zero failures. All 36 paired generated texts match;
the fixed completion probes and profile requests also match. Ranges are the three repeats of each client's mean.
Paired TTFT speedups are 2.51–3.92× at 32 input tokens, 6.84–7.54× at 256, and 7.23–7.87× at 1,024. Output throughput
improves by 1.67–1.69×, 4.21–4.22×, and 6.15–6.30× respectively. Decode latency remains similar, consistent with the
identical decode plans.

| Prefill | Input tokens | Mean TTFT, ms | Mean TPOT, ms | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Sequential | 32 | 162.81–204.74 | 4.03–5.36 | 97.05–97.18 |
| Chunked | 32 | 41.52–81.45 | 3.70–5.01 | 162.21–164.07 |
| Sequential | 256 | 1,378.59–1,399.17 | 5.39–6.08 | 20.41–20.43 |
| Chunked | 256 | 182.99–204.69 | 5.40–6.08 | 85.92–86.08 |
| Sequential | 1,024 | 6,764.76–6,927.00 | 6.37–8.60 | 4.45–4.56 |
| Chunked | 1,024 | 878.31–943.54 | 6.36–8.49 | 28.03–28.05 |

The short profiles confirm captured execution: 60 graph launches for sequential prefill and 51 for chunked prefill
across three identical five-token prompts with sixteen generated tokens. Sampler launches fall from 60 to 48;
chunked prefill omits those launches entirely, while the sequential program's intermediate sampler calls return
without sampling. The profile is a short structural check, not the serving latency measurement above.

## Manual fragment selection

The selected width-16 pre-attention fragment measures 89.4 µs; post-attention measures 128.6 µs in the fresh-cache
replay. Each captured run uses five warmups and twenty iterations, checks against eager, and uses deployable NVCC
optimization. These are fragment observations, not whole-model measurements.

The pre-attention cooperative candidate measured 899.1 µs. Direct projection schedules with measured cooperative
normalizations are faster. Two tensor-core proposals for this fragment fail exact-pin integrity and are rejected;
their fallback observations are not reported as tensor-core results. Post-attention's direct candidate measured
244.2 µs; the tensor-core gate/up and down projections reduce that to 128.6 µs. A global cooperative post-attention
candidate fails strict eager agreement at one of 16,384 elements and is excluded. All failed trials remain in the
archive. The original one-token evidence was unchanged in that experiment; the main merge below refreshes affected rows.

## Numerical qualification

Sequential execution retains the existing per-position absolute limits, reference-token agreement, and whole-prompt
RMS comparison. Chunked execution has no intermediate prompt logits, so it retains the absolute limits and token
agreement on every emitted logit vector and adds a matched sequential comparison for its shorter RMS window.
The same prefix is teacher-forced through sequential execution; the chunked RMS must meet the reference-based
bound or be no worse than that matched sequential baseline. This extra baseline floor is explicit in the test and
measurements. It is not described as the original whole-prompt RMS check.

The initial tail-only application of the old RMS formula rejects `heldout_context_4096`: probability TV RMS is
0.0011139 against a 0.0009766 epsilon floor. The previously qualified sequential artifact also exceeds that floor on
exactly those final 17 positions (0.0011164); its original whole-prompt comparison passes. Changing prefill to the
cooperative projection schedule gives 0.0011216 and does not repair this window mismatch. Both failed checks are
retained. The paired test preserves the sequential whole-prompt contract and checks the new output window against
the baseline on equal terms. Two additional prompts, including another 4,096-position case, were fixed before
running that revised qualification. The absolute error limits, token agreement, and precision mode are unchanged.
The chunked RMS criterion includes the explicit matched-baseline floor described above.

The expanded checkpoint/HTTP run finishes with 38 passes and one sequential-only failure in 1,244.01 seconds.
All nineteen chunked cases pass across 369 emitted logit vectors, including both fresh prompts. Their maximum
relative L2 error is 0.007152 and maximum probability TV is 0.004583; every argmax matches both references.
All nineteen paired diagnostic output sequences match sequential execution exactly. Eighteen of
nineteen sequential cases pass. Across 15,712 sequential positions, the maximum relative L2 error is 0.018090 and
maximum probability TV is 0.016688, with the one token mismatch described below. The native HTTP lifecycle test
passes, including seeded streaming, stop handling, cancellation recovery, and shutdown. These are correctness
measurements and diagnostic transfers, not serving timings.

The new long prompt also exposes an existing sequential token-agreement gap at prompt position 1,064. Native
execution chooses token 576; both references choose 785, with an FP32 logit margin of 0.026894. The relative L2 error
is 0.001810 and probability TV is 0.006687, within the absolute limits, but token agreement still fails. A separate
sequential-only artifact reproduces token 576 when the prompt is truncated to 1,065 tokens and one token is generated.
This uses no chunked program. The new qualification case remains a failure rather than an expected failure or a
relaxed token check. It limits the numerical claim and needs a separate investigation of the existing decode path.

## Memory and padding

At context capacity 4,096, the decode plan allocates 1,977,820,968 bytes, including 448 MiB of KV state and a
262,144-byte scratch slab. Prefill shares 615 named allocations and adds a 917,824-byte scratch slab plus the
65,536-byte final rotary-query output: 983,360 bytes (0.94 MiB) resident growth. These are execution-plan allocation
sizes, excluding driver state, loaded modules, graph storage, and allocator overhead.

Loading currently allocates and uploads prefill before lending decode's shared regions. The temporary plan-buffer
total is 3,621,821,540 bytes, compared with 1,978,804,328 resident bytes after sharing. Removing that duplicate load
allocation is separate work. The serving recipe also records process-external GPU memory observations, which include
other desktop activity and are not substituted for the plan accounting.

| Prompt tokens | Useful prefill rows | Chunks | Scheduled prefill rows | Padded rows | Final decode row |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 31 | 2 | 32 | 1 | 1 |
| 256 | 255 | 16 | 256 | 1 | 1 |
| 1,024 | 1,023 | 64 | 1,024 | 1 | 1 |

Projection work still executes for padded rows. One-token prompts bypass prefill. Width three in the tiny-model test
covers complete and partial chunks, including padding beyond context capacity, graph reuse, request reset, EOS,
zero output budgets, and seeded sampling.

## Evidence and limits

Run `2026-09-27_06-56-54` (`20260927T065654Z`) begins at 06:56:54 UTC on September 27, 2026, on clean source
`ac92448e4493ff7406f99a6192d6127ec32b5149`. All four rows succeed: baseline profile `285c9ca00372`, baseline serving
`7564b9a30c55`, chunked profile `d10739759c9e`, and chunked serving `0b8e77879a19`.

Hardware is one NVIDIA GeForce RTX 4080 with 16,376 MiB, an Intel Core i9-14900K, and driver 595.91.07. Software is
Ubuntu 24.04.5 LTS, NVCC 13.3.73, cuBLAS 13.6.0.2, PyTorch 2.11.0, Transformers 5.14.1, and vLLM 0.23.0 as the client.
Observed total GPU use is 2,847 MiB for the sequential serving process and 2,858 MiB for chunked serving; those
observations include desktop and driver activity. The allocator accounting above is the controlled memory comparison.

The sequential manifest SHA-256 is `9f0e068ca50cbeaeb3f018272c9a0becfcde935353166bfa667d3e27b4a1f42c`; the chunked
manifest is `f76d26fad3c85ec8012ce1e2b5e59cffd66d625fdfbbe86c27810c0b74e4b94c`. A fresh canonical export has identical
plans, weights, and binaries to the artifact used for qualification; 1,170 payloads were compared. The raw identity
files record the shared server binary and tokenizer hashes as well.

[The platform archive](results_rtx4080x1.tar.gz) retains all four system-only experiment records, each lane's
`c1_i{32,256,1024}_r{1,2,3}.json` and logs, profile summaries and fixed requests, and software and artifact identities.
`qualification-final/` contains all 38 numerical records; `qualification/` and `qualification-coop/` retain the
initial failed qualification and the alternate schedule trial. `checks/` contains fragment candidates, strict
replays, the independent sequential-only failure reproduction, allocation layouts, and validation logs.
`ANONYMIZATION.txt` describes the publication redactions. Original records and binary profiler captures stay local.

Final default suite: 7,421 passed and 795 skipped in 865.50 seconds. The checkpoint qualification above is opt-in
and is not included in that passing default count. All 20 Rust unit tests pass. Python lint, Rust formatting,
and Clippy pass. New checkpoint-case durations are recorded for test scheduling.

Results come from one shared desktop GPU and sequential lanes. They do not establish an optimal chunk width,
continuous batching, paged KV, other-model support, or an advantage over stock vLLM. The checkpoint artifact and
model weights remain local. The publication copy removes machine/account identities, GPU UUIDs, PCI/network
addresses, filesystem metadata, and uptime while preserving software versions and numeric measurements.
