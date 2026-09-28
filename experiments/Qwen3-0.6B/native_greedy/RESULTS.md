# Parallel greedy selection on RTX 4080

Greedy decode time per output token falls by 1.51–1.81× across paired repeats. Short-prompt output throughput rises
by 1.42×. All 36 paired measured completions match. Long-prompt throughput improves only about 2% because sequential
prefill is unchanged.

## Results

Ranges below span the three repeat means; comparisons pair the same input length and seed. These are serving
measurements, not profiler timings.

| Sampler | Input tokens | Mean TTFT (ms) | Mean TPOT (ms) | Output tokens/s |
| --- | ---: | ---: | ---: | ---: |
| Serial | 32 | 166.852–241.469 | 7.236–9.630 | 68.669–68.770 |
| Parallel | 32 | 162.587–204.123 | 4.009–5.358 | 97.267–97.404 |
| Serial | 256 | 1378.809–1414.274 | 9.178–10.338 | 18.829–18.835 |
| Parallel | 256 | 1372.536–1393.482 | 5.365–6.049 | 20.510–20.513 |
| Serial | 1024 | 6737.742–6830.063 | 9.573–12.783 | 4.484–4.490 |
| Parallel | 1024 | 6731.700–6778.551 | 6.317–8.450 | 4.575–4.588 |

Paired TPOT ratios are 1.795–1.805× at 32 input tokens, 1.709–1.712× at 256, and 1.513–1.515× at 1,024.
Output throughput ratios are respectively 1.415–1.418×, 1.089×, and 1.020–1.022×. TTFT ranges overlap; these data
support a decode improvement, not a prefill improvement. Some repeat means shift between TTFT and TPOT while total
throughput stays stable, so the paired ratios and full-request throughput are more informative than a single mean.

The profiles contain sixty sampler launches each, including twelve intermediate prefill returns. Sampler GPU time
falls from 235.095 ms to 11.066 ms, or 21.24×; its share falls from 42.4% to 3.3%. Median sampler duration falls
from 4.873 ms to 0.231 ms. Total profiled GPU kernel time is 554.700 ms versus 335.007 ms. The profiles include
initial graph capture, so they locate the improvement but do not replace the repeated serving measurements.

The remaining sampler cost is small enough that a second reduction stage is not justified by this experiment.
Chunked prefill is the next larger opportunity for long requests.

## Question and protocol

Does replacing the native sampler's serial greedy scan improve serving with the qualified manual Qwen3 schedules?
The baseline and candidate use identical model weights, compiled model plans, precision, context, and server binary.
Only the bundled native CUDA source and sampler launch geometry differ. The source change is confined to greedy
selection and the thread-zero guard around the existing positive-temperature sampler.

The checkpoint is `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`. Both artifacts were
prepared on main `7be6b330`, with the candidate sampler from `0f20e386`. Each export uses an isolated tuning cache,
standard math, strict evidence, and the [manual golden](../native_manual_schedules/golden/rtx4080_sm89.json).
No prior or search selects schedules. All bound weight bytes and the compiled model plans compare equal.
The unchanged manifest does not carry compiler provenance; preparation records and the explicit source revisions
above identify this comparison. The manifests are identical; `qualification/artifact-comparison.json` records distinct
SHA-256 hashes of both decode plans, which include the CUDA sources and launch geometry. The recipe now hashes those
plans alongside its other identity files; the measured run predates that additional identity-only command.

The [recipe](recipe.yaml) runs two artifacts through two modes on one local RTX 4080:

- Profiling: Nsight Systems CUDA node tracing over three fixed-prompt requests, each producing sixteen tokens.
  This includes first-request graph capture and intermediate prefill sampler calls that immediately return.
- Serving: concurrency one, greedy decoding, exactly 32 output tokens with EOS ignored, and input lengths
  32, 256, and 1,024. Each length has three repeats with seeds 1–3, four measured requests and one warmup per repeat.
  The vLLM benchmark client uses identical seeds for both artifacts. These are 72 measured requests across both.

The server uses CUDA graphs in both modes. The recipe holds the shared GPU lock throughout each row. Rows run
sequentially, baseline before candidate; desktop activity and execution order remain timing limitations. The recipe
reuses prebuilt artifacts, so preparation and compilation are outside measured serving time. Set `BASELINE_ARTIFACT`
and `TUNED_ARTIFACT` to the serial and parallel serving bundles, then run:

```sh
emmy bench experiments/Qwen3-0.6B/native_greedy --local
```

## Scope

The new sampler reduces the vocabulary in one 128-thread block using 512 bytes of shared memory. It preserves exact
argmax choices, lowest-ID ties, signed-zero ties, rejection of every nonfinite logit, and skipped intermediate prefill
steps. Positive-temperature sampling retains its existing serial arithmetic and RNG. The artifact format and Rust
runtime protocol are unchanged. Existing artifacts must be re-exported to receive the optimization.

Positive-temperature latency was not benchmarked. This experiment does not claim optimized prefill, production
batching, or an advantage over stock vLLM. Sequential prefill, continuous batching, and paged KV remain separate work.

## Retained evidence

Run `2026-09-27_01-17-38` began at 01:17:38 UTC on September 27, 2026. All four rows succeeded:
`rtx4080x1_lbaseline_mprofile_285c9ca00372`, `rtx4080x1_lbaseline_mserve_7564b9a30c55`,
`rtx4080x1_lparallel_mprofile_f266a167bbfe`, and `rtx4080x1_lparallel_mserve_8934f53b04d8`.
The hardware is one RTX 4080 with 16 GiB VRAM and an Intel Core i9-14900K. Software is NVIDIA driver 595.91.07,
CUDA toolkit 13.3.73, cuBLAS 13.6.0.2, PyTorch 2.11.0, Transformers 5.14.1, NumPy 2.3.5, and vLLM 0.23.0 as client.

[The platform archive](results_rtx4080x1.tar.gz) retains the four system records, both profiles' `kernels.csv`,
per-repeat serving JSON and logs, fixed-prompt outputs, software inventories, and artifact identity records.
Serving measurements are in each serving row's `c1_i{32,256,1024}_r{1,2,3}.json`. Profile requests are in
`request{1,2,3}.json`. The `qualification/` directory holds correctness and gate logs and artifact comparisons.
`ANONYMIZATION.txt` describes removed machine identities, addresses, account paths, filesystem details, and normalized
tar metadata. Numeric JSON measurements and software versions are preserved. Raw originals, binary profiler traces,
and model weights remain local.

## Correctness and validation

Five reduction cases cover vocabulary sizes 1, 31, 128, 129, and 151,936. They check NumPy argmax agreement for
random logits, extreme finite logits, cross-thread and cross-iteration ties, signed zeros, and nonfinite values at
both ends. Both ordinary execution and graph replay pass, and intermediate prefill preserves the prior selection.
The existing independent nucleus-distribution test also passes with the new launch geometry.

Twelve paired checkpoint cases compare exact token IDs for English, arithmetic, and Japanese prompts, greedy and
seeded temperature/top-p sampling, with and without CUDA graphs. All 192 generated tokens per artifact match.
This isolates sampler equivalence on the same compiled model; it does not repeat the broader model-quality claim
from the manual-schedule qualification. All 36 paired benchmark completions also match exactly.

The scoped correctness run passes six cases. The correctness-lane duration and recipe run passes fifty tests, with
new case durations recorded. Cargo workspace tests, Ruff, Rustfmt, and Clippy pass.

The full suite finishes with **7,410 passed, 774 skipped, and one failure** in 786.48 seconds. The failure is
`test_expert_program_fp8_indirect_compose`, outside the native sampler. On an isolated checkout of main `7be6b330`,
with a rebuilt current-source runtime extension and a fresh tuning cache, it fails identically: 30 of 32 elements
mismatch, with greatest absolute difference 1.25390625. Neither its implementation nor its expectations are changed
here. The full suite is therefore not claimed green.

The initial suite used the previously installed in-process extension. After refreshing it without changing dependency
versions, all six sampler checks pass again in 38.51 seconds. The serving measurements and paired checkpoint checks
use the separately rebuilt current-source standalone binaries throughout. `full-suite.log`, `main-fp8.log`, and
`rebuilt-runtime-sampling.log` preserve these validation limits and results.
