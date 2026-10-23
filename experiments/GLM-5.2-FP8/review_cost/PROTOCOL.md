# Synthetic review inference costs

This experiment measures serving throughput for growing review sessions. It uses the FP8 checkpoint and FP8 KV cache
from the earlier serving experiment, with prefix caching enabled and MTP proposing three output tokens. It does not
measure review quality or establish that FP8 preserves recall.

Each session starts with distinct random tokens drawn from the model tokenizer's ordinary vocabulary. Sessions share
256 prefix tokens to approximate a system prompt and tool schemas. Each subsequent call resends the exact previous
prompt and generated token IDs, followed by 512 new input tokens. Calls within a session are sequential, with no tool
wait. Separate sessions run concurrently. A session stays on one data-parallel rank using vLLM's rank header.

| Scenario | Mean input per call | Output per call | Calls per session | Sessions per sweep point |
| --- | ---: | ---: | ---: | ---: |
| Median-shaped | 32,000 | 330 | 12 | 128 |
| P90-shaped | 60,000 | 590 | 28 | 128 |
| 130:1 ratio | 42,900 | 330 | 12 | 128 |

The initial prompt length centers linear context growth on the named input length. Median and p90 are separate
synthetic scenarios, not a reconstructed traffic distribution. The third scenario preserves the reported aggregate
input/output ratio. The 512-token increment and 256-token common prefix are assumptions, since traces are unavailable.

Every scenario sweeps 16, 32, 64, 96 and 128 active sessions against one deployment. The server admits at most 16
requests per rank, or 128 across the node. The client writes raw request start/end times, prompt lengths, backend
usage (including cached tokens), finish reasons, and total elapsed time. A missing cached-token count or shortened
output fails the workload. Each sweep point uses fresh prompts. Cold first calls and final drain are included.

Cache hit rate is measured. The user's 92% is a comparison target, not a value injected into usage. Cold first calls,
context growth, block boundaries, eviction and routing can change the achieved rate. Keeping generated tokens in the
next prompt allows reuse of decode work too. Synthetic random prompts do not reproduce real code or MTP acceptance.

Foreground total input tokens per second, split into cached and uncached input, over each complete measured window.
Include reviews per hour and cost per review. With this fixed workload, total input throughput and review throughput
rank configurations identically. More uncached input work caused by eviction is not an improvement by itself. These
rates include decoding and are not isolated prefill benchmarks. Interpret server phase timings alongside GPU counters:
overlapping per-request prefill and decode durations do not measure exclusive GPU time in either phase.

The best measured concurrency is the initial candidate. Check that throughput has stopped improving before declaring
an optimum; repeat the candidate and compare affinity and MTP if they materially affect the result. The billing
calculation is uncached input times its rate, plus cached input times its rate, plus output times its rate. A mixed
workload determines one break-even constraint, not three independent marginal prices. Report any allocation or
relative-price assumption explicitly, and charge each GPU-hour only once. Show full-utilization economics separately
from serving 260 reviews per day on a continuously rented node.

The recipe retains an `initial` phase and a `refinement` phase, selected with `--filter phase=initial` or
`--filter phase=refinement`. The refinement measures 56/64/72 concurrent median sessions, 24/32/40 p90 sessions, and
40/48/56 sessions for the 130:1 workload, with fresh seeds. The overlapping points provide independent repeat evidence.
These brackets follow the initial measurements: increasing concurrency can evict the session prefixes and reduce
throughput. If a point clearly loses cache reuse and throughput, an operator may stop that client; retain the failed
row, its partial log, the stopping reason, and the unmeasured points. Partial points are not completed throughput
measurements. Later refinement includes all three scenarios rather than concealing an earlier failed row.

The user narrowed the search to the median workload after the initial sweeps. Preserve the earlier p90 and 130:1
evidence and explain any deliberately stopped follow-up rows. The objective is maximum sustained throughput, with no
TTFT constraint. The `prefill` phase measures a 2,048-token baseline at 64 concurrent reviews, then token budgets of
8,192 at 48/64 reviews and 16,384 at 32/48 reviews. Every point runs 128 reviews; seed offsets give overlapping
concurrencies identical prompts across token budgets. Check the resulting KV capacity and cache retention as well as
completed throughput; a larger token budget can change memory available for prefixes. Extend a winning boundary and
confirm the useful setting. Costs for other scenarios remain outside this narrower optimization claim.

Capture timestamped per-GPU activity, memory activity, memory allocation, power and clocks every second with
`nvidia-smi`, and raw server metrics every five seconds. Session start/end records carry Unix timestamps to align
telemetry with each measured window. Keep deployment and idle periods separate from client measurements. GPU activity
does not measure SM occupancy or useful compute efficiency: communication kernels can keep a GPU active. Use activity
gaps, power, queue depth, cache usage and throughput together to guide the next configuration.

The `controls` phase follows the measured memory limit: the 8,192-token MTP3 deployment exhausted memory available
for KV blocks. First test a 0.96 memory budget at 72/80 sessions with the original 2,048-token prefill budget. Then
disable MTP and measure 64/96/128 sessions at 2,048 prefill tokens, and 48/64 at 8,192 tokens. The MTP-off rows admit
24 requests per rank so the server limit does not stop the measured ladder at 128. These controls test whether draft
weights, activation memory and cache capacity prevent useful batching. Keep any allocation failures; do not treat
failed deployment as zero throughput. Repeat a useful winner with enough reviews to sustain several waves of work.

`scripts/rent_spot.py` retries exhausted GCP capacity using Emmy's provisioning and owned-lease mechanisms. It discovers
which zones advertise the requested machine, visits the selected zones sequentially, and waits five minutes between
exhausted sweeps. Authentication and configuration errors stop it. With recipe arguments, it invokes `emmy bench` on
the rented host and deletes the owned VM after the benchmark exits, including failure. A rent-only invocation leaves
the lease available to the caller. The experiment is complete only after the raw evidence and cost analysis are saved.
