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

The best measured concurrency is the initial candidate. Check that throughput has stopped improving before declaring
an optimum; repeat the candidate and compare affinity and MTP if they materially affect the result. The billing
calculation is uncached input times its rate, plus cached input times its rate, plus output times its rate. A mixed
workload determines one break-even constraint, not three independent marginal prices. Report any allocation or
relative-price assumption explicitly, and charge each GPU-hour only once. Show full-utilization economics separately
from serving 260 reviews per day on a continuously rented node.

`scripts/rent_spot.py` retries exhausted GCP capacity using Emmy's provisioning and owned-lease mechanisms. It discovers
which zones advertise the requested machine, visits the selected zones sequentially, and waits five minutes between
exhausted sweeps. Authentication and configuration errors stop it. With recipe arguments, it invokes `emmy bench` on
the rented host and deletes the owned VM after the benchmark exits, including failure. A rent-only invocation leaves
the lease available to the caller. The experiment is complete only after the raw evidence and cost analysis are saved.
