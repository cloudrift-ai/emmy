# DeepSeek V4 Flash 0731 serving on 16× V100 SXM3

## Conclusion

Both arms now serve this checkpoint and both produced serving numbers. The fork is **6.1× faster per output token**
and **23× faster to first token**. The two arms were measured in separate invocations rather than one alternating run,
so this is a directional comparison, not the balanced A/B described under "What this run does not establish".

Getting there meant fixing one kernel. `pre1.k_linear_mean_reduce_7fce9f` at M=1 carried a single measured candidate
in the golden, at 29.7 s, and under strict evidence the election had no alternative to elect. Single-token decode ran
at a flat ~61 s per layer, so no generation request ever returned: each died on the engine's 300 s `sample_tokens`
deadline, and raising that only moved the failure to the 600 s NCCL collective watchdog. Four placement cuts take that
program to 473.6 µs whole-program, which took time per output token from **5.565 s to 0.899 s**.

Three programs were fixed earlier in this work — `pre4096` by 594×, and the decode program's two hot kernels by 4.3×
and 5.6×. The M=1 kernel above is the same `k_linear_mean_reduce` as `pre4096`, recomputing the same loop-invariant
dot products inside the same output-channel sweep, and it took the same four cuts; only the m4096 shape had been
recorded. What now dominates is `post.decode.m16` at 1,149× its roofline floor and `post.decode.m1` at 293×, neither
of which has been recorded.

At a 4,096-token context the fork serves this checkpoint cleanly: 480 requests across fifteen rows, zero failures,
and three workload shapes that differ far more in repeat stability than the shapes themselves suggest. Single-stream
decode is the steadiest measurement on this stack by a wide margin — 6.46 tok/s on all five repeats, with mean time
per output token inside a 0.2 ms band. The eight-way concurrent shape is the least steady, spanning 29.03 to
38.31 tok/s across its five repeats; a comparison built on that row alone could show a 30% difference from run-to-run
noise and nothing else.

This also supersedes the earlier 30.79 tok/s figure as a reference point. That number was measured at a 1,048,576-token
context, which the Emmy arm cannot hold, so it was never a legitimate baseline for a compiler comparison. The rows
below are at the envelope both arms can share.

On 2026-09-15 the repository golden itself served the checkpoint for the first time — `main` plus #807, strict
evidence, nothing host-local in the process: 2.03 s per output token and 44 s to first token at 2,275 input
tokens, 13.8× and about 12× off the fork. The gap to the 0.899 s above is one election: the repository golden
runs the M=1 post-attention decode at 45 ms per layer where the host-local file elected 18 ms. Details under
"The repository golden serves".

On 2026-09-17 `main` served again after the identity re-key of #804, from the golden of #826: 3.30 s per output
token and 45.5 s to first token at 2,275 input tokens. Time to first token is where it was. Decode is 1.27 s slower
because strict evidence now refuses the M=1 tier, so single-token decode rides the width-16 twins. Details under
"Main after the identity re-key".

## Measurements

| Concurrency | Input → output | Repeats | Output tok/s, mean ± SD | Range | Mean TPOT | Mean TTFT | Failed |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 128 → 128 | 5 | 33.71 ± 3.24 | 29.03 – 38.31 | 221.6 ms | 2.50 s | 0 |
| 4 | 1024 → 256 | 5 | 19.99 ± 0.39 | 19.32 – 20.41 | 179.7 ms | 5.41 s | 0 |
| 1 | 2048 → 512 | 5 | 6.46 ± 0.00 | 6.46 – 6.47 | 147.7 ms | 3.77 s | 0 |

Spread is the population standard deviation over the five repeats. Per-repeat output throughput:

- Concurrency 8: 38.31, 29.03, 33.71, 31.58, 35.94 tok/s
- Concurrency 4: 19.88, 20.01, 20.32, 19.32, 20.41 tok/s
- Concurrency 1: 6.46, 6.46, 6.46, 6.47, 6.46 tok/s

Every repeat redeploys the server, so each row carries its own model load and warm-up; the stability above is therefore
across fresh processes, not across clients hitting one long-lived server.

The single-stream row is the one worth building the Emmy comparison on. Its inter-token latency and time per output
token agree to two decimal places within a repeat and to 0.2 ms across repeats, so a real difference between arms would
be visible far below the noise floor of the other two shapes.

## The Emmy arm

The Emmy arm serves. The numbers below come from a serving boot on the same host driven by direct HTTP requests, not
from `emmy bench`, so they carry no experiment records and no archive. The compiler measurements that follow them are
kept because they explain where the time goes.

### Serving measurements

Boot on 2026-09-12, `--strict-evidence`, `gpu_memory_utilization` 0.90, prefix caching on, the M=1 decode tier
enabled. Greedy decoding, single stream, streamed responses timestamped per chunk.

| Shape | TTFT | TPOT mean | TPOT range | Output tok/s |
| --- | ---: | ---: | ---: | ---: |
| 5 in → 33 out | 7.05 s | 0.899 s | 0.894 – 0.906 | 1.11 |

The first decode step of a request costs 1.85 s rather than 0.90 s: each layer's M=1 programs are CUDA-graph captured
on first use, and a program is captured once per layer per worker. Thirty-one steady steps span 1.3%.

Against the fork's single-stream row, which is the steadiest measurement on this stack:

| | 1Cat fork | Emmy | Ratio |
| --- | ---: | ---: | ---: |
| TTFT | 3.77 s (2,048 in) | 88.22 s (2,405 in) | 23× |
| Mean time per output token | 147.7 ms | 899 ms | 6.1× |
| Output tok/s, decode only | 6.77 | 1.11 | 6.1× |

The time-to-first-token row is from the pre-fix boot and is unchanged by this work: prefill runs the m16 and m4096
twins, which the fix did not touch. `post.chunk.m4096` at 317× its ~1,955 µs floor is 619 ms per layer and is what
dominates it.

Correctness at the serving level: greedy completions are coherent English ("Red, blue, and green are three classic
colors often used as primary colors in light-based systems (RGB)…"). There is no eager twin for a golden Loop IR
target on this model, so no CLI-level numerical verdict was obtainable for the cut schedule alone. The cuts are code
motion — hoisting a loop-invariant dot out of a sweep — which is why the equivalent `pre4096` cuts were bit-exact.

### The repository golden serves (2026-09-15)

Every number above was measured against a host-local golden; the repository's own golden could not answer a
request. On 2026-09-15 it did, from `main` at `7e9336e6` plus #807, booted with `--strict-evidence` and an empty
tune DB so the golden was the only measured evidence in the process. The boot reached health in twelve minutes —
engine init, profiling, KV creation and warm-up took 83 s against 949 s on 2026-09-11 — and strict evidence
refused the same single fork as before, the expert `m256` twin on all sixteen workers. KV capacity was 76,337
tokens on the first stage and 78,722 on the second.

Greedy decoding, single stream, streamed responses timestamped per chunk. Prefix caching is on in the launcher, so
a repeated prompt's time to first token is a cache hit; the cold column is the first request at each shape.

| Shape | TTFT, cold | TTFT, repeat | TPOT mean | TPOT range | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 in → 33 out | 6.14 s | 3.37 s | 2.032 s | 2.019 – 2.045 s | 0.49 |
| 2,275 in → 9 out | 44.23 s | 16.02 s | 2.089 s | 2.085 – 2.095 s | 0.48 |

The first decode step of the first request cost 4.20 s, the per-layer graph captures; every later step, including
the first of the second request, was inside the range above. Against the fork's single-stream row: 13.8× per
output token (2.03 s against 147.7 ms) and about 12× to first token (44.2 s at 2,275 input tokens against 3.77 s
at 2,048) — directional, for the reasons under "What this run does not establish".

**Where the time goes now.** The boot's roofline audit, identical on the first layer of each stage, beside the
2026-09-12 boot that produced the 0.899 s:

| Program | Measured | Over floor | 2026-09-12, host-local golden |
| --- | ---: | ---: | ---: |
| `pre.chunk.m4096` | 2.74 ms | 92× | 107× |
| `post.decode.m1` | 44.7 ms | 750× | 294× |
| `post.decode.m16` | 68.3 ms | 1,145× | 1,156× |
| `post.chunk.m4096` | 688 ms | 352× | 321× |

The pre decode programs no longer appear at all: #793 took the pre-attention family from seconds to
microseconds, which is why time to first token at 2.3k input tokens halved from 88 s. The post family is
untouched. And one election went backwards: the host-local golden elected an M=1 post-attention decode at about
18 ms per layer; the repository golden, which carries the same cut since #799, elects 44.7 ms. Forty-three layers
of that difference is the 1.1 s per token between the two boots. Why a recorded cut loses the election is the
open question ahead of any new recording.

**Greedy agreement against the fork**, the four-prompt corpus of 2026-08-26 at temperature 0, 32 tokens each,
compared with the fork arm's dumps of 2026-08-27:

| Prompt | Agreement | At the divergence |
| --- | ---: | --- |
| code | 32 / 32 | — |
| medium | 32 / 32 | — |
| short | 5 / 32 | ` Spain` (−1.114) against ` Italy` (−1.324) — the same near-tie as in August |
| long | 1 / 32 | ` is` (−1.075) against `.` (−0.869) — agreed on all 32 in August |

Both divergences are ties of about 0.2 nats and both continuations are coherent, but the long prompt — the one
that spills past the sliding window into the compressed and indexed attention layers — did not diverge in August.
This is the greedy half of gate (d) only; no tensor-level comparison was run. Evidence on the host under
`~/serve-evidence/boot20-*` and `emmy_arm_boot20.json`.

**What blocked it until now.** #801 found that with the expert rows recorded, the expert compile died at plan
construction with a `RangeOp` node the backend could not place. That was not a lowering gap in the expert path.
#793 had spelled the mxfp4 nibble shift as a two-lane range under a broadcast; the constant-fold pass deferred the
range to the broadcast as its maximal root, and the broadcast root keeps scalar computation lazy for the kernel to
inline. Nothing lowers a range, so it reached the kernel as an input no plan could feed. Neither `emmy compile
--ir cuda` nor a `--golden --realization` replay reaches plan construction, which is why the failure was invisible
off the serving path. #807 folds a range where it stands; the expert kernels change identity as a result, and
#801's rows still deploy by structural match.

### Main after the identity re-key (2026-09-17)

#804 re-keyed every kernel identity on 2026-09-15 and left this golden's rows behind; #815, #823, #825 and #826
brought them back. On 2026-09-17 `main` at `3b5cc4ca` booted from the golden of #826 with `--strict-evidence` and an
empty tune DB: health in fifteen minutes, engine init 81.7 s, KV capacity 75,759 and 78,127 tokens. Same probe as
2026-09-15, greedy, single stream, streamed.

| Shape | TTFT, cold | TTFT, repeat | TPOT mean | TPOT range | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 in → 33 out | 6.33 s | 3.60 s | 3.302 s | 3.290 – 3.319 s | 0.30 |
| 2,275 in → 9 out | 45.51 s | 16.61 s | 3.352 s | 3.345 – 3.363 s | 0.30 |

| Program | Measured | Over floor | 2026-09-15 |
| --- | ---: | ---: | ---: |
| `pre.chunk.m4096` | 2.73 ms | 92× | 2.74 ms |
| `post.decode.m1` | not deployed | — | 44.7 ms |
| `post.decode.m16` | 69.8 ms | 1,175× | 68.3 ms |
| `post.chunk.m4096` | 629.7 ms | 322× | 688 ms |

Prefill is unchanged and decode is slower, for one reason: strict evidence refuses the M=1 post twin and the expert
M=1 twin, so a single-token step runs the width-16 programs — 69.8 ms per layer against the 44.7 ms the M=1 program
cost on 2026-09-15, which is the 1.27 s per token. Three kernel sets stand between `main` and an M=1 tier, and none
is a stale row any more. The M=1 division cut mints a piece that no recorded schedule fits and that does not build
under the schedule the prior picks: the cut splicer renames a workspace read but not the values derived from it, so
the piece declares two values twice and nvcc refuses it. The expert M=1 cut now mints one kernel where it minted two,
and the prior's schedule for it runs about 4 s per launch, past the bench budget; it needs a tune. And #799's cut,
re-recorded on `main` at 777 µs, is still refused at its residual kernel, whose only measured row is the all-OFF
schedule — that refusal is not understood yet. The fourth, the `9e578e` cut, was re-recorded at 9.4 ms for the whole
M=1 program and elects under strict evidence.

Two boots failed before this one, and each named a way a golden can decode in full and still not deploy. The strict
decode accepts a row when any kernel of its cut set enumerates it; the deploy needs the kernel the row names to
enumerate it. Seven receipts sat on the wrong one of several same-shaped kernels, strict evidence refused the
width-4096 prefill twin, and vLLM had no prefill bucket to start with. Then the boot hung in the roofline audit, which
has no time limit: a schedule row that carries the identity a cut fork is offered on reads as the fused kernel's own
receipt, two empty rows of the pre-attention targets carried it with 19 µs and 130 µs, and the fused arm outbid the
measured cut — one fused kernel whose single launch ran past fifteen minutes. Whether the same mechanism explains the
election of 2026-09-15 is not established: that boot ran a tree from before #804, whose rows carried their own
identities. No greedy-agreement run was made on this boot. Evidence on the host under `~/serve-evidence/boot21-*`,
`boot22-*`, `boot23-*` and `elect825-*`.

Main moved again the same day, after this boot: #813 retuned the post family and re-recorded both M=1 post cuts, and
#827 fixed the splicer defect above. These numbers describe `main` at `3b5cc4ca`; the next strict boot from main is
owed.

### Main after #813's retune (2026-09-18)

`main` at `483e4cb7` — #813's retune of the post-attention family, #818's typed buffer roles, #827's cut splicer fix
and #826's re-key — did not boot from the golden as merged. Strict evidence refused the symbolic post-attention twin
at one piece of #813's twenty-seam route, on that piece's own cut fork, with no measured row spelling an arm. The
symbolic twin is required, so the workers died sixteen minutes in. The same election replayed on the host refuses all
three post twins (dynamic, m16 and m4096) at the same node, and refuses on the tree at #813's merge before #827, so
#827 is not the cause.

Two things about #813's rows explain it, and neither is visible to the strict decode. Three of the dynamic twin's
receipts and four of the M=1 post twin's carry an empty schedule row; the deploy's evidence index drops a record with
no knobs, so a piece whose only receipt is empty has no measured row at its cut fork. And the m16 and m4096 sets hold
receipts for eight of the seventeen pieces the route mints on `main`. #813 recorded on an earlier tree and re-keyed
its rows onto `main`, where the pooled strict decode accepted every one of them.

The three post twins were re-recorded on this host under #813's route, pinned, with a fresh tune DB, into a copy of
the golden: the record appends a knob row for every uncovered piece and re-times the rest. Each twin then elects its
route under strict evidence on `main`.

| Twin | #813's row | Re-recorded here | Pieces with a receipt |
| --- | ---: | ---: | ---: |
| `post-sym` `3836f9` @ dynamic | 117.8 ms | 64.9 ms | 17 → 17 (3 were empty) |
| `post16` `8e1e80` @ m16 | 18.6 ms | 9.9 ms | 8 → 17 |
| `post4096` `366777` @ m4096 | 126.1 ms | 536.8 ms | 8 → 17 |

The m4096 number is honest and bad: the nine pieces #813 never recorded take the prior's schedules here, four of
them at 92–140 ms each, and the whole twin is worse than the 126 ms #813's row claims for a kernel set that never
deployed. The dynamic twin's `a47f22fa9713` piece is the same story at 24.7 ms against an empty row that said 178 µs.
Both are tuning work on the pieces, not blockers.

The boot from that copy died one refusal further, at the `k_div_35` kernel of the same twin: its recorded schedule
puts a cooperative reduce on two seams that #813's schedule codec no longer allows together ("a second scheduled root
on a projection its outputs do not partition by root"), either alone decodes, and the deploy's own warning says one
measured row matches none of 468 offered candidates. Those are the four rows that went red at #813's merge; at 3.5–20
µs the kernel is not a cost, but under strict evidence a required twin with one unvouched fork does not boot. Main's
own pick for it is poor — the prior puts the cooperative reduce on a nested seam, 140 µs at dynamic width and 492 µs
at m4096 — so the two single-seam forms were benched as A/B rows and the better one recorded at each width, as new
rows beside the red ones, which stay so the gate keeps naming the change that dropped their schedule.

| `k_div_35` width | red row (two coop) | main's pick | one coop-t seam, recorded |
| --- | ---: | ---: | ---: |
| dynamic | 3.5 µs | 140.4 µs | 16.7 µs |
| m16 | 3.7 µs | 4.7 µs | 2.5 µs |
| m4096 | 7.3 µs | 492.0 µs | 53.5 µs |
| m1 | 19.9 µs | 2.1 µs | main's pick kept |

With those rows the boot came up: `main` at `483e4cb7`, strict, empty tune DB, health seventeen minutes after
launch, engine init 51.5 s, KV capacity 76,043 and 78,419 tokens. The only strict refusals are the two expert
twins, as in every boot since 2026-09-11; the M=1 tier deployed for the first time since #804. Same probe as before,
greedy, single stream, streamed; the long prompt tokenizes to 2,155 tokens on this boot.

| Shape | TTFT, cold | TTFT, repeat | TPOT mean | TPOT range | Output tok/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| 5 in → 33 out | 3.73 s | 0.96 s | 0.267 s | 0.260 – 0.274 s | 3.75 |
| 2,155 in → 9 out | 29.32 s | 2.70 s | 0.327 s | 0.320 – 0.335 s | 3.06 |

| Program | Measured | Over floor | 2026-09-17 | 2026-09-15 |
| --- | ---: | ---: | ---: | ---: |
| `pre.chunk.m4096` | 25.4 ms | 850× | 2.73 ms | 2.74 ms |
| `post.decode.m1` | 3.2 – 3.6 ms | 57× | not deployed | 44.7 ms |
| `post.decode.m16` | 13.6 ms | 229× | 69.8 ms | 68.3 ms |
| `post.chunk.m4096` | 599 ms | 306× | 629.7 ms | 688 ms |

Decode is 12.4× faster than the previous boot and 3.4× faster than the 0.899 s per token of 2026-09-12, which was
the best this host had produced: the M=1 post twin runs in 3.4 ms per layer where the m16 twin it replaced ran
69.8 ms, and #813's Sinkhorn cut is inside both. Against the pinned fork's 0.147 s per token from the same host the
repository arm is now 1.8× slower, directional as before. Time to first token at the long prompt fell from 45.5 s
to 29.3 s with the m4096 post twin barely moved; that twin does not run for this prompt (see 2026-09-19 below), so
the gain is most likely in the symbolic twins its prefill rides, which was not measured. One program went
backwards: the m4096 pre-attention twin elects the same three-kernel cut as before but measures 25.4 ms against
2.73 ms. Evidence on the host under `~/serve-evidence/boot24-*`, `boot25-*`,
`boot26-*`, `elect826-*`, `rec826*` and `ab826-*`.

The cause is the cut's residual receipt, not the compiler. Its row, `WORK: t128, REDUCE: coop`, was recorded while
the kernel binder ignored a cooperative reduce on that residual, so the time it carries is the serial kernel's: one
thread per output cell, the four-element reduce serial inside. Before #813 that row and the all-serial row render
the same source byte for byte. #813 made the binder honour the row, and 128 threads now share a four-element
reduce in one block per output cell. The schedule is lowered correctly and it is a bad schedule. Strict evidence
cannot see the change, because the spelling is still offered and only its meaning moved; the boot's roofline audit
is what caught it. The m16 and dynamic twins' residual receipts carried the same spelling. The three receipts are
re-recorded from `main` as the serial row, and a second plain row of the same kernel and spelling is dropped at
m4096 and at dynamic, where it would outbid the serial row:

| Residual of | Recorded row on `main` | Serial row on `main` | Figure the golden carried | Program, before → after |
| --- | ---: | ---: | ---: | ---: |
| `pre16` (m16) | 90.8 µs | 2.3 µs | 2.3 µs | 1.55 → 1.27 ms |
| `pre4096` (m4096) | 22,574 µs | 224 µs | 228 µs | 25.1 → 2.82 ms |
| `pre-sym` (dynamic) | 115 µs | 31.1 µs | 31.1 µs | 4.71 → 4.78 ms |

The serial kernel reproduces the carried figure at every width, which is what says the rows measured it. The
dynamic program does not move: its 4.2 ms first piece varies by more between runs than the residual gains. The m1
twin is not affected: its recorded rows and their serial respelling build the same kernels on `main`. One V100,
strict evidence, an empty tune DB per run; logs on the host under `~/serve-evidence/recpre4096-*` and `recpre3-*`.

**The re-recorded file did not boot, and one row was why (2026-09-19).** A strict boot of `main` at `cab3b735` with
the golden as merged died after 24 minutes: every rank refused the m16 decode twins at a piece's cut fork, and
without them the runner rejects the engine's 4,112-token step budget. The m16 pre-attention set carried two rows of
one kernel, a cooperative row and a row spelling a cross-CTA split (`REDUCE@reduce: g4a`). Both carried 3.8912 µs,
and the cooperative row won the tie. The re-record above retimed the cooperative row to 5.73 µs, so the split row
won, and the pieces its split mints have no rows. Replayed on one card: the merged golden refuses on `main` and on
the tree before the later merges, the previous boot's golden elects its three kernels, and the merged golden minus
the split row elects the same three. No other kernel in the file has a split row as its fastest of several. The
row is dropped, and the boot on that file serves:

| Measure | 2026-09-19 | 2026-09-18 |
| --- | ---: | ---: |
| `pre.chunk.m4096` per layer | 3.10 ms | 25.4 ms |
| `post.decode.m1` / `post.decode.m16` / `post.chunk.m4096` per layer | 3.2 – 3.9 / 13.6 / 599 ms | 3.2 – 3.6 / 13.6 / 599 ms |
| Time per output token, 5 → 33 tokens | 0.266 s | 0.267 s |
| Time to first token, 5 tokens (cold / repeat) | 3.74 s / 0.97 s | 3.73 s / 0.96 s |
| Time to first token, 2,155 tokens (cold / repeat) | 29.32 s / 2.68 s | 29.32 s / 2.70 s |

The long prompt's time to first token did not move although its chunk twin got 22 ms per layer faster, because
that twin did not run: the runner takes a chunk twin only for a step of exactly 4,096 tokens, and one request under
a 4,096-token context limit never makes one. A 2,155-token prefill rides the symbolic twins, so they, not the m4096
pieces, are what a single request's time to first token is made of; the m4096 twins matter once concurrent prompts
fill a step. Health took 29 minutes against 17: the first compile on each rank took 790 to 860 s in both boots of
this tree, against 107 s in the previous boot, and a second boot did not shorten it. The cause is not found.
Evidence on the host under `~/serve-evidence/boot27-*`, `boot28-*` and `elect27-*`.

**The symbolic post twin, measured at a long prompt's width, and seven receipts re-recorded (2026-09-19).** `run
--bench` runs a dynamic row at its stored 512-token hint and has no flag for another width, so the twin was benched
from scratch copies of the golden with the hint rewritten, strict, one V100. Every width elects the same 21 kernels
with the same schedules, so the figures are the serving program's:

| Width | 512 | 1,024 | 2,155 | 4,095 |
| --- | ---: | ---: | ---: | ---: |
| Symbolic post twin, per layer | 64.9 ms | 131.2 ms | 275.8 ms | 524.7 ms |

It is linear, about 128 µs per token, and at 2,155 tokens it is 11.9 s of the 29.3 s to first token over 43 layers.
The symbolic pre twin is 4.9 ms there and does not matter. Four pieces are 94% of the post twin. The largest, 104 ms
at 2,155 tokens, spelled two cooperative reduces over 256 threads and launched one block per output cell — the
schedule #843 removed from the pre-attention residuals, here honestly measured and simply bad. The m16 and m4096 post
twins carried it on three pieces each. The seven receipts are re-recorded from `main` at `94c94378` as the serial
row, strict, an empty tune DB per run, unpinned:

| Twin | Pieces, before → after | Program, before → after |
| --- | ---: | ---: |
| `post-sym` (dynamic, at the 512 hint) | 24,729 → 178 µs | 64.9 → 40.3 ms |
| `post16` (m16) | 3 × 358 → 4.1 – 4.5 µs | 9.88 → 8.85 ms |
| `post4096` (m4096) | 96,960 / 92,483 / 92,466 → 1,404 / 1,213 / 1,089 µs | 536.8 → 258.1 ms |

No kernel's fastest row changed hands, the file keeps its row count, and its strict decode passes. A strict boot of
that file serves, health in 21 minutes:

| Measure | This boot | Previous boot |
| --- | ---: | ---: |
| Time to first token, 2,155 tokens (cold / repeat) | 24.90 s / 2.48 s | 29.32 s / 2.68 s |
| Time to first token, 5 tokens (cold / repeat) | 3.67 s / 0.92 s | 3.74 s / 0.97 s |
| Time per output token, 5 → 33 tokens | 0.267 s | 0.266 s |
| `post.chunk.m4096` / `post.decode.m16` per layer | 318.5 / 12.5 ms | 599 / 13.6 ms |

The 4.4 s the long prompt gained is what the one symbolic piece predicted (103 ms × 43 layers). What is left of the
symbolic post twin at 2,155 tokens is 173 ms per layer, and three pieces scheduled `WORK: t256, REDUCE: coop-t` are
156 ms of it (72, 42 and 42 ms); they are the next thing to tune. The m4096 record run exits 1 because the pinned
replay of its lead row no longer compiles ("direct atomic REDUCE writes each partial into f16 output storage"); the
unmodified golden fails the same way on this tree, so it predates this change, and the election itself builds and
runs. Evidence on the host under `~/serve-evidence/symbench-*`, `recpost7-*` and `boot29-*`.

**The three matmul pieces, tiled on tensor cores (2026-09-19).** The three pieces are the block's plain linear
layers, 4,096 → 2,048 twice and 2,048 → 4,096 once, and their rows ran them as cooperative dot products with the
tile site left empty. Other V100 goldens tile a dynamic matmul with `WORK: w2x2, TILE: mma_m8n8k4_f16_f32/f4x4/k8,
STAGE: d2/smem`, so that and five neighbours were benched per receipt from scratch goldens with one receipt
respelled, strict, one V100 each. The m4096 twin's matmul piece was already tiled, with a schedule 51× off, and got
the same treatment:

| Piece | Recorded row | Best rows measured | Worst tiled row measured |
| --- | ---: | ---: | ---: |
| dynamic, 4,096 → 2,048 (two kernels) | 9,457 µs | 142 µs (`w2x2 f4x4/k8 d2/smem`), 148, 176 | 26,591 µs (`w8x4 f4x4/k2 d2/smem`) |
| dynamic, 2,048 → 4,096 | 17,244 µs | 414 µs (`w2x4 f4x2/k8 d2/smem`), 415, 515 | 18,787 µs (`w8x4 f4x4/k2 d2/smem`) |
| m4096 matmul piece | 140,438 µs (`w8x4 f4x4/k2 d2/smem`) | 2,745 µs (`w2x4 f4x2/k8 d2/smem`), 2,956, 3,135 | 131,322 µs (`w4x4 f4x4/k8 d2/smem`) |

Staging is what makes the tile pay: the same tile without `STAGE` measured 895 µs against 142, and on the
2,048 → 4,096 piece the unstaged rows are not offered. Recorded from `main` at `c7f852b5`, strict, an empty tune DB
per run, unpinned: the symbolic post twin goes 40.3 → 4.84 ms at its 512-token hint and the m4096 post twin
258.1 → 120.6 ms. At long widths the symbolic twin is now 18.8 ms at 2,155 tokens (was 172.6) and 35.0 ms at 4,095,
which is 3.4× faster than the static m4096 twin at the same width. No kernel's fastest row changed hands, the row
count is unchanged, the strict decode passes, and the boot serves, health in 22 minutes:

| Measure | This boot | Previous boot |
| --- | ---: | ---: |
| Time to first token, 2,155 tokens (cold / repeat) | 18.34 s / 2.21 s | 24.90 s / 2.48 s |
| Time to first token, 5 tokens (cold / repeat) | 3.69 s / 0.92 s | 3.67 s / 0.92 s |
| Time per output token, 5 → 33 tokens | 0.267 s | 0.267 s |
| `post.chunk.m4096` per layer | 172.8 ms | 318.5 ms |

The 6.6 s gained is what the bench predicted (154 ms × 43 layers), and the long prompt's completion is unchanged
word for word, which is the only correctness evidence this target has: it has no eager twin, so a recorded row is
checked against the election's own output. The symbolic post twin is now 0.8 s of the 18.3 s to first token, so the
rest of a long prompt's prefill is elsewhere: the experts, attention, and first-request warm-up, none of them
measured here. Evidence on the host under `~/serve-evidence/ab-*`, `ab4k-*`, `recmm-*`, `symbenchmm-*` and
`boot30-*`.

**Where a decode step goes, and the M=1 post twin (2026-09-19).** This model serves eager — a hyper-connection MoE
host-syncs in its routed combine, so vLLM's decode capture is off — and each Emmy program replays its own CUDA
graph. The boot audit times an uncaptured launch loop, so its 3.4 ms for `post.decode.m1` overstates what serving
pays; a torch-profiler trace of eleven single-stream decode steps on all sixteen workers is the measurement. Per
step and per pipeline stage (22 and 21 layers; the profiled step is 0.310 s against 0.267 s unprofiled):

| | Stage 0 | Stage 1 |
| --- | ---: | ---: |
| Emmy kernels | 65 ms, 1,687 launches | 61 ms, 1,612 launches |
| NCCL all-reduce | 45 ms, 44 calls, median 1.0 ms | 51 ms, 42 calls, median 1.2 ms |
| Sparse attention (the fork's kernel) | 11 ms | 11 ms |
| Everything else on the GPU | 18 ms | 18 ms |

The stages run one after the other, so a token is about 126 ms of Emmy kernels, 96 ms of all-reduce, 22 ms of
attention and 36 ms of the rest. The all-reduce belongs to the host: the cards see each other over PCIe only, vLLM
disables its custom all-reduce there, and every call costs a millisecond whatever the arm. The largest Emmy kernels
per layer were the M=1 post lead's residual at 680 µs and its sum-of-squares piece at 149 µs, the expert program at
487 + 285 µs (the M=1 expert twin is refused, so a wider tier runs), the two M=1 pre-attention pieces at 202 µs
each, `k_div_4` and `k_div_30` at 143 µs each, and the `9e578e` cut's pieces at about 0.45 ms together.

The two post rows were fully serial schedules: the residual looped 4,096 and 16,384 elements inside each of four
threads, and the other piece summed 16,384 squares in one thread. Benched per receipt from scratch goldens, strict:

| Piece | Recorded row | Rows measured |
| --- | ---: | --- |
| residual | 667.6 µs, serial | `WORK: t32` 95.8, `t64` 52.8, `t128` 27.6, `t256` 15.5 µs |
| sum of squares | 149.5 µs, serial | `t64 coop` 5.4, `t128 coop` 3.7, `t256 coop` 3.1 µs; `t256 coop-t` is not offered |

Recorded from `main` at `c7f852b5` with the two best rows, the set goes from 851 to 52 µs per layer; no kernel's
fastest row changed hands and the strict decode passes. The boot on that file serves, health in 14 minutes:

| Measure | This boot | Previous boot |
| --- | ---: | ---: |
| Time per output token, 5 → 33 tokens | 0.235 s | 0.267 s |
| Time per output token, 2,155 → 9 tokens | 0.295 s | 0.325 s |
| Time to first token, 2,155 tokens (cold / repeat) | 18.32 s / 2.22 s | 18.34 s / 2.21 s |

The 33 ms gained is 0.78 ms × 43 layers, what the bench predicted, and the completions are unchanged. Against the
fork's 0.147 s the repository arm is now 1.6× slower per output token. The two M=1 pre-attention pieces stay open:
each of their 128 threads recomputes the 16,384-element statistic before its share of the dot product, and the
hand-pinned further cuts tried here were either not reproducible or fell back into the slow fused kernel. Evidence
on the host under `~/serve-evidence/prof31/`, `abm1-*`, `recm1-*`, `pin-pre1-*` and `boot32-*`.

**The expert program at decode (2026-09-20).** Strict evidence refuses the M=1 expert twin at every boot, and it
cannot be tuned into shape. Its golden row is a lone placement cut with no piece rows; under that cut the target
lowers to a 67 µs piece and a residual that runs 1.04 s per launch, and neither kernel offers a single schedule
knob: eight tensor-core pins all came back unreproducible, the tile realized as unset. At one row the MXFP4 expert
matmul has no tile or reduce site, so that twin needs the compiler, not a row. Single-token steps therefore ride the
m16 expert twin, 487 + 285 µs per call in the trace, and that twin's two rows were the prior's picks. Forty-eight
candidates over four rounds, a scratch golden with one receipt respelled each, strict, one V100 per candidate, from
`main` at `9607133e`:

| Piece | Recorded row, as it measures on this tree | Best rows measured | Worst row measured |
| --- | ---: | --- | ---: |
| gate / up | 554 µs (`w4x2 f1x1/k4 d2/smem`) | 265 µs (`w2x1 f2x1/k8 d2/smem`), 282, 300 | 5,163 µs (`w2x2 f4x4/k8 d2/smem`) |
| down | 310 µs (`w1x2 f1x2/k4 d1/smem`) | 159 µs (`w2x4 f1x1/k8 d1/smem`), 165, 167 | 511 µs (`w1x2 f1x4/k8 d1/smem`) |

The down piece takes `d1/smem` staging only; every `d2/smem` row, deeper `k16` tiles and asynchronous staging are
not offered. The tile that won the post-attention matmuls, `f4x4/k8`, is the worst row here: the expert rows are
narrow and a wide tile pads them. Recorded with the two best rows the twin goes from 864 to 430 µs in the program;
no kernel's fastest row changed hands and the strict decode passes. The boot on that file serves, health in 25
minutes on a tree new to the host:

| Measure | This boot | Previous boot |
| --- | ---: | ---: |
| Time per output token, 5 → 33 tokens | 0.214 s | 0.235 s |
| Time per output token, 2,155 → 9 tokens | 0.272 s | 0.295 s |
| Time to first token, 2,155 tokens (cold / repeat) | 18.16 s / 1.98 s | 18.32 s / 2.22 s |

The bench predicted 17 ms per token (434 µs over about 40 expert calls) and the boot gained 21; the two boots are
also one compiler merge apart (#847), so the last few milliseconds are not attributed. Completions are unchanged.
Against the fork's 0.147 s the repository arm is 1.46× slower per output token. Evidence on the host under
`~/serve-evidence/pin-exp1-*`, `abm1-e16*`, `recm1-exp16*` and `boot33-*`.

**Main's restamp took the golden off the card, and the recorder put it back (2026-09-22).** #864 restamped every
golden to the Loop IR target format and dropped 17 DeepSeek V100 rows: eight expert tensor-core tiles staged at depth
2, a ring the compiler no longer offers on Volta because it returned wrong answers on nine of sixteen measured grids,
and nine post rows "whose kernels #829 reshaped" — among them, in every post twin, the receipt of the piece that
broadcasts the second matmul operand. All 398 remaining rows decode, yet `main` at `dab7bce9` refuses all five twins
under strict evidence: the single-token and symbolic post twins at that piece's cut fork, the width-16 and 4,096 post
twins at their root's schedule fork, the width-16 expert twin at its gate/up piece. Replays on the #864, #863, #866
and #861 trees, each with the golden as of that commit, refuse identically, so the restamp itself is the cause. It had
also keyed four split rows and three tile rows onto their leads' identities (`.a64df778c19d` on the post16 lead,
`.20e5861ce454` and `.ef6794e8214b` on the post4096 lead, `.3cb64a167eb2` and three post4096 tiles onto siblings), and
a 12 µs split row on the root's identity outbids an 8.8 ms cut: the root splits instead of cutting and the split
kernel's serial schedule hangs the bench. Dropping those rows and recording each twin greedy from `main`'s tree gave
every live piece a receipt, with the prior choosing double-cooperative reduces for the uncovered residuals (24 ms at
the 512 hint, 94 ms at width 4,096, 358 µs at width 16); those were respelled serial (144 µs, 1.1 ms, 4 µs) and the
expert pieces re-tiled at depth 1 from kernel-scoped A/Bs (`w2x1 f1x1/k8` best of fourteen for the width-16 gate/up;
`w4x1 f2x4/k4` for both width-4,096 pieces), then every twin re-recorded strict. Per layer, strict election of the old
file on the #860 tree → this file on `main`: post m1 406 → 449 µs; post m16 8,847 → 5,966; post symbolic at the 512
hint 4,836 → 3,625; post m4096 120.6 → 84.6 ms; expert m16 430 → 719 µs; expert symbolic at the 512 hint 2.44 → 22.6
ms; expert m4096 22.5 → 14.2 ms. Three of those are the compiler, not the rows, and stay open: cooperative reductions
run 2–30× slower on `main` than on `9607133e` for the same spelling (a `coop-t` t256 piece 61.8 → 133.5 µs; the
symbolic residual's single `coop` t128, 303 µs before, is 11 ms now, so serial wins everywhere); the depth-2 ring is
gone (gate/up 265 → 490 µs at width 16, the m4096 expert pieces 2.5 → 6.8 and 7.4 ms); and the symbolic expert twin's
root piece offers no tensor-core tile any more (`t32x8 f2x26`, 20.4 ms against 1.8 ms at depth 2 before), which is
what a single request's prefill pays. Booted strict from that file (sha e4631af5), the server died three and a half
minutes in, before any post twin was compiled: the symbolic expert twin's root kernel under serving's fresh trace
carries an identity no row names, although the same twin replayed from the golden's stored Loop IR had just elected
and measured. `emmy trace --serving-twins` on the two trees explains it. On `9607133e` the serving twins lower to
exactly the golden's kernel families: 152 kernels, 36 per post twin, the two large fused softmax-matmul kernels every
tuning round since #799 targeted. On `dab7bce9` they lower to 188: 45 per post twin, the fused kernels split into a
linear-mean reduce, a softmax, two matmuls, twelve broadcast adds and three sums, and the four trees between them put
the change at #863 (`f6bd311b`, the reduction-dependency and coordinate normalization), not at #862 as its title
suggested. The decode gate and `emmy run --golden` replay stored Loop IR and cannot see this, and the runner builds
the expert group before a layer's pre and post twins, so the boot's first refusal names an expert kernel while every
post row is just as unreachable. No row of this golden deploys on `main`, and the re-recorded file is kept on the host
as evidence, not committed. The follow-up (2026-09-23) found two causes. The fusion moved because #863's merge rule
leaves a copy beside the consumers departing with an unfusable chain instead of materializing it in the region; on
this model that output is what later merges grow around, and restoring the pre-#863 region gives every serving twin
the golden's kernel set again (draft PR #875). The same hunk moves Qwen3.8's kernel sets the other way (the AWQ layer
42 → 40 kernels, EXL3 42 → 38, GPTQ 42 → 40), and #861 recorded on the 42-kernel sets, so the rule needs a condition
or one of the two goldens a re-record. And the boot from the fixed tree still refuses at the symbolic expert twin:
#863 also normalized the Loop IR of every kernel, this golden's stored loops were never re-lowered for it, and the
decode gate cannot tell because it replays the stored loop. Deploying on `main` therefore needs the rule settled and
the golden's programs re-lowered with every row re-keyed onto the new identities. Evidence on the host under
`~/serve-evidence/elect34-*`, `elect34b-*`, `rec34-*`, `ab34*`, `boot34-*`, `twins33.yaml`, `twins34.yaml` and
`twins-<sha>.yaml`.

**The golden restamped onto the fresh lowering (2026-09-23).** With #875's fix in the tree, every stored program was
lowered again and each stored target replaced by the fresh kernel that writes the same outputs: 151 of 151 targets
matched, 135 loops changed, and the pool is byte-identical to the serving inventory `emmy trace --serving-twins`
writes on the host, so the golden now describes what serving builds. Rows were re-keyed to the kernels they name (the
root by its lift, the pieces of a cut by an unchanged identity, an equal structural signature, or mint order), and
#863's re-associated trees left 105 rows spelling site paths that no longer exist — an old `map.1/inner` contraction
is a `map.1/reduce` with its multiplier hoisted to the root, deep map/inner nests are reduce chains — so their keys
were moved to the closest site of the fresh tree (same extent, the cone's reads and size, depth), and a row whose
value the fresh kernel does not offer at any site became a proposal the host re-measured. Fourteen rows could not be
carried at all and were dropped: the tensor-core tiles of four single-token divide kernels (the fresh kernels offer no
tile) and ten cut receipts whose three cooperative sites the fresh pieces never offer together; 59 rows the recorder
had written from its isolated re-bench with identities no replay mints were already dead and are gone too. The pooled
decode gate is blind to all of this — the input file decoded row by row while 74 of its rows were no evidence — so two
audits decided instead: the deploy's own evidence index (`evidence_rows`, under the recorded precision regime) and per
twin the kernels the route mints against the rows that vouch for them. Of the input's 435 rows, 330 carried over
unchanged, 11 kept their measurement under a re-spelled key, 23 became proposals the host re-measured, 14 were
dropped, and the host record runs added the rest: the committed file holds 465 measured rows over 149 targets, every
one of the 152 stored loops the fresh lowering's. The deploy's evidence index reads 448 of them as live evidence (365
of the input's 424), and of the 335 kernels the nine serving twins mint, 324 have a measured row (283 of 330 before);
the 11 without are the single-token expert and pre twins' pieces, which the runner treats as warnings. Eight of the
nine twins elected strict from the file on the first pass; the single-token post twin needed its cut sets recorded
once more and then elected too. Booted strict from this file with `EMMY_FAST_MATH=0` (boot36b), the server reached its
serving state in 25 minutes and answered the probe: at 5 prompt tokens TTFT 3.50 s cold and 0.73 s warm, at 2,155
prompt tokens 68.7 s cold and 5.53 s warm (18.2 and 2.0 s on boot33), TPOT 0.56 s per token at both widths (0.215 on
boot33). The roofline audit puts the width-16 post twin at 8.06 ms per layer (136× its floor), the width-4,096 post
twin at 104 ms (53×) and the pre twin at 3.1 ms (104×): decode pays the serial divide kernels and the forkless pieces
this tree lowers, prefill the cut reductions no cooperative schedule survives on. The probe's text is worse than
boot33's: the 5-token completion is number-and-slash noise where boot33 continued into a JSON fragment, the
2,155-token one repeats the passage's phrases where boot33 continued it verbatim. The width-16 expert twin's election
reports its random-input reproducer 9,931 outliers over a budget of 4 (mean difference 132 on outputs whose mean is
−143; the symbolic expert twin 140 within a tolerance of 578); no earlier log on the host prints that check for the
expert twins, so whether it is new cannot be shown from the records. The width-4,096 post twin's reproducer returns
NaN as it did on 2026-09-17. Both are the compiler's kernels under rows a strict election accepted, and the next
tuning round has to start from a correctness audit of the m16 expert and the post twins on this tree. The boot itself
then found a third cause, older than any row: #868 made fast math the default, `emmy serve` publishes no precision pin
to its workers, and every row of this golden is recorded under `FAST_MATH: False`, so since 2026-09-22 a serving
worker's evidence index has been empty and a strict boot refused at the first kernel it compiled — boot34, boot35 and
boot36 all died there, whatever the rows said, while `emmy run --realization` kept passing because it replays under
the row's own pins. Pinning `EMMY_FAST_MATH=0` in the serving container restores the regime the rows were measured in;
the rows' regime and the deploy's must agree, and today nothing checks it. Three compiler findings fell out, none
fixed here. The fresh divide kernels offer one reduce site with a schedule, and cooperative reduction at it does not
compile — `float v125` is declared once per half of the split accumulator — while the serial schedule runs 272 µs at
width 16 where the old cooperative row ran 7.6 µs and 7.9 ms at width 4,096 where it ran 32 µs; the GPU-free repro is
`emmy compile --golden … --realization post16.k_div_50_reduce… --target sm_70 --ir cuda` under
`EMMY_KNOBS=REDUCE@map.1/map.1/reduce=coop,WORK=t128`. The transposed cooperative reduce is not offered on any fresh
kernel of this model. And the prior picks cooperative reduces everywhere: on the symbolic post twin's two residual
pieces they cost 28 and 14 ms until a register split at the top reduce with the nested sites off
(`REDUCE@map.1/inner=r2`) brought them to 189 and 177 µs and the twin's election to 3.5 ms per layer at the 512 hint,
under the 4.8 ms of the old file. Every other repository golden except Qwen3.5-122B, two hardware files and, since
#869, the Gemma 4 RTX 5090 golden is stale the same way — #863 kept the old stored loops everywhere — and a new gate,
`test_stored_targets_are_the_fresh_lowering`, decodes each golden's stored targets against a fresh lowering of its own
programs, with those files as strict xfails until each is restamped on its card.

**Rebased onto main at #883 (2026-09-23).** #883 splits a cut piece's grid coordinate that its operands read only as a
quotient and a remainder into the two axes it stands for, and the two residual pieces of each post twin's cut — the
`k_linear_softmax_matmul_mean_reduce` kernel at widths 16 and 4,096 and in the symbolic twin — are exactly that shape:
their extent-16,384 axis, read as `a1 / 4096` beside `a1 % 4096`, is now an axis of 4 over an axis of 4,096, so they
are new kernels under new identities. The eleven rows that named the old six — the serial rows, the register splits
pinned above, the cooperative rows — describe kernels the compiler no longer mints: five stopped decoding and all
eleven left the evidence index, and they are dropped rather than carried, since a measurement of a kernel with another
body says nothing about the new one. The single-token post twin's twenty pieces have one free axis each and are
untouched. The committed file holds 454 measured rows over the same 149 targets; the deploy's evidence index reads 437
of them, and 318 of the 335 kernels the nine twins mint have a measured row. The six new pieces open one-arm forks,
the kind the strict election passed with a warning for the eleven single-token pieces above, so the twins still elect
from the file, but each post twin runs those two pieces at whatever the one arm is until they are measured on the
card, and the three cut receipts that price the routes carry the old pieces' time. Next on the host: record the three
post leads on main.


### The M=1 decode tier: what broke and what now guards it

The tier exists as an optimization. The bucket twins already cover `T=1` by padding up; the M=1 twins exist only
because the contractions demote to faster planar forms at one row. Nothing checked that they were in fact faster.

On this model they were not. `pre1.k_linear_mean_reduce_7fce9f` at M=1 had one measured candidate at 29.7 s against
57.9 ms for the bucket twin it replaced, and both of its candidates bench-fail outright
(`HungKernelError: kernel did not complete within 2000 ms`). Decode ran at a flat ~61 s per layer — flat from the
first layer to the last, which is what ruled out first-use cost as the explanation.

Its Loop IR shows the defect directly. A 16,384-long dot product that depends only on the stream index sits inside a
4,096-wide output-channel sweep, so it is recomputed 4,096 times, and the structure appears twice in the program:

```
for a1 in 0..4096:              # output channel
    for a2 in 0..4:             # stream
        for a3 in 0..16384:     # depends on a2 only — recomputed for every a1
            acc1 <- add(acc1, v8)
```

Two things were changed. Four placement cuts hoist those dots out of the sweeps, giving 473.6 µs whole-program, and
that schedule is recorded into the golden so the election prices it at 410.9 µs against the old 29.7 s row. And the
runner now drops the M=1 tier at boot when its twins do not measure faster than the bucket twins they replace, so a
golden without a good M=1 schedule costs the optimization rather than the server.

The boot audit let this run unseen for three rounds, and that is fixed too. It exempted any program whose roofline
floor sat under 20 µs — as this program's did — on the reasoning that "a mispick there costs little in absolute
terms". A small floor bounds what a healthy program costs, never what a broken one does. Programs with no usable
floor are now judged on absolute cost instead, and every warning carries the measured time.

### The boot reaches a serving state

On 2026-09-11 a boot with `--strict-evidence` came up: `/health` returned 200, `/v1/models` listed the checkpoint, and
vLLM logged `Application startup complete`. Engine init — profile, KV-cache creation and model warm-up — took 949 s.

Strict evidence is what made the boot possible, and the mechanism is worth recording. An earlier boot without the flag
logged 16 prior-clip warnings — one per worker — reporting a latency-proxy exponent of 996 against a shipped
artifact that peaks near 28. Past that clip every candidate scores identically, so the ranking degenerates to
enumeration order. The warning fires once per process, so it appeared once early and then held silently for the
whole run. Under
`--strict-evidence` that boot logged **zero** prior clips: every kernel that ran was decided by a measured row.

Strict evidence refused exactly one fork, identically on all 16 workers — the expert `m256` twin, on its `030_cut`
fork, with no measured row spelling a kernel-set arm. That width falls back to a wider tier and the boot continues; the
`m256` expert program is absent from the golden entirely, because the runner's expert prefill tier is a hardcoded
constant that the pinned serving config has no field to declare.

### Where the time goes

The boot's own roofline audit, consistent between the first and last layer:

| Program | Over roofline floor | Floor |
| --- | ---: | ---: |
| `pre.chunk.m4096` | 64,604× | ~30 µs |
| `post.decode.m1` | 1,273× | ~59 µs |
| `post.decode.m16` | 1,154× | ~59 µs |
| `post.chunk.m4096` | 343× | ~1,955 µs |

The recorded golden the boot deployed from carries 903 realizations, 295 of them measured. Summing the measured
per-kernel latencies within each program:

| Program | Measured sum | Worst single kernel |
| --- | ---: | ---: |
| `pre4096` | 43.2 s | 19.6 s |
| `pre1` | 29.7 s | 29.7 s |
| `pre16` | 17.2 s | 17.2 s |
| `expert-sym@mxfp4` | 1.61 s | 1.38 s |
| `expert1@mxfp4` | 1.33 s | 1.33 s |
| `post4096` | 0.82 s | 0.45 s |
| `post1` | 0.12 s | 0.045 s |
| `post16` | 0.086 s | 0.047 s |

The `pre` family dominates by three orders of magnitude, and every one of those programs is the same
`k_linear_mean_reduce` kernel. `pre1` and `pre16` are the decode path, which is what the RPC deadline measures.

### Three programs were recomputing work inside sweeps

Benched end to end on one card, `pre4096`'s greedy pick was a two-kernel placement split totalling 1,923,598 µs —
whole-program 1,923,912 µs, agreeing with the boot audit's 64,604× against a ~30 µs floor. A 4,096-row mean-reduce
costing 1.92 s is not a search problem, so we decoded what it computes.

It computes, per token, an RMS statistic over 16,384 hidden values and **four length-16,384 dot products**, then
sigmoid-mixes four 4,096-wide streams using those four coefficients. The dots depend on token and stream only. The
generated code recomputed them for **every one of 4,096 output channels** — 1.10 trillion dot-product terms per kernel
against 268 million if computed once per token, or **8,192× redundant work**.

Four placement cuts hoist them out of the channel sweeps. The same shape appears in the decode program's two hot
kernels: `9e578e` ran sixteen long dot products on a **single thread** (`if (_gid < 1)`), and `4e26cc` evaluated its
sixteen mixing logits 352 times across four threads.

| Target | Greedy pick | With cuts | Factor | Correctness vs greedy |
| --- | ---: | ---: | ---: | --- |
| `pre4096` | 1,923,598 µs (2 kernels) | **3,238 µs** (5) | **594×** | pass, max abs 1.2e-4 (fp16 level) |
| `post1` `9e578e` @ m1 | 42,278 µs (20 kernels) | **9,866 µs** (22) | **4.3×** | pass, exact (0.0 / 0.0) |
| `post1` `4e26cc` @ m1 | 30,016 µs (1 kernel) | **5,347 µs** (7) | **5.6×** | pass, exact (0.0 / 0.0) |

No new compiler capability was needed — the better structure was already expressible. The cuts move where work happens
rather than reassociating reductions, which is why the two decode results are bit-exact; `pre4096` differs only at fp16
rounding, and the golden's own alternative schedule for it shows the same magnitude.

### The election picks them up unaided

Recording each pick with `emmy run --golden … --bench --record-greedy` under the full pin list, merging those rows
into the serving golden, and re-booting with `--strict-evidence` moves the boot's own audit:

| Program | Before | After |
| --- | ---: | ---: |
| `pre.chunk.m4096` | 64,746× | **107×** |
| `post.decode.m1` | 1,275× | **294×** |
| `post.chunk.m4096` | 320× | 321× |
| `post.decode.m16` | 1,156× | 1,156× |

The last two are untouched because only the `m1` shapes were recorded. The first two match the bench measurements
(605× and 4.35×) closely enough to corroborate them independently. Nothing was hand-pinned in the serving boot: the
recorded rows win the election on price.

### Four things that cost a day, recorded so they cost nobody else one

- **A `perf` row times one CUDA op.** When the schedule under test splits a kernel, the row is a fragment. Reading it
  as the program's latency overstates the result by orders of magnitude. The check that catches it is whether the
  boot's roofline ratio moves.
- **A `PLACE` pin replaces the entire placement decision**; it does not add to it. Pinning 2 of a program's 21 cuts
  silently discards the other 19 and produced a hanging 3-kernel program with no diagnostic saying so.
- **`emmy tune` defaults to a 2 s cumulative-GPU-time bench budget.** On a kernel whose baseline is ~1 s per launch,
  warm-up plus one measured iteration exhausts it, and a run records **zero** valid latencies while appearing to search
  normally. `EMMY_BENCH_RUN_TIMEOUT_S` raises it.
- **A replay must use the golden the boot uses.** Benching one of these kernels against a golden with no rows for it
  produced a 60-second hang for a program the boot runs at 42 ms.

## What this run does not establish

- **It is not a balanced A/B.** The two arms were measured in separate invocations, not in one run with their order
  alternated inside each repeat the way the RTX 5090 gemma-4 experiment balances time and thermal drift. The gap is
  large enough that ordering cannot explain it, but the numbers are directional, not a controlled comparison. There is
  also no baked Emmy image for this model, so the single-image, two-entrypoint mechanism that A/B uses does not exist
  here yet.
- **The two arms did not run the same envelope.** The fork rows are at `gpu_memory_utilization` 0.80 with prefix
  caching disabled; the Emmy boot needed 0.90 and ran with prefix caching on. The Emmy shapes are 2,405 and 5 input
  tokens against the fork's 2,048, and 9 and 33 output tokens against its 512. The two Emmy rows also come from
  different boots: the time-to-first-token row predates the M=1 fix, which did not touch the prefill
  twins. Whichever values a joint recipe adopts,
  both arms must share them.
- **The Emmy rows are one repeat each.** The fork rows are five repeats with a reported spread; the Emmy rows are
  single runs, and their stability claim rests on the spread of decode steps within a run, not across runs.
- **The Emmy row needs a golden carrying the recorded M=1 schedule.** Against the golden this experiment shipped
  with, the M=1 decode tier is now dropped at boot and time per output token falls back to about 5.6 s. As of 2026-09-15 the repository golden carries #799's rows for that
  schedule and still elects the slower kernel; see "The repository golden serves".
- **The Emmy rows came from direct HTTP requests**, not from `emmy bench`, so they have no experiment records and are
  not in the archive.
- **It is not a regression check against the August run.** That run used different prompt shapes, a different context
  length and four repeats, so the two are not comparable row for row.
- **Correctness was checked only by the deployment smoke test**, which asks one arithmetic question and reads one
  token back. No token-level or numerical agreement was measured.

## Protocol

`emmy bench experiments/golden-bench-2026/serving_deepseek_v4_flash_0731_v100x16 --ssh <host> --no-teardown` against a
pre-allocated 16× V100 SXM3 host. Fifteen rows: five repeats crossed with three zipped workload shapes. Each row
deploys the server, runs the deployment smoke test, then runs one benchmark pass with greedy decoding
(`temperature: 0`, `seed: 0`) and `ignore_eos`, so every request produces exactly its requested output length. The
three shapes send 64 prompts at concurrency 8, 24 at concurrency 4, and 8 at concurrency 1.

Engine: TP8 × PP2, `gpu_memory_utilization` 0.80, `max_model_len` 4096, `max_num_batched_tokens` 4096, block size 256,
`dtype` float16, `deepseek_v4_fp8` weight quantization and an FP8 KV cache, prefix caching disabled. The engine
allocated 57,594 tokens of GPU KV cache. Model load and warm-up took about 151 s per row, of which weight loading was
about 25 s; benchmark time was 242–345 s for the concurrent shapes and about 661 s for the single-stream shape.

## Machine and software

Ubuntu 24.04.1 LTS, kernel 6.8.0-124-generic, two Intel Xeon Platinum 8168 (80 logical CPUs), 1,437 GB RAM, sixteen
Tesla V100-SXM3-32GB behind twelve NVSwitches, driver 580.159.03, nvcc 12.9.86, cuBLAS 12.9.2.10, Docker 29.5.3.
Engine image `cloudriftai/1cat-vllm-deepseek-v4-flash-0731:1.2.3-d76126608`, resolved digest
`sha256:276240257b224097876b5b6db8f0d32484dff6a6f168d6b03d6df188e5c65bc1`, vLLM `1.2.3.dev87+gd76126608`. Model
revision `7872f01b1d1fe23eabc4c98b48bffcef5a386062`.

The experiment records report repository revision `f38b3a3b`, which is the checkout the `emmy` console script imports
from. The run was driven from a worktree at `29ec3b40`; the recipe used is byte-identical to the one on `main`, and
`emmy/benchmark`, `emmy/provisioning` and `emmy/deployment` are byte-identical between the two revisions, so the
orchestration that executed is the code the records name.

## Run and archive

Timestamp `2026-09-10T17:41:53Z`, run ID `20260910T174153Z`, fifteen rows, all `succeeded`, one run ID across every
record. Archive `results_v100x16.tar.gz`, root member `2026-09-10_17-41-53/`, 48 members: one `*.experiment.yaml`,
one `*.benchmark.log` and one `*.server.log` per row, plus the two orchestration logs. Row names carry their shape,
for example `v100x16_mc1_np8_ril2048_rol512_r0_ffaecb951bfc`.

The host is pre-allocated and privately owned, so its address and login were replaced with `<redacted-host>` and
`<redacted-user>` throughout the archived records and logs before archiving. That substitution is the only edit made
to what `emmy bench` produced; every timing, count and configuration value is untouched, and the records still parse
and still carry one run ID and a terminal status each.
