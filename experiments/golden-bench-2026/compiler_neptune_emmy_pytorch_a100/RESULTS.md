# Neptune, modern PyTorch, and Emmy on A100

## October 14 decode: smaller key chunks and cooperative finalizers

The final A100 decode sweep has a geometric mean Emmy / paired PyTorch latency ratio of 0.934237, or 6.58% lower
latency. Seven of eight shape means favor Emmy; the 32768-token cache remains 1.84% slower. Two separately controlled
changes lead to this selection: smaller key chunks and cooperative split finalizers. The complete official row
passes all 16 strict source checks. Its 16 additional stored-program timing replays remain numerically unchecked.

This follow-up tests two profiling hypotheses on the retained A100. The first halves the partial kernel's key chunk
from 128 to 64 while holding its split count, two compute warps, output width and precision controls fixed. The second
lets the split finalizer use existing legal reduction schedules instead of forcing a serial finalizer.
Neither change weakens accuracy checks or enables fast arithmetic.

### Profiling and controlled qualification

The preceding 8K profile measured about 134.29 MB of DRAM reads for both Emmy's partial kernel and the selected
FlashAttention reference kernel, close to one read of the 134.22 MB logical K/V input. Emmy nevertheless generated
401.09 MB of total L2 traffic versus 261.71 MB for the reference and showed 8.99% versus 11.04% active warps. This
motivated testing a smaller shared-memory slab, followed by parallelizing the small split finalizer.
The profile used application replay with cache flushing and clock control both disabled. Its reference launch could not
be tied to an exact captured `torch.compile` iteration, and Nsight Systems did not capture CUDA activity. These are
per-kernel diagnostics, not a matched hot-cache whole-forward comparison. Failed captures and raw reports remain
available; L2 traffic is total traffic, not read traffic or a hit rate.

The K64 comparison uses the existing seeded decode operator, five warmups and twenty captured iterations. Each
invocation has a fresh database, explicit schedule and four false precision gates; eager PyTorch, full-graph
`torch.compile` and Emmy run on the same inputs. Forty strict calls cover initial screens and three alternating pairs
at each of five selected lengths. All pass. The source and complete schedule are invariant within each arm; only the
two coupled tile choices differ, and each finalizer source remains unchanged. The 256/512 and differently shaped 2048
choices are left alone.

| Cache length | K128 Emmy median (µs) | K64 Emmy median (µs) | K128 paired Emmy/TC median | K64 paired Emmy/TC median |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 21.530 | 15.162 | 1.06706 | 0.74974 |
| 4096 | 64.717 | 61.824 | 1.03745 | 0.99116 |
| 8192 | 114.432 | 109.340 | 1.05030 | 1.00405 |
| 16384 | 213.197 | 207.872 | 1.04506 | 1.02343 |
| 32768 | 411.136 | 406.016 | 1.05407 | 1.04191 |

Every K64 candidate wins its three baseline pairs. At 1K the candidate's absolute times vary from 15.122 to 19.547 µs,
with its paired reference moving too; the ratios stay between 0.7493 and 0.7521. The paired-ratio improvement is the
stable conclusion there. Its baseline input, two-kernel boundary, full schedules and CUDA sources exactly match the
preceding official 1K run, whose timings used a different one-warmup/fifteen-iteration protocol. No cause is assigned
to the process variation.

At 8K, K64 reduces shared memory from 48 to 24 KiB and register use from 255 to 164, with no local spills in either
arm. The reported theoretical occupancy rises from 9% to 19%; this is a resource estimate, not a new active-warp
counter measurement. The earlier profile's cache and reference-attribution limitations remain unchanged.

Finalizer screening holds K64 fixed and compares serial, r2, r4 and cooperative-transposed reduction. Only the
cooperative choice uses 128 threads.
Three dedicated GPU tests first verify the actual merge of partial states for uneven five-part tails, empty states and
all-empty output under the unchanged strict tolerance. All pass. Before timing, all five K64 serial source vectors
are also shown byte-identical across the compiler update. Initial 8K/16K screens favor the cooperative choice; r2/r4
are retained as diagnostic observations. Three alternating pairs confirm the cooperative choice at both lengths.
Every candidate beats its serial control and its paired PyTorch reference. These compare K64 against K64; the earlier
K128-to-K64 phase remains a separate controlled comparison.

| Cache length | K64 serial median (µs) | K64 cooperative median (µs) | Serial paired Emmy/TC median | Cooperative paired Emmy/TC median |
| --- | ---: | ---: | ---: | ---: |
| 1024 | 15.073 | 18.971 | 0.74730 | 0.72948 |
| 4096 | 61.760 | 59.456 | 0.98959 | 0.95313 |
| 8192 | 109.227 | 106.610 | 1.00349 | 0.97775 |
| 16384 | 208.077 | 201.933 | 1.02454 | 0.98967 |
| 32768 | 406.187 | 397.312 | 1.04071 | 1.01652 |

The incremental whole-forward reductions are 2.40% at 8K and 2.95% at 16K. The exact partial source and schedule stay
unchanged; only the finalizer changes. The same cooperative choice also passes three pairs at the remaining lengths;
all paired ratios improve.
At 1K the finalizer confirmation again enters two common timing ranges. Serial Emmy/reference pairs are
19.566/26.060, 15.037/20.169 and 15.073/20.169 µs; cooperative pairs are 14.713/20.169, 19.035/26.045 and
18.971/26.060 µs. Every paired-ratio comparison improves, with the median moving from 0.74730 to 0.72948, but the raw
median reverses because the slower range appears in opposite arms. This supports a paired-ratio effect, not a raw
median latency improvement for that phase. No cause is assigned and no repeat is discarded.

The five strict writer runs preserve their exact qualified CUDA and full schedules. The minimal candidates preserve
all five stored programs, nineteen kernels, seven routes and nineteen original measured rows. They add ten measured
schedules and refresh five whole-target seed latencies. The raw writer retains explicit four-false precision pins;
the promoted form uses the original standard regime only after proving effective precision equivalence.

All five freshness checks pass. On the actual A100, a fresh database and strict-evidence compile without hand
schedule pins select exactly the recorded CUDA and complete tile schedules, including the 1K cooperative finalizer.
These proofs precede promotion. All 84 controlled source calls and five stored-program writer calls pass strict
eager checks. Their maximum absolute error is 0.0001220703125, and the largest mean absolute error among the
controlled calls is 0.0000065922. Emmy uses the unchanged `rtol=atol=0.001`; the paired PyTorch compiler uses the
existing scaled comparison. No tolerance is changed.

### Final official results

The unchanged complete eight-length decode-causal row succeeds. The table uses the mean of two independent
minimum-of-15 captured source measurements, with one warmup. Each ratio divides mean Emmy latency by mean paired
PyTorch latency. These source calls use the same seeded inputs within each invocation. Their protocol is separate
from the five-warmup/20-iteration controlled comparisons above.

| Cache length | Emmy, µs | Paired PyTorch, µs | Emmy / PyTorch |
| ---: | ---: | ---: | ---: |
| 256 | 13.810 | 15.495 | 0.891235 |
| 512 | 19.588 | 20.404 | 0.960017 |
| 1024 | 19.142 | 26.075 | 0.734086 |
| 2048 | 34.887 | 35.155 | 0.992356 |
| 4096 | 60.608 | 64.411 | 0.940959 |
| 8192 | 107.008 | 108.969 | 0.982004 |
| 16384 | 201.933 | 204.103 | 0.989366 |
| 32768 | 396.971 | 389.789 | 1.018424 |
| Geometric mean of ratios | — | — | 0.934237 |

All 16 strict source comparisons pass, with maximum absolute error 0.000244140625 and largest mean absolute error
0.0000112729. All 32 ordered CUDA source and full schedule vectors match the expected selections: the five new
recordings and the three unchanged accepted shapes. The retained task DB contains only imported golden rows;
official calls disable measurement recording. The 16 stored-program timing replays explicitly mark numerical
correctness unchecked and are not extra accuracy passes or inputs to this table.

The two strict source observations vary by 0.53% at 8K and have identical Emmy times at 16K and 32K. Larger variation
remains at 2K and 4K: Emmy spreads are 5.19% and 4.31%, and paired PyTorch spreads are 11.60% and 6.32%. At 2K the
individual ratios are 0.965 and 1.023, crossing parity despite a favorable mean. Unchecked replay timings vary by
as much as 26.27% at 1K. Every observation is retained. The final geometric mean describes this run; comparison
with a preceding run is not a controlled optimization estimate. The two alternating-pair phases above establish
the selected changes, including the explicit 1K absolute-median caveat.

### Provenance and retained evidence

The K64 qualification uses `0a701074d9df59461f61a60f55cf209e0794ddc2`. The finalizer qualification uses
`cc0ef7bb1ae36a766794b54740ce32a6a7b9cc8e`. Recorded commit-tree equivalence receipts preserve the relationship
to rebased revisions without rewriting the original measurement attribution. The official run is
`2026-10-14_02-44-52`, ID `20261014T024452Z`, at clean revision `58a746d73c91427164ed1a1c286a107cbbc5999f`.
It starts at 02:44:52 UTC and finishes at 02:52:33 UTC. The A100-SXM4-40GB has 108 SMs, PyTorch 2.13.0+cu130,
NVCC 13.0.88 and driver 580.178.04. All 467 declared official inputs match the committed revision, imports resolve
to that checkout, and the reused native runtime retains SHA-256
`9ae29b503857bbfb10a69a6fa2261c3b7bf7fa53edaf1a5098d0389a14449e38`. The dependency lists before and after are
byte-identical. No package was installed and no local GPU ran. Comparing all 467 common finalizer-qualification
and official inputs finds only the five intended golden changes. The compiler and native runtime source remain
unchanged between those two phases. The A100 is idle after the terminal run and remains rented.

The final H100 validation suite passes 7,032 tests with 880 skipped and 131 warnings in 2,178.20 seconds at
`98d49d5307ca15f4d6a93b0b932316b3ef129249`. Compiler, native runtime and test sources are identical to the
official A100 revision; only the five promoted experiment goldens differ. The explicit freshness and unpinned
selection checks above cover those files. The full-suite logs, environment, source and copy checks are retained
under `diagnostics/review/full-suite-98d49d530/`. This is remote validation; no full local suite ran.

The current `results_a100x1.tar.gz` is rooted at `2026-10-14_02-44-52/`. Its unchanged system-only record is
`a100x1_lemmy_od-c_b16ee7234dbb.experiment.yaml`, with terminal status `succeeded`. Official raw evidence is under
`a100x1_lemmy_od-c_b16ee7234dbb/evidence/emmy-tcompile/`. The table and all source/schedule joins are preserved in
`diagnostics/official-numeric-receipt.json`; exact command, imports, staged inputs, dependencies, idle state and remote
digests are under `diagnostics/official/`. All 1,581 official raw files match their remote digests. Qualification,
rejected schedules, strict writer DBs, original and promoted rows, unpinned selection, and independent measurement
and compiler reviews are under `diagnostics/`. Earlier profiling, including failed captures and attribution limits,
is retained in `diagnostics/prior-profile/`.

The archive retains the entire preceding platform archive byte-for-byte as
`historical/results_a100x1-before-profile-optimizations.tar.gz`, SHA-256
`058a4f103632154560dafa881e554ebbf6d6e3252c158b2fd232b3a8ffa559b1`, with 177,079,809 bytes. The current archive
SHA-256 is `614efcdce2c894aafed93ebe32550af11f2301de289a2cf9e88da6d5152dc08d`, with 201,241,783 bytes. All
16,248 direct files byte-match the retained tree. The credential scan finds no matches in those files or in 42,334
regular files inside 35 recursively inspected archives. The scan traverses the retained history afresh, verifies
all preliminary file hashes, and scans additions and changes before sealing; no historical scan count is reused.

## October 14 selected long causal-decode split refresh

Fewer partitions improve the existing two-warp causal-decode schedules at lengths 8192 and 16384. Three alternating
fresh-process pairs reduce median Emmy latency by 3.47% and 5.01%, respectively. Both lengths still trail their paired
PyTorch compiler by roughly 4–5%. The partial tile geometry, staging, precision controls, and the other six
lengths remain unchanged. No compiler or prior changes are required.

### Controlled qualification

The one-token operator uses seeded FP16 inputs, batch 1, 32 query and KV heads, and head width 128. It attends the
entire existing cache, with `is_causal=False`. All four precision gates remain false. Qualification uses revision
`bdbf964787e21b820d5e5a4c696cd633de3be818`, deployment O3, five warmups, 20 captured iterations, a fresh DB per
invocation, and unchanged strict eager tolerances of `rtol=atol=0.001`. All backends in each comparison use the
same inputs. The retained A100-SXM4-40GB uses PyTorch 2.13.0+cu130 and the unchanged native runtime.

The selected split counts change from 32 to 16 at length 8192, and from 64 to 32 at length 16384. Each partition
therefore processes 512 cache tokens instead of 256. Both the partial and finalizer bodies change; each new route
requires fresh measurements of both pieces. Doubling the original split counts was correct but slower at both
lengths, so those alternatives remain diagnostics only.

| Length | Original partitions | Selected partitions | Original median Emmy, µs | Selected median Emmy, µs | Original median paired ratio | Selected median paired ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8192 | 32 | 16 | 118.016 | 113.920 | 1.081595 | 1.042448 |
| 16384 | 64 | 32 | 224.000 | 212.787 | 1.091930 | 1.047344 |

Each candidate wins all three alternating pairs in both absolute latency and its own Emmy / paired PyTorch ratio.
Candidate latency spans 113.280–114.560 µs at 8192 and 212.582–213.197 µs at 16384, corresponding to 1.13% and 0.29%
repeat spreads. The complete ordered CUDA sources and schedules are invariant within each arm. Component timings
show less finalizer work, consistent with the smaller reduction, but their isolated sums are not whole-forward
timings and are not the optimization estimate.

The earlier bounded GQA-prefill probes did not improve performance. Reducing four warps to two at the original
128-key chunk increased register spills and nearly doubled latency at both 2048 and 8192. At 8192, reducing the
key chunk to 64 and matching the QK score width removed spills and reduced register use, but the four-warp version
still lost: 6420.480 µs versus the fresh 6231.040 µs control. Their paired PyTorch ratios were 1.125673 and 1.099566.
The two-warp/64-key combination was slower again, at 7815.168 µs and ratio 1.379609. These probes retain the existing
FAST_EXP-only prefill regime. The first four prefill diagnostics measured eager and Emmy only; they supply no
paired PyTorch-compiler claim. All seven pass strict eager correctness. No prefill schedule is promoted.

All 20 decode qualification and recording calls pass strict eager correctness, with maximum absolute error
0.000030517578125. The seven prefill diagnostics also pass, with maximum absolute error 0.0009765625. No tolerance
is changed. These two workload groups use their separate fixed numerical regimes as described above.

The two promoted files retain their original programs, kernels, routes, and six measured rows. They add four
kernels, two routes, and four measured rows, and refresh two seed latencies. New redundant false precision pins
are normalized only after proving identical effective precision decisions; original writer files remain intact.
Both strict stored-program recordings reproduce the qualified ordered CUDA and schedule lists. Both fresh-lowering
checks pass. Subsequent compilations on the actual A100 host, using fresh DBs and no hand schedule or precision pins,
select exactly those same sources from measured golden evidence.

### Final official results

The unchanged complete decode-causal recipe row succeeds at all eight lengths. Its final geometric mean Emmy /
paired PyTorch ratio is 1.008442, leaving a 0.84% gap. Three of eight point estimates favor Emmy; five still trail
PyTorch. The table uses the mean of two independent minimum-of-15 captured source measurements, with one warmup.
Each ratio divides mean Emmy latency by mean paired PyTorch latency. This is separate from the controlled
five-warmup/20-iteration comparisons above.

| Length | Emmy, µs | Paired PyTorch, µs | Emmy / PyTorch |
| ---: | ---: | ---: | ---: |
| 256 | 13.817 | 15.475 | 0.893 |
| 512 | 19.584 | 20.407 | 0.960 |
| 1024 | 27.917 | 26.126 | 1.069 |
| 2048 | 32.945 | 33.857 | 0.973 |
| 4096 | 67.891 | 64.548 | 1.052 |
| 8192 | 113.856 | 109.117 | 1.043 |
| 16384 | 213.197 | 204.907 | 1.040 |
| 32768 | 411.392 | 391.285 | 1.051 |
| Geometric mean of ratios | — | — | 1.008442 |

All 16 strict source comparisons pass, with maximum absolute error 0.000244140625 and largest mean absolute error
0.0000112729. The recipe's 16 stored-program timing replays mark numerical correctness unchecked and are not extra
accuracy passes. All 32 ordered CUDA source and complete schedule vectors match the expected selections: the two
new recordings at 8192/16384, and the preceding accepted sources at the other six lengths. The task DB contains
imported golden evidence only; official calls disable measurement recording.

The changed lengths are stable in the two strict source observations: Emmy's repeat spread is 0.56% at 8192 and
0.38% at 16384. Unchanged lengths 2048 and 4096 show larger absolute variation, 17.48% and 9.15%, despite identical
sources. Their paired PyTorch times vary by 7.21% and 6.85%. At 2048 the individual ratios are 0.927 and 1.016,
crossing parity while their mean favors Emmy. This unexplained variation means the new overall geometric mean
must not be interpreted as a controlled improvement over a different run. The 3.47% and 5.01% alternating-pair
results above are the optimization estimates. Separate unchecked replay timings vary by up to 28.21% at 1024;
they remain in the evidence and are not mixed into the source-comparison table.

### Provenance and retained evidence

The official run is `2026-10-14_01-01-47`, ID `20261014T010147Z`, at clean revision
`46c41fc5b0439cd82bf4492b7d385c40da60ddf3`. It starts at 01:01:47 UTC and completes at 01:09:30 UTC. All 467 declared
inputs match that revision. Comparing the 462 common qualification and official inputs finds only the two intended
golden changes; compiler and native runtime source are unchanged. Imports resolve to the measured checkout,
and the native runtime remains byte-identical. PyTorch is 2.13.0+cu130, NVCC is 13.0.88, and the driver is
580.178.04. The dependency lists before and after are byte-identical. No package was installed. The unused Neptune
tuning artifact retains its committed LFS pointer; the initial manifest mismatch against locally materialized
bytes and its correction are preserved. No local GPU workload ran. The rented A100 remains available with no
active GPU process.

The preceding split-refresh snapshot is rooted at `2026-10-14_01-01-47/`. Its unchanged system-only record is
`a100x1_lemmy_od-c_b16ee7234dbb.experiment.yaml`, with terminal status `succeeded`. Raw official measurements are
under `a100x1_lemmy_od-c_b16ee7234dbb/evidence/emmy-tcompile/`. `diagnostics/official-numeric-receipt.json` joins
every reported value to those files and verifies every source and schedule. `diagnostics/official/` holds the
source/import checks, exact command, dependency lists, process-state receipt, and remote raw-file digests.
The qualification, rejected prefill and larger-split arms, paired comparisons, strict writers, exact DB joins,
candidate construction, unpinned selection, and independent reviews remain under `diagnostics/`.

The preceding complete platform archive is preserved byte-for-byte as
`historical/results_a100x1-before-split-refresh.tar.gz`, SHA-256
`fe4528ce7231461d424def0966b0cf75e4c02d7564a8c7ec5171747a049321cd`. It retains the previous decode refresh and all
earlier platform history. That split-refresh archive SHA-256 is
`058a4f103632154560dafa881e554ebbf6d6e3252c158b2fd232b3a8ffa559b1`, with 177,079,809 bytes. All 5,667 direct
files byte-match the retained tree, and all 1,565 official raw files match their remote digests. The credential
scan found no matches in 5,666 direct files, the separately scanned final claims review, or 33,515 regular files
inside 32 recursively inspected archives. No historical scan count is reused.

## October 13 selected causal decode refresh

Fewer compute warps per block improve the standard-math causal-decode sweep without changing its arithmetic,
Loop IR bodies, or split decisions. Controlled comparisons reduce the geometric mean Emmy / paired
`torch.compile` ratio from 1.176 to 1.020, a 13.3% reduction in the ratio. The fresh official ratio is 1.024,
leaving a 2.4% gap. The largest improvement is at length 2048. The longer lengths gain less and still trail the
paired PyTorch compiler. This selected rerun does not update Neptune or the other four attention families.

### Selected schedules and controlled evidence

The operator uses seeded FP16 inputs, batch 1, one query token, 32 query and KV heads, and head width 128. The
query attends the entire existing cache (`is_causal=False`); the family name does not imply a triangular mask
for this one-token input. Umbrella fast math, fast exponentials, reduced-precision MMA accumulation, and FP8 MMA
are all disabled. The strict comparison stays at `rtol=atol=0.001`.

All lengths change `WORK=w4x1` to `w2x1`, except length 2048, which selects `w1x1` and widens the outer PV tile
from 32 to 128 output columns. The original four-warp geometry computes tensor-core query rows beyond the one
useful query. The wider 2048 tile also avoids recomputing its score for four separate output-column tiles. The
existing split counts, inner QK tiles, staging choices, and finalizer sources are unchanged. Widening the 8192
tile was correct but slower than the selected two-warp schedule at its original width, so it was rejected.

Qualification uses revision `693546d681d4ac87e433a788c258f6693d887556`, deployable O3, five warmups, 20 captured
iterations, a fresh DB per invocation, and explicit standard precision pins. Every source invocation measures
Emmy and full-graph `torch.compile` against the same eager inputs. Three alternating baseline/candidate pairs
were run at lengths 2048 and 8192; the other six lengths have one alternating pair in addition to their initial
baseline and candidate observations. The table summarizes the confirmation batch, using the median paired
ratio where three observations exist.

| Length | Original Emmy / paired PyTorch | Selected Emmy / paired PyTorch |
| ---: | ---: | ---: |
| 256 | 1.027108 | 0.891746 |
| 512 | 1.262595 | 0.959459 |
| 1024 | 1.252930 | 1.067816 |
| 2048 | 1.606660 | 0.981102 |
| 4096 | 1.066688 | 1.040397 |
| 8192 | 1.103194 | 1.084799 |
| 16384 | 1.112388 | 1.097408 |
| 32768 | 1.072129 | 1.055118 |
| Geometric mean | 1.176225 | 1.019969 |

At length 2048, median Emmy latency falls from 50.368 to 30.508 µs, a 39.4% reduction. Original observations span
50.240–50.368 µs and selected observations 30.332–30.685 µs. The selected paired ratios are 0.973835, 1.045888,
and 0.981102: Emmy beats its paired PyTorch compiler in two of three observations. At 8192, median Emmy latency
falls from 120.101 to 118.528 µs, a consistent but modest 1.31% reduction. At 1024, both the selected Emmy and its
paired PyTorch time shift substantially between processes; their ratio remains 1.064–1.068. The cause of that
shared absolute-time variation is unknown, so its isolated absolute reduction is not the optimization estimate.

All 49 whole-program qualification and recording invocations pass the unchanged strict check, including the
rejected wide-8192 diagnostic. Their largest absolute error is 0.000244140625. Ten isolated candidate observations
also pass their same-input greedy comparison, with zero reported differences; those isolated checks are distinct
from the whole-program eager proofs. A multi-arm isolated invocation retains the final arm's source dump, while
the JSON preserves every arm's runtime source hash and complete schedule.

The promoted golden files preserve all eight programs, 20 stored kernels, six routing decisions, and 14 original
measured rows. They add eight measured schedules and refresh eight whole-target seed latencies. The old finalizer
measurements remain intact; their fresh remeasurements stay in the raw writer evidence. Only new
redundant false precision pins are normalized to the existing standard spelling after proving equal effective
gates and DB contexts. All eight fresh-lowering checks pass. Eight subsequent strict-evidence compilations on
the A100 host, using its actual card metadata, a fresh DB, and no hand schedule or precision pins, select ordered
CUDA source lists identical to the qualified recordings. Neither hardware goldens nor prior weights change.

### Fresh official results and retained evidence

The selected official experiment row succeeded, with all 32 invocations completing. The table uses the mean of
two independent minimum-of-15 captured source measurements, with one warmup. Its ratios divide mean Emmy by
mean paired PyTorch latency. These final measurements are separate from the five-warmup/20-iteration controlled
comparisons above. Lengths 256 and 512 launch one kernel; the other six lengths launch a partial and finalizer.

| Length | Emmy, µs | Paired PyTorch, µs | Emmy / PyTorch |
| ---: | ---: | ---: | ---: |
| 256 | 13.832 | 15.485 | 0.893 |
| 512 | 19.595 | 20.422 | 0.960 |
| 1024 | 27.729 | 26.054 | 1.064 |
| 2048 | 30.543 | 30.358 | 1.006 |
| 4096 | 64.956 | 62.518 | 1.039 |
| 8192 | 118.784 | 109.101 | 1.089 |
| 16384 | 224.640 | 203.684 | 1.103 |
| 32768 | 411.904 | 391.459 | 1.052 |
| Geometric mean of ratios | — | — | 1.024 |

Only two of eight final point estimates favor Emmy. All 16 strict actual-source checks pass, with maximum
absolute error 0.000244140625 and largest mean absolute error 0.0000112729. The unchanged recipe's 16 golden
replays enforce measured evidence and report timing, but mark numerical correctness unchecked; they are not
additional accuracy passes. The eight prior stored-program recordings supply their separate strict eager proofs.
Every official ordered source and complete schedule list matches qualification. The task-local DB imports
committed measurements only; all official invocations disable recording, so repeats cannot learn new schedules.

Among the strict source comparisons, Emmy's largest within-length repeat spread is 0.74%, using `(max-min)/min`.
The largest paired PyTorch and ratio spreads occur at length 2048: 7.67% and 7.92%. Emmy is 30.508 and 30.579 µs
there while paired PyTorch is 31.479 and 29.236 µs, yielding ratios 0.969 and 1.046. Thus the final mean narrowly
trails PyTorch despite the controlled median favoring Emmy. Both results are retained; the reference-time change
does not justify another rerun.
The separate timing-only golden replays vary more, up to 14.54% at length 256, despite unchanged source lists.
Their observations remain in the archive and are not mixed into the source-comparison table.

The run is `2026-10-13_23-39-25`, ID `20261013T233925Z`, at clean revision
`bea18c1c3744161dda600a4391ef43a800ba961e`. It runs the unchanged official recipe locally on the remote
A100-SXM4-40GB, retaining the existing Python environment and a byte-identical native runtime. No package was
installed. All 467 declared inputs match the committed source; the unused Neptune tuning artifact remains its
committed LFS pointer and is not read by this selected Emmy lane. Imports resolve to the measured checkout.
PyTorch is 2.13.0+cu130, CUDA is 13.0.88, and the driver is 580.178.04. The before/after dependency lists are
byte-identical. No local GPU workload ran, and the rented A100 remains available with no active GPU process.

The preceding decode snapshot, now retained inside the current archive, is rooted at `2026-10-13_23-39-25/`.
Its unchanged system-only record is
`a100x1_lemmy_od-c_b16ee7234dbb.experiment.yaml`; its task directory contains `artifacts.tar.gz`, the logs,
dumped IR, software receipts, and all 32 records under `evidence/emmy-tcompile/json/`. All 1,565 downloaded
run files match their remote bytes and digests. `diagnostics/` retains all qualification, rejected alternatives,
recordings, DBs, commands, candidates, freshness and unpinned source checks, and independent reviews.
`diagnostics/official/numeric-receipt.json` joins every final value and source vector to its raw record;
the adjacent manifests and import receipts establish source and environment provenance.

The preceding complete platform archive is retained at `historical/results_a100x1-before-decode.tar.gz`, with
SHA-256 `8405c7bf56f51978707eebd80caa6ca0a4fdaa913fa4d39fe55e1a3b780d4fa1`. It preserves the earlier causal,
GQA, and global prefill refreshes and the original full matrix. The two legacy top-level A100 system records are
preserved verbatim under `historical/legacy-system-records/` before removal from the experiment directory.

That preceding archive has SHA-256 `fe4528ce7231461d424def0966b0cf75e4c02d7564a8c7ec5171747a049321cd`.
All 9,703 direct files byte-match the retained tree. The credential scan found no matches in 9,702 direct files
scanned before its receipt was written, or in 20,692 regular files inside 29 recursively inspected archives.

## October 13 selected causal and GQA prefill refresh

Causal and grouped-query prefill now use the narrowly qualified `FAST_EXP` variant while keeping every original
per-length schedule. Umbrella `FAST_MATH` remains off and MMA accumulation remains FP32. Controlled comparisons
reduce the geometric mean Emmy / paired `torch.compile` ratio from 1.200 to 1.057 for causal prefill and from 1.219
to 1.081 for GQA prefill. These are 11.9% and 11.3% reductions in the paired ratio, respectively. The final official
gaps are 5.9% and 8.0%. Both families still trail PyTorch on most shapes. This is an arithmetic tradeoff accepted by
the unchanged strict tolerance, not a standard-math optimization or a new five-family comparison with Neptune.

### Fresh official results

Both selected experiment rows succeeded. The official replay covers seeded FP16 attention at batch 1 and head
width 128, with 32 query and KV heads for causal prefill and 64 query / 8 KV heads for GQA. Both are causal. Each
value is the mean of two independent minimum-of-15 CUDA-graph measurements, with one warmup. Emmy, eager PyTorch,
and full-graph `torch.compile` use the same inputs within each invocation. Every Emmy invocation launches one kernel.

| Length | Causal Emmy, µs | Causal paired PyTorch, µs | Causal ratio | GQA Emmy, µs | GQA paired PyTorch, µs | GQA ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.387 | 23.843 | 1.023 | 40.562 | 35.982 | 1.127 |
| 512 | 58.995 | 58.434 | 1.010 | 70.577 | 70.065 | 1.007 |
| 1024 | 134.437 | 119.954 | 1.121 | 194.867 | 191.181 | 1.019 |
| 2048 | 332.971 | 335.701 | 0.992 | 613.888 | 566.784 | 1.083 |
| 4096 | 1113.600 | 1031.168 | 1.080 | 2123.776 | 1926.144 | 1.103 |
| 8192 | 3627.008 | 3384.320 | 1.072 | 6242.304 | 5684.224 | 1.098 |
| 16384 | 12282.880 | 11312.640 | 1.086 | 24282.112 | 22067.712 | 1.100 |
| 32768 | 48643.583 | 44475.904 | 1.094 | 96435.711 | 86948.864 | 1.109 |
| Geometric mean of ratios | — | — | 1.059 | — | — | 1.080 |

The remaining average gaps are 5.9% for causal prefill and 8.0% for GQA. Only one of eight causal point estimates
favors Emmy; all eight GQA estimates favor PyTorch. These descriptive results do not establish statistical wins.
All 32 official strict checks pass at unchanged `rtol=atol=0.001`, with maximum absolute error 0.0009765625.
All 32 complete schedule dictionaries and runtime CUDA hashes match the qualified and recorded selected arms.

Causal length 8192 has the largest absolute repeat spread: Emmy changes from 3998.720 to 3255.296 µs while paired
PyTorch changes from 3752.960 to 3015.680 µs. Using `(max-min)/min`, these spreads are 22.84% and 24.45%; the paired
ratio changes only 1.31%. The largest causal paired-ratio spread is 2.90% at length 2048. All other causal raw
backend spreads are at most 1.54%. GQA's maximum spreads are 0.90% for Emmy, 0.91% for PyTorch, and 0.92% for the
paired ratio. The archive preserves every observation; the source of the shared absolute-time shifts is unknown.

### Selected arithmetic and controlled evidence

Only the independent exponential decision changes: `FAST_MATH=false`, `FAST_EXP=true`, `F16_MMA_F32_ACC=false`,
and `FP8_MMA=false`. All 16 original schedules, programs, kernel bodies, and routing decisions are preserved.
At every length, the complete generated CUDA source differs from the standard arm only by `expf` becoming
`__expf`. The new measured rows come from fresh single-regime recording databases; standard-math measurements
were not reused. Each golden's isolated timing equals its recording DB median. Its whole-target seed latency
comes from that invocation's whole-forward receipt, which is a separate measurement.

Qualification ran at revision `301b7ab1907cf19b45c5ad29f29382413c97af53`, with five warmups, 20 captured iterations,
deployable O3, a fresh DB per process, and explicit schedule and precision pins. The table uses the first matched
standard/candidate pair at each of eight lengths; repeated 2048 observations are reported separately below.

| Family | Standard Emmy / paired PyTorch | Selected Emmy / paired PyTorch | Reduction in paired ratio |
| --- | ---: | ---: | ---: |
| Causal prefill | 1.199736 | 1.056763 | 11.92% |
| GQA prefill | 1.218828 | 1.080602 | 11.34% |

At length 2048, three alternating pairs give median paired ratios of 1.134945 → 0.996951 for causal prefill and
1.256881 → 1.085740 for GQA. Some absolute timing shifts affect Emmy and its paired PyTorch baseline together.
The second causal pair even makes the candidate's absolute time slower while its paired ratio improves. These
observations do not identify a clock cause, and an absolute-time change alone is not the optimization estimate.
Inputs are seeded in the unchanged source and shared by all backends within an invocation; cross-process input
bytes were not independently hashed.

Four `WORK=w8x1` probes at lengths 2048 and 8192 also pass strict accuracy, but lose to the selected schedules.
Their paired ratios are 1.098 and 1.154 for causal, and 1.217 and 1.183 for GQA. They remain rejected diagnostics.
Removing the inner asynchronous stage emits identical CUDA for the inspected schedules, so it was not timed as
an independent arm. All 44 qualification invocations and all 16 recording invocations pass the original strict
check; the largest absolute error is 0.0009765625. Neither hardware goldens nor prior weights change.

### Reproduction and retained evidence

The official run is `2026-10-13_22-08-07`, ID `20261013T220807Z`, at clean revision
`6bc0b5256fc3b20af3d010ed6076c9f57ab3fca1`. It selects `lane=emmy` and the causal and GQA prefill operators on the
retained A100-SXM4-40GB, using PyTorch 2.13.0+cu130, CUDA 13.0.88, driver 580.178.04, and deployable O3. All 467
staged input files match that revision. Qualification and recording use revision
`301b7ab1907cf19b45c5ad29f29382413c97af53`; runtime CUDA is byte-identical at every selected length.

The recipe now explicitly imports its staged checkout. On this run, the existing setup fallback rebuilt the
missing native extension and reinstalled editable Emmy in the reused cloud environment. The rebuilt native
binary has the same SHA-256 as the qualified runtime, and every package version is unchanged. The import paths,
source manifests, setup log, native digest, and dependency comparison are retained. The local Python environment
was not changed, and no local GPU workload was run.

The preceding causal/GQA prefill archive is rooted at `2026-10-13_22-08-07/`. It contains the two unchanged system-only
records `a100x1_lemmy_op-c_ebd1c70330df.experiment.yaml` and `a100x1_lemmy_op-g_c33c3729250e.experiment.yaml`, their
matching `_artifacts.tar.gz` files, and the client logs. Each nested artifact holds all 16 source receipts under
`evidence/emmy-tcompile/json/`, their logs, setup status, GPU and software information, and task-local evidence.
Both downloaded artifacts match their remote size and SHA-256.

`diagnostics/` retains qualification and recording receipts, databases, commands, generated CUDA and dumps,
original schedules, rejected alternatives, independent reviews, promotion audits, and final numeric/source
checks. An offline inspection with only an SM80 capability target initially differed at a short causal shape:
its default SM count differed from the real A100. That diagnostic is retained separately; actual-card runtime
hashes are the comparison above. It did not require a compiler change. A subsequent strict-evidence compile on
the live A100 selects all 16 exact measured rows without manual schedule or precision pins. Each uses a fresh DB;
all emitted CUDA hashes match the recordings. Those receipts are retained under `diagnostics/actual-card/`.

The untouched preceding archive is nested at `historical/results_a100x1-before-causal-gqa.tar.gz`, with SHA-256
`b4c96af44019e14db2b93957c374dd7a8f093ecd73c0ef003d6de84a65cc1e82`. It preserves the selected global refresh and,
in turn, the earlier full matrix and Neptune evidence. Neither Neptune nor either decode family was rerun here.
The earlier five-family win counts remain tied to their original standard-math run; they are not recomputed by
mixing this selected arithmetic into historical measurements.

That prefill archive has SHA-256 `8405c7bf56f51978707eebd80caa6ca0a4fdaa913fa4d39fe55e1a3b780d4fa1`.
All 10,110 direct files byte-match the retained run tree. The credential scan found no matches in the 10,109
files scanned before its receipt was written, or in 7,552 regular files inside 116 nested archives.

## Historical October 13 selected global prefill refresh

Global prefill now uses an explicitly qualified `FAST_EXP` variant, with `FAST_MATH` off and FP32 MMA accumulation.
Across eight controlled lengths, its geometric mean Emmy / paired `torch.compile` ratio falls from 1.369 to 1.076,
a 21.4% reduction. The final official replay gives 1.074, so a 7.4% gap remains and only two of eight point estimates
favor Emmy. All 16 official strict checks pass unchanged. This is a partial improvement in one attention family,
not a standard-math speedup or a new reproduction of the paper's five-family result.

### Fresh official results

The selected row covers seeded noncausal FP16 attention at batch 1, 32 heads, head width 128, and equal query/key
lengths 256–32768. Each value below is the mean of two independent minimum-of-15 CUDA-graph measurements, with
one warmup. Each process measures Emmy and its eager and full-graph PyTorch 2.13 baselines on the same inputs.
Every Emmy invocation launches one kernel, so the kernel receipt and whole-forward timing coincide.

| Length | Emmy, µs | Paired `torch.compile`, µs | Emmy / paired `torch.compile` |
| ---: | ---: | ---: | ---: |
| 256 | 23.966 | 20.224 | 1.185 |
| 512 | 53.120 | 53.760 | 0.988 |
| 1024 | 144.823 | 148.773 | 0.973 |
| 2048 | 506.368 | 452.864 | 1.118 |
| 4096 | 1824.768 | 1728.512 | 1.056 |
| 8192 | 5704.192 | 5191.680 | 1.099 |
| 16384 | 22709.248 | 20816.896 | 1.091 |
| 32768 | 91827.202 | 83856.899 | 1.095 |
| Geometric mean of ratios | — | — | 1.074 |

The maximum two-process Emmy timing spread and paired-ratio spread are both 0.608%, using `(max-min)/min`.
The largest absolute error is 0.00048828125; every source proof passes the unchanged `rtol=atol=0.001` check.
All 16 complete schedule dictionaries and CUDA source hashes match the corresponding qualified candidate.
These descriptive point estimates do not establish statistical superiority at the two faster shapes.

The selected schedule uses `WORK=w4x1`, outer PV tile `mma_m16n8k16_f16_f32/f1x16/k8`, inner QK tile
`mma_m16n8k16_f16_f32/f1x16/k4`, and one asynchronous shared-memory stage on the outer carrier. Its four precision
pins are `FAST_MATH=false`, `FAST_EXP=true`, `F16_MMA_F32_ACC=false`, and `FP8_MMA=false`. Raw JSON calls this
`lane=fm` because one precision-trading knob is enabled; umbrella fast math remains off. No cut or fusion change
was made, and hardware goldens and prior weights are unchanged.

### Controlled comparison and rejected alternatives

The controlled eight-length comparison uses five warmups and 20 captured iterations at deployable O3, a fresh DB
for each process, explicit evidence scope, and no evidence writes. At 2048, three alternating standard/candidate
pairs also leave schedule pins unset so the corresponding frozen golden selects its measured row. Standard
Emmy / paired PyTorch ratios are 1.401130, 1.401130, and 1.402260; candidate ratios are 1.117514, 1.117514, and
1.116528. The median ratio drops 20.2%. Median absolute Emmy time drops from 634.880 to 506.368 µs. The third
candidate process is faster for both backends, without evidence attributing that shift to clocks. Input bytes
were not independently hashed between processes; the source fixes the random seed, and inputs are shared by the
backends within each invocation. The raw receipts preserve every timing and error magnitude.

The exact 2048 attention identity is absent from the A100 hardware golden. Default hardware scope falls through
to the prior and chooses a 4333.6 µs schedule, rather than a competing measured row. Standard-math tile/stage
alternatives do not close the gap; the best smaller-chunk observation is 610.3 µs. Refused schedules and unchecked
isolated AB timings remain diagnostic evidence, not qualification. Only direct-source strict checks accept the
selected candidate. Every selected length improves its paired ratio in the controlled sweep.

A value-wait overlap compiler prototype was discarded after three alternating pairs: medians were 635.392 µs
before and 634.880 µs after, a 0.08% difference. A broader `FAST_MATH=true` candidate gave 497.152 µs in one
observation, only 1.7% below the narrow variant's 505.856 µs observation; it was not selected. Nsight Compute
records lower tensor activity and more LSU instructions than Torch flash for the original kernel, but similar
measured occupancy. Those observations do not establish one causal bottleneck.

### Reproduction and retained history

The official run is `2026-10-13_20-09-51`, ID `20261013T200951Z`, at clean revision
`e52a264b75e234fec3c9b5136d5ccdefcac65666`. It ran only `lane=emmy, operator=prefill_global`, using the retained
A100-SXM4-40GB, PyTorch 2.13.0+cu130, CUDA 13.0.88, driver 580.178.04, and deployable O3. All 467 staged source
files match the committed inputs. Qualification used `b4c01478cdfc42d4c1170acc4e4643a38a34e30b`; recordings used
that revision for 2048 and `27cecb6f401bf1b63b48ad458f835f3690ce13f9` for the other seven lengths. Their selected
CUDA sources are byte-identical to the final official run at every length.

Every recording started with a new single-regime DB and an unmeasured candidate seed. Each final row's measured
latency equals its CLI receipt and sole measured DB row. Those recordings did not mix standard and `FAST_EXP`
evidence. The final compiler additionally partitions independent precision decisions in evidence contexts, so a
later mixed-regime DB cannot silently substitute a standard-math measurement for this variant.

The global-refresh archive, retained inside the preceding prefill archive, contains its system-only
row and raw artifact, numeric receipt, qualification diagnostics, source attestations, and the untouched previous
archive under
`historical/results_a100x1-before-prefill.tar.gz`. That historical archive retains all 15 earlier rows, all other
Emmy families, and the pinned Neptune profiles; none is relabeled as a fresh measurement here. Its SHA-256 is
`ce1420d612d68ca5de1101b7544b4471387bcc3a493f95e32d4ef645246a3c71`. Neptune schedules and its artifact were not
retuned or rerun for this selected global refresh. The earlier 40-shape counts below remain tied to their original
standard-math run and are not recomputed by mixing in this arithmetic variant.

The global-refresh archive SHA-256 is `b4c96af44019e14db2b93957c374dd7a8f093ecd73c0ef003d6de84a65cc1e82`.
All 140 direct files byte-match the retained run tree. The current remote artifact matches its downloaded copy
by digest and size. The credential-pattern scan found no matches across the direct files and 7,312 regular files
inside 113 nested archives, including the untouched historical evidence.

## Historical October 13 full matrix refresh

All 15 experiment rows succeeded, but the current compiler does not reproduce the paper's 34/40 wins over Neptune.
Using the historical convention—one-kernel prefill timings and separate decode kernel sums—the current count is
18/40. Whole-forward timings give 20/40 lower point estimates. GQA decode remains faster on all eight shapes; the
other four families are slower than current Inductor by geometric mean. These are descriptive counts, not
statistically established wins: separate process measurements vary substantially on some shapes.

### Earlier same-run Emmy and paired PyTorch 2.13

Every family completed all eight lengths, 256–32768, and both strict source repetitions. All 80 source proofs passed
`rtol=atol=0.001`; the largest absolute error was 0.0009765625. Decode also completed all 32 separate exact-golden
replays. This is 112 unique CLI invocations, not 160: prefill's 48 source invocations also perform its exact-pin replay.

Each whole-forward value is the mean of two independent minimum-of-15 CUDA-graph measurements, with one warmup.
Emmy, eager, and full-graph `torch.compile` are measured in the same source invocation. The kernel-receipt column
uses those same single-kernel prefill values, but sums the explicit per-kernel receipts from the two separate decode
replays. Current `greedy.total_us` is whole-forward time; decode kernel sums come from `greedy.kernels[].us`.
Neptune is the mean of 15 GPU-projected NVTX ranges for the selected artifact schedule. Ratios below one favor Emmy.

| Family | Row status | Whole Emmy / paired Inductor | Whole Emmy / Neptune | Kernel receipts / Neptune | Lower than Neptune: whole / kernel receipts |
| --- | --- | ---: | ---: | ---: | ---: |
| Global prefill | Succeeded, 8/8 | 1.367 | 1.070 | 1.070 | 2/8 / 2/8 |
| Causal prefill | Succeeded, 8/8 | 1.203 | 1.027 | 1.027 | 4/8 / 4/8 |
| GQA prefill | Succeeded, 8/8 | 1.216 | 1.005 | 1.005 | 4/8 / 4/8 |
| Causal decode | Succeeded, 8/8 | 1.198 | 1.109 | 1.121 | 2/8 / 0/8 |
| GQA decode | Succeeded, 8/8 | 0.377 | 0.799 | 0.845 | 8/8 / 8/8 |

These are geometric means over eight shape ratios. The fresh paired Inductor comparison is independent of the
September comparison: compiler, driver, and other environment drift prevent attributing historical changes solely
to the compiler. The historical kernel-receipt win counts were 6, 7, 7, 6, and 8 respectively. No selected schedule
was retuned to remove the regressions.

Whole-forward repeat spreads reach 11.8% for global prefill, 26.2% for causal prefill, 21.9% for GQA prefill, 29.1%
for causal decode, and 0.91% for GQA decode, measured as `(max-min)/min`. Decode kernel-receipt spreads reach 28.7%
and 27.0%. Several shifts affect Emmy and its same-process baseline together; the largest paired-ratio spread is
11.9%. Separate replay kernel sums can therefore exceed a separately measured whole-forward mean; subtracting
those columns does not estimate launch overhead. Neptune ratios are directional comparisons between environments,
not paired end-to-end results. `current-common-attention.csv` preserves every shape and both repetitions.

### Pinned artifact coverage and baseline selection

All ten artifact rows succeeded, with 80/80 profiles and 476 provider/shape groups containing exactly 15 numbered
repetitions each (7,140 timed ranges). There were no profile timeouts, export failures, or runner exceptions.
Nineteen common-prefill shapes retain the historical Tri Dao Triton mismatch: global at 256–1024, and all causal
and GQA prefill lengths. Both non-Neptune ALiBi providers disagree on all 16 ALiBi shapes. Other supported providers
passed the artifact's cross-provider check, whose reference is one Neptune manual schedule rather than an
independent oracle.

This table belongs entirely to the pinned PyTorch 2.6 artifact. For each shape, its numerator is the lowest
mean-of-15 GPU range among valid non-Neptune artifact providers; its denominator is the lowest mean-of-15 among
available manual and saved tuned Neptune schedules. The table is the geometric mean of those eight ratios.
Values above one favor Neptune. This is separate from the paired PyTorch 2.13 comparison above.

| Artifact family | Row status | Fastest valid artifact provider / Neptune | Eligible non-Neptune providers |
| --- | --- | ---: | --- |
| Global prefill | Succeeded, 8/8 | 0.801 | Common prefill set; Tri Dao Triton also eligible at 2048–32768 |
| Causal prefill | Succeeded, 8/8 | 0.779 | Common prefill set |
| GQA prefill | Succeeded, 8/8 | 0.770 | Common prefill set |
| Causal decode | Succeeded, 8/8 | 1.033 | Flex, torch(mem-eff), Tri Dao CUTLASS, FlashInfer, Triton xformer |
| GQA decode | Succeeded, 8/8 | 1.207 | Flex, torch(mem-eff), Tri Dao CUTLASS, FlashInfer |
| ALiBi prefill | Succeeded, 8/8 | Excluded | None: Flex and Tri Dao CUTLASS both mismatch |
| ALiBi decode | Succeeded, 8/8 | Excluded | None: Flex and Tri Dao CUTLASS both mismatch |
| Softcap prefill | Succeeded, 8/8 | 1.038 | Flex |
| Softcap decode | Succeeded, 8/8 | 5.048 | Flex |
| Windowed prefill | Succeeded, 8/8 | 0.661 | Flex and Tri Dao CUTLASS |

The common prefill set is Flex, torch(flash), torch(cudnn), Triton OpenAI, Tri Dao CUTLASS, and FlashInfer.
`neptune-provider-timings.csv` preserves provider names, repetition counts, mismatch flags, and original artifact
member paths. Original per-repetition CSVs and profiles remain authoritative. Softcap decode still differs
substantially from the paper's 1.86x compiler comparison, so the complete Neptune paper comparison remains unresolved.

Selected Neptune mean times relative to `paper-baselines.csv` have family geometric means of 0.997x, 0.997x, 1.018x,
0.992x, and 1.001x for the five common families in table order. One shape is materially slower: GQA prefill at 16384
is 32.785 ms versus 28.992 ms for the same archived manual schedule. It was retained unchanged. Selected Neptune
repeat CV reaches 7.62% among common families and 12.60% across all artifact families (ALiBi prefill).

### Qualification, capture, and provenance

All 40 Emmy golden inputs were freshly lowered and re-recorded on this exact A100 with five warmups, 20 iterations,
standard math, deployable `-O3`, and strict checks. They contain 54 measured schedule rows and 14 split routes, with
no old timings copied from retired YAML files. Qualification source was
`f60a7e2fd8eb949041ae078c68fbe5d8b3a86feb`. Every official invocation's emitted kernel source hashes and complete knob
dictionaries exactly match its qualification receipt; all 40 staged golden hashes match the committed inputs.

Global and GQA prefill share an abbreviated remote directory within each lane. The later GQA artifacts retain
global files and nest the previous global artifact; their per-run databases and cubin directories are also reused.
They are not fresh per operator. Coverage uses operator-qualified filenames; the 16 duplicated native global JSONs
and eight duplicated external global CSVs are byte-identical. `receipt-verification.csv` records all 112 unique native
receipts. This reuse did not alter any selected kernel or schedule, as the receipt and input-hash checks above confirm.

The artifact replays schedules selected in historical run `2026-08-16_00-41-38` on an A100-SXM4-80GB. This run
re-measures those schedules on the A100-SXM4-40GB; it does not repeat the 128-trial search or claim that the saved
choices are newly optimal for the 40GB card. Neptune remains at revision
`3aa55c12ac822337e630b809b0d9eabb11eee5d3` in image
`evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`, with
PyTorch 2.6.0+cu124 and Nsight Systems 2025.3.1.90. The host lane uses PyTorch 2.13.0+cu130, CUDA 13.0.88, driver
580.178.04, and one NVIDIA A100-SXM4-40GB on `bench-cgo27-a100-1012-2255-2fec` (12 logical Xeon CPU threads).
This 40GB snapshot is retained inside the latest archive; legacy 80GB records and the mixed-card historical
archive remain.

The capture harness now requests CUDA/NVTX only, with CPU sampling, context-switch sampling, and CUDA backtraces
disabled. Selective SQLite export retains every table read by the GPU-projection report. Applied to the same original
profile, full and selective exports produced byte-identical CSV. Before/after capture validation retained all 165
measured ranges and identical GPU-operation counts; ten stable provider means moved by at most 0.6%. Flex moved
12.7% with 17.35% repeat CV and is not treated as a capture-induced speedup. Profile plus export fell from 342.7 to
20.7 seconds in that validation. All 15 final rows were restarted from one clean source after the harness change.

The final run is `2026-10-13_07-25-52`, ID `20261013T072552Z`, clean revision
`3284a500c6f5189f0ec3f6d53f22108e11a64257`. Its archive is now nested as
`historical/results_a100x1-before-prefill.tar.gz` (SHA-256
`ce1420d612d68ca5de1101b7544b4471387bcc3a493f95e32d4ef645246a3c71`), containing all 15 system-only row records,
per-row artifacts, the three numeric CSVs, `golden-recording.tar.gz`, and `capture-diagnostics.tar.gz`.
The diagnostics preserve interrupted invocation
`20261013T070012Z`, raw before/after profiles, CSVs, commands, source hashes, phase durations, and logs. Completed
derived SQLite caches are omitted; preserved Nsight profiles reproduce them.
The completed secret-pattern scan found no matches in 4,467 text-like files, including nested archives.

The sections below preserve earlier measurements. Their replaced 40GB snapshot also remains in Git history; the
retained top-level 80GB records describe a different exact card and its historical archive.

## Historical conclusion (2026-09-10)

The compiler now matches FlashAttention-2 on causal and GQA prefill and is ahead of Neptune on both, on the same
NVIDIA A100-SXM4-40GB the Neptune paper used. Re-measured on 2026-09-10 after the chunk-loop density change (the
staged slab swizzle hoisted per lane, each chunk folded at the advanced pivot), with no golden retuned: causal prefill
is 1.02x of eager and 0.87x of Neptune by geometric mean over the eight lengths, GQA prefill 1.03x and 0.88x, each
ahead of Neptune at seven of eight lengths. Global prefill, whose stream has no early stop, improved from 1.54x to
1.17x of eager and is ahead of Neptune at six of eight lengths (0.94x) but still trails FlashAttention-2. GQA decode,
replayed from the split-KV goldens, is 0.81x of Neptune and ahead at all eight lengths; on the paper's scale the
family's geometric mean over Inductor is 0.27 for Emmy against 0.34 for Neptune. Causal decode, the one family the
compiler could not schedule at all, is now ahead of Neptune too: binding the size-one query row it had lost lets it
reach the fragment tiers, and its eight re-recorded goldens are 0.84x of Neptune by geometric mean, ahead at six of
eight lengths, each shape 2.5x to 6.8x faster than the row it replaces. That column is a manual re-recording rather
than a lane rerun; the section below says what that costs.

The two earlier implementation results still stand under the new numbers: the causal early stop on the chunk tier
(masked chunks skipped rather than folded), which halved the long causal and GQA rows, and the split-KV partial that
stores the chunk tier's row states whole, which is what put GQA decode ahead. The remaining gaps are global prefill's
schedule (its goldens pin a weaker row than causal's; the causal geometry measured 10% faster at 2048 keys and 12%
slower at 512, so the retune is per length) and causal decode's register pressure, which is what keeps its two
longest lengths level with Neptune rather than ahead.

## Prefill and GQA decode after the chunk-loop density change

The three prefill families and GQA decode were re-measured on 2026-09-10 on the same host at revision
`3394fd03cbeac90b76e812875d2812b801c683d6`, after the compiler learned to hoist the staged slab swizzle per lane and
to fold each attention chunk at the advanced pivot, so the chunk's P·V accumulates straight into the carrier. No golden
was retuned: the lane replayed the committed rows, and the difference to the previous section is the compiler's.

The lane ran through `emmy bench --local --filter lane=emmy`, one invocation for `operator=prefill_*` and one for
`operator=decode_gqa`, each setup measured twice at deployable `-O3` with one warmup and 15 captured iterations and
strict eager correctness at `rtol=1e-3, atol=1e-3`. All 32 setups completed both repetitions and every Emmy
measurement passed correctness. Each latency below is the mean over the two repetitions of the minimum of 15; Neptune
is the replay experiment's mean of 15 (`paper-baselines.csv`). For GQA decode the Emmy latency is the replay's kernel
sum for the full kernel-set receipt row, as in the split-KV section below.

| Operator | Emmy / eager | Emmy / `torch.compile` | Emmy / Neptune | Emmy wins vs Neptune | previous Emmy / eager |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.17x | 1.17x | 0.94x | 6/8 | 1.54x |
| Prefill causal | 1.02x | 1.02x | 0.87x | 7/8 | 1.41x |
| Prefill GQA | 1.03x | 1.03x | 0.88x | 7/8 | 1.44x |
| Decode GQA | 0.04x | 0.34x | 0.81x | 8/8 | 0.05x |

Lower ratios favor Emmy; the summaries are geometric means over the eight lengths, and the previous column is the
section below this one, measured 2026-09-09.

**Prefill causal**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.4 | 23.8 | 24.3 | 26.6 | 1.02 | 0.92 | 1.15 |
| 512 | 58.6 | 58.8 | 58.7 | 54.2 | 1.00 | 1.08 | 1.13 |
| 1024 | 124.7 | 122.4 | 120.3 | 129.3 | 1.02 | 0.96 | 1.42 |
| 2048 | 315.6 | 335.4 | 336.4 | 375.4 | 0.94 | 0.84 | 1.39 |
| 4096 | 1056.3 | 1042.4 | 1040.9 | 1129.8 | 1.01 | 0.93 | 1.50 |
| 8192 | 3438.6 | 3362.8 | 3360.8 | 4389.3 | 1.02 | 0.78 | 1.56 |
| 16384 | 11872.8 | 11354.1 | 11311.1 | 16581.8 | 1.05 | 0.72 | 1.60 |
| 32768 | 47180.8 | 44305.4 | 44499.5 | 61207.6 | 1.06 | 0.77 | 1.63 |

**Prefill GQA**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 36.1 | 36.1 | 36.3 | 35.8 | 1.00 | 1.01 | 1.06 |
| 512 | 69.7 | 70.6 | 70.2 | 81.3 | 0.99 | 0.86 | 1.31 |
| 1024 | 197.8 | 193.1 | 192.3 | 217.8 | 1.02 | 0.91 | 1.41 |
| 2048 | 578.0 | 567.0 | 561.4 | 619.2 | 1.02 | 0.93 | 1.51 |
| 4096 | 2017.8 | 1925.6 | 1922.0 | 2113.7 | 1.05 | 0.95 | 1.56 |
| 8192 | 6015.5 | 5710.8 | 5708.3 | 8522.9 | 1.05 | 0.71 | 1.58 |
| 16384 | 23493.6 | 22295.0 | 22136.3 | 28992.0 | 1.05 | 0.81 | 1.62 |
| 32768 | 93751.8 | 88208.9 | 88154.1 | 100425.6 | 1.06 | 0.93 | 1.60 |

**Prefill global**

| keys | Emmy (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / eager | Emmy / Neptune | previous Emmy / eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 22.9 | 20.1 | 20.0 | 24.1 | 1.14 | 0.95 | 1.31 |
| 512 | 57.7 | 54.0 | 53.6 | 64.9 | 1.07 | 0.89 | 1.30 |
| 1024 | 160.8 | 149.6 | 148.5 | 191.8 | 1.07 | 0.84 | 1.36 |
| 2048 | 556.0 | 454.7 | 452.6 | 551.7 | 1.22 | 1.01 | 1.59 |
| 4096 | 1999.9 | 1721.3 | 1716.2 | 2001.6 | 1.16 | 1.00 | 1.54 |
| 8192 | 6612.5 | 5508.6 | 5492.7 | 7502.6 | 1.20 | 0.88 | 1.61 |
| 16384 | 25008.6 | 20663.3 | 20809.2 | 29564.6 | 1.21 | 0.85 | 1.63 |
| 32768 | 103875.1 | 81058.8 | 84664.8 | 92427.5 | 1.28 | 1.12 | 2.11 |

**Decode GQA** (the replay's kernel sum; the whole-forward capture, eager and Inductor from the same lane)

| keys | Emmy, kernel sum (us) | Emmy, whole (us) | eager (us) | Inductor (us) | Neptune (us) | Emmy / Inductor | Emmy / Neptune | previous kernel sum (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 9.7 | 11.4 | 74.8 | 17.2 | 13.1 | 0.57 | 0.74 | 9.2 |
| 512 | 11.1 | 12.5 | 110.0 | 27.5 | 13.6 | 0.40 | 0.81 | 17.5 |
| 1024 | 13.1 | 14.7 | 206.3 | 28.6 | 16.7 | 0.46 | 0.79 | 17.3 |
| 2048 | 14.7 | 16.3 | 392.2 | 41.8 | 19.8 | 0.35 | 0.74 | 18.5 |
| 4096 | 22.3 | 24.3 | 915.5 | 94.7 | 30.2 | 0.24 | 0.74 | 25.2 |
| 8192 | 45.0 | 46.7 | 1509.4 | 131.1 | 52.9 | 0.34 | 0.85 | 42.6 |
| 16384 | 78.7 | 80.7 | 3511.8 | 312.8 | 87.8 | 0.25 | 0.90 | 82.3 |
| 32768 | 137.0 | 141.6 | 6739.5 | 564.7 | 143.2 | 0.24 | 0.96 | 146.5 |

Each decode setup's replay directory holds one record per golden row; the finalize's receipt and the routing row
pin the partial kernel's knobs only through the route and replay it on a scalar tile there (hundreds of
microseconds to milliseconds), exactly as in the previous archive, so the table reads the full kernel-set receipt.
The reference arm's Emmy number is again not reported for decode: without golden evidence the greedy deploys the
prior's pick. The two `op-g` rows (global and GQA prefill) share one task directory name, so the GQA row's artifact
archive also carries the global row's evidence files; each setup file was read once.

The run-to-run spread on this host is the one noted below: eager itself moved by up to 25% between separate
processes at 2048 keys during the development measurements, so the same-process ratios are what the tables rest on,
and short-row ratios within a few percent of 1.00 are ties.

- Prefill run: `2026-09-10_07-23-33`, run ID `20260910T072333Z`; GQA-decode run: `2026-09-10_07-41-21`, run ID
  `20260910T074121Z`
- Git revision: `3394fd03cbeac90b76e812875d2812b801c683d6`; dirty: false
- Host, GPU, driver and toolkit as in "Run and system" below; PyTorch 2.13.0 with CUDA 13.0 in the staged repo's
  venv
- Archive: `results_a100x1.tar.gz` (both run directories with their records, per-row artifacts and logs); SHA-256
  `41115796b3f07a6958b505fa4ffeef922d5c9057a9832fb3a9eed6cf67401872`

## Causal decode after the bound row

Causal decode was the one family the compiler could not schedule. A query of one token per head leaves no axis for
the loop nest to iterate, so normalization inlined that coordinate as a constant and the contraction that remained
shared every coordinate it had with the value it multiplies. `TileOp.contracts` refuses such a term — a B that moves
with the row it is contracted against is no slab per tile — so the family fell to the per-cell tier, where each of
the 128 output channels re-ran the score's own 128-step contraction. No pin in the schedule space avoided that, so
the committed goldens were the best rows of a space that could not express the kernel.

Binding the coordinate back as an extent-one axis gives the term a row, and decode then traces to the same fused
chunk-tier kernel prefill uses. All eight goldens were re-traced and re-recorded on 2026-09-10 on the same
A100-SXM4-40GB host. Five of the eight also take a cross-CTA key-range split, whose width scales to hold roughly 256
keys per partition.

| Sequence | Emmy (us) | Eager (us) | `torch.compile` (us) | Neptune (us) | Emmy / Neptune | Split | Emmy before (us) |
| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| 256 | 8.817 | 22.528 | 19.456 | 15.162 | 0.58x | none | 24.680 |
| 512 | 15.360 | 27.648 | 24.576 | 19.027 | 0.81x | none | 38.841 |
| 1024 | 18.725 | 35.840 | 30.720 | 27.484 | 0.68x | `g8k` | 73.169 |
| 2048 | 42.583 | 48.128 | 41.984 | 45.496 | 0.94x | `g8k` | 148.541 |
| 4096 | 63.982 | 70.144 | 62.976 | 74.266 | 0.86x | `g16k` | 294.468 |
| 8192 | 117.617 | 119.296 | 110.592 | 122.328 | 0.96x | `g32k` | 607.300 |
| 16384 | 220.910 | 218.112 | 203.264 | 216.662 | 1.02x | `g64k` | 1384.960 |
| 32768 | 410.820 | 409.088 | 390.144 | 410.271 | 1.00x | `g64k` | 2775.040 |

By geometric mean over the eight lengths Emmy is 0.84x of Neptune and 0.74x of eager, ahead of Neptune at six of
eight lengths and level with it at the two longest. Every shape is 2.5x to 6.8x faster than the row it replaces. The
family that was 3.5x behind Neptune is now ahead of it.

**How these numbers were produced, and what that costs.** The Emmy column is not a lane rerun. Each value is the
routing row of a re-recorded golden — the kernel-set total from `emmy run --golden … --bench --record-greedy`'s
isolated re-bench, equal to the sum of its per-kernel receipts — measured during a manual pin sweep, not by
`emmy bench` on this recipe. The eager, `torch.compile` and Neptune columns are unchanged from the archived lane. So
every ratio in the table pairs a fresh Emmy measurement against a reference measured in a different session, and the
references are not neutral: this host measured eager at 31-35 us on the 2048-key shape during the sweep, against the
48.128 us the archived lane recorded. Taking that faster eager instead would move the 2048 row from 0.88x of eager to
about 1.3x. Rerunning `emmy bench … --filter lane=emmy --filter operator=decode_causal` would put both halves in one
process and settle it; until that runs, read the Emmy-versus-Neptune column as indicative rather than as a paired
measurement.

**The two schedule facts the sweep established**, both of which make a naive sweep of this family misleading:

- The geometry is a matched diagonal. The score tile's column count must equal the carrier's chunk width (`16 * k`).
  Off it — chunk 64 against 128 columns, or 128 against 64 — the identical kernel measures about 9300 us at 2048
  keys, 160x slower.
- A `TILE@map.1/twist` pin does not resolve on a split PIECE, whose tree spells the route `@twist`. The pin is
  ignored without complaint and the compiler picks that piece's geometry itself. Pinning only the one spelling made
  the split look harmful (137 us at 1024 keys, 511 at 4096); pinning both holds the geometry fixed, and the split is
  then worth 1.4x to 2x at every length.

## Split-KV GQA decode

The eight GQA-decode goldens now pin FlashAttention-2's split-KV: the key range splits across `n` CTAs
(`REDUCE@map.1/twist=g<n>k`, about 256 keys per CTA, `n` = 2 at 256 keys through 32 at 16384 and 32768), the partial
keeps the fused kernel's tensor-core chunk tier and writes its three carrier states to an f32 workspace, and a
serial finalize merges them per cell. This became recordable when the chunk tier learned to store a per-row carried
state whole (the pivot and the denominator broadcast into a fragment beside the expectation); before, the split's
partial fell to scalar tiles and the fused single-wave row ran on eight CTAs. The lane re-ran over the new goldens at
revision `38f7f6d9e` on 2026-09-10 on the same host: all eight setups completed both deployable-O3 replays and
both strict source references. Each Emmy latency is the replay's kernel sum, the mean over the two repetitions of the
minimum of 15.

| keys | Emmy replay, kernel sum (us) | Emmy whole (us) | Inductor, lane (us) | Inductor, paper baseline (us) | Neptune (us) | Emmy / Inductor | Neptune / Inductor |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 9.2 | 10.3 | 15.5 | 25.6 | 13.1 | 0.36 | 0.51 |
| 512 | 17.5 | 19.1 | 19.5 | 39.9 | 13.6 | 0.44 | 0.34 |
| 1024 | 17.3 | 18.9 | 26.8 | 41.5 | 16.7 | 0.42 | 0.40 |
| 2048 | 18.5 | 20.0 | 41.9 | 57.3 | 19.8 | 0.32 | 0.35 |
| 4096 | 25.2 | 26.9 | 92.4 | 95.2 | 30.2 | 0.26 | 0.32 |
| 8192 | 42.6 | 43.8 | 132.1 | 164.9 | 52.9 | 0.26 | 0.32 |
| 16384 | 82.3 | 84.4 | 310.3 | 314.9 | 87.8 | 0.26 | 0.28 |
| 32768 | 146.5 | 149.7 | 555.5 | 615.9 | 143.2 | 0.24 | 0.23 |

geometric mean over 8 lengths: Emmy / Inductor = 0.312, Neptune / Inductor = 0.335

Emmy is ahead of Neptune at 5 of the eight lengths and at 0.93x of its time by geometric mean; on the paper's
scale (Neptune's mean of 15 over Inductor's minimum, `paper-baselines.csv`) the family reads 0.31 for Emmy
against 0.34 for Neptune. The lane's own Inductor column is the reference arm's whole-forward capture in the
same process; the paper-baseline column is the earlier replay experiment's, which the paper table normalizes by.
The reference arm's Emmy number is not reported for decode: without golden evidence the greedy deploys the prior's
pick. The 64-way split at 32768 keys recorded well by hand pin but is outside the cut pass's offered widths, so the
unpinned deploy fell through to a scalar loop; the golden pins the 32-way split, which deploys.

The raw lane records are in `results_a10040_split_kv_decode_gqa.tar.gz`; `emmy_decode_gqa_lane.csv` carries the table
as numbers.

## Manually tuned Emmy on A100 40GB

The full run measured revision `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`. A second complete GQA-decode row at
`1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c` replaces its 2048 schedule with the qualified fused schedule. No result
uses `emmy tune`: each golden was selected manually, checked with five warmups and 20 measurements, then accepted only
after two fresh correct measurements.

All 40 shapes completed two deployable-O3 strict golden replays and two source-reference measurements. Every source
run passed eager correctness at `rtol=1e-3, atol=1e-3`. Each latency is the arithmetic mean of the two repetitions;
each repetition reports the minimum of 15 captured GPU measurements. Decode latency sums the realized golden kernels.

| Operator | Emmy / eager | Emmy / `torch.compile` | Emmy / Neptune | Emmy wins vs Neptune |
| --- | ---: | ---: | ---: | ---: |
| Prefill global | 1.54x | 1.54x | 1.23x | 0/8 |
| Prefill causal | 1.41x | 1.42x | 1.20x | 0/8 |
| Prefill GQA | 1.44x | 1.44x | 1.23x | 0/8 |
| Decode causal | 3.12x | 3.47x | 3.54x | 0/8 |
| Decode GQA | 0.05x | 0.42x | 0.93x | 5/8 |

Lower ratios favor Emmy. The corresponding per-shape Emmy latency is:

| Sequence | Prefill global (us) | Prefill causal (us) | Prefill GQA (us) | Decode causal (us) | Decode GQA (us) |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 26.435 | 28.032 | 38.183 | 24.680 | 9.200 |
| 512 | 70.473 | 66.731 | 93.510 | 38.841 | 17.500 |
| 1024 | 203.366 | 172.459 | 274.432 | 73.169 | 17.300 |
| 2048 | 728.576 | 462.592 | 844.800 | 148.541 | 18.500 |
| 4096 | 2670.592 | 1582.592 | 3007.488 | 294.468 | 25.200 |
| 8192 | 8174.592 | 5226.496 | 8929.280 | 607.300 | 42.600 |
| 16384 | 33263.617 | 17671.679 | 35146.751 | 1384.960 | 82.300 |
| 32768 | 172432.899 | 70041.088 | 137703.423 | 2775.040 | 146.500 |

After the early stop moved from the loop IR to the chunk tier (revision `4598f8f17`), the same goldens replayed on the
same card: causal prefill measured 28.0 us at 256 keys (28.032 above), 61.8 at 512 (66.731), 1233.9 at 4096 (1582.592)
and 4542.5 at 8192 (5226.496); GQA prefill measured 2337.8 at 4096 (3007.488). Repeated replays of the 256-key row spread
from 22.5 to 29.4 us and the 512-key row from 55.0 to 66.9, so the short rows carry a run-to-run spread of about 20% on
this VM. The tables above keep the full-run values.

`paper-emmy-a10040.csv` contains the exact 40 Emmy, eager, Inductor, and Neptune values behind both tables; its
GQA-decode Emmy column carries the split-KV lane values above, which replace the earlier fused and split rows.

## Historical tuned Emmy decode-causal follow-up

The follow-up measured the eight committed decode-causal goldens at revision
`5642d020259d0e09d49cbdab04e8e96408616b3e`. All eight shapes completed two deployable-O3 golden replays and two
strict source-reference invocations: 32/32 required invocations succeeded. Every replay realized exactly the two
expected golden schedules, every source comparison passed, and the largest Emmy absolute error was `2.59e-4` under
the experiment's `1e-3` tolerance.

Each latency below is the arithmetic mean of two independently launched repetitions; each repetition reports the
minimum of 15 captured GPU measurements. Replay latency is the sum of the two golden kernels. The final column compares
the same-input untuned greedy Emmy latency recorded beside each replay, so values above 1.00x favor the tuned schedules.

| Sequence | Tuned replay (us) | Eager (us) | `torch.compile` (us) | Untuned greedy Emmy (us) | Tuned vs greedy |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 24.522 | 13.102 | 12.651 | 73.179 | 2.98x |
| 512 | 38.306 | 17.262 | 16.603 | 149.285 | 3.90x |
| 1024 | 59.048 | 24.064 | 22.357 | 311.296 | 5.27x |
| 2048 | 125.897 | 38.912 | 33.280 | 640.000 | 5.08x |
| 4096 | 274.970 | 57.856 | 53.248 | 1281.024 | 4.66x |
| 8192 | 535.962 | 93.696 | 88.576 | 2551.808 | 4.76x |
| 16384 | 1152.640 | 163.328 | 175.616 | 5081.600 | 4.41x |
| 32768 | 2334.208 | 323.072 | 316.416 | 9318.400 | 3.99x |

Across the eight shapes, tuned replay was 4.32x faster than untuned greedy Emmy by geometric mean. It remained 3.82x
slower than eager and 4.02x slower than `torch.compile`. This row qualifies decode-causal only; it does not qualify the
other four Emmy operator families or change the broader Neptune comparison.

## Historical starter comparison

This table retains the full starter sweep. Its Emmy column describes the original untuned run; the tuned decode-causal
follow-up above is reported separately so historical failures are not rewritten.

The Neptune columns report `Inductor latency / Neptune latency`, so values above 1.00x favor Neptune. "Best" selects
the fastest measured manual or tuned Neptune schedule; "manual" uses only the artifact's fixed manual schedules. Each
summary is the geometric mean over eight sequence lengths from 256 through 32768.

| Operator | Inductor vs eager | Neptune vs Inductor, best | Neptune vs Inductor, manual | Full tunes | Valid Emmy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Prefill global | 1.02x | 0.84x | 0.82x | 8/8 | 8/8 |
| Prefill causal | 1.05x | 0.90x | 0.89x | 5/8 | 7/8 |
| Prefill GQA | 1.05x | 0.86x | 0.84x | 6/8 | 7/8 |
| Decode causal | 1.01x | 1.07x | 0.99x | 7/8 | 0/8 |
| Decode GQA | 7.67x | 3.01x | 2.90x | 8/8 | 0/8 |

Best-available Neptune beat Inductor on 15 of the 40 individual shapes: two prefill, five decode-causal, and all eight
decode-GQA setups. The decode-GQA speedup increased with context, from 1.68x at sequence 256 to 4.54x at sequence
32768. By contrast, Neptune's best prefill results ranged from 0.76x to 1.19x Inductor across individual shapes.

## Alignment with the Neptune paper

The [Neptune paper](https://arxiv.org/abs/2510.08726) reports the geometric-mean speedup of Neptune over the fastest
manually optimized library in Table 4. To compare with that table, the replay values below use the arithmetic mean of
the 15 projected GPU times for each implementation and then the geometric mean across the eight sequence lengths.
This is the closest aggregation available in the durable Nsight exports to the paper's mean-of-15 kernel rule. Values
above 1.00x favor Neptune.

| Operator | Paper, A100 | A100 40GB replay | Difference |
| --- | ---: | ---: | ---: |
| Prefill global | 0.84x | 0.80x | -5.1% |
| Prefill causal | 0.81x | 0.78x | -4.0% |
| Prefill GQA | 0.80x | 0.79x | -1.3% |
| Decode causal | 0.99x | 1.03x | +3.7% |
| Decode GQA | 1.24x | 1.21x | -2.4% |

The five comparable library results agree within 1.3--5.1%. Both the paper and this exact-card replay find that
optimized libraries beat Neptune on the three common prefill operators, while Neptune is competitive on causal
decode and ahead on GQA decode.

### Reconstructing the normalized paper table

`paper-baselines.csv` extracts the 40 shared shape measurements used by the paper. For each shape, the Neptune value
is the arithmetic mean of 15 projected GPU ranges for each available manual or tuned schedule, followed by selection
of the lower schedule mean. The Inductor value is the minimum of 15 captured, whole-forward CUDA-event measurements.
The per-family reproduced value is the geometric mean of `neptune_mean_us / inductor_min_us` over the eight sequence
lengths. This gives 1.20, 1.14, 1.15, 0.99, and 0.34 for global prefill, causal prefill, GQA prefill, causal decode,
and GQA decode, respectively. Lower values favor Neptune because current Inductor is normalized to one.

`paper-table.csv` records the published library-relative ratios, their artifact replay, and the displayed values after
putting both results on the current Inductor scale. For each operator, the bridge is

```text
paper Neptune / current Inductor
  = replayed Neptune / current Inductor
  * replayed library / Neptune
  / published library / Neptune
```

The Neptune paper publishes only two-decimal family ratios. The final column in `paper-table.csv` therefore records
the displayed paper value directly; recomputing from the rounded intermediate columns can differ by 0.01.

To audit `paper-baselines.csv`, use the sibling `compiler_neptune_replay_a100/results_a10040.tar.gz` archive. Neptune
ranges are under `nsys-stats/<operator>-b1-s<sequence>.csv`; current PyTorch JSON rows are under
`evidence/emmy-tcompile/json/`. Warmup ranges are excluded.

The paper's Table 2 reports Neptune relative to Triton, FlexAttention, TVM, and Mirage, rather than to the manually
optimized libraries. The pinned artifact revision leaves its TVM runners disabled, so this experiment cannot claim a
complete reproduction of every Table 2 cell. Its PyTorch 2.6 runners also select specialized SDPA, cuDNN, or CUTLASS
paths; they are not equivalent to the full-graph PyTorch 2.13 lane added here.

The paper does not publish per-shape latency tables or raw plot data. Its absolute attention results are throughput
plots at sequence length 8192 over varying batch sizes. Representative measurements from the 40GB replay are below,
reported as `Neptune / torch.compile` in microseconds. Neptune is the mean of 15 projected GPU ranges; Inductor is the
mean of two independently launched minimum-of-15 measurements.

| Operator | Sequence 2048 | Sequence 32768 |
| --- | ---: | ---: |
| Prefill global | 551.7 / 453.6 | 92,427.5 / 84,806.1 |
| Prefill causal | 375.4 / 334.8 | 61,207.6 / 45,048.8 |
| Prefill GQA | 619.2 / 563.2 | 100,425.6 / 88,315.4 |
| Decode causal | 45.5 / 38.9 | 410.3 / 395.8 |
| Decode GQA | 19.8 / 57.3 | 143.2 / 615.9 |

The absolute trend is coherent with the ratios: prefill remains close but favors current PyTorch, causal decode
converges toward parity, and Neptune's decode-GQA advantage grows with context length.

## Published artifact coverage

All ten operator families and all eight sequence lengths produced Nsight profiles. Sixty-four of 80 tuning jobs
completed their 128-trial search; 16 reached the 30-minute per-setup limit. Timed-out rows still profile the available
manual and partial tuned schedules and are not described as fully tuned.

The speedup column compares the fastest available Neptune schedule with the fastest valid non-Neptune runner in the
published artifact. Values above 1.00x favor Neptune. The artifact uses PyTorch 2.6, so this table characterizes
Neptune's original comparison environment; the modern Inductor comparison above is the more relevant baseline.

| Published operator | Full tunes | Profiles | Neptune vs fastest valid artifact runner | Validity note |
| --- | ---: | ---: | ---: | --- |
| Prefill global | 8/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton rows at 256–1024 |
| Prefill causal | 5/8 | 8/8 | 0.78x | Excludes the mismatching Tri Dao Triton runner |
| Prefill GQA | 6/8 | 8/8 | 0.76x | Excludes the mismatching Tri Dao Triton runner |
| Decode causal | 7/8 | 8/8 | 1.04x | No cross-runner mismatch |
| Decode GQA | 8/8 | 8/8 | 1.21x | No cross-runner mismatch |
| Prefill ALiBi | 6/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Decode ALiBi | 8/8 | 8/8 | Excluded | Flex and CUTLASS disagree with Neptune on all shapes |
| Prefill softcap | 4/8 | 8/8 | 1.04x | Compared with Flex |
| Decode softcap | 8/8 | 8/8 | 4.96x | Compared with Flex |
| Prefill windowed | 4/8 | 8/8 | 0.68x | Compared with CUTLASS |

The harness treats a Neptune manual schedule as its correctness reference. Agreement from the other runners supports
the non-ALiBi rows, but it is not an independent oracle for Neptune. ALiBi is therefore excluded rather than assigning
the disagreement to either side.

SoftCap decode is the remaining performance-reproduction outlier. This replay reports 4.96x over Flex, while the
paper's A100 compiler table reports 1.86x over its best compiler baseline. The modern PyTorch lane does not implement
SoftCap, so this result should not be treated as reproduced until that difference is explained.

## Protocol and limitations

- Neptune ran revision `3aa55c12ac822337e630b809b0d9eabb11eee5d3` in the pinned image
  `evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca`.
  The artifact environment recorded PyTorch 2.6.0 with CUDA 12.4 and Nsight Systems 2025.3.1.
- The current Emmy lane reconstructs global, causal, and GQA attention for prefill and decode through
  `emmy run -c ... --bench`. It uses PyTorch 2.13.0 with CUDA 13.0, deployable O3, full-graph Inductor in
  `max-autotune-no-cudagraphs` mode, one warmup, 15 measured iterations, and strict correctness.
- The old starter lane used untuned Emmy and retained its failures. The current table replaces only that Emmy result;
  it does not rewrite the historical run below.
- The paper comparison uses the arithmetic mean of 15 projected Neptune GPU ranges. The PyTorch/Emmy lane uses the
  mean of two independent minimum-of-15 CUDA-event measurements. Both are GPU-time measurements from separate
  processes and software environments, so their ratios are kernel-level evidence rather than an end-to-end result.
- The softcap, ALiBi, and windowed families have no current PyTorch/Emmy twin in this experiment. Their table only
  reproduces the runners shipped in Neptune's artifact.

## Run and system

- Chunk-loop density lane (prefill and GQA decode): 4/4 rows succeeded on 2026-09-10 at revision `3394fd03c`, same
  host as below
- Split-KV GQA-decode lane: 8/8 setups succeeded on 2026-09-10 at revision `38f7f6d9e`, same host as below
- Status: 5/5 full rows succeeded; the corrected GQA-decode row also succeeded
- Full run: `20260909T081524Z`; corrected GQA-decode run: `20260909T093117Z`
- Full-run Git revision: `326fb0210f65d6d373ea72e2f8b2cfbad8e2359a`; dirty: false
- Corrected-row Git revision: `1b1d6aa0cdc3a0e8d5fd4070c82daaa17c60cd9c`; dirty: false
- Host: `bench-codex-a100-0908-0933-d43f`; Ubuntu 24.04.4 LTS; kernel `6.17.0-1022-gcp`
- CPU: Intel Xeon at 2.20 GHz, x86_64, 12 logical CPUs; memory: 89616363520 bytes
- GPU: NVIDIA A100-SXM4-40GB, 40960 MiB, UUID `GPU-be299b90-0ff5-e1b4-db53-28465b6f874b`
- NVIDIA driver: `580.173.02`; host NVCC: `12.9.41`; host cuBLAS: `12.9.0.13`
- Docker client/server: `29.8.0` / `29.8.0`

The host was supplied for this work and remains running.

## Historical starter run and system

- Status: succeeded
- Result timestamp: 2026-08-16T00:41:38Z; run ID: `20260816T004138Z`
- Experiment row: `compiler_neptune_emmy_pytorch_a100_recovery/a100x1`; row ID: `e246bb6279fd`
- Git revision: `2550211d9c93e522ea4f9eb81e39735f4ab64d07`; dirty: false
- Host: `riftvm`; Ubuntu 24.04.1 LTS; kernel `6.8.0-51-generic`
- CPU: AMD EPYC 7742 64-Core Processor, x86_64, 15 logical CPUs; memory: 221634367488 bytes
- GPU: NVIDIA A100-SXM4-80GB, 81920 MiB, UUID `GPU-b0354a1a-37c2-086d-f6fe-953b6fac5c3e`
- NVIDIA driver: `580.65.06`; host NVCC: `12.9.86`; host cuBLAS: `12.9.1.4`
- Docker client/server: `28.5.1` / `28.5.1`

The source run (`20260815T040818Z`) completed all Neptune work in 71393.66 seconds, then failed because the host lane
started outside the staged repository. The successful 2353.12-second recovery verified the immutable source archive's
SHA-256 (`775fb71d3eac78f0371c1014b9945b29d17f41202347f8115e8703db5a4c14ca`), retained its failed status, and ran
only the missing host lane. The durable `recipe.yaml` contains the corrected working directory for clean future runs.

## Durable files

- Exact A100 40GB comparison: `paper-emmy-a10040.csv` (prefill and GQA-decode Emmy, eager and Inductor columns from
  the 2026-09-10 lane; causal decode's Emmy column re-recorded 2026-09-10 after the bound row, its eager, Inductor
  and Neptune columns still from the 2026-09-09 run)
- Causal-decode bound-row sweep: `results_a10040_decode_causal_bound_row.tar.gz`; SHA-256
  `e92d9b0f49963597c94931cc0ad3edf44f3dbbe37272965177750c264f79d9b5`. Holds the pin-sweep and split-width tables, every
  `--record-greedy` and trace log behind the eight re-recorded goldens, the environment freeze and the revision.
  Logs and tables only — this was a manual sweep, so it carries no experiment record, JSON rows or Nsight profiles,
  and the lane has NOT been rerun for this family.
- 2026-09-10 prefill and GQA-decode lane: `results_a100x1.tar.gz` (historical version in Git)
- Split-KV GQA-decode lane: `emmy_decode_gqa_lane.csv` and `results_a10040_split_kv_decode_gqa.tar.gz` (per-setup JSON
  records, logs and the status table)
- Neptune paper reconstruction: `paper-baselines.csv` and `paper-table.csv`
- Current system records and composite task artifacts: five rows under `2026-09-09_08-15-24/` and the corrected
  GQA-decode row under `2026-09-09_09-31-17/`, both retained in the raw-results archive
- Starter experiment record: `a100x1_e246bb6279fd.experiment.yaml`
- Tuned decode-causal experiment record: `a100x1_lemmy_od-c_9f8816b4a4fb.experiment.yaml`; SHA-256
  `07a4b79cf046bfb16b766bc830974dc42f5cc291c87874c3de7f25d7fb7b81d3`
- Raw-results archive: `results.tar.gz`; SHA-256
  `0871843d4d8eb9232c3260143f7da63322a2978318895a54a570a9f91f6dafc8`
- Archived roots: `2026-08-16_00-41-38/`, `2026-08-24_22-35-24/`, `2026-09-09_08-15-24/`, and
  `2026-09-09_09-31-17/`
- Starter composite task artifact: `a100x1_artifacts.tar.gz`; SHA-256
  `015951d7cccf187c69dd2712bcaf966f3de179b53508942312a0e8e6cc31e4b5`
- Tuned decode-causal composite task artifact: `a100x1_lemmy_od-c_9f8816b4a4fb_artifacts.tar.gz`; SHA-256
  `6c288facacb05cf46b20c4d7be8a6bf56c1495a96ffdbe58c79f41b744412d4b`
- Raw evidence includes 80 `.nsys-rep` profiles, 80 CSV exports, all tune/profile logs, 40 modern PyTorch JSON rows,
  Emmy dumps and logs, all current replay/reference JSON rows, environment freezes, runner hashes, source/recovery
  status files, and all run records/logs.
