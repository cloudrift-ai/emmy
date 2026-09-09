# RTX 5090 attention goldens

This directory receives the recorded goldens for the RTX 5090 attention comparison. The measurement recipe does not
tune and fails when any requested golden is absent.

For each operator, batch, and sequence length that `operators.sh` offers, print the exact source with:

```bash
experiments/golden-bench-2026/compiler_attention_rtx5090/operators.sh OPERATOR BATCH SEQUENCE_LENGTH
```

Trace that source, tune it on the RTX 5090, validate it, and record it as
`OPERATOR-bBATCH-sSEQUENCE_LENGTH.golden.yaml`. Commit every offered setup before running the Emmy lane. Keep the
tuning records outside the measurement recipe; the committed golden is the schedule input that the recipe replays.

## What blocks recording them

Nothing here has been recorded yet. The blocker written down here earlier — a freshly traced
`F.scaled_dot_product_attention` reaching a kernel no recorded row could name, and running at **20015 us** against
**14 us** eager — **no longer reproduces**. It was measured at `366caa52d`; on `origin/main` at `bdd26c539` the same
trace deploys the card's recorded schedule and measures **18.6 us**, and `emmy compile --strict-evidence` passes on
it. Something between those commits closed it. Do not plan around the old account.

Measured on an RTX 5090 at `bdd26c539` (CUDA 13.0, driver 580.173.02, torch 2.14.0, deployable `-O3`):

| setup | eager | Emmy | note |
| --- | ---: | ---: | --- |
| causal MHA, 1 x 8 x 512 x 128 | 14 us | 18.6 us | deploys the recorded row; `--strict-evidence` passes |
| causal MHA prefill, 1 x 32 x 1024 x 128 | 86 us | 143299 us | no measured row for its `030_cut` fork |
| the same, under a hand pin of the 512 schedule | 95 us | 225 us | an untuned schedule borrowed from another shape |
| decode, 1 x 32 x 1 x 128 against 4096 keys | 59 us | 15921 us | no measured row, and no tile choice to record |

So what is left is recording, not reachability. Three things make that work rather than a formality.

**The longer sequences have no evidence.** `--strict-evidence` names the kernel and its `030_cut` fork as having no
measured row, which is exactly what a recorded golden supplies. The routes a trace produces are the routes the
recorded rows use (`TILE@map.1/twist`, `TILE@map.1/twist.1/inner`, `STAGE@…`, `WORK`), so a hand pin reaches them —
`EMMY_KNOBS="TILE@map.1/twist=…,WORK=…"` — and `emmy run --golden PATH --bench --record-greedy` writes what it
measures back.

**Pinning is slow.** Causal prefill at 32 heads and 1024 keys compiles to CUDA in 34 s unpinned; the same program
under a hand pin resolved in 1629 s. Recording every offered setup this way is a long campaign, and the setups run to
32768 keys.

**Decode has no schedule worth recording yet.** A one-query attention against a long key length fuses into a twisted
carrier, but that kernel offers only `WORK`, `REDUCE`, `LOOPIFY` and `RASTER` — no `TILE`, so the output tile cannot
be widened to give a whole head to one CTA. An earlier sweep over the families it does offer, at `366caa52d`, landed
every reachable pin within four percent of one wall, the cross-CTA split included: it adds CTAs that each still
recompute the softmax. Widening the output tile is a design question to settle before decode goldens are worth
recording.

## One thing worth knowing before tuning

The same attention program compiles to two different kernel identities depending on what its input tensors are
called: `k_sdpa_deb5ea` when the query, key and value are anonymous expressions, `k_sdpa_69f335` when they are named
`q`, `k` and `v`. Both measure the same (18.5 and 18.6 us) and both find the recorded evidence, so it costs nothing
here — but two identical kernels compile twice and do not share a cache entry, and a golden's stored `identity` is
written under one spelling. Record from the source the recipe replays.
