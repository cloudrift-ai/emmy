# Paged KV cache on the native Qwen3-0.6B path

## Status: generates on a V100 over a paged cache, with the schedule pinned

Both exports produce a pack whose 28 layers' K and V are page tables, the Rust runtime loads it and
allocates the pages, and it generates:

```
step 0..3: Ok(None)          prefill, consuming "The capital of France is"
step 4: Ok(Some(12095))      first generated token
step 5: Ok(Some(13))
step 6: Ok(Some(576))
```

That needed one thing beyond paging: a pinned schedule. Unpinned, the first token step never
completes — and the cause is not paging.

## Why the unpinned build does not complete

Tracing the step's launches shows both paged kernels completing and the stall landing on the fused
post-attention kernel that follows them:

```
launch native_embed          ok
launch pre0.reshape*         ok
launch native_rope_cache     ok      <- paged cache write
launch native_attention      ok      <- paged cache read
launch post0.add_2 (k_linear_mean_reduce_d72eb5)   never returns
```

That kernel stalls on this GPU with no paging, no pack and no Rust runtime involved. Compiling the
`post` half of one decoder layer at a single row and running it through the Python CUDA backend
reproduces it in about two minutes, and emmy's own watchdog names it:

```
HungKernelError: kernel 'k_linear_mean_reduce_d72eb5 (iter 0)' did not complete within 60000 ms
```

It is not a deadlock. A tuning run on the same card shows variants of the same family completing
after 43 seconds, so the 60-second watchdog was cutting off work that would eventually finish. The
blocker is a schedule that costs tens of seconds per launch on sm_70.

## What did not find a schedule

| attempt | outcome |
| --- | --- |
| `emmy tune --layer 0` | wrong kernel set — that compiles the unsplit layer, whose maximal fusion is `k_sdpa_linear_mean_reduce`; the export compiles the `pre`/`post` split |
| working golden of `pre`/`post` at sm_70, tuned 2.5 h | 2236 rows measured; first of two targets unfinished, so `post` was never reached |
| export reading that tune DB | unchanged — no row for the blocking kernel |
| `EMMY_PLACE=cut` | does not complete |
| `UNROLL`, `VECTORIZE_*`, `LOOPIFY` pins | do not complete, and leave the kernel identity unchanged — these knobs do not reach this schedule |

## The pin, and what a golden does and does not do

The repository's own V100 evidence carries working rows for this model — `_tune/main-fixes-v100/`
records Qwen3-0.6B schedules measured on a Tesla V100 — and they are all scalar tier. Pinning that
spelling makes the post kernel run:

```
EMMY_TILE=          # empty: scalar tier, no tensor-core atom
EMMY_WORK=t128
EMMY_REDUCE=coop
EMMY_STAGE=
```

`TILE=''` is the load-bearing one. Unpinned, the greedy pick reaches for a Volta tensor-core tile
whose schedule does not complete a launch here. With the pins the export drops from 10.3s to 0.8s
and the pack generates.

The same schedule is recorded as a golden — `emmy/recipes/Qwen3-0.6B/golden/v100_sm70.yaml`, ten
realizations over the three sub-blocks the native export compiles, written by
`emmy run --golden ... --record-greedy` with the pins active. **It does not replace the pins.** An
export from the golden alone takes the slow schedule again: the pins collapse the search space,
while a golden ranks candidates inside it. `--strict-evidence` says exactly what is missing:

```
strict evidence: kernel 'k_linear_mean_reduce_83fb98' has no measured evidence for its 030_cut
fork (no measured row spells a kernel-set arm)
```

So the golden covers the schedules but not the kernel-set decision above them. Retiring the pins
means recording those cut arms too; until then the recipe passes both, and the pins are what
actually decide.

## What it costs

Recording the golden measured the scalar tier at a single row:

| sub-block | scalar tier, one row, V100 |
| --- | ---: |
| `pre` (q/k/v projections) | 69 ms |
| `post` (o_proj + residual + MLP) | 249 ms |

Those are milliseconds where microseconds belong. The reason is visible in the bench rows: one of
the two `pre` kernels launches at **grid 1** — a single CTA on an 80-SM card. At one row there is
no M to spread, so the scalar tier serializes what a tensor-core tile would parallelize. That is
the whole of the twenty-minute generation, and it is a property of the tier, not of paging.

## Next

1. **Record the cut-fork rows** so the golden alone pins the build and the env pins can be
   dropped. `--strict-evidence` names each missing one, so this is mechanical, not a search.
2. **Retry unpinned after the Volta work.** `#874` reworks exactly the tensor-core path the greedy
   pick was taking here, so the pin may stop being necessary.
3. **Qualify on a 4080 or 5090** to measure what paging costs against a schedule that is fast as
   well as correct. The scalar tier answers "is it correct"; it cannot answer "what does it cost".
