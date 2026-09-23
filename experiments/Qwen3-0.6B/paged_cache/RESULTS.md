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

## The pin

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
whose schedule costs tens of seconds per launch here. The recipe exports these, so the behavioural
test runs a fully defined kernel rather than whatever the search happens to pick.

| | unpinned | pinned scalar tier |
| --- | --- | --- |
| export | 10.3 s | 0.8 s |
| first token step | never completes | completes |
| 8-step generation | — | 3 tokens after a 5-token prompt |

It is correct, not fast: those seven steps took roughly twenty minutes of wall clock. The scalar
tier is the right choice for a behavioural test and the wrong one for a perf claim.

## Next

1. **Retry unpinned after the Volta work.** `feature/volta-trans-b-crosswise` and #872 rework
   exactly the tensor-core path the greedy pick was taking here, so the pin may stop being
   necessary — and if it is still necessary, that is worth knowing.
2. **Qualify on a 4080 or 5090** to measure what paging costs against schedules that are fast as
   well as correct. 139 golden records already exist for this model there
   (`../native_baseline/golden/rtx4080_sm89.yaml`).
