# Paged KV cache on the native Qwen3-0.6B path

## Status: paging works; generation is blocked by a compiler hang on sm_70

Both exports produce a pack whose 28 layers' K and V are page tables, and the Rust runtime loads it
and allocates the pages. Generation does not complete on the V100 — but the cause is not paging.

Tracing the step's launches shows both paged kernels completing and the hang landing on the fused
post-attention kernel that follows them:

```
launch native_embed          ok
launch pre0.reshape*         ok
launch native_rope_cache     ok      <- paged cache write
launch native_attention      ok      <- paged cache read
launch post0.add_2 (k_linear_mean_reduce_d72eb5)   hangs
```

That kernel hangs on this GPU with no paging, no pack and no Rust runtime involved. Compiling the
`post` half of one Qwen3-0.6B decoder layer at a single row and running it through the Python CUDA
backend reproduces it in about two minutes, and emmy's own watchdog names it:

```
HungKernelError: kernel 'k_linear_mean_reduce_d72eb5 (iter 0)' did not complete within 60000 ms
```

| check | V100 (sm_70) |
| --- | --- |
| paged read/write, compiler-rendered kernels (`tests/paged.rs`) | passes |
| paged cache write + read, hand-written native kernels | both launches complete |
| `post` sub-block alone, no paging anywhere | **hangs** |
| `tests/compiler/e2e/` (attention coverage, WS deadlock) | 14 passed |
| full token step | hangs, at the post kernel |

`EMMY_PLACE=cut` changes the fusion and does not help. The schedule is a greedy unmeasured pick on a
GPU this model was never tuned for — its recorded results are from an RTX 4080
(`../native_runtime/`, `../native_baseline/`).

## It is not a deadlock

A tuning run on the same card shows variants of the same kernel family **completing after 43
seconds**:

```
kernel 'k_sdpa_linear_mean_reduce_bc3bdd__place_06a90023ca (iter 0)' completed after 43.58s of waiting
[tune] backend.benchmark failed (... exceeded 2.0s of GPU time — variant marked bench_fail)
```

So the 60-second watchdog was cutting off work that would eventually finish. The blocker is a
schedule that costs tens of seconds per launch on sm_70, not a hang, which is why a fully measured
schedule — a golden row, pinned — is the fix rather than a code change.

## The tuning attempt, and why it did not produce one

`emmy tune Qwen/Qwen3-0.6B --layer 0 --seq-len 1 --bench` ran for about an hour and wrote no
golden: 105 bench_fail markers against ~40 candidates, 26 launches completing only after tens of
seconds.

It also tuned the wrong kernel set. `--layer 0` compiles the **unsplit** decoder layer, whose
maximal fusion is `k_sdpa_linear_mean_reduce`; the native export compiles the `pre` / `post` split
wrappers instead, and the kernel that blocks generation is `post`'s `k_linear_mean_reduce`. The two
share a family name and nothing else.

## Next

1. Tune what the native export actually compiles — the `post` sub-block at one row — and pin the
   result as a golden the export reads via `--golden`. That is another multi-hour run on the card.
2. Or run the same export on a 4080 or 5090, where the path has recorded results, to qualify paging
   end to end on a GPU whose schedules are known good.
