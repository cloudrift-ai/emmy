# Emmy blockers on full attention, RTX 5090

Goal: record what stops `experiments/golden-bench-2026/compiler_attention_rtx5090` from having an Emmy lane, so the
next attempt starts from evidence instead of repeating the search. The baseline lane of that experiment is complete
and archived; every number below was measured beside it on the same card, at `366caa52d`, CUDA 13.0, driver
580.173.02, torch 2.14.0.

**None of this is a recent regression.** The headline miss reproduces unchanged at `39d9f91ef` — the commit that
recorded this card's causal attention rows — and at every commit since.

## The one that matters: a traced SDPA cannot reach its own recorded row

Causal MHA at batch 1, 8 heads, 512 keys, head dimension 128 is *in* this card's repository hardware golden as
`attention.hd128.causal.fused`, recorded at **14.91 us**. Replaying that realization reproduces it:

```bash
emmy run --golden emmy/compiler/pipeline/search/goldens/rtx5090_sm120.yaml \
  --realization attention.hd128.causal.fused --bench --bench-backends eager
# Emmy 18.6 us greedy, 16.8 us on the recorded [fm] row; eager 14 us
```

Tracing the same program from source reaches a different kernel entirely:

```bash
emmy run -c "torch.manual_seed(0);q=torch.randn(1,8,512,128,dtype=torch.float16);\
k=torch.randn(1,8,512,128,dtype=torch.float16);v=torch.randn(1,8,512,128,dtype=torch.float16);\
F.scaled_dot_product_attention(q,k,v,is_causal=True,enable_gqa=False)" --bench --bench-backends eager
# Emmy 20015 us; eager 14 us
```

The two Graph IR programs are the same: three f16 inputs of that shape, one `torch.sdpa` with `is_causal: true`, one
constant, target `origins: [scaled_dot_product_attention]`. They differ only in input names. Yet the golden's program
lowers to `k_sdpa_deb5ea` with schedule sites spelled `@map.1/twist` and `@map.1/twist.1/inner`, while the traced one
lowers to `k_sdpa_69f335` with sites under `@inner`, `@inner.1/map.1/inner` and
`@inner.1/map.2/map.1/twist.1/inner`.

Because the site routes differ, the recorded row names no site the traced program has, so it is not evidence for it,
so the greedy pick falls through to the prior — and the prior's pick is 1340x slower than the row that exists for that
exact shape.

**This is circular, and that is the blocker.** The recorded row can only be found once the compile has already taken
the structure that produces those routes, and the structure is chosen by the greedy using the prior. `--record-greedy`
writes routing rows precisely to break this, but it can only record the structure the greedy already took.

Pinning does not break it either. `PLACE=fuse` leaves the routes unchanged. Pinning the recorded row's own knobs by
their recorded spelling fails to match:

```
unreproducible pin: TILE@map.1/twist=... realized TILE@inner.1/map.2/map.1/twist.1/inner=mma_m16n8k16_f16_f32/f1x1
```

So there is no schedule to hand-pin and record. The `tune-kernels` method that #756 used — "measured by hand pin on
the card rather than by the tuner" — does not apply to a program the pins cannot address.

Worth checking next: what makes the two programs lower differently. They are identical at Graph IR, so the divergence
is downstream — the `--golden --realization` path replays an embedded program while `-c` traces through
`torch.export`. If that difference is incidental rather than semantic, closing it would make every recorded attention
row reachable from source and unblock the lane on its own.

## `acc4`: most of the worker inventory does not compile

```
AssertionError: materialize: kernel 'k_sdpa_...' reads names it never binds: ['acc4']
```

`materialize` calls this "always a compiler bug" in its own docstring, and it fires across the mma schedule space for
attention. On causal MHA prefill at batch 1, 32 heads, 1024 keys, sweeping `WORK` alone:

| `WORK` | outcome |
| --- | --- |
| `w2x2`, `w1x8`, `w8x2`, `w1x4` | `acc4` assertion |
| `w2x4`, `w4x4` | nvcc compile failure |
| `w4x2` | compiles — 284782 us, against 94.3 us for the best baseline |

Four of seven inventories do not compile. It is not GQA-specific, though GQA hits it on the greedy pick too. Smallest
repro, about a minute:

```bash
emmy compile -c "torch.manual_seed(0);q=torch.randn(1,8,128,64,dtype=torch.float16);\
k=torch.randn(1,2,128,64,dtype=torch.float16);v=torch.randn(1,2,128,64,dtype=torch.float16);\
F.scaled_dot_product_attention(q,k,v,is_causal=False,enable_gqa=True)" --ir cuda
```

Two query heads per KV head is not enough — `q_heads=4, kv_heads=1` compiles, `q_heads=8, kv_heads=2` does not, so a
degenerate KV-head axis hides it.

`acc4` is the recomputed QK score in the second pass of the fold tree — the value the flash epilogue reads back after
the softmax twist. Some schedules emit a reference to it that nothing binds.

## The greedy pick hangs

On causal MHA prefill at batch 1, 32 heads, 1024 keys the greedy pick is a **scalar** thread tile — `WORK=t64x8`,
`TILE@inner=f4x10`, grid 416, block 512 — on an f16 attention that has tensor cores available. It does not finish one
launch:

```
HungKernelError: kernel 'k_sdpa_d8a514 (iter 0)' did not complete within 60000 ms
```

Every A/B run on that shape pays the 60 s watchdog before any pinned row benches.

## Decode: the schedule space cannot express flash-decode

For decode the output is (heads x head_dim) cells and the greedy places **one CTA per output cell**, each reducing
over the whole key length:

```
k_sdpa_07cc15  5471.5 us  grid 4096  block 128  WORK=t128  REDUCE@map.1/twist=coop
```

For 32 heads and head dimension 128 that is 4096 CTAs where 32 would do, and each of a head's 128 output cells
recomputes that head's entire score vector — 128x redundant work. Against 40.8 us for the best baseline at batch 1 and
4096 keys, Emmy is 134x slower.

No knob moves it. The only families this kernel offers are `WORK`, `REDUCE`, `LOOPIFY` and `RASTER`; `TILE` is not
applicable, so the output tile cannot be widened to give a whole head to one CTA. Everything reachable lands within
4 percent of the same wall:

| pin | latency |
| --- | --- |
| greedy (`t128`, `coop`) | 5471.5 us |
| `WORK=t128,REDUCE=coop/r2` | 5435.9 us |
| `WORK=t256,REDUCE=coop` | 5461.6 us |
| `WORK=t64,REDUCE=coop/r4` | 5651.5 us |
| `WORK=t128,REDUCE=g8k/coop` | 5320.4 + 3.3 us (partial + combine) |

The cross-CTA split is the interesting one and it barely helps, because splitting keys does not remove the redundancy
across the dim axis — it only adds CTAs that each still recompute the softmax.

`LOOPIFY` is a readability knob (byte-identical CUDA) and is not a lever here.

## Reading order for whoever picks this up

1. The routing mismatch is the only blocker that gates the others. Fix it and the recorded rows become reachable, the
   greedy stops falling through to the prior, and the hang and the decode geometry may both stop mattering because
   the good structure is the one that gets picked.
2. `acc4` is worth fixing regardless — it is an assertion the compiler itself calls a bug, and it removes most of the
   mma inventory from the search.
3. Decode needs a schedule the space cannot currently spell. That is a design question, not a tuning one.

## What is already done

`experiments/golden-bench-2026/compiler_attention_rtx5090` has a complete, archived baseline lane on this card — 59
setups against SDPA, Inductor, FlexAttention, FlashAttention-2, cuDNN and TileLang. When the Emmy lane can run, it has
something to be compared against, and `golden/README.md` records the same blockers where the goldens would go.
