# Qwen3.8-27B at EXL3 5.0 bpw on one V100 SXM3 32GB

Compiler-qualified 2026-09-13 against repository revision `9095cb297` on `riftuser@66.172.10.131`.

This recipe has a golden and **no serving lane**. Nothing serves this checkpoint on this card: stock vLLM has
neither sm_70 support nor an EXL3 quantization method, and the 1Cat-vLLM Volta fork does not read EXL3 either. The
golden below is compiler evidence — measured CUDA kernels for the model's Gated DeltaNet chunk path — and it does
not imply that the model can be served here. It cannot.

## What was measured

| Item | Value |
| --- | --- |
| Model | `turboderp/Qwen3.8-27B-exl3@a35e75a73baee51da709329d19294245cbeeb5d8` (5.0 bpw body, 6-bit head, 4-bit MTP) |
| GPU | 1 x NVIDIA Tesla V100-SXM3-32GB, compute capability 7.0, CUDA 12.9 |
| Architecture | 64 decoder layers: 48 `linear_attention` (Gated DeltaNet), 16 `full_attention`, plus 1 MTP layer |
| Inventory | 124 distinct kernels traced from two archetypes (layer 0 and layer 3), covering all 64 layers |
| Golden | 15 measured Gated DeltaNet chunk kernels, each recorded with `emmy run --record-greedy` |
| Reference | eager PyTorch (torch 2.13.0+cu126), same inputs, 5 warm-up / 20 measured iterations |

**The golden holds 15 of the 124 traced kernels, and the other 109 were never candidates.** A repository golden
requires every realization VERIFIED — knobs plus paired positive timings — and the 109 are traced inventory
carrying only a `FAST_MATH` pin, with nothing measured to promote. The 15 are the whole of the loss: they are the
only kernels in the inventory that lost to eager by more than a microsecond.

## The result

Every one of the 15 is a chunk of the Gated DeltaNet delta rule. Export unrolls that chunk loop, so chunk *k*
carries O(*k*) work and the kernels form a gradient rather than a family of equals.

| kernel | eager us | emmy us | vs eager |
| --- | ---: | ---: | ---: |
| `k_slice_unsqueeze_reduce_4b94bb` | 3840 | **186** | **20.60x** |
| `k_slice_unsqueeze_reduce_d90fea` | 3850 | **312** | 12.34x |
| `k_slice_unsqueeze_reduce_ac7e88` | 3861 | **392** | 9.86x |
| `k_slice_unsqueeze_reduce_006570` | 3871 | **532** | 7.28x |
| `k_slice_unsqueeze_reduce_cc2c8e` | 3927 | 644 | 6.10x |
| `k_slice_unsqueeze_reduce_d4b52b` | 3982 | 827 | 4.81x |
| `k_slice_unsqueeze_reduce_287e6f` | 4036 | 994 | 4.06x |
| `k_slice_unsqueeze_reduce_d126a6` | 4086 | 1641 | 2.49x |
| `k_slice_unsqueeze_reduce_99707e` | 4205 | 2050 | 2.05x |
| `k_slice_unsqueeze_reduce_7e169c` | 4269 | 2560 | 1.67x |
| `k_slice_unsqueeze_reduce_124b26` | 4346 | 3109 | 1.40x |
| `k_slice_unsqueeze_reduce_0d1c72` | 4425 | 3778 | 1.17x |
| `k_slice_unsqueeze_reduce_6169c0` | 4557 | 4577 | 1.00x |
| `k_slice_unsqueeze_reduce_955202` | 4632 | 5466 | 0.85x |
| `k_slice_unsqueeze_reduce_1faf06` | 3263 | 5979 | 0.55x |

Twelve beat eager, one is at parity, two do not.

The two worst before any of this were not merely slow. `k_slice_unsqueeze_reduce_cc2c8e` measured **1,339,612 us**
and `k_slice_unsqueeze_reduce_006570` **648,737 us** against eager's ~3,900 — and ten of the fifteen could not be
measured at all, exceeding the 60-second kernel timeout. The whole inventory gave away at least 2.05 seconds per
call, a floor rather than a total, because those ten never returned a number.

## What fixed it

The greedy elected a single fused kernel at **grid 1** — one CTA for the entire reduce. A `PLACE=cut` placement
splits it into pieces that fill the grid. Nothing about the cut was missing: it was offered, it realized, and it
was fast. It was simply never picked, because of a pricing defect diagnosed separately from this recipe, and that
defect is still open.

**The golden is what closes it for a deploy.** A recorded row outranks the prior: the greedy's measured-evidence
pick reads these rows before it consults the model, and a strict-evidence compile takes the kernel set the row
names. So the fifteen rows below make a deploy take the cut whether or not the pricing defect is ever fixed — it
costs an unrecorded kernel, not a recorded one. That is why this recipe carries a golden rather than waiting for a
compiler change.

## Recording the route, and why the spelling is recorded per row

Every row here is recorded under the explicit scoped pin `PLACE@map.1/reduce=cut`, and the golden stores that key
rather than a bare `PLACE`. That is load-bearing. The three placements below are three different decisions on one
kernel, not three spellings of one:

| pin | emmy us | vs eager | kernels in the set |
| --- | ---: | ---: | --- |
| bare `PLACE=cut` | 536 | 7.23x | **5** — incl. `k_add_20__place_…` at t32x8/f2x4 and a grid-24 piece |
| `PLACE@map.1/reduce=cut` | **532** | **7.28x** | **4** — grid 6144 t32/coop, 1536 f4, 7296 t128/coop, grid 1 |
| `PLACE@map.2/map=cut` | 648780 | 0.01x | the fused arm the greedy elects unaided |

The first two agree on time to within this box's clock ramp and disagree on the kernel SET. A bare `PLACE` is also
ambiguous on replay and will not strict-decode. The route was verified on `006570` and applied uniformly; each row
carries its own measurement, so a row where the route was wrong shows up in its own number — none did, and none
came back near the third line above.

## The three that do not beat eager, as a bounded conclusion

This is not an open to-do. The schedule space for these kernels is exhausted: the cut is taken, the pieces fill the
grid, and the remaining cost is the shape of the traced program rather than how it is scheduled. Eager sits flat at
3,263–4,632 us across all fifteen while Emmy climbs monotonically with chunk index, because `torch.export` unrolled
the chunk loop and chunk *k* carries O(*k*) work. The three laggards are the tail chunks, and no placement or
schedule pin reaches them — the fix would be a different traced program, upstream of the compiler.

The same gradient was measured independently on the GPTQ Int4 quantization of this architecture, on the same box,
by a separate agent: eager flat, Emmy climbing with chunk index, same conclusion. Two quantizations and two
independent measurements make this a finding about Emmy's Gated DeltaNet path rather than about either checkpoint.

## Correctness

All 15 were accuracy-checked against eager, not merely timed. `emmy run` requests an accuracy comparison unless the
run is an ncu child or `--quantize` was passed — neither applies — and on failure it exits before the backend table
renders. Every one of the 15 printed a table, so every one passed. The `reference_backend: same-input-greedy` each
row carries is a *timing* reference and does not by itself establish this.

No recorded row uses a tensor-core atom: `mma_m8n8k4` appears nowhere in the file. Every row here is a reduce or
pointwise piece rather than a contraction that would take Volta's atom, so this golden has no exposure to the
tile-shape miscompilation found on the `mma_m8n8k4` path.

## Not established

- **Serving.** Nothing reads EXL3 on sm_70. No engine, no image, no throughput number.
- **The full-attention path.** Layer 3's 7 kernels are traced and in the inventory but none is in the golden; they
  were not among the losses.
- **The MTP path.** Traced by neither archetype here.
- **`-O3` deployment latency.** These numbers are the default build; no `-O3` replay was recorded.
