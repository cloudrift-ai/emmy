# Ministral-3-14B-Instruct-2512 on one RTX 5090

This entry covers the Emmy compiler inventory only. No engine was deployed and no serving benchmark was run, so the
recipe stays an `onboarding` / `untested` shell and makes no serving claim. The golden is committed because it is
complete on its own: every decoder layer maps to a measured kernel set.

## Compiler inventory, 2026-10-03

| Item | Value |
| --- | --- |
| Model | `mistralai/Ministral-3-14B-Instruct-2512@29439f81c2be264d8d393273f99e7db9c0961120` |
| GPU | 1 x NVIDIA GeForce RTX 5090, compute capability 12.0, driver 580.173.02 |
| Toolchain | CUDA 13.0 (nvcc 13.0), torch 2.13.0+cu130, transformers 5.14.1, deployable `-O3` |
| Golden | `recipes/Ministral-3-14B-Instruct-2512/golden/rtx5090_sm120.json` |
| Contents | 8 traced programs, 48 kernels, 12 routing rows, 72 measured schedule rows, no unmeasured proposals |

The checkpoint stores every decoder linear as FP8 (e4m3) with one scale per tensor, and one calibrated activation
scale per linear input: static W8A8. Embeddings, the output head and the vision tower are not quantized. The decoder
is 40 structurally identical layers (hidden 5120, MLP 16384, 32 heads of 128, 8 key/value heads). The FP8 shards are
14.65 GiB, which is why FP8 is the form that leaves room for a KV cache on a 32 GB card.

### Coverage

The golden is in the serving-twin form: each layer is split into the half before attention (norm, activation
quantize, the q/k/v projections) and the half after it (output projection, residual, norm, the MLP). Attention itself
stays with the engine in that form. All 40 layers share one profile, so layer 0 is the representative of every layer
index. Each half is recorded at widths 1, 32 and 4096 and as the any-width program, in the standard and the fast-math
lane: 16 realizations over 8 target kernels. The trace used decode bucket 32, the width-1 tier on, 4128 batched
tokens and one fast-math warm shape.

The embedding lookup and the final norm with the output head are not serving twins and are not in the golden. Both
were measured as separate programs: the embedding at 2.7 us against 4.1 us for `torch.compile`, and the head at
3.37 ms against 3.38 ms once the norm is cut from the 131072-wide matrix multiply.

### Results

Whole-program latency of every kernel the route selects, 5 warm-ups and 100 iterations, Emmy and both references
timed on the same random inputs in one run. Eager and `torch.compile` replay the same traced program, including its
FP8 encode and decode steps; they are not the Transformers FP8 kernel. The any-width programs ran at 512 tokens.

| Target | Lane | Emmy | `torch.compile` | Eager | Emmy vs `torch.compile` |
| --- | --- | ---: | ---: | ---: | ---: |
| pre, any-width | fast math | 172.3 us | 213.2 us | 576.9 us | 1.24x |
| pre, any-width | standard | 174.7 us | 214.1 us | 581.9 us | 1.23x |
| pre, 1 | fast math | 18.4 us | 20.5 us | 368.7 us | 1.11x |
| pre, 1 | standard | 19.9 us | 20.5 us | 364.8 us | 1.03x |
| pre, 32 | fast math | 47.1 us | 57.4 us | 378.2 us | 1.22x |
| pre, 32 | standard | 47.1 us | 57.3 us | 378.2 us | 1.22x |
| pre, 4096 | fast math | 819.4 us | 1308.3 us | 2680.0 us | 1.60x |
| pre, 4096 | standard | 829.8 us | 1313.9 us | 2681.2 us | 1.58x |
| post, any-width | fast math | 1061.7 us | 1832.0 us | 5728.3 us | 1.73x |
| post, any-width | standard | 1078.9 us | 1872.3 us | 5736.0 us | 1.74x |
| post, 1 | fast math | 176.7 us | 171.0 us | 4508.0 us | 0.97x |
| post, 1 | standard | 180.3 us | 170.5 us | 4512.1 us | 0.95x |
| post, 32 | fast math | 355.8 us | 759.0 us | 4535.7 us | 2.13x |
| post, 32 | standard | 359.6 us | 759.4 us | 4536.3 us | 2.11x |
| post, 4096 | fast math | 8151.8 us | 11116.1 us | 17711.7 us | 1.36x |
| post, 4096 | standard | 8171.5 us | 11079.5 us | 17787.9 us | 1.36x |

The width-1 post half is the one target not ahead. Its five kernels sum to 148 us and the captured program takes
177 us; `torch.compile` moved between 170 and 183 us across runs. Its largest kernel, the gate and up projections,
reads 168 MB of FP8 weights in 104 us whatever the schedule, so it is bound by memory traffic.

### How the kernel sets are built

Every row needs its cuts pinned. With no measurement in scope the compiler keeps each half as one fused kernel and
runs the projections as scalar loops; on the post half that exceeded the bench's 60 s limit even at 32 tokens.

- **Pre half:** the activation codes are cut out as their own kernel (norm and quantize), and the k/v pair is cut from
  q, so each projection has its own output grid and reads both multiplicands as FP8 bytes.
- **Post half:** the residual stream after the output projection is cut first, because fusion otherwise computes that
  projection three times; then the activation codes in front of the output projection, of the gate and up pair, and
  of the down projection.

With those cuts the projections run on the native FP8 tensor-core instruction (`mma_m16n8k32_e4m3_f32`) at widths 32
and up, and on cooperative scalar reductions at width 1. That instruction is offered to an unpinned compile only in
the fast-math lane. The standard-lane rows name the same tiles explicitly: on this card they match the exact scalar
path almost element for element (below), and without them the standard lane runs the same route ten to forty times
slower (3.8 ms against 0.36 ms on the 32-wide post half, 7.2 ms against 0.17 ms on the any-width pre half).

### Accuracy

All 16 rows pass the bench's default accuracy check against eager. None passes the strict elementwise check
(`rtol = atol = 1e-3`), for a reason that is not specific to a schedule:

| Target (fast math) | Elements outside the strict tolerance | Max abs | Mean abs |
| --- | ---: | ---: | ---: |
| pre, any-width | 261 of 2,097,152 | 0.0078 | 0.00036 |
| pre, 1 | 1 of 4,096 | 0.0039 | 0.00035 |
| pre, 32 | 11 of 131,072 | 0.0078 | 0.00035 |
| pre, 4096 | 2,638 of 16,777,216 | 0.0117 | 0.00036 |
| post, any-width | 2,345,959 of 2,621,440 | 4.63 | 0.53 |
| post, 1 | 4,583 of 5,120 | 2.66 | 0.50 |
| post, 32 | 147,821 of 163,840 | 3.88 | 0.56 |
| post, 4096 | 18,808,578 of 20,971,520 | 5.06 | 0.54 |

Eager multiplies each FP8 code by its scale and rounds the product to FP16 before the matrix multiply. Emmy multiplies
the raw codes and applies the two scales once, after the sum. On the pre half that is a mean difference of about
0.00036 with a handful of larger elements. The post half has two more quantizers downstream, and the bench draws
random FP8 weight bits and random scales, so that small difference changes codes and grows: its worst elements read
88.9 against 85.1 and -132.0 against -135.9.

The control for this reading is the exact scalar route in the standard lane, which uses no tensor-core tile. It fails
the strict check the same way: 264 of 2,097,152 elements on the any-width pre half, with the same worst element and
the same two values as the tensor-core row, and 148,027 of 163,840 on the 32-wide post half (mean abs 0.56). So the
difference is between Emmy's lowering of this algebra and the eager replay, not between tiers.

With the real checkpoint the question goes away. The serving runner was booted for layer 0 with this golden as its
only evidence, fed real embedding rows, and compared with the declared W8A8 math computed in float64 from the shard
tensors. At widths 1, 32 and 40 every element of q, k, v and of the post half's output is inside `rtol = atol = 1e-3`:

| Output | Relative L2 error | Max abs | Elements outside 1e-3 |
| --- | ---: | ---: | ---: |
| q, k, v (pre half) | 2e-5 to 4e-5 | 0.0039 | 0 |
| layer output (post half) | 2.4e-4 to 5.9e-4 | 0.0010 | 0 |

The post half's attention input in that check was synthetic (normal, standard deviation 0.1); everything else was the
checkpoint's own.

### Gaps and findings

- **No evidence, no usable kernel.** The unpinned pick is a scalar loop that hangs at every width. This is a prior
  finding: the golden's rows are what fix it.
- **A small producer kernel gets an empty schedule.** In three pre-half programs the pointwise quantize kernel left
  after the norm statistics are cut ran at 0.5 to 3.8 ms until its work shape was pinned (2 to 20 us after). The rows
  carry that pin.
- **The whole-layer form is not tractable on this host.** One fused layer with attention has 86 cut seams; its
  unpinned pick hangs and a pinned compile grew past 40 GB of host memory twice.
- **Serving compiles these programs, but is not qualified.** The serving runner now keeps a static-FP8 trunk coded
  instead of decoding it to about 27 GiB of FP16 values, and a one-layer boot under strict evidence compiled exactly
  this golden's 48 kernels. Nothing beyond that layer was run.

### Not done

Engine baseline, Emmy serving configuration and image, serving benchmark, context and capability probes, and HF
parity are all outstanding. Multimodal input was not exercised.

### Reproduce

```bash
cp recipes/Ministral-3-14B-Instruct-2512/golden/rtx5090_sm120.json /tmp/ministral-5090.json
EMMY_TUNE_DB=/tmp/ministral-replay.db EMMY_NVCC_FLAGS= emmy run --golden /tmp/ministral-5090.json --bench \
  --strict-evidence --bench-backends eager,tcompile,emmy
```
