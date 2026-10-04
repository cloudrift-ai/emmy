# GDN serving: follow-ups that Qwen3.8 NVFP4 serving does not need

Status: open, written 2026-10-02 during the work on PR #1023 (GDN serving). This file collects what the GDN serving
work found on the way and did not fix, because serving Qwen3.8 NVFP4 does not depend on it. The serving design itself
is in `emmy/serving/ARCHITECTURE.md`.

GDN is the gated DeltaNet layer of Qwen3.5 / Qwen3.8 (`layer_types` value `linear_attention`). A GDN layer carries
per-request recurrent state: a matrix state `S` and a convolution history `H`. A GDN program is the `gdn<W>` program
that serves one such layer at a static width `W`, the number of tokens one call takes.

A placement cut, spelled `PLACE@<route>=cut` in `EMMY_KNOBS`, splits a fused kernel into pieces, each its own kernel.

## Compiler and tooling bugs

### `python -m tests.serving.regen` fails while the serving tests' golden holds GDN programs

`python -m tests.serving.regen` builds every runner configuration of `tests/serving/helpers.py`. It then writes one
schedule row per kernel, plus a routing row per kernel-set decision, through `complete`
(`tests/compiler/realization/helpers.py`). On the width-1 GDN program, `complete` compiles each kernel by itself, and
one of them fails:

    ValueError: Buffer name 'conv1d_acc2__steps0__acc0' already has a producer

The error comes from `Graph.splice`. The kernel is `k_conv1d_acc2_steps0_pointwise`, the kernel that carries the
convolution history. It compiles without error inside its whole program; only a compile of that kernel by itself
fails.

Consequence: we recorded the GDN schedule rows and routing rows of `tests/serving/goldens/serving.golden.json` by
hand, from whole-program compiles and from single-kernel compiles under `EMMY_KNOBS` pins. A change that alters the
structural identity of a GDN kernel needs the same manual work until someone fixes this bug.

A full regeneration has a second cost, independent of this bug: it builds every configuration with no golden in
scope and rewrites every row from what that compile picks. At the time of writing the team treats the schedule prior
as broken, and trusts only hand-pinned golden rows and `EMMY_KNOBS` pins. So review such a rewrite row by row before
committing it.

### `kernel_tile` can return a parent-level tile for a cut remainder

`complete` reads each compiled kernel's identity through `kernel_tile(node.op)`. For the remainder piece of a cut
BF16 matmul kernel (`k_matmul_reduce_6c937b` of the width-16 GDN program), the kernel's stamp matched no tile on
the op's source chain, and `kernel_tile` fell back to the first tile, the parent's. A row recorded that way carries
the parent's identity; the live compile's schedule fork then finds no row for the remainder and strict evidence
fails. The GDN rows of the serving tests' golden were corrected by hand (one kernel and one row added). Any
recording that goes through `complete` can hit the same fallback.

### Two cuts together crash the lowering of a width-1 GDN kernel

Lowering kernel `k_linear_mean_conv1d_reduce_290483` of the width-1 GDN program alone, with both of these cuts
pinned:

    EMMY_KNOBS='PLACE@map.2/map.1/inner.1/map.1/inner.1/map.1/inner=cut,PLACE@map.1/map.3/map=cut'

raises `ValueError: Assign 'v2': arg 'in6' not defined`. The second cut alone does not fail. A greedy search over
cuts found the pair; the recorded rows avoid it. Nobody re-ran this pair in isolation after the search.

### One cut of the float16 width-1 GDN kernel yields a piece that nvcc rejects

In float16 the whole width-1 GDN program fuses into one kernel, `k_conv1d_linear_mean_reduce_79c704`. In float32 the
same program has three kernels, so the float32 cuts do not apply to it. Cutting that kernel at this seam:

    PLACE@map.3/map.1/inner.1/map.1/inner.1/map.1/inner.1/map.2/reduce.1/inner.1/map.10/map.1/inner.1/map=cut

mints a piece whose CUDA source does not compile: nvcc reports `identifier "in0__s0" is undefined` on the line
`v1 = in0__s0 * in2;`. The lowering passes accept the piece; the error appears only when the program builds its
kernels. The recorded rows cut the kernel at 44 other seams and leave this one fused.

### A text-only Qwen3.5 NVFP4 checkpoint keeps its quantized weights unspelled in serving

`EmmyGenRunner.from_model` addresses a coded trunk's constants by checkpoint key. It derives each key from the
parameter's identity through the model's reversed key renamer, which yields `model.language_model.layers.N.*` for
Qwen3.5. The Qwen3.8 NVFP4 checkpoint uses that key layout, and there the runner's GDN programs have the kernel
identities of the serving twins. A checkpoint saved by `Qwen3_5ForCausalLM.save_pretrained` uses `model.layers.N.*`
instead. For it the runner looks up a key that the checkpoint does not hold, the NVFP4 speller finds nothing, and
the program keeps a BF16 weight. Serving-twin capture matches constants by suffix and spells them, so the two sides
lower to different kernels and golden rows recorded on the twins do not apply. We saw this on a synthetic checkpoint
with one NVFP4 linear in a GDN layer; attention layers were not compared.

### GDN programs and packs

A pack is the on-disk store of compiled plans that a serving image boots from.

- The pack-hit path of `EmmyGenRunner.from_model` decides whether the static decode and prefill twins exist, by
  looking for the `pre` program of the first layer this runner serves. When that layer is a GDN layer, the program
  does not exist, so a pack hit would switch both twins off for the attention layers of a hybrid model.
- Nobody has saved or loaded a pack with GDN programs yet.

## Speed

Serving speed of GDN layers was out of scope for the first version of GDN serving (the `gdn<W>` programs in
`EmmyGenRunner`). Measurements below are from the tiny test model (hidden size 64) on an RTX 5080 Laptop, float32,
with every scheduling knob off (`WORK`, `TILE`, `REDUCE`, `STAGE`, `RASTER` all empty), the form the existing rows of
`tests/serving/goldens/serving.golden.json` use.

- **A fused GDN kernel needs cuts.** With no cut, the fused kernel recomputes its producers inside every output
  cell; one width-1 call took 165 s. The recorded rows cut `k_linear_mean_conv1d_reduce_290483` (width 1) into
  4 pieces, and `k_linear_matmul_mean_reduce_a19fd0` (width 4) and `k_linear_matmul_mean_reduce_5d3ccd` (width 16)
  into 7 pieces each. A call then takes 0.3 ms at width 1, 18.6 ms at width 4 and 24.5 ms at width 16.
- **The convolution kernel is the next cost.** The convolution kernel of the width-16 program,
  `k_conv1d_linear_mean_reduce_72cf22`, is uncut and takes 13 ms of the 24.5 ms. Cutting it at every cut its
  placement fork offers brought a width-16 call to 8 ms in an experiment whose rows are not in the golden.
- **More widths.** The first version builds widths 1, the decode bucket and the prefill bucket, and decomposes a
  request's tokens greedily into them. With all powers of two up to the prefill bucket the decomposition is the
  binary expansion of the token count. Example with a decode bucket of 16 and a prefill bucket of 64: 63 tokens cost
  6 calls with powers of two, and 18 calls with widths 64, 16 and 1. The price is more programs to compile at boot
  and more rows to record.
- **A masked program.** Hugging Face's chunked computation already pads its inputs to a multiple of 64 inside the
  layer. It pads after the projections, where the padding does not change the state. A program with a length input
  could expose that padded length, so one wide program would serve any shorter length. A narrow program may already
  cost nearly as much as the width-64 program for the same reason; nobody has measured it.
- **A mix of both:** decompose the large part of a length into wide programs, then mask the remainder.
- **One call for several requests.** The first version runs the GDN programs for one request after another. The
  programs have a batch axis of size 1.
- **State copies.** Each GDN call copies the request's state into the program's buffers and back. Aliasing the
  program's state buffers onto the request's memory would remove the copies.
