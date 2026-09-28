# Native fp4 matmuls cannot use TMA staging

## Summary

On sm_120, a W4A4 NVFP4 matmul (4-bit weights and 4-bit activations) can run on the native fp4 tensor-core instruction
`mma_m16n8k64_e2m1_f32`. The code calls that schedule the *block-scaled cell*. It multiplies the packed 4-bit codes
directly, and it applies each 16-value block's scale inside the instruction.

For this cell, the scheduler offers cp.async staging only: `STAGE=dN/smem-async`, N = 1–4, with and without `/p2`. It
never offers TMA (`dN/smem-tma`). Here `dN` is the number of buffered K steps; `/p2` adds a second register buffer.
Pinning TMA fails:

```
ValueError: STAGE pin 'd2/smem-tma' does not resolve for this contraction
```

The same matmul with 16-bit activations (W4A16) offers TMA at every depth, and so does plain f16. That includes TMA
copies of packed 4-bit weight bytes. In these reproducers, the missing path is specific to the fp4 instruction.
This limits the available schedules; it does not show that TMA would be faster than cp.async.

## Reading the schedule difference

Abbreviated Tile IR for the fp4 contraction; operands and outputs are omitted:

```text
Observed with the cp.async pin:
Fold[…] contraction ⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 STAGE=d3/smem-async⟩

Expected with TMA support (illustrative; the pin currently fails):
Fold[…] contraction ⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 STAGE=d2/smem-tma⟩
```

`TILE` chooses the matrix instruction and fragment layout; `STAGE` chooses how operands reach shared memory.
The matrix instruction stays the same. Only the method for copying stored codes and scales into shared memory changes.

## Reproduce

From the repository root, inside `nix develop`, with a fresh empty tune DB so no recorded row interferes:

```sh
rm -f /tmp/fp4-tma.db; export EMMY_TUNE_DB=/tmp/fp4-tma.db
PROG='
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4096, 1024, bias=False)
    def forward(self, x):
        return self.b(x)
M().half()(torch.randn(16, 4096).half())'

# 1. cp.async on the fp4 cell: compiles.
EMMY_KNOBS='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d3/smem-async' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4 --ir tile -c "$PROG"

# 2. TMA on the fp4 cell: fails.
EMMY_KNOBS='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d2/smem-tma' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4 --ir tile -c "$PROG"
# ValueError: STAGE pin 'd2/smem-tma' does not resolve for this contraction
```

The two-channel form (two weights, each with its own accumulator) fails the same way. Here two `nn.Linear(4096, 1024)`
read `a = x + 1`, and the program multiplies their outputs, so fusion builds one contraction with two weight channels:

```sh
PROG2='
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4096, 1024, bias=False)
        self.c = nn.Linear(4096, 1024, bias=False)
    def forward(self, x):
        a = x + 1
        return self.b(a) * self.c(a)
M().half()(torch.randn(16, 4096).half())'
EMMY_KNOBS='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d2/smem-tma' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4 --ir tile -c "$PROG2"
# ValueError: STAGE pin 'd2/smem-tma' does not resolve for this contraction
```

The STAGE values surviving the scheduler's legality checks, in both programs: `''`, `d1/smem-async`,
`d1/smem-async/p2`, …, `d4/smem-async/p2`. There is no `smem-tma` value. The empty value (`''`) means no shared-memory staging and applies
only to scalar tiles.

CUDA of repro 1 (`--ir cuda`) shows the cp.async form. Four shared buffers (activation codes, weight codes, and a
scale buffer for each) rotate through a three-slot ring:

```
__shared__ __align__(16) unsigned char _a_smem[...];      // activation codes
__shared__ __align__(16) unsigned char _b_smem[...];      // weight codes
__shared__ __align__(16) __nv_fp8_e4m3 _as_smem[...];     // activation block scales
__shared__ __align__(16) __nv_fp8_e4m3 _bs_smem[...];     // weight block scales
for (int _ks = 0; _ks < 4096; _ks += 256) {
    emmy_cp_async_cg(&_a_smem[((_ks / 256 + 2) % 3 * 32 + _fa / 8) * 144 + …], &x_static_fp4_bits[…]);
    emmy_cp_async_cg(&_b_smem[(_ks / 256 + 2) % 3 * 128 * 144 + …], &p_b_weight_bits[…]);
    …
    emmy_cp_async_wait<2>();
```

## Compare: W4A16 and f16 offer TMA

With `--quantize nvfp4`, both weights and activations are 4-bit; the activation encode runs separately in these
reproducers. Under `--quantize nvfp4-w4a16`, only the weights are quantized. `PROG`'s matmul uses the 16-bit
instruction on weights decoded in registers. Its available STAGE values include all 16 copy values: `d1`–`d4` × `smem-async`/`smem-tma` × with and without `/p2`. Plain
f16 (no `--quantize`) has the same 16.

With `EMMY_KNOBS='STAGE=d2/smem-tma'`, the W4A16 program compiles to `⟨TILE=mma_m16n8k16_f16_f32/f2x2/k8
STAGE=d2/smem-tma⟩`. Its CUDA copies the packed weight bytes by TMA:

```
void k_linear_…(…, const CUtensorMap* __restrict__ _desc_a, const CUtensorMap* __restrict__ _desc_b) {
    __shared__ __align__(1024) __half _a_smem[8192];
    __shared__ __align__(128) unsigned char _b_smem[4096];   // packed 4-bit weight bytes
    __shared__ unsigned long long _mbar[2];
    for (int _ks = 0; _ks < 4096; _ks += 128) {
        …
        mbarrier_wait_parity(&_mbar[_ks / 128 % 2], _ks / 128 / 2 % 2);
```

## Cause

The fp4 cell has its own staging resolver, `_block_scaled_warp_stage` in `emmy/compiler/ir/schedule/staging.py` (line
373). It returns `None` for every transport except `smem-async`. Its docstring says why: "cp.async (the
four-descriptor TMA box copy is not built — a missing-code fact, stated where the code would live)". For the
multi-channel form, `_stage_candidates` in `emmy/compiler/ir/schedule/classic/refusals.py` (line 454) also keeps only
`smem-async`. This confirms a missing implementation. It does not establish which TMA layouts or depths will be legal.

## Fix criteria

Done when all of these hold for both `PROG` and `PROG2`, at `--target sm_120 --quantize nvfp4`:

- **Loop IR:** unchanged. `--ir loop` output is byte-identical before and after.
- **Tile IR:**
  - With an fp4-cell TILE pinned (`mma_m16n8k64_e2m1_f32/…`), the available STAGE values include `dN/smem-tma` and
    `dN/smem-tma/p2` for N = 1–4, beside the existing `smem-async` values.
  - `EMMY_KNOBS='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d2/smem-tma'` compiles, and the contraction line reads
    `⟨TILE=mma_m16n8k64_e2m1_f32/f1x2/k4 STAGE=d2/smem-tma⟩`.
- **CUDA** (`--ir cuda`, same pin):
  - One `const CUtensorMap*` kernel parameter per stored buffer: activation codes, activation scales, and codes and
    scales for each weight channel.
  - `cp.async.bulk.tensor` fills each of those buffers in an `__align__(128)` or wider shared buffer, with one
    mbarrier per ring slot and `mbarrier_wait_parity` before the tensor-core instructions read it.
  - No `emmy_cp_async_cg` remains for those buffers inside the K loop.
  - With `d2` or deeper and K = 4096, the kernel issues the copy of K step i+1 before the tensor-core instructions of
    step i.
  - The instruction is still `emmy_mma_m16n8k64_e2m1_f32`, with the scales applied inside it.
- **Correctness:** on an sm_120 card, for `PROG` and `PROG2` at depths 1, 2 and 3, the TMA schedule's outputs equal
  the `smem-async` schedule's outputs (same TILE, same inputs) to the last bit, since only the copy mechanism differs.
  The [inline strict report](strict-fails-for-inline-quantize-programs.md) explains why the current benchmark
  comparison cannot establish correctness for these repros: it can use different weight snapshots and compares
  against unquantized eager. Validate a reference for the same quantized graph before relying on a strict result.
- **No regression:** the `smem-async` values stay available, and their CUDA is unchanged.

## Out of scope

- An activation whose 4-bit codes this same matmul computes has no buffer to copy. That can happen when its quantize
  fuses in, although in the programs above it runs as a separate kernel. That buffer stays compute-filled, as today.
- K steps shorter than 256 values stay unavailable for this cell. A scale row covers 16 values per byte, and cp.async
  copies whole 16-byte chunks, so a step needs at least 256 values (`_block_scaled_warp_stage` docstring). One
  consequence: a K-split kernel whose share of K is under 256 cannot use the fp4 cell.

## Appendix: listing the available options

Save as `list_forks.py` and run it in place of `emmy`, for example `./venv/bin/python list_forks.py compile --target
sm_120 --quantize nvfp4 --ir tile -c "$PROG" > /dev/null`. It prints each fork the compile meets, including forks
inside cut pieces that the default pick explores while pricing a cut. Each fork gets its cut options, its count of
fp4-cell TILE values, and the STAGE values of its schedule leaves.

```python
import sys
import emmy.compiler.pipeline.search.policy.greedy as G
from emmy.compiler.pipeline.fork import iter_leaves, leaf_knobs
from emmy.compiler.pipeline.knob import decision_view
from emmy.compiler.pipeline.pipeline import _option_decision

orig = G.greedy_decide


def wrapped(*args, **kwargs):
    decide = orig(*args, **kwargs)

    def logged(fp):
        root = getattr(fp.root_op, "knobs", None) or {}
        print(f"=== fork at {fp.node_id} ({getattr(fp.root_op, 'name', '?')})", file=sys.stderr)
        for splice in fp.splices:
            print(f"   CUT {_option_decision(splice, root)}", file=sys.stderr)
        stages, tiles, n = set(), set(), 0
        for leaf in iter_leaves(fp.variants):
            n += 1
            knobs = decision_view(leaf_knobs(leaf))
            stages.update(str(v) for k, v in knobs.items() if k.split("@")[0] == "STAGE")
            tiles.update(str(v) for k, v in knobs.items() if k.split("@")[0] == "TILE")
        fp4 = sum("e2m1" in t for t in tiles)
        print(f"   leaves={n} fp4_tiles={fp4} STAGE={sorted(stages)}", file=sys.stderr)
        return decide(fp)

    return logged


for module in list(sys.modules.values()):
    if module is not None and getattr(module, "greedy_decide", None) is orig:
        module.greedy_decide = wrapped
from emmy.emmy import main  # noqa: E402

sys.exit(main())
```
