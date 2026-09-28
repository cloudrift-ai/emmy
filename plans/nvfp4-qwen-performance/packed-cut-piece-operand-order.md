# Cutting packed-weight matmuls reverses their operands and loses staging

## Summary

Take two matmuls of different widths that read one computed activation, with 4-bit NVFP4 weights and 16-bit
activations (W4A16). Fusion puts both in one kernel. The placement cut `PLACE@map.1/inner=cut` splits them into two
kernels; the cut code calls each one a *piece*.

In each piece, the matmul's first operand is the decoded weight and its second is the activation, and its internal
free-axis order is weight row then token. The store still writes the logical output in token-then-weight-row order.
The packed-weight staging expects the reverse order, so it does not apply. Each piece can then only use the smem
compute fill (`STAGE=d1/smem`), where the compute threads decode every weight element into a 16-bit shared buffer.
Pinning cp.async or TMA fails:

```
ValueError: STAGE pin 'd2/smem-async' does not resolve for this contraction
```

When K is split four ways across thread blocks (`REDUCE=g4k`) together with the cut, the split's kernels come out
activation first, and every cp.async and TMA option is available. The plain f16 version of the program also orients its pieces activation first.
The staging matcher rejects the weight-first form. Why the K-split path chooses the other orientation remains
untraced.

## Expected Tile IR

The observed Tile IR excerpt under **Reproduce** puts `operand[v4]` (weight decode) before `operand[v5]` (activation).
The following is **composed expected IR**, with those operand subtrees reversed and copy staging available.
`…` omits the unchanged weight-decode arithmetic and accumulator operations:

```text
=== 0: k_linear_reduce_e63dae__place_943d6e4dd9 ===
    place  free=(a1, a0)  grid=(a1, a0)
    work   w1x2
    Fold[a2 in 0..4096] contraction   ⟨TILE=mma_m16n8k16_f16_f32/f1x8/k8 STAGE=d2/smem-async⟩
    ├─ operand[v5]: Fold  free   ‹computed›
    │  ├─ operand[in5]: load x[a1, a2]   ‹materialized›
    │  └─ lift: λ(in5) -> (v5)
    │       v5 = add(1, in5)
    ├─ operand[v4]: Fold  free   ‹computed›
    │  ├─ operand[in2]: load p_b_weight_bits[a0, (((a2 / 16) * 8) + ((a2 % 16) / 2))]   ‹materialized›
    │  ├─ operand[in4]: load p_b_weight_scale_bits[a0, (a2 / 16)]   ‹materialized›
    │  …
    …
    outputs
    └─ linear[a1, a0] = acc0
```

The returned tensor is still token first (`a1`) and weight-row second (`a0`). The change exposes the packed weight in
the B operand position, where the staging matcher recognizes it; it does not change the mathematical output layout.

## Reproduce

From the repository root, inside `nix develop`, with a fresh empty tune DB so no recorded row interferes.
`KNOB@site=value` pins one schedule-tree node; `REDUCE=` disables K splitting. A K split produces partial results
across thread blocks and then combines them in a finishing kernel.

```sh
rm -f /tmp/cut-order.db; export EMMY_TUNE_DB=/tmp/cut-order.db
PROG='
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4096, 1024, bias=False)
        self.c = nn.Linear(4096, 512, bias=False)
    def forward(self, x):
        a = x + 1
        return self.b(a), self.c(a)
M().half()(torch.randn(16, 4096).half())'

# 1. Cut, no K split: weight-first pieces, fill only.
EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE=' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4-w4a16 --ir tile -c "$PROG"

# 2. The same, asking for cp.async: fails.
EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE=,STAGE=d2/smem-async' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4-w4a16 --ir tile -c "$PROG"
# ValueError: STAGE pin 'd2/smem-async' does not resolve for this contraction

# 3. Cut plus K split: activation-first kernels, all transports available.
EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE=g4k' \
  ./venv/bin/emmy compile --target sm_120 --quantize nvfp4-w4a16 --ir tile -c "$PROG"
```

Loop IR (`--ir loop`) writes each product weight first. Here `v4` is B's decoded weight and `v5` is `x + 1`:

```
f32 v6 = multiply(v4, v5)
acc0 <- add(acc0, v6)
```

Tile IR of repro 1, first piece. The weight decode is operand 0. The output index `linear[a1, a0]` uses the reverse of
the internal free-axis order: `a0` runs over the weight's 1024 rows and `a1` over the 16 tokens. The logical output is
already token first; this is not an incorrect external transpose.

```
=== 0: k_linear_reduce_…__place_… ===
    Fold[a2 in 0..4096] contraction   ⟨TILE=mma_m16n8k16_f16_f32/f1x8/k8 STAGE=d1/smem⟩
    ├─ operand[v4]: Fold  free   ‹computed›
    │  ├─ operand[in2]: load p_b_weight_bits[a0, (((a2 / 16) * 8) + ((a2 % 16) / 2))]   ‹materialized›
    │  ├─ operand[in4]: load p_b_weight_scale_bits[a0, (a2 / 16)]   ‹materialized›
    ├─ operand[v5]: Fold  free   ‹computed›
    │  ├─ operand[in5]: load x[a1, a2]   ‹materialized›
    outputs
    └─ linear[a1, a0] = acc0
```

Available STAGE values for each piece in repro 1: `''` (scalar tiles only) and `d1/smem`.

Tile IR of repro 3, one K-split kernel. The activation is operand 0. `Fold[a3 in 0..1024 ⊂ a3]` marks a K loop over
this kernel's quarter of K.

```
    Fold[a3 in 0..1024 ⊂ a3] contraction   ⟨…⟩
    ├─ operand[v1]: Fold  free   ‹computed›
    │  ├─ operand[in4]: load x[a1, ((1024 * a0) + a3)]   ‹materialized›
    ├─ operand[v5]: Fold  free   ‹computed›
    │  ├─ operand[in2]: load p_c_weight_bits[a2, …]   ‹materialized›
```

Available STAGE values there: `''`, `d1/smem`, and all 16 copy values: `d1`–`d4` × `smem-async`/`smem-tma` × with and
without `/p2`, where `/p2` adds a second register buffer. The `d1`–`d4` prefix is the number of buffered K steps.

## Compare: plain f16

The same program without `--quantize`, with `EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE='`, gives pieces with `x + 1` as
operand 0 and the stored weight (`load linear_wt[a2, a1]`) as operand 1. So the order is right there. These pieces are
still limited to `''`, `d1/smem`, `d2/smem`, because an f16 matmul whose activation is computed only gets the compute
fill. That is a separate limit.

## Cause

`Fold.__post_init__` in `emmy/compiler/ir/pure/fold.py` (lines 316–345) puts each bilinear contraction into a
canonical orientation, with A in `operands[0]`:

- **Several channels** (a fused two-weight kernel): A is the operand all products share. So the fused kernel is
  oriented correctly.
- **One product over two stored operands:** they are oriented by memory layout.
- **One product with a computed operand:** "a computed operand keeps the order its former chose" (line 330). The
  *former* is the code that built the Fold, and it follows the Loop IR's spelling.

Both operands of a cut piece are computed: the weight decode and `x + 1`. So the piece keeps `multiply(v4, v5)`,
weight first.

The packed-weight staging's matcher, `match_packed_b_node` in `emmy/compiler/ir/schedule/packing.py` (line 317), reads
A from the first operand and requires every B operand to be a packed-weight decode. Here the only B operand is `x +
1`, which is not one, so the matcher returns `None`. The contraction then gets the generic treatment for a computed
operand: the compute fill only. The intended packed-weight path copies raw weight bytes into shared memory and
decodes them in registers. In the inspected TMA form, compute threads still load and convert the scales separately.

The K split rebuilds its kernels from the contraction with re-indexed operands
(`emmy/compiler/pipeline/passes/tile/_split.py`), and they come out activation first. I did not trace why that path
orients differently.

## Fix criteria

Done when, for `PROG` at `--target sm_120 --quantize nvfp4-w4a16` with `EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE='`:

- **Loop IR:** unchanged. `--ir loop` output is byte-identical before and after.
- **Tile IR:**
  - In each piece, operand 0 is the activation (the Fold holding `load x[…]` and the `+ 1`), and operand 1 is the
    weight decode (the Fold holding `load p_*_weight_bits[…]`).
  - Each piece preserves the existing logical output layout: `linear[<token axis>, <weight-row axis>]`, regardless of
    internal axis names.
  - Each piece's available STAGE values include all 16 copy values listed under repro 3.
  - Adding `STAGE=d2/smem-async` or `STAGE=d2/smem-tma` to the pin compiles, and the piece's contraction line shows
    that STAGE.
- **CUDA** (`--ir cuda`, with the TMA pin):
  - Each piece holds its weight in an `unsigned char` buffer (packed bytes), and `cp.async.bulk.tensor` fills it
    through a `const CUtensorMap*` kernel parameter.
  - No weight buffer is `__half _b_smem[…]` filled element by element with decoded values.
  - The compute threads write the activation buffer, evaluating `x + 1` from `x`.
- **Correctness:** on an sm_120 card, the outputs under `EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE=,STAGE=d2/smem-tma'`
  match the outputs of the fused default schedule on the same inputs, within f16 rounding of a 4096-long sum. For
  these inline quantized repros, the [strict report](strict-fails-for-inline-quantize-programs.md) identifies a
  reference mismatch. Use the same original weight snapshot and a validated reference for the quantized computation;
  the current strict failure neither proves a kernel error nor establishes correctness.
- **No regression:**
  - The fused equal-width two-weight kernel and the K-split kernels keep their current operand order and available
    values.
  - Flash attention's score contractions, the case the orientation comment in `fold.py` protects, still orient both
    contractions of one kernel the same way. The attention tests under `tests/compiler` pass.

## Out of scope

- Whether the default pick, which has no measured rows for this card, should choose the cut.
- The f16 compute-fill limit mentioned under *Compare*.

## Appendix: listing the available options

Save as `list_forks.py` and run it in place of `emmy`, for example `./venv/bin/python list_forks.py compile --target
sm_120 --quantize nvfp4-w4a16 --ir tile -c "$PROG" > /dev/null`. It prints each fork the compile meets, including
forks inside cut pieces that the default pick explores while pricing a cut. Each fork gets its cut options and the
STAGE values of its schedule leaves.

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
