# An f16 matmul whose activation is computed in the kernel cannot copy its stored weights by TMA

## Summary

Some 16-bit matmuls compute their activation inside the kernel, for example `x + 1` or a norm fused in front of a
projection. The scheduler gives such a matmul only the smem compute fill, where the compute threads write the
activation into shared memory. Under `STAGE=d2/smem` the stored weights already arrive through a two-deep cp.async
ring beside that fill.

What is never available: TMA copies of the stored weights, and the `/p2` register double buffer. Pinning them fails:

```
ValueError: STAGE pin 'd2/smem-tma' does not resolve for this contraction
```

The same program with packed 4-bit weights (W4A16) gets exactly this combination: the compute threads fill the
activation while TMA copies the weights. So the pattern exists in emmy, but only on the packed-weight path.

## Terms

- **Tile IR** (`emmy compile --ir tile`) prints each matmul as `Fold[k …] contraction` with its operands in order. The
  first operand is **A**, the activation here; the rest are **B**, one per weight.
- A **channel** is one weight with its accumulator. A fused gate-and-up pair is one contraction with two channels.
- **STAGE** is the knob for how operands reach shared memory. Its value is a depth `dN` (how many K steps of buffers
  rotate, the *ring*) plus a transport:
  - `smem` is the smem compute fill: the compute threads write shared memory themselves. At depth 2 they also prefetch
    the stored B buffers by cp.async.
  - `smem-async` copies every staged operand with cp.async.
  - `smem-tma` copies with TMA bulk copies.
  - A suffix `/p2` adds a second register buffer between shared memory and the tensor-core instruction.
  - `''` means no shared-memory staging; operands are read straight from global memory.
- **Available options** below means the values that appear in a fork's schedule leaves after the scheduler's own
  checks. Those are exactly the values a pin can reach. The script in the appendix prints them.
- **Knob pins** go in `EMMY_KNOBS` as comma-separated `KNOB@site=value`. The site (`map.1/inner.1/map`) names one node
  in the kernel's schedule tree.

## Reproduce

From the repository root, inside `nix develop`, with a fresh empty tune DB:

```sh
rm -f /tmp/f16-computed-a.db; export EMMY_TUNE_DB=/tmp/f16-computed-a.db
PROG='
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4096, 1024, bias=False)
        self.c = nn.Linear(4096, 1024, bias=False)
    def forward(self, x):
        a = x + 1
        return self.b(a) * self.c(a)
M().half()(torch.randn(16, 4096).half())'

# 1. Default: one kernel, compute fill.
./venv/bin/emmy compile --target sm_120 --ir tile -c "$PROG"
#   … contraction   ⟨TILE=mma_m16n8k16_f16_f32/f2x2/k2 STAGE=d2/smem⟩

# 2. TMA: fails.
EMMY_KNOBS='STAGE=d2/smem-tma' ./venv/bin/emmy compile --target sm_120 --ir tile -c "$PROG"
# ValueError: STAGE pin 'd2/smem-tma' does not resolve for this contraction

# 3. Cut the `x + 1` computation into its own kernel first: TMA now resolves,
#    at the cost of an extra kernel and a round trip of the activation through memory.
EMMY_KNOBS='PLACE@map.1/inner.1/map=cut,STAGE=d2/smem-tma' ./venv/bin/emmy compile --target sm_120 --ir tile -c "$PROG"
```

Tile IR of repro 1. The activation is computed (operand 0 holds `load x` and the `+ 1`), and the two weights are
stored:

```
    ├─ operand[acc0, acc1]: Fold[a2 in 0..4096] contraction   ⟨TILE=mma_m16n8k16_f16_f32/f2x2/k2 STAGE=d2/smem⟩   ‹computed›
    │  ├─ operand[v0]: Fold  free   ‹computed›
    │  │  ├─ operand[in3]: load x[a0, a2]   ‹materialized›
    │  ├─ operand[in1]: load linear_1_wt[a2, a1]   ‹materialized›
    │  ├─ operand[in2]: load linear_wt[a2, a1]   ‹materialized›
```

Available STAGE values: `''`, `d1/smem`, `d2/smem`.

A single-weight f16 matmul with a computed activation is limited the same way. Example: two `nn.Linear(4096, 1024)`
and `nn.Linear(4096, 512)` over `x + 1`, split apart with `EMMY_KNOBS='PLACE@map.1/inner=cut,REDUCE='`. Each piece is
limited to the same three values.

## Compare: packed 4-bit weights

With `--quantize nvfp4-w4a16`, `PROG` keeps the same computed `x + 1` as A, and each weight becomes a decode of packed
4-bit bytes. Available STAGE values: `''`, `d1/smem`, and all 16 copy values, `d1`–`d4` × `smem-async`/`smem-tma` ×
with and without `/p2`. The default pick is `⟨TILE=mma_m16n8k16_f16_f16/f4x8/k2 STAGE=d2/smem-tma/p2⟩`. Its staging
code, `_packed_warp_stage` in `emmy/compiler/ir/schedule/staging.py` (line 274), copies the weight bytes and lets the
compute threads fill the activation buffer.

## Cause

`_stage_candidates` in `emmy/compiler/ir/schedule/classic/refusals.py` (lines 455–458) chooses the transports a matmul
may use:

```python
if _needs_fill(tile, node, choice.tile):
    candidates: tuple[Stage, ...] = fill_stage_moves()
    if tile.packed_reading(node)[0] is not None:
        candidates = (*candidates, *stage_moves(warp=True, ctx=target))
```

- A computed operand makes `_needs_fill` true.
- The copy transports (`stage_moves`) are added only when `packed_reading(node)[0]` finds packed weights. Index 0 is
  the packed-weight reading; index 1 is the native fp4 pair.
- An f16 matmul with stored weights has no packed reading, so it keeps the fill values only.

The fill resolver `resolve_fill_stage` (`staging.py`, line 694) states the constraint a fix must respect: "the
byte-copy / cp.async / TMA transports move bytes and cannot evaluate a producer cone". A computed activation therefore
has to stay on the compute fill. What is missing is a way to put the stored weights of such a matmul on TMA while the
fill writes the activation. The packed path already builds exactly that.

Deeper compute-fill rings are not part of this bug. `fill_stage_moves` (`refusals.py`, line 423) leaves out depths 3
and 4 on purpose: measured on a Qwen3 gate/up matmul, they ran 1.38× and 2.22× slower than depth 2 on an A100.

## Fix criteria

Done when, for `PROG` at `--target sm_120` without quantization:

- **Loop IR:** unchanged. `--ir loop` output is byte-identical before and after.
- **Tile IR:**
  - The two-channel contraction's available STAGE values include `dN/smem-tma` and `dN/smem-tma/p2` for at least N = 1
    and 2, beside the existing `d1/smem` and `d2/smem`.
  - `EMMY_KNOBS='STAGE=d2/smem-tma'` compiles without any cut, and the contraction line shows `STAGE=d2/smem-tma`.
  - The single-weight piece described under *Reproduce* gets the same values.
- **CUDA** (`--ir cuda`, TMA pin):
  - One `const CUtensorMap*` kernel parameter per weight channel.
  - `cp.async.bulk.tensor` fills each weight buffer in a ring of at least two slots, with an mbarrier per slot and
    `mbarrier_wait_parity` before the tensor-core instructions read it.
  - The compute threads write the activation buffer, evaluating `x + 1` from `x`.
  - There is no separate kernel for `x + 1`.
  - With K = 4096, the kernel issues the copy of the next K step's weights before the tensor-core instructions of the
    current step.
- **Correctness:** on an sm_120 card, `EMMY_KNOBS='STAGE=d2/smem-tma' ./venv/bin/emmy run -c "$PROG" --bench --strict`
  exits 0. `--strict` fails the run unless emmy's outputs match eager's.
- **No regression:**
  - `d1/smem` and `d2/smem` stay available, with unchanged CUDA.
  - Matmuls whose operands are all stored keep their current values.
  - The sm_70 rule in `resolve_fill_stage` still refuses a depth-2 fill ring on instructions without cp.async.

## Out of scope

- Ringing the compute-filled activation buffer itself.
- Deeper compute-fill rings (see *Cause*).
- Which form the default pick chooses, fused or with `x + 1` cut out.

## Appendix: listing the available options

Save as `list_forks.py` and run it in place of `emmy`, for example `./venv/bin/python list_forks.py compile --target
sm_120 --ir tile -c "$PROG" > /dev/null`. It prints each fork the compile meets, including forks inside cut pieces
that the default pick explores while pricing a cut. Each fork gets its cut options and the STAGE values of its
schedule leaves.

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
