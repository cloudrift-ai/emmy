# Inline quantized benchmarks use mismatched weights and an unquantized strict reference

## Summary

`emmy run -c "<program>" --quantize nvfp4|nvfp4-w4a16 --bench --strict` fails for every knob setting tried on the
program below, with an error as large as the outputs. These failures cannot distinguish a kernel error from a bad
reference comparison. The finding concerns this inline benchmark path; it does not establish that every quantized
program or checkpoint-based correctness check fails.

## Reading the reference mismatch

This is a dataflow sketch from source review, not IR output. `Q` includes the selected quantization and its scales.

```text
Observed, unseeded:                 Expected strict comparison (illustrative):
Emmy: Q(W_parent, X_worker)         Emmy:  quantized graph(weights, scales, inputs)
eager: W_worker, X_worker           oracle: same semantics, weights, scales, inputs
compare at rtol = atol = 1e-3       compare with a justified numerical tolerance
```

Rerunning the snippet creates different worker weights. Seeding aligns the original snapshots for this reproducer,
but unquantized eager still computes a different program. Its delta can report quantization error separately.

## Two reference problems

1. **Emmy and eager run different weights.** The bench worker runs the `-c` code a second time, which draws new random
   weights and inputs for its eager module. Emmy's packed weights still come from the checkpoint the parent process
   wrote from the first run's weights. With `torch.manual_seed(0)` at the top of the program both runs draw the same
   weights. The reported error drops 10–14×; attributing the residual entirely to quantization is unverified.
2. **The strict reference still uses unquantized eager at 1e-3.** Even with the same original weight snapshot,
   quantization changes the program. Its error against eager can exceed `rtol = atol = 1e-3` even for a correct
   implementation; the seeded runs below reportedly still fail. `emmy/commands/run.py` says the eager delta should
   be informational under `--quantize`, and names the numpy backend on the same graph as the intended oracle.
   That oracle is not wired into the inspected inline benchmark comparison. The non-strict call disables eager
   gating (`accuracy=not (skip_accuracy or quantized)`), but still passes `strict_accuracy=strict_correctness`.
   For this frontend-runnable path, `valid_proof` in `_strict_benchmark_errors` requires an `"eager"` proof at
   `rtol == atol == 1e-3`. Strict mode also requires positive captured timings from every requested backend. Other
   paths can accept `"same-input-greedy"`; the restriction is not universal.

## Review status

Source review at `a98fd4f8` confirms the parent/worker weight-binding mismatch for the unseeded reproducer and the
strict comparison against unquantized eager. No GPU runs were repeated during review. The measured errors and seeded
improvement below are the original investigator's reported results, not independently verified here.

**Unverified:** whether all residual seeded error comes from quantization, whether every tested kernel is correct,
and how broadly other inline programs fail. Agreement across schedules does not establish kernel correctness. The
original claim that quantization *always* exceeds 1e-3 is too broad: it depends on the data and quantization error.
The `--ab` behavior and calibration effects discussed below also remain unverified at runtime.

## Reproduce

From the repository root inside `nix develop`. `-c` runs a Python snippet whose final statement calls the module.
`nvfp4` quantizes weights and activations (W4A4); `nvfp4-w4a16` quantizes only weights (W4A16). The fresh tune DB keeps
earlier measurements out of the greedy pick: the schedule chosen without hand pins.

```sh
rm -f /tmp/strict.db; export EMMY_TUNE_DB=/tmp/strict.db
PROG='
class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.b = nn.Linear(4096, 1024, bias=False)
    def forward(self, x):
        return self.b(x)
M().half()(torch.randn(16, 4096).half())'

# 1. Unseeded, quantized: error as large as the outputs.
./venv/bin/emmy run -c "$PROG" --quantize nvfp4-w4a16 --bench --strict --warmup 2 --iters 5
./venv/bin/emmy run -c "$PROG" --quantize nvfp4 --bench --strict --warmup 2 --iters 5
EMMY_KNOBS='TILE=f4x2,WORK=t16x8' ./venv/bin/emmy run -c "$PROG" --quantize nvfp4 --bench --strict --warmup 2 --iters 5
EMMY_KNOBS='TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d1/smem-async' \
  ./venv/bin/emmy run -c "$PROG" --quantize nvfp4 --bench --strict --warmup 2 --iters 5

# 2. Seeded, quantized: quantization-sized error, still a failure at 1e-3.
./venv/bin/emmy run -c "torch.manual_seed(0)
$PROG" --quantize nvfp4-w4a16 --bench --strict --warmup 2 --iters 5

# 3. Unseeded, not quantized: passes (exit 0).
./venv/bin/emmy run -c "$PROG" --bench --strict --warmup 2 --iters 5
```

`TILE=f4x2,WORK=t16x8` pins a scalar schedule (no tensor cores); `TILE=mma_m16n8k64_e2m1_f32/…` pins the native fp4
tensor-core instruction. Reported measurements from 2026-09-28 at commit `a98fd4f8`, RTX 5080 Laptop GPU:

| run | mean abs error | max abs error |
| --- | ---: | ---: |
| W4A16, unseeded, greedy compile | 0.658 | 3.27 |
| W4A4, unseeded: greedy, scalar pin, fp4 pin | 0.647–0.662 | 3.24–3.99 |
| W4A16, seeded | 0.047 | 0.245 |
| W4A4, seeded | 0.065 | 0.376 |
| 16-bit, unseeded | passes | passes |

The layer's outputs have a standard deviation of about 0.58: 4096 inputs from N(0, 1) times weights from U(−1/64, 1/64).
Two unrelated outputs of that size differ by a mean of about 0.65, which is what the unseeded quantized runs show.

## Cause of the weight mismatch

- The parent quantizes its traced module's weights into a checkpoint. `write_quantized_checkpoint`
  (`emmy/compiler/loader/synthesize.py`, lines 160–186) renames each quantized constant's `source_path` to
  `l<i>.weight`.
- The parent sends the worker both `"code": args.code` and `"input": quantized_checkpoint` (`run.py`, lines 396–398).
- The worker calls `load_or_trace` (`_bench_worker.py`, line 222), which prefers `code` (`compile.py`, lines 706–709).
  It runs the `-c` code again, unseeded, so the module gets new weights and new inputs.
- The worker's module parameters are named `b.weight`, so they cannot bind the renamed constants (`run.py`, lines
  2958–2970). Those bind from the parent's checkpoint instead (`_bench_worker.py`, line 242; `run.py`, lines 2972–2979).
  `compile.py` (lines 686–688) describes this path: re-run the code, then bind constants from the checkpoint.
- Result: emmy computes with the parent's weights, and the worker's eager forward with the worker's. Both see the
  worker's inputs.
- W4A4 also uses the parent's calibrated activation `input_scale`. `_spell_static_fp4_quantize` in
  `emmy/compiler/loader/quant.py` creates a checkpoint-backed `ConstantOp` with `source_path=scale_key`, not a literal.
  Calibration uses the parent's activations; execution uses the worker's. **Unverified hypothesis:** this difference
  contributes to the observed residual error. Different calibration and inference inputs are normal for static
  quantization, so that difference alone does not establish a second bug.
- Without `--quantize` there is no checkpoint, so both sides bind the worker module's parameters. That is why the 16-bit
  program passes.
- Without `--bench`, `emmy run` binds the parent's module and checkpoint together (`run.py`, line 352), with no weight
  mismatch.

## Fix criteria

1. **Consistent data provenance.** Emmy's quantized parameters and the eager comparison originate from the same
   original weight snapshot, and both execute on the same inputs. A reference for the quantized graph uses the same
   packed weights and calibration scales as that graph. Seeding should not be required to align the snapshots.
   **Unverified expectation:** removing the unrelated weights should reduce the unseeded error toward the seeded
   results. Their exact error values are observations, not acceptance thresholds.
2. **A reference that fits a quantized program.** Strict validation checks the quantized computation against a
   trusted reference for those semantics, with a justified numerical tolerance. The numpy backend on the same graph
   is the candidate named in `run.py`; its coverage and suitability for these repros remain unverified. The original
   proposal also suggested unquantized eager with a quantization-aware tolerance; whether that would distinguish
   kernel errors reliably is unverified. Merely loosening the eager tolerance until these runs pass is insufficient.
   Correct greedy and pinned schedules should pass; the current failures do not prove those schedules correct.
3. **Wrong results still fail.** Validation must detect an intentionally incorrect quantized output as well as accept
   a known-correct one.
4. **Visible reference.** The `--strict` result line and the `--json` `strict` payload name the reference used.

## Notes

- **Unverified:** the `--ab` wrong-answer check compares each pinned row with the greedy row on the greedy row's
  inputs. It holds the same weights only because the rows reuse the greedy row's bound constants, which assumes node
  ids repeat across traces. Under W4A4 each unseeded `--ab` row synthesizes its own calibrated `input_scale`; whether
  the subsequent constant binding preserves that difference or replaces it with the greedy row's scale has not been
  checked.
- **Source-based expectation, not independently reproduced:** under `--strict` in the `-c` path, `run.py` benches
  `--ab` rows without a strict proof, so they fail with "lacks strict eager correctness" (`run.py`, lines 439–449 and
  2523–2524). With `--quantize`, the greedy row's failure skips them first (line 434).
- Not examined: checkpoint-based runs (`emmy run <model id>` or `--golden <file>`), which take other paths.
