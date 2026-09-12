# Block-scaled FP8 linear kernels at decode width on RTX 5090

Date: 2026-09-10. Source base: `fa54b4621` plus the change this report ships with. GPU: NVIDIA GeForce RTX 5090
(`sm_120`), driver 580.173.02. Software: CUDA 13.0, PyTorch 2.13.0+cu130, vLLM 0.29.0. Emmy uses deployable nvcc
defaults (`-O3`).

This is the block-scaled FP8 companion to the [NVFP4 linear showcase](2026-09-10_nvfp4-linear-rtx5090.md), measured
under the same timing contract. The checkpoint form is the one the official Qwen, DeepSeek and GLM FP8 releases ship:
e4m3 weights with one f32 scale per 128x128 block, and dynamic e4m3 activations with one scale per token and per
128-wide K group. `Qwen/Qwen3-0.6B-FP8` supplies the shapes. Each timed invocation starts from BF16 activations,
computes the activation scales, quantizes, and runs the block-scaled matrix multiplication with BF16 output. Weight
quantization is outside every timed region.

## Results at M=16

The table reports the pooled median over 40 samples from two fresh processes. Each sample is the per-invocation
latency of 200 replays of a CUDA graph that holds exactly one complete linear route, with all implementations
interleaved in every process.

| Model use | M x K x N | Emmy | vLLM default: CUTLASS | vLLM Triton | CUTLASS / Emmy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B merged QKV | 16 x 1024 x 4096 | **8.19 us** | 8.20 us | 14.34 us | 1.00x |
| Qwen3-0.6B O | 16 x 2048 x 1024 | **8.20 us** | 11.32 us | 23.36 us | **1.38x** |
| Qwen3-0.6B merged gate/up | 16 x 1024 x 6144 | **8.20 us** | 8.20 us | 14.34 us | 1.00x |
| Qwen3-0.6B down | 16 x 3072 x 1024 | **10.25 us** | 14.34 us | 32.87 us | **1.40x** |

Emmy is 1.18x faster than vLLM's selected CUTLASS route and 2.30x faster than its Triton route by geometric mean. Both
of those quantize the activation exactly as Emmy does, per token and per 128-wide K group.

vLLM also ships two weight-only kernels that run this checkpoint on this GPU, Marlin and Humming. Its selector does not
choose them here, and they skip the activation quantization entirely: they multiply BF16 activations by the dequantized
weights, which is a different and more accurate function. Humming is the fastest implementation measured.

| Model use | Emmy | Marlin (weight-only) | Humming (weight-only) |
| --- | ---: | ---: | ---: |
| merged QKV | 8.19 us | 6.15 us | 6.14 us |
| O | 8.20 us | 10.24 us | 6.14 us |
| merged gate/up | 8.20 us | 8.19 us | 6.14 us |
| down | 10.25 us | 10.25 us | 6.14 us |

Humming is 1.41x faster than Emmy by geometric mean; Marlin and Emmy are level (0.98x).

## Device time explains the ties

The replay latencies above land on steps of about 2.05 us. On this machine a CUDA graph replay costs a multiple of that
step whatever it contains: one trivial kernel measures 2.05 us, three or four measure 4.10 us, and six measure 6.14 us.
Two routes whose GPU time falls in the same step therefore tie, which is what happens on both K=1024 rows. The
profiler's per-kernel durations, taken after the timing windows, separate them:

| Model use | Emmy quantize + matmul | CUTLASS quantize + scale transpose + GEMM | Humming |
| --- | ---: | ---: | ---: |
| merged QKV | 2.40 + 3.81 = 6.35 us | 1.06 + 0.74 + 5.26 = 7.05 us | 4.22 us |
| O | 2.40 + 4.93 = 7.32 us | 1.04 + 0.74 + 7.94 = 9.71 us | 3.85 us |
| merged gate/up | 2.34 + 4.42 = 6.76 us | 1.06 + 0.74 + 5.34 = 7.13 us | 4.35 us |
| down | 2.37 + 7.07 = 9.45 us | 1.02 + 0.74 + 10.77 = 12.54 us | 4.65 us |

Emmy's matrix multiplication is faster than CUTLASS's GEMM on every row, and on the two K=1024 rows it matches
Humming's complete call. Two things separate Emmy from Humming. The activation quantize costs 2.4 us, more than twice
vLLM's 1.06 us, and Humming does not quantize at all. On the two rows with N=1024 and a long K, Emmy's matmul has 32 to
64 CTAs for 170 SMs; a cross-CTA K split would add parallelism there, but it cannot be pinned by hand on this kernel set
(below).

## Correctness

Every sample and output check is stored in the `*-fair-run[12].json` receipts beside this report. Emmy's output differs
from CUTLASS's by at most 0.34% to 0.51% of the largest output magnitude; Triton's differs from CUTLASS's by 0.04%. The
two are not bit-identical by construction. The checkpoint program Emmy compiles rounds the quantized activation and the
dequantized weight to BF16 before a BF16 tensor-core product, which is what its spelled algebra says, while CUTLASS
multiplies the e4m3 values exactly and applies both scales in f32. The weight-only kernels differ by 2.5%, the size of
the activation quantization they skip.

## What Emmy runs

Two kernels. The first computes each token's 128-wide group maxima, the scale, and the quantized-and-restored
activation. The second is the matrix multiplication:

```text
PLACE=cut
WORK=w1x4                              (w1x2 on O)
TILE=mma_m16n8k16_bf16_f32/f1x1/k8     (f1x2 on merged gate/up)
STAGE=d2/smem-async
```

The weight's e4m3 bytes copy verbatim into a shared-memory slab through a two-slot `cp.async` ring, a 128-element K
chunk at a time, while its block scales fill a small f32 slab. The fragment load converts each lane's byte pair with one
hardware conversion, multiplies both values by the block's f32 scale and rounds once to BF16, then a BF16 `m16n8k16`
tensor-core instruction accumulates in f32. The weight crosses global memory at one byte per element and no
dequantized tile is ever written.

That staging is what changed. The same tensor-core tile was already offered before, but its weight was a computed
operand filled synchronously, cell by cell, with a K-strided byte gather: 9.7 us for the q projection's matmul, against
2.9 us staged. The staged drain performs the fill's own arithmetic, so the two are bit-identical; the tests hold that on
both copy transports and both 16-bit fragment dtypes.

## Decode width (M=1)

The same four shapes at one token, under the same protocol. At M=1 a tensor-core tile wastes fifteen of its sixteen
rows and every padded row re-reads the one activation row, so the route uses the scalar tier instead: one warp per
output element folds K cooperatively, reading the e4m3 bytes and their block scales straight from global memory.

```text
PLACE=cut
WORK=t32
REDUCE=coop
```

| Model use | M x K x N | Emmy | vLLM default: CUTLASS | vLLM Triton | Marlin (weight-only) | Humming (weight-only) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged QKV | 1 x 1024 x 4096 | **6.15 us** | 8.19 us | 18.43 us | 4.25 us | 6.14 us |
| O | 1 x 2048 x 1024 | **6.15 us** | 10.24 us | 22.56 us | 8.19 us | 6.14 us |
| merged gate/up | 1 x 1024 x 6144 | **8.19 us** | 8.19 us | 22.41 us | 6.15 us | 6.15 us |
| down | 1 x 3072 x 1024 | **8.20 us** | 13.19 us | 32.78 us | 8.22 us | 6.14 us |

Emmy is 1.37x faster than CUTLASS and 3.31x faster than Triton by geometric mean; Marlin is 1.10x and Humming 1.15x
faster than Emmy. In device time Emmy's quantize plus matmul is 1.38 + 3.94, 1.79 + 3.44, 1.38 + 4.74 and 2.18 + 4.42
us, against 6.19, 8.81, 6.38 and 11.69 us for CUTLASS, and 4.80, 4.20, 5.55 and 4.70 us for Humming. The receipts are
the `*-m1-*` files.

## What changed since the first version of this report

The first version measured the model's fused q projection with its q-norm at M=1 against eager and torch.compile
pipelines, and reported Emmy 4x slower than torch.compile. Three findings replaced that picture:

- **The activation grouping was wrong.** Emmy spelled one activation scale per token. transformers, vLLM and SGLang
  quantize these checkpoints per token and per 128-wide K group, and Emmy now does too.
- **The M=1 slowness was placement, not a missing tensor-core atom.** The 58 us kernel in that table recomputed the
  whole projection in 16 CTAs to derive each head's norm statistic, before a second kernel computed it again.
- **The matched baseline runs on this machine.** vLLM 0.29.0 is installed beside the compiler, so every number above
  comes from one process per run and one input set.

## What does not work yet

- **Fusing the quantize into the matmul is slower.** The matmul's A fill can now evaluate each row's group maximum once
  per staged chunk instead of once per slab cell, and the fused kernel is bit-identical to the two-kernel route. It
  measures 27.8 us on merged QKV because the synchronous fill waits on global memory twice per chunk with nothing to
  overlap it. The two-kernel route stays the deployed choice.
- **A cross-CTA K split cannot be pinned by hand here.** The route's two kernels share the bare `REDUCE` key, so a split
  pin also splits the quantize kernel's group maximum and the pinned compile refuses.
- **No native e4m3 tensor-core route.** A kernel that multiplies e4m3 values natively and applies each block's scale to
  the f32 accumulator per K chunk would double the tensor-core rate. At M=16 the matmul is limited by memory traffic,
  where the BF16 route already beats CUTLASS; the native route is the prefill lever.
- **The model's decode layer over-fuses.** With grouped activation scales, the q and k projections fuse into the
  attention kernel and the MLP becomes one kernel that recomputes the gate and up products inside the down
  projection's operand. Neither runs within the lane's budget, so the committed decode golden keeps only the norm and
  quantize kernels, and the prefill golden only its six small kernels. The first version's decode golden checked the q
  and k projections; that coverage waits on a fusion change.

## Commands for the reported numbers

Run from this checkout on the RTX 5090 with vLLM installed. nvcc must be on `PATH`, or vLLM's fast kernels do not build.

```bash
export CUDA_HOME=/usr/local/cuda PATH=/usr/local/cuda/bin:$PATH
R=evaluation_results/2026-09-10_fp8-block-linear-rtx5090

bench_twice () {
  label="$1"; k="$2"; n="$3"
  for run in 1 2; do
    python scripts/bench_quant_linear.py --format fp8-block \
      --m 16 --k "$k" --n "$n" --warmup 20 --samples 20 --calls 200 \
      --emmy-ir "$R/$label-kernel.json" --json "$R/$label-fair-run$run.json"
  done
  python scripts/bench_quant_linear.py --format fp8-block \
    --m 16 --k "$k" --n "$n" --warmup 20 --samples 20 --calls 200 --profile-calls 20 \
    --emmy-ir "$R/$label-kernel.json" --json "$R/profile-$label.json"
}

bench_twice qkv-merged 1024 4096
bench_twice o 2048 1024
bench_twice gateup-merged 1024 6144
bench_twice down 3072 1024
```

The M=1 rows run the same function with `--m 1` and the `-m1` labels.

Each `*-kernel.json` is the Kernel IR `emmy run` dumps for the pinned route, and `*-tile.txt` its readable Tile IR:

```bash
dump () {
  label="$1"; k="$2"; n="$3"; pins="$4"
  EMMY_KNOBS="$pins" emmy run \
    --code "torch.nn.Linear($k,$n,bias=False,dtype=torch.bfloat16)(torch.randn(16,$k,dtype=torch.bfloat16))" \
    --quantize fp8-block --bench --bench-backends emmy --warmup 20 --iters 100 --no-record-nodes \
    --dump-dir "/tmp/$label"
  cp "/tmp/$label/08_lowering_kernel.json" "$R/$label-kernel.json"
  cp "/tmp/$label/07_lowering_tile.kernels.txt" "$R/$label-tile.txt"
}

A=mma_m16n8k16_bf16_f32
dump qkv-merged 1024 4096 "PLACE=cut,WORK=w1x4,TILE=$A/f1x1/k8,STAGE=d2/smem-async"
dump o 2048 1024 "PLACE=cut,WORK=w1x2,TILE=$A/f1x1/k8,STAGE=d2/smem-async"
dump gateup-merged 1024 6144 "PLACE=cut,WORK=w1x4,TILE=$A/f1x2/k8,STAGE=d2/smem-async"
dump down 3072 1024 "PLACE=cut,WORK=w1x4,TILE=$A/f1x1/k8,STAGE=d2/smem-async"
```

The M=1 routes are dumped the same way from `torch.randn(1,$k,...)` under `EMMY_KNOBS="PLACE=cut,WORK=t32,REDUCE=coop"`.

The table aggregation:

```bash
python - <<'PY'
import json, math, statistics
from pathlib import Path

root = Path("evaluation_results/2026-09-10_fp8-block-linear-rtx5090")
refs = ("vllm_cutlass", "vllm_triton", "vllm_marlin_w8a16", "vllm_humming_w8a16")
ratios = {name: [] for name in refs}
for label in ("qkv-merged", "o", "gateup-merged", "down"):
    rows = [json.loads((root / f"{label}-fair-run{run}.json").read_text()) for run in (1, 2)]
    medians = {n: statistics.median([s for row in rows for s in row["timing_us"][n]["samples"]]) for n in rows[0]["timing_us"]}
    for name in refs:
        ratios[name].append(medians[name] / medians["emmy"])
    print(label, medians)
for name, values in ratios.items():
    print(name, "geomean", math.prod(values) ** (1 / len(values)))
PY
```

## Scope

These are projection results: the activation quantize and the matmul, without the RMSNorm that precedes them in a
model, RoPE, attention, or the MLP nonlinearity. The schedules are pinned per shape at the measured widths.
