"""Phase-1 prototype: dynamic FP4 down projection versus serial and cooperative CUDA."""

import argparse
import hashlib
import logging
import os
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)
K, N = 17408, 5120
NAME = "k_linear_00520c"
REALIZATION = "post-sym-dense-full@nvfp4.k_linear_00520c.dynamic.fm.0b40fd02b533"
GOLDEN = "recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json"
VARIANTS = {
    "native": ("TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d2/smem-async,WORK=w4x1", 128),
    "serial": ("TILE=,STAGE=,REDUCE=,WORK=", 256),
    "coop": ("TILE=,STAGE=,REDUCE=coop,WORK=t512", 512),
}
GUARD = 256
SENTINEL = 0x7FC0


def bf16(value):
    """Round float32 to bfloat16 with round-to-nearest, ties-to-even."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def floats(bits):
    """Interpret bfloat16 bits as float32 without a framework conversion."""
    return (bits.astype(np.uint32) << 16).view(np.float32)


def compile_variant(label, scratch):
    """Compile this exact golden body once, keeping num_tokens as a runtime argument."""
    knobs, block = VARIANTS[label]
    source = scratch / f"{label}.cu"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "emmy.emmy",
            "compile",
            "--golden",
            GOLDEN,
            "--realization",
            REALIZATION,
            "--target",
            "sm_120",
            "--ir",
            "cuda",
            "-o",
            str(source),
        ],
        env=dict(os.environ, EMMY_KNOBS=knobs),
        check=True,
        timeout=115,
    )
    text = source.read_text()
    text = text[text.index("#include") :]
    assert f"__launch_bounds__({block})" in text and "int num_tokens)" in text
    assert ("mma.sync.aligned.m16n8k64" in text) == (label == "native")
    source.write_text(text)
    cubin = source.with_suffix(".cubin")
    subprocess.run(["nvcc", "--cubin", "-arch=sm_120a", "--use_fast_math", str(source), "-o", str(cubin)], check=True, timeout=115)
    module = cp.RawModule(path=str(cubin))
    logger.info("%s compiled once source_sha256=%s", label, hashlib.sha256(text.encode()).hexdigest())
    return module.get_function(NAME)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/codex-1/sym-fp4-bench"))
    parser.add_argument("--tokens", type=int, nargs="+", default=[1, 17, 40, 63])
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if args.iterations < 20 or args.warmup < 1:
        parser.error("at least 20 measured iterations and one warmup required")
    if any(tokens < 1 for tokens in args.tokens):
        parser.error("token counts must be positive")
    args.scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    tokens_max = max(args.tokens)
    lut = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    byte = np.arange(256, dtype=np.uint16)
    table = bf16(np.stack([lut[byte & 15], lut[byte >> 4]], axis=1))
    wbits = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
    xbits = rng.integers(0, 256, (tokens_max, K // 2), dtype=np.uint8)
    # All eight e4m3 values have nonzero mantissas; no scale is a power of two.
    scale_codes = np.array([33, 35, 37, 39, 41, 43, 45, 47], dtype=np.uint8)
    ws = rng.choice(scale_codes, (N, K // 16))
    xs = rng.choice(scale_codes, (tokens_max, K // 16))
    residual = bf16(rng.normal(size=(tokens_max, N)).astype(np.float32))

    def decode(bits, scales):
        codes = np.stack([bits & 15, bits >> 4], axis=-1).reshape(bits.shape[0], K)
        exponent = ((scales >> 3) & 15).astype(np.int32) - 7
        scale = np.ldexp(1 + (scales & 7).astype(np.float64) / 8, exponent)
        # These scale/global-scale/code products fit BF16 exactly. Float64 matmul is
        # independent of the compiler's CUDA reduction and packed-instruction path.
        return lut[codes].astype(np.float64) * np.repeat(scale, 16, axis=1) * 0.125

    x = decode(xbits, xs)
    reference = np.empty((tokens_max, N), dtype=np.float32)
    for start in range(0, N, 128):
        reference[:, start : start + 128] = x @ decode(wbits[start : start + 128], ws[start : start + 128]).T
    reference = floats(bf16(floats(bf16(reference)) + floats(residual)))
    inputs = tuple(
        cp.asarray(a) for a in (np.array([0.125], np.float32), np.array([0.125], np.float32), ws, xs, wbits, xbits, table, table, residual)
    )
    device = cp.cuda.runtime.getDeviceProperties(0)["name"]
    logger.info("device=%s N=%d K=%d tokens=%s seed=%d scale_codes=%s", device, N, K, args.tokens, args.seed, scale_codes.tolist())
    logger.info("5080 timings are sanity checks only; 5090 measurements are required for the prototype decision.")
    kernels = {label: compile_variant(label, args.scratch) for label in args.variants}
    for tokens in args.tokens:
        for label, kernel in kernels.items():
            _, block = VARIANTS[label]
            # Fixed launch geometry of these three prototype schedules, not a general launcher.
            grid = ((tokens + 63) // 64) * (N // 16) if label == "native" else tokens * N // (256 if label == "serial" else 1)
            storage = cp.full(tokens * N + 2 * GUARD, SENTINEL, dtype=cp.uint16)
            out = storage[GUARD:-GUARD]

            def launch(kernel=kernel, grid=grid, block=block, out=out, tokens=tokens):
                kernel((grid,), (block,), (*inputs, out, np.int32(tokens)))

            launch()
            cp.cuda.get_current_stream().synchronize()
            actual = floats(cp.asnumpy(out)).reshape(tokens, N)
            np.testing.assert_allclose(actual, reference[:tokens], rtol=1e-3, atol=1e-3)
            np.testing.assert_array_equal(cp.asnumpy(storage[:GUARD]), SENTINEL)
            np.testing.assert_array_equal(cp.asnumpy(storage[-GUARD:]), SENTINEL)
            logger.info(
                "tokens=%d %s correctness=PASS max_abs=%g nonzero=%d/%d guards=PASS",
                tokens,
                label,
                np.max(np.abs(actual - reference[:tokens])),
                np.count_nonzero(actual),
                actual.size,
            )
            if args.check_only:
                continue
            for _ in range(args.warmup):
                launch()
            start, end = cp.cuda.Event(), cp.cuda.Event()
            timings = []
            for _ in range(args.iterations):
                start.record()
                launch()
                end.record()
                end.synchronize()
                timings.append(cp.cuda.get_elapsed_time(start, end) * 1000)
            logger.info("tokens=%d %s median_us=%.3f iterations=%d", tokens, label, np.median(timings), args.iterations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
