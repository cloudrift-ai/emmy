"""Phase-1-only FP4 unit-N prototype: compile, check NumPy, compare identical inputs."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

import cupy as cp
import numpy as np

logger = logging.getLogger(__name__)
K, N = 17408, 5120
NAME = "k_linear_7f95ba"
REALIZATION = "gdn1-dense-linear@nvfp4.k_linear_7f95ba.m1.fm.ab7077b55776"


def bf16(value):
    """Round float32 to bfloat16 with round-to-nearest, ties-to-even."""
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) >> 16).astype(np.uint16)


def floats(bits):
    return (bits.astype(np.uint32) << 16).view(np.float32)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scratch", type=Path, default=Path("/tmp/codex-2/fp4"))
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.iterations < 20:
        parser.error("at least 20 iterations required")
    args.scratch.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    lut = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    byte = np.arange(256, dtype=np.uint16)
    table = bf16(np.stack([lut[byte & 15], lut[byte >> 4]], axis=1))
    wbits = rng.integers(0, 256, (N, K // 2), dtype=np.uint8)
    xbits = rng.integers(0, 256, (1, K // 2), dtype=np.uint8)
    # Positive finite e4m3 powers of two keep the decoded BF16 operands exact.
    scale_codes = np.array([32, 40, 48], dtype=np.uint8)
    ws = rng.choice(scale_codes, (N, K // 16))
    xs = rng.choice(scale_codes, (1, K // 16))
    residual = bf16(rng.normal(size=N).astype(np.float32))

    def decode(bits, scales):
        codes = np.stack([bits & 15, bits >> 4], axis=-1).reshape(bits.shape[0], K)
        exponent = ((scales >> 3) & 15).astype(np.int32) - 7
        scale = np.ldexp(1 + (scales & 7).astype(np.float64) / 8, exponent)
        return lut[codes].astype(np.float64) * np.repeat(scale, 16, axis=1) * 0.125

    x = decode(xbits, xs)[0]
    reference = np.empty(N, dtype=np.float32)
    for start in range(0, N, 128):
        reference[start : start + 128] = decode(wbits[start : start + 128], ws[start : start + 128]) @ x
    reference = floats(bf16(floats(bf16(reference)) + floats(residual)))
    inputs = tuple(
        cp.asarray(a) for a in (np.array([0.125], np.float32), np.array([0.125], np.float32), ws, xs, wbits, xbits, table, table, residual)
    )
    device = cp.cuda.runtime.getDeviceProperties(0)["name"]
    logger.info("device=%s shape=M1,N%d,K%d seed=%d", device, N, K, args.seed)
    logger.info("5080 timings are sanity checks only; only the 5090 run is performance evidence.")
    variants = (
        ("native", "TILE=mma_m16n8k64_e2m1_f32/f1x2/k4,STAGE=d2/smem-async,WORK=w4x1", 80, 128),
        ("scalar", "WORK=t512,REDUCE=coop-t", 160, 512),
    )
    for label, knobs, grid, block in variants:
        source = args.scratch / f"{label}.cu"
        env = dict(os.environ, EMMY_KNOBS=knobs)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "emmy.emmy",
                "compile",
                "--golden",
                "recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json",
                "--realization",
                REALIZATION,
                "--target",
                "sm_120",
                "--ir",
                "cuda",
                "-o",
                str(source),
            ],
            env=env,
            check=True,
            timeout=115,
        )
        text = source.read_text()
        source.write_text(text[text.index("#include") :])
        cubin = source.with_suffix(".cubin")
        subprocess.run(["nvcc", "--cubin", "-arch=sm_120a", "--use_fast_math", str(source), "-o", str(cubin)], check=True, timeout=115)
        module = cp.RawModule(path=str(cubin))
        kernel = module.get_function(NAME)
        out = cp.full(N, 0x7FC0, dtype=cp.uint16)

        def launch(kernel=kernel, grid=grid, block=block, out=out):
            kernel((grid,), (block,), (*inputs, out))

        launch()
        cp.cuda.get_current_stream().synchronize()
        actual = floats(cp.asnumpy(out))
        np.testing.assert_allclose(actual, reference, rtol=1e-3, atol=1e-3)
        logger.info("%s correctness PASS max_abs=%g nonzero=%d/%d", label, np.max(np.abs(actual - reference)), np.count_nonzero(actual), N)
        for _ in range(10):
            launch()
        start, end = cp.cuda.Event(), cp.cuda.Event()
        timings = []
        for _ in range(args.iterations):
            start.record()
            launch()
            end.record()
            end.synchronize()
            timings.append(cp.cuda.get_elapsed_time(start, end) * 1000)
        logger.info("%s median_us=%.3f iterations=%d", label, np.median(timings), args.iterations)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
