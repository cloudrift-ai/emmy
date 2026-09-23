"""Supervised native generation using binary token files and the existing worker protocol."""

from __future__ import annotations

import math
import tempfile
from pathlib import Path

import numpy as np

from emmy.compiler.backend.native import NativeWorker

DEFAULT_TIMEOUT_SECONDS = 120


def validate_sampling(temperature, top_p, seed):
    """Validate native request controls before starting a worker or loading a tokenizer."""
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("temperature must be finite and nonnegative")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    if not isinstance(seed, int) or not 0 <= seed < 2**64:
        raise ValueError("seed must be an unsigned 64-bit integer")


async def generate_tokens(
    pack, prompt_ids, *, max_new_tokens, capture=False, temperature=0.0, top_p=1.0, seed=0, timeout=DEFAULT_TIMEOUT_SECONDS
):
    """Run the complete cached loop in Rust; retire the process on failure or timeout."""
    validate_sampling(temperature, top_p, seed)
    worker = NativeWorker()
    try:
        with tempfile.TemporaryDirectory(prefix="emmy-generation-") as directory:
            prompt = Path(directory) / "prompt.bin"
            output = Path(directory) / "tokens.bin"
            np.asarray(prompt_ids, dtype="<i8").tofile(prompt)
            await worker.run_job({"op": "load_generation", "root": str(Path(pack).resolve())}, wall_timeout_s=timeout)
            await worker.run_job(
                {
                    "op": "generate",
                    "prompt": str(prompt),
                    "max_new_tokens": max_new_tokens,
                    "capture": capture,
                    "output": str(output),
                    "sampling": {"temperature": temperature, "top_p": top_p, "seed": seed},
                },
                wall_timeout_s=timeout,
            )
            return np.fromfile(output, dtype="<i8").tolist()
    finally:
        await worker.aclose()
