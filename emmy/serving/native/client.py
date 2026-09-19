"""Supervised native generation using binary token files and the existing worker protocol."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from emmy.compiler.backend.native import NativeWorker

DEFAULT_TIMEOUT_SECONDS = 120


async def generate_tokens(pack, prompt_ids, *, max_new_tokens, capture=False, timeout=DEFAULT_TIMEOUT_SECONDS):
    """Run the complete cached loop in Rust; retire the process on failure or timeout."""
    worker = NativeWorker()
    try:
        with tempfile.TemporaryDirectory(prefix="emmy-generation-") as directory:
            prompt = Path(directory) / "prompt.bin"
            output = Path(directory) / "tokens.bin"
            np.asarray(prompt_ids, dtype="<i8").tofile(prompt)
            await worker.run_job({"op": "load_generation", "root": str(Path(pack).resolve())}, wall_timeout_s=timeout)
            await worker.run_job({"op": "generate", "prompt": str(prompt), "max_new_tokens": max_new_tokens,
                                  "capture": capture, "output": str(output)}, wall_timeout_s=timeout)
            return np.fromfile(output, dtype="<i8").tolist()
    finally:
        await worker.aclose()
