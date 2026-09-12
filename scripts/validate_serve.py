#!/usr/bin/env python
"""Validate emmy's generative serving of a checkpoint on a real GPU, against HF eager.

The correctness gate of the serving-image release (docker/vllm-emmy-serve): never warm a cache
from a server whose output disagrees with HF, because the disagreement gets baked in.

Runs a **sequential** A/B so a big model still fits one card (HF and emmy are never resident at
once): compute HF fp16 greedy references in a throwaway subprocess (its GPU memory is fully
reclaimed on exit), then start ``emmy serve --generate`` in a subprocess, query
``/v1/completions`` greedily, and compare. Prints a per-prompt side-by-side + a PASS/FAIL on the
first generated token (the strong correctness signal; fp16 numerics between two runtimes can
drift a few tokens in, so a full-text match is not required — eyeball the continuations for
coherence).

    ./venv/bin/python scripts/validate_serve.py --model google/gemma-4-12B-it

Prereqs: the `serving` extra + cupy installed, a CUDA toolkit whose nvcc supports the target arch,
and HF access to the checkpoint (`export HF_TOKEN=...` when gated). The first `emmy serve` boot
compiles every layer — minutes.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPTS = [
    "The capital of France is",
    "The three primary colors are",
    "Water is made of hydrogen and",
    "Once upon a time, in a small village,",
]


def _hf_refs(model: str, prompts: list[str], max_tokens: int) -> list[dict]:
    """HF fp16 greedy references. Run as a subprocess (``--hf-worker``) so its ~24 GB of GPU
    weights are fully freed before emmy takes the card."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emmy.compiler.loader.safetensors import split_revision
    from emmy.compiler.trace.huggingface import load_quantized_twin, quantized_checkpoint_dir

    repo, revision = split_revision(model)
    tok = AutoTokenizer.from_pretrained(repo, revision=revision)
    quant_dir = quantized_checkpoint_dir(model)
    if quant_dir is not None:
        # Packed checkpoints are not runnable HF modules: loading them directly either engages a
        # transformers quantizer or puts the packed carrier into a dense weight. Emmy's established
        # accuracy twin strips that engine declaration and loads the independently decoded values.
        m = load_quantized_twin(quant_dir, torch.float16)
    else:
        m = AutoModelForCausalLM.from_pretrained(repo, revision=revision, dtype=torch.float16)
    m = m.to("cuda").eval()
    refs = []
    for p in prompts:
        ids = tok(p, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = m.generate(**ids, max_new_tokens=max_tokens, do_sample=False)
        gen = out[0][ids.input_ids.shape[1] :].tolist()
        refs.append({"prompt": p, "text": tok.decode(gen)})
    return refs


def _wait_health(port: str, proc: subprocess.Popen, timeout_s: int) -> None:
    url = f"http://localhost:{port}/health"
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"emmy serve exited early (code {proc.returncode}) — check its log above")
        try:
            with urllib.request.urlopen(url, timeout=2):
                return
        except (urllib.error.URLError, ConnectionError, TimeoutError):
            time.sleep(3)
    raise TimeoutError(f"server not healthy within {timeout_s}s")


def _completion(port: str, model: str, prompt: str, max_tokens: int) -> tuple[str, str]:
    """Returns (text, diag) — diag carries finish_reason / token count / first top-logprobs,
    so an empty or wrong completion is diagnosable from the transcript alone."""
    body = json.dumps({"model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 0, "logprobs": 3}).encode()
    req = urllib.request.Request(f"http://localhost:{port}/v1/completions", data=body, headers={"content-type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        resp = json.load(r)
    choice = resp["choices"][0]
    lp = choice.get("logprobs") or {}
    first_top = (lp.get("top_logprobs") or [{}])[0]
    diag = (
        f"finish={choice.get('finish_reason')} completion_tokens={resp.get('usage', {}).get('completion_tokens')} "
        f"first_tokens={lp.get('tokens', [])[:3]!r} first_top={ {k: round(v, 2) for k, v in list(first_top.items())[:3]} }"
    )
    return choice["text"], diag


def _first_token(s: str) -> str:
    parts = s.strip().split()
    return parts[0] if parts else ""


def _serve_invocation(args) -> tuple[list[str], dict[str, str]]:
    """The serving command and environment whose exact shape the reference gates."""
    cmd = [
        args.emmy,
        "serve",
        "--generate",
        args.model,
        "--max-model-len",
        args.max_model_len,
        "--port",
        args.port,
        "--gpu-memory-utilization",
        args.gpu_mem_util,
    ]
    if args.max_num_batched_tokens:
        cmd += ["--max-num-batched-tokens", args.max_num_batched_tokens]
    if args.golden:
        cmd += ["--golden", args.golden]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    env = os.environ.copy()
    if args.decode_bucket is not None:
        env["EMMY_GEN_DECODE_BUCKET"] = str(args.decode_bucket)
    return cmd, env


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF checkpoint to serve + reference.")
    ap.add_argument("--max-tokens", type=int, default=16, help="greedy tokens to generate per prompt.")
    ap.add_argument(
        "--prompt-count",
        type=int,
        default=len(PROMPTS),
        choices=range(1, len(PROMPTS) + 1),
        help="number of built-in prompts to compare.",
    )
    ap.add_argument("--port", default="8000")
    ap.add_argument("--emmy", default="./venv/bin/emmy", help="path to the emmy CLI in the serving venv.")
    ap.add_argument("--max-model-len", default="4096", help="vLLM --max-model-len (smaller ⇒ less KV cache).")
    ap.add_argument(
        "--max-num-batched-tokens",
        default=None,
        help="vLLM --max-num-batched-tokens. This sizes the runner's PER-LAYER activation buffers "
        "(each layer's program retains its own), so lowering it cuts emmy's memory a lot.",
    )
    ap.add_argument(
        "--gpu-mem-util", default="0.9", help="vLLM --gpu-memory-utilization (lower to leave room for the emmy runner's on-GPU weights)."
    )
    ap.add_argument("--health-timeout", type=int, default=1800, help="seconds to wait for first-boot compile.")
    ap.add_argument("--golden", help="golden YAML whose measured routes the serving comparison must deploy.")
    ap.add_argument("--enforce-eager", action="store_true", help="forward vLLM --enforce-eager to the serving subprocess.")
    ap.add_argument(
        "--decode-bucket",
        type=int,
        help="static decode width (sets EMMY_GEN_DECODE_BUCKET for the serving subprocess).",
    )
    ap.add_argument("--hf-worker", action="store_true", help=argparse.SUPPRESS)  # internal: emit HF refs as JSON
    args = ap.parse_args()

    if args.hf_worker:
        print(json.dumps(_hf_refs(args.model, PROMPTS[: args.prompt_count], args.max_tokens)))
        return 0

    print(f"[1/3] HF fp16 greedy references for {args.model} (subprocess; frees the GPU when done)...", flush=True)
    refs = json.loads(
        subprocess.check_output(
            [
                sys.executable,
                __file__,
                "--hf-worker",
                "--model",
                args.model,
                "--max-tokens",
                str(args.max_tokens),
                "--prompt-count",
                str(args.prompt_count),
            ]
        )
    )

    print("[2/3] starting `emmy serve --generate` (first boot compiles every layer — minutes)...", flush=True)
    serve_cmd, serve_env = _serve_invocation(args)
    serve = subprocess.Popen(serve_cmd, env=serve_env)
    try:
        _wait_health(args.port, serve, args.health_timeout)
        print("[3/3] querying emmy /v1/completions and comparing to HF...\n", flush=True)
        matches = 0
        for r in refs:
            got, diag = _completion(args.port, args.model, r["prompt"], args.max_tokens)
            ok = _first_token(got) == _first_token(r["text"])
            matches += ok
            print(f"  {'PASS' if ok else 'FAIL'} | {r['prompt']!r}")
            print(f"       hf  : {r['text']!r}")
            print(f"       emmy: {got!r}")
            print(f"       diag: {diag}")
        n = len(refs)
        print(f"\n{matches}/{n} first-token matches vs HF fp16. Eyeball the continuations above for coherence.")
        return 0 if matches == n else 1
    finally:
        serve.send_signal(signal.SIGINT)
        try:
            serve.wait(timeout=30)
        except subprocess.TimeoutExpired:
            serve.kill()


if __name__ == "__main__":
    raise SystemExit(main())
