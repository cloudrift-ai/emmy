"""Prepare a checkpoint-owned native serving artifact and launch prebuilt Rust binaries."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)
DEFAULT_CONTEXT = 4096


def options(arguments):
    """Parse only the supported native options; never silently forward a vLLM option."""
    parser = argparse.ArgumentParser(prog="emmy serve --native", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--revision")
    parser.add_argument("--max-model-len", type=int, default=DEFAULT_CONTEXT)
    parser.add_argument("--native-pack", type=Path)
    args = parser.parse_args(arguments)
    if not 1 <= args.max_model_len <= DEFAULT_CONTEXT or not 1 <= args.port <= 65535:
        raise ValueError("native context must be 1–4096 and port must be 1–65535")
    return args


def prepare(model, revision, root, context, golden, strict):
    """Export weights and the same checkpoint's tokenizer/template into one serving bundle."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from emmy import config
    from emmy.compiler.backend.gpu_lock import gpu_lock
    from emmy.serving.native.prepare import export_model

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
        raise ValueError("native serving requires a fast tokenizer and a single checkpoint chat template")
    lm = AutoModelForCausalLM.from_pretrained(model, revision=revision, dtype=torch.float16).eval().cpu()
    eos = lm.generation_config.eos_token_id
    eos = [eos] if isinstance(eos, int) else (eos or [])
    with gpu_lock(), config.golden_file_override(golden), config.strict_evidence_override(strict):
        export_model(lm, root, context_length=context, eos_ids=eos)
    tokenizer.backend_tokenizer.save(str(root / "tokenizer.json"))
    (root / "chat_template.jinja").write_text(tokenizer.chat_template)
    (root / "serving.json").write_text(json.dumps({"model": model, "revision": revision, "context_length": context}))


def command(model, opts, root, executable="emmy-server"):
    """Build the native argv without importing compiler or model dependencies."""
    return [executable, "--artifact", str(root), "--model", model, "--host", opts.host, "--port", str(opts.port),
            "--max-model-len", str(opts.max_model_len)]


def launch(args, arguments):
    """Select an existing bundle or prepare one, then replace Python with the native server."""
    from emmy.commands.serve import _child_env, _serve_and_bench, _vllm_bin, build_bench_cmd
    from emmy.compiler.loader.safetensors import split_revision

    if not args.generate or args.stock:
        raise ValueError("--native requires --generate and is incompatible with --stock")
    opts = options(arguments)
    model, pinned = split_revision(args.model)
    if pinned and opts.revision and pinned != opts.revision:
        raise ValueError("conflicting model revisions")
    revision = opts.revision or pinned
    if opts.native_pack and (args.golden or args.strict_evidence):
        raise ValueError("golden and strict evidence apply to preparation, not an existing native pack")
    root = opts.native_pack or Path(tempfile.gettempdir()) / "emmy-native-prepare"
    serve = command(model, opts, root)
    bench = build_bench_cmd(model, port=str(opts.port), max_concurrency=args.max_concurrency, num_prompts=args.num_prompts,
                           random_input_len=args.random_input_len, seed=args.bench_seed, generate=True,
                           random_output_len=args.random_output_len)
    if args.dry_run:
        if not opts.native_pack:
            logger.info("Prepare native artifact: model=%s revision=%s context=%d golden=%s strict=%s",
                        model, revision, opts.max_model_len, args.golden, args.strict_evidence)
        logger.info("%s", shlex.join(serve))
        if args.bench:
            logger.info("%s", shlex.join(bench))
        return
    binary = shutil.which("emmy-server")
    if not binary:
        raise FileNotFoundError("emmy-server is not installed; install the matching native binary package before serving")
    if opts.native_pack:
        metadata = json.loads((root / "serving.json").read_text())
        if metadata["model"] != model or metadata["revision"] != revision or opts.max_model_len > metadata["context_length"]:
            raise ValueError("native pack model, revision, or context does not match the request")
    else:
        # Export publishes a fresh directory. Keep the resulting bundle for deliberate reuse.
        root = Path(tempfile.mkdtemp(prefix="emmy-native-")) / "artifact"
        prepare(model, revision, root, opts.max_model_len, args.golden, args.strict_evidence)
        logger.info("Prepared native serving artifact: %s", root)
    serve = command(model, opts, root.resolve(), binary)
    env = _child_env()
    if args.bench:
        bench[0] = _vllm_bin()
        _serve_and_bench(serve, bench, str(opts.port), env=env, health_timeout_s=args.health_timeout)
    else:
        os.execve(binary, serve, env)
