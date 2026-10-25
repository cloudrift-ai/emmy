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


def options(arguments):
    """Parse only the supported native options; never silently forward a vLLM option."""
    from emmy.serving.native.prepare import MAX_CONTEXT
    from emmy.serving.release import DTYPES, KV_CACHE_DTYPES

    parser = argparse.ArgumentParser(prog="emmy serve --native", allow_abbrev=False)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--revision")
    parser.add_argument("--runner", default="auto", help="vLLM's runner; the native server generates only")
    parser.add_argument("--max-model-len", type=int, help="Context (default: the checkpoint's window, or a reused pack's)")
    parser.add_argument("--page-tokens", type=int, help="Tokens of KV cache per page (default: one page spanning the context)")
    parser.add_argument("--kv-cache-dtype", default="auto", choices=tuple(KV_CACHE_DTYPES), help="KV cache dtype (default: the model's)")
    parser.add_argument("--native-pack", type=Path)
    parser.add_argument("--prefill-size", type=int, help="Native export chunk width (default: 16; 1 disables chunking)")
    parser.add_argument("--dtype", default="auto", choices=("auto", *DTYPES), help="Model dtype (default: the checkpoint's)")
    args = parser.parse_args(arguments)
    if not 1 <= (args.max_model_len or 1) <= MAX_CONTEXT or not 1 <= args.port <= 65535:
        raise ValueError(f"native context must be 1–{MAX_CONTEXT} and port must be 1–65535")
    if args.prefill_size is not None and (args.native_pack or not 1 <= args.prefill_size <= MAX_CONTEXT):
        raise ValueError(f"prefill size requires preparation and must be 1–{MAX_CONTEXT}")
    return args


def prepare(model, revision, root, context, golden, strict, *, page_tokens=None, prefill_size=None, dtype="auto", kv_cache_dtype="auto"):
    """Export weights and the same checkpoint's tokenizer/template into one serving bundle; ``context`` defaults to the
    checkpoint's window."""
    from transformers import AutoConfig, AutoTokenizer

    from emmy.serving.native.prepare import default_context, export_checkpoint

    tokenizer = AutoTokenizer.from_pretrained(model, revision=revision)
    if not tokenizer.is_fast or not isinstance(tokenizer.chat_template, str):
        raise ValueError("native serving requires a fast tokenizer and a single checkpoint chat template")
    context = context or default_context(AutoConfig.from_pretrained(model, revision=revision))
    export_checkpoint(
        model,
        revision,
        root,
        dtype=dtype,
        golden=golden,
        strict=strict,
        context_length=context,
        page_tokens=page_tokens,
        prefill_size=prefill_size,
        kv_cache_dtype=kv_cache_dtype,
    )
    tokenizer.backend_tokenizer.save(str(root / "tokenizer.json"))
    (root / "chat_template.jinja").write_text(tokenizer.chat_template)
    (root / "serving.json").write_text(json.dumps({"model": model, "revision": revision, "context_length": context}))


def command(model, opts, root, executable="emmy-server"):
    """Build the native argv without importing compiler or model dependencies; the server takes the pack's context
    when none is named."""
    context = [] if opts.max_model_len is None else ["--max-model-len", str(opts.max_model_len)]
    return [executable, "--artifact", str(root), "--model", model, "--host", opts.host, "--port", str(opts.port), *context]


def launch(args, arguments):
    """Select an existing bundle or prepare one, then replace Python with the native server."""
    from emmy import config
    from emmy.commands.serve import _child_env, _serve_and_bench, _vllm_bin, build_bench_cmd, serving_runner
    from emmy.compiler.loader.safetensors import split_revision

    if args.stock:
        raise ValueError("--native is incompatible with --stock")
    opts = options(arguments)
    model, pinned = split_revision(args.model)
    if pinned and opts.revision and pinned != opts.revision:
        raise ValueError("conflicting model revisions")
    revision = opts.revision or pinned
    # The runner probe reads the pinned checkpoint, so the pin reaches it as vLLM's own flag.
    probe = arguments if opts.revision or not pinned else [*arguments, "--revision", pinned]
    if serving_runner(model, probe) != "generate":
        raise ValueError("--native serves a generate runner only; pass --runner generate for a checkpoint vLLM would pool")
    preparation = (opts.page_tokens is not None, opts.dtype != "auto", opts.kv_cache_dtype != "auto", config.gen_experts_gpu() is not None)
    if opts.native_pack and (args.golden or args.strict_evidence or any(preparation)):
        raise ValueError(
            "golden, strict evidence, page size, dtype, KV cache dtype and EMMY_GEN_EXPERTS_GPU apply to preparation, not an existing pack"
        )
    root = opts.native_pack or Path(tempfile.gettempdir()) / "emmy-native-prepare"
    serve = command(model, opts, root)
    bench = build_bench_cmd(
        model,
        port=str(opts.port),
        max_concurrency=args.max_concurrency if args.max_concurrency is not None else 1,
        num_prompts=args.num_prompts,
        random_input_len=args.random_input_len,
        seed=args.bench_seed,
        generate=True,
        random_output_len=args.random_output_len,
    )
    if args.dry_run:
        if not opts.native_pack:
            logger.info(
                "Prepare native artifact: model=%s revision=%s context=%s page_tokens=%s prefill_size=%s dtype=%s kv_cache_dtype=%s "
                "golden=%s strict=%s",
                model,
                revision,
                opts.max_model_len or "the checkpoint's window",
                opts.page_tokens,
                opts.prefill_size if opts.prefill_size is not None else "default",
                opts.dtype,
                opts.kv_cache_dtype,
                args.golden,
                args.strict_evidence,
            )
        logger.info("%s", shlex.join(serve))
        if args.bench:
            logger.info("%s", shlex.join(bench))
        return
    binary = shutil.which("emmy-server")
    if not binary:
        raise FileNotFoundError("emmy-server is not installed; install the matching native binary package before serving")
    if opts.native_pack:
        metadata = json.loads((root / "serving.json").read_text())
        if metadata["model"] != model or metadata["revision"] != revision or (opts.max_model_len or 0) > metadata["context_length"]:
            raise ValueError("native pack model, revision, or context does not match the request")
    else:
        # Export publishes a fresh directory. Keep the resulting bundle for deliberate reuse.
        root = Path(tempfile.mkdtemp(prefix="emmy-native-")) / "artifact"
        prepare(
            model,
            revision,
            root,
            opts.max_model_len,
            args.golden,
            args.strict_evidence,
            page_tokens=opts.page_tokens,
            prefill_size=opts.prefill_size,
            dtype=opts.dtype,
            kv_cache_dtype=opts.kv_cache_dtype,
        )
        logger.info("Prepared native serving artifact: %s", root)
    serve = command(model, opts, root.resolve(), binary)
    env = _child_env()
    if args.bench:
        bench[0] = _vllm_bin()
        _serve_and_bench(serve, bench, str(opts.port), env=env, health_timeout_s=args.health_timeout)
    else:
        os.execve(binary, serve, env)
