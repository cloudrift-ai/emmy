"""Native CLI routing rejects unsupported engine settings before model loading."""

import argparse

import pytest

from emmy.commands.serve import _add_own_flags, _split_own_flags
from emmy.serving.native.launch import command, launch, options
from emmy.serving.native.prepare import MAX_CONTEXT


def arguments(*flags):
    parser = argparse.ArgumentParser()
    _add_own_flags(parser, suppress_defaults=False)
    args = parser.parse_args([])
    args.model = "Qwen/Qwen3-0.6B"
    args.vllm_args = list(flags)
    forwarded = _split_own_flags(args)
    return args, forwarded


def test_native_probes_the_runner_of_the_pinned_revision(monkeypatch):
    """The pin in ``MODEL@rev`` reaches the runner probe as ``--revision`` before any config is read."""
    seen = []

    def runner(model, vllm_args):
        seen.append((model, list(vllm_args)))
        return "generate"

    monkeypatch.setattr("emmy.commands.serve.serving_runner", runner)
    args, forwarded = arguments("--native", "--dry-run")
    args.model = "Qwen/Qwen3-0.6B@abc"
    launch(args, forwarded)
    assert seen == [("Qwen/Qwen3-0.6B", ["--revision", "abc"])]


def test_native_dry_run(caplog):
    args, forwarded = arguments(
        "--native",
        "--runner",
        "generate",
        "--dry-run",
        "--revision",
        "pinned",
        "--golden",
        "working.json",
        "--strict-evidence",
        "--port",
        "8123",
        "--prefill-size",
        "16",
        "--kv-cache-dtype",
        "fp8",
        "--speculative-config",
        '{"method": "mtp", "num_speculative_tokens": 2}',
    )
    with caplog.at_level("INFO"):
        launch(args, forwarded)
    assert "revision=pinned" in caplog.text
    assert "drafts=2" in caplog.text
    assert "strict=True" in caplog.text
    assert "prefill_size=16" in caplog.text
    assert "kv_cache_dtype=fp8" in caplog.text
    # The context defaults to the checkpoint's window, which the server reads from the pack.
    assert "context=the checkpoint's window" in caplog.text
    assert "emmy-server" in caplog.text and "--max-model-len" not in caplog.text
    assert "--port 8123" in caplog.text


@pytest.mark.parametrize(
    "flags", [("--stock", "--runner", "generate"), ("--runner", "pooling"), ("--runner", "generate", "--revision", "b")]
)
def test_native_rejects_incompatible_modes(flags):
    args, forwarded = arguments("--native", "--dry-run", *flags)
    args.model += "@a"
    with pytest.raises(ValueError):
        launch(args, forwarded)


@pytest.mark.parametrize("flags", [["--tensor-parallel-size", "2"], ["--dtype", "float32"], ["--enforce-eager"]])
def test_native_rejects_forwarded_engine_options(flags):
    with pytest.raises(SystemExit):
        options(flags)


def test_native_command_and_capacity():
    opts = options(["--native-pack", "/tmp/prepared", "--max-model-len", "128"])
    assert command("model", opts, opts.native_pack)[-2:] == ["--max-model-len", "128"]
    assert options(["--max-model-len", str(MAX_CONTEXT)]).max_model_len == MAX_CONTEXT
    with pytest.raises(ValueError):
        options(["--max-model-len", str(MAX_CONTEXT + 1)])


def test_native_page_size_divides_the_context(caplog):
    """The page size is a preparation choice: it must divide the context, it is logged with the
    other preparation settings, and it cannot be applied to an already prepared pack."""
    assert options(["--max-model-len", "128", "--page-tokens", "16"]).page_tokens == 16
    args, forwarded = arguments("--native", "--runner", "generate", "--dry-run", "--page-tokens", "16")
    with caplog.at_level("INFO"):
        launch(args, forwarded)
    assert "page_tokens=16" in caplog.text
    for flag, value in (("--page-tokens", "16"), ("--kv-cache-dtype", "fp8")):
        args, forwarded = arguments("--native", "--runner", "generate", "--dry-run", "--native-pack", "/tmp/prepared", flag, value)
        with pytest.raises(ValueError):
            launch(args, forwarded)


def test_a_prepared_pack_keeps_its_expert_budget(monkeypatch):
    """The expert budget, like the page size, is a preparation choice a reused pack cannot change."""
    monkeypatch.setenv("EMMY_GEN_EXPERTS_GPU", "4")
    args, forwarded = arguments("--native", "--runner", "generate", "--dry-run", "--native-pack", "/tmp/prepared")
    with pytest.raises(ValueError, match="EMMY_GEN_EXPERTS_GPU"):
        launch(args, forwarded)


@pytest.mark.parametrize(
    "flags", [["--prefill-size", "0"], ["--prefill-size", str(MAX_CONTEXT + 1)], ["--prefill-size", "16", "--native-pack", "pack"]]
)
def test_prefill_size_requires_preparation(flags):
    with pytest.raises(ValueError, match="prefill size"):
        options(flags)


@pytest.mark.parametrize(
    "config",
    ['{"method": "ngram", "num_speculative_tokens": 2}', '{"method": "mtp", "num_speculative_tokens": -1}', '{"method": "mtp"}', "[2]"],
)
def test_native_drafts_with_an_mtp_head_only(config):
    """Native drafts with the checkpoint's multi-token-prediction head, a count of tokens it prepares the pack with."""
    assert options(["--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}']).drafts == 2
    assert options([]).drafts == 0
    with pytest.raises(ValueError, match="draft"):
        options(["--speculative-config", config])
    with pytest.raises(ValueError, match="preparation"):
        options(["--native-pack", "pack", "--speculative-config", '{"method": "mtp", "num_speculative_tokens": 2}'])
