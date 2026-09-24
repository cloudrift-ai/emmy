"""Native CLI routing rejects unsupported engine settings before model loading."""

import argparse

import pytest

from emmy.commands.serve import _add_own_flags, _split_own_flags
from emmy.serving.native.launch import command, launch, options


def arguments(*flags):
    parser = argparse.ArgumentParser()
    _add_own_flags(parser, suppress_defaults=False)
    args = parser.parse_args([])
    args.model = "Qwen/Qwen3-0.6B"
    args.vllm_args = list(flags)
    forwarded = _split_own_flags(args)
    return args, forwarded


def test_native_dry_run(caplog):
    args, forwarded = arguments(
        "--native", "--generate", "--dry-run", "--revision", "pinned", "--golden", "working.yaml", "--strict-evidence", "--port", "8123"
    )
    with caplog.at_level("INFO"):
        launch(args, forwarded)
    assert "revision=pinned" in caplog.text
    assert "strict=True" in caplog.text
    assert "emmy-server" in caplog.text
    assert "--max-model-len 4096" in caplog.text
    assert "--port 8123" in caplog.text


@pytest.mark.parametrize("flags", [("--stock", "--generate"), (), ("--generate", "--revision", "b")])
def test_native_rejects_incompatible_modes(flags):
    args, forwarded = arguments("--native", "--dry-run", *flags)
    args.model += "@a"
    with pytest.raises(ValueError):
        launch(args, forwarded)


@pytest.mark.parametrize("flags", [["--tensor-parallel-size", "2"], ["--dtype", "bfloat16"], ["--enforce-eager"]])
def test_native_rejects_forwarded_engine_options(flags):
    with pytest.raises(SystemExit):
        options(flags)


def test_native_command_and_capacity():
    opts = options(["--native-pack", "/tmp/prepared", "--max-model-len", "128"])
    assert command("model", opts, opts.native_pack)[-2:] == ["--max-model-len", "128"]
    with pytest.raises(ValueError):
        options(["--max-model-len", "4097"])
