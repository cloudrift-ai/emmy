"""The existing run command accepts artifact comparisons without tracing a model."""

import argparse
import json
from unittest.mock import AsyncMock

import pytest

from emmy.commands.run import handle_run, register_run_command


def parse(flags):
    parser = argparse.ArgumentParser()
    register_run_command(parser.add_subparsers())
    return parser.parse_args(["run", *flags])


def test_run_pack_delegates_without_compilation(tmp_path, monkeypatch):
    from emmy.compiler.backend import native

    result = {"format": 1, "rows": []}
    benchmark = AsyncMock(return_value=result)
    monkeypatch.setattr(native, "benchmark_pack", benchmark)
    output = tmp_path / "result.json"
    handle_run(parse(["--pack", "bundle", "--bench", "--warmup", "2", "--iters", "7", "--json", str(output)]))
    benchmark.assert_awaited_once_with("bundle", warmup=2, iterations=7)
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize(
    "flags",
    [
        [],
        ["--bench"],
        *[
            ["--bench", "--json", "result.json", *extra]
            for extra in (["--code", "x"], ["--layer", "0"], ["--nvcc-flags", ""], ["--adapter", "dit"], ["--seed", "1"])
        ],
    ],
)
def test_run_pack_rejects_incomplete_or_mixed_modes(flags):
    with pytest.raises(SystemExit) as error:
        handle_run(parse(["--pack", "bundle", *flags]))
    assert error.value.code == 2
