"""Native Qwen3 preparation rejects unsupported models before CUDA work."""

import pytest

from emmy.serving.native.prepare import validate_model
from tests.serving.helpers import qwen3_model


def test_configuration_rejections():
    model = qwen3_model(2).half()
    validate_model(model, 8)
    model.config.quantization_config = {"quant_method": "fp8"}
    with pytest.raises(ValueError, match="unquantized"):
        validate_model(model, 8)
    del model.config.quantization_config
    for length in (0, 65, 4097):
        with pytest.raises(ValueError, match="context"):
            validate_model(model, length)
    model.float()
    with pytest.raises(ValueError, match="FP16"):
        validate_model(model, 8)
    model.half()
    model.config.layer_types[0] = "sliding_attention"
    with pytest.raises(ValueError, match="sliding"):
        validate_model(model, 8)


def test_native_cli_modes_reject_unsupported_sampling():
    import argparse

    from emmy.commands.generate import handle_generate, register_generate_command

    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    for mode in ("--native-pack", "--export-native"):
        args = parser.parse_args(["generate", "unused", mode, "artifact", "--top-k", "5"])
        with pytest.raises(ValueError, match="top-k"):
            handle_generate(args)
    args = parser.parse_args(["generate", "unused", "--native-pack", "artifact", "--max-new-tokens", "-1"])
    with pytest.raises(ValueError, match="nonnegative"):
        handle_generate(args)


def test_native_only_arguments_fail_before_loading_a_model():
    import argparse

    from emmy.commands.generate import handle_generate, register_generate_command

    parser = argparse.ArgumentParser()
    register_generate_command(parser.add_subparsers())
    for options in (["--capture"], ["--context-length", "8"], ["--golden", "unused"], ["--strict-evidence"]):
        with pytest.raises(ValueError, match="require"):
            handle_generate(parser.parse_args(["generate", "unused", *options]))


@pytest.mark.parametrize(
    "temperature,top_p,seed", [(-1, 1, 0), (float("nan"), 1, 0), (1, 0, 0), (1, 1.1, 0), (1, 1, -1), (1, 1, 2**64), (1, 1, True)]
)
def test_invalid_sampling_controls(temperature, top_p, seed):
    from emmy.serving.native.client import validate_sampling

    with pytest.raises(ValueError):
        validate_sampling(temperature, top_p, seed)
