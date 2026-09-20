"""Native Qwen3 preparation rejects unsupported models before CUDA work."""

import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from emmy.serving.native.prepare import validate_model


def tiny_model():
    torch.manual_seed(71)
    config = Qwen3Config(vocab_size=32, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                        num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=32)
    return Qwen3ForCausalLM(config).half().eval()


def test_configuration_rejections():
    model = tiny_model()
    validate_model(model, 8)
    for length in (0, 33, 4097):
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
        args = parser.parse_args(["generate", "unused", mode, "artifact", "--temperature", "0.5"])
        with pytest.raises(ValueError, match="greedy"):
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
