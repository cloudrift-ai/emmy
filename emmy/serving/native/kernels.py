"""CUDA source bundled for native Qwen3 artifact preparation."""

from importlib.resources import files

SOURCE = files(__package__).joinpath("kernels.cu").read_text(encoding="utf-8")
