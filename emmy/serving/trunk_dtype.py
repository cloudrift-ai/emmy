"""Map the engine's dtype to the serving trunk's dtype."""

import logging

import torch

logger = logging.getLogger(__name__)

_TRUNK_DTYPE = {torch.float16: "float16", torch.bfloat16: "bfloat16", torch.float32: "float32"}


def _trunk_dtype_str(torch_dtype, *, allow_bf16=True) -> str:
    dtype_str = _TRUNK_DTYPE.get(torch_dtype)
    if dtype_str is None or (dtype_str == "bfloat16" and not allow_bf16):
        logger.warning("[serving] --dtype %s unsupported by the emmy trunk; computing in float16", torch_dtype)
        return "float16"
    return dtype_str
