"""The closed sets the crate declares agree with the Python registries that name the same things."""

from __future__ import annotations

from emmy.emmy_ir import DTYPES

import emmy.compiler.dtype as dtype_registry
from emmy.compiler.ir.expression import BUILTINS


def test_the_crate_knows_every_registered_dtype_and_no_other() -> None:
    assert set(DTYPES) == set(dtype_registry._BY_NAME)


def test_the_built_ins_are_the_gpu_coordinates() -> None:
    assert {name.split(".")[0] for name in BUILTINS} == {"thread_idx", "block_idx", "block_dim", "grid_dim", "warp_size"}
