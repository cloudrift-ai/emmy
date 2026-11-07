"""The closed sets the crate declares agree with the Python registries that name the same things."""

from __future__ import annotations

from emmy.emmy_ir import DTYPES

import emmy.compiler.dtype as dtype_registry


def test_the_crate_knows_every_registered_dtype_and_no_other() -> None:
    assert set(DTYPES) == set(dtype_registry._BY_NAME)
