"""Name resolution for the named scalar ops -- which spellings reach which implementation."""

from __future__ import annotations

import numpy as np
import pytest

from emmy.compiler.ir.elementwise import ElementwiseImpl


def test_right_shift_spellings_normalize_to_the_canonical_name():
    """``>>`` and ``torch.bitwise_right_shift`` must reach the one op the renderers know.

    ``right_shift`` is what the CUDA and loop render targets, the torch reference and the statement
    renderer all key on, and what the quantized-weight spellers emit directly. Two other spellings
    arrive from traced expressions and neither resolved: ``a >> b`` traces to ``__rshift__``, absent
    from both numpy and ``_NAME_TO_FN`` so it failed at construction, and ``bitwise_right_shift``
    traces under its own name, which numpy aliases -- so it constructed and then failed to render.

    Both now normalize. That is what lets a packed-int4 decode cone be written as a traced
    expression: until it did, such a cone existed only inside the checkpoint spellers, so a
    miscompilation reachable only through one could not be minimized outside a real checkpoint.
    """
    canonical = ElementwiseImpl("right_shift")
    for spelling in ("__rshift__", "bitwise_right_shift"):
        impl = ElementwiseImpl(spelling)
        assert impl.name == canonical.name == "right_shift"
        assert impl.arity == canonical.arity == 2
        assert int(impl(np.int32(64), np.int32(3))) == 8


def test_an_unaliased_unknown_spelling_still_raises():
    """The alias table is two entries, not a general bitwise sweep: anything else still raises."""
    with pytest.raises(ValueError, match="unknown elementwise op name"):
        ElementwiseImpl("__lshift__")
