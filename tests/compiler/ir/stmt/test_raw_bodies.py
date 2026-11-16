"""The raw bodies of a corpus lowering, and the normalization rules stated over them.

A raw body is what a loop op is constructed over, before normalization; a stored kernel is what one of them
normalizes to. The rules a port of normalization has to keep are stated over these bodies: the normal form of a raw
body is final, and the stored kernel keeps its identity.
"""

import pytest

from emmy.compiler.ir.stmt import normalize_body, parse_body
from tests.compiler.realization import helpers

#: Cases across the families whose raw bodies the rules run over; each lowers in well under a second.
CASES = [
    "attention/sdpa-hd64-wgmma-value-sm90.json",
    "attention/sdpa-hd128-softmax-v-mma.json",
    "fused/gdn-recurrence-state-kv.json",
    "fused/nvfp4-gate-up-requant-place-cut.json",
    "matmul/bf16-mma-sm120.json",
    "matmul/f16-mma-f16acc-tma.json",
    "reduce/computed-provider-coop-t-sm70.json",
    "reduce/cross-cta-sum-kernel.json",
    "reduce/online-softmax-4x128.json",
    "reduce/rms-norm-cut-sweep-work.json",
    "scan/f32-cumsum-warp-prefix.json",
]


@pytest.mark.parametrize("case_path", CASES)
def test_the_raw_bodies_of_a_lowering_normalize_to_their_stored_kernel(case_path):
    case = helpers.load_case(helpers.CASES_DIR / case_path)
    raw = helpers.raw_bodies(case)
    assert raw
    normal = [normalize_body(body) for body in raw]
    for body in normal:
        # Computed afresh from the text: a body already in normal form answers ``normalize_body`` from its mark.
        assert normalize_body(parse_body(str(body))) == body, "the normal form is final"
    identities = {body.identity(structural=False).key for body in normal}
    stored = case.target.op().identity_key(structural=False)
    assert stored in identities, "the stored kernel is what one raw body normalizes to"
