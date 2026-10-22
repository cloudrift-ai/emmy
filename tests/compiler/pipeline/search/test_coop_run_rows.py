"""A width-1 product's schedule pool offers ``coop/v<n>`` runs, and the prior tells a run from its plain band.

A run is the same threads and the same split as its plain ``coop`` band, so before the featurizer read it the two
scored alike and the catalog kept runs out (a tie dropped reproduction nodes). The run rows are where an fp8 or f16
weight row reads as one vector load per lane, the fastest rows of every width-1 product measured on the V100.
"""

from __future__ import annotations

from emmy.compiler.context import Context
from emmy.compiler.ir.schedule import Reduce, Work
from emmy.compiler.ir.schedule.catalog import COOP_RUNS
from emmy.compiler.pipeline.knob import family_value
from emmy.compiler.pipeline.search.autotune import schedule_space
from emmy.compiler.pipeline.search.features import knob_features
from tests.compiler.passes.test_fp8_coop_run import _graph


def test_a_width_one_pool_offers_every_run_of_each_plain_band():
    _, rows = schedule_space(_graph(), Context.from_target((7, 0)))
    plans = [Reduce.parse(family_value(row, "REDUCE"), Work.parse(row.get("WORK", ""))) for row in rows if family_value(row, "REDUCE")]
    plain = {plan.coop for plan in plans if plan.coop > 1 and plan.reg == 1 and plan.coop_columns == 1 and not plan.coop_transposed}
    runs = {(plan.coop, plan.coop_columns) for plan in plans if plan.coop_columns > 1 and not plan.coop_transposed}
    assert plain, "the pool offers a cooperative band"
    assert runs == {(coop, run) for coop in plain for run in COOP_RUNS}


def test_the_featurizer_tells_a_run_from_its_plain_band():
    plain = knob_features({"WORK": "t256", "REDUCE": "coop"})
    runs = [knob_features({"WORK": "t256", "REDUCE": f"coop/v{run}"}) for run in COOP_RUNS]
    assert plain["D_reduce_run"] == 0.0
    assert [feats["D_reduce_run"] for feats in runs] == [1.0, 2.0, 3.0, 4.0]
    assert all(
        {k: v for k, v in feats.items() if k != "D_reduce_run"} == {k: v for k, v in plain.items() if k != "D_reduce_run"} for feats in runs
    )
