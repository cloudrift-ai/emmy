"""Cold benchmark CLI selections retain the per-kernel timing boundary."""

import pytest

from emmy.commands.run import _resolve_backends
from emmy.compiler.pipeline.search.pins import pinned_knobs


def test_cold_cache_defaults_to_emmy_and_refuses_peer_forward_timings(monkeypatch):
    monkeypatch.delenv("EMMY_BENCH_BACKENDS", raising=False)
    with pinned_knobs({"COLD_CACHE": True}):
        assert _resolve_backends(None) == {"emmy"}
        assert _resolve_backends("emmy") == {"emmy"}
        for peer in ("eager", "tcompile"):
            with pytest.raises(ValueError, match="internal cache reuse"):
                _resolve_backends(peer)
    with pinned_knobs({"COLD_CACHE": False}):
        assert _resolve_backends(None) == {"emmy", "eager"}


@pytest.mark.parametrize(
    "args,reason",
    [(["--cold-cache"], "requires --bench"), (["--cold-cache", "--bench", "--record", "--golden", "absent.json"], "whole-row comparisons")],
)
def test_cold_cache_refuses_incompatible_cli_modes_before_work(run_cli, args, reason):
    code, stdout, stderr = run_cli("run", *args)
    assert code == 2
    assert reason in stdout + stderr
