"""Regenerate the serving-generation lane's golden — ``python -m tests.serving.regen``.

Builds every runner in :data:`helpers.RUNNERS` once, capturing the graph handed to each plan
compile, and writes them as one inventory completed to **one row per kernel of the set** — the
same ``helpers.complete`` the realization corpus authors its cases with, reading the compile's
realized row per kernel — into the live card's capability file (:func:`helpers.golden_path`).

Run it when a runner shape changes, when a new one joins the table, or when strict evidence starts
reporting a kernel the golden does not decide (a kernel identity or schedule codec moved). Those
reports are the signal that this needs running; they are never something to work around by
loosening the scope.

The two halves also run apart, since only the capture needs the card and the Gated DeltaNet shapes
make both slow (about 80 minutes of compiles each with six workers): ``--capture PATH`` writes the
inventory to ``PATH`` and its graphs' layouts and pins beside it, ``--complete PATH`` completes that
inventory into its card's file, with no GPU.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def _captured_graphs() -> tuple[dict, dict, dict]:
    """Every distinct graph this lane compiles, keyed by a readable name, with the weight layouts each compiled under
    and its runner's cut pins (:data:`helpers.CUTS`).

    The runner shapes come through the plan cache — ``PlanTemplateCache.resolve`` sees each graph on
    its way to the compiler, the one seam every split's plan passes through — and the attention-split
    wrappers are traced directly, since they never reach a runner.

    """
    from emmy.compiler.backend.plan_cache import PlanTemplateCache
    from emmy.compiler.pipeline.search.pins import pinned_knobs
    from emmy.serving.gen_runner import EmmyGenRunner
    from tests.serving import helpers

    graphs: dict = {}
    settled: dict = {}
    pins: dict = {}
    cuts: dict = {}

    class _Capturing(PlanTemplateCache):
        def resolve(self, graph, compile_plan, settled_layouts=()):
            missed = self.misses
            plan = super().resolve(graph, compile_plan, settled_layouts)
            if self.misses > missed:  # a hit is the same binding-neutral graph, already recorded
                name = f"g{len(graphs):03d}"
                graphs[name] = graph.copy()
                settled[name] = dict(settled_layouts)  # the weight layouts the runner compiled it with
                pins[name] = dict(cuts)  # its runner's cut pins, which the completion takes again
            return plan

    # ONE cache across every runner, the way the lane's session fixture shares it: two shapes with
    # a layer program in common contribute it once, and the capture sees each distinct graph once.
    cache = _Capturing()
    for runner_id, (make_model, kwargs) in helpers.RUNNERS.items():
        logger.info("[regen] building %s ...", runner_id)
        before = len(graphs)
        cuts = helpers.CUTS.get(runner_id, {})
        with pinned_knobs(cuts):
            EmmyGenRunner.from_model(make_model(), plan_cache=cache, **kwargs)
        logger.info("[regen]   %s contributed %d new graph(s) (%d total)", runner_id, len(graphs) - before, len(graphs))
    for case_id in helpers.WRAPPERS:
        logger.info("[regen] tracing wrapper %s ...", case_id)
        graphs[f"w-{case_id}"] = helpers.wrapper_graph(case_id)
    return graphs, settled, pins


def _capture(inventory: Path) -> None:
    """Capture every graph into ``inventory``, and the layouts and pins each compiled under beside it."""
    from emmy.compiler.context import Context
    from emmy.compiler.pipeline.search.golden import write_trace_inventories

    graphs, settled, pins = _captured_graphs()
    logger.info("[regen] %d distinct graph(s); writing the inventory", len(graphs))
    inventory.unlink(missing_ok=True)
    write_trace_inventories(graphs, inventory, ctx=Context.probe())
    _sidecar(inventory).write_text(json.dumps({"settled": settled, "pins": pins}, indent=1, sort_keys=True))


def _complete(inventory: Path) -> None:
    """Complete a captured inventory into its card's file (:func:`helpers.golden_path`)."""
    from emmy.compiler.pipeline.search.golden import GoldenFile
    from tests.compiler.realization import helpers as corpus
    from tests.serving import helpers

    captured = json.loads(_sidecar(inventory).read_text())
    document = corpus.complete(GoldenFile.load(inventory), captured["settled"], captured["pins"])
    logger.info("[regen]   %d kernel(s), %d row(s), %d decision(s)", len(document.kernels), len(document.rows), len(document.routing))
    destination = helpers.golden_path(tuple(document.compute_cap))
    destination.parent.mkdir(parents=True, exist_ok=True)
    document.dump(destination, overwrite=True)
    logger.info("[regen] wrote %s", destination)


def _sidecar(inventory: Path) -> Path:
    return inventory.with_name(inventory.name + ".capture.json")


def main(argv: list[str] | None = None) -> int:
    import argparse

    from tests.serving import helpers

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(prog="python -m tests.serving.regen", description=__doc__.split("\n", 1)[0])
    half = parser.add_mutually_exclusive_group()
    half.add_argument("--capture", type=Path, metavar="PATH", help="capture only: write the inventory to PATH")
    half.add_argument("--complete", type=Path, metavar="PATH", help="complete the inventory at PATH, GPU-free")
    args = parser.parse_args(argv)
    if args.capture is not None:
        _capture(args.capture)
    elif args.complete is not None:
        _complete(args.complete)
    else:
        inventory = helpers.golden_path().with_suffix(".inventory.tmp")
        _capture(inventory)
        _complete(inventory)
        inventory.unlink()
        _sidecar(inventory).unlink()
    return 0


if __name__ == "__main__":
    sys.exit(main())
