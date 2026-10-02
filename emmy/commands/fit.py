"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The fitter entry point: a CatBoost ranker fit over the golden groups of a dataset ``emmy db export`` wrote
(:class:`~emmy.compiler.pipeline.search.dataset.Dataset`, the directory the positional argument names). The
feature view is a projection of the dataset's full featurization, taken here. Any written artifact can be
pointed at with ``EMMY_OFFLINE_FILE`` and A/B'd against the shipped one.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and the full-train artifact at the path the second positional argument names, in the shipped
``weights/schedule.json`` format. The metrics layout (``full_train`` +
a ``cv`` holdout/train/gap block, both carrying ``prior/report.py`` summaries) is documented on
:mod:`emmy.compiler.pipeline.search.prior.fit.cv`, which owns all the fold machinery;
the run itself is :func:`~emmy.compiler.pipeline.search.prior.fit.run.run_fit`. This
module owns the CLI, the trainer wiring, and the file writing.
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

from emmy import storage
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.dataset import DEFAULT_FEATURES, PLACEMENT_FEATURES, Dataset, feature_view, repo_commit
from emmy.compiler.pipeline.search.prior.fit import catboost as fit_catboost
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit.run import run_fit

logger = logging.getLogger(__name__)


def register_fit_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "fit",
        help="Fit the offline prior (a CatBoost ranker over a golden dataset) and cross-validate it, writing a metrics file",
    )
    parser.add_argument("dataset", help="Dataset directory written by `emmy db export`, e.g. _data/dataset.")
    parser.add_argument("--iterations", type=int, default=fit_catboost.CatBoostTrainer.iterations, help="Boosting iterations (trees).")
    parser.add_argument("--depth", type=int, default=fit_catboost.CatBoostTrainer.depth, help="Tree depth.")
    parser.add_argument("--learning-rate", type=float, default=fit_catboost.CatBoostTrainer.learning_rate, help="Boosting learning rate.")
    parser.add_argument(
        "--negatives",
        type=int,
        default=fit_catboost.DEFAULT_NEGATIVES,
        help="Sampled negatives per pool per round (every golden matched into the pool is a positive).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=fit_catboost.DEFAULT_ROUNDS,
        help="Fit rounds — the first draws negatives uniformly, each further one mines hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--folds",
        type=int,
        default=fit_cv.DEFAULT_FOLDS,
        help="Cross-validation folds, grouped by shape so goldens sharing a candidate pool are held out together "
        f"(default {fit_cv.DEFAULT_FOLDS}; 0 skips cross-validation).",
    )
    parser.add_argument(
        "--features",
        default=None,
        help="Feature view: comma-separated names, trailing '*' = prefix glob, leading '-' excludes (recorded in "
        "metrics + provenance). Default: the schedule view (dataset.DEFAULT_FEATURES) or the placement view.",
    )
    parser.add_argument(
        "weights",
        help="Weights artifact to write — emmy/compiler/pipeline/search/prior/weights/schedule.json is the shipped offline prior "
        "(README, 'Fit the offline prior'); any other path is a candidate to A/B through EMMY_OFFLINE_FILE.",
    )
    parser.add_argument("--out", default=None, help="Run dir (default: _tune/fits/<timestamp>/).")
    parser.set_defaults(func=handle_fit)


def _trainer(args, names: list[str]):
    """The trainer and its recorded hyperparameters. ONE trainer serves both the shippable model and every fold: a
    tree ensemble has no warm start, so a fold model cannot inherit anything from the held-out golden."""
    trainer = fit_catboost.CatBoostTrainer(
        feature_names=tuple(names),
        iterations=args.iterations,
        depth=args.depth,
        learning_rate=args.learning_rate,
        negatives=args.negatives,
        rounds=args.rounds,
        random_state=args.seed,
    )
    params = {
        "iterations": args.iterations,
        "depth": args.depth,
        "learning_rate": args.learning_rate,
        "negatives": args.negatives,
        "rounds": args.rounds,
        "objective": "QuerySoftMax",
    }
    return trainer, params


def _log_cells(metrics: dict) -> None:
    """The run's summaries as one line per card per split — the same rows ``metrics.json`` carries, so what
    scrolls past and what is written down cannot disagree. Indexes rather than defends: every key read here
    is one the same process wrote a few lines earlier, so a shape mismatch should raise rather than render
    a line full of ``None``."""
    full, cv = metrics["full_train"], metrics["cv"]
    train = {c["axes"]["gpu"]: c["metrics"]["rank"]["median"] for c in cv.get("summaries", []) if c["axes"]["cv_split"] == "train"}
    gap = cv.get("gap", {})
    for summary in full["summaries"] + [c for c in cv.get("summaries", []) if c["axes"]["cv_split"] == "holdout"]:
        cv_split, gpu, rank = summary["axes"]["cv_split"], summary["axes"]["gpu"], summary["metrics"]["rank"]
        line = f"{cv_split:<11} {gpu:<34} n={summary['groups']:<3} median={rank['median']} (optimistic {rank['median_optimistic']})"
        if cv_split == "full_train":
            skipped = full["skipped"][gpu]
            line += f" unranked={skipped['unranked']} out_of_scope={skipped['out_of_scope']}"
        else:
            line += f" train={train.get(gpu)} gap={gap.get(gpu)}"
        logger.info("%s", line)
    # A card every one of whose goldens was skipped has no summary at all — say so rather than let it vanish.
    for gpu, skipped in full["skipped"].items():
        if gpu not in {c["axes"]["gpu"] for c in full["summaries"]}:
            logger.info("%-11s %-34s no ranked groups  unranked=%d out_of_scope=%d", "full_train", gpu, *skipped.values())


def handle_fit(args) -> None:
    out_dir = Path(args.out) if args.out else Path("_tune/fits") / time.strftime("%Y%m%d-%H%M%S")
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        dataset = Dataset.load(args.dataset)
    except (OSError, ValueError) as exc:
        logger.error("%s", exc)
        sys.exit(2)
    space = dataset.provenance.get("space", "schedule")
    view = args.features or (PLACEMENT_FEATURES if space == "placement" else DEFAULT_FEATURES)
    keep = feature_view(view)
    groups, skipped = dataset.golden, dataset.skipped
    names = sorted({n for c in groups for n in c.feat_names if keep(n)})
    n_dyn = sum(1 for c in groups if c.dynamic)
    # A group is a candidate pool and may carry several verified rows, so the group count alone no longer says
    # how much supervision the fit saw — both numbers travel together, into the header and the provenance.
    # ``merged`` is the positives beyond one per pool; the builder logs the record-level count, which also
    # counts a golden recorded twice at one config (it pins a row already pinned, so it adds no positive).
    positives = sum(len(c.golden_ids) for c in groups)
    logger.info(
        "  %d static + %d dynamic golden groups (%d positives, %d merged), %d features, %d skipped",
        len(groups) - n_dyn,
        n_dyn,
        positives,
        positives - len(groups),
        len(names),
        len(skipped),
    )

    trainer, trainer_params = _trainer(args, names)
    header = {
        # The rows the pools were read from: two fits are comparable only when they were computed over the
        # same golden files, and a file's digest in the source name is what says so.
        "source": str(args.dataset),
        "dataset": dataset.provenance,
        "dropped": dataset.dropped["golden"],
        "seed": args.seed,
        "space": space,
        "feat_ver": features.FEATURIZER_VERSION,
        "features": view,
        "folds": args.folds,
        # Two fits are comparable only when they drew the same way: a sampled fit's ranks are RAW
        # ranks within the draw, and ``per_golden`` prints the true pool size beside them.
        "pool_sample": dataset.provenance["pool_sample"],
        # A group IS a candidate pool; positives are the verified rows marked in it, and a pool can hold more
        # than one. Recorded so a metrics file whose group count dropped against an earlier fit says why,
        # instead of looking like lost data.
        "groups": {"total": len(groups), "positives": positives, "merged": positives - len(groups)},
        "repo_commit": repo_commit(),
        "trainer_params": trainer_params,
    }
    import datetime  # noqa: PLC0415

    metrics, fit = run_fit(groups, skipped, trainer=trainer, folds=args.folds, header=header)

    provenance = {
        "fitted": datetime.date.today().isoformat(),
        "script": "emmy fit",
        "args": {"seed": args.seed, **trainer_params},
        "space": space,
        "features": view,
        "sources": dataset.provenance["sources"],
        "pool_sample": dataset.provenance["pool_sample"],
        "groups": {"static": len(groups) - n_dyn, "dynamic": n_dyn},
        "positives": positives,
        "notes": fit.notes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    storage.write_json(Path(args.weights), fit.model.to_artifact(provenance=provenance, space=space), indent=1)
    logger.info("wrote %s", args.weights)

    _log_cells(metrics)
    logger.info("wrote %s", out_dir / "metrics.json")
