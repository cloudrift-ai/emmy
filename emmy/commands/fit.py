"""``emmy fit`` — fit an offline-prior artifact and cross-validate it, writing a per-run
metrics file.

The fitter entry point: one pipeline, one switch — ``--trainer`` (model class: the incumbent ``linear``
weights or a ``catboost`` ranker) — over the golden pools of the dataset DB
(:func:`~emmy.compiler.pipeline.search.ranking.build_golden_groups`; ``--db`` names another instance).
Both trainers write the same artifact shape, distinguished by its ``kind`` field, so either can be pointed at
with ``EMMY_OFFLINE_FILE`` and A/B'd against the other.

A run writes ``<out>/metrics.json`` — the deterministic, diff-able record two fits are
compared by (same header inputs → identical content; the run dir name, not the file,
carries the timestamp) — and ``<out>/weights.json``, the full-train artifact in the
shipped ``offline_weights.json`` format. The metrics layout (``full_train`` +
a ``cv`` holdout/train/gap block, both carrying ``prior/report.py`` summaries) is documented on
:mod:`emmy.compiler.pipeline.search.prior.fit.cv`, which owns all the fold machinery;
the run itself is :func:`~emmy.compiler.pipeline.search.prior.fit.run.run_fit`. This
module owns the CLI, the trainer wiring, and the file writing.
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path

from emmy import config, storage
from emmy.commands.dataset import golden_dataset
from emmy.compiler.pipeline.search import features
from emmy.compiler.pipeline.search.data.group import DEFAULT_FEATURES
from emmy.compiler.pipeline.search.pool import DEFAULT_SAMPLE
from emmy.compiler.pipeline.search.prior.fit import catboost as fit_catboost
from emmy.compiler.pipeline.search.prior.fit import cv as fit_cv
from emmy.compiler.pipeline.search.prior.fit import linear as fit_linear
from emmy.compiler.pipeline.search.prior.fit.run import run_fit
from emmy.compiler.pipeline.search.prior.linear_model import LinearModel
from emmy.compiler.pipeline.search.ranking import build_golden_groups

logger = logging.getLogger(__name__)


def register_fit_command(subparsers) -> None:
    parser = subparsers.add_parser(
        "fit",
        help="Fit the offline prior and cross-validate it (linear trainer x golden dataset), writing a metrics file",
    )
    parser.add_argument("--trainer", choices=("linear", "catboost"), default="linear")
    parser.add_argument(
        "--db",
        help="Dataset DB whose golden pools are the training data (default: the dataset DB — EMMY_DATASET_DB, else "
        "~/.cache/emmy/dataset.db — filled by `emmy dataset import`).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=0,
        help="linear only: random weight vectors before coordinate descent (default 0: descent-from-seed, the incumbent practice).",
    )
    parser.add_argument(
        "--l2",
        type=float,
        default=fit_linear.DEFAULT_L2,
        help="linear only: raw-space L2 penalty strength in the fit loss (default: the declared tie-breaker strength; 0 disables).",
    )
    parser.add_argument("--iterations", type=int, default=500, help="catboost only: boosting iterations.")
    parser.add_argument(
        "--negatives",
        type=int,
        default=fit_catboost.DEFAULT_NEGATIVES,
        help="catboost only: sampled negatives per pool per round (every golden matched into the pool is a positive).",
    )
    parser.add_argument(
        "--rounds",
        type=int,
        default=fit_catboost.DEFAULT_ROUNDS,
        help="catboost only: fit rounds — the first draws negatives uniformly, each further one mines hard negatives.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--pool-sample",
        type=int,
        default=DEFAULT_SAMPLE,
        help=f"Candidates drawn per pool during enumeration (default {DEFAULT_SAMPLE}; 0 enumerates every row). "
        "Recorded in the metrics header and the artifact provenance - two fits are comparable only when it matches.",
    )
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
        "metrics + provenance). Default: the trainer's own view — the full D_* set for 'linear', and for "
        "'catboost' that set minus the features a tree re-derives from the columns it keeps.",
    )
    parser.add_argument(
        "--artifact",
        nargs="?",
        const="",
        default=None,
        help="Also write the fitted weights artifact to this path (no value: the repo-checked offline_weights.json).",
    )
    parser.add_argument("--out", default=None, help="Run dir (default: _tune/fits/<timestamp>-<trainer>/).")
    parser.set_defaults(func=handle_fit)


def _write_artifact(path: Path, model, provenance: dict) -> None:
    """Write one weights artifact — the JSON, plus the model's binary sidecar when it has one.

    The sidecar is named after the JSON (``weights.json`` → ``weights.cbm``) and recorded RELATIVE in the JSON,
    so the pair travels together: copied into a run directory, rsynced to a tuning box, or checked in beside the
    shipped weights. Naming it after its own JSON is what lets two artifacts share a directory without one
    silently overwriting the other's model.

    Which classes have a sidecar is the MODEL's business, not this function's: it asks for ``model_file`` in the
    artifact and writes ``blob`` only if the model put the key there. A linear artifact is self-contained and
    simply does not."""
    artifact = model.to_artifact(provenance=provenance, model_file=f"{path.stem}.cbm")
    storage.write_json(path, artifact, indent=2)
    if "model_file" in artifact:
        (path.parent / artifact["model_file"]).write_bytes(model.blob)


def _repo_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — a fit outside a git checkout still gets a metrics file
        return "unknown"


def _linear_trainers(args, names: list[str]):
    """The linear summary's trainer pair and the hyperparameters its metrics header records.

    Full-train seeds from the incumbent artifact's weights; fold models seed from ZEROS
    (``warm_start=False``) — the incumbent's weights were fit on every golden, so warm-starting a fold from
    them would leak each held-out golden into its own holdout model. The scalar params seed from the incumbent
    either way: two numbers a fold fit re-derives, not a per-golden memory."""
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    raw = storage.read_json(config.offline_path() or _DEFAULT_FILE)
    if not isinstance(raw, dict) or "scale" not in (raw.get("params") or {}):
        raise SystemExit(
            f"no usable incumbent weights artifact to seed from at {config.offline_path() or _DEFAULT_FILE} "
            f"(needs a 'params' block carrying 'scale')"
        )
    # Lenient read (``LinearModel.from_artifact`` does not version-gate): a refit after a featurizer
    # change is exactly when versions mismatch, and a stale key simply seeds 0.0. A pre-2026-08-05
    # artifact whose params block still lists the retired gate weights simply loses them here — they
    # are linear terms now. ``scale`` rides along on the model, rank-neutral and never fitted.
    incumbent = LinearModel.from_artifact(raw)
    trainer = fit_linear.LinearTrainer(feature_names=tuple(names), init=incumbent, samples=args.samples, l2=args.l2, random_state=args.seed)
    fold_trainer = replace(trainer, warm_start=False)
    params = {
        "samples": args.samples,
        "l2": args.l2,
        "objective": getattr(trainer.objective, "__name__", repr(trainer.objective)),
        "full_train_seed_weights": "incumbent" if trainer.warm_start else "zeros",
        "fold_seed_weights": "incumbent" if fold_trainer.warm_start else "zeros",
    }
    return trainer, fold_trainer, params, incumbent


def _catboost_trainers(args, names: list[str]):
    """The tree summary's trainer and its recorded hyperparameters. ONE trainer serves both the shippable model and
    every fold: a tree ensemble has no warm start, so there is no seeding policy to differ on and no way for a
    fold model to inherit anything from the held-out golden."""
    trainer = fit_catboost.CatBoostTrainer(
        feature_names=tuple(names),
        iterations=args.iterations,
        negatives=args.negatives,
        rounds=args.rounds,
        random_state=args.seed,
    )
    params = {
        "iterations": args.iterations,
        "negatives": args.negatives,
        "rounds": args.rounds,
        "depth": trainer.depth,
        "learning_rate": trainer.learning_rate,
        "objective": "QuerySoftMax",
    }
    return trainer, trainer, params, None


# Each trainer's factory and its default feature view. The views differ because the model classes do: the
# linear one needs the engineered step / fold / interaction features, having no way to form them, and the
# tree re-derives every one of them from the columns ``fit_catboost.TREE_FEATURES`` keeps. ``--features`` overrides
# either, which is how the two views are compared on one model class.
TRAINERS = {
    "linear": (_linear_trainers, DEFAULT_FEATURES),
    "catboost": (_catboost_trainers, fit_catboost.TREE_FEATURES),
}


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
    for f, why in cv.get("fold_detail", {}).get("excluded", {}).items():
        logger.info("cv fold %s EXCLUDED: %s", f, why)


def handle_fit(args) -> None:
    from emmy.compiler.pipeline.search.prior.offline import _DEFAULT_FILE  # noqa: PLC0415

    out_dir = Path(args.out) if args.out else Path("_tune/fits") / f"{time.strftime('%Y%m%d-%H%M%S')}-{args.trainer}"
    out_dir.mkdir(parents=True, exist_ok=True)

    make_trainers, default_view = TRAINERS[args.trainer]
    view = args.features or default_view

    db_path, pools, dropped = golden_dataset(args.db)
    logger.info("Building golden pools from %s (each under its own card's context) ...", db_path)
    groups, skipped = build_golden_groups(pools, view, sample=args.pool_sample, seed=args.seed)
    names = sorted({n for c in groups for n in c.feat_names})
    n_dyn = sum(1 for c in groups if c.dynamic)
    # A group is a candidate pool and may carry several verified rows, so the group count alone no longer says
    # how much supervision the fit saw — both numbers travel together, into the header and the provenance.
    # ``merged`` is the positives beyond one per pool; the builder logs the record-level count, which also
    # counts a golden recorded twice at one config (it pins a row already pinned, so it adds no positive).
    positives = sum(len(c.golden_ids) for c in groups)
    logger.info(
        "  %d static + %d dynamic golden groups (%d positives, %d merged), %d D_* features, %d skipped",
        len(groups) - n_dyn,
        n_dyn,
        positives,
        positives - len(groups),
        len(names),
        len(skipped),
    )

    trainer, fold_trainer, trainer_params, incumbent = make_trainers(args, names)
    header = {
        "trainer": args.trainer,
        # The rows the pools were read from: two fits are comparable only when they were computed over the
        # same golden files, and a file's digest in the source name is what says so.
        "source": str(db_path),
        "sources": dict(Counter(row.source for pool in pools for row in pool.rows)),
        "dropped": dropped,
        "seed": args.seed,
        "feat_ver": features.FEATURIZER_VERSION,
        "features": view,
        "folds": args.folds,
        # Two fits are comparable only when they drew the same way: a sampled fit's ranks are RAW
        # ranks within the draw, and ``per_golden`` prints the true pool size beside them.
        "pool_sample": args.pool_sample,
        # A group IS a candidate pool; positives are the verified rows marked in it, and a pool can hold more
        # than one. Recorded so a metrics file whose group count dropped against an earlier fit says why,
        # instead of looking like lost data.
        "groups": {"total": len(groups), "positives": positives, "merged": positives - len(groups)},
        "repo_commit": _repo_commit(),
        "trainer_params": trainer_params,
    }
    import datetime  # noqa: PLC0415

    metrics, fit = run_fit(groups, skipped, trainer=trainer, fold_trainer=fold_trainer, folds=args.folds, header=header)

    model, notes = fit.model, fit.notes
    # Shipping policy, and the reason ``run_fit`` hands back a fit rather than an artifact: a LINEAR fit with no
    # dynamic groups would otherwise ship with no dynamic weight set at all, so carry the incumbent's forward —
    # loudly, in the provenance notes, never silently. The tree model has no second weight set to be missing.
    if isinstance(model, LinearModel) and model.weights_dynamic is None:
        # ``is not None``, not truthiness: an incumbent that legitimately pruned every dynamic
        # coordinate carries an EMPTY set, and that is still its answer, not a missing one.
        carried = incumbent.weights_dynamic if incumbent.weights_dynamic is not None else model.weights
        source = "incumbent" if incumbent.weights_dynamic is not None else "the static fit"
        model = replace(model, weights_dynamic=carried)
        notes = f"{notes}; dynamic set carried from {source}"
    provenance = {
        "fitted": datetime.date.today().isoformat(),
        "script": "emmy fit",
        "args": {"trainer": args.trainer, "seed": args.seed, **trainer_params},
        "features": view,
        "pool_sample": args.pool_sample,
        "groups": {"static": len(groups) - n_dyn, "dynamic": n_dyn},
        "positives": positives,
        "notes": notes,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    _write_artifact(out_dir / "weights.json", model, provenance)
    if args.artifact is not None:
        artifact_path = Path(args.artifact) if args.artifact else _DEFAULT_FILE
        _write_artifact(artifact_path, model, provenance)
        logger.info("wrote %s", artifact_path)

    _log_cells(metrics)
    logger.info("wrote %s", out_dir / "metrics.json")
