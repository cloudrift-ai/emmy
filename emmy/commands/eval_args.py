"""Shared CLI vocabulary for ``emmy eval``: the tune-DB flags of the per-kernel views (:func:`add_db_args`), the
dataset argument of ``eval prior`` (:func:`add_dataset_args`), and the prior-file override — so every command names
a DB or a dataset through one spelling instead of opening files by hand."""

from __future__ import annotations

import os
from pathlib import Path

from emmy import config


def add_db_args(parser, *, with_min_variants: bool = False) -> None:
    """Register the tune-DB flags of a per-kernel view: ``--db`` and ``--kernel``, plus the regret analysis's
    grouping threshold when ``with_min_variants``."""
    parser.add_argument(
        "--db",
        help="Tune DB to read (default: EMMY_TUNE_DB, else ~/.cache/emmy/autotune.db), whose rows carry the kernel "
        "sources they name kernels by.",
    )
    parser.add_argument("--kernel", help="Filter by substring of the kernel C identifier.")
    if with_min_variants:
        parser.add_argument(
            "--min-variants", type=int, default=8, help="Skip kernels with fewer than this many measured variants (default: 8)."
        )


def add_dataset_args(parser) -> None:
    """Register ``eval prior``'s data source: the dataset directory ``emmy db export`` wrote, and which of its pools
    to report over."""
    parser.add_argument("dataset", help="Dataset directory written by `emmy db export`, e.g. _data/dataset.")
    parser.add_argument(
        "--pools",
        choices=["golden", "measured"],
        default="golden",
        help="Which pools to report over: 'golden' (each golden row ranked among the candidates its kernel offers) or "
        "'measured' (every benched pool: Spearman and regret, what a wrong pick costs). Default: golden.",
    )
    parser.add_argument(
        "--kernel",
        help="Filter by substring: the pool's kernel C name for --pools golden; the op label (e.g. 'matmul', 'reduce', "
        "'free=512') for --pools measured.",
    )


def resolve_offline_arg(args) -> None:
    """Publish ``--offline-file`` into the env (``EMMY_OFFLINE_FILE``) so the
    offline prior loads from it."""
    if getattr(args, "offline_file", None):
        os.environ[config.OFFLINE_FILE] = str(Path(args.offline_file).expanduser())
