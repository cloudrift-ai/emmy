"""Shared CLI vocabulary for ``emmy eval``: the dataset argument of ``eval prior`` (:func:`add_dataset_args`) and
the prior-file override."""

from __future__ import annotations

import os
from pathlib import Path

from emmy import config


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
