"""``Dataset`` — the training data as a directory a person can inspect: ``manifest.json`` carries everything
textual (each group's identity, labels and feature names, each golden pool with its rows and its kernel's
definition, and the provenance a reader needs), and one ``.npy`` beside it per group holds that group's feature
matrix, which a JSON document is the wrong place for. Written by ``emmy db export``, read by ``emmy fit`` and
``emmy eval prior``; a fit or a report names the directory it read, and two of them are comparable when the
manifests match. Exporting the same DB twice writes the same bytes: the manifest carries no clock.

Nothing here reads a DB: the export (``db/export.py``) builds the groups and hands them over, so this package knows
the shape of the data and not where it came from. The leaf values (:class:`~.kernel.KernelDef`,
:class:`~.pool.GoldenPool`, :class:`~.pool.GoldenRow`) are wire classes and write themselves; a group's wire is its
fields minus the matrix, spelled here beside the file that holds the matrix. Kernel definitions are interned once
by identity — a kernel recorded in two regimes is one definition — and each pool names its kernel by index. The
identity travels beside the definition (``identities``): a dataset is a build product, read back under the version
it was written at, so a reader names a pool without lifting its kernel again.
"""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from emmy.compiler.pipeline.search.dataset.group import GoldenGroup, Group, MeasuredGroup
from emmy.compiler.pipeline.search.dataset.pool import GoldenPool
from emmy.compiler.pipeline.search.features import FEATURIZER_VERSION

logger = logging.getLogger(__name__)

FORMAT = "emmy-dataset"
VERSION = 2
MANIFEST = "manifest.json"


def repo_commit() -> str:
    """The checkout's short commit, ``unknown`` outside a git checkout — what a dataset and a fit record as the
    compiler that produced them."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, check=True, timeout=10)
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 — an export outside a git checkout still gets a manifest
        return "unknown"


@dataclass(frozen=True)
class Dataset:
    """The exported training data: the golden groups (each a candidate pool with its verified rows marked and the
    pools it folded), the measured groups (each a set of benched rows labelled with their microseconds), the golden
    rows that landed in no group with the reason, the rows the admission rule dropped by reason (``golden`` and
    ``measured``), and the provenance — the DB the export read, the sources its rows came from, the pool sample and
    seed, the featurizer version and the compiler commit."""

    golden: list[GoldenGroup]
    measured: list[MeasuredGroup]
    skipped: list[tuple[str, str, str]]
    dropped: dict[str, dict[str, int]]
    provenance: dict

    def dump(self, path: str | Path) -> Path:
        """Write the dataset directory at ``path``: the matrices first, then the manifest, so a directory without a
        manifest is a write that did not finish. An existing directory is replaced only when it is itself a dataset
        — anything else is refused rather than deleted."""
        out = Path(path)
        if out.exists():
            if not (out.is_dir() and _is_dataset(out / MANIFEST)):
                raise RuntimeError(f"{out} exists and is not a dataset — refusing to replace it")
            shutil.rmtree(out)
        kernels: list[dict] = []
        index: dict[str, int] = {}  # exact identity -> position; its keys, in order, are the manifest's ``identities``
        golden = [_write_group(out, group, f"golden/{i:04d}.npy", index, kernels) for i, group in enumerate(self.golden)]
        measured = [_write_group(out, group, f"measured/{i:04d}.npy", index, kernels) for i, group in enumerate(self.measured)]
        manifest = {
            "format": FORMAT,
            "version": VERSION,
            "provenance": self.provenance,
            "kernels": kernels,
            "identities": list(index),
            "golden": golden,
            "measured": measured,
            "skipped": [list(entry) for entry in self.skipped],
            "dropped": self.dropped,
        }
        (out / MANIFEST).write_text(json.dumps(manifest, indent=1) + "\n")
        return out

    @classmethod
    def load(cls, path: str | Path) -> Dataset:
        """The dataset at ``path``. A directory that is no dataset, another format version, or an export under
        another featurizer version is refused by name: its columns are spelled in another vocabulary, and the fix
        is a re-export, never a guess."""
        base = Path(path)
        if not _is_dataset(base / MANIFEST):
            raise FileNotFoundError(f"no dataset at {base} — write one with `emmy db export`")
        manifest = json.loads((base / MANIFEST).read_text())
        if manifest.get("version") != VERSION:
            raise ValueError(f"{base} is a version {manifest.get('version')!r} dataset; this code reads version {VERSION} — re-export it")
        provenance = manifest["provenance"]
        if provenance.get("feat_ver") != FEATURIZER_VERSION:
            raise ValueError(
                f"{base} was exported under featurizer version {provenance.get('feat_ver')!r}; this code reads "
                f"{FEATURIZER_VERSION} — re-export it (`emmy db export`)"
            )
        if provenance.get("compiler") != (current := repo_commit()):
            logger.info("dataset %s was exported at commit %s; this checkout is %s", base, provenance.get("compiler"), current)
        kernels = list(zip(manifest["kernels"], manifest["identities"], strict=True))
        return cls(
            [_read_group(base, wire, kernels) for wire in manifest["golden"]],
            [_read_group(base, wire, kernels) for wire in manifest["measured"]],
            [tuple(entry) for entry in manifest["skipped"]],
            manifest["dropped"],
            provenance,
        )


def _is_dataset(manifest: Path) -> bool:
    try:
        return json.loads(manifest.read_text()).get("format") == FORMAT
    except (OSError, ValueError, AttributeError):
        return False


def _write_group(out: Path, group: Group, matrix: str, index: dict[str, int], kernels: list[dict]) -> dict:
    """``group``'s matrix written at ``matrix`` under ``out`` and its manifest entry: the fields every group has, then
    a golden group's verified rows and the pools it folded (each pool's kernel interned into ``kernels``), or a
    measured group's microseconds and regime."""
    (out / matrix).parent.mkdir(parents=True, exist_ok=True)
    np.save(out / matrix, group.feats)
    wire = {
        "key": group.key,
        "name": group.name,
        "tier": group.tier,
        "gpu": group.gpu,
        "shape": group.shape,
        "dynamic": group.dynamic,
        "feat_names": list(group.feat_names),
        "total": group.total,
        "matrix": matrix,
    }
    if isinstance(group, GoldenGroup):
        wire["golden_ids"] = list(group.golden_ids)
        wire["pools"] = [_pool_wire(pool, index, kernels) for pool in group.pools]
    elif isinstance(group, MeasuredGroup):
        wire["h_opt"] = group.h_opt
        wire["latency_us"] = group.latency_us.tolist()
    return wire


def _pool_wire(pool: GoldenPool, index: dict[str, int], kernels: list[dict]) -> dict:
    wire = pool.to_wire()
    identity = pool.kernel.exact_identity
    if identity not in index:
        index[identity] = len(kernels)
        kernels.append(wire["kernel"])
    wire["kernel"] = index[identity]
    return wire


def _read_group(base: Path, wire: dict, kernels: list[tuple[dict, str]]) -> Group:
    common = (
        wire["key"],
        wire["name"],
        wire["tier"],
        wire["gpu"],
        wire["shape"],
        wire["dynamic"],
        tuple(wire["feat_names"]),
        np.load(base / wire["matrix"]),
        wire["total"],
    )
    if "golden_ids" in wire:
        pools = []
        for pool in wire["pools"]:
            definition, identity = kernels[pool["kernel"]]
            read = GoldenPool.from_wire({**pool, "kernel": definition})
            read.kernel.keyed(identity)  # the manifest's own: a reader names the pool without lifting its kernel
            pools.append(read)
        return GoldenGroup(*common, golden_ids=tuple(wire["golden_ids"]), pools=tuple(pools))
    return MeasuredGroup(*common, latency_us=np.asarray(wire["latency_us"], dtype=float), h_opt=wire["h_opt"])
