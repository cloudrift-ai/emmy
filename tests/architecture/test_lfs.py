"""Git LFS guard: every tracked file whose ``filter`` attribute is ``lfs`` is stored as a pointer, never as its
payload.

``.gitattributes`` names the filter; only ``git lfs install`` wires it, on the machine that runs ``git add``. A
freeze file a tune wrote on a rented card, copied back and added where the install never ran, would land in history
as a plain blob — hundreds of kilobytes a clone carries forever, and a pull that LFS never serves. Git only, no GPU,
no fixtures."""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_POINTER = "version https://git-lfs.github.com/spec/v1"


def _git(*args: str, stdin: str | None = None) -> str:
    return subprocess.run(["git", *args], cwd=_REPO_ROOT, input=stdin, check=True, capture_output=True, text=True).stdout


def test_every_lfs_tracked_file_is_stored_as_a_pointer() -> None:
    """The index's blob of every path whose attribute says LFS starts with the pointer header: a payload there was
    added where ``git lfs install`` never ran. The fix is in the message."""
    if shutil.which("git") is None or not (_REPO_ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    oids = {}
    for line in _git("ls-files", "-s").splitlines():
        meta, path = line.split("\t", 1)
        oids[path] = meta.split()[1]
    fields = _git("check-attr", "-z", "--stdin", "filter", stdin="\0".join(oids)).split("\0")
    lfs = [path for path, _attr, value in zip(fields[0::3], fields[1::3], fields[2::3], strict=False) if value == "lfs"]
    assert lfs, "no LFS-tracked file: the attributes moved?"
    payloads = [path for path in lfs if not _git("cat-file", "-p", oids[path]).startswith(_POINTER)]
    assert not payloads, (
        f"stored as payload, not as an LFS pointer: {payloads} — run `git lfs install`, then `git add --renormalize <path>` and commit"
    )
