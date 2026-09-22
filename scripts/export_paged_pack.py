#!/usr/bin/env python
"""Export a standalone pack whose output is a paged KV cache.

The program is one step of a cache fill: it reads a chunk of new keys and writes them into the
cache at the runtime position ``past``. The cache is never one allocation — the pack declares it
paged, so the Rust runtime owns its pages and binds their table. ``crates/emmy-runtime/tests/
paged.rs`` runs the result; point ``EMMY_PAGED_PACK`` at the directory this prints.

    ./venv/bin/python scripts/export_paged_pack.py /tmp/paged-pack
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

KV_HEADS, ROW, CHUNK, PAGE = 2, 8, 4, 8


class Step(torch.nn.Module):
    """A chunk of new keys, standing in for a K projection."""

    def forward(self, chunk):
        return torch.tanh(chunk)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out", type=Path, help="directory to write the pack into")
    args = parser.parse_args()

    from emmy.compiler.backend.cuda.backend import CudaBackend
    from emmy.compiler.backend.pack import save_executable
    from emmy.compiler.backend.plan import plan_from_graph
    from emmy.compiler.trace.torch import trace_module

    graph = trace_module(Step(), (torch.zeros(1, KV_HEADS, CHUNK, ROW),))
    graph.rename_node(graph.inputs[0], "chunk")
    graph.rename_node(graph.outputs[0], "cache")
    # The step's output holds only its own CHUNK keys; ``past`` says which pages of the cache they
    # land in, and how many pages the cache has is the runtime's business, not the pack's.
    graph.hints.set("cuda.paged_buffers", (("cache", 2, PAGE, "past"),))

    plan = plan_from_graph(CudaBackend().compile(graph))
    root = save_executable(args.out, {"step": plan}, bindings={"step": {}}, key={})
    print(root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
