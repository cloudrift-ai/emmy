"""The realization corpus through the Kernel IR to PTX compiler, without a card: every kernel a case lowers to
compiles to PTX, and every PTX text assembles for the case's declared capability at ptxas's default level and at
``-O0``. The realization corpus runs the cases on the card.
"""

from __future__ import annotations

import re

import pytest

from emmy.compiler.backend.cuda import ptxas
from emmy.compiler.backend.plan import KernelSpec, merge_kernels
from emmy.compiler.ir.cuda import CudaOp
from tests.compiler.ptx import launch
from tests.compiler.realization.helpers import CASES_DIR, assembles, case_files, load_case, lowered

_CASES = [path.relative_to(CASES_DIR).as_posix() for path in case_files()]


def _texts(path: str) -> tuple[tuple[int, int], dict[str, str]]:
    """The declared capability of the case at ``path`` and the PTX of each kernel it launches. A kernel the compiler
    refuses raises: it is a bug, of the compiler or of the pass that wrote the kernel."""
    case = load_case(CASES_DIR / path)
    graph, _taken = lowered(case, case.context())
    return case.compute_cap, {node.op.kernel_name: node.op.kernel_source for node in graph.nodes.values() if isinstance(node.op, CudaOp)}


@pytest.mark.parametrize("path", _CASES)
def test_a_case_compiles_to_ptx_and_its_ptx_assembles(path):
    """Every kernel of the case compiles. A kernel's name stands in its PTX as its entry and in its parameter and
    shared symbols alone, so the kernel merge may rename it in the text."""
    if ptxas.ptxas_path() is None:
        pytest.skip("ptxas is not installed")
    capability, texts = _texts(path)
    assert texts
    for name, text in texts.items():
        code = "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("//"))
        symbols = re.findall(rf"(?<![\w$.%]){re.escape(name)}(?:_param_\d+|_shared_\d+)?(?![\w$])", code)
        assert len(symbols) == code.count(name), f"{name} stands in its PTX outside its own symbols"
    if not assembles(capability):
        pytest.skip(f"ptxas does not assemble sm_{capability[0]}{capability[1]}; {len(texts)} kernels compiled to PTX")
    for name, text in texts.items():
        for level in launch.LEVELS:
            ptxas.compile_to_cubin(text, name, level=level)


def test_a_merged_ptx_kernel_is_renamed_in_its_symbols():
    """Two kernels of one name and different PTX: the second takes a new name in its entry and in every symbol the
    printer derives from it, and still assembles."""
    if ptxas.ptxas_path() is None:
        pytest.skip("ptxas is not installed")
    _capability, texts = _texts("reduce/combine-amax-coop-warp.json")
    [(name, text)] = texts.items()
    kernels = {name: KernelSpec(source="taken")}
    renamed = merge_kernels(kernels, {name: KernelSpec(source=text)})[name]
    merged = kernels[renamed].source
    assert renamed != name and f".entry {renamed}(" in merged and f"{renamed}_param_0" in merged
    assert name not in merged.replace(renamed, "")
    ptxas.compile_to_cubin(merged, renamed)
