"""The realization corpus through the Kernel IR to PTX compiler, without a card: every kernel a case lowers to
compiles to PTX, and every PTX text assembles for the case's declared capability at ptxas's default level and at
``-O0``. The realization corpus runs the cases on the card.
"""

from __future__ import annotations

import re

import pytest

from emmy.compiler.backend.cuda import ptxas
from emmy.compiler.backend.plan import KernelSpec, merge_kernels
from tests.compiler.ptx import corpus, launch
from tests.compiler.realization.helpers import CASES_DIR, assembles, case_files, load_case

_CASES = [path.relative_to(CASES_DIR).as_posix() for path in case_files()]


@pytest.mark.parametrize("path", _CASES)
def test_a_case_compiles_to_ptx_and_its_ptx_assembles(path, tmp_path):
    """A kernel the compiler refuses is a bug, of the compiler or of the pass that wrote the kernel. A kernel's name
    stands in its PTX as its entry and in its parameter and shared symbols alone, so the kernel merge may rename it
    in the text."""
    if ptxas.ptxas_path() is None:
        pytest.skip("ptxas is not installed")
    case = load_case(CASES_DIR / path)
    outcomes = corpus.launched(*corpus.compiled(case))
    refused = {outcome.kernel: outcome.error or outcome.no_rule for outcome in outcomes if outcome.ptx is None}
    assert outcomes and not refused, refused
    arch = "sm_{}{}".format(*case.compute_cap)
    texts = {outcome.kernel: outcome for outcome in outcomes if outcome.ptx is not None}
    for name, outcome in texts.items():
        code = "\n".join(line for line in outcome.ptx.splitlines() if not line.lstrip().startswith("//"))
        symbols = re.findall(rf"(?<![\w$.%]){re.escape(name)}(?:_param_\d+|_shared_\d+)?(?![\w$])", code)
        assert len(symbols) == code.count(name), f"{name} stands in its PTX outside its own symbols"
    if not assembles(case.compute_cap):
        pytest.skip(f"ptxas does not assemble {arch}; {len(texts)} kernels compiled to PTX")
    for outcome in texts.values():
        for level in launch.LEVELS:
            launch.assemble(outcome.ptx, level, tmp_path, outcome.arch_specific, arch)


def test_a_merged_ptx_kernel_is_renamed_in_its_symbols(tmp_path):
    """Two kernels of one name and different PTX: the second takes a new name in its entry and in every symbol the
    printer derives from it, and still assembles."""
    if ptxas.ptxas_path() is None:
        pytest.skip("ptxas is not installed")
    case = load_case(CASES_DIR / "reduce" / "combine-amax-coop-warp.json")
    [outcome] = corpus.launched(*corpus.compiled(case))
    assert outcome.ptx is not None
    kernels = {outcome.kernel: KernelSpec(source="taken")}
    renamed = merge_kernels(kernels, {outcome.kernel: KernelSpec(source=outcome.ptx)})[outcome.kernel]
    text = kernels[renamed].source
    assert renamed != outcome.kernel and f".entry {renamed}(" in text and f"{renamed}_param_0" in text
    assert outcome.kernel not in text.replace(renamed, "")
    launch.assemble(text, None, tmp_path, arch="sm_{}{}".format(*case.compute_cap))
