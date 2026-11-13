"""The realization corpus through the Kernel IR to PTX compiler.

Without a card: every kernel a case lowers to either compiles to PTX or is refused for a missing rule, never for
anything else, and every PTX text assembles for the case's declared capability at ptxas's default level and at
``-O0``. On the card: a case whose every kernel compiles to PTX gives, built from PTX, what it gives built from CUDA
text by nvcc, on the same inputs. A sample of each family runs by default; ``pytest -m sweep`` runs every case.
"""

from __future__ import annotations

import functools
import re
import tempfile
from pathlib import Path

import pytest
from emmy.emmy_ir import compile_ptx, parse_body

from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.backend.plan import KernelSpec, merge_kernels
from emmy.compiler.dtype import F32
from emmy.compiler.ir.kernel.interface import Interface, Param
from tests.compiler.helpers import device_compute_capability, requires_cuda
from tests.compiler.ptx import corpus, differential, launch
from tests.compiler.realization.helpers import CASES_DIR, case_files, load_case

_CASES = [path.relative_to(CASES_DIR).as_posix() for path in case_files()]

#: The cases the default lane runs both ways, and whether the two builds give the same bits: they do where no
#: kernel of the case holds a math function, since the fusion decision picks the same pairs in each of these.
SAMPLE = {
    "matmul/f32-scalar-staging-gmem-64x64x64.json": True,
    "matmul/f16-scalar-register-tile.json": True,
    "matmul/bf16-scalar-sync-d2-sm120.json": True,
    "reduce/combine-amax-coop-warp.json": True,
    "reduce/combine-l2-ilp-coop.json": True,
    "reduce/cross-cta-sum-atomic.json": True,
    "reduce/online-softmax-2x64.json": False,
    "qwen3emb/rmsnorm-layer-s128.json": True,
    "qwen3emb/matmul-qproj-s128.json": True,
    "qwen3emb/sdpa-s128.json": False,
    "fused/head-major-views-coop-reader.json": True,
    "fused/gdn-conv-join-in-place-name.json": False,
    "attention/rmsnorm-qk-sdpa-composed-cut.json": False,
    "scan/f32-cumsum-odd-extent.json": True,
}


@functools.cache
def _assembles(compute_cap: tuple[int, int]) -> bool:
    """Whether the installed ptxas takes the card at all: the one of CUDA 13.0 no longer takes ``sm_70``."""
    major, minor = compute_cap
    body = "Grid(axes=(i in 0..256), decode=(thread_idx.x), block_threads=256):\n    f32 x = load a[i]\n    f32 o[i] = x"
    interface = Interface((Param("a", "input", F32, "a"), Param("o", "output", F32, "o")), 256, {}, 0, False)
    probe = compile_ptx("probe", parse_body(body), interface, sm=major * 10 + minor)
    with tempfile.TemporaryDirectory() as scratch:
        try:
            launch.assemble(probe, None, Path(scratch), arch=f"sm_{major}{minor}")
        except AssertionError:
            return False
    return True


@pytest.mark.parametrize("path", _CASES)
def test_a_case_compiles_to_ptx_or_names_the_missing_rule_and_its_ptx_assembles(path, tmp_path):
    """A kernel the compiler raises on for any reason but a missing rule is a bug, of the compiler or of the pass
    that wrote the kernel. A kernel's name stands in its PTX as its entry and in its parameter and shared symbols
    alone, so the kernel merge may rename it in the text."""
    if nvcc.nvcc_path() is None:
        pytest.skip("the CUDA toolkit is not installed")
    case = load_case(CASES_DIR / path)
    outcomes = corpus.launched(*corpus.compiled(case))
    errors = {outcome.kernel: outcome.error for outcome in outcomes if outcome.error}
    assert not errors, errors
    arch = "sm_{}{}".format(*case.compute_cap)
    texts = {outcome.kernel: outcome for outcome in outcomes if outcome.ptx is not None}
    for name, outcome in texts.items():
        code = "\n".join(line for line in outcome.ptx.splitlines() if not line.lstrip().startswith("//"))
        symbols = re.findall(rf"(?<![\w$.%]){re.escape(name)}(?:_param_\d+|_shared_\d+)?(?![\w$])", code)
        assert len(symbols) == code.count(name), f"{name} stands in its PTX outside its own symbols"
    if not _assembles(case.compute_cap):
        pytest.skip(f"ptxas does not assemble {arch}; {len(texts)} kernels compiled to PTX")
    for outcome in texts.values():
        for level in launch.LEVELS:
            launch.assemble(outcome.ptx, level, tmp_path, outcome.arch_specific, arch)


def test_a_merged_ptx_kernel_is_renamed_in_its_symbols(tmp_path):
    """Two kernels of one name and different PTX: the second takes a new name in its entry and in every symbol the
    printer derives from it, and still assembles."""
    if nvcc.nvcc_path() is None:
        pytest.skip("the CUDA toolkit is not installed")
    case = load_case(CASES_DIR / "reduce" / "combine-amax-coop-warp.json")
    [outcome] = corpus.launched(*corpus.compiled(case))
    assert outcome.ptx is not None
    kernels = {outcome.kernel: KernelSpec(source="taken")}
    renamed = merge_kernels(kernels, {outcome.kernel: KernelSpec(source=outcome.ptx)})[outcome.kernel]
    text = kernels[renamed].source
    assert renamed != outcome.kernel and f".entry {renamed}(" in text and f"{renamed}_param_0" in text
    assert outcome.kernel not in text.replace(renamed, "")
    launch.assemble(text, None, tmp_path, arch="sm_{}{}".format(*case.compute_cap))


def _both_ways(path: str) -> differential.Difference:
    case = load_case(CASES_DIR / path)
    if device_compute_capability() != case.compute_cap:
        pytest.skip("the case declares another card")
    if not differential.whole(case):
        pytest.skip("a kernel of the case has no PTX rule yet")
    return differential.compare(case)


@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("path", SAMPLE)
def test_a_sampled_case_gives_through_ptx_what_it_gives_through_cuda(path):
    difference = _both_ways(path)
    assert difference.same_bits or not SAMPLE[path], f"{path}: the builds differ by {difference.largest:.3e} of the output's peak"


@pytest.mark.sweep
@requires_cuda
@pytest.mark.xdist_group("cuda")
@pytest.mark.parametrize("path", _CASES)
def test_a_case_gives_through_ptx_what_it_gives_through_cuda(path):
    """Within the corpus's own tolerance; ``python -m tests.compiler.ptx.differential`` prints how far apart."""
    _both_ways(path)
