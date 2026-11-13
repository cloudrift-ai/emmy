"""One corpus case through both code generators on the card: its program built from CUDA text by nvcc and from PTX
by ptxas, run on the same inputs. ``python -m tests.compiler.ptx.differential`` prints, for every case the card
runs whose every kernel compiles to PTX, whether the two builds give the same bits and how far apart they are."""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

from emmy import config
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from tests.compiler.helpers import device_compute_capability
from tests.compiler.ptx.corpus import compiled, kernel_code, launched
from tests.compiler.realization.helpers import Case, _has_narrow_operand, _tolerance, case_files, load_case, lowered, seeded_inputs


@dataclass(frozen=True)
class Difference:
    """The outputs of one case from the two builds: whether every bit is the same, and the largest difference of
    an element as a fraction of its output's largest magnitude."""

    same_bits: bool
    largest: float


def whole(case: Case) -> bool:
    """Whether every kernel ``case`` launches compiles to PTX. GPU-free."""
    outcomes = launched(*compiled(case))
    return bool(outcomes) and all(outcome.ptx is not None for outcome in outcomes)


def outputs(case: Case, code: str) -> dict[str, np.ndarray]:
    """The outputs of ``case`` on the card, its kernels generated as ``code`` says (``config.KERNEL_CODES``) and
    every one of them that way."""
    from emmy.compiler.backend.cuda.backend import CudaBackend  # noqa: PLC0415 — needs the runtime extension

    with kernel_code(code):
        graph, _taken = lowered(case, Context.probe())
    sources = [node.op.kernel_source for node in graph.nodes.values() if isinstance(node.op, CudaOp)]
    assert sources and all(nvcc.is_ptx(source) == (code != config.KERNEL_CODE_CUDA) for source in sources)
    program = case.program()
    result, _ = CudaBackend().run(graph, input_data=dict(seeded_inputs(program)))
    return {name: np.asarray(result.outputs[name]) for name in program.outputs}


def compare(case: Case) -> Difference:
    """Run ``case`` both ways and hold the PTX build to the CUDA build within the corpus's own tolerance."""
    cuda, ptx = outputs(case, config.KERNEL_CODE_CUDA), outputs(case, config.KERNEL_CODE_PTX_ONLY)
    narrow = _has_narrow_operand(case.program())
    same, largest = True, 0.0
    for name, want in cuda.items():
        got = ptx[name]
        assert got.dtype == want.dtype and got.shape == want.shape, f"{case.id}: output {name}"
        same = same and got.tobytes() == want.tobytes()
        if want.dtype.kind == "f":
            np.testing.assert_allclose(got, want, err_msg=f"{case.id}: output {name}", equal_nan=True, **_tolerance(narrow, want))
            finite = np.isfinite(want) & np.isfinite(got)
            peak = float(np.max(np.abs(want[finite]), initial=0.0))
            if peak:
                largest = max(largest, float(np.max(np.abs(got[finite].astype(np.float64) - want[finite]), initial=0.0)) / peak)
        else:
            np.testing.assert_array_equal(got, want, err_msg=f"{case.id}: output {name}")
    return Difference(same, largest)


def main() -> None:
    """One line a case on stdout (case, ``same bits`` or the largest difference), the counts on stderr."""
    live = device_compute_capability()
    same = differing = 0
    for path in case_files():
        case = load_case(path)
        if case.compute_cap != live or not whole(case):
            continue
        difference = compare(case)
        same += difference.same_bits
        differing += not difference.same_bits
        print(f"{case.id}\t{'same bits' if difference.same_bits else f'{difference.largest:.3e}'}", flush=True)
    print(f"\n{same + differing} cases through both: {same} with the same bits, {differing} within tolerance", file=sys.stderr)


if __name__ == "__main__":
    main()
