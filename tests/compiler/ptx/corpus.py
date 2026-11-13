"""The realization corpus through the Kernel IR to PTX compiler: each case lowered, and what became of each of its
kernels. GPU-free. ``python -m tests.compiler.ptx.corpus`` prints the table."""

from __future__ import annotations

import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

from emmy.compiler.context import Context
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel import ptx
from tests.compiler.realization.helpers import CASES_DIR, Case, case_files, load_case, lowered


@dataclass(frozen=True)
class Outcome:
    """One kernel of a lowered case: its PTX, or the statement or operation the compiler has no rule for, or the
    error the compiler raised on it, which is a bug."""

    kernel: str
    arch_specific: bool
    ptx: str | None = None
    no_rule: str | None = None
    error: str | None = None


def compiled(case: Case, ctx: Context | None = None) -> tuple[object | None, list[Outcome]]:
    """``case`` lowered at ``ctx`` (its own declared capability when omitted), and the outcome of each kernel in the
    order the lowering reached it. The graph is ``None`` where the compiler refused a kernel, which ends the
    lowering: the last outcome names the refusal."""
    outcomes: list[Outcome] = []
    compile_kernel = ptx.compile_kernel

    def recording(kernel, interface, **decisions):  # noqa: ANN001, ANN003, ANN202 — the compiler entry's signature
        name, arch = kernel.name, interface.arch_specific
        try:
            text = compile_kernel(kernel, interface, **decisions)
        except ptx.NoRule as refusal:
            outcomes.append(Outcome(name, arch, no_rule=str(refusal)))
            raise
        except Exception as error:  # noqa: BLE001 — the report's product is the error
            outcomes.append(Outcome(name, arch, error=f"{type(error).__name__}: {error}"))
            raise
        outcomes.append(Outcome(name, arch, ptx=text))
        return text

    ptx.compile_kernel = recording
    try:
        graph, _taken = lowered(case, ctx or case.context())
    except Exception:
        if not outcomes or outcomes[-1].ptx is not None:
            raise
        graph = None
    finally:
        ptx.compile_kernel = compile_kernel
    return graph, outcomes


def launched(graph, outcomes: list[Outcome]) -> list[Outcome]:  # noqa: ANN001 — a lowered graph
    """The outcomes of the kernels the lowered graph launches: a search may lower a kernel it then does not keep.
    With no graph, the refusal that ended the lowering."""
    if graph is None:
        return outcomes[-1:]
    names = {node.op.kernel_name for node in graph.nodes.values() if isinstance(node.op, CudaOp)}
    last = {outcome.kernel: outcome for outcome in outcomes}
    return [last[name] for name in sorted(names) if name in last]


def reason(message: str) -> str:
    """What a refusal names, without the statement it stands at."""
    return message.split(": `", 1)[0].removeprefix("no PTX rule for ")


def _row(path: str) -> tuple[str, tuple[int, int], list[Outcome]]:
    case = load_case(CASES_DIR / path)
    return case.id, case.compute_cap, launched(*compiled(case))


def main() -> None:
    """One line a kernel on stdout (case, capability, kernel, verdict), the counts by reason on stderr."""
    paths = [path.relative_to(CASES_DIR).as_posix() for path in case_files()]
    no_rule: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    kernels = whole = 0
    with ProcessPoolExecutor(max_workers=min(6, os.cpu_count() or 1)) as pool:
        rows = list(pool.map(_row, paths, chunksize=4))
    for case, (major, minor), outcomes in rows:
        kernels += len(outcomes)
        whole += all(outcome.ptx is not None for outcome in outcomes)
        for outcome in outcomes:
            verdict = "ptx" if outcome.ptx is not None else f"no rule: {outcome.no_rule}" if outcome.no_rule else f"ERROR: {outcome.error}"
            print(f"{case}\tsm_{major}{minor}\t{outcome.kernel}\t{verdict}")
            if outcome.no_rule:
                no_rule[reason(outcome.no_rule)] += 1
            if outcome.error:
                errors[reason(outcome.error)] += 1
    in_ptx = kernels - sum(no_rule.values()) - sum(errors.values())
    print(f"\n{len(rows)} cases, {whole} with every kernel in PTX; {kernels} kernels, {in_ptx} in PTX", file=sys.stderr)
    for title, counts in (("no rule", no_rule), ("compiler error", errors)):
        for what, count in counts.most_common():
            print(f"{count:5d}  {title}: {what}", file=sys.stderr)


if __name__ == "__main__":
    main()
