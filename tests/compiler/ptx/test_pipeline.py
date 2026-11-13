"""``EMMY_KERNEL_CODE``: which generator writes a kernel's code in the CUDA lowering, and when the lowering falls
back from the PTX compiler to the CUDA renderer. GPU-free but for the pack's key."""

from __future__ import annotations

import pytest

from emmy import config
from emmy.compiler.backend.cuda import nvcc
from emmy.compiler.backend.plan import plan_from_dict, plan_from_graph, plan_to_dict
from emmy.compiler.ir.cuda import CudaOp
from emmy.compiler.ir.kernel import ptx
from tests.compiler.helpers import requires_cuda
from tests.compiler.ptx.corpus import kernel_code
from tests.compiler.realization.helpers import CASES_DIR, load_case, lowered

#: A case of two kernels, one of which the test makes the compiler refuse as having no rule, and a case of one kernel.
MIXED = "matmul/f16-mma-splitk-deferred.json"
COVERED = "reduce/combine-amax-coop-warp.json"


def _sources(path: str, code: str | None) -> tuple[dict[str, CudaOp], object]:
    """The kernels ``path`` lowers to under the switch's value ``code`` (unset when ``None``), and its graph."""
    case = load_case(CASES_DIR / path)
    with kernel_code(code):
        graph, _taken = lowered(case, case.context())
    return {node.op.kernel_name: node.op for node in graph.nodes.values() if isinstance(node.op, CudaOp)}, graph


def test_the_default_renders_cuda_and_ptx_falls_back_for_a_missing_rule_alone(monkeypatch):
    compile_kernel = ptx.compile_kernel

    def missing_one(kernel, interface, **decisions):  # noqa: ANN001, ANN003, ANN202 — the compiler entry's signature
        # Every corpus kernel compiles: the refusal of one stands for a statement with no rule yet.
        if kernel.name.endswith("__partial"):
            raise ptx.NoRule(f"no PTX rule for the test's missing rule: `{kernel.name}`")
        return compile_kernel(kernel, interface, **decisions)

    monkeypatch.setattr(ptx, "compile_kernel", missing_one)
    default, _ = _sources(MIXED, None)
    assert len(default) == 2 and not any(nvcc.is_ptx(op.kernel_source) for op in default.values())
    mixed, _ = _sources(MIXED, config.KERNEL_CODE_PTX)
    assert sorted(nvcc.is_ptx(op.kernel_source) for op in mixed.values()) == [False, True]
    # The kernel the compiler has no rule for is the text the default renders; the launch is the same either way.
    for name, op in mixed.items():
        if not nvcc.is_ptx(op.kernel_source):
            assert op == default[name]
        else:
            assert (op.arg_order, op.grid, op.block, op.smem_bytes) == (
                default[name].arg_order,
                default[name].grid,
                default[name].block,
                default[name].smem_bytes,
            )
    with pytest.raises(ptx.NoRule, match="no PTX rule for"):
        _sources(MIXED, config.KERNEL_CODE_PTX_ONLY)


def test_an_error_that_is_no_missing_rule_is_raised_and_not_rendered_around(monkeypatch):
    def broken(kernel, interface, **decisions):  # noqa: ANN001, ANN003, ANN202
        raise ValueError("PTX verifier: a register is read undefined")

    monkeypatch.setattr(ptx, "compile_kernel", broken)
    with pytest.raises(ValueError, match="PTX verifier"):
        _sources(COVERED, config.KERNEL_CODE_PTX)
    _sources(COVERED, None)


def test_an_unknown_value_of_the_switch_is_refused(monkeypatch):
    monkeypatch.setenv(config.KERNEL_CODE, "nvrtc")
    with pytest.raises(ValueError, match="EMMY_KERNEL_CODE"):
        config.kernel_code()


def test_a_ptx_kernel_keeps_its_name_out_of_its_identity_and_round_trips_in_a_plan():
    sources, graph = _sources(COVERED, config.KERNEL_CODE_PTX_ONLY)
    [(name, op)] = sources.items()
    text = op.kernel_source
    assert nvcc.is_ptx(text) and f".entry {name}(" in text
    other = "k_renamed"
    renamed = CudaOp(**{**op.__dict__, "kernel_source": text.replace(name, other), "kernel_name": other})
    assert renamed.identity_key() == op.identity_key()
    plan = plan_from_graph(graph)
    assert plan_to_dict(plan_from_dict(plan_to_dict(plan))) == plan_to_dict(plan)
    assert plan.kernels[name].source == text
    # A kernel for a card with no wait on the grid ahead says so, and is launched serialized.
    assert plan.kernels[name].dependent_launch == (load_case(CASES_DIR / COVERED).compute_cap >= ptx.GRID_WAIT_CAPABILITY)


def test_a_ptx_cubin_is_keyed_without_the_nvcc_flags(monkeypatch):
    sources, _ = _sources(COVERED, config.KERNEL_CODE_PTX_ONLY)
    [(name, op)] = sources.items()
    cuda, _ = _sources(COVERED, None)
    keys = {}
    for flags in ("", "-Xcicc -O1"):
        monkeypatch.setenv(config.NVCC_FLAGS, flags)
        keys[flags] = (nvcc._cubin_key(op.kernel_source, name, "sm_120"), nvcc._cubin_key(cuda[name].kernel_source, name, "sm_120"))
    assert keys[""][0] == keys["-Xcicc -O1"][0] and keys[""][1] != keys["-Xcicc -O1"][1]


@requires_cuda
@pytest.mark.xdist_group("cuda")
def test_a_pack_is_keyed_on_the_generator_only_off_the_default():
    from emmy.compiler.backend.pack import _environment

    with kernel_code(None):
        assert "kernel_code" not in _environment()
    with kernel_code(config.KERNEL_CODE_PTX):
        assert _environment()["kernel_code"] == config.KERNEL_CODE_PTX
