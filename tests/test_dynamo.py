"""``torch.compile(..., backend="emmy")`` against eager torch."""

import pytest

from emmy.compiler.pipeline.search.golden.format import GoldenFile
from tests.compiler.helpers import requires_cuda

torch = pytest.importorskip("torch")


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()


def _compare(fn, *inputs, **options):
    """Run ``fn`` eagerly and compiled on copies of ``inputs``; the outputs and the inputs must agree after."""
    eager_inputs, compiled_inputs = [x.clone() for x in inputs], [x.clone() for x in inputs]
    with torch.no_grad():
        expected = fn(*eager_inputs)
        actual = torch.compile(fn, backend="emmy", dynamic=False, options=options)(*compiled_inputs)
    torch.testing.assert_close(actual, expected, atol=2e-3, rtol=2e-2)
    torch.testing.assert_close(compiled_inputs, eager_inputs, atol=2e-3, rtol=2e-2)


def test_emmy_is_a_registered_dynamo_backend():
    assert "emmy" in torch._dynamo.list_backends()


def test_unknown_option_is_refused():
    compiled = torch.compile(torch.nn.Linear(4, 4), backend="emmy", options={"fast": True})
    with torch.no_grad(), pytest.raises(Exception, match="unknown emmy options"):
        compiled(torch.randn(2, 4))


@requires_cuda
@pytest.mark.parametrize("dynamic", [None, False, True])
def test_dynamic_shapes_are_refused(dynamic):
    """Emmy compiles one shape. A graph Dynamo made dynamic, ``mark_dynamic`` included, is refused rather than run."""
    compiled = torch.compile(lambda x: x.exp() + 1, backend="emmy", dynamic=dynamic)
    x = torch.randn(4, 8, device="cuda")
    torch._dynamo.mark_dynamic(x, 0)
    with torch.no_grad(), pytest.raises(Exception, match="emmy compiles static shapes"):
        compiled(x)


@requires_cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_compiled_module_matches_eager(dtype):
    model = torch.nn.Sequential(torch.nn.Linear(64, 64), torch.nn.RMSNorm(64), torch.nn.Softmax(-1)).cuda().to(dtype)
    _compare(model, torch.randn(4, 64, device="cuda", dtype=dtype))


@requires_cuda
@pytest.mark.parametrize(
    "write",
    [
        lambda x, cache: x.add_(1),
        lambda x, cache: cache[:, 1:3].copy_(x[:, :2]),
        lambda x, cache: cache[:, 2].copy_(x[:, 0]),
        lambda x, cache: (x.mul_(3), cache.mul_(2)),
    ],
    ids=["add_", "slice", "select", "two_writes"],
)
def test_in_place_writes_reach_the_callers_tensors(write):
    def fn(x, cache):
        write(x, cache)
        return x.sum(-1) + cache.sum(-1)

    _compare(fn, torch.randn(4, 4, device="cuda"), torch.randn(4, 4, device="cuda"))


@requires_cuda
def test_write_into_a_non_contiguous_input():
    def fn(x):
        x.mul_(2)
        return x.sum(0)

    _compare(fn, torch.randn(8, 4, device="cuda").t())


@requires_cuda
def test_a_returned_written_tensor_is_the_input_itself():
    def fn(x):
        return x.mul_(2)

    x = torch.randn(8, device="cuda")
    with torch.no_grad():
        assert torch.compile(fn, backend="emmy", dynamic=False)(x) is x


@requires_cuda
def test_buffer_state_persists_across_calls():
    class Counter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.register_buffer("count", torch.zeros(4))

        def forward(self, x):
            self.count.add_(1)
            return x * self.count

    eager, compiled = Counter().cuda(), Counter().cuda()
    run = torch.compile(compiled, backend="emmy", dynamic=False)
    x = torch.randn(4, device="cuda")
    with torch.no_grad():
        for _ in range(3):
            torch.testing.assert_close(run(x), eager(x))
    torch.testing.assert_close(compiled.count, eager.count)


@requires_cuda
def test_inputs_that_share_memory_after_compiling_go_unnoticed_as_with_inductor():
    """A known limit of Dynamo, not of this backend: no guard checks whether two input tensors share memory, so a
    graph compiled for separate inputs runs on overlapping ones and reads ``b`` before ``a``'s write, where eager reads
    it after. Inductor gives the same answer. If this fails, Dynamo learned to guard it; drop the test."""

    def fn(a, b):
        a.add_(1)
        return b * 2

    def aliased_call(**compile_kwargs):
        torch._dynamo.reset()
        compiled = torch.compile(fn, dynamic=False, **compile_kwargs)
        compiled(torch.zeros(4, 4, device="cuda"), torch.zeros(3, 4, device="cuda"))  # compiled for separate inputs
        a = torch.zeros(4, 4, device="cuda")
        return compiled(a, a[1:])

    with torch.no_grad():
        a = torch.zeros(4, 4, device="cuda")
        eager = fn(a, a[1:])
        emmy, inductor = aliased_call(backend="emmy"), aliased_call()
    torch.testing.assert_close(emmy, inductor)
    assert not torch.equal(emmy, eager)


@requires_cuda
def test_settings_reach_the_compile():
    model = torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.GELU()).cuda()
    _compare(model, torch.randn(4, 32, device="cuda"), knobs={"FAST_MATH": "0"}, nvcc_flags="-lineinfo")


@requires_cuda
def test_trace_option_writes_a_working_golden(tmp_path):
    path = tmp_path / "working.json"
    compiled = torch.compile(torch.nn.Linear(16, 16).cuda(), backend="emmy", dynamic=False, options={"trace": str(path)})
    with torch.no_grad():
        compiled(torch.randn(2, 16, device="cuda"))
        compiled(torch.randn(3, 16, device="cuda"))  # a second graph joins the same file
    assert GoldenFile.load(path).kernels
