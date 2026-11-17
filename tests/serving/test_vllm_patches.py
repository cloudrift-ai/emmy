"""The dummy-run seq-lens clamp (``serving/vllm_patches.py``) against a stub runner.

The real defect needs a CUDA capture warmup to fire; here we verify the patch mechanics —
over-limit seq lens are clamped on both buffers before delegation, legal ones pass through
untouched, and re-applying the patch never double-wraps.
"""

import sys
import types

import torch

from emmy.serving.vllm_patches import clamp_dummy_run_seq_lens, one_token_prompt_rows_replay_decode_graphs


class _RunnerBase:
    def __init__(self, lens: list[int], max_model_len: int):
        self.optimistic_seq_lens_cpu = torch.tensor(lens, dtype=torch.int32)
        self.seq_lens = torch.zeros(len(lens), dtype=torch.int32)
        self.max_model_len = max_model_len
        self.built = 0

    def _build_attention_metadata(self, *args, **kwargs):
        self.built += 1
        return "metadata"


def _install_stub(monkeypatch) -> type:
    class Runner(_RunnerBase):  # fresh class per test: the patch mutates it
        pass

    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("vllm.v1.worker.gpu_model_runner")
    mod.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu_model_runner", mod)
    return Runner


def test_over_model_len_dummy_seq_lens_are_clamped(monkeypatch):
    _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    runner = sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner([4128, 4128, 0], max_model_len=4096)
    assert runner._build_attention_metadata(num_tokens=1) == "metadata"
    assert runner.built == 1
    assert runner.optimistic_seq_lens_cpu.tolist() == [4096, 4096, 0]
    assert runner.seq_lens.tolist() == [4096, 4096, 0]


def test_legal_seq_lens_pass_through_untouched(monkeypatch):
    _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    runner = sys.modules["vllm.v1.worker.gpu_model_runner"].GPUModelRunner([4096, 17, 0], max_model_len=4096)
    runner._build_attention_metadata()
    assert runner.optimistic_seq_lens_cpu.tolist() == [4096, 17, 0]
    assert runner.seq_lens.tolist() == [0, 0, 0]  # no refresh issued


def test_patch_is_idempotent(monkeypatch):
    runner_cls = _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    once = runner_cls._build_attention_metadata
    clamp_dummy_run_seq_lens()
    assert runner_cls._build_attention_metadata is once


def test_newer_runner_without_optimistic_seq_lens_passes_through(monkeypatch):
    runner_cls = _install_stub(monkeypatch)
    clamp_dummy_run_seq_lens()
    runner = runner_cls([4], max_model_len=4096)
    del runner.optimistic_seq_lens_cpu
    assert runner._build_attention_metadata() == "metadata"
    assert runner.built == 1


class _V2Runner:
    """Model Runner V2's dispatch input: the stock count refuses every step that still holds a prompt token."""

    def __init__(self, speculative_steps: int):
        self.num_speculative_steps = speculative_steps

    def _get_uniform_decode_token_count(self, scheduler_output, dummy_run):
        return None


def _install_v2_stub(monkeypatch) -> type:
    class Runner(_V2Runner):  # fresh class per test: the patch mutates it
        pass

    for name in ("vllm", "vllm.v1", "vllm.v1.worker", "vllm.v1.worker.gpu"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    mod = types.ModuleType("vllm.v1.worker.gpu.model_runner")
    mod.GPUModelRunner = Runner
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.gpu.model_runner", mod)
    return Runner


def _step(*rows: int):
    return types.SimpleNamespace(num_scheduled_tokens={f"r{i}": n for i, n in enumerate(rows)})


def test_one_token_rows_replay_the_decode_graph(monkeypatch):
    runner_cls = _install_v2_stub(monkeypatch)
    one_token_prompt_rows_replay_decode_graphs()
    runner = runner_cls(speculative_steps=0)
    assert runner._get_uniform_decode_token_count(_step(1), False) == 1
    assert runner._get_uniform_decode_token_count(_step(1, 1, 1, 1), False) == 1


def test_wider_rows_and_speculative_steps_keep_the_stock_count(monkeypatch):
    runner_cls = _install_v2_stub(monkeypatch)
    one_token_prompt_rows_replay_decode_graphs()
    assert runner_cls(speculative_steps=0)._get_uniform_decode_token_count(_step(1, 175), False) is None
    assert runner_cls(speculative_steps=0)._get_uniform_decode_token_count(_step(2), False) is None
    assert runner_cls(speculative_steps=2)._get_uniform_decode_token_count(_step(1), False) is None


def test_one_token_patch_is_idempotent(monkeypatch):
    runner_cls = _install_v2_stub(monkeypatch)
    one_token_prompt_rows_replay_decode_graphs()
    once = runner_cls._get_uniform_decode_token_count
    one_token_prompt_rows_replay_decode_graphs()
    assert runner_cls._get_uniform_decode_token_count is once
