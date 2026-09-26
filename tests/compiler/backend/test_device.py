"""Runtime contexts follow the host's selected logical GPU without requiring multiple cards."""

import sys
from unittest.mock import Mock, call

import pytest
import torch

from emmy import emmy_runtime
from emmy.compiler.backend.cuda import device as devices


@pytest.fixture
def runtime_devices(monkeypatch):
    monkeypatch.setattr(devices, "_DEVICES", {})
    factory = Mock(side_effect=lambda ordinal: Mock(ordinal=ordinal))
    monkeypatch.setattr(emmy_runtime, "Device", factory)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    return factory


def test_current_device_selects_and_reuses_context(runtime_devices, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    seventh = devices.device()
    assert seventh.ordinal == 7
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    assert devices.device().ordinal == 0
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    assert devices.device() is seventh
    assert runtime_devices.call_args_list == [call(7), call(0)]


@pytest.mark.parametrize("missing_torch", [False, True])
def test_runtime_without_torch_cuda_uses_visible_device_zero(runtime_devices, monkeypatch, missing_torch):
    if missing_torch:
        monkeypatch.setitem(sys.modules, "torch", None)
    else:
        monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=AssertionError("CUDA unavailable")))
    assert devices.device().ordinal == 0
    runtime_devices.assert_called_once_with(0)


def test_failed_host_selection_does_not_fall_back_to_zero(runtime_devices, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", Mock(side_effect=RuntimeError("device selection failed")))
    with pytest.raises(RuntimeError, match="device selection failed"):
        devices.device()
    assert devices.compute_capability() is None
    runtime_devices.assert_not_called()


def test_context_probe_does_not_create_a_context(runtime_devices, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    assert not devices.context_poisoned()
    runtime_devices.assert_not_called()
    devices.device()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 3)
    assert not devices.context_poisoned()
    runtime_devices.assert_called_once_with(7)


def test_context_probe_checks_only_the_selected_device(runtime_devices, monkeypatch):
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    bad = devices.device()
    bad.synchronize.side_effect = RuntimeError("poisoned")
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    good = devices.device()
    assert not devices.context_poisoned()
    good.synchronize.assert_called_once_with()
    bad.synchronize.assert_not_called()
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 7)
    assert devices.context_poisoned()
