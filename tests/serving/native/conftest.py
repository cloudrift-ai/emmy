"""Opt-in checkpoint qualification; ordinary tests use a hermetic tiny model."""


def pytest_addoption(parser):
    parser.addoption("--native-checkpoint", help="Local dense Qwen3 checkpoint for full-model native qualification")
    parser.addoption("--native-artifact", help="Prepared native artifact matching --native-checkpoint")
    parser.addoption("--native-decode-steps", type=int, default=15, help="Checkpoint decode steps after prefill")
