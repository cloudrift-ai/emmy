"""Tests for the Host abstraction and driver/CUDA provisioning."""

import asyncio

import click
import pytest

from emmy.provisioning.host import LocalHost, RemoteHost
from emmy.provisioning.proxy import DOCKER_PROXY_DROPIN, NO_PROXY, docker_proxy_dropin
from emmy.provisioning.remote import _configure_docker_proxy, _ensure_nvidia_versions, _matches, provision_remote

PROXY = "http://10.0.0.1:3128"


def test_matches_prefix():
    assert _matches("595.58.03", "595")
    assert _matches("550.127.05", "550")
    assert not _matches("595.58.03", "550")
    assert _matches("12.4.1", "12.4")
    assert not _matches("12.5.0", "12.4")
    assert not _matches(None, "550")


def test_local_host_sudo_raises():
    host = LocalHost()
    with pytest.raises(click.ClickException, match="Refusing to run privileged"):
        asyncio.run(host.run("apt-get update", sudo=True))


def test_local_host_non_sudo_runs():
    host = LocalHost()
    rc, out = asyncio.run(host.run("echo hello", capture=True))
    assert rc == 0
    assert out == "hello"


def test_local_host_dry_run():
    host = LocalHost(dry_run=True)
    # dry-run still refuses sudo
    with pytest.raises(click.ClickException):
        asyncio.run(host.run("apt-get update", sudo=True))
    # non-sudo dry-run is a no-op
    rc, out = asyncio.run(host.run("echo hello"))
    assert rc == 0


def test_remote_host_dry_run_logs(caplog):
    host = RemoteHost("user@host", None, 22, dry_run=True)
    with caplog.at_level("INFO"):
        rc, _ = asyncio.run(host.run("apt-get update", sudo=True))
    assert rc == 0
    assert any("sudo apt-get update" in r.message for r in caplog.records)


def test_remote_host_exports_its_proxy_to_every_command(caplog):
    host = RemoteHost("user@host", None, 22, dry_run=True, proxy=PROXY)
    with caplog.at_level("INFO"):
        asyncio.run(host.run("apt-get update", sudo=True))
        asyncio.run(host.run("nvidia-smi"))
    exports = (
        f"export HTTP_PROXY={PROXY} HTTPS_PROXY={PROXY} NO_PROXY={NO_PROXY} http_proxy={PROXY} https_proxy={PROXY} no_proxy={NO_PROXY};"
    )
    assert [r.message for r in caplog.records] == [
        f"[dry-run] ssh user@host: sudo {exports} apt-get update",
        f"[dry-run] ssh user@host: {exports} nvidia-smi",
    ]


class _ProxyHost(LocalHost):
    """A host behind PROXY whose drop-in and proxy check answer as scripted; every other command succeeds."""

    def __init__(self, dropin: str | None = None, curl=(0, "401")):
        super().__init__()
        self.proxy = PROXY
        self.dropin = dropin
        self.curl = curl
        self.calls: list[tuple[str, bool]] = []

    async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
        self.calls.append((cmd, sudo))
        if cmd.startswith(f"cat {DOCKER_PROXY_DROPIN}"):
            return (0, self.dropin) if self.dropin is not None else (1, "")
        if cmd.startswith("curl -sS"):
            return self.curl
        return 0, ""


def test_configure_docker_proxy_writes_the_dropin_and_restarts_the_daemon(caplog):
    host = _ProxyHost()
    with caplog.at_level("INFO"):
        asyncio.run(_configure_docker_proxy(host, dry_run=False))
    ((write, sudo),) = [(cmd, sudo) for cmd, sudo in host.calls if "printf" in cmd]
    assert sudo
    assert docker_proxy_dropin(PROXY) in write  # the exact content, single-quoted for the shell
    assert write.endswith(f"> {DOCKER_PROXY_DROPIN} && systemctl daemon-reload && systemctl restart docker")
    messages = [r.message for r in caplog.records]
    assert f"Configuring Docker daemon proxy {PROXY} on local" in messages
    assert f"local: proxy {PROXY} reaches https://registry-1.docker.io/v2/ (HTTP 401)" in messages


def test_configure_docker_proxy_leaves_a_matching_dropin_alone(caplog):
    host = _ProxyHost(dropin=docker_proxy_dropin(PROXY).strip())
    with caplog.at_level("INFO"):
        asyncio.run(_configure_docker_proxy(host, dry_run=False))
    assert not [cmd for cmd, sudo in host.calls if sudo]
    assert f"Docker daemon proxy {PROXY} already configured on local" in [r.message for r in caplog.records]


def test_configure_docker_proxy_rewrites_a_dropin_naming_another_proxy():
    host = _ProxyHost(dropin=docker_proxy_dropin("http://10.0.0.2:3128").strip())
    asyncio.run(_configure_docker_proxy(host, dry_run=False))
    assert [cmd for cmd, sudo in host.calls if sudo and "systemctl restart docker" in cmd]


def test_configure_docker_proxy_fails_the_deploy_when_the_proxy_does_not_answer():
    host = _ProxyHost(curl=(7, "curl: (7) Failed to connect to 10.0.0.1 port 3128: Connection refused"))
    with pytest.raises(RuntimeError, match=r"local: proxy http://10\.0\.0\.1:3128 does not reach .*Connection refused"):
        asyncio.run(_configure_docker_proxy(host, dry_run=False))


def test_configure_docker_proxy_warns_only_when_the_proxy_is_a_hostname(caplog):
    named = _ProxyHost()
    named.proxy = "http://squid.corp:3128"
    with caplog.at_level("WARNING"):
        asyncio.run(_configure_docker_proxy(named, dry_run=False))
        assert any("the host itself must be able to resolve squid.corp" in r.message for r in caplog.records)
        caplog.clear()
        asyncio.run(_configure_docker_proxy(_ProxyHost(), dry_run=False))
    assert not caplog.records


def test_provision_remote_configures_the_proxy_once_docker_is_there():
    host = _ProxyHost()
    asyncio.run(provision_remote(host, skip_nvidia=True))
    commands = [cmd for cmd, _ in host.calls]
    assert commands.index("command -v docker") < commands.index(f"cat {DOCKER_PROXY_DROPIN} 2>/dev/null")
    plain = _ProxyHost()
    plain.proxy = None
    asyncio.run(provision_remote(plain, skip_nvidia=True))
    assert not [cmd for cmd, _ in plain.calls if "proxy" in cmd]


def test_remote_host_build_args_includes_sudo():
    host = RemoteHost("user@host", "/tmp/key", 2222, dry_run=True)
    args = host._build_args("ls")
    assert "ssh" in args[0]
    assert args[-1] == "ls"
    assert "user@host" in args
    assert "-p" in args and "2222" in args
    assert "-i" in args and "/tmp/key" in args


def test_ensure_nvidia_skip_when_matching(caplog):
    """If installed driver matches requested, no sudo is invoked."""

    class FakeHost(LocalHost):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[str, bool]] = []

        async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
            self.calls.append((cmd, sudo))
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            if sudo:
                raise AssertionError(f"sudo unexpectedly invoked: {cmd}")
            return 0, ""

    host = FakeHost()
    with caplog.at_level("INFO"):
        installed = asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))
    assert installed is False
    assert any("already matches" in r.message for r in caplog.records)


def test_ensure_nvidia_install_on_mismatch_raises_locally():
    """LocalHost rejects the sudo install command when driver mismatches."""

    class FakeHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600):
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            return await super().run(cmd, sudo=sudo, capture=capture, timeout=timeout)

    host = FakeHost()
    with pytest.raises(click.ClickException, match="cuda"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="550", cuda_version=None))


def test_ensure_nvidia_install_failure_raises():
    """Regression: when `apt-get install cuda-drivers-...` exits non-zero
    (e.g. unmet dependencies on a CloudRift base image), the harness must
    raise loudly rather than silently proceeding to a half-installed state.

    Before the fix, `_ensure_nvidia_versions` ignored the exit code and
    returned `installed=True`, the caller rebooted, and the bench produced
    empty result tables an hour later with no indication why.
    """
    from emmy.provisioning.remote import _ensure_nvidia_versions

    class FailingAptHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            self_cmd = cmd
            if "nvidia-smi" in self_cmd:
                return 0, "550.54.15"  # mismatched driver triggers install path
            if "test -d /usr/local/cuda" in self_cmd:
                return 1, ""  # cuda toolkit not present
            if "cuda-keyring" in self_cmd or "apt-get update" in self_cmd:
                return 0, ""
            if "dpkg --configure" in self_cmd or "fix-broken" in self_cmd:
                return 0, ""
            if "apt-get install" in self_cmd and "nvidia-open" in self_cmd:
                return (
                    100,
                    "E: Unmet dependencies. Try 'apt --fix-broken install' with no packages",
                )
            return 0, ""

    host = FailingAptHost()
    with pytest.raises(RuntimeError, match="nvidia-open.*failed"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))


def test_ensure_nvidia_cuda_install_silent_failure_raises():
    """Regression: apt-get install of cuda-toolkit-X-Y can return rc=0 yet not
    actually create /usr/local/cuda-X.Y (kept-back packages, etc.). The post-
    install dir check must catch this and raise."""
    from emmy.provisioning.remote import _ensure_nvidia_versions

    class SilentlyFailingHost(LocalHost):
        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            if "nvidia-smi" in cmd:
                return 0, "595.58.03"
            if "test -d /usr/local/cuda" in cmd:
                return 1, ""  # never present, even after install
            if "cuda-keyring" in cmd or "apt-get update" in cmd:
                return 0, ""
            if "dpkg --configure" in cmd or "fix-broken" in cmd:
                return 0, ""
            if "apt-get install" in cmd:
                return 0, ""  # apt lies and exits 0
            return 0, ""

    host = SilentlyFailingHost()
    with pytest.raises(RuntimeError, match="reported success but"):
        asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version="13.2"))


def test_ensure_nvidia_purges_old_packages_first():
    """Regression: when installing a new driver, the harness must purge any
    pre-existing nvidia/libnvidia packages first. CloudRift base images ship
    nvidia-driver-510 from the Ubuntu archive; without a purge step, the
    cuda repo's libnvidia-* (= 595.58.03-1ubuntu1) cannot unpack over the
    older files and the install fails."""
    from emmy.provisioning.remote import _ensure_nvidia_versions

    class TrackingHost(LocalHost):
        def __init__(self):
            super().__init__()
            self.commands: list[str] = []

        async def run(self, cmd, *, sudo=False, capture=False, timeout=600, **kwargs):
            self.commands.append(cmd)
            if "nvidia-smi" in cmd:
                return 0, "510.47.03"  # old version triggers install
            if "test -d /usr/local/cuda" in cmd:
                return 0, ""  # cuda toolkit not requested in this test
            if "cuda-keyring" in cmd or "apt-get update" in cmd:
                return 0, ""
            if "dpkg --configure" in cmd or "fix-broken" in cmd:
                return 0, ""
            if "apt-get purge" in cmd:
                return 0, ""
            if "apt-get install" in cmd and "nvidia-open" in cmd:
                return 0, ""
            return 0, ""

    host = TrackingHost()
    asyncio.run(_ensure_nvidia_versions(host, driver_version="595", cuda_version=None))
    # Find the install command and the purge command, ensure purge came first.
    purge_idx = next((i for i, c in enumerate(host.commands) if "apt-get purge" in c and "nvidia-*" in c), -1)
    install_idx = next((i for i, c in enumerate(host.commands) if "apt-get install" in c and "nvidia-open" in c), -1)
    assert purge_idx >= 0, f"purge step never invoked. commands: {host.commands}"
    assert install_idx >= 0, f"install step never invoked. commands: {host.commands}"
    assert purge_idx < install_idx, f"purge must run before install (purge at {purge_idx}, install at {install_idx})"
