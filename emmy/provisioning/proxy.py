"""An HTTP proxy for a host that reaches the internet only through one.

Such a host pulls images, downloads model weights and installs packages through the proxy. The Docker
daemon reads it from a systemd drop-in; every other process reads it from the environment, where Go and
Docker honour the upper-case spelling and curl only the lower-case one, so both are always set.
"""

import argparse
import shlex
from urllib.parse import urlsplit

from emmy.redact import register_secret

# Never sent through the proxy: the host itself and the private ranges a VLAN or a Docker network lives in.
NO_PROXY = "localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"
DOCKER_PROXY_DROPIN = "/etc/systemd/system/docker.service.d/http-proxy.conf"


def proxy_url(text: str) -> str:
    """argparse type: ``http://HOST:PORT`` or ``https://HOST:PORT``; credentials are allowed and kept out of logs."""
    parts = urlsplit(text)
    try:
        port = parts.port
    except ValueError:
        port = None
    if parts.scheme not in ("http", "https") or not parts.hostname or port is None or parts.path not in ("", "/"):
        raise argparse.ArgumentTypeError(f"expected http://HOST:PORT or https://HOST:PORT, got {text!r}")
    if parts.query or parts.fragment:
        raise argparse.ArgumentTypeError(f"expected http://HOST:PORT or https://HOST:PORT, got {text!r}")
    if parts.password:
        register_secret(text)
    return text


def add_proxy_argument(parser) -> None:
    """Register ``--vm-proxy URL`` on a deploy target that reaches a host over SSH."""
    parser.add_argument(
        "--vm-proxy",
        type=proxy_url,
        default=None,
        metavar="URL",
        help="HTTP proxy the target host reaches the internet through (http://HOST:PORT or https://HOST:PORT); "
        "the Docker daemon, the weight download and package installs on the host all use it",
    )


def proxy_env(url: str) -> dict[str, str]:
    """The proxy as environment variables, in both spellings."""
    upper = {"HTTP_PROXY": url, "HTTPS_PROXY": url, "NO_PROXY": NO_PROXY}
    return {**upper, **{name.lower(): value for name, value in upper.items()}}


def proxy_exports(url: str) -> str:
    """A shell prefix that exports the proxy to the command after it."""
    return "export " + " ".join(f"{name}={shlex.quote(value)}" for name, value in proxy_env(url).items()) + ";"


def docker_proxy_dropin(url: str) -> str:
    """The systemd drop-in that points the Docker daemon at the proxy."""
    return f'[Service]\nEnvironment="HTTP_PROXY={url}" "HTTPS_PROXY={url}" "NO_PROXY={NO_PROXY}"\n'
