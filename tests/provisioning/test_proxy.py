"""The proxy a host reaches the internet through: URL validation, its environment, the Docker drop-in."""

import argparse

import pytest

from emmy.provisioning.proxy import NO_PROXY, docker_proxy_dropin, proxy_env, proxy_exports, proxy_url
from emmy.redact import redact_secrets

URL = "http://10.0.0.1:3128"


@pytest.mark.parametrize(
    "text",
    ["http://10.0.0.1:3128", "https://squid.corp:8080", "http://[fd00::1]:3128", "http://10.0.0.1:3128/", "http://user:pw@10.0.0.1:3128"],
    ids=["http-ip", "https-name", "ipv6", "trailing-slash", "credentials"],
)
def test_proxy_url_accepts_http_and_https_with_host_and_port(text):
    assert proxy_url(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "10.0.0.1:3128",
        "socks5://10.0.0.1:1080",
        "http://10.0.0.1",
        "http://:3128",
        "http://10.0.0.1:port",
        "http://10.0.0.1:70000",
        "http://10.0.0.1:3128/path",
        "http://10.0.0.1:3128?x=1",
        "",
    ],
    ids=["no-scheme", "socks", "no-port", "no-host", "port-not-a-number", "port-out-of-range", "path", "query", "empty"],
)
def test_proxy_url_rejects_everything_else(text):
    with pytest.raises(argparse.ArgumentTypeError, match="expected http://HOST:PORT or https://HOST:PORT"):
        proxy_url(text)


def test_proxy_url_with_credentials_is_redacted_from_logs():
    url = "http://relay:hunter22@10.0.0.1:3128"
    proxy_url(url)
    assert redact_secrets(f"Configuring Docker daemon proxy {url} on host") == "Configuring Docker daemon proxy *** on host"
    assert redact_secrets(URL) == URL  # no credentials, nothing to hide


def test_proxy_env_has_both_spellings():
    assert proxy_env(URL) == {
        "HTTP_PROXY": URL,
        "HTTPS_PROXY": URL,
        "NO_PROXY": NO_PROXY,
        "http_proxy": URL,
        "https_proxy": URL,
        "no_proxy": NO_PROXY,
    }


def test_proxy_exports_prefixes_a_shell_command():
    assert proxy_exports(URL) == (
        f"export HTTP_PROXY={URL} HTTPS_PROXY={URL} NO_PROXY={NO_PROXY} http_proxy={URL} https_proxy={URL} no_proxy={NO_PROXY};"
    )


def test_docker_proxy_dropin_content():
    assert docker_proxy_dropin(URL) == (
        "[Service]\n"
        f'Environment="HTTP_PROXY={URL}" "HTTPS_PROXY={URL}" '
        '"NO_PROXY=localhost,127.0.0.1,::1,10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"\n'
    )
