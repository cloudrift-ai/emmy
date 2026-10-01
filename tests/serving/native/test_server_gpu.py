"""Opt-in native HTTP qualification using an already prepared checkpoint artifact."""

import json
import os
import shutil
import socket
import subprocess
import time

import httpx
import pytest

pytestmark = pytest.mark.xdist_group("cuda")


def test_checkpoint_http(request):
    artifact = request.config.getoption("--native-artifact")
    binary = shutil.which("emmy-server")
    if not artifact or not binary:
        pytest.skip("requires --native-artifact and emmy-server")
    from emmy.compiler.backend.gpu_lock import gpu_lock

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    model = "Qwen/Qwen3-0.6B"
    with (
        gpu_lock(),
        subprocess.Popen(
            [binary, "--artifact", artifact, "--model", model, "--port", str(port)],
            env={**os.environ, "PATH": "/nonexistent"},
            stdout=subprocess.DEVNULL,
        ) as server,
    ):
        try:
            with httpx.Client(base_url=f"http://127.0.0.1:{port}", timeout=60) as client:
                deadline = time.monotonic() + 30
                while True:
                    assert server.poll() is None
                    try:
                        if client.get("/health").status_code == 200:
                            break
                    except httpx.ConnectError:
                        pass
                    assert time.monotonic() < deadline
                    time.sleep(0.05)
                assert client.get("/v1/models").json()["data"][0]["id"] == model
                body = {
                    "model": model,
                    "messages": [{"role": "user", "content": "Say hello in one sentence."}],
                    "max_tokens": 20,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "seed": 42,
                }
                response = client.post("/v1/chat/completions", json=body)
                response.raise_for_status()
                result = response.json()
                assert result["usage"]["prompt_tokens"] == 18
                assert result["usage"]["completion_tokens"] <= 20
                assert result["choices"][0]["finish_reason"] in ("stop", "length")
                text = result["choices"][0]["message"]["content"]
                assert "<think>" not in text
                assert text
                replay = client.post("/v1/chat/completions", json=body).json()
                assert replay["choices"] == result["choices"]
                with client.stream(
                    "POST", "/v1/chat/completions", json={**body, "stream": True, "stream_options": {"include_usage": True}}
                ) as stream:
                    stream.raise_for_status()
                    chunks = [line[6:] for line in stream.iter_lines() if line.startswith("data: ")]
                assert chunks[-1] == "[DONE]"
                values = [json.loads(chunk) for chunk in chunks[:-1]]
                assert "".join(c["choices"][0]["delta"].get("content", "") for c in values if c["choices"]) == text
                assert values[-1]["usage"] == result["usage"]
                fixed = client.post("/v1/chat/completions", json={**body, "max_tokens": 24, "ignore_eos": True}).json()
                assert fixed["usage"]["completion_tokens"] == 24
                assert fixed["choices"][0]["finish_reason"] == "length"
                stop = text[:3]
                stopped = client.post("/v1/chat/completions", json={**body, "stop": stop}).json()
                assert stopped["choices"][0]["message"]["content"] == ""
                assert stopped["choices"][0]["finish_reason"] == "stop"
                zero = client.post("/v1/completions", json={"model": model, "prompt": "Hello", "max_tokens": 0}).json()
                assert zero["usage"]["completion_tokens"] == 0
                assert zero["choices"][0]["finish_reason"] == "length"
                assert client.post("/v1/completions", json={"model": model, "prompt": "Hello", "max_tokens": 4096}).status_code == 400
                # A long output ensures the request is still active when its client disconnects.
                with client.stream(
                    "POST", "/v1/completions", json={"model": model, "prompt": "Count: 1, 2,", "max_tokens": 4000, "stream": True}
                ) as stream:
                    stream.raise_for_status()
                    for line in stream.iter_lines():
                        if line.startswith("data: "):
                            assert client.post("/v1/completions", json={"model": model, "prompt": "hi"}).status_code == 429
                            break
                deadline = time.monotonic() + 10
                while True:
                    response = client.post("/v1/completions", json={"model": model, "prompt": "hi", "max_tokens": 0})
                    if response.status_code != 429:
                        assert response.status_code == 200
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                # Closing a non-streaming request must also cancel its pending generation.
                payload = json.dumps({"model": model, "prompt": "Count: 1, 2,", "max_tokens": 4000, "ignore_eos": True}).encode()
                deadline = time.monotonic() + 10
                while True:
                    raw = socket.create_connection(("127.0.0.1", port))
                    raw.settimeout(0.1)
                    raw.sendall(
                        (
                            "POST /v1/completions HTTP/1.1\r\nHost: localhost\r\nContent-Type: application/json\r\n"
                            f"Content-Length: {len(payload)}\r\n\r\n"
                        ).encode()
                        + payload
                    )
                    try:
                        rejection = raw.recv(4096)
                    except TimeoutError:
                        break  # Accepted: a fixed 4,000-token response cannot have completed yet.
                    raw.close()
                    assert rejection.startswith(b"HTTP/1.1 429"), rejection
                    assert time.monotonic() < deadline
                try:
                    assert client.post("/v1/completions", json={"model": model, "prompt": "hi", "max_tokens": 0}).status_code == 429
                finally:
                    raw.close()
                while True:
                    response = client.post("/v1/completions", json={"model": model, "prompt": "hi", "max_tokens": 0})
                    if response.status_code != 429:
                        assert response.status_code == 200
                        break
                    assert time.monotonic() < deadline
                    time.sleep(0.02)
                with client.stream(
                    "POST",
                    "/v1/completions",
                    json={"model": model, "prompt": "Count: 1, 2,", "max_tokens": 4000, "stream": True, "ignore_eos": True},
                ) as stream:
                    stream.raise_for_status()
                    for line in stream.iter_lines():
                        if line.startswith("data: "):
                            server.terminate()
                            break
        finally:
            server.terminate()
            try:
                server.wait(timeout=10)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait()
        assert server.returncode == 0
