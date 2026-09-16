"""Native execution through the benchmark worker's shared deadline and process ownership."""

from __future__ import annotations

import json
import shutil

from emmy.compiler.backend.cuda.program import _AsyncBenchWorker


class NativeWorker(_AsyncBenchWorker):
    """One trusted static program per worker; failures retire it without replaying a job."""

    _ATTEMPTS = 1

    def __init__(self, *, executable: str | None = None, device_id: int | None = None):
        super().__init__(device_id=device_id)
        self.executable = executable or shutil.which("emmy-runtime-worker")
        if not self.executable:
            raise FileNotFoundError("emmy-runtime-worker is not installed on PATH; build and install the matching Cargo binary")

    def _command(self) -> list[str]:
        return [self.executable]

    @staticmethod
    def _encode(request: dict) -> bytes:
        payload = json.dumps({"version": 1, "command": request}).encode()
        if len(payload) > 1024 * 1024:
            raise ValueError("native control frame exceeds 1 MiB")
        return payload

    @staticmethod
    def _decode(body: bytes) -> dict:
        response = json.loads(body)
        if response.get("version") != 1:
            raise ValueError("unsupported native response version")
        if "error" in response:
            return {"ok": False, "error": response["error"], "_retire_worker": True}
        return {"ok": True, **response["result"]}
