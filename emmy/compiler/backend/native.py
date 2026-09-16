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


class PythonPackWorker(_AsyncBenchWorker):
    """Reference the identical artifact through the existing persistent Python dispatcher."""

    _ATTEMPTS = 1

    @staticmethod
    def _encode(request: dict) -> bytes:
        return _AsyncBenchWorker._encode({"pack_command": request})


class PackReference:
    """Worker-local reference state; never used for native model execution."""

    def __init__(self):
        self.program = None
        self.plan = None
        self.stream = None

    def command(self, request: dict) -> dict:
        from pathlib import Path
        from time import perf_counter

        import cupy as cp
        import numpy as np

        from emmy.compiler.backend.cuda.program import CompiledProgram
        from emmy.compiler.backend.plan import plan_from_dict

        started = perf_counter()
        op = request["op"]
        if op == "release":
            if self.stream is not None:
                self.stream.synchronize()
            self.program = self.plan = self.stream = None
            cp.get_default_memory_pool().free_all_blocks()
            return {"released": True}
        if op == "load":
            self.command({"op": "release"})
            root = Path(request["root"]).resolve()

            def member(rel):
                path = (root / rel).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("artifact member escapes bundle")
                return path

            manifest = json.loads(member("manifest.json").read_text())
            if manifest.get("format") != 1 or manifest.get("standalone") != 1:
                raise ValueError("unsupported standalone pack format")
            arch = cp.cuda.Device().compute_capability
            if manifest["environment"]["arch"] != f"sm_{arch}":
                raise ValueError("artifact GPU architecture mismatch")
            name = request["program"]
            self.plan = plan_from_dict(json.loads(member(manifest["programs"][name]).read_text()))
            by_name = {b.name: b for b in self.plan.buffers}
            data = {
                key: np.frombuffer(member(rel).read_bytes(), dtype=by_name[key].dtype.np).reshape(by_name[key].resolve_shape({}))
                for key, rel in manifest["bindings"][name].items()
            }
            if not set(self.plan.inputs) <= data.keys():
                raise ValueError("reference benchmark requires bound inputs")
            self.stream = cp.cuda.Stream(non_blocking=True)
            with self.stream:
                self.program = CompiledProgram.build_from_plan(self.plan, data, cubin_dir=root / "cubin")
            self.stream.synchronize()
            return {"loaded": True, "load_ms": (perf_counter() - started) * 1000, "load_times_ms": self.program.load_times_ms}
        if self.program is None:
            raise ValueError("no loaded program")
        if op != "run":
            raise ValueError(f"unsupported reference operation: {op}")
        warmup, iterations, capture = request["warmup"], request["iterations"], request["capture"]
        if not 0 <= warmup <= 1_000_000 or not 1 <= iterations <= 1_000_000:
            raise ValueError("invalid iteration count")
        with self.stream:
            phase = perf_counter()
            if capture:
                self.program.run_once()
                self.stream.synchronize()
                self.program.capture_program_graph()
            preparation_ms = (perf_counter() - phase) * 1000
            execute = self.program.replay_program_graph if capture else self.program.run_once
            phase = perf_counter()
            for _ in range(warmup):
                execute()
            self.stream.synchronize()
            warmup_ms = (perf_counter() - phase) * 1000
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record()
            phase = perf_counter()
            for _ in range(iterations):
                execute()
            end.record()
            submission_ms = (perf_counter() - phase) * 1000
            phase = perf_counter()
            end.synchronize()
            time_ms = cp.cuda.get_elapsed_time(start, end) / iterations
            completion_wait_ms = (perf_counter() - phase) * 1000
            phase = perf_counter()
            for name, path in request["outputs"].items():
                if name not in self.plan.outputs:
                    raise ValueError(f"unknown output {name}")
                Path(path).write_bytes(cp.asnumpy(self.program.arrays[name]).tobytes())
            output_ms = (perf_counter() - phase) * 1000
        return {
            "time_ms": time_ms, "captured": capture, "run_ms": (perf_counter() - started) * 1000, "output_ms": output_ms,
            "metrics": {"preparation_ms": preparation_ms, "warmup_ms": warmup_ms,
                        "submission_ms": submission_ms, "completion_wait_ms": completion_wait_ms},
        }


async def benchmark_pack(root, *, warmup: int, iterations: int, repeats: int = 3, executable: str | None = None) -> dict:
    """Measure the same static artifact with both dispatchers and worker lifetimes.

    Event windows exclude file I/O and IPC. Parent round trips include those costs. Load
    measurements combine module loading, allocation, and initial upload. No model speedup
    is inferred here; outputs are checked directly against the Python dispatcher.
    """
    import tempfile
    from pathlib import Path
    from time import perf_counter

    import numpy as np

    from emmy.compiler.backend.plan import plan_from_dict

    if not 0 <= warmup <= 1_000_000 or not 1 <= iterations <= 1_000_000 or repeats < 1:
        raise ValueError("invalid benchmark counts")
    root = Path(root).resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    rows = []
    reloads = []
    with tempfile.TemporaryDirectory(prefix="emmy-runtime-") as temporary:
        for program, relative in manifest["programs"].items():
            plan = plan_from_dict(json.loads((root / relative).read_text()))
            buffers = {b.name: b for b in plan.buffers}
            reference = {}
            for runtime in ("python", "rust"):
                for capture in (False, True):
                    for lifetime in ("persistent", "one-shot"):
                        worker = NativeWorker(executable=executable) if runtime == "rust" else PythonPackWorker()
                        try:
                            for repeat in range(repeats):
                                load_ms = None
                                load_result = None
                                if lifetime == "one-shot" or repeat == 0:
                                    before = perf_counter()
                                    load_result = await worker.run_job({"op": "load", "root": str(root), "program": program}, wall_timeout_s=60)
                                    load_ms = (perf_counter() - before) * 1000
                                outputs = {name: str(Path(temporary) / f"output-{i}.bin") for i, name in enumerate(plan.outputs)}
                                before = perf_counter()
                                result = await worker.run_job(
                                    {"op": "run", "warmup": warmup, "iterations": iterations, "capture": capture, "outputs": outputs},
                                    wall_timeout_s=60,
                                )
                                roundtrip_ms = (perf_counter() - before) * 1000
                                for name, path in outputs.items():
                                    value = np.frombuffer(Path(path).read_bytes(), dtype=buffers[name].dtype.np)
                                    if name not in reference:
                                        reference[name] = value.copy()
                                    np.testing.assert_array_equal(value, reference[name], err_msg=f"{runtime}/{program}/{name}")
                                rows.append({
                                    "program": program, "runtime": runtime, "capture": capture, "lifetime": lifetime, "repeat": repeat,
                                    "load_roundtrip_ms": load_ms, "run_roundtrip_ms": roundtrip_ms, "time_ms": result["time_ms"],
                                    "outputs_equal": True,
                                    "load": load_result, "run": result,
                                })
                                if lifetime == "one-shot":
                                    await worker.aclose()
                            if lifetime == "persistent":
                                before = perf_counter()
                                result = await worker.run_job({"op": "load", "root": str(root), "program": program}, wall_timeout_s=60)
                                reloads.append({"program": program, "runtime": runtime, "capture": capture,
                                                "roundtrip_ms": (perf_counter() - before) * 1000, "load": result})
                        finally:
                            await worker.aclose()
    return {"format": 1, "artifact": str(root), "warmup": warmup, "iterations": iterations, "repeats": repeats,
            "rows": rows, "reloads": reloads}
