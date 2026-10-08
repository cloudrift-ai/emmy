"""Deploy orchestration: run_deploy, run_teardown, deploy, teardown."""

import asyncio
import base64
import json
import logging
import math
import re
import shlex
import struct
import zlib

from emmy.deploy.compose import generate_compose, generate_nginx_conf, service_name
from emmy.deploy.log_phases import decompose_model_load, parse_engine_load_phases
from emmy.deploy.params import DeployParams, Service
from emmy.provisioning.proxy import proxy_env
from emmy.provisioning.ssh_transport import make_run_cmd, make_write_file
from emmy.recipe.types import Recipe
from emmy.timing import (
    PHASE_IMAGE_PULL,
    PHASE_MODEL_DOWNLOAD,
    PHASE_MODEL_LOAD_AND_WARMUP,
    PHASE_SMOKE_TEST,
    PhaseTimer,
)

logger = logging.getLogger(__name__)

# The window one service may take to answer /health: container start + weight load into
# GPU + CUDA graph capture + warmup. Polled every HEALTH_INTERVAL seconds.
HEALTH_TIMEOUT = 3600
HEALTH_INTERVAL = 10
SMOKE_TIMEOUT = 600
SMOKE_INTERVAL = 10


def _solid_png(size: int, rgb: tuple[int, int, int]) -> bytes:
    """A one-color RGB PNG from the format's three chunks, so the fixture needs no imaging dependency."""

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))

    header = struct.pack(">IIBBBBB", size, size, 8, 2, 0, 0, 0)  # 8-bit RGB, no interlace
    rows = (b"\x00" + bytes(rgb) * size) * size  # filter byte 0 before each row
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b"")


# The image smoke test's fixture: a red square above any vision processor's minimum size.
_RED_SQUARE_DATA_URL = "data:image/png;base64," + base64.b64encode(_solid_png(128, (255, 0, 0))).decode()


def _tone_wav(seconds: float, hz: int, rate: int = 16000) -> bytes:
    """A mono 16-bit PCM WAV of one sine tone, so the fixture needs no audio dependency."""
    samples = b"".join(struct.pack("<h", int(8000 * math.sin(2 * math.pi * hz * i / rate))) for i in range(int(seconds * rate)))
    fmt = struct.pack("<HHIIHH", 1, 1, rate, rate * 2, 2, 16)
    body = b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt + b"data" + struct.pack("<I", len(samples)) + samples
    return b"RIFF" + struct.pack("<I", len(body)) + body


# The audio smoke test's fixture: one second of a 440 Hz tone at the 16 kHz rate speech encoders resample to.
_TONE_WAV_BASE64 = base64.b64encode(_tone_wav(1.0, 440)).decode()


async def baked_hf_cache(run_cmd, image):
    """The image's own HF cache directory, when it ships one — else None.

    A prebuilt per-model serving image (``docker/vllm-emmy-serve/``) bakes the model
    snapshot under its own HF_HOME and sets HF_HUB_OFFLINE=1, so it never reaches the
    hub. That pair is the image declaring itself self-contained: pointing HF_HOME at a
    host cache instead hides the baked snapshot while offline mode stays on, which makes
    the download step fail outright — and would re-fetch the full weights if it didn't.
    """
    fmt = "{{range .Config.Env}}{{println .}}{{end}}"
    # stream=False is what makes run_cmd return stdout rather than pass it through.
    rc, out, _ = await run_cmd(f"docker image inspect {image} --format '{fmt}'", stream=False, timeout=60)
    if rc != 0 or not out:
        return None
    env = dict(line.split("=", 1) for line in out.splitlines() if "=" in line)
    return env.get("HF_HOME") if env.get("HF_HUB_OFFLINE") == "1" else None


async def _fail_with_logs(run_cmd, message: str) -> bool:
    """Log a failed deploy step with the containers' last log lines; always False."""
    logger.error(message)
    logger.error("Container logs:")
    await run_cmd("docker compose logs --tail=100", timeout=60, log_output=True)
    return False


def _share_device(a: Service, b: Service) -> bool:
    """Whether two services may see the same GPU (a service without device ids sees every GPU)."""
    return a.gpu_device_ids is None or b.gpu_device_ids is None or bool(set(a.gpu_device_ids) & set(b.gpu_device_ids))


async def _wait_healthy(run_cmd, name: str, port: int, dry_run: bool) -> bool:
    """Poll a service's /health on its host port until it answers, or fail when a service exits.

    Short polls rather than one long ``compose up --wait``: a reset SSH connection then costs
    one probe, not the deploy. A container that exits before it is healthy fails at once.
    """
    logger.info(f"Waiting for {name} to become healthy...")
    elapsed = 0
    while elapsed < HEALTH_TIMEOUT:
        rc, _, _ = await run_cmd(f"curl -sf http://localhost:{port}/health", stream=False, timeout=30, log_output=True)
        if rc == 0 or dry_run:
            return True
        rc_ps, exited, _ = await run_cmd("docker compose ps -q --status exited --status dead", stream=False, timeout=30)
        if rc_ps == 0 and exited.strip():
            return await _fail_with_logs(run_cmd, f"A service exited before {name} became healthy")
        await asyncio.sleep(HEALTH_INTERVAL)
        elapsed += HEALTH_INTERVAL
    logger.error(f"Health check for {name} timed out after {HEALTH_TIMEOUT}s")
    return False


def _request(recipe: Recipe, *, example: bool = False) -> tuple[str, dict]:
    """The endpoint path and JSON body a recipe answers: embeddings, completion, or
    chat. The smoke test asks for 2+2; the printed curl example says hello."""
    model = recipe.request_model_name
    if recipe.is_embedding:
        return "/v1/embeddings", {"model": model, "input": "Hello" if example else "What is 2+2?"}
    if recipe.model.smoke_test == "completion":
        if example:
            return "/v1/completions", {"model": model, "prompt": "Hello", "max_tokens": 64}
        return "/v1/completions", {"model": model, "prompt": "2 + 2 =", "max_tokens": 16, "temperature": 0}
    content = "Hello" if example else "What is 2+2? Answer with just the number."
    return "/v1/chat/completions", {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 64 if example else 128,
    }


def _image_request(recipe: Recipe) -> tuple[str, dict]:
    """The chat request the image smoke test sends: name the color of an inline red square."""
    return "/v1/chat/completions", {
        "model": recipe.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What color is this image? Answer with one word."},
                    {"type": "image_url", "image_url": {"url": _RED_SQUARE_DATA_URL}},
                ],
            }
        ],
        "max_tokens": 128,
    }


def _audio_request(recipe: Recipe) -> tuple[str, dict]:
    """The chat request the audio smoke test sends: describe an inline tone."""
    return "/v1/chat/completions", {
        "model": recipe.model_name,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "input_audio", "input_audio": {"data": _TONE_WAV_BASE64, "format": "wav"}},
                    {"type": "text", "text": "Describe this audio in a few words."},
                ],
            }
        ],
        "max_tokens": 128,
    }


async def _smoke_test(run_cmd, service: Service, name: str, check_smoke_output: bool) -> bool:
    """Probe one service until it answers (the first request may be slow after warmup).

    A standalone deploy checks model-specific content, then sends one inline image and one inline
    audio clip when the recipe declares those inputs; benchmark callers request transport
    readiness only and retain the response for later review.
    """
    path, body = _request(service.recipe)
    check = _smoke_response_check(service.recipe, check_smoke_output=check_smoke_output)
    if not await _probe(run_cmd, service.port, path, body, check, name, log_response=not check_smoke_output):
        return False
    modalities = service.recipe.model.input_modalities
    if check_smoke_output and "image" in modalities:
        path, body = _image_request(service.recipe)
        if not await _probe(run_cmd, service.port, path, body, _check_image_response, f"{name} (image input)"):
            return False
    if check_smoke_output and "audio" in modalities:
        path, body = _audio_request(service.recipe)
        return await _probe(run_cmd, service.port, path, body, _check_audio_response, f"{name} (audio input)")
    return True


async def _probe(run_cmd, port: int, path: str, body: dict, check, name: str, *, log_response: bool = False) -> bool:
    """Send one request until the service answers it, then judge the answer with ``check``."""
    # The body is a shell word: quote it so no recipe value can end the string early.
    data = shlex.quote(json.dumps(body))
    smoke_cmd = f"curl --fail-with-body -s http://localhost:{port}{path} -H 'Content-Type: application/json' -d {data}"
    deadline = asyncio.get_event_loop().time() + SMOKE_TIMEOUT
    while asyncio.get_event_loop().time() < deadline:
        rc, stdout, _ = await run_cmd(smoke_cmd, stream=False, timeout=180)
        if rc != 0 or not stdout.strip():
            # Server not ready yet, keep retrying
            await asyncio.sleep(SMOKE_INTERVAL)
            continue
        if log_response:
            logger.info("Smoke response: %s", stdout)
        verdict, detail = check(stdout)
        if verdict == "retry":
            # Malformed/empty response, server may still be starting
            await asyncio.sleep(SMOKE_INTERVAL)
            continue
        if verdict == "pass":
            logger.info(f"Smoke test passed for {name}.")
            return True
        # Valid response, wrong content — broken model
        return await _fail_with_logs(run_cmd, f"Smoke test failed for {name}: {detail}")
    return await _fail_with_logs(run_cmd, f"Smoke test for {name} timed out after {SMOKE_TIMEOUT}s. The endpoint is not ready.")


async def run_deploy(
    run_cmd,
    write_file,
    services: list[Service],
    model_dir,
    hf_token,
    host,
    dry_run=False,
    load_balancer=False,
    port_mappings=None,
    timer: PhaseTimer | None = None,
    check_smoke_output: bool = True,
    proxy: str | None = None,
):
    """Shared deploy orchestration.

    Args:
        run_cmd: async callable(command, stream=True, timeout=600) -> (returncode, stdout, stderr)
        write_file: async callable(path, content) -> None - writes file to target
        services: the engine containers of this deployment, in start order
        model_dir: model cache directory path
        hf_token: HuggingFace token
        host: hostname/IP for endpoint display
        dry_run: if True, skip sleep in health polling
        load_balancer: put nginx on 8080 in front of every service (replica fan-out)
        port_mappings: optional list of (internal, external) port tuples
        timer: optional PhaseTimer; deploy step durations are recorded into it. A
            throwaway timer is used when None, so the timing log lines still print.
        check_smoke_output: whether a valid smoke response must also satisfy the
            model-specific deployment check. Benchmark orchestration disables this
            semantic judgment and uses the probe only for API readiness.
        proxy: HTTP proxy URL the host reaches the internet through; the weight download and every
            engine service get it in their environment
    """
    timer = timer or PhaseTimer()
    names = [service_name(index, service) for index, service in enumerate(services)]

    # Generate and write compose file, and the nginx config when there is a load balancer
    await write_file("docker-compose.yaml", generate_compose(services, model_dir, hf_token, load_balancer, proxy=proxy))
    if load_balancer:
        await write_file("nginx.conf", generate_nginx_conf(names))

    # Step 1: Pull images. --ignore-pull-failures keeps a locally-built image
    # usable before it's pushed to a registry (e.g. testing a fresh
    # vllm-emmy build with `bench --local`), but compose's exit code is
    # unreliable across versions (non-zero for a registry-lookup miss on
    # some, zero on others) — so ignore it and enforce the actual
    # invariant instead: after the pull, every image the compose file
    # references must exist locally. Missing images abort here with their
    # names (instead of a confusing failure later); a never-pulled image
    # that exists locally proceeds with a stale-copy warning.
    logger.info("Pulling images...")
    async with timer.ameasure(PHASE_IMAGE_PULL):
        rc, _, _ = await run_cmd("docker compose pull --ignore-pull-failures", timeout=1800, log_output=True)
    # stream=False, else run_cmd passes stdout through and returns "" — which left this
    # guard iterating an empty list, so a genuinely missing image fell through to the
    # confusing later failure the check exists to replace.
    _, images_out, _ = await run_cmd("docker compose config --images", stream=False, timeout=60)
    missing = []
    for image in images_out.split():
        rc_i, _, _ = await run_cmd(f"docker image inspect {image}", timeout=60)
        if rc_i != 0:
            missing.append(image)
    if missing:
        logger.error(f"Images neither pullable nor available locally: {', '.join(missing)}")
        return False
    if rc != 0:
        logger.warning("Image pull reported failures, but every image exists locally — proceeding (local copies may be stale)")

    # Step 2: Model weights. An image that ships its own snapshot needs none of this —
    # and cannot do it, since it runs offline by construction. Defer to its cache and
    # rewrite the compose file so the service keeps the baked HF_HOME too (the first
    # write had to happen before the pull, which is what `docker compose pull` reads).
    baked = set()
    for image in dict.fromkeys(service.recipe.engine.llm.image for service in services):
        baked_hf_home = await baked_hf_cache(run_cmd, image)
        if baked_hf_home:
            if any(
                service.recipe.engine.llm.vllm and service.recipe.engine.llm.vllm.lora_adapter
                for service in services
                if service.recipe.engine.llm.image == image
            ):
                logger.error("A pinned LoRA adapter needs a regular vLLM image with a writable model cache, not %s", image)
                return False
            logger.info(f"Image {image} ships its model cache at {baked_hf_home} (offline) — skipping download")
            baked.add(image)
    if baked:
        await write_file(
            "docker-compose.yaml", generate_compose(services, model_dir, hf_token, load_balancer, baked_images=baked, proxy=proxy)
        )
    download_items = []
    for service in services:
        image = service.recipe.engine.llm.image
        if image in baked:
            continue
        download_items.append((image, service.recipe.model_name, service.recipe.model.revision))
        adapter = service.recipe.engine.llm.vllm.lora_adapter if service.recipe.engine.llm.vllm else None
        if adapter:
            download_items.append((image, adapter.huggingface, adapter.revision))
    downloads = dict.fromkeys(download_items)
    proxy_args = "".join(f" -e {name}={value}" for name, value in proxy_env(proxy).items()) if proxy else ""
    async with timer.ameasure(PHASE_MODEL_DOWNLOAD):
        for image, model_name, revision in downloads:
            logger.info(f"Downloading model {model_name}...")
            revision_arg = f" --revision {revision}" if revision else ""
            dl_cmd = (
                f"docker run --rm"
                f" -e HUGGING_FACE_HUB_TOKEN={hf_token}"
                f" -e HF_HOME={model_dir}{proxy_args}"
                f" -v {model_dir}:{model_dir}"
                f" --entrypoint bash"
                f" {image}"
                f" -c 'HF_HUB_ENABLE_HF_TRANSFER=1 hf download {model_name}{revision_arg}'"
            )
            rc, _, _ = await run_cmd(dl_cmd, timeout=7200, log_output=True)
            if rc != 0:
                logger.error(f"Failed to download model {model_name}")
                return False

    # Step 3: Clean up old containers
    logger.info("Cleaning up old containers...")
    await run_cmd("docker compose down", timeout=300, log_output=True)

    # Steps 4-5: start each service detached, then poll its /health. vLLM asserts at start-up
    # that free memory covers its whole fraction, so a service waits for every earlier service
    # it shares a GPU with; services on other GPUs start at once. nginx, then autoheal, come last.
    async with timer.ameasure(PHASE_MODEL_LOAD_AND_WARMUP):
        logger.info("Starting services...")
        healthy: set[int] = set()
        for index, service in enumerate(services):
            for earlier in range(index):
                if earlier not in healthy and _share_device(services[earlier], service):
                    if not await _wait_healthy(run_cmd, names[earlier], services[earlier].port, dry_run):
                        return False
                    healthy.add(earlier)
            rc, _, _ = await run_cmd(f"docker compose up -d {names[index]}", timeout=600, log_output=True)
            if rc != 0:
                return await _fail_with_logs(run_cmd, f"Failed to start {names[index]}")
        for index, service in enumerate(services):
            if index not in healthy and not await _wait_healthy(run_cmd, names[index], service.port, dry_run):
                return False
        if load_balancer:
            rc, _, _ = await run_cmd("docker compose up -d nginx", timeout=600, log_output=True)
            if rc != 0:
                return await _fail_with_logs(run_cmd, "Failed to start nginx")
            if not await _wait_healthy(run_cmd, "nginx", 8080, dry_run):
                return False
        # Last, so a slow first load can never be mistaken for a hung engine.
        rc, _, _ = await run_cmd("docker compose up -d autoheal", timeout=600, log_output=True)
        if rc != 0:
            return await _fail_with_logs(run_cmd, "Failed to start autoheal")

    # Best-effort: break the warmup window into startup / weights_load / torch_compile /
    # engine_warmup / cuda_graph_capture by scraping the first service's logs. The leaves sum
    # to model_load_and_warmup; absent/degraded to a single `other` when the format differs.
    if not dry_run:
        rc_logs, logs, _ = await run_cmd(f"docker compose logs --no-color {names[0]}", stream=False, timeout=120)
        if rc_logs == 0 and logs:
            raw = parse_engine_load_phases(logs, services[0].recipe.engine.llm.engine_name)
            mlw = timer.phases.get(PHASE_MODEL_LOAD_AND_WARMUP, 0.0)
            for name, seconds in decompose_model_load(raw, mlw).items():
                timer.record(name, seconds)

    # Step 6: Print endpoint info (external ports, e.g. through CloudRift NAT)
    port_map = dict(port_mappings or [])
    for service in services:
        logger.info(f"\nEndpoint: http://{host}:{port_map.get(service.port, service.port)}/v1")
        logger.info(f"Model: {service.recipe.model_name}")
    if load_balancer:
        logger.info(f"\nLoad balancer: http://{host}:{port_map.get(8080, 8080)}/v1")
    logger.info(f"Instances: {len(services)}")
    logger.info(f"Status: {'dry-run (not deployed)' if dry_run else 'deployed'}")

    # Step 7: Probe inference on every service.
    if not dry_run:
        logger.info("\nRunning smoke test...")
        async with timer.ameasure(PHASE_SMOKE_TEST):
            for index, service in enumerate(services):
                if not await _smoke_test(run_cmd, service, names[index], check_smoke_output):
                    return False

    # Print curl example
    logger.info("\nExample curl:")
    for service in services:
        path, body = _request(service.recipe, example=True)
        payload = "\n".join("    " + line for line in json.dumps(body, indent=2).splitlines()).lstrip()
        logger.info(
            f"  curl http://{host}:{port_map.get(service.port, service.port)}{path} \\\n"
            f"    -H 'Content-Type: application/json' \\\n"
            f"    -d '{payload}'"
        )

    return True


def _chat_answer(stdout: str) -> str:
    """The assistant text of a /v1/chat/completions response; empty when malformed or not ready."""
    try:
        message = json.loads(stdout)["choices"][0]["message"]
        return message.get("content") or message.get("reasoning_content") or message.get("reasoning") or ""
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return ""


def _check_chat_response(stdout: str) -> tuple[str, str]:
    """Validate a /v1/chat/completions smoke response.

    Returns ``("pass" | "fail" | "retry", detail)`` — ``retry`` means the
    response was malformed/empty (server may still be starting)."""
    answer = _chat_answer(stdout)
    if not answer:
        return "retry", ""
    if "4" in answer:
        return "pass", ""
    return "fail", f"model returned wrong answer: {answer!r}"


def _check_image_response(stdout: str) -> tuple[str, str]:
    """Validate the image smoke response: the model must name the red square's color."""
    answer = _chat_answer(stdout)
    if not answer:
        return "retry", ""
    if re.search(r"\bred\b", answer, re.IGNORECASE):
        return "pass", ""
    return "fail", f"model did not see the red image: {answer!r}"


def _check_audio_response(stdout: str) -> tuple[str, str]:
    """Validate the audio smoke response: the engine decoded the clip and the model answered. A tone has
    no words to check, so any answer passes; a missing audio stack fails the request itself."""
    return ("pass", "") if _chat_answer(stdout) else ("retry", "")


def _check_readiness_response(stdout: str) -> tuple[str, str]:
    """Accept any nonempty JSON API response without interpreting model content."""
    try:
        body = json.loads(stdout)
    except json.JSONDecodeError:
        return "retry", ""
    return ("pass", "") if body else ("retry", "")


def _smoke_response_check(recipe: Recipe, *, check_smoke_output: bool):
    """Select semantic deployment checking or protocol-only benchmark readiness."""
    if not check_smoke_output:
        return _check_readiness_response
    if recipe.is_embedding:
        return _check_embedding_response
    if recipe.model.smoke_test == "completion":
        return _check_completion_response
    return _check_chat_response


def _check_completion_response(stdout: str) -> tuple[str, str]:
    """Validate the base-model completion smoke response."""
    try:
        answer = json.loads(stdout)["choices"][0]["text"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "retry", ""
    if not answer:
        return "retry", ""
    if "4" in answer:
        return "pass", ""
    return "fail", f"model returned wrong answer: {answer!r}"


def _check_embedding_response(stdout: str) -> tuple[str, str]:
    """Validate a /v1/embeddings smoke response: a non-empty vector of finite
    floats with L2 norm ≈ 1 (the pooler normalizes; garbage/NaN models fail)."""
    try:
        body = json.loads(stdout)
        vec = body["data"][0]["embedding"]
    except (json.JSONDecodeError, KeyError, IndexError, TypeError):
        return "retry", ""
    if not isinstance(vec, list) or not vec:
        return "retry", ""
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in vec):
        return "fail", "embedding contains non-finite values"
    norm = math.sqrt(sum(v * v for v in vec))
    if not 0.9 <= norm <= 1.1:
        return "fail", f"embedding L2 norm {norm:.4f} outside [0.9, 1.1] (dim={len(vec)})"
    return "pass", ""


async def run_teardown(run_cmd):
    """Tear down: docker compose down."""
    logger.info("Tearing down...")
    rc, _, _ = await run_cmd("docker compose down", timeout=300, log_output=True)
    if rc == 0:
        logger.info("Teardown complete.")
    else:
        logger.error("Teardown failed.")
    return rc == 0


async def deploy(params: DeployParams, timer: PhaseTimer | None = None, *, check_smoke_output: bool = True) -> bool:
    """Deploy a recipe to a server via SSH. Single entry point."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    write_file = make_write_file(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    host = params.server.split("@")[-1] if "@" in params.server else params.server
    return await run_deploy(
        run_cmd,
        write_file,
        params.services,
        params.model_dir,
        params.hf_token,
        host,
        params.dry_run,
        load_balancer=params.load_balancer,
        port_mappings=params.port_mappings,
        timer=timer,
        check_smoke_output=check_smoke_output,
        proxy=params.proxy,
    )


async def teardown(params: DeployParams) -> bool:
    """Teardown containers on a server."""
    run_cmd = make_run_cmd(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    return await run_teardown(run_cmd)
