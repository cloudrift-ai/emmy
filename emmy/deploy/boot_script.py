"""Boot script for a VM that serves a model by itself.

`deploy cloud` provisions a VM over SSH. A VM rented by a customer has nobody connected to it,
so this renders the same deployment as a shell script for the VM's first boot (CloudRift runs it
as root through cloud-init). The script only writes files and installs a systemd unit; the unit
does the slow part (Docker, weights, engine start) in the background so the VM reports ready
promptly. After a reboot the unit only waits for the engines Docker restarted, since everything
is cached by then (``INSTALLED_FILE``). Progress is the one-line file ``STATUS_FILE``.
"""

import re
import shlex

import httpx

from emmy.deploy.compose import generate_compose, generate_nginx_conf, service_name, serving_port
from emmy.deploy.orchestrate import HEALTH_INTERVAL, HEALTH_TIMEOUT, IMAGE_ENV_FORMAT, model_download_command, model_downloads
from emmy.deploy.params import Service
from emmy.provisioning.remote import (
    DOCKER_INSTALL,
    FABRIC_MANAGER_INSTALL,
    NVIDIA_CONTAINER_TOOLKIT_INSTALL,
    NVIDIA_CUDA_REPO_SETUP,
)

SERVE_DIR = "/opt/emmy-model"
STATUS_FILE = f"{SERVE_DIR}/status"
INSTALLED_FILE = f"{SERVE_DIR}/installed"
UNIT_NAME = "emmy-model.service"
MODEL_DIR = "/hf_models"
HF_MODEL_API = "https://huggingface.co/api/models/{}"
# Restarts a crash-looping engine reaches before the script stops waiting for it.
CRASH_RESTARTS = 3
# Lands in a YAML file, an nginx string and a shell heredoc, so only characters that are inert in all three.
API_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,128}$")
_HEREDOC = "EMMY_BOOT_EOF"
_FABRIC_MANAGER = shlex.quote(
    f'{NVIDIA_CUDA_REPO_SETUP}; driver="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n1)"; '
    f'[ -n "$driver" ]; {FABRIC_MANAGER_INSTALL}'
)


async def gated_models(model_names: list[str]) -> list[str]:
    """The models among ``model_names`` that Hugging Face gates behind an access agreement."""
    gated = []
    async with httpx.AsyncClient(timeout=30) as client:
        for name in model_names:
            response = await client.get(HF_MODEL_API.format(name))
            response.raise_for_status()
            if response.json().get("gated"):
                gated.append(name)
    return gated


def check_api_key(api_key: str) -> None:
    """Raise ValueError unless ``api_key`` can be written into the compose file and the script as is."""
    if not API_KEY_PATTERN.match(api_key):
        raise ValueError("The API key must hold 16-128 characters from A-Z, a-z, 0-9, '_' and '-'")


def render_boot_script(services: list[Service], load_balancer: bool, api_key: str) -> str:
    """The first-boot script for ``services``, serving behind ``api_key``.

    Raises ValueError for what the VM cannot serve unattended: an unsafe ``api_key``, a pinned driver
    or CUDA version (it needs a reboot the script cannot drive) and an engine other than vLLM (only
    vLLM reads ``VLLM_API_KEY``). Weights are downloaded without a Hugging Face token: the VM belongs
    to whoever rents it, so no credential of ours may land on it, and the caller refuses gated models
    (``gated_models``) before rendering. An image that ships its own model cache is refused on the VM,
    since only inspecting the pulled image tells.
    """
    check_api_key(api_key)
    for service in services:
        deploy, llm = service.recipe.deploy, service.recipe.engine.llm
        if deploy.driver_version or deploy.cuda_version:
            raise ValueError("The recipe pins an NVIDIA driver or CUDA version, which needs a reboot a boot script cannot drive")
        if llm.engine_name != "vllm":
            raise ValueError(f"The endpoint is protected through VLLM_API_KEY, which {llm.engine_name} does not read")
    names = [service_name(index, service) for index, service in enumerate(services)]
    engines = " ".join(names)
    compose = generate_compose(services, MODEL_DIR, "", load_balancer, api_key=api_key)
    port = serving_port(load_balancer, services)
    nvidia = not any((service.recipe.deploy.gpu or "").startswith("AMD") for service in services)

    install = [
        "status installing",
        f"command -v docker >/dev/null || {{ {DOCKER_INSTALL}; }} || fail 'Docker install'",
    ]
    if nvidia:
        install += [
            f"command -v nvidia-ctk >/dev/null || {{ {NVIDIA_CONTAINER_TOOLKIT_INSTALL}; }} || fail 'NVIDIA Container Toolkit install'",
            "if ls /proc/driver/nvidia-nvswitch/devices/ 2>/dev/null | grep -q . \\",
            "    && ! systemctl is-active --quiet nvidia-fabricmanager; then",
            f"  bash -c {_FABRIC_MANAGER} || fail 'Fabric Manager install'",
            "fi",
        ]
    install += ["status downloading", "docker compose pull --quiet || fail 'image pull'"]
    for image in dict.fromkeys(service.recipe.engine.llm.image for service in services):
        install.append(
            f"docker image inspect {image} --format {shlex.quote(IMAGE_ENV_FORMAT)} | grep -qx HF_HUB_OFFLINE=1"
            " && fail 'image ships its own model cache; not supported yet'"
        )
    for image, model_name, revision in model_downloads(services):
        install.append(f"{model_download_command(image, model_name, revision, MODEL_DIR, '')} || fail 'download of {model_name}'")
    install.append(f"touch {INSTALLED_FILE}")

    # `restart: unless-stopped` keeps a crashing engine `restarting`/`running`, never `exited`, so the wait also counts
    # its restarts. autoheal comes last, as in `run_deploy`, so a slow first load is never taken for a hung engine.
    serve = [
        "#!/bin/bash",
        "set -uo pipefail",
        f"cd {SERVE_DIR}",
        f'status() {{ echo "$1" > {STATUS_FILE}; }}',
        'fail() { status "failed: $1"; exit 1; }',
        f"if [ ! -e {INSTALLED_FILE} ]; then",
        *(f"  {line}" for line in install),
        "fi",
        "status starting",
        f"docker compose up -d {engines} nginx || fail 'engine start'",
        f"for _ in $(seq 1 {HEALTH_TIMEOUT // HEALTH_INTERVAL}); do",
        f"  if curl -sf -m 5 http://localhost:{port}/health >/dev/null; then",
        "    docker compose up -d autoheal || fail 'autoheal start'",
        "    status ready",
        "    exit 0",
        "  fi",
        f'  [ -n "$(docker compose ps -q --status exited --status dead {engines})" ] && fail "engine exited"',
        f"  docker inspect --format '{{{{.RestartCount}}}}' {engines} | awk '$1 >= {CRASH_RESTARTS} {{ n++ }} END {{ exit !n }}' \\",
        "    && fail 'engine keeps crashing'",
        f"  sleep {HEALTH_INTERVAL}",
        "done",
        "fail 'engine did not become healthy'",
    ]

    unit = "\n".join(
        [
            "[Unit]",
            "Description=Serve the model this VM was rented for",
            "Wants=network-online.target docker.service",
            "After=network-online.target docker.service",
            "",
            "[Service]",
            "Type=oneshot",
            "RemainAfterExit=yes",
            "TimeoutStartSec=0",
            f"ExecStart={SERVE_DIR}/serve.sh",
            "",
            "[Install]",
            "WantedBy=multi-user.target",
        ]
    )

    # CloudRift joins this script, its own commands and the VM-ready webhook into one runcmd shell script, so
    # `set -e` and any failure stay inside the subshell and the webhook always runs. The files holding the key are
    # written from a nested subshell under `umask 077`, so only root can read them.
    lines = ["(", "set -e", f"mkdir -p {SERVE_DIR} {MODEL_DIR}", f"echo pending > {STATUS_FILE}"]
    lines += _heredoc(f"{SERVE_DIR}/serve.sh", "\n".join(serve))
    lines += _heredoc(f"/etc/systemd/system/{UNIT_NAME}", unit)
    lines += ["(", "umask 077"]
    lines += _heredoc(f"{SERVE_DIR}/docker-compose.yaml", compose)
    lines += _heredoc(f"{SERVE_DIR}/nginx.conf", generate_nginx_conf(names, api_key=api_key))
    lines += [
        ")",
        f"chmod +x {SERVE_DIR}/serve.sh",
        "systemctl daemon-reload",
        f"systemctl enable {UNIT_NAME}",
        f"systemctl start --no-block {UNIT_NAME}",
        ")",
    ]
    return "\n".join(lines) + "\n"


def _heredoc(path: str, content: str) -> list[str]:
    return [f"cat > {path} <<'{_HEREDOC}'", content.rstrip("\n"), _HEREDOC]
