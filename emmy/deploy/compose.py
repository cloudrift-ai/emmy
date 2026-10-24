"""Docker Compose and nginx config generation."""

from typing import Any

import yaml

from emmy.deploy.params import Service
from emmy.provisioning.proxy import proxy_env
from emmy.recipe.engines import build_engine_args
from emmy.recipe.types import Recipe

AUTOHEAL_IMAGE = "willfarrell/autoheal:1.2.0"
LOAD_BALANCER_PORT = 8080


def _env_items(extra_env) -> list[tuple[str, str]]:
    """``extra_env`` as (key, value) pairs, whether it's a dict or a string.

    The schema declares a dict, but matrix recipes (and their dryrun tests) write
    space-separated ``K=V`` strings — e.g. ``"EMMY_FAST_MATH=1 EMMY_GEN_DECODE_BUCKET=32"``
    — and the variant expansion passes them through verbatim. Accept both."""
    if isinstance(extra_env, str):
        return [tuple(kv.split("=", 1)) for kv in extra_env.split()]
    return list(extra_env.items())


def _render_docker_options(docker_options: dict[str, Any]) -> str:
    """Render docker_options dict as indented YAML lines for a compose service."""
    if not docker_options:
        return ""
    lines = []
    for key, value in docker_options.items():
        fragment = yaml.dump({key: value}, default_flow_style=False).rstrip("\n")
        indented = "\n".join("    " + line for line in fragment.splitlines())
        lines.append(indented)
    return "\n" + "\n".join(lines)


def service_name(index: int, service: Service) -> str:
    """Compose service (and container) name of the service at ``index``."""
    return f"{service.recipe.engine.llm.engine_name}_{index}"


def replica_services(recipe: Recipe, gpu_device_ids: list[int] | None = None) -> tuple[list[Service], bool]:
    """Fan a resolved recipe out into its containers.

    When ``deploy.gpu_count`` holds several instances of the recipe's parallelism, each replica
    gets its own device slice and host port behind a load balancer; otherwise one service sees
    the given devices (every GPU when None). Returns the services and whether they need nginx.
    """
    per_instance = recipe.engine.llm.gpus_per_instance
    count = max(1, recipe.deploy.gpu_count // per_instance) if recipe.deploy.gpu is not None else 1
    if count == 1:
        return [Service(recipe, gpu_device_ids)], False
    return [Service(recipe, list(range(i * per_instance, (i + 1) * per_instance)), 8000 + i) for i in range(count)], True


def serving_port(load_balancer: bool, services: list[Service]) -> int:
    """Host port the model answers on: the load balancer's when there are replicas."""
    return LOAD_BALANCER_PORT if load_balancer else services[0].port


def generate_compose(services: list[Service], model_dir, hf_token, load_balancer=False, baked_images=frozenset(), proxy=None, api_key=None):
    """Build docker-compose.yaml string from the services of one deployment.

    One engine service per entry, named ``{engine}_{i}``, pinned to its device ids (``count: all``
    when it has none) and published on its host port; with ``load_balancer``, an nginx service on
    8080 in front of all of them; and an autoheal service that restarts an unhealthy engine service.
    Nothing declares ``depends_on``: the orchestrator starts the services in the order their GPUs
    allow and polls each one itself.

    ``baked_images``: the images that ship their own HF cache (see ``baked_hf_cache``).
    Setting HF_HOME on such an image would hide the snapshot it baked in, so the override is
    dropped and the image's own value stands — UNLESS the engine args name a model beyond the
    baked one (a ``--speculative-config`` drafter): the baked cache holds only the one snapshot
    and the image pins ``HF_HUB_OFFLINE=1``, so the extra model can resolve only from the host
    cache, and the override returns.

    ``proxy``: an HTTP proxy URL; every engine service gets it in its environment, in both spellings, so
    the weight download inside the container goes through it. nginx only talks to the services.

    ``api_key``: the deployment serves the public behind this key, unattended (``deploy boot-script``). vLLM's own
    ``VLLM_API_KEY`` check covers only ``/v1/*``, so it is set on every engine service but no engine port is
    published: nginx, on ``serving_port``, is the only way in, and ``generate_nginx_conf(api_key=...)`` checks the
    key. autoheal gets no restart policy, so after a reboot only the boot script starts it, once the engines are
    healthy again.
    """
    compose = "services:\n"
    proxy_lines = "".join(f"\n      - {name}={value}" for name, value in proxy_env(proxy).items()) if proxy else ""
    api_key_line = f"\n      - VLLM_API_KEY={api_key}" if api_key else ""

    for index, service in enumerate(services):
        recipe = service.recipe
        llm = recipe.engine.llm
        model_name = recipe.model_name
        engine_args = build_engine_args(llm, model_name, recipe.model.revision, model_dir=model_dir)
        command_str = "\n      ".join(engine_args)
        extra_env_lines = "".join(f"\n      - {k}={v}" for k, v in _env_items(llm.extra_env))
        needs_extra_model = any("--speculative-config" in a for a in engine_args)
        hf_home_line = "" if llm.image in baked_images and not needs_extra_model else f"\n      - HF_HOME={model_dir}"
        docker_options_lines = _render_docker_options(llm.docker_options)
        entrypoint_line = f"\n    entrypoint: {llm.entrypoint}" if llm.entrypoint else ""

        if recipe.deploy.gpu is not None and recipe.deploy.gpu.startswith("AMD"):
            gpu_section = (
                "    devices:\n      - /dev/kfd:/dev/kfd\n      - /dev/dri:/dev/dri\n    group_add:\n      - video\n      - render"
            )
        else:
            if service.gpu_device_ids is None:
                gpu_config = "count: all"
            else:
                gpu_config = "device_ids: [" + ", ".join(f"'{g}'" for g in service.gpu_device_ids) + "]"
            gpu_section = (
                "    deploy:\n"
                "      resources:\n"
                "        reservations:\n"
                "          devices:\n"
                f"            - driver: nvidia\n"
                f"              {gpu_config}\n"
                "              capabilities: [gpu]"
            )

        name = service_name(index, service)
        ports_section = "" if api_key else f'    ports:\n      - "{service.port}:8000"\n'
        compose += f"""
  {name}:
    image: {llm.image}
    container_name: {name}{entrypoint_line}
{gpu_section}
    volumes:
      - {model_dir}:{model_dir}
    environment:
      - HUGGING_FACE_HUB_TOKEN={hf_token}{hf_home_line}{extra_env_lines}{proxy_lines}{api_key_line}
{ports_section}    shm_size: '16gb'
    ipc: host
    restart: unless-stopped
    labels:
      - autoheal=true{docker_options_lines}
    command: >
      {command_str}
    healthcheck:
      test: ["CMD", "bash", "-c", "curl -f http://localhost:8000/health"]
      interval: 10s
      timeout: 10s
      retries: 180
      start_period: 1200s
"""

    if load_balancer or api_key:
        compose += f"""
  nginx:
    image: nginx:alpine
    container_name: nginx_lb
    ports:
      - "{serving_port(load_balancer, services)}:{LOAD_BALANCER_PORT}"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
    restart: unless-stopped
"""

    # `restart: unless-stopped` acts only when the process exits. An engine whose core has died can
    # leave its API server waiting forever for open streams to close: the container stays up, the
    # health check fails, and nothing restarts it. autoheal restarts an engine service once Docker
    # marks it unhealthy; it touches only containers labelled autoheal=true.
    compose += f"""
  autoheal:
    image: {AUTOHEAL_IMAGE}
    container_name: autoheal
    environment:
      - AUTOHEAL_CONTAINER_LABEL=autoheal
    volumes:
      - /var/run/docker.sock:/var/run/docker.sock
    restart: {'"no"' if api_key else "unless-stopped"}
"""

    return compose


def generate_nginx_conf(names: list[str], api_key: str | None = None):
    """Generate nginx config with a least_conn upstream over the named services.

    With ``api_key`` it is the public gateway of an unattended deployment (see ``generate_compose``): ``GET /health``
    passes without the key, ``/v1/`` needs ``Authorization: Bearer <api_key>``, and every other path is a 404, since
    vLLM leaves its routes outside ``/v1/`` open. The map lookup ignores case; the engine's own exact ``VLLM_API_KEY``
    check behind it does not. The engines are resolved again while nginx runs, since a restarted container can come
    back on another address.
    """
    if api_key:
        return _gateway_nginx_conf(names, api_key)
    upstream_servers = "\n".join(f"        server {name}:8000;" for name in names)

    return f"""worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    upstream llm_backend {{
        least_conn;
{upstream_servers}
    }}

    server {{
        listen 8080;

        location / {{
            proxy_pass http://llm_backend;
            proxy_http_version 1.1;
            proxy_set_header Connection "";
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;

            proxy_connect_timeout 600s;
            proxy_send_timeout 600s;
            proxy_read_timeout 600s;

            proxy_buffering off;
        }}
    }}
}}
"""


def _gateway_nginx_conf(names: list[str], api_key: str) -> str:
    upstream_servers = "\n".join(f"        server {name}:8000 resolve;" for name in names)

    return f"""worker_processes auto;

events {{
    worker_connections 4096;
}}

http {{
    resolver 127.0.0.11 valid=10s ipv6=off;
    map_hash_bucket_size 256;

    map $http_authorization $authorized {{
        default 0;
        "Bearer {api_key}" 1;
    }}

    upstream llm_backend {{
        zone llm_backend 64k;
        least_conn;
{upstream_servers}
    }}

    server {{
        listen {LOAD_BALANCER_PORT};
        client_max_body_size 100m;

        proxy_http_version 1.1;
        proxy_set_header Connection "";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_connect_timeout 600s;
        proxy_send_timeout 3600s;
        proxy_read_timeout 3600s;

        proxy_buffering off;

        location = /health {{
            limit_except GET {{
                deny all;
            }}
            proxy_pass http://llm_backend;
        }}

        location /v1/ {{
            if ($authorized = 0) {{
                return 401;
            }}
            proxy_pass http://llm_backend;
        }}

        location / {{
            return 404;
        }}
    }}
}}
"""
