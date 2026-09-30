"""A deployment plan: several models on one VM, each pinned to its GPU devices."""

import json
from dataclasses import dataclass
from pathlib import Path

from emmy.deploy.params import Service
from emmy.recipe import resolve_for_hardware, resolve_recipe_dir

PLAN_SCHEMA_VERSION = 1
# What the fractions of the models sharing one GPU device may add up to; the rest is the
# engines' own memory outside their budgets (CUDA contexts, allocator slack).
SHARED_DEVICE_BUDGET = 0.95


@dataclass
class Plan:
    """The VM to rent and the services to run on it, in start order."""

    gpu: str
    gpu_count: int
    recipes: list[str]  # the recipe each model named, for the result file
    services: list[Service]


def load_plan(path: str | Path) -> Plan:
    """Read and validate a plan file; every error names the model it concerns.

    Each model resolves its recipe's matrix entry by exact match on the plan's GPU, its device
    count and its memory fraction, and must run as one container (the entry's GPU count equals
    its parallelism). The models sharing a device must fit the shared budget together. Model i
    becomes the service on host port 8000 + i.
    """
    plan = json.loads(Path(path).read_text())
    if plan.get("schema_version") != PLAN_SCHEMA_VERSION:
        raise ValueError(f"plan schema_version must be {PLAN_SCHEMA_VERSION}, got {plan.get('schema_version')!r}")
    gpu, gpu_count, models = plan["gpu"], plan["gpu_count"], plan["models"]
    if not isinstance(gpu, str) or not gpu:
        raise ValueError("plan gpu must be a GPU name")
    if not isinstance(gpu_count, int) or isinstance(gpu_count, bool) or gpu_count < 1:
        raise ValueError("plan gpu_count must be a positive integer")
    if not isinstance(models, list) or not models:
        raise ValueError("plan models must be a non-empty list")

    services = []
    tenants: dict[int, list[float]] = {}
    for index, model in enumerate(models):
        name, fraction, devices = model["recipe"], model["gpu_memory_utilization"], model["gpu_device_ids"]
        where = f"model {index} ({name})"
        valid_ids = isinstance(devices, list) and devices and all(isinstance(d, int) and not isinstance(d, bool) for d in devices)
        if not valid_ids or len(set(devices)) != len(devices) or not all(0 <= d < gpu_count for d in devices):
            raise ValueError(f"{where}: gpu_device_ids must be distinct integers below {gpu_count}")
        try:
            recipe = resolve_for_hardware(resolve_recipe_dir(name), gpu, len(devices), fraction)
        except (FileNotFoundError, ValueError) as exc:
            raise ValueError(f"{where}: {exc}") from exc
        if recipe.deploy.gpu_count != recipe.engine.llm.gpus_per_instance:
            raise ValueError(
                f"{where}: its matrix entry spreads {recipe.deploy.gpu_count} GPUs over replicas, but a plan runs one "
                f"container per model; the entry must declare data parallelism instead"
            )
        for device in devices:
            tenants.setdefault(device, []).append(fraction)
        services.append(Service(recipe, list(devices), 8000 + index))

    for device, fractions in sorted(tenants.items()):
        if len(fractions) > 1 and round(sum(fractions), 6) > SHARED_DEVICE_BUDGET:
            raise ValueError(
                f"GPU device {device} is shared by models whose fractions add up to {sum(fractions):g}, above {SHARED_DEVICE_BUDGET}"
            )
    pins = {(service.recipe.deploy.driver_version, service.recipe.deploy.cuda_version) for service in services}
    if len(pins) > 1:
        raise ValueError(f"the recipes disagree on their driver/CUDA version pins: {sorted(pins, key=str)}")
    return Plan(gpu, gpu_count, [model["recipe"] for model in models], services)
