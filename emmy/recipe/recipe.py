"""Recipe loading and deep merge."""

import os
import re

import yaml

from emmy.recipe.engines import banned_extra_arg_flags
from emmy.recipe.lifecycle import recipe_is_runnable, recipe_lifecycle, validate_recipe_tags
from emmy.recipe.types import LLMConfig, Recipe


def deep_merge(base, override):
    """Recursive dict merge. Override wins for scalars."""
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def validate_extra_args(extra_args, engine="vllm"):
    """Raise ValueError if extra_args contains flags managed by named recipe fields."""
    banned = banned_extra_arg_flags(engine)
    tokens = extra_args.split()
    found = []
    for token in tokens:
        flag = token.split("=")[0]
        if flag in banned:
            found.append(flag)
    if found:
        raise ValueError(
            f"extra_args contains flags managed by named fields: {', '.join(sorted(found))}. "
            f"Use the corresponding recipe YAML keys instead."
        )


INPUT_MODALITIES = ("text", "image")


def validate_input_modalities(value) -> tuple[str, ...]:
    """Return a recipe's ``model.input_modalities``: ``("text",)`` when unset, else the declared list."""
    if value is None:
        return ("text",)
    if not isinstance(value, list) or not value or not all(isinstance(m, str) for m in value):
        raise ValueError(f"model.input_modalities must be a non-empty list of strings, got {value!r}")
    unknown = sorted(set(value) - set(INPUT_MODALITIES))
    if unknown:
        raise ValueError(f"model.input_modalities has unknown entries {unknown}; allowed: {', '.join(INPUT_MODALITIES)}")
    if len(set(value)) != len(value) or "text" not in value:
        raise ValueError(f"model.input_modalities must list text and each modality once, got {value!r}")
    return tuple(value)


def _disables_image_input(extra_args: str) -> bool:
    """Whether the engine flags drop the vision tower or zero the per-prompt image limit."""
    if re.search(r"--language-model-only\b", extra_args):
        return True
    _, flag, rest = extra_args.partition("--limit-mm-per-prompt")
    if not flag:
        return False
    value = rest.split(" --", 1)[0]  # this flag's value, up to the next flag
    return re.search(r'"image"\s*:\s*0(?![.\d])|\bimage=0(?![.\d])', value) is not None


def validate_image_input(config: dict) -> tuple[str, ...]:
    """Validate ``model.input_modalities`` against one resolved recipe config and return it.

    ``image`` needs a generative task and engine flags that keep image input on: a recipe
    served with ``--language-model-only`` or a zero image limit must not declare it.
    """
    model = config.get("model") or {}
    modalities = validate_input_modalities(model.get("input_modalities"))
    if "image" not in modalities:
        return modalities
    if model.get("task", "generate") != "generate":
        raise ValueError("model.input_modalities: image requires model.task: generate")
    llm = (config.get("engine") or {}).get("llm") or {}
    engine = llm.get("sglang") or llm.get("vllm") or {}
    if _disables_image_input(engine.get("extra_args", "")):
        raise ValueError(
            "model.input_modalities declares image, but extra_args disable image input "
            "(--language-model-only or --limit-mm-per-prompt with image 0)"
        )
    return modalities


def _load_raw_config(recipe_dir) -> dict:
    """Load recipe.yaml and return raw dict (with matrices still present)."""
    recipe_path = os.path.join(recipe_dir, "recipe.yaml")
    if not os.path.isfile(recipe_path):
        raise FileNotFoundError(f"Recipe file not found: {recipe_path}")

    with open(recipe_path) as f:
        return yaml.safe_load(f)


def validate_docker_options(docker_options: dict) -> None:
    """Raise ValueError if docker_options contains keys managed by the compose template."""
    conflicts = set(docker_options) & _MANAGED_COMPOSE_KEYS
    if conflicts:
        raise ValueError(
            f"docker_options contains keys managed by the compose template: "
            f"{', '.join(sorted(conflicts))}. Remove them from docker_options."
        )


# Keys already rendered by generate_compose() — must not appear in docker_options.
_MANAGED_COMPOSE_KEYS = frozenset(
    {
        "image",
        "container_name",
        "entrypoint",
        "deploy",
        "devices",
        "group_add",
        "volumes",
        "environment",
        "ports",
        "shm_size",
        "ipc",
        "command",
        "healthcheck",
        "restart",
    }
)


def _validate_and_build(config: dict) -> Recipe:
    """Validate extra_args and docker_options, then build Recipe from config dict."""
    validate_recipe_tags(config.get("tags"))
    if not recipe_is_runnable(config):
        raise ValueError(f"Recipe is disabled by its '{recipe_lifecycle(config)}' lifecycle tag")

    has_command = "command" in config and config["command"] is not None
    has_engine_llm = bool(config.get("engine", {}).get("llm"))

    if has_command and has_engine_llm:
        raise ValueError("Recipe must specify exactly one of 'engine.llm' or 'command', not both.")

    task = config.get("model", {}).get("task", "generate")
    if task not in ("generate", "embed"):
        raise ValueError(f"model.task must be 'generate' or 'embed', got {task!r}")
    smoke_test = config.get("model", {}).get("smoke_test", "chat")
    if smoke_test not in ("chat", "completion"):
        raise ValueError(f"model.smoke_test must be 'chat' or 'completion', got {smoke_test!r}")
    if task == "embed" and smoke_test != "chat":
        raise ValueError("model.smoke_test is only configurable for model.task: generate")
    validate_image_input(config)
    revision = config.get("model", {}).get("revision")
    if revision is not None and (not isinstance(revision, str) or not re.fullmatch(r"[A-Za-z0-9._/-]+", revision)):
        raise ValueError(f"model.revision must be a non-empty Hugging Face revision, got {revision!r}")

    if has_command:
        # Command recipes don't go through engine extra_args validation.
        return Recipe.from_dict(config)
    llm_dict = config.get("engine", {}).get("llm", {})
    if "sglang" in llm_dict:
        engine = "sglang"
        extra_args = llm_dict.get("sglang", {}).get("extra_args", "")
    else:
        engine = "vllm"
        extra_args = llm_dict.get("vllm", {}).get("extra_args", "")
    validate_extra_args(extra_args, engine=engine)
    if engine == "vllm" and llm_dict.get("vllm", {}).get("lora_adapter"):
        conflicts = {token.split("=")[0] for token in extra_args.split()} & {"--enable-lora", "--max-lora-rank", "--lora-modules"}
        if conflicts:
            raise ValueError(f"extra_args conflicts with vllm.lora_adapter: {', '.join(sorted(conflicts))}")
    validate_docker_options(llm_dict.get("docker_options", {}))

    return Recipe.from_dict(config)


def load_recipe(recipe_dir):
    """Load recipe.yaml and return base Recipe (no matrix expansion).

    Strips the 'matrices' section before building the Recipe.
    """
    config = _load_raw_config(recipe_dir)
    config.pop("matrices", None)
    return _validate_and_build(config)


def resolve_for_hardware(
    recipe_dir: str, gpu_name: str, gpu_count: int | None = None, gpu_memory_utilization: float | None = None
) -> "Recipe":
    """Load recipe and resolve the matrix entry for the given hardware.

    Expands the full matrix (cross/zip), keeps the entries naming ``gpu_name``, then picks:
    1. With ``gpu_memory_utilization``: the entry whose deploy.gpu_count and fraction both match
       exactly, and nothing else — a plan names the qualified entry it wants.
    2. Otherwise, among the entries whose deploy.gpu_count matches exactly, the highest fraction
       (the whole-GPU qualification); ties keep the first declared.
    3. Divisible match: gpu_count is a multiple of the entry's deploy.gpu_count (for scale-out).
       The largest entry count that divides evenly wins.
    4. Name-only match: the first entry for the GPU (when gpu_count is None).

    If no matrices section exists, returns the base recipe.
    Raises ValueError if no match is found.
    """
    from emmy.recipe.matrix import build_override, expand_matrix

    config = _load_raw_config(recipe_dir)
    matrices = config.pop("matrices", None)

    if not matrices:
        return _validate_and_build(config)

    base_fraction = (config.get("engine") or {}).get("llm", {}).get("gpu_memory_utilization", LLMConfig.gpu_memory_utilization)

    def build(combo):
        return _validate_and_build(deep_merge(config, build_override(combo)))

    def fraction(combo):
        return combo.get("engine.llm.gpu_memory_utilization", base_fraction)

    combinations = expand_matrix(matrices)
    candidates = [combo for combo in combinations if combo.get("deploy.gpu") == gpu_name]
    if not candidates:
        available_gpus = sorted({combo["deploy.gpu"] for combo in combinations if combo.get("deploy.gpu") is not None})
        raise ValueError(f"No matrix entry matches GPU '{gpu_name}'. Available GPUs: {', '.join(available_gpus)}")

    if gpu_count is None:
        return build(candidates[0])

    exact = [combo for combo in candidates if combo.get("deploy.gpu_count", 1) == gpu_count]
    if gpu_memory_utilization is not None:
        for combo in exact:
            if fraction(combo) == gpu_memory_utilization:
                return build(combo)
        raise ValueError(
            f"No matrix entry for GPU '{gpu_name}' x{gpu_count} at gpu_memory_utilization={gpu_memory_utilization}. "
            f"Available fractions: {sorted({fraction(combo) for combo in exact})}"
        )
    if exact:
        return build(max(exact, key=fraction))

    divisible = [combo for combo in candidates if gpu_count % combo.get("deploy.gpu_count", 1) == 0]
    if divisible:
        return build(max(divisible, key=lambda combo: combo.get("deploy.gpu_count", 1)))

    entry_counts = sorted({combo.get("deploy.gpu_count", 1) for combo in candidates})
    raise ValueError(f"No matrix entry for GPU '{gpu_name}' matches gpu_count={gpu_count}. Available counts: {entry_counts}")
