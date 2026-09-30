"""Deploy parameters dataclasses."""

from dataclasses import dataclass, field

from emmy.recipe.types import Recipe


@dataclass
class Service:
    """One engine container: its resolved recipe, the GPU devices it may see, its host port.

    ``gpu_device_ids`` is None when the container may see every GPU (``count: all``). The
    container port is always 8000; ``port`` is the host port published for it.
    """

    recipe: Recipe
    gpu_device_ids: list[int] | None = None
    port: int = 8000


@dataclass
class DeployParams:
    """All parameters needed for a single deployment. Serializable for future API use."""

    server: str  # user@host or IP
    ssh_key: str  # path to SSH private key
    ssh_port: int = 22
    services: list[Service] = field(default_factory=list)
    load_balancer: bool = False  # nginx on 8080 in front of every service (replica fan-out)
    model_dir: str = "/hf_models"
    hf_token: str = ""
    dry_run: bool = False
    port_mappings: list[tuple[int, int]] = field(default_factory=list)
