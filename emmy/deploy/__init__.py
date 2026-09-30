"""Deploy library: compose generation, deploy orchestration, scale-out strategies."""

from emmy.deploy.compose import (
    generate_compose,
    generate_nginx_conf,
    replica_services,
    service_name,
)
from emmy.deploy.orchestrate import (
    deploy,
    run_deploy,
    run_teardown,
    teardown,
)
from emmy.deploy.params import DeployParams, Service
from emmy.deploy.scale_out import (
    DEFAULT_STRATEGY,
    STRATEGIES,
    DataParallelismScaleOutStrategy,
    ReplicaParallelismScaleOutStrategy,
    ScaleOutStrategy,
)

__all__ = [
    "DEFAULT_STRATEGY",
    "DataParallelismScaleOutStrategy",
    "DeployParams",
    "ReplicaParallelismScaleOutStrategy",
    "STRATEGIES",
    "ScaleOutStrategy",
    "Service",
    "deploy",
    "generate_compose",
    "generate_nginx_conf",
    "replica_services",
    "run_deploy",
    "run_teardown",
    "service_name",
    "teardown",
]
