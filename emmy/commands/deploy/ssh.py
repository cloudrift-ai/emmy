"""SSH deploy target CLI handler."""

import asyncio
import logging
import os
import sys

from emmy.deploy import DEFAULT_STRATEGY, STRATEGIES, DeployParams, replica_services
from emmy.deploy import (
    deploy as deploy_entry,
)
from emmy.deploy import (
    teardown as teardown_entry,
)
from emmy.deploy.plan import load_plan
from emmy.detect import detect_remote_gpus
from emmy.provisioning.host import RemoteHost
from emmy.provisioning.remote import provision_remote
from emmy.provisioning.ssh_target import parse_ssh_target
from emmy.recipe import resolve_for_hardware, resolve_recipe_dir
from emmy.redact import register_secret
from emmy.timing import PHASE_REMOTE_PROVISION, PhaseTimer

logger = logging.getLogger(__name__)


def handle_ssh(args):
    """Handle the SSH deploy target."""
    asyncio.run(_handle_ssh(args))


async def _handle_ssh(args):
    if args.ssh:
        if args.server is not None or args.ssh_port is not None:
            logger.error("--ssh cannot be combined with --server / --ssh-port")
            sys.exit(2)
        user, host, port = parse_ssh_target(args.ssh)
        server = f"{user}@{host}"
    elif args.server:
        logger.warning("--server is deprecated; use --ssh USER@HOST[:PORT] instead. This flag will be removed in a future release.")
        if args.ssh_port is not None:
            logger.warning("--ssh-port is deprecated; encode the port in --ssh USER@HOST:PORT.")
        server = args.server
        port = args.ssh_port if args.ssh_port is not None else 22
    else:
        logger.error("--ssh USER@HOST[:PORT] is required")
        sys.exit(2)

    # GPU detection (overridable via CLI flags)
    if not args.gpu or not args.gpu_count:
        detected_name, detected_count = await detect_remote_gpus(server, args.ssh_key, port)
    gpu_name = args.gpu or detected_name
    gpu_count = args.gpu_count or detected_count
    logger.info(f"GPU: {gpu_count}x {gpu_name}")

    if args.plan:
        # A plan names the host it was written for; on an existing host that shape must be the one found.
        try:
            plan = load_plan(args.plan)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.error(f"Invalid plan {args.plan}: {exc}")
            sys.exit(1)
        if (gpu_name, gpu_count) != (plan.gpu, plan.gpu_count):
            logger.error(f"Plan {args.plan} is for {plan.gpu_count}x {plan.gpu}, but the host has {gpu_count}x {gpu_name}")
            sys.exit(1)
        services, load_balancer = plan.services, False
        for index, service in enumerate(services):
            devices = ",".join(map(str, service.gpu_device_ids))
            logger.info(
                f"Model {index}: {service.recipe.model_name} on GPU {devices} at gpu_memory_utilization="
                f"{service.recipe.engine.llm.gpu_memory_utilization}, port {service.port}"
            )
    else:
        recipe = resolve_for_hardware(resolve_recipe_dir(args.recipe), gpu_name, gpu_count)
        recipe = STRATEGIES[args.scale_out_strategy]().apply(recipe, gpu_count)
        services, load_balancer = replica_services(recipe)
    deploy_config = services[0].recipe.deploy  # a plan validated that every recipe pins the same versions

    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    register_secret(hf_token)
    params = DeployParams(
        server=server,
        ssh_key=args.ssh_key,
        ssh_port=port,
        services=services,
        load_balancer=load_balancer,
        model_dir=args.model_dir,
        hf_token=hf_token,
        dry_run=args.dry_run,
    )
    skip_nvidia = deploy_config.gpu is not None and deploy_config.gpu.startswith("AMD")
    host = RemoteHost(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    timer = PhaseTimer()
    async with timer.ameasure(PHASE_REMOTE_PROVISION):
        await provision_remote(
            host,
            skip_nvidia=skip_nvidia,
            driver_version=deploy_config.driver_version,
            cuda_version=deploy_config.cuda_version,
        )

    if args.teardown:
        return await teardown_entry(params)

    if not await deploy_entry(params, timer=timer):
        sys.exit(1)

    logger.info("\nTiming:")
    for line in timer.format_table().splitlines():
        logger.info(line)


def register_ssh_target(subparsers):
    """Register the SSH deploy target."""
    parser = subparsers.add_parser("ssh", help="Deploy via SSH to a remote server")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", help="Path to recipe directory")
    source.add_argument(
        "--plan",
        help="Plan file: the models to run on the host, each pinned to its GPU devices (see the deploy command "
        "reference); the host must have exactly the plan's GPU name and count. Exclusive with --recipe",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/mnt/models", help="Model cache directory")
    parser.add_argument("--teardown", action="store_true", help="Stop containers instead of deploying")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument(
        "--ssh",
        default=None,
        metavar="USER@HOST[:PORT]",
        help="SSH target (e.g. user@host or user@host:2222). Default port: 22",
    )
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH key path")
    # Deprecated — kept for backwards compatibility. Prefer --ssh USER@HOST[:PORT].
    parser.add_argument("--server", default=None, help="[DEPRECATED] SSH address (user@host); use --ssh instead")
    parser.add_argument("--ssh-port", type=int, default=None, help="[DEPRECATED] SSH port; encode it in --ssh USER@HOST:PORT")
    parser.add_argument("--gpu", default=None, help="Override GPU name (skips detection)")
    parser.add_argument("--gpu-count", type=int, default=None, help="Override GPU count (skips count detection)")
    parser.add_argument(
        "--scale-out-strategy",
        choices=list(STRATEGIES.keys()),
        default=DEFAULT_STRATEGY,
        help=f"Scale-out strategy (default: {DEFAULT_STRATEGY})",
    )
    parser.set_defaults(func=handle_ssh)
