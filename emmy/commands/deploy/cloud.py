"""Cloud deploy target CLI handler."""

import asyncio
import logging
import os
import sys

from emmy.deploy import DeployParams, replica_services
from emmy.deploy import (
    deploy as deploy_entry,
)
from emmy.deploy.plan import load_plan
from emmy.provisioning.cloud import (
    provision_cloud_vm,
    read_public_key_files,
)
from emmy.provisioning.errors import CapacityExhausted, TerminalProvisionError
from emmy.provisioning.host import RemoteHost
from emmy.provisioning.lease import add_lease_arguments, lease_observer
from emmy.provisioning.remote import provision_remote
from emmy.recipe import resolve_for_hardware, resolve_recipe_dir
from emmy.redact import register_secret
from emmy.storage import write_json
from emmy.timing import PHASE_REMOTE_PROVISION, PHASE_VM_PROVISION, PhaseTimer

logger = logging.getLogger(__name__)

RESULT_SCHEMA_VERSION = 1

# ── CLI handler ────────────────────────────────────────────────────


def handle_cloud(args):
    """CLI handler for 'deploy cloud'."""
    asyncio.run(_handle_cloud(args))


async def _handle_cloud(args):
    if args.plan:
        if args.gpu or args.gpu_count:
            logger.error("--gpu and --gpu-count select a matrix entry for --recipe; a plan names its entries itself")
            sys.exit(2)
        try:
            plan = load_plan(args.plan)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            logger.error(f"Invalid plan {args.plan}: {exc}")
            sys.exit(1)
        gpu, gpu_count, services, load_balancer, names = plan.gpu, plan.gpu_count, plan.services, False, plan.recipes
    else:
        if not args.gpu or not args.gpu_count:
            logger.error("--gpu and --gpu-count are required with --recipe")
            sys.exit(2)
        recipe = resolve_for_hardware(resolve_recipe_dir(args.recipe), args.gpu, args.gpu_count)
        services, load_balancer = replica_services(recipe)
        gpu, gpu_count, names = recipe.deploy.gpu, recipe.deploy.gpu_count, [args.recipe] * len(services)
    logger.info(f"GPU: {gpu_count}x {gpu}")
    for index, service in enumerate(services):
        devices = "all" if service.gpu_device_ids is None else ",".join(map(str, service.gpu_device_ids))
        logger.info(
            f"Model {index}: {service.recipe.model_name} on GPU {devices} at gpu_memory_utilization="
            f"{service.recipe.engine.llm.gpu_memory_utilization}, port {service.port}"
        )

    ssh_key = os.path.expanduser(args.ssh_key)
    hf_token = args.hf_token or os.environ.get("HF_TOKEN", "")
    register_secret(hf_token)

    try:
        observer = lease_observer(args, gpu, gpu_count)
        extra_authorized_keys = read_public_key_files(args.authorized_key)
    except (FileNotFoundError, ValueError) as exc:
        logger.error(str(exc))
        sys.exit(1)

    providers_config = None
    if args.billing_exempt or args.network:
        providers_config = {"cloudrift": {}}
        if args.billing_exempt:
            providers_config["cloudrift"]["billing_exempt"] = True
        if args.network:
            providers_config["cloudrift"]["network"] = args.network

    # One host port per service, plus the load balancer's. A plan rents exactly its shape:
    # a larger box would only idle the extra devices while billing for them.
    ports = [22, *(service.port for service in services), *([8080] if load_balancer else [])]
    timer = PhaseTimer()
    async with timer.ameasure(PHASE_VM_PROVISION):
        try:
            conn = await provision_cloud_vm(
                gpu_name=gpu,
                gpu_count=gpu_count,
                ssh_key=ssh_key,
                providers_config=providers_config,
                server_name=args.name,
                dry_run=args.dry_run,
                provider=args.provider,
                extra_authorized_keys=extra_authorized_keys,
                allocation_observer=observer,
                exact_gpu_count=args.plan is not None,
                ports=ports,
            )
        except (CapacityExhausted, TerminalProvisionError, RuntimeError, ValueError) as exc:
            logger.error(str(exc))
            sys.exit(1)
    if conn is None:
        logger.error("Error: VM provisioning failed.")
        sys.exit(1)

    params = DeployParams(
        server=conn.address,
        ssh_key=ssh_key,
        ssh_port=conn.ssh_port,
        services=services,
        load_balancer=load_balancer,
        model_dir=args.model_dir,
        hf_token=hf_token,
        dry_run=args.dry_run,
        port_mappings=conn.port_mappings,
    )
    host = RemoteHost(params.server, params.ssh_key, params.ssh_port, dry_run=params.dry_run)
    deploy_config = services[0].recipe.deploy  # a plan validated that every recipe pins the same versions
    async with timer.ameasure(PHASE_REMOTE_PROVISION):
        await provision_remote(host, driver_version=deploy_config.driver_version, cuda_version=deploy_config.cuda_version)

    if not await deploy_entry(params, timer=timer):
        sys.exit(1)

    if args.result_json and not args.dry_run:
        port_map = dict(conn.port_mappings)
        result = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "cloud_instance_id": conn.delete_info[1],
            "models": [
                {"index": index, "recipe": names[index], "endpoint": f"http://{conn.host}:{port_map.get(service.port, service.port)}/v1"}
                for index, service in enumerate(services)
            ],
        }
        write_json(args.result_json, result, indent=2)
        logger.info(f"Result written to {args.result_json}")

    logger.info("\nTiming:")
    for line in timer.format_table().splitlines():
        logger.info(line)


def register_cloud_target(subparsers):
    """Register the cloud deploy target."""
    parser = subparsers.add_parser("cloud", help="Provision a cloud VM and deploy via SSH")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--recipe", help="Path to recipe directory (with --gpu and --gpu-count)")
    source.add_argument(
        "--plan",
        help="Plan file: the VM to rent and the models to run on it, each pinned to its GPU devices "
        "(see the deploy command reference); exclusive with --recipe",
    )
    parser.add_argument("--name", default="cloud-deploy", help="VM name prefix (default: cloud-deploy)")
    parser.add_argument("--ssh-key", default="~/.ssh/id_ed25519", help="SSH private key path")
    parser.add_argument(
        "--authorized-key",
        action="append",
        default=None,
        metavar="PATH",
        help="Extra SSH public key file to install in the VM's authorized_keys (repeatable)",
    )
    parser.add_argument("--hf-token", default=None, help="HuggingFace token (default: $HF_TOKEN)")
    parser.add_argument("--model-dir", default="/hf_models", help="Model cache directory")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    parser.add_argument("--billing-exempt", action="store_true", help="Skip billing for CloudRift (admin-only)")
    parser.add_argument(
        "--network",
        default=None,
        help="CloudRift network name (must exist in target datacenter; default: provider picks a public network)",
    )
    parser.add_argument("--gpu", help="GPU name (selects matching matrix entry; required with --recipe)")
    parser.add_argument("--gpu-count", type=int, help="GPU count (selects matching matrix entry; required with --recipe)")
    parser.add_argument(
        "--provider",
        choices=["gcp", "cloudrift"],
        default=None,
        help="Force cloud provider (default: first listed for the GPU in the hardware table)",
    )
    add_lease_arguments(parser)
    parser.add_argument("--result-json", metavar="PATH", help="Write the instance id and one endpoint per model here on success")
    parser.set_defaults(func=handle_cloud)
