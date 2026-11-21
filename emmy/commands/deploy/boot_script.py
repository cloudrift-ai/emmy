"""`deploy boot-script`: the first-boot script for a VM that serves a model by itself."""

import asyncio
import json
import logging
import os
import sys

import httpx

from emmy.deploy import replica_services
from emmy.deploy.boot_script import check_api_key, gated_models, render_boot_script, serving_port
from emmy.deploy.orchestrate import model_downloads
from emmy.hardware import resolve_instance_type
from emmy.provisioning.cloudrift import select_image_url
from emmy.recipe import resolve_for_hardware, resolve_recipe_dir

logger = logging.getLogger(__name__)

BOOT_SCRIPT_SCHEMA_VERSION = 1
# Read from the environment rather than argv so the key never shows in a process listing.
API_KEY_ENV = "SERVE_API_KEY"


def handle_boot_script(args):
    """CLI handler for 'deploy boot-script'."""
    asyncio.run(_handle_boot_script(args))


async def _handle_boot_script(args):
    api_key = os.environ.get(API_KEY_ENV, "")
    try:
        check_api_key(api_key)
    except ValueError as exc:
        logger.error(f"{API_KEY_ENV}: {exc}")
        sys.exit(2)

    recipe = resolve_for_hardware(resolve_recipe_dir(args.recipe), args.gpu, args.gpu_count)
    services, load_balancer = replica_services(recipe)
    try:
        script = render_boot_script(services, load_balancer, api_key)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(2)

    try:
        gated = await gated_models([model_name for _, model_name, _ in model_downloads(services)])
    except httpx.HTTPError as exc:
        logger.error(f"Could not confirm the model is downloadable without a Hugging Face token: {exc}")
        sys.exit(1)
    if gated:
        logger.error(f"Hugging Face gates {', '.join(gated)}; a VM without our token cannot download it")
        sys.exit(2)

    # Not the logger: it redacts the API key, which the script must carry.
    print(
        json.dumps(
            {
                "schema_version": BOOT_SCRIPT_SCHEMA_VERSION,
                "image_url": select_image_url(resolve_instance_type("cloudrift", args.instance_type, args.gpu_count)),
                "port": serving_port(load_balancer, services),
                "script": script,
            },
            sort_keys=True,
        )
    )


def register_boot_script_target(subparsers):
    """Register the boot-script deploy target."""
    parser = subparsers.add_parser(
        "boot-script",
        help="Print the first-boot script for a CloudRift VM that serves a model by itself (JSON)",
    )
    parser.add_argument("--recipe", required=True, help="Recipe name or directory")
    parser.add_argument("--gpu", required=True, help="GPU name (selects the matrix entry)")
    parser.add_argument("--gpu-count", type=int, required=True, help="GPU count (selects the matrix entry)")
    parser.add_argument(
        "--instance-type",
        required=True,
        metavar="TYPE",
        help="CloudRift base instance type the VM is rented as (picks the OS image)",
    )
    parser.set_defaults(func=handle_boot_script)
