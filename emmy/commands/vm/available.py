"""``emmy vm available GPU…`` — which of the named GPUs CloudRift can rent right now."""

import asyncio
import json
import logging
import os
import sys

logger = logging.getLogger(__name__)


def _handle_available(args) -> None:
    from emmy.provisioning.candidates import rentable  # noqa: PLC0415
    from emmy.provisioning.cloudrift import list_available_instance_types  # noqa: PLC0415
    from emmy.redact import register_secret  # noqa: PLC0415

    api_key = os.environ.get("CLOUDRIFT_API_KEY")
    if not api_key:
        logger.error("CLOUDRIFT_API_KEY is required")
        sys.exit(2)
    register_secret(api_key)
    try:
        available = asyncio.run(list_available_instance_types(api_key))
    except Exception as exc:  # noqa: BLE001 — a provider error is the command's error
        logger.error(str(exc))
        sys.exit(1)
    print(json.dumps([gpu for gpu in dict.fromkeys(args.gpus) if rentable(gpu, args.gpu_count, "cloudrift", available)]))


def register_available_target(action_subparsers) -> None:
    parser = action_subparsers.add_parser(
        "available", help="Print, as a JSON list, which of the named GPUs CloudRift can rent right now, in the order given"
    )
    parser.add_argument("gpus", nargs="+", help="GPU names from the hardware table, e.g. 'NVIDIA GeForce RTX 5090'.")
    parser.add_argument("--gpu-count", type=int, default=1, help="Exact GPU count per VM (default: 1).")
    parser.set_defaults(func=_handle_available)
