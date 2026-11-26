"""Cycle GCP zones until one exact GPU allocation is ready, optionally run Emmy recipes and release it."""

import argparse
import asyncio
import json
import logging
import signal
import sys
from pathlib import Path

from emmy.benchmark.config import load_config
from emmy.hardware import GPU_GCP_ZONES, GPU_INSTANCE_TYPES, resolve_instance_type
from emmy.provisioning.cloud import provision_cloud_vm
from emmy.provisioning.lease import VmLeaseObserver, delete_owned_vm, lease_is_active, load_owned_lease
from emmy.provisioning.shell import run_shell_cmd

logger = logging.getLogger(__name__)


async def rent(args):
    """Retry exhausted capacity; provider authentication and configuration errors still stop the run."""
    if args.lease.exists():
        lease = load_owned_lease(args.lease, args.owner)
        if await lease_is_active(lease):
            if lease["request"] != {"gpu": args.gpu, "gpu_count": args.gpu_count}:
                raise RuntimeError("Existing active lease does not match the requested GPU and count")
            if lease["status"] != "active":
                raise RuntimeError("Existing lease is not ready; inspect it before restarting")
            return lease["vm"]["ssh_target"]
    machine = next(resolve_instance_type(p, t, args.gpu_count) for p, t in GPU_INSTANCE_TYPES[args.gpu] if p == "gcp")
    rc, output, error = await run_shell_cmd(
        ["gcloud", "compute", "machine-types", "list", f"--filter=name={machine}", "--format=json"], timeout=120
    )
    if rc:
        raise RuntimeError(error)
    available = {row["zone"].rsplit("/", 1)[-1] for row in json.loads(output)}
    preferred = args.zones or GPU_GCP_ZONES[args.gpu]
    zones = [zone for zone in preferred if zone in available]
    if not args.zones:
        zones += sorted(available - set(zones))
    if not zones:
        raise ValueError(f"No zones advertise {machine}")
    GPU_GCP_ZONES[args.gpu] = zones
    observer = VmLeaseObserver(args.lease, args.owner, args.gpu, args.gpu_count)
    config = load_config(args.config)
    while True:
        connection = await provision_cloud_vm(
            args.gpu,
            args.gpu_count,
            str(args.ssh_key.expanduser()),
            providers_config=config.get("providers"),
            server_name=args.name,
            provider="gcp",
            provisioning_model="SPOT",
            exact_gpu_count=True,
            allocation_observer=observer,
        )
        if connection:
            return load_owned_lease(args.lease, args.owner)["vm"]["ssh_target"]
        logger.info("All zones exhausted; retrying in %s seconds", args.retry_seconds)
        await asyncio.sleep(args.retry_seconds)


async def run(args):
    target = await rent(args)
    logger.info("Spot node ready; connection recorded in %s", args.lease)
    if not args.recipes:
        return 0
    process = None
    try:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "emmy.emmy",
            "bench",
            *args.recipes,
            "--ssh",
            target,
            "--ssh-key",
            str(args.ssh_key.expanduser()),
            "--config",
            args.config,
            *[part for value in args.filters for part in ("--filter", value)],
        )
        return await process.wait()
    finally:
        if process is not None and process.returncode is None:
            process.terminate()
            await process.wait()
        await delete_owned_vm(args.lease, args.owner)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--lease", type=Path, required=True)
    parser.add_argument("--owner", required=True)
    parser.add_argument("--name", default="spot")
    parser.add_argument("--zones", nargs="+", help="Preferred zones, in order; defaults to every advertised zone")
    parser.add_argument("--retry-seconds", type=int, default=300)
    parser.add_argument("--ssh-key", type=Path, default=Path("~/.ssh/id_ed25519"))
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--filter", dest="filters", action="append", default=[], help="Pass a KEY=PATTERN filter to emmy bench; repeatable")
    parser.add_argument("recipes", nargs="*", help="After renting, bench these recipes and delete the owned VM")
    args = parser.parse_args()
    if args.retry_seconds < 1 or args.gpu_count < 1:
        parser.error("GPU count and retry interval must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    # asyncio.run handles SIGINT by cancelling the main task; use the same path for SIGTERM.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    return asyncio.run(run(args))


if __name__ == "__main__":
    sys.exit(main())
