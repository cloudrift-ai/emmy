"""Deploy command layer — registers deploy subcommands."""

from emmy.commands.deploy.boot_script import register_boot_script_target
from emmy.commands.deploy.cloud import register_cloud_target
from emmy.commands.deploy.local import register_local_target
from emmy.commands.deploy.ssh import register_ssh_target

__all__ = [
    "register_boot_script_target",
    "register_cloud_target",
    "register_local_target",
    "register_ssh_target",
]
