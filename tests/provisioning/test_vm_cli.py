"""``emmy vm available`` and ``emmy vm delete cloudrift --tag`` — the two CLI forms the renting workflows call."""

import argparse
import json

from emmy.commands.vm import register_vm_command
from emmy.provisioning import cloudrift


def _run(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    register_vm_command(parser.add_subparsers())
    args = parser.parse_args(argv)
    args.func(args)


def test_available_names_the_gpus_cloudrift_can_rent_in_the_order_given(monkeypatch, capsys):
    async def instance_types(api_key):
        assert api_key == "key"
        return {"rtx59-7-50-400-ec.1", "rtx59-7-50-400-ec.2"}

    monkeypatch.setenv("CLOUDRIFT_API_KEY", "key")
    monkeypatch.setattr(cloudrift, "list_available_instance_types", instance_types)
    _run(["vm", "available", "NVIDIA Tesla V100 SXM3 32GB", "NVIDIA GeForce RTX 5090", "NVIDIA H100 80GB", "Not A GPU"])
    assert json.loads(capsys.readouterr().out) == ["NVIDIA GeForce RTX 5090"]


def test_delete_by_tag_terminates_every_vm_carrying_the_comma_separated_tags(monkeypatch):
    calls = []

    async def terminate(api_key, tags, api_url, *, audit_attempts, audit_delay):
        calls.append((api_key, tags, audit_attempts, audit_delay))

    monkeypatch.setenv("CLOUDRIFT_API_KEY", "key")
    monkeypatch.setattr(cloudrift, "terminate_instances_by_tags", terminate)
    _run(["vm", "delete", "cloudrift", "--tag", "emmy,workflow:golden-fill", "--audit-attempts", "3", "--audit-delay", "1"])
    assert calls == [("key", ["emmy", "workflow:golden-fill"], 3, 1.0)]
