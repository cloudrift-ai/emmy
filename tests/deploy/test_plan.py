"""Plan files: several models on one VM, validated before anything is rented."""

import json

import pytest
import yaml

from emmy.deploy.plan import load_plan

GPU = "NVIDIA H200 141GB"


def _recipe_dir(tmp_path, name, *, gpu_count=1, tensor_parallel_size=1, fractions=(0.9,), driver_version=None):
    """A recipe with one H200 matrix entry per fraction."""
    directory = tmp_path / name
    directory.mkdir()
    entries = []
    for fraction in fractions:
        entry = {"deploy.gpu": GPU, "deploy.gpu_count": gpu_count, "engine.llm.gpu_memory_utilization": fraction}
        if driver_version:
            entry["deploy.driver_version"] = driver_version
        entries.append(entry)
    (directory / "recipe.yaml").write_text(
        yaml.safe_dump(
            {
                "model": {"huggingface": f"org/{name}"},
                "engine": {"llm": {"tensor_parallel_size": tensor_parallel_size, "vllm": {"image": "example/vllm"}}},
                "matrices": entries,
            }
        )
    )
    return str(directory)


def _plan(tmp_path, models, gpu_count=1):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 1, "gpu": GPU, "gpu_count": gpu_count, "models": models}))
    return path


def _model(recipe, fraction, devices):
    return {"recipe": recipe, "gpu_memory_utilization": fraction, "gpu_device_ids": devices}


def test_plan_resolves_each_model_onto_its_own_service(tmp_path):
    big = _recipe_dir(tmp_path, "big", fractions=(0.9, 0.55))
    small = _recipe_dir(tmp_path, "small", fractions=(0.9, 0.35))

    plan = load_plan(_plan(tmp_path, [_model(big, 0.55, [0]), _model(small, 0.35, [0])]))

    assert (plan.gpu, plan.gpu_count, plan.recipes) == (GPU, 1, [big, small])
    assert [(service.gpu_device_ids, service.port) for service in plan.services] == [([0], 8000), ([0], 8001)]
    assert [service.recipe.engine.llm.gpu_memory_utilization for service in plan.services] == [0.55, 0.35]
    assert [service.recipe.model_name for service in plan.services] == ["org/big", "org/small"]


def test_plan_unknown_recipe_names_the_model(tmp_path):
    with pytest.raises(ValueError, match=r"model 0 \(nowhere\): No recipe directory"):
        load_plan(_plan(tmp_path, [_model("nowhere", 0.9, [0])]))


def test_plan_requires_an_exact_fraction_entry(tmp_path):
    recipe = _recipe_dir(tmp_path, "one", fractions=(0.9,))
    with pytest.raises(ValueError, match=r"model 0 \(.*one\): .*Available fractions: \[0.9\]"):
        load_plan(_plan(tmp_path, [_model(recipe, 0.5, [0])]))


@pytest.mark.parametrize("devices", [[], [0, 0], [2], [-1], ["0"], [True]])
def test_plan_rejects_bad_device_ids(tmp_path, devices):
    recipe = _recipe_dir(tmp_path, "one")
    with pytest.raises(ValueError, match="gpu_device_ids must be distinct integers below 2"):
        load_plan(_plan(tmp_path, [_model(recipe, 0.9, devices)], gpu_count=2))


def test_plan_shared_device_budget(tmp_path):
    recipe = _recipe_dir(tmp_path, "one", fractions=(0.95, 0.6, 0.55, 0.4))

    with pytest.raises(ValueError, match="GPU device 0 is shared by models whose fractions add up to 1, above 0.95"):
        load_plan(_plan(tmp_path, [_model(recipe, 0.6, [0]), _model(recipe, 0.4, [0])]))
    assert len(load_plan(_plan(tmp_path, [_model(recipe, 0.55, [0]), _model(recipe, 0.4, [0])])).services) == 2
    assert len(load_plan(_plan(tmp_path, [_model(recipe, 0.95, [0])])).services) == 1  # a single tenant keeps its own value


def test_plan_rejects_an_entry_that_needs_replicas(tmp_path):
    recipe = _recipe_dir(tmp_path, "wide", gpu_count=2, tensor_parallel_size=1)
    with pytest.raises(ValueError, match="spreads 2 GPUs over replicas"):
        load_plan(_plan(tmp_path, [_model(recipe, 0.9, [0, 1])], gpu_count=2))


def test_plan_rejects_disagreeing_driver_pins(tmp_path):
    pinned = _recipe_dir(tmp_path, "pinned", driver_version="550")
    plain = _recipe_dir(tmp_path, "plain")
    with pytest.raises(ValueError, match="driver/CUDA version pins"):
        load_plan(_plan(tmp_path, [_model(pinned, 0.9, [0]), _model(plain, 0.9, [1])], gpu_count=2))


def test_plan_rejects_other_schema_versions(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(json.dumps({"schema_version": 2, "gpu": GPU, "gpu_count": 1, "models": []}))
    with pytest.raises(ValueError, match="schema_version must be 1"):
        load_plan(path)
