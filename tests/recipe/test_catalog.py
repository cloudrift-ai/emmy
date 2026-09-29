"""Recipe catalog inventory and onboarding-stub creation."""

import pytest
import yaml

from emmy.recipe.catalog import CATALOG_SCHEMA_VERSION, create_recipe_stub, recipe_inventory, recipe_inventory_document

GPU = "NVIDIA H200 141GB"


def _write_recipe(root, name, model_id, tags, *, rationale="Useful model."):
    path = root / name / "recipe.yaml"
    path.parent.mkdir(parents=True)
    path.write_text(
        yaml.safe_dump(
            {
                "tags": tags,
                "model": {"huggingface": model_id, "rationale": rationale, "task": "generate"},
                "engine": {"llm": {"context_length": 8192, "vllm": {"image": "example/vllm"}}},
                "matrices": [
                    {"deploy.gpu": GPU, "deploy.gpu_count": 1},
                    {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2, "engine.llm.context_length": 16384},
                ],
            },
            sort_keys=False,
        )
    )
    return path


def test_recipe_inventory_filters_tags_and_reports_deployments(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "ready", "org/ready", ["maintained"])
    _write_recipe(root, "other", "org/other", ["best-effort"])

    inventory = recipe_inventory(root, ("maintained",))

    assert inventory == [
        {
            "path": str(root / "ready" / "recipe.yaml"),
            "name": "ready",
            "model_id": "org/ready",
            "tags": ["maintained"],
            "task": "generate",
            "runnable": True,
            "deployments": [
                {"gpu": GPU, "gpu_count": 1, "gpu_memory_utilization": 0.9, "context_length": 8192},
                {"gpu": "NVIDIA B200", "gpu_count": 2, "gpu_memory_utilization": 0.9, "context_length": 16384},
            ],
            "rationale": "Useful model.",
            "heat": None,
        }
    ]


def test_recipe_inventory_document_is_versioned(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "ready", "org/ready", ["maintained"])

    document = recipe_inventory_document(root)

    assert document["schema_version"] == CATALOG_SCHEMA_VERSION == 2
    assert [recipe["name"] for recipe in document["recipes"]] == ["ready"]


def test_recipe_inventory_keeps_entries_that_differ_only_by_memory_fraction(tmp_path):
    """A recipe that may share its GPU lists one deployment per fraction, each with its own context."""
    root = tmp_path / "recipes"
    recipe = _write_recipe(root, "shared", "org/shared", ["maintained"])
    config = yaml.safe_load(recipe.read_text())
    config["matrices"] = {
        "zip": {
            "deploy.gpu": [GPU, GPU],
            "deploy.gpu_count": [1, 1],
            "engine.llm.gpu_memory_utilization": [0.9, 0.3],
            "engine.llm.context_length": [131072, 32768],
        }
    }
    recipe.write_text(yaml.safe_dump(config, sort_keys=False))

    assert recipe_inventory(root)[0]["deployments"] == [
        {"gpu": GPU, "gpu_count": 1, "gpu_memory_utilization": 0.9, "context_length": 131072},
        {"gpu": GPU, "gpu_count": 1, "gpu_memory_utilization": 0.3, "context_length": 32768},
    ]


def test_bundled_qwen3_30b_a3b_lists_its_shared_gpu_entry(recipes_dir):
    """The bundled recipe that may share an H200 exposes both of its qualified entries."""
    record = next(record for record in recipe_inventory(recipes_dir) if record["name"] == "Qwen3-30B-A3B-Instruct-2507")
    assert record["deployments"] == [
        {"gpu": GPU, "gpu_count": 1, "gpu_memory_utilization": 0.9, "context_length": 262144},
        {"gpu": GPU, "gpu_count": 1, "gpu_memory_utilization": 0.55, "context_length": 131072},
    ]


def test_recipe_inventory_rejects_invalid_heat(tmp_path):
    root = tmp_path / "recipes"
    recipe = _write_recipe(root, "ready", "org/ready", ["maintained"])
    recipe.write_text(recipe.read_text().replace("  rationale: Useful model.\n", "  rationale: Useful model.\n  heat: 101\n"))

    with pytest.raises(ValueError, match="heat must be an integer from 0 to 100"):
        recipe_inventory(root)


def test_disabled_recipe_is_not_runnable_but_keeps_proposed_deployments(tmp_path):
    root = tmp_path / "recipes"
    recipe = _write_recipe(root, "old", "org/old", ["obsolete"])

    record = recipe_inventory(root)[0]

    assert record["runnable"] is False
    assert record["deployments"][0]["context_length"] == 8192
    assert recipe.is_file()


def test_create_recipe_stub_writes_native_deployment_matrix(tmp_path):
    root = tmp_path / "recipes"
    recipe = create_recipe_stub(
        root,
        "org/new-model",
        "Strong current serving demand.",
        "generate",
        [
            {"deploy.gpu": GPU, "deploy.gpu_count": 1},
            {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2},
        ],
    )

    config = yaml.safe_load(recipe.read_text())
    assert config == {
        "tags": ["onboarding", "untested"],
        "model": {
            "huggingface": "org/new-model",
            "rationale": "Strong current serving demand.",
            "heat": 0,
            "task": "generate",
        },
        "matrices": [
            {"deploy.gpu": GPU, "deploy.gpu_count": 1},
            {"deploy.gpu": "NVIDIA B200", "deploy.gpu_count": 2},
        ],
    }
    assert list(config["model"])[:2] == ["huggingface", "rationale"]


@pytest.mark.parametrize("heat", [-1, 101, True, "90"])
def test_create_recipe_stub_rejects_invalid_heat(tmp_path, heat):
    root = tmp_path / "recipes"

    with pytest.raises(ValueError, match="heat must be an integer from 0 to 100"):
        create_recipe_stub(
            root,
            "org/new-model",
            "Strong current serving demand.",
            "generate",
            [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
            heat,
        )


def test_create_recipe_stub_uses_organization_when_checkpoint_directory_exists(tmp_path):
    root = tmp_path / "recipes"
    _write_recipe(root, "new-model", "other/new-model", ["best-effort"])

    recipe = create_recipe_stub(
        root,
        "org/new-model",
        "Different organization checkpoint.",
        "generate",
        [{"deploy.gpu": GPU, "deploy.gpu_count": 1}],
    )

    assert recipe == root / "org--new-model" / "recipe.yaml"
