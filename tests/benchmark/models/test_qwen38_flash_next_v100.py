"""Configuration gates for the Qwen3.8-Flash-Next V100 fork image."""

import re
from pathlib import Path

from emmy.recipe import load_recipe


def test_onecat_v151_image_is_reproducibly_pinned(project_root) -> None:
    dockerfile = Path(project_root, "docker/1cat-vllm-sm70/Dockerfile.v1.5.1").read_text()
    assert "12.8.1-devel-ubuntu24.04@sha256:4b9ed5fa" in dockerfile
    assert "ONECAT_REPO=https://github.com/cloudrift-ai/1Cat-vLLM.git" in dockerfile
    assert "ONECAT_COMMIT=248d877c549c5800be8a05de5743f173364a7d5a" in dockerfile
    assert "ONECAT_BASE_COMMIT=589a5d1e1a9ea7c42d533eed724a7d67c5f1779d" in dockerfile
    assert 'test "$(git -C /src rev-parse HEAD)" = "${ONECAT_COMMIT}"' in dockerfile
    assert 'merge-base --is-ancestor "${ONECAT_BASE_COMMIT}" "${ONECAT_COMMIT}"' in dockerfile
    assert "ARG ONECAT_VERSION=1.5.1" in dockerfile
    assert 'VLLM_VERSION_OVERRIDE="${ONECAT_VERSION}" CMAKE_BUILD_TYPE=Release' in dockerfile
    assert "ARG TRANSFORMERS_VERSION=5.19.0" in dockerfile
    assert '"transformers==${TRANSFORMERS_VERSION}"' in dockerfile
    assert "TORCH_CUDA_ARCH_LIST=7.0" in dockerfile
    assert "VLLM_SM70_" not in dockerfile  # the fork's switches belong to each recipe, not the image


def test_publish_recipe_names_the_pinned_fork_build(project_root) -> None:
    dockerfile = Path(project_root, "docker/1cat-vllm-sm70/Dockerfile.v1.5.1").read_text()
    commit = re.search(r"ARG ONECAT_COMMIT=([0-9a-f]{40})", dockerfile)[1]
    recipe = load_recipe(str(Path(project_root, "experiments/Qwen3.8-Flash-Next/fork_serving_v100_sxm3")))
    assert recipe.engine.llm.image == f"cloudriftai/1cat-vllm-qwen3.8-flash-next-nvfp4:1.5.1-{commit[:9]}"
