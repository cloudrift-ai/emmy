"""Configuration checks for the 2026 compiler-submission experiments."""

import subprocess
from pathlib import Path

import pytest

from emmy.benchmark.command_workload import build_substitution_map, render_command
from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe

EXP = Path("experiments/golden-bench-2026")


@pytest.mark.parametrize(
    "path",
    [
        pytest.param(path, id=path.name)
        for path in sorted((Path(__file__).resolve().parents[3] / EXP / "kernels" / "golden").glob("*.golden.json"))
    ],
)
def test_kernel_golden_matches_fresh_lowering(path) -> None:
    from emmy.compiler.pipeline.search.golden import GoldenFile, restamp

    document = GoldenFile.load(path)
    assert document.targets() and any(row.measured for row in document.rows)
    fresh, report = restamp(document)
    assert fresh == document, "\n".join(report.lines())


def _experiment(project_root: str, name: str) -> str:
    return str(Path(project_root) / EXP / name)


def _kernel_tasks(project_root: str, study: str):
    tasks = enumerate_tasks([_experiment(project_root, "kernels")])
    return [task for task in tasks if task.variant.params["study"] == study]


def test_common_kernel_corpus_is_small_and_identical(project_root) -> None:
    platforms = {
        "NVIDIA Tesla V100 SXM3 32GB",
        "NVIDIA Tesla V100 SXM2 16GB",
        "NVIDIA A100 80GB",
        "NVIDIA A100 40GB",
        "NVIDIA H100 80GB",
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H200 141GB",
        "NVIDIA B200",
    }
    replayed = {
        "NVIDIA A100 80GB": "a100",
        "NVIDIA A100 40GB": "a100",
        "NVIDIA H100 80GB": "h100",
        "NVIDIA Tesla V100 SXM2 16GB": "v100",
        "NVIDIA GeForce RTX 5090": "rtx5090",
        "NVIDIA GeForce RTX 4090": "rtx4090",
    }
    recipe_dir = _experiment(project_root, "kernels")
    recipe = load_recipe(recipe_dir)
    tasks = _kernel_tasks(project_root, "common")
    assert recipe.kind == "command"
    assert len(tasks) == len(platforms) * 2
    assert {task.recipe.deploy.gpu for task in tasks} == platforms
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {"Qwen/Qwen3-0.6B@c1899de289a04d12100db370d81485cdf75e47ca"}
    replay_tasks = [task for task in tasks if task.recipe.deploy.gpu in replayed]
    searched_tasks = [task for task in tasks if task.recipe.deploy.gpu not in replayed]
    for gpu, short in replayed.items():
        assert {task.variant.params["golden"] for task in replay_tasks if task.recipe.deploy.gpu == gpu} == {
            f"qwen3-06b-s1_{short}",
            f"qwen3-06b-s512_{short}",
        }
    assert all(task.variant.params["budget"] == 0 for task in replay_tasks)
    assert all(task.variant.params["golden"] == "" for task in searched_tasks)
    assert all(task.variant.params["budget"] == 12 for task in searched_tasks)
    for task in tasks:
        fast_exp = task.recipe.deploy.gpu == "NVIDIA H100 80GB" and task.variant.params["seq_len"] == 512
        assert task.recipe.command.env["EMMY_FAST_EXP"] == str(int(fast_exp))

    run = recipe.command.run
    assert "./venv/bin/emmy trace" in run
    assert "torch==2.13.0" in run
    assert "./venv/bin/emmy tune" not in run
    assert '--kernel "$$kernel"' in run
    assert '--tune "$budget"' in run
    assert "--record-greedy" in run
    assert "./venv/bin/emmy run" in run
    assert "for repeat in 0 1 2 3 4" in run
    assert "--golden $task_dir/working.json --bench --strict" in run
    assert 'if [ -n "$golden" ]; then\n  evidence_args=(--strict-evidence --no-record-evidence)' in run
    assert "export EMMY_TUNE_DB=$task_dir/verification/repeat-$$repeat.db" in run
    assert run.count('"$${evidence_args[@]}"') == 2
    assert '"$model_ref" --layer "$layer" --seq-len "$seq_len" --bench' in run
    assert "EMMY_GOLDEN_FILE=$task_dir/working.json" in run
    assert "--bench-backends eager,tcompile,emmy" in run
    assert "--bench-backends eager,emmy" in run
    assert "torch-compile.status" in run
    assert "scripts/" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "Cargo.toml",
        "Cargo.lock",
        "crates",
        "experiments/golden-bench-2026/kernels/recipe.yaml",
        "experiments/golden-bench-2026/kernels/golden",
    ]
    assert recipe.command.strict is True
    assert recipe.command.result_files == ["artifacts.tar.gz"]
    assert "./venv/bin/python -m pip freeze --all" in run
    assert "tar -C $task_dir" in run


def test_native_fp8_kernel_corpus_is_separate_and_identical(project_root) -> None:
    tasks = _kernel_tasks(project_root, "fp8-common")
    assert len(tasks) == 8
    assert {task.recipe.deploy.gpu for task in tasks} == {
        "NVIDIA GeForce RTX 4090",
        "NVIDIA GeForce RTX 5090",
        "NVIDIA H200 141GB",
        "NVIDIA B200",
    }
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {
        "RedHatAI/Qwen3-0.6B-FP8-dynamic@068a9040b238d65f5c1064f10232bb15b96c0ff0"
    }
    assert all(task.variant.params["fp8_mma"] == 1 for task in tasks)
    for task in tasks:
        command = render_command(
            task.recipe.command.run,
            build_substitution_map(task.variant, [0], "/repo", "/task"),
        )
        assert "--require-kernel-source" not in command
        assert "EMMY_FP8_MMA=1" in command


def test_quantized_support_check_covers_three_formats(project_root) -> None:
    recipe_dir = _experiment(project_root, "quantized_kernels_rtx5090")
    recipe = load_recipe(recipe_dir)
    tasks = enumerate_tasks([recipe_dir])
    assert len(tasks) == 4
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA GeForce RTX 5090"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    by_format: dict[str, list] = {}
    for task in tasks:
        by_format.setdefault(task.variant.params["format"], []).append(task)
    assert set(by_format) == {"nvfp4", "awq", "trellis"}
    assert {task.variant.params["seq_len"] for task in by_format["nvfp4"]} == {1, 512}
    assert all(task.variant.params["seq_len"] == 1 for task in by_format["awq"] + by_format["trellis"])

    run = recipe.command.run
    assert "./venv/bin/emmy trace" in run
    assert "torch==2.13.0" in run
    assert "./venv/bin/emmy tune" not in run
    # The CLI owns target enumeration; recipes do not parse a private golden schema.
    assert "--golden $task_dir/working.json --bench" in run
    assert 'd["configs"]' not in run
    assert "--bench --strict --no-record-evidence" in run
    assert "--bench-backends eager,emmy" in run
    assert "tcompile" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "requirements.txt",
        "Makefile",
        "Cargo.toml",
        "Cargo.lock",
        "crates",
        "experiments/golden-bench-2026/quantized_kernels_rtx5090/recipe.yaml",
    ]
    assert recipe.command.strict is True


def test_native_fp8_large_layer_supplement_is_bounded(project_root) -> None:
    tasks = _kernel_tasks(project_root, "fp8-large-layer")
    assert len(tasks) == 4
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA H200 141GB", "NVIDIA B200"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["layer"] for task in tasks} == {0}
    assert {task.variant.params["model_ref"] for task in tasks} == {
        "RedHatAI/Qwen3-32B-FP8-dynamic@c6732fc26128341172e4005bad34aafa51c32866"
    }
    assert all(task.variant.params["budget"] == 8 for task in tasks)


def test_large_layer_corpus_is_bounded_and_not_labeled_tp8(project_root) -> None:
    tasks = _kernel_tasks(project_root, "large-layer")
    assert len(tasks) == 8
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA H200 141GB", "NVIDIA B200"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert {task.variant.params["layer"] for task in tasks} == {0, 3}
    assert {task.variant.params["seq_len"] for task in tasks} == {1, 512}
    assert {task.variant.params["model_ref"] for task in tasks} == {"Qwen/Qwen3.6-27B@6a9e13bd6fc8f0983b9b99948120bc37f49c13e9"}
    for task in tasks:
        command = render_command(
            task.recipe.command.run,
            build_substitution_map(task.variant, list(range(8)), "/repo", "/task"),
        )
        assert "--loop-targets" not in command
        assert "--golden /task/working.json --bench --strict" in command
        assert "--bench-backends eager,tcompile,emmy" in command
        assert "--bench-backends eager,emmy" in command


def test_convergence_check_is_one_shape_and_three_runs(project_root) -> None:
    tasks = _kernel_tasks(project_root, "convergence")
    assert len(tasks) == 3
    assert {task.variant.params["trial"] for task in tasks} == {0, 1, 2}
    assert all(task.recipe.deploy.gpu == "NVIDIA H200 141GB" for task in tasks)
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    assert all(task.variant.params["seq_len"] == 512 for task in tasks)

    fp8_tasks = _kernel_tasks(project_root, "fp8-convergence")
    assert len(fp8_tasks) == 3
    assert {task.variant.params["trial"] for task in fp8_tasks} == {0, 1, 2}
    assert all(task.variant.params["model_ref"].startswith("RedHatAI/Qwen3-0.6B-FP8-dynamic@") for task in fp8_tasks)


def test_search_ablation_is_executable(project_root) -> None:
    tasks = _kernel_tasks(project_root, "search-ablation")
    assert len(tasks) == 4
    assert {task.variant.params["budget"] for task in tasks} == {0, 4, 12, 48}
    assert all("--golden $task_dir/working.json --bench --strict" in task.recipe.command.run for task in tasks)
    assert all(task.recipe.deploy.gpu == "NVIDIA H200 141GB" for task in tasks)
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)


def test_mpk_megakernel_lane_is_pinned_and_paired(project_root) -> None:
    directory = _experiment(project_root, "serving_mpk_qwen3_8b_a100")
    tasks = enumerate_tasks([directory])
    # One row per A100 part. The archived rows are the 80GB part's; the 40GB part has 1555 GB/s
    # against 2039, so it measures the same lane at its own bandwidth and is never merged into them.
    assert [task.recipe.deploy.gpu for task in tasks] == ["NVIDIA A100 80GB", "NVIDIA A100 40GB"]
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)
    recipe = load_recipe(directory)
    assert recipe.kind == "command"

    run = recipe.command.run
    # Pinned external sources: the mirage mpk-branch revision and the Qwen3-8B checkpoint revision.
    assert "5c28cc68dc621cc9448c5c9882ef9e21fdc85884" in run
    assert "b968826d9c46dd6066d109eabc6255188de91218" in run
    # The MPK baseline/megakernel demo pair and stock vLLM are the only paths.
    assert "demo/qwen3/demo.py" in run
    assert "--use-mirage" in run
    assert "vllm==0.23.0" in run
    # No Emmy arm runs here yet — the generative plugin cannot boot Qwen3-8B on sm_80 (#785).
    assert "emmy serve" not in run
    assert "EMMY_GEN_DECODE_BUCKET" not in run
    # ninja is not a vllm dependency, and its engine shells out to it BY NAME: both the install and
    # the venv bin on PATH are load-bearing, and without either every stock repeat fails to boot.
    assert "vllm==0.23.0 ninja" in run
    assert "export PATH=$repo_dir/venv/bin:" in run
    assert "for repeat in 0 1 2 3 4" in run
    # The suite's deterministic serving controls on the single-stream decode point.
    assert "--ignore-eos --temperature 0 --seed 0" in run
    assert "--max-concurrency 1" in run
    assert "--no-enable-prefix-caching" in run


def test_neptune_emmy_pytorch_a100_share_one_experiment(project_root) -> None:
    directory = _experiment(project_root, "compiler_neptune_emmy_pytorch_a100")
    tasks = enumerate_tasks([directory])
    recipe = load_recipe(directory)
    assert recipe.kind == "command"
    assert {task.recipe.deploy.gpu for task in tasks} == {"NVIDIA A100 40GB"}
    assert all(task.recipe.deploy.gpu_count == 1 for task in tasks)

    # One row per (lane, operator): the artifact's ten operators, and the five common operators
    # compared against current PyTorch from a pre-tuned golden.
    common = ("prefill_global", "prefill_causal", "prefill_gqa", "decode_causal", "decode_gqa")
    artifact_only = ("prefill_alibi", "decode_alibi", "prefill_softcap", "decode_softcap", "prefill_windowed")
    lanes: dict[str, set[str]] = {}
    for task in tasks:
        lanes.setdefault(task.variant.params["lane"], set()).add(task.variant.params["operator"])
    assert lanes == {
        "neptune": set(common + artifact_only),
        "emmy": set(common),
    }
    assert len(tasks) == 15

    run = recipe.command.run
    assert "evanzhao16/neptune-env@sha256:724d07594bc817f0fe94267b2d0dbdc6e29d3ae4a7e3516e553a6d9327bfebca" in run
    assert "3aa55c12ac822337e630b809b0d9eabb11eee5d3" in run
    assert "torch==2.13.0" in run
    assert "[ ! -x venv/bin/emmy ]" in run
    assert "import numpy, torch, sys" in run
    assert "EMMY_TUNE_DB=$task_dir/autotune.db" in run
    assert "pip freeze --all" in run
    assert "tar -C $task_dir" in run
    # Each row runs exactly one lane for exactly one operator.
    assert 'case "$lane" in' in run
    assert 'bash /experiment/run.sh "$operator" /experiment/tuning.tar.gz' in run
    assert '"$$EXPERIMENT/run_emmy.sh" $repo_dir/venv/bin/emmy "$operator"' in run
    # Tuning is the tune-kernels skill's job: the recipe only replays what it committed.
    assert "emmy tune" not in run
    assert recipe.command.stage == [
        "emmy",
        "pyproject.toml",
        "README.md",
        "LICENSE",
        "Cargo.toml",
        "Cargo.lock",
        "crates",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/tuning_neptune_a100x1.tar.gz",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_neptune.py",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/operators.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_emmy.sh",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/run_pytorch.py",
        "experiments/golden-bench-2026/compiler_neptune_emmy_pytorch_a100/golden",
    ]
    assert recipe.command.strict is True
    assert recipe.command.result_files == ["artifacts.tar.gz"]
    assert "git apply" not in run

    assert not (Path(directory) / "neptune-inductor.patch").exists()
    assert not (Path(project_root) / EXP / "compiler_neptune_inductor_a100").exists()
    assert not (Path(project_root) / EXP / "compiler_neptune_emmy_tcompile_a100").exists()

    # The artifact runner takes its one operator from the row instead of looping over a private list.
    neptune_runner = (Path(directory) / "run.sh").read_text()
    assert "operator=$1" in neptune_runner
    assert "operators=(" not in neptune_runner
    assert "sequence_lengths=(256 512 1024 2048 4096 8192 16384 32768)" in neptune_runner
    assert "--n-trials 128" in neptune_runner
    assert 'nsys profile -o "/results/profiles/$setup" --trace=cuda,nvtx' in neptune_runner
    assert "--sample=none --cpuctxsw=none --cudabacktrace=none --wait=primary" in neptune_runner
    assert 'profile "$operator" "1,$sequence_length" --repeat 15' in neptune_runner
    assert "neptune-setup-status.tsv" in neptune_runner
    assert "tune_status=ok:no-valid-schedule" in neptune_runner
    assert 'profile_status="$profile_status:mismatch"' in neptune_runner
    assert 'profile_status="$profile_status:runner-failure"' in neptune_runner
    assert 'profile_status="$profile_status:no-saved-optimized-kernel"' in neptune_runner
    assert "tune_status=reused" in neptune_runner
    assert 'tar -xzf "$tuning_archive" -C logs --wildcards "neptune-tuning/${operator}-*"' in neptune_runner
    assert 'test "$successful_profiles" -gt 0' in neptune_runner
    assert 'torch.__version__.split("+")[0]' in neptune_runner
    assert '"2.6.0"' in neptune_runner
    assert "git apply" not in neptune_runner

    neptune_entry = (Path(directory) / "run_neptune.py").read_text()
    assert "NVIDIA A100-SXM4-80GB" in neptune_entry
    assert "NVIDIA A100-SXM4-40GB" in neptune_entry
    assert "NeptuneGQARunner.e = ours.NeptuneGQARunner.create_flex_from_schedulers" in neptune_entry
    assert 'runpy.run_module("scripts.neptune_bench", run_name="__main__")' in neptune_entry

    # One checked-in definition of the common operators, read both by the comparison lane and by the
    # trace that produces a committed golden, so the tuned and benched programs cannot drift apart.
    operators_path = Path(directory) / "operators.sh"
    assert operators_path.stat().st_mode & 0o111
    operators = operators_path.read_text()
    for operator in common:
        assert operator in operators
    for excluded in artifact_only:
        assert excluded not in operators
    assert "SEQUENCE_LENGTHS=(256 512 1024 2048 4096 8192 16384 32768)" in operators
    assert "enable_gqa=True" in operators
    assert "q_length=1" in operators
    assert '"$q_input.reshape(1,8,8,1,128)' in operators
    assert '"$q_input,$k_input,$v_input,is_causal=$is_causal' in operators
    assert "q_input=q" in operators
    assert "local q_input=$q" in operators
    assert '"q=$q;"' in operators
    assert "is_causal=False).reshape(1,64,1,128)" in operators
    assert 'operator_code "$1" "$2" || exit' in operators
    prefill_source = subprocess.check_output([operators_path, "prefill_causal", "256"], text=True)
    decode_source = subprocess.check_output([operators_path, "decode_causal", "256"], text=True)
    assert "q=torch.randn" not in prefill_source
    assert "F.scaled_dot_product_attention(torch.randn" in prefill_source
    assert "q=torch.randn" in decode_source
    assert "F.scaled_dot_product_attention(q,k,v" in decode_source
    assert not (Path(directory) / "run_tune.sh").exists()

    # The comparison lane only measures: it replays the committed golden and never tunes.
    emmy_runner_path = Path(directory) / "run_emmy.sh"
    assert emmy_runner_path.stat().st_mode & 0o111
    emmy_runner = emmy_runner_path.read_text()
    assert "operators.sh" in emmy_runner
    assert '"$emmy" run --golden "$golden" --bench --bench-backends emmy' in emmy_runner
    assert 'run --golden "$golden" --bench --strict' not in emmy_runner
    assert '"$emmy" run -c "$source_code" --bench --strict --bench-backends eager,tcompile,emmy' in emmy_runner
    assert 'EMMY_KNOBS="$reference_knobs"' in emmy_runner
    assert "golden_reference_knobs" in emmy_runner
    assert "emmy tune" not in emmy_runner
    assert "timeout --signal=TERM --kill-after=30s 1500s" in emmy_runner
    assert "setup-status.tsv" in emmy_runner
    assert "replay_1\\treplay_2\\treference_1\\treference_2" in emmy_runner
    assert emmy_runner.count("for repetition in 1 2") == 2
    assert '"$results/json/$setup.replay-$repetition"' in emmy_runner
    assert '"$results/json/$setup.reference-$repetition.json"' in emmy_runner
    assert "replay_statuses=(exact-pin-reference exact-pin-reference)" in emmy_runner
    assert '[[ "$operator" == prefill_* ]]' in emmy_runner
    assert '"${reference_statuses[0]}" = ok' in emmy_runner
    assert '"${reference_statuses[1]}" = ok' in emmy_runner
    assert 'test "$missing_goldens" -eq 0' in emmy_runner
    assert 'test "$successful_setups" -eq "${#SEQUENCE_LENGTHS[@]}"' in emmy_runner
    assert "run_pytorch.py" in emmy_runner
    assert 'reference_statuses+=("pytorch-only:emmy-failed:$reference_status")' in emmy_runner

    pytorch_runner = (Path(directory) / "run_pytorch.py").read_text()
    assert 'mode="max-autotune-no-cudagraphs"' in pytorch_runner
    assert "torch.testing.assert_close" in pytorch_runner
    assert '"captured_whole_forward"' in pytorch_runner


def test_every_command_variant_renders(project_root) -> None:
    root = Path(project_root) / EXP
    rendered = 0
    for recipe_path in sorted(root.glob("*/recipe.yaml")):
        for task in enumerate_tasks([str(recipe_path.parent)]):
            if task.recipe.command is None:
                continue
            substitutions = build_substitution_map(
                task.variant,
                list(range(task.recipe.deploy.gpu_count)),
                "/repo",
                "/task",
            )
            command = render_command(task.recipe.command.run, substitutions)
            assert "/repo" in command
            assert "/task" in command
            subprocess.run(["bash", "-n"], input=command, text=True, check=True)
            rendered += 1
    assert rendered == 69


def test_kernel_tuning_inventory_keeps_file_local_refs(project_root, tmp_path) -> None:
    """A C name may name several definitions; tuning must select every measured file reference exactly."""
    import shlex
    import sys

    from emmy.compiler.pipeline.search.golden import GoldenFile, Kernel, Measurements, Row

    body = {"inputs": [], "outputs": [], "nodes": []}
    document = GoldenFile(
        compute_cap=(12, 0),
        kernels=[Kernel(loop_ir=body, name="k_shared", key=ref, formed=True) for ref in ("k_shared", "k_shared#2", "k_shared#3")],
        rows=[
            Row(name="proposal", kernel="k_shared"),
            *[
                Row(name=label, kernel=ref, knobs={}, measurements=Measurements(emmy_us=1))
                for label, ref in (("first", "k_shared#2"), ("first-again", "k_shared#2"), ("second", "k_shared#3"))
            ],
        ],
    )
    path = document.dump(tmp_path / "working.json")
    command = load_recipe(_experiment(project_root, "kernels")).command.run
    inventory_line = next(line for line in command.splitlines() if "GoldenFile.load" in line)
    script = shlex.split(inventory_line.rstrip().removesuffix("\\"))[2]
    result = subprocess.run([sys.executable, "-c", script, str(path)], check=True, text=True, capture_output=True)
    assert result.stdout.splitlines() == ["k_shared#2", "k_shared#3"]


@pytest.mark.parametrize("tuning,record,verification", [(0, 0, 0), (3, 0, 0), (0, 4, 0), (0, 0, 5), (3, 4, 5)])
@pytest.mark.parametrize("home_relative", [False, True])
def test_kernel_recipe_preserves_failures_and_completes_whole_target_checks(
    project_root, tmp_path, monkeypatch, tuning, record, verification, home_relative
):
    """A rejected candidate or final recording must not discard model diagnostics or any strict replay."""
    task = next(task for task in _kernel_tasks(project_root, "fp8-common") if task.recipe.deploy.gpu == "NVIDIA GeForce RTX 5090")
    bin_dir = tmp_path / "venv" / "bin"
    bin_dir.mkdir(parents=True)
    python = bin_dir / "python"
    python.write_text(
        "#!/bin/sh\n"
        'case "$*" in\n'
        "  *__version__*) echo 2.13.0 ;;\n"
        "  *get_device_capability*) echo sm_120 ;;\n"
        '  *GoldenFile.load*) printf "good\\nbad\\n" ;;\n'
        "esac\n"
    )
    emmy = bin_dir / "emmy"
    emmy.write_text(
        "#!/bin/sh\n"
        '[ "$PYTHONPATH" = "$PWD" ] || exit 97\n'
        'output=trace\nwhile [ "$#" -gt 0 ]; do\n'
        '  if [ "$1" = --json ]; then shift; output=${1##*/}; fi\n'
        "  shift\ndone\n"
        'printf "%s\\n" "$output" >> "$MOCK_CALLS"\n'
        'case "$output" in\n'
        '  bad.json) exit "$MOCK_TUNING_STATUS" ;;\n'
        '  record-final) exit "$MOCK_RECORD_STATUS" ;;\n'
        '  repeat-2) exit "$MOCK_VERIFICATION_STATUS" ;;\n'
        "esac\n"
    )
    python.chmod(0o755)
    emmy.chmod(0o755)
    calls = tmp_path / "calls.txt"
    monkeypatch.setenv("PYTHONPATH", "/older-checkout")
    monkeypatch.setenv("HOME", str(tmp_path.parent))
    for key, value in {
        "MOCK_CALLS": calls,
        "MOCK_TUNING_STATUS": tuning,
        "MOCK_RECORD_STATUS": record,
        "MOCK_VERIFICATION_STATUS": verification,
    }.items():
        monkeypatch.setenv(key, str(value))
    task_dir = tmp_path / "task"
    repo_dir = f"~/{tmp_path.name}" if home_relative else str(tmp_path)
    command = render_command(task.recipe.command.run, build_substitution_map(task.variant, [0], repo_dir, str(task_dir)))
    result = subprocess.run(["bash"], input=command, text=True, capture_output=True, timeout=10)
    assert result.returncode == (verification or record or tuning), result.stderr
    assert calls.read_text().splitlines() == [
        "trace",
        "initial",
        "record-initial",
        "good.json",
        "bad.json",
        "record-final",
        "model.json",
        *(f"repeat-{index}" for index in range(5)),
    ]
    assert (task_dir / "tuning.status").read_text().strip() == str(tuning)
    assert (task_dir / "record-final.status").read_text().strip() == str(record)
    assert [(task_dir / "verification" / f"repeat-{index}.status").read_text().strip() for index in range(5)] == [
        "0",
        "0",
        str(verification),
        "0",
        "0",
    ]
