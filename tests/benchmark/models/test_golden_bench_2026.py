"""Configuration checks for the 2026 compiler-submission experiments."""

import subprocess
from pathlib import Path

from emmy.benchmark.command_workload import build_substitution_map, render_command
from emmy.benchmark.tasks import enumerate_tasks
from emmy.recipe import load_recipe

EXP = Path("experiments/golden-bench-2026")


def _experiment(project_root: str, name: str) -> str:
    return str(Path(project_root) / EXP / name)


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
    assert rendered == 21
