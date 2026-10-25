<p align="center">
  <a href="https://pypi.org/project/emmy-ml/"><img src="https://img.shields.io/pypi/v/emmy-ml" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

**Compile → Benchmark → Deploy.** Emmy compiles PyTorch models into CUDA kernels, benchmarks LLM inference, and
deploys models on local or cloud GPUs with vLLM and SGLang.

[Compile](#compile) · [Tune](#tune) · [Deploy](#deploy) · [Benchmark](#benchmark) · [Serve](#serve) · [Recipes](#recipes)

See Emmy in action: [Outperforming vLLM on Gemma4-12B](https://www.cloudrift.ai/blog/optimizing-gemma-4-12b-rtx).

## Install

Requires Python 3.12 or newer.

```bash
pip install emmy-ml
emmy --version
```

The package includes ready-to-use recipes. Remote deployment and benchmarking do not require a local GPU or CUDA
installation. For gated Hugging Face models, accept the model's license and set `HF_TOKEN`.

## Compile

Compile a PyTorch operation, compare its speed with PyTorch, or inspect the generated CUDA:

```bash
pip install "emmy-ml[compile]"

# Compile an operation
emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))"

# Benchmark against eager PyTorch and torch.compile
emmy run --bench -c "nn.Softmax(dim=-1)(torch.randn(1,28,2048,2048))"

# Inspect the generated CUDA
emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir cuda
```

Running compiled kernels requires an NVIDIA GPU and the CUDA toolkit (`nvcc`); source builds also need Rust.
For GPUs older than Turing, use CUDA 12 and a PyTorch build that supports your GPU. Fast math is enabled by default;
set `EMMY_FAST_MATH=0` to disable it.

You can also compile a custom module:

```bash
emmy compile -c "
class LN(nn.Module):
    def forward(self, x):
        m = x.mean(-1, keepdim=True)
        v = ((x - m) ** 2).mean(-1, keepdim=True)
        return (x - m) * torch.rsqrt(v + 1e-6)
LN()(torch.randn(64, 2048))"
```

For Python integration, use `torch.compile(model, backend="emmy", dynamic=False)` under `torch.no_grad()`.
Read the [compiler guide](emmy/compiler/ARCHITECTURE.md) or the
[compiler walkthrough](https://www.cloudrift.ai/blog/building-gpu-compiler-from-scratch-1) for more.

### Lowered kernel example

**Readable Schedule**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir tile`
```
kernel k_rms_norm_reduce  inputs: rms_norm_mean_count, rms_norm_eps, x, p_weight  outputs: rms_norm
    in0 = load rms_norm_mean_count[0]
    in1 = load rms_norm_eps[0]
    Tile(axes=(a0:256=THREAD, a1:32=BLOCK)):
        x_smem = Stage(x, origin=(0, a1, 0), slab=(a2:2048@2)) async
        p_weight_smem = Stage(p_weight, origin=(0), slab=(a3:2048@0)) async
        StridedLoop(a2 = a0; < 2048; += 256):  # reduce
            in2 = load x_smem[a2]
            v0 = multiply(in2, in2)
            acc0 <- add(acc0, v0)
        v1 = divide(acc0, in0)
        v2 = add(v1, in1)
        v3 = rsqrt(v2)
        StridedLoop(a3 = a0; < 2048; += 256):  # free
            in3 = load x_smem[a3]
            in4 = load p_weight_smem[a3]
            v4 = multiply(in3, v3)
            v5 = multiply(v4, in4)
            rms_norm[0, a1, a3] = v5
```

**Optimized CUDA kernel**: `emmy compile -c "nn.RMSNorm(2048)(torch.randn(1,32,2048))" --ir cuda`

```c
extern "C" __global__
__launch_bounds__(256) void k_rms_norm_reduce(const float* x, const float* p_weight, float* rms_norm) {
    float in0 = 2048.0f;
    float in1 = 1e-06f;
    {
        int a1 = blockIdx.x;
        int a0 = threadIdx.x;
        float acc0 = 0.0f;
        __syncthreads();
        __shared__ float x_smem[2048];
        for (int x_smem_flat = a0; x_smem_flat < 2048; x_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&x_smem[x_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&x[a1 * 2048 + x_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        __shared__ float p_weight_smem[2048];
        for (int p_weight_smem_flat = a0; p_weight_smem_flat < 2048; p_weight_smem_flat += 256) {
            {
                unsigned int _smem_addr = __cvta_generic_to_shared(&p_weight_smem[p_weight_smem_flat]);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 4;\n"
                             :: "r"(_smem_addr), "l"(&p_weight[p_weight_smem_flat])
                             : "memory");
            }
        }
        asm volatile("cp.async.commit_group;\n" ::: "memory");
        asm volatile("cp.async.wait_group 0;\n" ::: "memory");
        __syncthreads();
        for (int a2 = a0; a2 < 2048; a2 += 256) {
            float in2 = x_smem[a2];
            float v0 = in2 * in2;
            acc0 += v0;
        }
        __shared__ float acc0_smem[256];
        acc0_smem[a0] = acc0;
        __syncthreads();
        for (int s = 128; s > 0; s >>= 1) {
            if (a0 < s) {
                acc0_smem[a0] = acc0_smem[a0] + acc0_smem[a0 + s];
            }
            __syncthreads();
        }
        __syncthreads();
        float acc0_b = acc0_smem[0];
        float v1 = acc0_b / in0;
        float v2 = v1 + in1;
        float v3 = rsqrtf(v2);
        for (int a3 = a0; a3 < 2048; a3 += 256) {
            float in3 = x_smem[a3];
            float in4 = p_weight_smem[a3];
            float v4 = in3 * v3;
            float v5 = v4 * in4;
            rms_norm[a1 * 2048 + a3] = v5;
        }
    }
}
```

## Tune

Try 20 schedules for a single kernel on your GPU, with cold-cache measurements:

```bash
emmy run --bench --cold-cache --tune 20 -c "nn.Linear(4096,4096,bias=False)(torch.randn(1,4096))"
```

Successful measurements are saved so later compiles can reuse them when the kernel and measurement settings match.

### Fit the priors

The schedule prior ranks kernel schedules; the placement prior chooses how to split work into kernels. Both guide
compilation when no compatible measurement is available. You can refit them from the repository's golden files in a
[source checkout](#contributing), without a GPU:

```bash
# Load the repository goldens and the measurement freeze into a separate database
emmy db import --db _data/dataset.db --fresh --repository

# Export a dataset for each prior
emmy db export --db _data/dataset.db _data/schedule --space schedule
emmy db export --db _data/dataset.db _data/placement --space placement

# Refit the shipped weights
emmy fit _data/schedule emmy/compiler/pipeline/search/prior/weights/schedule.json
emmy fit _data/placement emmy/compiler/pipeline/search/prior/weights/placement.json

# Evaluate both priors
emmy eval prior _data/schedule
emmy eval prior _data/placement
```

To experiment without replacing the shipped weights, fit a separate candidate and compare it on the same dataset:

```bash
emmy fit _data/schedule _data/schedule-candidate.json --folds 0
emmy eval prior _data/schedule \
  --offline-file emmy/compiler/pipeline/search/prior/weights/schedule.json \
  --compare-to _data/schedule-candidate.json --json _data/schedule-comparison.json
```

Adjust `--iterations`, `--depth`, or `--learning-rate` to try different fits. `--folds 0` skips cross-validation;
omit it to use the default five folds. Each fit writes metrics under `_tune/fits/`, and the comparison command writes
a JSON report. See the [CLI reference](emmy/commands/ARCHITECTURE.md) for all tuning and fitting options.

### Keep what a tune measured

`emmy run --bench --tune N` writes every clean row it measures into the tune DB of the machine it ran on, and
freezes the same rows into one file beside the card's under `emmy/compiler/pipeline/search/freezes/`, the
measurement freeze the repository keeps, one golden file per card in git LFS: `--repository` reads it beside the
goldens, and the nightly refresh reports the prior's regret over its pools. Commit the tune's file with your change
(run `git lfs install` once on the machine that commits); the nightly refresh folds the files into each card's file
on `main`. A tune on a rented card writes the file into the copy there: copy it back next to the card's file, as you
do the working golden. `EMMY_FREEZE_DIR` names another directory or, set empty, none.

A tune DB from a machine with no checkout is merged by hand (`~/.cache/emmy/autotune.db`, or the file `EMMY_TUNE_DB`
names, copied over):

```bash
emmy db import --db _data/freeze.db --fresh emmy/compiler/pipeline/search/freezes autotune.db
emmy db freeze --db _data/freeze.db --out emmy/compiler/pipeline/search/freezes
```

A schedule measured on both sides takes the tune DB's row, faster or slower, a kernel the compiler no longer lowers
drops out, and merging the same tune DB again changes nothing. Commit the rewritten files.

## Deploy

Choose a recipe that matches your GPU. This Gemma recipe targets one RTX 5090:

```bash
emmy recipe list

# Local Docker Compose
emmy deploy local --recipe gemma-4-12B-it

# An existing server
emmy deploy ssh --recipe gemma-4-12B-it --ssh user@host

# Rent a cloud VM and deploy
emmy deploy cloud --recipe gemma-4-12B-it
```

A recipe name creates an editable copy in the current directory. You can also pass a recipe directory with
`--recipe path/to/recipe`. Deployment leaves the server running and prints its endpoint.

Local deployment needs Docker Compose and NVIDIA container support. Cloud deployment needs provider credentials:
`CLOUDRIFT_API_KEY` for CloudRift, or an authenticated `gcloud` CLI for GCP.
See the [deployment reference](emmy/commands/ARCHITECTURE.md) for teardown, provider options, and multi-model plans.

## Benchmark

From a [source checkout](#contributing), run an experiment on cloud GPUs, your machine, or existing servers:

```bash
# Rent GPUs, run one experiment, and tear down
emmy bench experiments/gemma-4-12B/gsm8k_mtp_rtx5090

# Select RTX 5090 variants across Gemma experiments
emmy bench experiments/gemma-4-12B/* --filter "deploy.gpu=*5090*"

# Use this machine or an existing server
emmy bench experiments/gemma-4-12B/gsm8k_mtp_rtx5090 --local
emmy bench experiments/gemma-4-12B/gsm8k_mtp_rtx5090 --ssh user@host
```

Results and logs are saved in a timestamped directory beside the experiment recipe. Cloud VMs are removed after the
run; existing hosts are kept. Local benchmarks use SSH to localhost, so local SSH access must be configured.
Browse [experiments and results](experiments/) or read the [benchmark reference](emmy/commands/ARCHITECTURE.md).

## Serve

Serve an embedding model through vLLM with Emmy-compiled kernels:

```bash
pip install "emmy-ml[compile,serving]"
emmy serve Qwen/Qwen3-Embedding-0.6B
```

Then send a request from another terminal:

```bash
curl localhost:8000/v1/embeddings -H 'Content-Type: application/json' \
  -d '{"model":"Qwen/Qwen3-Embedding-0.6B","input":"Hello"}'
```

For a one-shot benchmark and a stock-vLLM comparison:

```bash
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32
emmy serve Qwen/Qwen3-Embedding-0.6B --bench --random-input-len 32 --stock
```

See the [serving guide](emmy/serving/ARCHITECTURE.md) for text generation and the
[native generation guide](emmy/serving/native/ARCHITECTURE.md) for the experimental Rust server.

## Recipes

A recipe sets the model, serving engine, GPU, and optional benchmark workload. Save this as `my-benchmark/recipe.yaml`
and run `emmy bench my-benchmark`:

```yaml
model:
  huggingface: "Qwen/Qwen3-0.6B"

engine:
  llm:
    tensor_parallel_size: 1
    context_length: 4096
    vllm:
      image: "vllm/vllm-openai:v0.23.0"

benchmark:
  num_prompts: 256
  random_input_len: 1024
  random_output_len: 128

# Cross-product: 2 GPUs × 2 concurrency settings = 4 variants
matrices:
  cross:
    deploy.gpu_count: 1
    deploy.gpu: ["NVIDIA GeForce RTX 5090", "NVIDIA H100 80GB"]
    zip:
      engine.llm.max_concurrent_requests: [32, 128]
      benchmark.max_concurrency: [32, 128]
```

Browse [serving recipes](recipes/) for existing configurations. The [recipe reference](emmy/recipe/ARCHITECTURE.md)
covers fields, parameter sweeps, LoRA adapters, and custom commands.

## Virtual Machine Management

```bash
# Rent a GPU server and save its lease for cleanup
emmy vm create gpu --gpu "NVIDIA GeForce RTX 5090" --gpu-count 1 --exact-gpu-count \
  --lease /tmp/emmy-vm.json --owner local-run --json

# Delete that server
emmy vm delete lease /tmp/emmy-vm.json --owner local-run
```

See the [VM reference](emmy/commands/ARCHITECTURE.md) for provider-specific commands and availability queries.

## Documentation

- [CLI reference](emmy/commands/ARCHITECTURE.md) — command options, tuning, refitting priors, and image publication
- [Compiler tutorials](docs/docs/tutorials/) — tracing, optimization, and measuring kernels
- [Glossary](GLOSSARY.md) — compiler and serving terminology

## Related Projects

These repositories provide the platform, serving runtimes, and deployment services that Emmy integrates with. The
CloudRift repositories are typically checked out beside Emmy, so use `../<repository-name>` for local code inspection
when that sibling exists, and verify the remote source when current compatibility matters. Some are internal and
require CloudRift organization access.

| Project | Relationship to Emmy |
| --- | --- |
| [CloudRift](https://github.com/cloudrift-ai/cloudrift) | GPU cloud platform and API/protocol implementation behind Emmy's CloudRift provisioning and deployments. |
| [1Cat-vLLM](https://github.com/cloudrift-ai/1Cat-vLLM) | Volta/SM70 vLLM fork and source for `docker/1cat-vllm-sm70`; consult it when stock vLLM lacks target-GPU support. |
| [Rift Provisioning](https://github.com/cloudrift-ai/rift-provisioning) | Ansible automation for CloudRift provider nodes and the CI and agent runner hosts used by Emmy workflows. |
| [Rift Relay](https://github.com/cloudrift-ai/rift-relay) | OpenAI-compatible inference proxy and model-offering service that consumes Emmy recipes and invokes Emmy deployments. |
| [vLLM](https://github.com/vllm-project/vllm) | Primary serving engine upstream and authoritative model-support, release, and image source. |
| [SGLang](https://github.com/sgl-project/sglang) | Alternative serving engine upstream and model-support source. |

## Project Structure

- [AGENTS.md](AGENTS.md) — shared coding-agent instructions; [CLAUDE.md](CLAUDE.md) imports the same guidance for
  Claude Code
- [.agents/skills/](.agents/skills/) — canonical repository skills; [.claude/skills/](.claude/skills/) exposes them
  through compatibility symlinks
- [prompts/](prompts/) — reusable agent prompts shared by repository skills and non-interactive workflows
- [.opencode/](.opencode/) — API-agent provider, permissions, and workflow profiles
- [.github/](.github/) — Pull-request checks, releases, cloud experiments, and model discovery/onboarding workflows
  (see [ARCHITECTURE.md](.github/ARCHITECTURE.md))
- [emmy/](emmy/) — Python package
  - [emmy.py](emmy/emmy.py) — CLI entrypoint
  - [logging_setup.py](emmy/logging_setup.py) — CLI logging configuration
  - [hardware.py](emmy/hardware.py) — GPU specs and instance type mapping
  - [detect.py](emmy/detect.py) — GPU detection via PCI sysfs (local and remote)
  - [redact.py](emmy/redact.py) — Secret redaction for logs and dumps
  - [commands/](emmy/commands/) — CLI layer (thin argparse handlers, see
    [ARCHITECTURE.md](emmy/commands/ARCHITECTURE.md))
    - [deploy/](emmy/commands/deploy/) — `deploy local`, `deploy ssh`, `deploy cloud` commands
    - [bench/](emmy/commands/bench/) — `bench` command
    - [vm/](emmy/commands/vm/) — `vm create/delete/audit` commands (GCP, CloudRift, owned leases)
    - [teardown.py](emmy/commands/teardown.py) — `teardown` command
    - [pull.py](emmy/commands/pull.py) — `pull` command (download HF model)
    - [trace.py](emmy/commands/trace.py) — `trace` command (PyTorch → Graph IR)
    - [compile.py](emmy/commands/compile.py) — `compile` command (decomposition → optimization → fusion → kernel/CUDA
      lowering)
    - [run.py](emmy/commands/run.py) — `run` command (compile + execute on CUDA backend, optional benchmarks)
    - [inspect_graph.py](emmy/commands/inspect_graph.py) — `inspect` command (graph summary)
  - [compiler/](emmy/compiler/) — PyTorch → Graph IR → CUDA compiler (see
    [ARCHITECTURE.md](emmy/compiler/ARCHITECTURE.md))
    - [graph.py](emmy/compiler/graph.py) — `Graph`, `Node`, `Tensor`, `Hints` container
    - [pretty.py](emmy/compiler/pretty.py) — graph dumps as Rust-shaped pseudocode (see
      [IR-PSEUDOCODE-TORCH.md](emmy/compiler/IR-PSEUDOCODE-TORCH.md))
    - [ir/](emmy/compiler/ir/) — per-dialect op definitions (torch / tensor / loop / kernel / cuda) (see
      [ARCHITECTURE.md](emmy/compiler/ir/ARCHITECTURE.md))
      - [schedule/](emmy/compiler/ir/schedule/) — schedule interfaces, reusable choices, and classic scheduling (see
        [ARCHITECTURE.md](emmy/compiler/ir/schedule/ARCHITECTURE.md))
    - [trace/](emmy/compiler/trace/) — PyTorch/HuggingFace → Graph IR capture (see
      [ARCHITECTURE.md](emmy/compiler/trace/ARCHITECTURE.md))
    - [pipeline/](emmy/compiler/pipeline/) — rewrite engine + passes + dump hooks (see
      [ARCHITECTURE.md](emmy/compiler/pipeline/ARCHITECTURE.md))
    - [backend/](emmy/compiler/backend/) — numpy / loop / CUDA execution (see
      [ARCHITECTURE.md](emmy/compiler/backend/ARCHITECTURE.md))
      - [cuda/](emmy/compiler/backend/cuda/) — CUDA backend internals (see
        [ARCHITECTURE.md](emmy/compiler/backend/cuda/ARCHITECTURE.md))
  - [recipe/](emmy/recipe/) — Recipe loading, dataclass types, engine flag mapping (see
    [ARCHITECTURE.md](emmy/recipe/ARCHITECTURE.md))
  - [serving/](emmy/serving/) — vLLM out-of-tree embedding plugin (see [ARCHITECTURE.md](emmy/serving/ARCHITECTURE.md))
  - [deploy/](emmy/deploy/) — Compose generation, deploy orchestration
  - [provisioning/](emmy/provisioning/) — Cloud provisioning, SSH transport, VM lifecycle
  - [benchmark/](emmy/benchmark/) — Benchmark tracking, config, task enumeration, execution
    (see [ARCHITECTURE.md](emmy/benchmark/ARCHITECTURE.md))
  - [planner/](emmy/planner/) — Groups benchmark tasks into execution groups for VM allocation
- [recipes/](recipes/) — The recommended serving configuration and per-GPU model goldens, one directory per model —
  what `emmy deploy` runs (see [ARCHITECTURE.md](recipes/ARCHITECTURE.md); benchmark grids belong in `experiments/`)
- [docker/](docker/) — Custom image builds ([vllm-emmy](docker/vllm-emmy/) — vLLM + the emmy plugin;
  [vllm-emmy-serve](docker/vllm-emmy-serve/) — prebuilt per-model images: warmed cubins + baked model snapshot;
  [1cat-vllm-sm70](docker/1cat-vllm-sm70/) — source-pinned 1Cat-vLLM runtimes and request-time GPU caches for Volta)
- [experiments/](experiments/) — Benchmark parameter sweeps, self-contained recipe + committed results —
  what `emmy bench` runs (see [ARCHITECTURE.md](experiments/ARCHITECTURE.md))
- [kernels/](kernels/) — Standalone CUDA kernel sources
- [crates/emmy-runtime/](crates/emmy-runtime/) — The Rust runtime: the executor behind every launch, the standalone
  pack loader and cached generation (design, hosts, and qualification in
  [ARCHITECTURE.md](crates/emmy-runtime/ARCHITECTURE.md)); [crates/emmy-runtime-py/](crates/emmy-runtime-py/) is
  its in-process host, the `emmy.emmy_runtime` extension setuptools-rust builds into the package
- [crates/emmy-server/](crates/emmy-server/) — Native HTTP adapter: text and image processing
  (see [ARCHITECTURE.md](crates/emmy-server/ARCHITECTURE.md))
- [docs/](docs/) — Docusaurus user-docs site (getting started, benchmarking, custom configurations, deployment)
- [tests/](tests/) — pytest tests (see [ARCHITECTURE.md](tests/ARCHITECTURE.md))
  - [compiler/passes/](tests/compiler/passes/) — compiler pass tests (see
    [ARCHITECTURE.md](tests/compiler/passes/ARCHITECTURE.md))
  - [compiler/realization/](tests/compiler/realization/) — the realization corpus: checked-in reproducers of pinned
    schedules, replayed as data (see [ARCHITECTURE.md](tests/compiler/realization/ARCHITECTURE.md))
  - [perf/](tests/perf/) — GPU perf comparison vs PyTorch (see [ARCHITECTURE.md](tests/perf/ARCHITECTURE.md))
- [scripts/](scripts/) — Analysis and visualization scripts
- [utils/](utils/) — Standalone utility scripts
- [config.yaml](config.yaml) — Benchmark configuration
- [Makefile](Makefile) — Build automation
- [pyproject.toml](pyproject.toml) — Package metadata and tool config

## Contributing

Source setup requires Python 3.12+ and a Rust toolchain:

```bash
git clone https://github.com/cloudrift-ai/emmy.git
cd emmy
make setup
```

Follow [STYLE.md](STYLE.md) and the [contribution instructions](AGENTS.md) for development, testing, and PRs.
Release procedures live in the [workflow documentation](.github/ARCHITECTURE.md).

## License

Licensed under the [Apache License 2.0](LICENSE).
