<p align="center">
  <a href="https://pypi.org/project/emmy-ml/"><img src="https://img.shields.io/pypi/v/emmy-ml" alt="PyPI"></a>
  <a href="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml"><img src="https://github.com/cloudrift-ai/emmy/actions/workflows/tests.yml/badge.svg" alt="Tests"></a>
  <a href="https://discord.gg/cloudrift"><img src="https://img.shields.io/discord/1150997934113030174?label=Discord" alt="Discord"></a>
</p>

**Compile → Benchmark → Deploy.** Emmy compiles PyTorch models into CUDA kernels, benchmarks LLM inference, and
deploys models on local or cloud GPUs with vLLM and SGLang.

[Compile](#compile) · [Deploy](#deploy) · [Benchmark](#benchmark) · [Serve](#serve) · [Recipes](#recipes)

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

# Try 20 kernel schedules with cold-cache measurements
emmy run --bench --cold-cache --tune 20 -c "nn.Linear(4096,4096,bias=False)(torch.randn(1,4096))"

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

<details>
<summary>Architecture index for contributors</summary>

Related repositories may be checked out at `../<repository-name>`.

| Area | Design docs |
| --- | --- |
| Compiler | [Overview](emmy/compiler/ARCHITECTURE.md), [tracing](emmy/compiler/trace/ARCHITECTURE.md), [IR](emmy/compiler/ir/ARCHITECTURE.md), [scheduling](emmy/compiler/ir/schedule/ARCHITECTURE.md), [Tile IR](emmy/compiler/ir/tile/ARCHITECTURE.md) |
| Optimization | [Pipeline and evidence](emmy/compiler/pipeline/ARCHITECTURE.md), [passes](emmy/compiler/pipeline/passes/ARCHITECTURE.md), [kernel lowering](emmy/compiler/pipeline/passes/lowering/kernel/ARCHITECTURE.md) |
| Execution | [Backends](emmy/compiler/backend/ARCHITECTURE.md), [CUDA](emmy/compiler/backend/cuda/ARCHITECTURE.md), [Rust runtime](crates/emmy-runtime/ARCHITECTURE.md) |
| Serving | [vLLM integration](emmy/serving/ARCHITECTURE.md), [native generation](emmy/serving/native/ARCHITECTURE.md), [native server](crates/emmy-server/ARCHITECTURE.md), [prebuilt images](docker/vllm-emmy-serve/ARCHITECTURE.md) |
| Deployment and benchmarks | [Commands](emmy/commands/ARCHITECTURE.md), [provisioning](emmy/provisioning/ARCHITECTURE.md), [benchmark runner](emmy/benchmark/ARCHITECTURE.md), [experiments](experiments/ARCHITECTURE.md) |
| Recipes | [Format and matrices](emmy/recipe/ARCHITECTURE.md), [serving configurations](recipes/ARCHITECTURE.md) |
| Tests | [Overview](tests/ARCHITECTURE.md), [passes](tests/compiler/passes/ARCHITECTURE.md), [realization corpus](tests/compiler/realization/ARCHITECTURE.md), [performance](tests/perf/ARCHITECTURE.md) |
| Automation | [Checks and releases](.github/ARCHITECTURE.md), [agent skills](.agents/skills/) |

</details>

## Related Projects

| Project | Relationship to Emmy |
| --- | --- |
| [CloudRift](https://github.com/cloudrift-ai/cloudrift) | GPU cloud platform and provisioning API. |
| [1Cat-vLLM](https://github.com/cloudrift-ai/1Cat-vLLM) | vLLM fork for Volta GPUs. |
| [Rift Provisioning](https://github.com/cloudrift-ai/rift-provisioning) | CloudRift node and runner setup. |
| [Rift Relay](https://github.com/cloudrift-ai/rift-relay) | Inference proxy that deploys Emmy recipes. |
| [vLLM](https://github.com/vllm-project/vllm) | Primary serving engine. |
| [SGLang](https://github.com/sgl-project/sglang) | Alternative serving engine. |

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
