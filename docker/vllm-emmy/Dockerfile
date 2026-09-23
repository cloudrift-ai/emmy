# vLLM serving image with the emmy embedding plugin baked in.
#
# Build from the repo root (the wheel must exist first — `make wheel`):
#   make vllm-emmy-image          # builds cloudriftai/vllm-emmy:<vllm-ver>-<git-sha>
#   make vllm-emmy-push
#
# Serve (same entrypoint as the stock image, plus the plugin flags):
#   docker run --gpus all -p 8000:8000 cloudriftai/vllm-emmy:TAG \
#     --model Qwen/Qwen3-Embedding-0.6B --runner pooling --enforce-eager \
#     --max-model-len 4096 --hf-overrides '{"architectures":["EmmyEmbedModel"]}'
ARG VLLM_VERSION=v0.23.0
ARG BASE_IMAGE=vllm/vllm-openai:${VLLM_VERSION}

# The runtime extension is built against the image's own Python and libc, so the wheel
# needs no manylinux story: a Rust toolchain in a throwaway stage, maturin, one build.
FROM ${BASE_IMAGE} AS runtime-builder
RUN curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain stable
ENV PATH=/root/.cargo/bin:$PATH
COPY Cargo.toml Cargo.lock /src/
COPY crates /src/crates
RUN pip install --no-cache-dir "maturin>=1.5,<2" \
    && maturin build --release --locked -m /src/crates/emmy-runtime-py/Cargo.toml -o /wheels

FROM ${BASE_IMAGE}

# The emmy wheel installs --no-deps so the image's pinned torch /
# transformers / numpy (vLLM's own dependency set) are never touched; the
# runtime deps the serving path actually uses are added explicitly. cppyy is
# deliberately absent — the CUDA serving path compiles via nvcc only.
# (vllm-openai:v0.23.0 ships CUDA 13.0 + nvcc; transformers 5.10.2 satisfies
# the compile pin.) frozendict is imported by the tensor IR itself, so every
# serving boot needs it; only the stock image happened to carry it transitively,
# and a fork base (1Cat-vLLM on Volta) fails at model construction without it.
COPY dist/emmy_ml-*.whl /tmp/wheels/
COPY --from=runtime-builder /wheels/emmy_runtime-*.whl /tmp/wheels/
RUN pip install --no-cache-dir --no-deps /tmp/wheels/emmy_ml-*.whl /tmp/wheels/emmy_runtime-*.whl \
    && pip install --no-cache-dir catboost pyyaml httpx "huggingface_hub[hf_transfer]" safetensors frozendict \
    && rm -rf /tmp/wheels

# Compiled-cubin cache under the HF cache mount (compose mounts the model dir
# there), so container restarts on the same VM skip recompiles. Recipes can
# override via extra_env.
ENV EMMY_CUBIN_CACHE=/root/.cache/huggingface/emmy-cubins
