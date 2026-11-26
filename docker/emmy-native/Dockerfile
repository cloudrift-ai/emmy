# A recipe's native lane as one serving image: Emmy's native server (`emmy-server`), its runtime worker and the pack
# prepared from the recipe golden, so a boot loads the pack and compiles nothing. `bake.sh` stages warm/ and builds it;
# see ARCHITECTURE.md.
#
#   docker run --gpus all -p 8000:8000 cloudriftai/emmy-native-<slug>:<emmy version>-<sha> --model <hf id> \
#     --host 0.0.0.0 --port 8000
#
# The base carries no CUDA toolkit: the pack holds every kernel as a cubin, and the worker loads the driver's libcuda,
# which the NVIDIA container runtime mounts. The binaries are built on the bake host, so the base's glibc must be at
# least the host's (Ubuntu 24.04 here).
ARG BASE_IMAGE=nvidia/cuda:13.0.1-base-ubuntu24.04
FROM ${BASE_IMAGE}

# curl for the compose health check.
RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates && rm -rf /var/lib/apt/lists/*

COPY warm/bin/emmy-server warm/bin/emmy-runtime-worker /usr/local/bin/
# The pack in sub-10 GB parts (`docker/vllm-emmy-serve/split_hf.sh`): Docker Hub refuses larger blobs.
COPY warm/pack_parts/p0 /opt/emmy/pack
COPY warm/pack_parts/p1 /opt/emmy/pack
COPY warm/pack_parts/p2 /opt/emmy/pack
COPY warm/pack_parts/p3 /opt/emmy/pack
COPY warm/pack_parts/p4 /opt/emmy/pack
COPY warm/pack_parts/p5 /opt/emmy/pack
COPY warm/pack_parts/p6 /opt/emmy/pack
COPY warm/pack_parts/p7 /opt/emmy/pack
COPY warm/pack_parts/p8 /opt/emmy/pack
COPY warm/pack_parts/p9 /opt/emmy/pack
COPY warm/pack_parts/p10 /opt/emmy/pack
COPY warm/pack_parts/p11 /opt/emmy/pack
COPY warm/pack_parts/p12 /opt/emmy/pack
COPY warm/pack_parts/p13 /opt/emmy/pack
COPY warm/pack_parts/p14 /opt/emmy/pack
COPY warm/pack_parts/p15 /opt/emmy/pack
COPY warm/pack_parts/p16 /opt/emmy/pack
COPY warm/pack_parts/p17 /opt/emmy/pack
COPY warm/pack_parts/p18 /opt/emmy/pack
COPY warm/pack_parts/p19 /opt/emmy/pack
COPY warm/pack_parts/p20 /opt/emmy/pack
COPY warm/pack_parts/p21 /opt/emmy/pack
COPY warm/pack_parts/p22 /opt/emmy/pack
COPY warm/pack_parts/p23 /opt/emmy/pack

# The labels `emmy publish` checks against the recipe's image reference.
ARG PUBLISH_VERSION
ARG PUBLISH_REVISION
ARG MODEL
ARG TARGET_GPU
LABEL ai.emmy.publish.family="emmy-native" \
      org.opencontainers.image.version="${PUBLISH_VERSION}" \
      org.opencontainers.image.revision="${PUBLISH_REVISION}" \
      ai.emmy.model.id="${MODEL}" \
      ai.emmy.target.gpu="${TARGET_GPU}"

ENTRYPOINT ["emmy-server", "--artifact", "/opt/emmy/pack"]
