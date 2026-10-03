#!/bin/bash
# Verify a baked per-model serving image on the target GPU: cold-start fully offline (HF_HUB_OFFLINE is
# baked in; no HF_TOKEN passed) — proving zero downloads — then issue one completion and
# assert the cubin set did not grow: an empty diff = 100% cache hit (zero nvcc compiles).
#
#   MODEL=google/gemma-4-12B-it IMAGE=cloudriftai/vllm-emmy-<slug>:TAG [GPU_DEVICE=1] ./verify.sh
set -euo pipefail
# Resolve the pinned config from MODEL via the naming schema — models/<slug>.env is the
# single source the bake reads too, so warm and release cannot drift apart.
cd "$(dirname "$0")"
MODEL="${MODEL:?set MODEL to the HF model id being released}"
SLUG=$(./model_slug.sh "$MODEL")
CONFIG="models/$SLUG.env"
[ -f "$CONFIG" ] || { echo "no pinned config for $MODEL (expected $CONFIG)" >&2; exit 1; }
set -a  # export the SERVE_* config so the -e pass-throughs below carry values
source "$CONFIG"
set +a

: "${IMAGE:?set IMAGE to the baked serving image to verify}"

# Every cache-key input is card-specific — the live-probed featurization, the golden picks,
# the memory headroom the config was swept for. A warm on the wrong GPU bakes a cache the
# released image can never hit, and the failure only surfaces ~30 min later in verify (or
# worse, never, on a card whose picks happen to coincide). Check it here, cheaply.
if [ -n "${SERVE_GPU:-}" ] && [ -z "${SKIP_GPU_CHECK:-}" ]; then
    # The name the compiler keys on, from the image's own emmy: nvidia-smi spells some cards
    # differently from the registry (a V100 reports "Tesla V100-SXM3-32GB", registered as
    # "NVIDIA Tesla V100 SXM3 32GB"). `|| true`: under `set -euo pipefail` a host with no GPU
    # would abort the script here with no message at all.
    live=$(docker run --rm --gpus "device=${GPU_DEVICE:-0}" --entrypoint python3 "$IMAGE" \
        -c 'from emmy.gpu import live_name; print(live_name() or "")' 2>/dev/null || true)
    if [ -n "$live" ] && [ "$live" != "$SERVE_GPU" ]; then
        echo "GPU mismatch: config pins '$SERVE_GPU', device ${GPU_DEVICE:-0} is '$live'." >&2
        echo "  Warm and verify must run on the card the config was swept for." >&2
        echo "  Override with SKIP_GPU_CHECK=1 only if you know the picks transfer." >&2
        exit 1
    fi
fi

# The pinned checkpoint revision is baked into the image ENV, so every boot below already
# serves the right rung — but only if this tag was built from THIS config. A tag built
# before the pin landed (or from another rung) serves different weights and still passes
# every check in this script: the cubin set is closed, the pack hits, the completion reads
# fine. Compare the two, once, off the image metadata.
baked_env=$(docker inspect --format '{{range .Config.Env}}{{println .}}{{end}}' "$IMAGE")
baked_value() { printf '%s\n' "$baked_env" | sed -n "s/^$1=//p" | head -1; }
check_baked() {
    local key="$1" expected="$2" actual
    actual=$(baked_value "$key")
    if [ "${actual:-}" != "$expected" ]; then
        echo "[verify] FAIL — $IMAGE bakes $key='${actual:-<default>}', the config pins '${expected:-<default>}'." >&2
        echo "  The image was built from a different config; re-bake with 'make serve-image MODEL=$MODEL'." >&2
        exit 1
    fi
}
check_baked SERVE_REVISION "${SERVE_REVISION:-}"
check_baked EMMY_GEN_EMBED_HOST "${SERVE_EMBED_HOST:-}"
check_baked EMMY_GEN_PREFILL_CAPACITY "${SERVE_PREFILL_CAPACITY:-}"
check_baked EMMY_GEN_PREFILL_BUCKET "${SERVE_PREFILL_BUCKET:-}"
check_baked EMMY_GEN_M1_TIER "${SERVE_M1_TIER:-}"
check_baked SERVE_V2_MODEL_RUNNER "${SERVE_V2_MODEL_RUNNER:-}"
check_baked SERVE_ENV "${SERVE_ENV:-}"
check_baked SERVE_CAPTURE_SIZES "${SERVE_CAPTURE_SIZES:-}"
check_baked SERVE_EXTRA_ARGS "${SERVE_EXTRA_ARGS:-}"

# A parallel boot runs one worker per card and each loads the pack of its pipeline stage, so every
# worker must log a hit: one that fell back recompiles its layers on every customer boot while the
# others' hits pass a single-line grep.
flag_value() {  # $1 = flag -> its value in SERVE_EXTRA_ARGS, or $2 when absent
    # shellcheck disable=SC2086 — SERVE_EXTRA_ARGS is a deliberately word-split flag list
    printf '%s\n' ${SERVE_EXTRA_ARGS:-} | awk -v flag="$1" -v none="$2" 'take { print; found = 1; exit } $0 == flag { take = 1 } END { if (!found) print none }'
}
WORKERS=$(( $(flag_value --tensor-parallel-size 1) * $(flag_value --pipeline-parallel-size 1) ))
pack_hits() { docker logs "$NAME" 2>&1 | grep -c "pack hit" || true; }

PORT="${PORT:-8000}"
GPUS="all"; [ -n "${GPU_DEVICE:-}" ] && GPUS="device=$GPU_DEVICE"
NAME="emmy-verify-$SLUG"

docker rm -f "$NAME" >/dev/null 2>&1 || true
# Cleanup rides an EXIT trap: under `set -e` an inline `docker rm` after a failing
# diagnostic pipeline never runs and leaks the container (seen 2026-07-23).
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 "$IMAGE"

before=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")
# A Triton compile at boot or request time writes a new entry into the baked cache, which is the
# signal checked below. vLLM's JIT monitor also warns on a kernel's first launch in a process after
# warmup, even when its binary loads straight from that cache: a runtime whose warmup does not launch
# its attention kernels (the 1Cat fork) warns on the first request with nothing compiled.
triton_files() { docker exec "$NAME" sh -c "find /opt/emmy/triton -type f | sort"; }
triton_before=$(triton_files)

# With a baked pack the boot skips the compiler frontend entirely and health arrives in
# ~weight-load time; without one (pack write skipped at warm) the per-layer CPU
# trace/lower/render runs uncached — the 48-layer gemma-4 boot takes ~25 min. Budget 40.
echo "[verify] waiting for /health (no downloads, no compiles; fast if the pack baked)..."
for _ in $(seq 1 240); do
    if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
    if [ -z "$(docker ps -q -f name=$NAME)" ]; then
        echo "[verify] server died:"; docker logs --tail 50 "$NAME"; exit 1
    fi
    sleep 10
done
curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[verify] timed out"; docker logs --tail 50 "$NAME"; exit 1; }

# Under HF_HUB_OFFLINE vLLM serves the model under the RESOLVED snapshot path, not the
# repo id — ask the server for its served name rather than assuming $SERVE_MODEL.
served=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
    -d "{\"model\": \"$served\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
    | head -c 400; echo
triton_after=$(triton_files)
if [ "$triton_before" != "$triton_after" ]; then
    echo "[verify] FAIL — Triton compiled at boot or request time despite the baked warm cache (new cache entries):" >&2
    diff <(echo "$triton_before") <(echo "$triton_after") | tail -10 >&2 || true
    exit 1
fi

after=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")
# When the image ships a pack, the boot must have actually used it — a silent fallback to
# the full compile (key/environment drift) still passes the cubin check but re-pays the
# ~25 min frontend on every customer boot, which is exactly what the pack exists to kill.
# The "pack hit" line is emmy's runner log; emmy.serving.register() attaches a log
# handler under the bare vLLM entrypoint precisely so it reaches docker logs here.
pack_baked=$(docker exec "$NAME" sh -c "find /opt/emmy/pack -name manifest.json 2>/dev/null | head -1")
# grep without -q: under pipefail, -q's early exit can SIGPIPE docker logs on a hit.
if [ -n "$pack_baked" ] && [ "$(pack_hits)" -lt "$WORKERS" ]; then
    echo "[verify] FAIL — a pack is baked but $(pack_hits) of $WORKERS worker(s) hit it (the rest fell back to full compile):"
    docker logs "$NAME" 2>&1 | grep -i "\[pack\]" | tail -5 || true
    exit 1
fi

if [ "$before" != "$after" ]; then
    echo "[verify] FAIL — new cubins compiled at runtime (cache miss):"
    diff <(echo "$before") <(echo "$after") || true
    exit 1
fi
echo "[verify] PASS — served offline with zero new cubins or Triton compiles ($(echo "$before" | wc -l) prebuilt)${pack_baked:+, pack hit on all $WORKERS worker(s)}"

# ---- the extra shapes ------------------------------------------------------------------
# Each shape warm.sh baked a pack for must ALSO hit it, for the same reason the pinned shape
# must: a shape whose pack silently misses re-pays the whole compiler frontend on every boot
# of that lane, and nothing else in the pipeline would notice. Boot the image once per extra
# shape with that shape's knobs and assert the hit. A miss here is a FAIL — the pack was baked,
# so a miss means a key mismatch between warm and serve, not an un-warmed lane.
for spec in ${SERVE_WARM_SHAPES:-}; do
    IFS=':' read -r b p m lane <<<"$spec"
    args=(-e "EMMY_GEN_DECODE_BUCKET=${b:-$SERVE_DECODE_BUCKET}")
    [ -n "$p" ] && args+=(-e "EMMY_GEN_PREFILL_BUCKET=$p")
    args+=(-e "SERVE_MAX_NUM_BATCHED_TOKENS=${m:-$SERVE_MAX_NUM_BATCHED_TOKENS}")
    [ "${lane:-}" = "fm" ] && args+=(-e "EMMY_FAST_MATH=1")
    echo "[verify] ---- extra shape $spec ----"
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    docker run -d --name "$NAME" --gpus "$GPUS" --ipc=host -p "$PORT":8000 "${args[@]}" "$IMAGE" >/dev/null
    shape_triton_before=$(triton_files)
    for _ in $(seq 1 240); do
        if curl -sf "http://localhost:$PORT/health" >/dev/null 2>&1; then break; fi
        if [ -z "$(docker ps -q -f name=$NAME)" ]; then
            echo "[verify] FAIL — server died at shape $spec:"; docker logs --tail 50 "$NAME"; exit 1
        fi
        sleep 10
    done
    curl -sf "http://localhost:$PORT/health" >/dev/null || { echo "[verify] FAIL — shape $spec timed out"; docker logs --tail 50 "$NAME"; exit 1; }
    shape_served=$(curl -sf "http://localhost:$PORT/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
    curl -sf "http://localhost:$PORT/v1/completions" -H 'Content-Type: application/json' \
        -d "{\"model\": \"$shape_served\", \"prompt\": \"The capital of France is\", \"max_tokens\": 20, \"temperature\": 0}" \
        >/dev/null
    if [ "$shape_triton_before" != "$(triton_files)" ]; then
        echo "[verify] FAIL — shape $spec compiled Triton kernels despite the baked warm cache:" >&2
        diff <(echo "$shape_triton_before") <(triton_files) | tail -10 >&2 || true
        exit 1
    fi
    shape_after=$(docker exec "$NAME" sh -c "find /opt/emmy/cubin -name '*.cubin' | sort")
    if [ "$(pack_hits)" -lt "$WORKERS" ]; then
        echo "[verify] FAIL — shape $spec baked a pack but $(pack_hits) of $WORKERS worker(s) hit it:"
        docker logs "$NAME" 2>&1 | grep -i "\[pack\]" | tail -5 || true
        exit 1
    fi
    [ "$shape_after" = "$after" ] || { echo "[verify] FAIL — shape $spec compiled new cubins at runtime:"; diff <(echo "$after") <(echo "$shape_after") || true; exit 1; }
    echo "[verify] PASS — shape $spec: pack-hit boot, zero new cubins or Triton compiles"
done
