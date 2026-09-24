#!/bin/sh
# The frozen generative serve entrypoint — the arg set `emmy serve --generate` builds,
# parameterized by the SERVE_* env (baked from models/<slug>.env). The warm run
# mounts this script into the plain vllm-emmy image and the baked image ships it, so
# the warmed and released invocations are literally the same — cache-key parity.
#
# The compilation-config mirrors emmy/commands/serve.py's generate path: FULL_DECODE_ONLY
# whole-step decode cudagraphs (capture sizes = the power-of-two ladder to max-num-seqs,
# with the decode bucket riding the list) and the fused rotary_embedding CustomOp (redundant
# since the plugin dispatches its RoPEs itself; kept because this rendered invocation is a
# cache-key input of every released image). --no-enable-prefix-caching matches the
# benchmark protocol (every request does full prefill work). Keep in sync with
# _gen_graph_args / build_serve_cmd in emmy/commands/serve.py.
#
# Checkpoint-specific values come from the same
# models/<slug>.env. Each default reproduces the invocation this script emitted before they
# existed, so a config that sets none of them renders byte-identically:
#
#   SERVE_REVISION       the checkpoint commit to serve. Empty = the repo's DEFAULT branch,
#                        which on a multi-rung repo (one branch per bit rate) is a different
#                        model than the one that was tuned and swept — warm.sh refuses an
#                        unpinned revision there. Pin by commit sha, never a branch name: a
#                        branch can be re-cut under the same name, and the offline boot
#                        resolves snapshots/<sha> directly.
#   SERVE_QUANT          the checkpoint's quantization method when vLLM has none for it
#                        (`exl3` today; see _is_exl3_model). vLLM refuses the boot at config
#                        parsing ("Unknown quantization method: exl3") though nothing in the
#                        engine needs the method: emmy owns every coded weight and the one
#                        vLLM-owned parameter (lm_head) is handled at load. So the override
#                        tells vLLM the model is unquantized, which for vLLM's purposes it is.
#   SERVE_CAPTURE_SIZES  the cudagraph capture ladder, as a JSON list. An MoE model must cap
#                        it at [1] (see _is_moe_model): single-token steps ride the runner's
#                        fixed-slot expert dispatch (fixed launch set, capture-legal) while
#                        wider decode steps keep the routed dispatch, which host-syncs and
#                        stays eager.
#   SERVE_EXTRA_ARGS     further pinned vLLM flags, word-split (e.g. `--kv-cache-dtype
#                        fp8_e4m3`). They belong in the pinned config because a flag that
#                        moves which programs the plugin builds is a cache-key input.
#   SERVE_EMBED_HOST / SERVE_PREFILL_CAPACITY / SERVE_PREFILL_BUCKET / SERVE_M1_TIER
#                        the memory/shape lane exported as EMMY_GEN_* by warm and baked into
#                        the image. They are compiler/pack-key inputs even though they do not
#                        add vLLM argv. A warm-shape prefill field overrides the pinned bucket.
#   SERVE_V2_MODEL_RUNNER
#                        opt into vLLM's V2 runner after model-specific serving qualification.

# Docker must declare the optional build ENV keys so verify can compare the baked image to
# its source config. Preserve the historical opt-out semantics at runtime: an empty build arg
# is UNSET, not an explicit empty value inherited by the server process.
[ -n "${EMMY_GEN_EMBED_HOST:-}" ] || unset EMMY_GEN_EMBED_HOST
[ -n "${EMMY_GEN_PREFILL_CAPACITY:-}" ] || unset EMMY_GEN_PREFILL_CAPACITY
[ -n "${EMMY_GEN_PREFILL_BUCKET:-}" ] || unset EMMY_GEN_PREFILL_BUCKET
[ -n "${EMMY_GEN_M1_TIER:-}" ] || unset EMMY_GEN_M1_TIER
# The standard lane is this entrypoint's default, whatever the compiler's is. #868 made fast math
# the compiler default, which would silently collapse the two serving lanes into one: every warm
# shape without an `:fm` suffix would bake the SAME f16-accumulate cubins its `:fm` twin does, no
# standard-lane pack would exist, and a deployment that asks for the standard lane — the golden's
# own default, and what the experiment recipes run — would miss the pack and pay the full compiler
# frontend on every boot. A caller wanting the other lane still says so, which is what the `:fm`
# shapes and the recipes' EMMY_FAST_MATH=1 do.
export EMMY_FAST_MATH="${EMMY_FAST_MATH:-0}"
if [ -n "${SERVE_V2_MODEL_RUNNER:-}" ]; then
    export VLLM_USE_V2_MODEL_RUNNER="$SERVE_V2_MODEL_RUNNER"
else
    unset VLLM_USE_V2_MODEL_RUNNER
fi

# The architectures override is what routes the model to the plugin; the quantization arm
# rides beside it. One spelling, shared with emmy/commands/serve.py's json.dumps output.
OVERRIDES='{"architectures": ["EmmyGenModel"]}'
if [ "${SERVE_QUANT:-}" = "exl3" ]; then
    OVERRIDES='{"architectures": ["EmmyGenModel"], "quantization_config": null}'
fi

SIZES="${SERVE_CAPTURE_SIZES:-[1, 2, 4, 8, 16, 32, 64, 128, 256]}"
COMPILE_CFG='{"cudagraph_mode": "FULL_DECODE_ONLY", "cudagraph_capture_sizes": '"$SIZES"', "custom_ops": ["+rotary_embedding"]}'

REVISION=""
if [ -n "${SERVE_REVISION:-}" ]; then
    REVISION="--revision ${SERVE_REVISION}"
fi

# shellcheck disable=SC2086 — $REVISION and $SERVE_EXTRA_ARGS are deliberately word-split
# flag lists, and both expand to nothing at all when unset.
exec python3 -m vllm.entrypoints.openai.api_server \
    --model "${SERVE_MODEL}" \
    $REVISION \
    --runner generate \
    --dtype float16 \
    --max-model-len "${SERVE_MAX_MODEL_LEN}" \
    --max-num-batched-tokens "${SERVE_MAX_NUM_BATCHED_TOKENS}" \
    --gpu-memory-utilization "${SERVE_GPU_MEM_UTIL}" \
    --no-enable-prefix-caching \
    --hf-overrides "${OVERRIDES}" \
    --compilation-config "${COMPILE_CFG}" \
    ${SERVE_EXTRA_ARGS:-} \
    "$@"
