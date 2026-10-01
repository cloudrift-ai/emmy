#!/bin/sh
# Measured BF16 mixed lane on one RTX 5090.
set -eu

recipe_root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd -P)
golden="$recipe_root/recipes/Qwen3.8-27B-NVFP4/golden/rtx5090_sm120.json"

# An inherited hand pin would override the measured schedule while this command
# still appears to use the golden. Keep experiments in the separate knobs helper.
inherited_pins=$(env | grep -E '^EMMY_(KNOBS|MLP_STATIC_KNOBS|MLP_PREFILL_KNOBS|PLACE|WORK|TILE|STAGE|REDUCE|RASTER|FAST_MATH)(@[^=]+)?=|^EMMY_TUNE_DB=' || true)
if [ -n "$inherited_pins" ]; then
    echo 'Remove inherited EMMY knob pins and EMMY_TUNE_DB before golden serving.' >&2
    exit 2
fi

# Strict evidence excludes prior guesses, while an old tune DB may still hold
# measured rows from a different experiment. Import only this golden per boot.
EMMY_TUNE_DB="$(mktemp -d /tmp/qwen38-golden-XXXXXX)/tune.db"
export EMMY_TUNE_DB

exec emmy serve Inferact/Qwen3.8-27B-NVFP4@6128240ebaf4eaa7bad2b3d1c72c37d677c5f462 \
    --runner generate --compile-scope mlp --dtype bfloat16 \
    --max-model-len 4096 --max-num-seqs 1 --max-num-batched-tokens 64 \
    --language-model-only --gpu-memory-utilization 0.97 \
    --golden "$golden" --strict-evidence "$@"
