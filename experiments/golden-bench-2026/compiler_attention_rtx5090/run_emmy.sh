#!/usr/bin/env bash
# Replay recorded RTX 5090 goldens and measure the same source with current PyTorch.
set -euo pipefail

if [ "$#" -ne 5 ]; then
  echo "usage: $0 EMMY OPERATOR BATCH RESULTS_DIR GOLDEN_DIR" >&2
  exit 2
fi

emmy=$1
operator=$2
batch=$3
results=$4
golden_dir=$5
here=$(cd "$(dirname "$0")" && pwd)
source "$here/operators.sh"

mkdir -p "$results/json" "$results/dumps" "$results/logs"
status_file=$results/setup-status.tsv
printf "operator\tbatch\tsequence_length\treplay\treference\n" > "$status_file"
successful_setups=0
missing_goldens=0
mapfile -t sequence_lengths < <(operator_sequence_lengths "$operator" "$batch")

for sequence_length in "${sequence_lengths[@]}"; do
  setup="${operator}-b${batch}-s${sequence_length}"
  golden=$golden_dir/$setup.golden.yaml
  if [ ! -f "$golden" ]; then
    printf "%s\t%s\t%s\tmissing-golden\tskipped\n" "$operator" "$batch" "$sequence_length" >> "$status_file"
    missing_goldens=$((missing_goldens + 1))
    continue
  fi
  source_code=$(operator_code "$operator" "$batch" "$sequence_length")

  if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 1200s \
    "$emmy" run --golden "$golden" --bench --bench-backends emmy \
    --warmup 1 --iters 10 --no-record-nodes \
    --json "$results/json/$setup.replay" --dump-dir "$results/dumps/$setup.replay" \
    2>&1 | tee "$results/logs/$setup.replay.log"; then
    replay_status=ok
  else
    replay_status=failed:$?
  fi
  if EMMY_NVCC_FLAGS= timeout --signal=TERM --kill-after=30s 1200s \
    "$emmy" run -c "$source_code" --bench --strict --bench-backends eager,tcompile,emmy \
    --warmup 1 --iters 10 --no-record-nodes --json "$results/json/$setup.reference.json" \
    2>&1 | tee "$results/logs/$setup.reference.log"; then
    reference_status=ok
  else
    reference_status=failed:$?
  fi
  if [ "$replay_status" = ok ] && [ "$reference_status" = ok ]; then
    successful_setups=$((successful_setups + 1))
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" \
    "$operator" "$batch" "$sequence_length" "$replay_status" "$reference_status" >> "$status_file"
done

# A missing or partial setup is not performance evidence. Preserve every status, then fail the row.
test "$missing_goldens" -eq 0
test "$successful_setups" -eq "${#sequence_lengths[@]}"
