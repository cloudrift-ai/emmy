#!/bin/bash
# Fill a golden's missing measurements on its own card:  fill.sh GOLDEN TASK_DIR BUDGET_S
#
# Works on a copy, TASK_DIR/working.json, and records in two phases, each row a strict record whose result the next
# row's compile reads as evidence:
#   1. every row with a schedule, as an isolated record of its own kernel (`run --kernel K --realization ROW`): a cut
#      piece under its row's knobs, a whole target under `PLACE=fuse` and its knobs. A strict whole-program record of
#      a target whose rows were all demoted refuses at its first fork, since nothing prices it yet; the isolated form
#      needs only the row's own kernel.
#   2. every whole target left: a seed with no schedule of its own, and a target with no torch.compile time, as a
#      `run --record` of the whole program, which now finds the rows phase 1 recorded.
# A row that fails is listed in failed.txt with its phase and stays missing; no new row starts once BUDGET_S is spent.
set -uo pipefail
golden=$1 task_dir=$2 budget_s=$3
emmy=${EMMY:-./venv/bin/emmy}
cp "$golden" "$task_dir/working.json"
: > "$task_dir/failed.txt"
"$emmy" golden list "$task_dir/working.json" --missing --json "$task_dir/missing.json" || exit 1
deadline=$((SECONDS + budget_s))

record() {  # $1 = one missing.json entry, $2 = its phase: piece, target or whole
  local entry=$1 phase=$2 row kernel knobs lane
  row=$(jq -r .row <<< "$entry")
  kernel=$(jq -r .kernel <<< "$entry")
  knobs=$(jq -r '.knobs // {} | to_entries | map("\(.key)=\(.value)") | join(",")' <<< "$entry")
  lane=$(jq -r 'if .pins.FAST_MATH == null then "" elif .pins.FAST_MATH then "1" else "0" end' <<< "$entry")
  local env_=()
  [ -n "$lane" ] && env_+=(EMMY_FAST_MATH="$lane")
  local cmd
  case $phase in
    piece) cmd=(env "${env_[@]}" EMMY_KNOBS="$knobs" "$emmy" run --kernel "$kernel" --record-greedy --strict) ;;
    target) cmd=(env "${env_[@]}" EMMY_KNOBS="PLACE=fuse,$knobs" "$emmy" run --kernel "$kernel" --record-greedy --strict) ;;
    whole) cmd=(env "${env_[@]}" "$emmy" run --record) ;;
  esac
  if ! timeout 1800 "${cmd[@]}" --golden "$task_dir/working.json" --realization "$row" --bench </dev/null \
    >> "$task_dir/fill.log" 2>&1; then
    echo "$phase $row" >> "$task_dir/failed.txt"
  fi
}

# Phase 1: the rows with a schedule. Phase 2: the whole targets, with what budget is left.
for phase in 1 2; do
  if [ $phase = 1 ]; then
    select='.missing == "emmy" and ((.knobs // {}) | length) > 0'
  else
    select='.missing == "tcompile" or (.missing == "emmy" and ((.knobs // {}) | length) == 0)'
  fi
  while IFS= read -r entry; do
    if [ "$SECONDS" -ge "$deadline" ]; then
      echo "budget spent in phase $phase; the remaining rows stay missing"
      break
    fi
    if [ $phase = 2 ]; then
      record "$entry" whole
    elif [ "$(jq -r .piece <<< "$entry")" = true ]; then
      record "$entry" piece
    else
      record "$entry" target
    fi
  done < <(jq -c ".[] | select($select)" "$task_dir/missing.json")
done
"$emmy" golden list "$task_dir/working.json" --missing --json "$task_dir/still_missing.json"
