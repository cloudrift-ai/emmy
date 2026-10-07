#!/usr/bin/env bash
# Git operations the emmy-onboarding-bot performs from workflows. The rolling model-discovery pull
# request is the one branch every agent workflow appends to, and the nightly refresh commits straight
# to main. Locating the rolling branch, rebasing it and pushing to main are identical everywhere, and
# the force-with-lease and moved-main guards are easy to get subtly wrong, so they live here rather
# than once per workflow.
set -euo pipefail

retry_gh() {
  for attempt in 1 2 3; do
    "$@" && return 0
    if [ "$attempt" -eq 3 ]; then
      return 1
    fi
    sleep "$attempt"
  done
}

# Commits as the bot and lets gh answer git's credential prompts with $GH_TOKEN.
bot_git_identity() {
  git config user.name "emmy-onboarding-bot"
  git config user.email "emmy-onboarding-bot[bot]@users.noreply.github.com"
  gh auth setup-git
}

# Writes exists/number/branch to $GITHUB_OUTPUT. An empty branch means no rolling work exists yet. The rolling PR is
# the one open PR on a branch starting with PREFIX or carrying one of the LABELS (default: the model-discovery PR).
find_rolling_pr() {
  local prefix=${1:-agents/model-discovery-} matches count number branch orphan_output labels_json filter
  local -a orphan_branches labels=("${@:2}")
  [ "${#labels[@]}" -gt 0 ] || labels=(model-discovery model-onboarding)
  labels_json=$(printf '%s\n' "${labels[@]}" | jq -R . | jq -sc .)
  filter="[.[] | select((.isCrossRepository | not) and (([.labels[].name] as \$have | $labels_json | any(. as \$label | \$have | index(\$label))) or (.headRefName | startswith(\"$prefix\"))))]"
  matches=$(retry_gh gh pr list --state open --limit 100 --json number,headRefName,isCrossRepository,labels --jq "$filter")
  count=$(jq length <<< "$matches")
  if [ "$count" -gt 1 ]; then
    echo "Expected at most one rolling PR on $prefix*, found $count" >&2
    return 1
  fi

  number=$(jq -r '.[0].number // empty' <<< "$matches")
  branch=$(jq -r '.[0].headRefName // empty' <<< "$matches")
  if [ -n "$number" ]; then
    echo "Updating rolling PR #$number from $branch"
    echo "exists=true" >> "$GITHUB_OUTPUT"
  else
    orphan_output=$(retry_gh gh api --paginate \
      "repos/$GITHUB_REPOSITORY/git/matching-refs/heads/$prefix" \
      --jq '.[].ref | sub("^refs/heads/"; "")')
    mapfile -t orphan_branches < <(printf '%s' "$orphan_output" | sort)
    if [ "${#orphan_branches[@]}" -gt 1 ]; then
      echo "Expected at most one unpaired $prefix* branch, found ${#orphan_branches[@]}" >&2
      return 1
    fi
    branch="${orphan_branches[0]:-}"
    if [ -n "$branch" ]; then
      echo "Adopting unpaired rolling branch $branch"
    fi
    echo "exists=false" >> "$GITHUB_OUTPUT"
  fi
  echo "number=$number" >> "$GITHUB_OUTPUT"
  echo "branch=$branch" >> "$GITHUB_OUTPUT"
}

# Rebases the checked-out rolling branch onto $BASE_BRANCH and pushes it back.
# Refuses when the branch moved after the caller selected it: another agent workflow is mid-run.
rebase_rolling_branch() {
  local original_head remote_head rebased_head attempt
  bot_git_identity
  git fetch origin \
    "+refs/heads/$BASE_BRANCH:refs/remotes/origin/$BASE_BRANCH" \
    "+refs/heads/$EXISTING_BRANCH:refs/remotes/origin/$EXISTING_BRANCH"

  original_head="$(git rev-parse HEAD)"
  remote_head="$(git rev-parse "origin/$EXISTING_BRANCH")"
  if [ "$original_head" != "$remote_head" ]; then
    echo "Discovery branch changed after it was selected; refusing to overwrite $remote_head" >&2
    return 1
  fi

  git rebase "origin/$BASE_BRANCH"
  rebased_head="$(git rev-parse HEAD)"
  if [ "$rebased_head" = "$original_head" ]; then
    return 0
  fi

  for attempt in 1 2 3; do
    if git push \
      --force-with-lease="refs/heads/$EXISTING_BRANCH:$original_head" \
      origin "HEAD:refs/heads/$EXISTING_BRANCH"; then
      return 0
    fi
    git fetch origin \
      "+refs/heads/$EXISTING_BRANCH:refs/remotes/origin/$EXISTING_BRANCH"
    if [ "$(git rev-parse "origin/$EXISTING_BRANCH")" = "$rebased_head" ]; then
      return 0
    fi
    if [ "$attempt" -eq 3 ]; then
      return 1
    fi
    sleep "$attempt"
  done
}

# Commits the named paths and pushes that commit onto main: `push_to_main MESSAGE PATH...`.
# Rebases over a main that moved only in $TOLERATED_PATHS (the other nightly jobs' files, as git
# pathspecs) and retries the push; any other move of main stops the push, so a stale result cannot
# overwrite newer work.
push_to_main() {
  local message=$1 base attempt path
  local -a excludes=() tolerated=()
  shift
  read -ra tolerated <<< "${TOLERATED_PATHS:-}"
  for path in "${tolerated[@]}"; do
    excludes+=(":(exclude)$path")
  done
  bot_git_identity
  git add -- "$@"
  git diff --cached --check
  git commit -m "$message"
  base=$(git rev-parse HEAD^)
  for attempt in 1 2 3; do
    git fetch origin main
    if ! git diff --quiet "$base" origin/main -- . "${excludes[@]}"; then
      echo "main changed beyond ${TOLERATED_PATHS:-the pushed paths}; leaving the commit unpushed" >&2
      return 1
    fi
    git rebase origin/main
    if git push origin HEAD:main; then
      return 0
    fi
  done
  return 1
}
