#!/usr/bin/env bash
# The rolling model-discovery pull request is the one branch every agent workflow appends to.
# Locating it and rebasing it are identical everywhere, and the force-with-lease guard is easy
# to get subtly wrong, so both live here rather than once per workflow.
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

# Writes exists/number/branch to $GITHUB_OUTPUT. An empty branch means no rolling work exists yet.
find_rolling_pr() {
  local matches count number branch orphan_output
  local -a orphan_branches
  matches=$(retry_gh gh pr list --state open --limit 100 \
    --json number,headRefName,isCrossRepository,labels \
    --jq '[.[] | select((.isCrossRepository | not) and (([.labels[].name] | index("model-discovery")) or ([.labels[].name] | index("model-onboarding")) or (.headRefName | startswith("agents/model-discovery-"))))]')
  count=$(jq length <<< "$matches")
  if [ "$count" -gt 1 ]; then
    echo "Expected at most one rolling model discovery PR, found $count" >&2
    return 1
  fi

  number=$(jq -r '.[0].number // empty' <<< "$matches")
  branch=$(jq -r '.[0].headRefName // empty' <<< "$matches")
  if [ -n "$number" ]; then
    echo "Updating rolling discovery PR #$number from $branch"
    echo "exists=true" >> "$GITHUB_OUTPUT"
  else
    orphan_output=$(retry_gh gh api --paginate \
      "repos/$GITHUB_REPOSITORY/git/matching-refs/heads/agents/model-discovery-" \
      --jq '.[].ref | sub("^refs/heads/"; "")')
    mapfile -t orphan_branches < <(printf '%s' "$orphan_output" | sort)
    if [ "${#orphan_branches[@]}" -gt 1 ]; then
      echo "Expected at most one unpaired model discovery branch, found ${#orphan_branches[@]}" >&2
      return 1
    fi
    branch="${orphan_branches[0]:-}"
    if [ -n "$branch" ]; then
      echo "Adopting unpaired discovery branch $branch"
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
  git config user.name "emmy-onboarding-bot"
  git config user.email "emmy-onboarding-bot[bot]@users.noreply.github.com"
  gh auth setup-git
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
