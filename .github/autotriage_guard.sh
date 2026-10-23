#!/usr/bin/env bash
# Merge a PR only if it passes these checks.
# Settings, including the mode, live in autotriage.conf next to this script.
# Without that file nothing is merged.
set -euo pipefail

here="$(dirname "$0")"
source "$here/autotriage_lib.sh"

# Check the argument is a PR number.
pr="${1:?usage: autotriage_guard.sh <pr-number>}"
[[ "$pr" =~ ^[0-9]+$ ]] || fail "'$pr' is not a PR number"
who="#$pr"

# Read settings (see autotriage_lib.sh for the format).
conf="$here/autotriage.conf"
load_conf "$conf"

# Check the settings themselves.
[[ "$mode" != conservative ]] || fail "mode is conservative, so the bot never merges"
for kind in "${merge_kinds[@]}"; do
  [[ "$kind" =~ ^(docs|deps|tested|reviewed|none)$ ]] || fail "bad merge kind '$kind'"
done
! matches_any none "${merge_kinds[@]}" || fail "merging is off (merge = none in $conf)"
[[ "$merge_method" =~ ^(squash|merge|rebase)$ ]] || fail "bad merge_method '$merge_method'"
[[ "$max_changed_lines" =~ ^[0-9]+$ ]] || fail "bad max_changed_lines '$max_changed_lines'"
[[ "$allow_no_ci" =~ ^(yes|no)$ ]] || fail "bad allow_no_ci '$allow_no_ci'"
[[ "$merge_own_fixes" =~ ^(yes|no)$ ]] || fail "bad merge_own_fixes '$merge_own_fixes'"
[[ -n "$base_branch" ]] || base_branch=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name)

# Fetch the PR.
info=$(gh pr view "$pr" --json state,isDraft,mergeable,baseRefName,headRefName,headRefOid,isCrossRepository,files,changedFiles,additions,deletions,labels,reviewDecision,statusCheckRollup)

# Check the PR is open.
[[ $(jq -r .state <<<"$info") == OPEN ]] || fail "not open"
# Check it isn't a draft.
[[ $(jq -r .isDraft <<<"$info") == false ]] || fail "is a draft"
# Check it targets the base branch.
[[ $(jq -r .baseRefName <<<"$info") == "$base_branch" ]] || fail "does not target $base_branch"
# Check it has no conflicts.
[[ $(jq -r .mergeable <<<"$info") == MERGEABLE ]] || fail "not mergeable (conflicts?)"
# Check no reviewer has requested changes.
[[ $(jq -r .reviewDecision <<<"$info") != CHANGES_REQUESTED ]] || fail "a reviewer requested changes"
# Check the bot may merge from this branch: never its own fixes, unless
# merge_own_fixes = yes, and never a blocked branch.
head=$(jq -r .headRefName <<<"$info")
[[ "$merge_own_fixes" == yes || "$head" != autofix/* ]] || fail "the bot's own fixes need a person to merge them"
! matches_any "$head" ${block_branches[@]+"${block_branches[@]}"} || fail "PRs from $head are never auto-merged"
# Check no label holds it back.
labels=$(jq -r '.labels[].name' <<<"$info")
while IFS= read -r label; do
  [[ -n "$label" ]] || continue
  ! matches_any "$label" ${block_labels[@]+"${block_labels[@]}"} || fail "has label '$label'"
done <<<"$labels"

# Check we can see every changed file (GitHub lists at most 100).
n_listed=$(jq '.files | length' <<<"$info")
[[ "$n_listed" -eq $(jq .changedFiles <<<"$info") ]] || fail "too many changed files to check"
files=$(jq -r '.files[].path' <<<"$info")

# Bot dependency PRs: must come from this repo and only touch deps_paths.
# They skip the size limit, since lock file diffs are long.
if matches_any "$head" ${deps_branches[@]+"${deps_branches[@]}"}; then
  matches_any deps "${merge_kinds[@]}" || fail "dependency PRs aren't auto-merged here"
  [[ $(jq -r .isCrossRepository <<<"$info") == false ]] || fail "dependency branch comes from a fork"
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    matches_any "$f" ${deps_paths[@]+"${deps_paths[@]}"} || fail "dependency PR changes $f"
  done <<<"$files"
else
  # Sort the PR by its files: docs only, code with tests, or code without.
  kind=docs
  has_test=false
  while IFS= read -r f; do
    [[ -n "$f" ]] || continue
    matches_any "$f" ${docs_paths[@]+"${docs_paths[@]}"} || kind=code
    if matches_any "$f" ${test_paths[@]+"${test_paths[@]}"}; then
      has_test=true
    fi
  done <<<"$files"
  if [[ "$kind" == code ]]; then
    kind=untested
    [[ "$has_test" == false ]] || kind=tested
  fi
  # Check that kind may be merged; "reviewed" covers every kind.
  case "$kind" in
    docs) what="docs-only PRs" ;;
    tested) what="code changes with tests" ;;
    untested) what="code changes without tests" ;;
  esac
  matches_any "$kind" "${merge_kinds[@]}" || matches_any reviewed "${merge_kinds[@]}" ||
    fail "$what aren't auto-merged here"
  # Check it is small enough.
  n_lines=$(jq '.additions + .deletions' <<<"$info")
  [[ "$max_changed_lines" -eq 0 || "$n_lines" -le "$max_changed_lines" ]] ||
    fail "$n_lines changed lines (limit $max_changed_lines)"
fi

# Block changes to protected files.
while IFS= read -r f; do
  [[ -n "$f" ]] || continue
  ! matches_any "$f" ${protected[@]+"${protected[@]}"} || fail "touches protected file $f"
done <<<"$files"

# True if the repo has CI of its own: a workflow besides autotriage's, or
# another CI service's config.
has_ci() {
  local f
  for f in "$here"/workflows/*.yml "$here"/workflows/*.yaml; do
    [[ -e "$f" ]] || continue
    case "${f##*/}" in
      autotriage.yml | autofix.yml | upgrade-dependencies.yml) ;;
      *) return 0 ;;
    esac
  done
  for f in .travis.yml .circleci .gitlab-ci.yml azure-pipelines.yml appveyor.yml \
    .appveyor.yml Jenkinsfile .buildkite .drone.yml bitbucket-pipelines.yml; do
    [[ -e "$here/../$f" ]] && return 0
  done
  return 1
}

# Check tests ran. With allow_no_ci, a repo with no CI at all may merge on
# review alone; any CI in the repo means checks are required.
n_checks=$(jq '.statusCheckRollup | length' <<<"$info")
if [[ "$n_checks" -eq 0 ]]; then
  [[ "$allow_no_ci" == yes ]] || fail "no CI checks have run"
  ! has_ci || fail "no CI checks have run, and this repo has CI"
fi
# Check tests passed.
bad=$(jq -r '.statusCheckRollup[]
  | (((.conclusion // "") | if . == "" then null else . end) // .status // .state) as $r
  | select(($r | ascii_upcase) as $u | ["SUCCESS","NEUTRAL","SKIPPED"] | index($u) | not)
  | "\(.name // .context): \($r)"' <<<"$info")
[[ -z "$bad" ]] || fail "CI not green: $(tr '\n' ';' <<<"$bad")"

# Merge the exact commit checked.
gh pr merge "$pr" "--$merge_method" --match-head-commit "$(jq -r .headRefOid <<<"$info")"
echo "MERGED #$pr"
