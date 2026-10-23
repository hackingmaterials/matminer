#!/usr/bin/env bash
# Commit Claude's change, only if it passes these checks. Claude runs this
# once, with no arguments, from the code checkout, as its only way to write.
# autofix.yml copies it outside the checkout first, next to autotriage.conf,
# autotriage_lib.sh, and base.sha (the commit Claude started from).
#
# For an issue: commit to autofix/issue-<n> and open a PR that fixes it
# (a draft, unless merge_own_fixes = yes). For a PR: add one commit to it.
# Posts the result, or why it refused, on the issue or PR. Uses the token
# in GH_TOKEN or GITHUB_TOKEN (the Claude app's, so CI runs on the result).
set -euo pipefail

here="$(cd "$(dirname "$0")" && pwd)"
source "$here/autotriage_lib.sh"
date >"$here/ran"

# Who asked, about what: set by autofix.yml, not by Claude.
number="${NUMBER:-}"
requester="${REQUESTER:-}"
[[ "$number" =~ ^[0-9]+$ ]] || fail "'$number' is not an issue or PR number"
who="#$number"
[[ "$requester" =~ ^[A-Za-z0-9][A-Za-z0-9-]*$ ]] || fail "'$requester' is not a GitHub username"

# From here on, a refusal is also posted where the change was asked for, and
# so is any unexpected error.
refuse() {
  posted=true
  gh issue comment "$number" --body "Not published: $1." >/dev/null || true
}
fail() {
  echo "REFUSED $who: $1"
  refuse "$1"
  exit 1
}
posted=false
on_exit() {
  local status=$?
  [[ $status -eq 0 || $posted == true ]] || refuse "an unexpected error stopped it; see the workflow log"
}
trap on_exit EXIT

# Read and check the settings.
load_conf "$here/autotriage.conf"
[[ "$mode" != conservative ]] || fail "mode is conservative, so the bot never commits"
[[ "$autofix_prs" =~ ^(on|off)$ ]] || fail "bad autofix_prs '$autofix_prs'"
[[ "$autofix_issues" =~ ^(on|off)$ ]] || fail "bad autofix_issues '$autofix_issues'"
[[ "$merge_own_fixes" =~ ^(yes|no)$ ]] || fail "bad merge_own_fixes '$merge_own_fixes'"
[[ "$autofix_max_files" =~ ^[0-9]+$ ]] || fail "bad autofix_max_files '$autofix_max_files'"
[[ "$autofix_max_changed_lines" =~ ^[0-9]+$ ]] || fail "bad autofix_max_changed_lines '$autofix_max_changed_lines'"
token="${GH_TOKEN:-${GITHUB_TOKEN:-}}"
[[ -n "$token" ]] || fail "no GitHub token"

# Check the target is open, and whether it is an issue or a PR.
info=$(gh api "repos/{owner}/{repo}/issues/$number")
[[ $(jq -r .state <<<"$info") == open ]] || fail "not open"
title=$(jq -r .title <<<"$info")
target="issue"
[[ $(jq 'has("pull_request")' <<<"$info") == false ]] || target="pr"
if [[ "$target" == pr ]]; then
  [[ "$autofix_prs" == on ]] || fail "autofix_prs is off in autotriage.conf"
else
  [[ "$autofix_issues" == on ]] || fail "autofix_issues is off in autotriage.conf"
fi

# Take Claude's summary and change out of the working tree, leaving it clean.
# Files the action resets to the default branch's version on PRs are left out.
summary=$(cat .autofix_summary.md 2>/dev/null || true)
rm -f .autofix_summary.md
[[ -n "$summary" ]] || summary="The autofix bot left no summary; see the workflow run."
patch="$here/autofix.patch"
git add -A
git diff --cached --binary -- . ':!.claude' ':!.claude-pr' ':!.claude.json' ':!.mcp.json' \
  ':!.gitmodules' ':!.ripgreprc' ':!CLAUDE.md' ':!CLAUDE.local.md' ':!.husky' >"$patch"
git reset -q --hard
git clean -fdq

# No change: report back and stop.
if [[ ! -s "$patch" ]]; then
  gh issue comment "$number" --body "$summary

No code changed."
  echo "NO CHANGE for #$number"
  exit 0
fi

base=$(cat "$here/base.sha")
[[ "$base" =~ ^[0-9a-f]{40}$ ]] || fail "bad base commit '$base'"
default=$(gh repo view --json defaultBranchRef --jq .defaultBranchRef.name)

# For a PR, check its branch first: in this repo, not the default branch,
# and unchanged since Claude started.
if [[ "$target" == pr ]]; then
  pr=$(gh pr view "$number" --json headRefName,headRefOid,isCrossRepository)
  [[ $(jq -r .isCrossRepository <<<"$pr") == false ]] ||
    fail "comes from a fork; the bot only commits to branches in this repository"
  branch=$(jq -r .headRefName <<<"$pr")
  [[ "$branch" != "$default" ]] || fail "its branch is the default branch"
  [[ $(jq -r .headRefOid <<<"$pr") == "$base" ]] || fail "the PR changed while the bot worked; ask again"
fi

# Check the patch applies to the commit Claude started from.
git switch -q --detach "$base"
git apply --check "$patch" || fail "the change does not apply to $base"

# Allow only edits and plain file creations or deletions: no renames, mode
# changes, executables, symlinks, or submodules.
plain='^ (create|delete) mode 100644 '
deletion='^ delete mode 100644 (.*)$'
shapes=$(git apply --summary "$patch")
deletions=""
while IFS= read -r line; do
  [[ -z "$line" || "$line" =~ $plain ]] || fail "the change has '${line# }'"
  if [[ "$line" =~ $deletion ]]; then
    deletions+="${BASH_REMATCH[1]}"$'\n'
  fi
done <<<"$shapes"

# Check each file: not binary (unless deleted), not protected, inside
# autofix_paths if any are listed. Count files, lines, and test files, and
# whether it adds or edits code: anything not deleted, docs, or a test.
numstat=$(git apply --numstat "$patch")
n_files=0
n_lines=0
has_test=false
has_code=false
while IFS=$'\t' read -r added deleted path; do
  [[ -n "$path" ]] || continue
  is_deleted=false
  ! grep -qxF -- "$path" <<<"$deletions" || is_deleted=true
  if [[ "$added" == - ]]; then
    [[ "$is_deleted" == true ]] || fail "the change has binary file $path"
    added=0 deleted=0
  fi
  [[ "$path" != \"* ]] || fail "the change has unusual file name $path"
  ! matches_any "$path" ${protected[@]+"${protected[@]}"} || fail "it touches protected file $path"
  if [[ ${#autofix_paths[@]} -gt 0 ]]; then
    matches_any "$path" "${autofix_paths[@]}" || fail "it changes $path, which is not in autofix_paths"
  fi
  if matches_any "$path" ${test_paths[@]+"${test_paths[@]}"}; then
    has_test=true
  elif [[ "$is_deleted" == false ]] && ! matches_any "$path" ${docs_paths[@]+"${docs_paths[@]}"}; then
    has_code=true
  fi
  n_files=$((n_files + 1))
  n_lines=$((n_lines + added + deleted))
done <<<"$numstat"
[[ "$n_files" -gt 0 ]] || fail "the change has no files"
[[ "$n_files" -le "$autofix_max_files" ]] || fail "$n_files files changed (limit $autofix_max_files)"
[[ "$n_lines" -le "$autofix_max_changed_lines" ]] ||
  fail "$n_lines lines changed (limit $autofix_max_changed_lines)"

bot_name="claude[bot]"
bot_email="209825114+claude[bot]@users.noreply.github.com"
old_bot_email="41898282+github-actions[bot]@users.noreply.github.com"
commit() {
  git apply --index "$patch"
  git -c user.name="$bot_name" -c user.email="$bot_email" commit -q -m "$1"
}
# Push with the token; the checkout keeps no credentials of its own.
push() {
  local auth
  auth=$(printf 'x-access-token:%s' "$token" | base64 | tr -d '\n')
  git -c http.extraheader="AUTHORIZATION: basic $auth" push -q "$@"
}

if [[ "$target" == pr ]]; then
  # A PR: add one commit to its branch. The lease makes the push fail if the
  # branch moved since the check above.
  git switch -q -C "$branch" "$base"
  commit "Apply changes requested by @$requester"
  push --force-with-lease="$branch:$base" origin "$branch"
  sha=$(git rev-parse --short HEAD)
  gh pr comment "$number" --body "$summary

Pushed $sha."
  echo "PUSHED $sha to #$number"
  exit 0
fi

# An issue: the base must be on the default branch, and a fix that adds or
# edits code must change a test file if test_paths are listed. Changes that
# only touch docs or delete files need no test.
git merge-base --is-ancestor "$base" "origin/$default" || fail "$base is not on $default"
[[ ${#test_paths[@]} -eq 0 || "$has_test" == true || "$has_code" == false ]] ||
  fail "no test changed; a code fix needs a regression test"

# Don't overwrite commits someone else pushed to the bot's branch.
branch="autofix/issue-$number"
old=$(git rev-parse -q --verify "refs/remotes/origin/$branch" || true)
if [[ -n "$old" ]]; then
  others=$(git log --format=%ae "origin/$default..$old" | grep -vxF -e "$bot_email" -e "$old_bot_email" || true)
  [[ -z "$others" ]] || fail "someone else pushed to $branch; not overwriting it"
fi

# Commit on a fresh branch from the base, and push it. The lease makes the
# push fail if the branch moved since the check above.
git switch -q -C "$branch" "$base"
commit "Fix #$number: $title"
push --force-with-lease="$branch:$old" origin "$branch"

# Open a PR, or update the one already open for this branch. Only with
# merge_own_fixes = yes is it ready to merge; otherwise it's a draft.
draft=(--draft)
if [[ "$merge_own_fixes" == yes ]]; then
  draft=()
fi
body="$summary

Fixes #$number"
url=$(gh pr list --head "$branch" --state open --json url --jq '.[0].url // empty')
if [[ -z "$url" ]]; then
  url=$(gh pr create ${draft[@]+"${draft[@]}"} --base "$default" --head "$branch" --title "Fix #$number: $title" --body "$body")
  verb=Opened
else
  gh pr edit "$url" --body "$body" >/dev/null
  verb=Updated
fi

# Tell the requester.
gh issue comment "$number" --body "$verb $url; please review."
echo "PUBLISHED $url"
