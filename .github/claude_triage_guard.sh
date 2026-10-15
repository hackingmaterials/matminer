#!/usr/bin/env bash
# Merge a PR only if it passes these checks.
set -euo pipefail

# Check the argument is a PR number.
pr="${1:?usage: claude_triage_guard.sh <pr-number>}"
[[ "$pr" =~ ^[0-9]+$ ]] || { echo "REFUSED: '$pr' is not a PR number"; exit 1; }

# Fetch the PR.
info=$(gh pr view "$pr" --json state,isDraft,mergeable,baseRefName,headRefName,headRefOid,isCrossRepository,files,statusCheckRollup)

# Print why and stop.
fail() { echo "REFUSED #$pr: $1"; exit 1; }

# Check the PR is open.
[[ $(jq -r .state <<<"$info") == OPEN ]] || fail "not open"
# Check it isn't a draft.
[[ $(jq -r .isDraft <<<"$info") == false ]] || fail "is a draft"
# Check it targets main.
[[ $(jq -r .baseRefName <<<"$info") == main ]] || fail "does not target main"
# Check it has no conflicts.
[[ $(jq -r .mergeable <<<"$info") == MERGEABLE ]] || fail "not mergeable (conflicts?)"

# Block fake dependency PRs from forks.
if [[ $(jq -r .headRefName <<<"$info") == auto-dependency-upgrades &&
      $(jq -r .isCrossRepository <<<"$info") == true ]]; then
  fail "auto-dependency-upgrades branch comes from a fork"
fi

# Block changes to .github/ or packaging files.
if jq -e '[.files[].path | select(startswith(".github/") or IN("pyproject.toml", "setup.cfg", "MANIFEST.in"))] | length > 0' <<<"$info" >/dev/null; then
  fail "touches .github/ or packaging files"
fi

# Check tests ran.
n_checks=$(jq '.statusCheckRollup | length' <<<"$info")
[[ "$n_checks" -gt 0 ]] || fail "no CI checks have run"
# Check tests passed.
bad=$(jq -r '.statusCheckRollup[]
  | (.conclusion // .state) as $r
  | select(($r | ascii_upcase) as $u | ["SUCCESS","NEUTRAL","SKIPPED"] | index($u) | not)
  | "\(.name // .context): \($r)"' <<<"$info")
[[ -z "$bad" ]] || fail "CI not green: $(tr '\n' ';' <<<"$bad")"

# Merge the exact commit checked.
gh pr merge "$pr" --squash --match-head-commit "$(jq -r .headRefOid <<<"$info")"
echo "MERGED #$pr"
