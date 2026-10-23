# Shared helpers for the gate scripts, autotriage_guard.sh (merging) and
# autofix_publish.sh (committing fixes), and for the workflows that read
# autotriage.conf. Source this file; don't run it.

# Print why and stop. Scripts set $who (e.g. "#12") to name the subject.
fail() {
  echo "REFUSED${who:+ $who}: $1"
  exit 1
}

# True if the string in $1 matches any of the bash globs that follow it.
matches_any() {
  local path="$1" pat
  shift
  for pat in "$@"; do
    # Unquoted on purpose: glob match, where "*" also matches "/".
    [[ "$path" == $pat ]] && return 0
  done
  return 1
}

# Read settings from the file in $1: "key = value" lines, where repeated keys
# add values. Unknown keys are errors, so a typo can't loosen a gate. Then
# fill in what the mode implies for anything the file leaves out.
load_conf() {
  local conf="$1" key value
  [[ -f "$conf" ]] || fail "no $conf, so this gate is closed"
  mode=""
  # The triage workflow.
  max_prs_per_run=5
  mentions=on
  # Merging (autotriage_guard.sh).
  merge_kinds=()
  allow_no_ci=no
  base_branch=""
  merge_method=squash
  max_changed_lines=0
  docs_paths=()
  block_labels=()
  block_branches=()
  deps_branches=()
  deps_paths=()
  # Fixing (autofix_publish.sh).
  autofix=""
  autofix_prs=""
  autofix_issues=""
  merge_own_fixes=""
  autofix_max_files=10
  autofix_max_changed_lines=200
  autofix_paths=()
  # Both.
  test_paths=()
  protected=()
  while IFS='=' read -r key value || [[ -n "$key" ]]; do
    key="${key//[[:space:]]/}"
    value="${value#"${value%%[![:space:]]*}"}"
    value="${value%"${value##*[![:space:]]}"}"
    case "$key" in
      "" | \#*) ;;
      mode) mode="$value" ;;
      max_prs_per_run) max_prs_per_run="$value" ;;
      mentions) mentions="$value" ;;
      merge) merge_kinds+=("$value") ;;
      allow_no_ci) allow_no_ci="$value" ;;
      base_branch) base_branch="$value" ;;
      merge_method) merge_method="$value" ;;
      max_changed_lines) max_changed_lines="$value" ;;
      docs_paths) docs_paths+=("$value") ;;
      block_label) block_labels+=("$value") ;;
      block_branch) block_branches+=("$value") ;;
      deps_branch) deps_branches+=("$value") ;;
      deps_paths) deps_paths+=("$value") ;;
      autofix) autofix="$value" ;;
      autofix_prs) autofix_prs="$value" ;;
      autofix_issues) autofix_issues="$value" ;;
      merge_own_fixes) merge_own_fixes="$value" ;;
      autofix_max_files) autofix_max_files="$value" ;;
      autofix_max_changed_lines) autofix_max_changed_lines="$value" ;;
      autofix_paths) autofix_paths+=("$value") ;;
      test_paths) test_paths+=("$value") ;;
      protected) protected+=("$value") ;;
      *) fail "unknown setting '$key' in $conf" ;;
    esac
  done <"$conf"

  # A missing mode means the safest one.
  [[ -n "$mode" ]] || mode=conservative
  [[ "$mode" =~ ^(conservative|normal|mayhem)$ ]] || fail "bad mode '$mode' in $conf"
  if [[ ${#merge_kinds[@]} -eq 0 ]]; then
    case "$mode" in
      normal) merge_kinds=(docs deps tested) ;;
      mayhem) merge_kinds=(docs deps tested reviewed) ;;
    esac
  fi
  if [[ -z "$autofix" ]]; then
    autofix=on
    [[ "$mode" != conservative ]] || autofix=off
  fi
  # autofix sets both kinds of fix; autofix_prs and autofix_issues override it.
  [[ -n "$autofix_prs" ]] || autofix_prs="$autofix"
  [[ -n "$autofix_issues" ]] || autofix_issues="$autofix"
  if [[ -z "$merge_own_fixes" ]]; then
    merge_own_fixes=no
    [[ "$mode" != mayhem ]] || merge_own_fixes=yes
  fi
  # Conservative never commits or merges, whatever the file says.
  if [[ "$mode" == conservative ]]; then
    autofix=off autofix_prs=off autofix_issues=off merge_own_fixes=no
  fi
  return 0
}
