#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=../launchers/glm52-dcp-prefill-policy.sh
source "${repo_root}/launchers/glm52-dcp-prefill-policy.sh"

assert_policy() {
  local expected="$1"
  shift
  local actual
  actual="$(resolve_glm52_dcp_prefill_policy "$@")"
  [[ "${actual}" == "${expected}" ]] || {
    printf 'policy(%s) = %s, expected %s\n' "$*" "${actual}" "${expected}" >&2
    exit 1
  }
}

# query gather, full CKV gather, owner merge, indexer shards, prefetch depth
assert_policy "1 0 0 0 0" 4 1 auto auto auto auto auto
assert_policy "1 0 0 0 0" 8 1 auto auto auto auto auto
assert_policy "0 0 0 0 0" 6 1 auto auto auto auto auto
assert_policy "1 1 1 0 1" 4 2 auto auto auto auto auto
assert_policy "1 1 1 0 1" 4 4 auto auto auto auto auto
assert_policy "1 1 1 0 1" 8 2 auto auto auto auto auto
assert_policy "1 1 1 2 1" 8 4 auto auto auto auto auto
assert_policy "1 1 1 4 1" 8 8 auto auto auto auto auto
assert_policy "0 0 1 0 0" 6 2 auto auto auto auto auto
assert_policy "0 0 1 0 0" 6 3 auto auto auto auto auto
assert_policy "0 0 1 0 0" 6 6 auto auto auto auto auto

# Explicit values are never rewritten by the automatic policy.
assert_policy "0 0 0 4 3" 8 4 0 0 0 4 3
assert_policy "1 0 1 2 0" 8 4 auto 0 auto auto auto
assert_policy "1 1 1 2 2" 6 3 1 1 1 2 2

echo "GLM-5.2 DCP prefill policy: PASS"
