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
assert_policy "1 1 0 4 0" 8 8 auto auto auto auto auto 0 0
assert_policy "0 0 1 0 0" 6 2 auto auto auto auto auto
assert_policy "0 0 1 0 0" 6 3 auto auto auto auto auto
assert_policy "0 0 1 0 0" 6 6 auto auto auto auto auto

# Explicit values are never rewritten by the automatic policy.
assert_policy "0 0 0 4 3" 8 4 0 0 0 4 3
assert_policy "1 0 1 2 0" 8 4 auto 0 auto auto auto
assert_policy "1 1 1 2 2" 6 3 1 1 1 2 2
assert_policy "1 1 1 2 0" 8 4 auto auto auto auto auto 0
assert_policy "1 1 1 2 1" 8 4 auto auto auto auto 1 0

emit_topology() {
  local mode="$1"
  local count="$2"
  local i j relation

  printf '\t'
  for ((i = 0; i < count; ++i)); do
    printf 'GPU%d\t' "${i}"
  done
  printf 'CPU Affinity\tNUMA Affinity\n'

  for ((i = 0; i < count; ++i)); do
    printf 'GPU%d\t' "${i}"
    for ((j = 0; j < count; ++j)); do
      if ((i == j)); then
        relation=X
      else
        case "${mode}" in
          paired)
            if ((i / 2 == j / 2)); then
              relation=PIX
            else
              relation=NODE
            fi
            ;;
          dual-phb)
            if ((i / 4 == j / 4)); then
              relation=PHB
            else
              relation=SYS
            fi
            ;;
          pxb) relation=PXB ;;
          *) return 2 ;;
        esac
      fi
      printf '%s\t' "${relation}"
    done
    printf '0-127\t0\n'
  done
}

assert_topology() {
  local expected="$1"
  local tp="$2"
  local dcp="$3"
  local gpus="$4"
  local mode="$5"
  local count="$6"
  local actual

  actual="$(emit_topology "${mode}" "${count}" | \
    classify_glm52_ckv_prefetch_topology "${tp}" "${dcp}" "${gpus}")"
  [[ "${actual}" == "${expected}" ]] || {
    printf 'topology(%s) = %s, expected %s\n' \
      "${mode} ${gpus}" "${actual}" "${expected}" >&2
    exit 1
  }
}

assert_topology \
  "safe:local-dcp-rings" 8 4 "0,1,2,3,4,5,6,7" paired 16
assert_topology \
  "unsafe:dcp-ring-lacks-local-links" 8 4 "0,2,4,6,8,10,12,14" paired 16
assert_topology \
  "unsafe:tp-crosses-numa" 8 4 "0,1,2,3,4,5,6,7" dual-phb 8
assert_topology \
  "safe:local-dcp-rings" 8 8 "0,1,2,3,4,5,6,7" pxb 8

echo "GLM-5.2 DCP prefill policy: PASS"
