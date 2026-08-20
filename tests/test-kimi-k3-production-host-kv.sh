#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo_root}/launchers/serve-kimi-k3-production-dspark-ii"

if [[ "${KIMI_HOST_KV_CAPTURE:-0}" == 1 ]]; then
  printf 'HOST_KV_BACKEND=%s\n' "${HOST_KV_BACKEND-}"
  printf 'LMCACHE_MODE=%s\n' "${LMCACHE_MODE-}"
  printf 'PYTORCH_CUDA_ALLOC_CONF=%s\n' "${PYTORCH_CUDA_ALLOC_CONF-}"
  printf 'MAX_NUM_BATCHED_TOKENS=%s\n' "${MAX_NUM_BATCHED_TOKENS-}"
  printf 'B12X_MOE_WORKSPACE_TOKEN_LIMIT=%s\n' "${B12X_MOE_WORKSPACE_TOKEN_LIMIT-}"
  printf 'B12X_W4A16_PREFILL_FUSED_SUM=%s\n' "${B12X_W4A16_PREFILL_FUSED_SUM-}"
  printf 'ARG=%s\n' "$@"
  exit 0
fi

capture_launcher() {
  env -u HOST_KV_BACKEND -u LMCACHE_MODE -u PYTORCH_CUDA_ALLOC_CONF \
    -u KV_OFFLOADING_SIZE -u MAX_NUM_BATCHED_TOKENS \
    -u B12X_MOE_WORKSPACE_TOKEN_LIMIT -u B12X_W4A16_PREFILL_FUSED_SUM \
    KIMI_HOST_KV_CAPTURE=1 \
    LMCACHE_WRAPPER="${BASH_SOURCE[0]}" \
    KIMI_DSPARK_LAUNCHER=/qualified/dspark-launcher \
    "$@"
}

assert_line() {
  local output=$1 expected=$2
  grep -Fxq -- "${expected}" <<<"${output}" || {
    printf 'Missing line: %s\nOutput:\n%s\n' "${expected}" "${output}" >&2
    exit 1
  }
}

assert_no_line() {
  local output=$1 rejected=$2
  if grep -Fxq -- "${rejected}" <<<"${output}"; then
    printf 'Unexpected line: %s\nOutput:\n%s\n' "${rejected}" "${output}" >&2
    exit 1
  fi
}

native_output="$(capture_launcher "${launcher}" --served-model-name test-model)"
assert_line "${native_output}" 'HOST_KV_BACKEND=native'
assert_line "${native_output}" 'LMCACHE_MODE=off'
assert_line "${native_output}" 'PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False'
assert_line "${native_output}" 'MAX_NUM_BATCHED_TOKENS=4102'
assert_line "${native_output}" 'B12X_MOE_WORKSPACE_TOKEN_LIMIT=4096'
assert_line "${native_output}" 'B12X_W4A16_PREFILL_FUSED_SUM=1'
assert_line "${native_output}" 'ARG=/qualified/dspark-launcher'
assert_line "${native_output}" 'ARG=--kv-offloading-size'
assert_line "${native_output}" 'ARG=32'
assert_line "${native_output}" 'ARG=--kv-offloading-backend'
assert_line "${native_output}" 'ARG=native'
assert_line "${native_output}" 'ARG=--served-model-name'
assert_line "${native_output}" 'ARG=test-model'

native_override_output="$(
  capture_launcher env KV_OFFLOADING_SIZE=48.5 "${launcher}"
)"
assert_line "${native_override_output}" 'ARG=48.5'

lmcache_output="$(capture_launcher env LMCACHE_MODE=ram "${launcher}")"
assert_line "${lmcache_output}" 'HOST_KV_BACKEND=lmcache'
assert_line "${lmcache_output}" 'LMCACHE_MODE=ram'
assert_no_line "${lmcache_output}" 'ARG=--kv-offloading-size'

uncached_output="$(capture_launcher env LMCACHE_MODE=off "${launcher}")"
assert_line "${uncached_output}" 'HOST_KV_BACKEND=off'
assert_line "${uncached_output}" 'LMCACHE_MODE=off'
assert_no_line "${uncached_output}" 'ARG=--kv-offloading-size'

explicit_lmcache_output="$(
  capture_launcher env HOST_KV_BACKEND=lmcache LMCACHE_MODE=disk "${launcher}"
)"
assert_line "${explicit_lmcache_output}" 'HOST_KV_BACKEND=lmcache'
assert_line "${explicit_lmcache_output}" 'LMCACHE_MODE=disk'

if capture_launcher env HOST_KV_BACKEND=unsupported "${launcher}" >/dev/null 2>&1; then
  echo 'Unsupported HOST_KV_BACKEND unexpectedly succeeded' >&2
  exit 1
fi

printf 'Kimi-K3 production host-KV launcher tests passed.\n'
