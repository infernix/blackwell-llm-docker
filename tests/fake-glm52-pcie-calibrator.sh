#!/usr/bin/env bash
set -euo pipefail

require_env() {
  local name="$1"
  local expected="$2"
  local actual="${!name:-}"
  if [[ "${actual}" != "${expected}" ]]; then
    printf '%s: expected %q, got %q\n' "${name}" "${expected}" "${actual}" >&2
    exit 1
  fi
}

require_env VLLM_ENABLE_PCIE_ALLREDUCE 1
require_env VLLM_PCIE_ALLREDUCE_BACKEND b12x
require_env VLLM_USE_B12X_PCIE_DMA 1
require_env VLLM_PCIE_DMA_FP8 0
require_env NCCL_PROTO LL,LL128,Simple

if [[ "${FAKE_PCIE_CALIBRATION_FAIL:-0}" == "1" ]]; then
  exit 1
fi

if [[ "${FAKE_PCIE_CALIBRATION_BANNER:-0}" == "1" ]]; then
  printf 'fake calibration diagnostic\n'
fi

printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
  "${FAKE_PCIE_CALIBRATION_STATUS:-measured}" \
  "${FAKE_PCIE_PREFETCH_DEPTH:-1}" \
  "${FAKE_PCIE_QUERY_SPLIT:-1}" \
  "${FAKE_PCIE_QUERY_SPLIT_MIN_CONTEXT_TOKENS:-8192}" \
  "${FAKE_PCIE_DMA_MIN_BYTES:-25165824}" \
  "${FAKE_PCIE_CALIBRATION_CACHE:-/tmp/fake-calibration.json}"
