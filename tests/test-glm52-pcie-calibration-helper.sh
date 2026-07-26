#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
helper="${repo_root}/launchers/serve-glm52-v19.sh"
calibrator="${repo_root}/tests/fake-glm52-pcie-calibrator.sh"

run_helper() {
  env \
    TP=8 \
    DCP=4 \
    GPUS=0,1,2,3,4,5,6,7 \
    PCIE_CALIBRATION_ONLY=1 \
    PCIE_CALIBRATOR="${calibrator}" \
    "$@" \
    "${helper}" 2>&1
}

assert_contains() {
  local output="$1"
  local expected="$2"
  grep -Fqx "${expected}" <<<"${output}" || {
    printf 'missing expected output: %s\n--- output ---\n%s\n' \
      "${expected}" "${output}" >&2
    exit 1
  }
}

measured="$(run_helper)"
assert_contains "${measured}" "VLLM_DCP_QUERY_SPLIT=1"
assert_contains "${measured}" \
  "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=8192"
assert_contains "${measured}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=1"
assert_contains "${measured}" "VLLM_PCIE_DMA_MIN_BYTES=25165824"
assert_contains "${measured}" "PCIE_CALIBRATION_STATUS=measured"

no_dma="$(run_helper FAKE_PCIE_DMA_MIN_BYTES=off)"
assert_contains "${no_dma}" "VLLM_PCIE_DMA_MIN_BYTES=off"

explicit="$(run_helper \
  DCP_QUERY_SPLIT=0 \
  DCP_CKV_GATHER=1 \
  DCP_CKV_PREFETCH_DEPTH=0 \
  PCIE_DMA_MIN_BYTES=12MB)"
assert_contains "${explicit}" "VLLM_DCP_QUERY_SPLIT=0"
assert_contains "${explicit}" \
  "VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS=0"
assert_contains "${explicit}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0"
assert_contains "${explicit}" "VLLM_PCIE_DMA_MIN_BYTES=12MB"
assert_contains "${explicit}" "PCIE_CALIBRATION_STATUS=skipped:all-explicit"

compressed="$(run_helper \
  F8_DMA=ring \
  DCP_CKV_PREFETCH_TOPOLOGY=safe)"
assert_contains "${compressed}" \
  "PCIE_CALIBRATION_STATUS=skipped:explicit-compressed-dma"
assert_contains "${compressed}" "VLLM_PCIE_DMA_MIN_BYTES=6MB"

fallback="$(run_helper \
  FAKE_PCIE_CALIBRATION_FAIL=1 \
  DCP_CKV_PREFETCH_TOPOLOGY=unsafe)"
assert_contains "${fallback}" \
  "PCIE_CALIBRATION_STATUS=failed:fallback-to-topology"
assert_contains "${fallback}" "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0"
assert_contains "${fallback}" "VLLM_PCIE_DMA_MIN_BYTES=6MB"

if run_helper PCIE_CALIBRATION_TIMEOUT=0 >/dev/null; then
  echo "zero calibration timeout was accepted" >&2
  exit 1
fi

echo "GLM-5.2 PCIe calibration helper: PASS"
