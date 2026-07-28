#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r6 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r6' <<<"${output}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r6' \
  <<<"${output}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r6' \
  <<<"${output}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output}"

echo 'Gilded Gnosis release composition: PASS'
