#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_r6="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r6 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r6' <<<"${output_r6}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r6' \
  <<<"${output_r6}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r6' \
  <<<"${output_r6}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output_r6}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output_r6}"
grep -Fxq 'lmcache_repo=https://github.com/LMCache/LMCache.git' <<<"${output_r6}"
grep -Fxq 'lmcache_ref=v0.5.2' <<<"${output_r6}"
grep -Fxq \
  'lmcache_commit=cd2c0d6a6a982ec5e334bae7704e1029c06d3c97' \
  <<<"${output_r6}"
grep -Fxq 'lmcache_patch=lmcache/glm52-dcp-v052.patch' <<<"${output_r6}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.2' <<<"${output_r6}"

output_r7="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r7 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r7' <<<"${output_r7}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r7' \
  <<<"${output_r7}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r7' \
  <<<"${output_r7}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output_r7}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output_r7}"
grep -Fxq \
  'lmcache_repo=https://github.com/local-inference-lab/LMCache.git' \
  <<<"${output_r7}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output_r7}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r7}"
grep -Fxq 'lmcache_patch=' <<<"${output_r7}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.3' <<<"${output_r7}"

echo 'Gilded Gnosis release composition: PASS'
