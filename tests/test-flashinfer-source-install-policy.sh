#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
dockerfile="${repo_root}/Dockerfile.vllm-b12x-cu132"

grep -Fq 'BUILD_NVEP=0' "${dockerfile}"
grep -Fq 'BUILD_NCCL_EP=0' "${dockerfile}"
grep -Fq 'BUILD_NIXL_EP=0' "${dockerfile}"
grep -Fq 'python -m pip install --no-build-isolation --no-deps -v .' "${dockerfile}"
grep -Fq 'nccl_version_before="$(python -m pip show nvidia-nccl-cu13' "${dockerfile}"
grep -Fq 'test "${nccl_version_after}" = "${nccl_version_before}"' "${dockerfile}"

echo 'FlashInfer source-install policy: PASS'
