#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

grep -Fq 'ARG TORCH_VERSION=2.12.0+cu132' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'ARG TORCHVISION_VERSION=0.27.0+cu132' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'cuda-python==13.2.0' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'nvidia-cuda-nvcc==13.2.78' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'nvidia-nvvm==13.2.78' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'COPY scripts/install-patched-nccl.sh /usr/local/bin/install-patched-nccl' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'md.version("nvidia-nccl-cu13")' \
  "${repo_root}/scripts/install-patched-nccl.sh"
grep -Fq 'if [[ -s "${vllm_patch}" ]]' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'if [[ -s "${b12x_patch}" ]]' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'COPY --from=vllm-build /usr/local/bin/lmcache-*.sh /usr/local/bin/' \
  "${repo_root}/Dockerfile.vllm-b12x-cu132"
grep -Fq 'vllm_cache_source="${VLLM_INTEGRATION_TREE:-${VLLM_COMMIT}}"' \
  "${repo_root}/build-vllm-b12x-cu132.sh"
grep -Fq 'b12x_cache_source="${B12X_INTEGRATION_TREE:-${B12X_COMMIT}}"' \
  "${repo_root}/build-vllm-b12x-cu132.sh"
if grep -Fq 'Clean GG composition requires' \
  "${repo_root}/build-vllm-b12x-cu132.sh"; then
  printf 'Clean source composition must not be restricted to one vLLM branch\n' >&2
  exit 1
fi

if [[ "${1:-}" == --source-only ]]; then
  exit 0
fi

output="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 ./build-infernal-invocation-cu132.sh
)"
grep -Fxq 'revision=r2' <<<"${output}"
grep -Fxq 'vllm_ref=dev/infernal-invocation' <<<"${output}"
grep -Fxq 'vllm_commit=c8d04a543e0e8b0896e60b8b11bec0bb2d780860' <<<"${output}"
grep -Fxq 'vllm_tree=344438d742b3cb3f3bd1851a0e9f33f4ebac64e0' <<<"${output}"
grep -Fxq 'vllm_patch=releases/infernal-invocation-r2/vllm/integration.patch' <<<"${output}"
grep -Fxq 'b12x_ref=master' <<<"${output}"
grep -Fxq 'b12x_commit=184d7d52ad630841d0c6caf962f8b9d36f38992a' <<<"${output}"
grep -Fxq 'b12x_tree=1584743fd972ead81619e8f8934cb7bca61571db' <<<"${output}"
grep -Fxq 'b12x_patch=releases/infernal-invocation-r2/b12x/integration.patch' <<<"${output}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output}"
grep -Fxq 'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' <<<"${output}"
grep -Fxq 'lmcache_tree=ccccdfc37f108ab674ac0418b5ac5fc1c8b0857e' <<<"${output}"
grep -Fxq 'lmcache_patch=releases/infernal-invocation-r2/lmcache/integration.patch' <<<"${output}"
grep -Fxq 'torch=2.13.0+cu132' <<<"${output}"
grep -Fxq 'torchvision=0.28.0+cu132' <<<"${output}"
grep -Fxq 'cutlass_ref=e6233cbac5d7c7a865c19c91cd684ceece19513c' <<<"${output}"
grep -Fxq 'cutlass_commit=e6233cbac5d7c7a865c19c91cd684ceece19513c' <<<"${output}"
grep -Fxq 'cutlass_dsl=4.6.2' <<<"${output}"
grep -Eq '^build_base_image=voipmonitor/vllm:infernal-invocation-cu132-build-base-torch2-13-0-cu132-cutlass4-6-2-base[0-9a-f]{12}$' <<<"${output}"
