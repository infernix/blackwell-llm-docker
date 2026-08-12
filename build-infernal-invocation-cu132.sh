#!/usr/bin/env bash
set -euo pipefail

# Build a source-locked CUDA 13.2 image from vLLM Infernal Invocation and
# B12X master. A release lock records the exact base commit, composed tree,
# patch digest, and included pull requests for each source repository.

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

release_revision=${RELEASE_REVISION:-r2}
release_date=${RELEASE_DATE:-20260812}
release_dir="patches/releases/infernal-invocation-${release_revision}"
release_patch_dir="${release_dir#patches/}"
refresh_locks=${REFRESH_RELEASE_LOCKS:-0}

case "${release_revision}" in
  r[1-9]|r[1-9][0-9]*) ;;
  *)
    printf 'RELEASE_REVISION must use the rN form; got %s\n' \
      "${release_revision}" >&2
    exit 2
    ;;
esac
if [[ ! "${release_date}" =~ ^[0-9]{8}$ ]]; then
  printf 'RELEASE_DATE must use YYYYMMDD; got %s\n' "${release_date}" >&2
  exit 2
fi
if [[ "${refresh_locks}" != 0 && "${refresh_locks}" != 1 ]]; then
  printf 'REFRESH_RELEASE_LOCKS must be 0 or 1\n' >&2
  exit 2
fi

compose_component() {
  local manifest=$1 output=$2
  if [[ "${refresh_locks}" == 1 ]]; then
    python3 scripts/compose_vllm_release.py "${manifest}" --output-dir "${output}"
  fi
  [[ -f "${output}/integration.lock.json" ]] || {
    printf 'Release lock is missing: %s/integration.lock.json\n' "${output}" >&2
    exit 1
  }
  [[ -f "${output}/integration.patch" ]] || {
    printf 'Release patch is missing: %s/integration.patch\n' "${output}" >&2
    exit 1
  }
}

compose_component manifests/vllm/infernal-invocation.json "${release_dir}/vllm"
compose_component manifests/b12x/infernal-invocation.json "${release_dir}/b12x"
compose_component manifests/lmcache/infernal-invocation.json "${release_dir}/lmcache"

vllm_lock="${release_dir}/vllm/integration.lock.json"
b12x_lock="${release_dir}/b12x/integration.lock.json"
lmcache_lock="${release_dir}/lmcache/integration.lock.json"

export VLLM_REPO="$(jq -er '.base.repository' "${vllm_lock}")"
export VLLM_REF="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${vllm_lock}")"
export VLLM_COMMIT="$(jq -er '.base.commit' "${vllm_lock}")"
export VLLM_INTEGRATION_TREE="$(jq -er '.result.tree' "${vllm_lock}")"
export VLLM_PATCH_FILE="${release_patch_dir}/vllm/integration.patch"
export VLLM_PATCH_SHA256="$(jq -er '.result.patch_sha256' "${vllm_lock}")"
export VLLM_INTEGRATION_LOCK_FILE="${vllm_lock}"
export REQUIRE_CLEAN_VLLM_COMPOSITION=1
export VERIFY_VLLM_BASE_HEAD=${VERIFY_VLLM_BASE_HEAD:-1}

export B12X_REPO="$(jq -er '.base.repository' "${b12x_lock}")"
export B12X_REF="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${b12x_lock}")"
export B12X_COMMIT="$(jq -er '.base.commit' "${b12x_lock}")"
export B12X_INTEGRATION_TREE="$(jq -er '.result.tree' "${b12x_lock}")"
export B12X_PATCH_FILE="${release_patch_dir}/b12x/integration.patch"
export B12X_PATCH_SHA256="$(jq -er '.result.patch_sha256' "${b12x_lock}")"
export B12X_INTEGRATION_LOCK_FILE="${b12x_lock}"
export REQUIRE_CLEAN_B12X_COMPOSITION=1
export VERIFY_B12X_BASE_HEAD=${VERIFY_B12X_BASE_HEAD:-1}

export LMCACHE_REPO="$(jq -er '.base.repository' "${lmcache_lock}")"
export LMCACHE_REF="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lmcache_lock}")"
export LMCACHE_COMMIT="$(jq -er '.base.commit' "${lmcache_lock}")"
export LMCACHE_INTEGRATION_TREE="$(jq -er '.result.tree' "${lmcache_lock}")"
export LMCACHE_PATCH_FILE="${release_patch_dir}/lmcache/integration.patch"
export LMCACHE_PATCH_SHA256="$(jq -er '.result.patch_sha256' "${lmcache_lock}")"
export LMCACHE_INTEGRATION_LOCK_FILE="${lmcache_lock}"
export REQUIRE_CLEAN_LMCACHE_COMPOSITION=1
export VERIFY_LMCACHE_BASE_HEAD=${VERIFY_LMCACHE_BASE_HEAD:-1}

export IMAGE="${IMAGE:-voipmonitor/vllm:infernal-invocation-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-fi1ac6942-cu132-${release_date}-${release_revision}}"
export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.26.1rc0+infernal.invocation.${release_revision}.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}.fi1ac6942.cu132.${release_date}}"

export TORCH_VERSION=${TORCH_VERSION:-2.13.0+cu132}
export TORCHVISION_VERSION=${TORCHVISION_VERSION:-0.28.0+cu132}
export TORCH_BUNDLED_NCCL_VERSION=${TORCH_BUNDLED_NCCL_VERSION:-2.29.7}
export CUTLASS_REF=${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}
export CUTLASS_COMMIT=${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}
export CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION:-4.6.2}
export TILELANG_VERSION=${TILELANG_VERSION:-0.1.12}
export TOKENSPEED_MLA_VERSION=${TOKENSPEED_MLA_VERSION:-0.1.8}
export TVM_FFI_VERSION=${TVM_FFI_VERSION:-0.1.11}
export QUACK_KERNELS_SPEC=${QUACK_KERNELS_SPEC:-quack-kernels==0.6.4}
export FASTSAFETENSORS_SPEC=${FASTSAFETENSORS_SPEC:-fastsafetensors>=0.3.3}
export HUMMING_KERNELS_SPEC=${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.12}
export VLLM_RUNTIME_EXTRA_PACKAGES=${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}

torch_base_id="$(tr '+.' '--' <<<"${TORCH_VERSION}")"
cutlass_base_id="$(tr '.' '-' <<<"${CUTLASS_DSL_VERSION}")"
base_contract_id="$({
  sed -n '1,/^# Stage 2:/p' Dockerfile.vllm-b12x-cu132
  sha256sum scripts/install-patched-nccl.sh
} | sha256sum | cut -c1-12)"
export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:infernal-invocation-cu132-build-base-torch${torch_base_id}-cutlass${cutlass_base_id}-base${base_contract_id}}"
if [[ -z "${BUILD_BASE_IMAGE+x}" ]]; then
  if docker image inspect "${BUILD_BASE_IMAGE_TAG}" >/dev/null 2>&1; then
    export BUILD_BASE_IMAGE=0
  else
    export BUILD_BASE_IMAGE=1
  fi
fi
export PUSH_BASE_IMAGE=${PUSH_BASE_IMAGE:-0}
export PIN_SOURCE_COMMITS=1
export MAX_JOBS=${MAX_JOBS:-64}
export VLLM_MAX_JOBS=${VLLM_MAX_JOBS:-64}
export NVCC_THREADS=${NVCC_THREADS:-1}
export VLLM_NVCC_THREADS=${VLLM_NVCC_THREADS:-1}

export NCCL_REPO=${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}
export NCCL_REF=${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}
export NCCL_COMMIT=${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}
export FLASHINFER_REPO=${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}
export FLASHINFER_REF=${FLASHINFER_REF:-integration/main-pr4393-pcie-ipc-qualified-20260807}
export FLASHINFER_COMMIT=${FLASHINFER_COMMIT:-1ac6942776b383c6b03c7a5805a22e72a3e3349f}
export FLASHINFER_BUILD_CUBIN=${FLASHINFER_BUILD_CUBIN:-0}
export DEEPGEMM_REPO=${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}
export DEEPGEMM_REF=${DEEPGEMM_REF:-a6b593d2826719dcf4892609af7b84ee23aaf32a}
export DEEPGEMM_COMMIT=${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}
export EXLLAMAV3_REPO=${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}
export EXLLAMAV3_REF=${EXLLAMAV3_REF:-a1-retile-sm120}
export EXLLAMAV3_COMMIT=${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}
export INSTANTTENSOR_REPO=${INSTANTTENSOR_REPO:-https://github.com/voipmonitor/InstantTensor.git}
export INSTANTTENSOR_REF=${INSTANTTENSOR_REF:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}
export INSTANTTENSOR_COMMIT=${INSTANTTENSOR_COMMIT:-49b4010afc1cae0441e71fe0b0bffc24fa05e932}
export LMCACHE_BUILD_VERSION=${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.5}
export XGRAMMAR_REPO=${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}
export XGRAMMAR_REF=${XGRAMMAR_REF:-v0.2.5}
export XGRAMMAR_COMMIT=${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}
export XGRAMMAR_VERSION=${XGRAMMAR_VERSION:-0.2.5}
export XGRAMMAR_TRANSFORMERS5_COMPAT=${XGRAMMAR_TRANSFORMERS5_COMPAT:-1}

export LAUNCHER_REPO=${LAUNCHER_REPO:-${VLLM_REPO}}
export LAUNCHER_REF=${LAUNCHER_REF:-${VLLM_REF}}
export LAUNCHER_COMMIT=${LAUNCHER_COMMIT:-${VLLM_COMMIT}}
export VLLM_REQUIRED_LAUNCHERS="${VLLM_REQUIRED_LAUNCHERS:-serve-ds4-flash.sh serve-ds4-flash-spark.sh serve-glm52.sh}"

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'revision=%s\nimage=%s\nversion=%s\nvllm_ref=%s\nvllm_commit=%s\nvllm_tree=%s\nvllm_patch=%s\nb12x_ref=%s\nb12x_commit=%s\nb12x_tree=%s\nb12x_patch=%s\nlmcache_ref=%s\nlmcache_commit=%s\nlmcache_tree=%s\nlmcache_patch=%s\ntorch=%s\ntorchvision=%s\ncutlass_ref=%s\ncutlass_commit=%s\ncutlass_dsl=%s\nbuild_base_image=%s\nflashinfer_commit=%s\nlauncher_commit=%s\n' \
    "${release_revision}" "${IMAGE}" "${VLLM_BUILD_VERSION}" \
    "${VLLM_REF}" "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" \
    "${VLLM_PATCH_FILE}" \
    "${B12X_REF}" "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" \
    "${B12X_PATCH_FILE}" "${LMCACHE_REF}" "${LMCACHE_COMMIT}" \
    "${LMCACHE_INTEGRATION_TREE}" "${LMCACHE_PATCH_FILE}" \
    "${TORCH_VERSION}" "${TORCHVISION_VERSION}" "${CUTLASS_REF}" \
    "${CUTLASS_COMMIT}" "${CUTLASS_DSL_VERSION}" "${BUILD_BASE_IMAGE_TAG}" \
    "${FLASHINFER_COMMIT}" "${LAUNCHER_COMMIT}"
  exit 0
fi

python3 -m pytest -q tests/test_compose_vllm_release.py
./tests/test-install-patched-nccl.sh
./tests/test-infernal-invocation-release-composition.sh --source-only
./build-vllm-b12x-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${TORCH_VERSION}" \
  '."local-inference.torch.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${VLLM_INTEGRATION_TREE}" \
  '."local-inference.vllm.integration.tree" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${B12X_INTEGRATION_TREE}" \
  '."local-inference.b12x.integration.tree" == $value' <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${IMAGE}" -c \
  'import torch; assert torch.__version__ == "2.13.0+cu132"; assert torch.version.cuda == "13.2"'
docker run --rm --entrypoint /usr/local/bin/serve-ds4-flash.sh \
  -e DRY_RUN=1 -e MODE=dspark -e DSPARK_TOKENS=5 -e MAX_NUM_SEQS=16 \
  -e GRAPH=auto -e LOAD_FORMAT=instanttensor "${IMAGE}"

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${IMAGE}"
fi
