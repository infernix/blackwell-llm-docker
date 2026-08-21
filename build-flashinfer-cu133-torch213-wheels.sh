#!/usr/bin/env bash
# Build the FlashInfer wheels consumed by CUDA 13.3 PyTorch 2.13 runtimes.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

base_image=${BASE_IMAGE:-voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2}
flashinfer_repo=${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}
flashinfer_ref=${FLASHINFER_REF:-integration/main-pr4393-pcie-ipc-qualified-20260807}
flashinfer_commit=${FLASHINFER_COMMIT:-1ac6942776b383c6b03c7a5805a22e72a3e3349f}
flashinfer_version=${FLASHINFER_VERSION:-0.6.18+cu133}
cutlass_dsl_version=${CUTLASS_DSL_VERSION:-4.6.2}
release_date=${RELEASE_DATE:-20260820}
revision=${REVISION:-r1}
image=${IMAGE:-voipmonitor/vllm:flashinfer-wheels-fi${flashinfer_commit:0:7}-cu133-torch213-${release_date}-${revision}}

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'The FlashInfer artifact recipe must be committed before build.\n' >&2
  git status --short >&2
  exit 1
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi

base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "FLASHINFER_REPO=${flashinfer_repo}" \
  --build-arg "FLASHINFER_REF=${flashinfer_ref}" \
  --build-arg "FLASHINFER_COMMIT=${flashinfer_commit}" \
  --build-arg "FLASHINFER_VERSION=${flashinfer_version}" \
  --build-arg "CUTLASS_DSL_VERSION=${cutlass_dsl_version}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.flashinfer-cu133-torch213-wheels \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
assert_label() {
  local key=$1 expected=$2
  jq -e --arg key "${key}" --arg expected "${expected}" \
    '.[$key] == $expected' <<<"${labels}" >/dev/null || {
      printf 'Image label %s does not match %s\n' "${key}" "${expected}" >&2
      exit 1
    }
}
assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.flashinfer.commit "${flashinfer_commit}"
assert_label local-inference.flashinfer.version "${flashinfer_version}"
assert_label local-inference.cutlass-dsl.version "${cutlass_dsl_version}"

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format 'image={{.Id}} size={{.Size}}'
printf '%s\n' "${image}"
