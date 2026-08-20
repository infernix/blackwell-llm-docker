#!/usr/bin/env bash
# Compile Kimi-K3 from verified Infernal Invocation and B12X merge stacks.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

source_root=patches/releases/kimi-k3-upstream-aligned-20260820
release_name="${RELEASE_NAME:-kimi-k3-upstream-aligned}"
release_date="${RELEASE_DATE:-20260820}"
revision="${REVISION:-r1}"
base_image="${BASE_IMAGE:-voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2}"

read_source_lock() {
  local component=$1 prefix=$2
  local lock="${source_root}/${component}/source.lock.json"
  test -f "${lock}" || {
    printf 'Missing source lock: %s\n' "${lock}" >&2
    exit 1
  }
  jq -e \
    '.schema_version == 1 and .component == $component and
     (.source_patches | length) == 0' \
    --arg component "${component}" "${lock}" >/dev/null

  export "${prefix}_REPO=$(jq -er '.source.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.source.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.source.commit' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.source.tree' "${lock}")"
  export "${prefix}_UPSTREAM_BASE=$(jq -er '.upstream_base.commit' "${lock}")"
  export "${prefix}_MERGE_HEADS=$(jq -r '[.pull_requests[].head] | join(",")' "${lock}")"
  export "${prefix}_PRS=$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
}

read_source_lock vllm VLLM
read_source_lock b12x B12X
read_source_lock lmcache LMCACHE

test "${VLLM_UPSTREAM_BASE}" = 337ef76dcd30198d8dd47f6c9e61ae1d8be73656
test "${B12X_UPSTREAM_BASE}" = c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5
test "${LMCACHE_UPSTREAM_BASE}" = "${LMCACHE_COMMIT}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'The clean runtime recipe must be committed before build.\n' >&2
  git status --short >&2
  exit 1
fi

if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi

base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"
source_lock_sha256="$({
  sha256sum \
    "${source_root}/vllm/source.lock.json" \
    "${source_root}/b12x/source.lock.json" \
    "${source_root}/lmcache/source.lock.json"
} | sha256sum | cut -d' ' -f1)"
cache_fingerprint="cu133-torch213-kimi-k3-clean-vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}-lmcache${LMCACHE_INTEGRATION_TREE:0:10}"
vllm_package_version="${VLLM_PACKAGE_VERSION:-0.26.1rc0+kimi.k3.aligned.vllm${VLLM_INTEGRATION_TREE:0:7}.b12x${B12X_INTEGRATION_TREE:0:7}}"
core_image="${CORE_IMAGE:-voipmonitor/vllm:kimi-k3-upstream-aligned-core-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-lmcache-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu133-torch213-${release_date}-${revision}}"

printf 'base=%s id=%s\n' "${base_image}" "${base_image_id}"
printf 'vllm=%s tree=%s base=%s prs=%s\n' \
  "${VLLM_COMMIT}" "${VLLM_INTEGRATION_TREE}" "${VLLM_UPSTREAM_BASE}" "${VLLM_PRS}"
printf 'b12x=%s tree=%s base=%s prs=%s\n' \
  "${B12X_COMMIT}" "${B12X_INTEGRATION_TREE}" "${B12X_UPSTREAM_BASE}" "${B12X_PRS}"
printf 'lmcache=%s tree=%s\n' "${LMCACHE_COMMIT}" "${LMCACHE_INTEGRATION_TREE}"
printf 'core_image=%s\nimage=%s\n' "${core_image}" "${image}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg 'VLLM_PATCH_FILE=' \
  --build-arg 'VLLM_PATCH_SHA256=' \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "VLLM_UPSTREAM_BASE=${VLLM_UPSTREAM_BASE}" \
  --build-arg "VLLM_MERGE_HEADS=${VLLM_MERGE_HEADS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg 'B12X_PATCH_FILE=' \
  --build-arg 'B12X_PATCH_SHA256=' \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "B12X_UPSTREAM_BASE=${B12X_UPSTREAM_BASE}" \
  --build-arg "B12X_MERGE_HEADS=${B12X_MERGE_HEADS}" \
  --build-arg "LMCACHE_REPO=${LMCACHE_REPO}" \
  --build-arg "LMCACHE_REF=${LMCACHE_REF}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg 'LMCACHE_PATCH_FILE=' \
  --build-arg 'LMCACHE_PATCH_SHA256=' \
  --build-arg "LMCACHE_INTEGRATION_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_INTEGRATION_LOCK_SHA256=${LMCACHE_INTEGRATION_LOCK_SHA256}" \
  --build-arg "LMCACHE_PRS=${LMCACHE_PRS}" \
  --build-arg "LMCACHE_UPSTREAM_BASE=${LMCACHE_UPSTREAM_BASE}" \
  --build-arg "LMCACHE_MERGE_HEADS=${LMCACHE_MERGE_HEADS}" \
  --build-arg 'LMCACHE_BUILD_VERSION=0.5.2+glm52dcp.5' \
  --build-arg 'INSTANTTENSOR_REPO=https://github.com/voipmonitor/InstantTensor.git' \
  --build-arg 'INSTANTTENSOR_COMMIT=49b4010afc1cae0441e71fe0b0bffc24fa05e932' \
  --build-arg 'INSTANTTENSOR_LIBAIO_REPO=https://pagure.io/libaio.git' \
  --build-arg 'INSTANTTENSOR_LIBAIO_COMMIT=1b18bfafc6a2f7b9fa2c6be77a95afed8b7be448' \
  --build-arg 'INSTANTTENSOR_LIBAIO_TREE=c9442e111b747e9329ea782c6edb9d13a827cc08' \
  --build-arg 'CUTLASS_DSL_VERSION=4.6.2' \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg 'FLASHINFER_VERSION=0.6.18+cu133' \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --build-arg "RELEASE_NAME=${release_name}-core" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.deepseek-infernal-invocation-cu133-torch213 \
  --tag "${core_image}" \
  .

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${core_image}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "LMCACHE_COMMIT=${LMCACHE_COMMIT}" \
  --build-arg "LMCACHE_TREE=${LMCACHE_INTEGRATION_TREE}" \
  --file Dockerfile.kimi-k3-production-clean-runtime \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e --arg expected "${source_lock_sha256}" \
  '."local-inference.runtime.source-lock.sha256" == $expected and
   ."local-inference.runtime.source-mode" == "compiled-installed-package" and
   ."local-inference.runtime.source-overlay" == "absent"' \
  <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${image}" -c \
  'import pathlib, vllm, b12x, lmcache; root=pathlib.Path("/opt/venv/lib/python3.12/site-packages"); assert pathlib.Path(vllm.__file__).resolve().is_relative_to(root); assert pathlib.Path(b12x.__file__).resolve().is_relative_to(root); assert pathlib.Path(lmcache.__file__).resolve().is_relative_to(root)'

if [[ "${RUN_NCCL_SMOKE:-0}" == 1 ]]; then
  docker run --rm --gpus all --ipc=host \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    --entrypoint torchrun "${image}" \
    --standalone --nproc-per-node=16 \
    /opt/local-inference/torch_nccl_smoke.py
fi

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
