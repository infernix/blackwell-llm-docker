#!/usr/bin/env bash
set -euo pipefail

# This builder consumes committed integration locks by default. Set
# KIMI_K3_COMPOSITION=compose only when refreshing the release artifacts from
# the pinned manifests; review and commit the resulting locks before building
# an image intended for publication.

cd "$(dirname "$0")"

release_name="${RELEASE_NAME:-kimi-k3-hh-runtime-r1}"
release_date="${RELEASE_DATE:-$(date -u +%Y%m%d)}"
revision="${REVISION:-r1}"
composition_mode="${KIMI_K3_COMPOSITION:-reproduce-r1}"

case "${composition_mode}" in
  reproduce-r1)
    composition_root="patches/releases/kimi-k3-hh-runtime-r1"
    ;;
  compose)
    composition_root="patches/generated/kimi-k3-hh-runtime"
    python3 scripts/compose_vllm_release.py \
      manifests/vllm/kimi-k3-hh-runtime.json \
      --output-dir "${composition_root}/vllm" >/dev/null
    python3 scripts/compose_vllm_release.py \
      manifests/b12x/kimi-k3-hh-runtime.json \
      --output-dir "${composition_root}/b12x" >/dev/null
    ;;
  *)
    printf 'KIMI_K3_COMPOSITION must be reproduce-r1 or compose\n' >&2
    exit 2
    ;;
esac

read_lock() {
  local component="$1"
  local prefix="$2"
  local lock="${composition_root}/${component}/integration.lock.json"
  local patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}" || { printf 'Missing composition lock: %s\n' "${lock}" >&2; exit 1; }
  test -f "${patch}" || { printf 'Missing integration patch: %s\n' "${patch}" >&2; exit 1; }
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" | sha256sum -c - >/dev/null

  export "${prefix}_REPO=$(jq -er '.base.repository' "${lock}")"
  export "${prefix}_REF=$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${lock}")"
  export "${prefix}_COMMIT=$(jq -er '.base.commit' "${lock}")"
  export "${prefix}_PATCH_FILE=${composition_root#patches/}/${component}/integration.patch"
  export "${prefix}_PATCH_SHA256=$(jq -er '.result.patch_sha256' "${lock}")"
  export "${prefix}_INTEGRATION_TREE=$(jq -er '.result.tree' "${lock}")"
  export "${prefix}_INTEGRATION_LOCK_SHA256=$(sha256sum "${lock}" | cut -d' ' -f1)"
  export "${prefix}_PRS=$(jq -er '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${lock}")"
}

read_lock vllm VLLM
read_lock b12x B12X

base_image="${BASE_IMAGE:-voipmonitor/vllm@sha256:820181fbbc975cd5291c411cda9771d58fecee1636d916f508f47230df20592b}"
if ! docker image inspect "${base_image}" >/dev/null 2>&1; then
  docker pull "${base_image}"
fi
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

if [[ "${composition_mode}" == reproduce-r1 ]] && [[ -n "$(git status --porcelain --untracked-files=all)" ]]; then
  printf 'Publication build requires a clean repository so the image revision identifies its complete recipe.\n' >&2
  git status --short >&2
  exit 1
fi

cache_fingerprint="vllm${VLLM_INTEGRATION_TREE:0:10}-b12x${B12X_INTEGRATION_TREE:0:10}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-vllm${VLLM_INTEGRATION_TREE:0:7}-b12x${B12X_INTEGRATION_TREE:0:7}-cu132-${release_date}-${revision}}"

printf 'release=%s\n' "${release_name}"
printf 'vllm=%s + %s -> %s\n' "${VLLM_COMMIT}" "${VLLM_PRS}" "${VLLM_INTEGRATION_TREE}"
printf 'b12x=%s + %s -> %s\n' "${B12X_COMMIT}" "${B12X_PRS}" "${B12X_INTEGRATION_TREE}"
printf 'base=%s (%s)\n' "${base_image}" "${base_image_id}"
printf 'image=%s\n' "${image}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "VLLM_REPO=${VLLM_REPO}" \
  --build-arg "VLLM_REF=${VLLM_REF}" \
  --build-arg "VLLM_COMMIT=${VLLM_COMMIT}" \
  --build-arg "VLLM_PATCH_FILE=${VLLM_PATCH_FILE}" \
  --build-arg "VLLM_PATCH_SHA256=${VLLM_PATCH_SHA256}" \
  --build-arg "VLLM_INTEGRATION_TREE=${VLLM_INTEGRATION_TREE}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${VLLM_INTEGRATION_LOCK_SHA256}" \
  --build-arg "VLLM_PRS=${VLLM_PRS}" \
  --build-arg "B12X_REPO=${B12X_REPO}" \
  --build-arg "B12X_REF=${B12X_REF}" \
  --build-arg "B12X_COMMIT=${B12X_COMMIT}" \
  --build-arg "B12X_PATCH_FILE=${B12X_PATCH_FILE}" \
  --build-arg "B12X_PATCH_SHA256=${B12X_PATCH_SHA256}" \
  --build-arg "B12X_INTEGRATION_TREE=${B12X_INTEGRATION_TREE}" \
  --build-arg "B12X_INTEGRATION_LOCK_SHA256=${B12X_INTEGRATION_LOCK_SHA256}" \
  --build-arg "B12X_PRS=${B12X_PRS}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "CACHE_FINGERPRINT=${cache_fingerprint}" \
  --file Dockerfile.kimi-k3-hh-runtime \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
assert_label() {
  local key="$1"
  local expected="$2"
  jq -e --arg key "${key}" --arg expected "${expected}" '.[$key] == $expected' <<<"${labels}" >/dev/null || {
    printf 'Image label %s does not match %s\n' "${key}" "${expected}" >&2
    exit 1
  }
}

assert_label local-inference.docker.commit "${docker_commit}"
assert_label local-inference.runtime.base-id "${base_image_id}"
assert_label local-inference.cache.fingerprint "${cache_fingerprint}"
assert_label local-inference.vllm.commit "${VLLM_COMMIT}"
assert_label local-inference.vllm.integration.prs "${VLLM_PRS}"
assert_label local-inference.vllm.integration.tree "${VLLM_INTEGRATION_TREE}"
assert_label local-inference.b12x.commit "${B12X_COMMIT}"
assert_label local-inference.b12x.integration.prs "${B12X_PRS}"
assert_label local-inference.b12x.integration.tree "${B12X_INTEGRATION_TREE}"

entrypoint="$(docker image inspect "${image}" --format '{{json .Config.Entrypoint}}')"
test "${entrypoint}" = '["/usr/local/bin/serve-kimi-k3-dspark"]'

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
