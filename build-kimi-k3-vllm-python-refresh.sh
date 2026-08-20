#!/usr/bin/env bash
# Build Kimi-K3 after a source-verified Python-only vLLM change.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

source_root=patches/releases/kimi-k3-upstream-aligned-20260820
vllm_lock="${source_root}/vllm/source.lock.json"
b12x_lock="${source_root}/b12x/source.lock.json"
lmcache_lock="${source_root}/lmcache/source.lock.json"
release_name="${RELEASE_NAME:-kimi-k3-upstream-aligned}"
release_date="${RELEASE_DATE:-20260820}"
revision="${REVISION:-r16}"

vllm_repo="$(jq -er '.source.repository' "${vllm_lock}")"
vllm_ref="$(jq -er '.source.ref | sub("^refs/heads/"; "")' "${vllm_lock}")"
vllm_commit="$(jq -er '.source.commit' "${vllm_lock}")"
vllm_tree="$(jq -er '.source.tree' "${vllm_lock}")"
vllm_base="$(jq -er '.upstream_base.commit' "${vllm_lock}")"
vllm_merge_heads="$(jq -r '[.pull_requests[].head] | join(",")' "${vllm_lock}")"
vllm_prs="$(jq -r '[.pull_requests[] | "\(.number)@\(.head)"] | join(",")' "${vllm_lock}")"
vllm_lock_sha256="$(sha256sum "${vllm_lock}" | cut -d' ' -f1)"
base_image="${BASE_IMAGE:-$(jq -er '.native_artifact_base.image' "${vllm_lock}")}"
base_vllm_commit="$(jq -er '.native_artifact_base.commit' "${vllm_lock}")"
base_vllm_tree="$(jq -er '.native_artifact_base.tree' "${vllm_lock}")"
changed_paths_sha256="$(jq -er '.native_artifact_base.changed_paths_sha256' "${vllm_lock}")"
b12x_tree="$(jq -er '.source.tree' "${b12x_lock}")"
lmcache_tree="$(jq -er '.source.tree' "${lmcache_lock}")"

test -z "$(git status --porcelain --untracked-files=all)" || {
  printf 'Commit the runtime recipe before building.\n' >&2
  git status --short >&2
  exit 1
}

docker image inspect "${base_image}" >/dev/null 2>&1 || docker pull "${base_image}"
base_image_id="$(docker image inspect "${base_image}" --format '{{.Id}}')"
base_labels="$(docker image inspect "${base_image}" --format '{{json .Config.Labels}}')"
jq -e \
  --arg commit "${base_vllm_commit}" \
  --arg tree "${base_vllm_tree}" \
  --arg b12x_tree "${b12x_tree}" \
  --arg lmcache_tree "${lmcache_tree}" \
  '."local-inference.vllm.commit" == $commit and
   ."local-inference.vllm.integration.tree" == $tree and
   ."local-inference.b12x.integration.tree" == $b12x_tree and
   ."local-inference.lmcache.integration.tree" == $lmcache_tree and
   ."local-inference.runtime.source-overlay" == "absent"' \
  <<<"${base_labels}" >/dev/null

source_lock_sha256="$({
  sha256sum "${vllm_lock}" "${b12x_lock}" "${lmcache_lock}"
} | sha256sum | cut -d' ' -f1)"
docker_commit="$(git rev-parse HEAD)"
vllm_package_version="${VLLM_PACKAGE_VERSION:-0.26.1rc0+kimi.k3.aligned.vllm${vllm_tree:0:7}.b12x${b12x_tree:0:7}}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-upstream-aligned-dspark-nativekv-vllm${vllm_tree:0:7}-b12x${b12x_tree:0:7}-cu133-torch213-${release_date}-${revision}}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "BASE_IMAGE=${base_image}" \
  --build-arg "BASE_IMAGE_ID=${base_image_id}" \
  --build-arg "BASE_VLLM_COMMIT=${base_vllm_commit}" \
  --build-arg "BASE_VLLM_TREE=${base_vllm_tree}" \
  --build-arg "VLLM_REPO=${vllm_repo}" \
  --build-arg "VLLM_REF=${vllm_ref}" \
  --build-arg "VLLM_COMMIT=${vllm_commit}" \
  --build-arg "VLLM_TREE=${vllm_tree}" \
  --build-arg "VLLM_UPSTREAM_BASE=${vllm_base}" \
  --build-arg "VLLM_MERGE_HEADS=${vllm_merge_heads}" \
  --build-arg "VLLM_PRS=${vllm_prs}" \
  --build-arg "VLLM_INTEGRATION_LOCK_SHA256=${vllm_lock_sha256}" \
  --build-arg "VLLM_CHANGED_PATHS_SHA256=${changed_paths_sha256}" \
  --build-arg "VLLM_PACKAGE_VERSION=${vllm_package_version}" \
  --build-arg "SOURCE_LOCK_SHA256=${source_lock_sha256}" \
  --build-arg "RELEASE_NAME=${release_name}" \
  --build-arg "RELEASE_DATE=${release_date}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --file Dockerfile.kimi-k3-vllm-python-refresh \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e \
  --arg commit "${vllm_commit}" \
  --arg tree "${vllm_tree}" \
  --arg base_id "${base_image_id}" \
  --arg lock "${source_lock_sha256}" \
  '."local-inference.vllm.commit" == $commit and
   ."local-inference.vllm.integration.tree" == $tree and
   ."local-inference.runtime.native-artifact-base-id" == $base_id and
   ."local-inference.runtime.native-artifact-reuse" == "python-only-diff-verified" and
   ."local-inference.runtime.source-lock.sha256" == $lock and
   ."local-inference.runtime.source-overlay" == "absent"' \
  <<<"${labels}" >/dev/null

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
