#!/usr/bin/env bash
# Build a source-neutral Kimi-K3 ABI and toolchain foundation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}"

source_image="${SOURCE_IMAGE:-voipmonitor/vllm@sha256:e009bb404211c67164f1009bda97823f35578285b6779a7614ed1f97c1f8c338}"
foundation_version="${FOUNDATION_VERSION:-cu133-torch213-fi1ac6942-rust195-20260822-r1}"
image="${IMAGE:-voipmonitor/vllm:kimi-k3-runtime-foundation-${foundation_version}}"

if [[ -n "$(git status --porcelain --untracked-files=all)" ]] \
    && [[ "${ALLOW_DIRTY_BUILD:-0}" != 1 ]]; then
  printf 'The runtime-foundation recipe must be committed before build.\n' >&2
  git status --short >&2
  exit 1
fi

if ! docker image inspect "${source_image}" >/dev/null 2>&1; then
  docker pull "${source_image}"
fi

source_image_id="$(docker image inspect "${source_image}" --format '{{.Id}}')"
docker_commit="$(git rev-parse HEAD)"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "SOURCE_IMAGE=${source_image}" \
  --build-arg "SOURCE_IMAGE_ID=${source_image_id}" \
  --build-arg "DOCKER_COMMIT=${docker_commit}" \
  --build-arg "FOUNDATION_VERSION=${foundation_version}" \
  --file Dockerfile.kimi-k3-runtime-foundation \
  --tag "${image}" \
  .

labels="$(docker image inspect "${image}" --format '{{json .Config.Labels}}')"
jq -e \
  --arg source "${source_image}" \
  --arg source_id "${source_image_id}" \
  '."local-inference.runtime.foundation.source-image" == $source and
   ."local-inference.runtime.foundation.source-image-id" == $source_id and
   ."local-inference.runtime.foundation.source-packages" == "absent" and
   ."local-inference.flashinfer.version" == "0.6.18+cu133" and
   ."local-inference.rust.toolchain" == "1.95"' \
  <<<"${labels}" >/dev/null

docker run --rm --entrypoint /opt/venv/bin/python "${image}" -c \
  'import importlib.metadata as m; names={d.metadata["Name"].lower() for d in m.distributions()}; assert not ({"vllm", "b12x", "lmcache"} & names); assert m.version("flashinfer-python") == "0.6.18+cu133"; assert m.version("instanttensor") == "0.1.9"'

if [[ "${PUSH_IMAGE:-0}" == 1 ]]; then
  docker push "${image}"
fi

docker image inspect "${image}" --format \
  'image={{.Id}} size={{.Size}} entrypoint={{json .Config.Entrypoint}}'
printf '%s\n' "${image}"
