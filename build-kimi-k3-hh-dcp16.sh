#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r2-20260805}"
DOCKER_COMMIT="$(git -C "${SCRIPT_DIR}" rev-parse HEAD)"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --build-arg "DOCKER_COMMIT=${DOCKER_COMMIT}" \
  --file "${SCRIPT_DIR}/Dockerfile.kimi-k3-hh-dcp16" \
  --tag "${IMAGE}" \
  "${SCRIPT_DIR}"

docker image inspect "${IMAGE}" --format \
  'image={{.Id}} vllm={{index .Config.Labels "local-inference-lab.vllm.commit"}} sparkinfer={{index .Config.Labels "local-inference-lab.sparkinfer.commit"}} base={{index .Config.Labels "local-inference-lab.runtime.base-digest"}}'
