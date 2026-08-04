#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-dspark-dcp16-pr238-pr118-20260804}"

DOCKER_BUILDKIT=1 docker build \
  --pull=false \
  --file "${SCRIPT_DIR}/Dockerfile.kimi-k3-hh-dcp16" \
  --tag "${IMAGE}" \
  "${SCRIPT_DIR}"

docker image inspect "${IMAGE}" --format \
  'image={{.Id}} vllm={{index .Config.Labels "local-inference-lab.vllm.commit"}} sparkinfer={{index .Config.Labels "local-inference-lab.sparkinfer.commit"}} base={{index .Config.Labels "local-inference-lab.runtime.base-digest"}}'
