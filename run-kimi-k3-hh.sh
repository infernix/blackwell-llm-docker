#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-dcp16-dspark}"
IMAGE="${IMAGE:-voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-20260805}"
HF_CACHE="${HF_CACHE:-/root/.cache/huggingface}"
PORT="${PORT:-8000}"
NAME="${NAME:-kimi-k3-hh-${PROFILE}}"

case "${PROFILE}" in
  dcp16-dspark)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-kda-mxfp8.sh
    ;;
  dcp16-dspark-full)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dspark7-dcp16-1m-hierarchical-bf16x2.sh
    ;;
  dcp16-no-dspark)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark.sh
    ;;
  dcp16-no-dspark-batch8)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dcp16-1m-no-dspark-batch8.sh
    ;;
  dcp8-no-dspark)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dcp8-1m-no-dspark-optimized.sh
    ;;
  dcp8-dspark)
    LAUNCHER=serve-kimi-k3-full-mxfp4-dspark7-dcp8-1m.sh
    ;;
  *)
    echo "Unknown profile: ${PROFILE}" >&2
    echo "Profiles: dcp16-dspark, dcp16-dspark-full, dcp16-no-dspark," >&2
    echo "          dcp16-no-dspark-batch8, dcp8-no-dspark, dcp8-dspark" >&2
    exit 2
    ;;
esac

exec docker run --rm \
  --name "${NAME}" \
  --gpus all \
  --ipc=host \
  --shm-size=64g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --security-opt label=disable \
  -p "${PORT}:8000" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  --entrypoint "/opt/kimi-k3-hh/vllm/${LAUNCHER}" \
  "${IMAGE}"
