#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo_root}/launchers/serve-glm52-v16.sh"
common_env=(
  DRY_RUN=1
  MODEL=/model
  TP=8
  DCP=1
  MTP=0
  GPUS=0,1,2,3,4,5,6,7
  MAX_NUM_SEQS=1
  GRAPH=6
)

default_output="$(env "${common_env[@]}" "${launcher}")"
grep -Fxq 'VLLM_WORKER_MULTIPROC_METHOD=spawn' <<<"${default_output}"

forkserver_output="$(env "${common_env[@]}" \
  VLLM_WORKER_MULTIPROC_METHOD=forkserver "${launcher}")"
grep -Fxq 'VLLM_WORKER_MULTIPROC_METHOD=forkserver' \
  <<<"${forkserver_output}"

if env "${common_env[@]}" VLLM_WORKER_MULTIPROC_METHOD=fork \
  "${launcher}" >/dev/null 2>&1; then
  echo "Unsafe fork method was unexpectedly accepted" >&2
  exit 1
fi

echo "GLM-5.2 worker multiprocessing policy: PASS"
