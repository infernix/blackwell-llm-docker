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
grep -Fxq 'SPARKINFER_NSA_TOPK_SELECTION_POLICY=bounded_compat' \
  <<<"${default_output}"

exact_output="$(env "${common_env[@]}" \
  NSA_TOPK_SELECTION_POLICY=exact "${launcher}")"
grep -Fxq 'SPARKINFER_NSA_TOPK_SELECTION_POLICY=exact' <<<"${exact_output}"

if env "${common_env[@]}" NSA_TOPK_SELECTION_POLICY=invalid \
  "${launcher}" >/dev/null 2>&1; then
  echo 'Invalid indexer selection policy unexpectedly succeeded' >&2
  exit 1
fi

echo 'GLM-5.2 indexer selection policy: PASS'
