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
  MOE_MODE=a16
  ONLINE_QUANT=mxfp8
  GPUS=0,1,2,3,4,5,6,7
  MAX_NUM_SEQS=1
  GRAPH=6
)

default_output="$(env "${common_env[@]}" "${launcher}")"
grep -Fxq 'QUANTIZATION_CONFIG_JSON=\{\"linear\":\{\"weight\":\"mxfp8\"\}\}' \
  <<<"${default_output}"
if grep -q 'kv_b_proj' <<<"${default_output}"; then
  echo "Default MXFP8 policy unexpectedly excludes kv_b_proj" >&2
  exit 1
fi

explicit_config='{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}'
explicit_output="$(env "${common_env[@]}" \
  QUANTIZATION_CONFIG_JSON="${explicit_config}" "${launcher}")"
grep -Fq 're:.\*kv_b_proj' <<<"${explicit_output}"

echo "GLM-5.2 online quantization policy: PASS"
