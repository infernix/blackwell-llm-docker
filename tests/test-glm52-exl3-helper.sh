#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT

output="$({
  DRY_RUN=1 \
  XDG_CACHE_HOME="${tmp_root}/cache" \
  TMPDIR="${tmp_root}/tmp" \
  MODEL=brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw \
  MODEL_REVISION=9297b9f1d53af5c67cffa01e30cc071a1ff7144b \
  SERVED_MODEL_NAME=GLM-5.2-EXL3-TR3-3.0bpw \
  TP=4 DCP=4 MTP=3 MAX_NUM_SEQS=8 GRAPH=32 \
  MAX_BATCHED_TOKENS=3072 MAX_MODEL_LEN=524288 \
  GPU_MEMORY_UTILIZATION=0.95 MOE_MODE=a16 MOE_BACKEND=b12x \
  MTP_MOE_BACKEND=triton MTP_DRAFT_SAMPLE_METHOD=greedy \
  QUANTIZATION=exl3 ONLINE_QUANT=none LOAD_FORMAT=safetensors \
  KV_CACHE_DTYPE=nvfp4_ds_mla ASYNC_SCHEDULING=0 \
  DCP_KV_CACHE_INTERLEAVE_SIZE=1 F8_DMA=0 \
  bash "${repo_root}/launchers/serve-glm52-v16.sh"
} 2>&1)"

grep -Fq 'glm52-exl3|exl3)' \
  "${repo_root}/launchers/serve-gilded-gnosis.sh"
grep -Fq 'brandonmusic/GLM-5.2-EXL3-TR3-3.0bpw' \
  "${repo_root}/launchers/serve-gilded-gnosis.sh"

grep -Fq 'QUANTIZATION=exl3' <<<"${output}"
grep -Fq 'ONLINE_QUANT=none' <<<"${output}"
grep -Fq 'KV_CACHE_DTYPE=nvfp4_ds_mla' <<<"${output}"
grep -Fq 'ASYNC_SCHEDULING=0' <<<"${output}"
grep -Fq 'MTP_MOE_BACKEND=triton' <<<"${output}"
grep -Fq 'MTP_DRAFT_SAMPLE_METHOD=greedy' <<<"${output}"
grep -Fq 'DCP_KV_CACHE_INTERLEAVE_SIZE=1' <<<"${output}"
grep -Fq -- '--tensor-parallel-size 4' <<<"${output}"
grep -Fq -- '--decode-context-parallel-size 4' <<<"${output}"
grep -Fq -- '--quantization exl3' <<<"${output}"
grep -Fq -- '--load-format safetensors' <<<"${output}"
grep -Fq -- '--no-async-scheduling' <<<"${output}"
grep -Fq -- '--max-num-seqs 8' <<<"${output}"
grep -Fq -- '--max-num-batched-tokens 3072' <<<"${output}"
grep -Fq -- '--max-model-len 524288' <<<"${output}"
grep -Fq -- '\"moe_backend\":\"triton\"' <<<"${output}"
grep -Fq -- '\"draft_sample_method\":\"greedy\"' <<<"${output}"
grep -Fq -- '\"cudagraph_capture_sizes\":\[4\,8\,12\,16\,20\,24\,28\,32\]' <<<"${output}"
grep -Fq -- '\"custom_ops\":\[\"all\"\]' <<<"${output}"

if grep -Fq -- '--quantization-config' <<<"${output}"; then
  echo 'EXL3 preset unexpectedly enables online quantization' >&2
  exit 1
fi
if grep -Fq 'VLLM_EXL3_ABI_SHIM=' <<<"${output}"; then
  echo 'EXL3 preset unexpectedly requires an ABI shim' >&2
  exit 1
fi

echo 'GLM-5.2 EXL3 helper: PASS'
