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
grep -Fxq 'VLLM_B12X_ABSORB_BMM=1' <<<"${default_output}"

exl3_output="$(env "${common_env[@]}" QUANTIZATION=exl3 "${launcher}")"
grep -Fq 'q_a_proj' <<<"${exl3_output}"
grep -Fq 'kv_a_proj_with_mqa' <<<"${exl3_output}"
grep -Fq 'lm_head' <<<"${exl3_output}"
if grep -Fq 'shared_experts' <<<"${exl3_output}"; then
  echo "Default EXL3 MXFP8 policy unexpectedly quantizes shared experts" >&2
  exit 1
fi

exl3_native_output="$(env "${common_env[@]}" QUANTIZATION=exl3 \
  ONLINE_QUANT=none "${launcher}")"
grep -Fxq 'ONLINE_QUANT=none' <<<"${exl3_native_output}"
grep -Fxq "QUANTIZATION_CONFIG_JSON=''" <<<"${exl3_native_output}"
if grep -Fq 'q_a_proj' <<<"${exl3_native_output}"; then
  echo "Native EXL3 policy unexpectedly enabled the MXFP8 overlay" >&2
  exit 1
fi

exl3_explicit_config='{"linear":{"weight":"mxfp8"},"shared_experts":{"weight":"mxfp8"}}'
exl3_explicit_output="$(env "${common_env[@]}" QUANTIZATION=exl3 \
  QUANTIZATION_CONFIG_JSON="${exl3_explicit_config}" "${launcher}")"
grep -Fq 'shared_experts' <<<"${exl3_explicit_output}"
if grep -Fq 'q_a_proj' <<<"${exl3_explicit_output}"; then
  echo "Explicit EXL3 MXFP8 policy was unexpectedly replaced" >&2
  exit 1
fi

if exl3_fp8_output="$(env "${common_env[@]}" QUANTIZATION=exl3 \
  ONLINE_QUANT=fp8 "${launcher}" 2>&1)"; then
  echo "EXL3 unexpectedly accepted the static block-FP8 online overlay" >&2
  exit 1
fi
grep -Fq 'EXL3 online overlays support MXFP8 weights' <<<"${exl3_fp8_output}"

native_output="$(env "${common_env[@]}" ONLINE_QUANT=none "${launcher}")"
grep -Fxq 'VLLM_B12X_ABSORB_BMM=0' <<<"${native_output}"

explicit_config='{"linear":{"weight":"mxfp8"},"ignore":["re:.*kv_b_proj"]}'
explicit_output="$(env "${common_env[@]}" \
  QUANTIZATION_CONFIG_JSON="${explicit_config}" "${launcher}")"
grep -Fq 're:.\*kv_b_proj' <<<"${explicit_output}"

disabled_output="$(env "${common_env[@]}" \
  VLLM_B12X_ABSORB_BMM=0 "${launcher}")"
grep -Fxq 'VLLM_B12X_ABSORB_BMM=0' <<<"${disabled_output}"

for dma_mode in i8_ring mx_ring; do
  dma_output="$(env "${common_env[@]}" F8_DMA="${dma_mode}" "${launcher}")"
  grep -Fxq "VLLM_PCIE_DMA_FP8=${dma_mode}" <<<"${dma_output}"
  grep -Fxq "SPARKINFER_PCIE_DMA_FP8=${dma_mode}" <<<"${dma_output}"
done

echo "GLM-5.2 online quantization policy: PASS"
