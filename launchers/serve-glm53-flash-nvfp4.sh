#!/usr/bin/env bash
set -euo pipefail

# NCCL interprets an empty graph-file value as a path. This single-node TP
# launcher does not use an external NCCL topology file.
unset NCCL_GRAPH_FILE

model=${MODEL:-local-inference-lab/GLM-5.3-Flash-NVFP4}
if (($# > 0)) && [[ "$1" != -* ]]; then
  model=$1
  shift
fi

served_model_name=${SERVED_MODEL_NAME:-GLM-5.3-Flash-NVFP4}
host=${HOST:-0.0.0.0}
port=${PORT:-8000}
tp=${TP:-4}
max_num_seqs=${MAX_NUM_SEQS:-16}
max_model_len=${MAX_MODEL_LEN:-262144}
max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS:-4096}
gpu_memory_utilization=${GPU_MEMORY_UTILIZATION:-0.90}
load_format=${LOAD_FORMAT:-instanttensor}
num_speculative_tokens=${NUM_SPECULATIVE_TOKENS:-${MTP:-0}}
speculator=${SPECULATOR:-mtp}
dflash_model=${DFLASH_MODEL:-incoai/GLM-5.3-Flash-DFlash2}
attention_backend=${ATTENTION_BACKEND:-B12X}
moe_backend=${MOE_BACKEND:-b12x}
linear_backend=${LINEAR_BACKEND:-b12x}
mtp_attention_backend=${MTP_ATTENTION_BACKEND:-B12X}
# The checkpoint's MTP expert is MXFP8. B12X only handles the target model's
# NVFP4 experts, while Humming implements the MTP MXFP8 1x32 quantization.
mtp_moe_backend=${MTP_MOE_BACKEND:-humming}
disable_custom_all_reduce=${DISABLE_CUSTOM_ALL_REDUCE:-1}

if [[ ! "${num_speculative_tokens}" =~ ^[0-9]+$ ]]; then
  printf 'NUM_SPECULATIVE_TOKENS/MTP must be a non-negative integer; got %s\n' \
    "${num_speculative_tokens}" >&2
  exit 2
fi

case "${speculator}" in
  mtp | dflash) ;;
  *)
    printf 'SPECULATOR must be mtp or dflash; got %s\n' "${speculator}" >&2
    exit 2
    ;;
esac

# A value of zero keeps B12X NVFP4 routed experts on the W4A4 path. Set the
# variable to one only when BF16 activations are required for compatibility.
export VLLM_B12X_MOE_FP4_FORCE_A16="${VLLM_B12X_MOE_FP4_FORCE_A16:-0}"
export VLLM_PLUGINS=

cmd=(
  /opt/venv/bin/vllm serve "${model}"
  --served-model-name "${served_model_name}"
  --host "${host}"
  --port "${port}"
  --tensor-parallel-size "${tp}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size 1
  --max-num-seqs "${max_num_seqs}"
  --max-model-len "${max_model_len}"
  --max-num-batched-tokens "${max_num_batched_tokens}"
  --gpu-memory-utilization "${gpu_memory_utilization}"
  --mamba-cache-mode align
  --enable-chunked-prefill
  --dtype bfloat16
  --kv-cache-dtype fp8
  --quantization modelopt_mixed
  --block-size 256
  --load-format "${load_format}"
  --attention-backend "${attention_backend}"
  --moe-backend "${moe_backend}"
  --linear-backend "${linear_backend}"
  --no-enable-flashinfer-autotune
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
)

case "${disable_custom_all_reduce}" in
  0) ;;
  1) cmd+=(--disable-custom-all-reduce) ;;
  *)
    printf 'DISABLE_CUSTOM_ALL_REDUCE must be 0 or 1; got %s\n' \
      "${disable_custom_all_reduce}" >&2
    exit 2
    ;;
esac

case "${ENABLE_PREFIX_CACHING:-1}" in
  0) ;;
  1) cmd+=(--enable-prefix-caching) ;;
  *)
    printf 'ENABLE_PREFIX_CACHING must be 0 or 1; got %s\n' \
      "${ENABLE_PREFIX_CACHING}" >&2
    exit 2
    ;;
esac

if ((num_speculative_tokens > 0)); then
  case "${speculator}" in
    mtp)
      cmd+=(
        --speculative-config
        "{\"method\":\"mtp\",\"num_speculative_tokens\":${num_speculative_tokens},\"moe_backend\":\"${mtp_moe_backend}\",\"attention_backend\":\"${mtp_attention_backend}\"}"
      )
      ;;
    dflash)
      cmd+=(
        --speculative-config
        "{\"method\":\"dflash\",\"model\":\"${dflash_model}\",\"num_speculative_tokens\":${num_speculative_tokens},\"kv_cache_dtype\":\"auto\"}"
      )
      ;;
  esac
fi

cmd+=("$@")

if [[ "${DRY_RUN:-0}" == 1 ]]; then
  printf 'GLM-5.3-Flash NVFP4 launch:'
  printf ' %q' "${cmd[@]}"
  printf '\n'
  exit 0
fi

exec "${cmd[@]}"
