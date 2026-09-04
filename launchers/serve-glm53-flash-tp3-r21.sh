#!/usr/bin/env bash
set -euo pipefail

# Strict GLM-5.3-Flash TP3 runtime policy (R21 launcher chain).
#
# This launcher mirrors the semantic intent of the TP3 runtime work done on
# older bases (PR #30 / the R17-era TP3 child) but is built on the R21
# cache-complete chain. It is fail-closed: every value that the TP3 runtime
# was shaped around is locked, and any caller attempt to override it is
# rejected before a container is started. Structural support only; the policy
# is recorded here without qualification claims until a hardware run.

readonly capture_launcher=/usr/local/bin/serve-glm53-flash-nvfp4-dflash2.sh

fail() {
  printf '%s\n' "$1" >&2
  exit 2
}

require_positive_integer() {
  local name=$1
  local value=$2
  [[ ${value} =~ ^[1-9][0-9]*$ ]] ||
    fail "${name} must be a positive integer; got ${value}"
}

# TP=3 is the entire point of this launcher; reject anything else up front.
case "${TP:-}" in
  3) ;;
  *) fail "R21 TP3 policy requires TP=3; got ${TP:-unset}" ;;
esac

case ${CACHE_MODE:-vram} in
  vram)
    if [[ ${LMCACHE_ENABLED:-0} != 0 ]]; then
      fail 'R21 TP3 dense-cache policy does not support LMCache'
    fi
    ;;
  native | lmcache)
    fail "R21 TP3 policy supports dense GPU cache only; got CACHE_MODE=${CACHE_MODE}"
    ;;
  *)
    fail "CACHE_MODE must be vram, native, or lmcache; got ${CACHE_MODE}"
    ;;
esac

if [[ ${LMCACHE_ENABLED:-0} != 0 ]]; then
  fail 'R21 TP3 dense-cache policy does not support LMCache'
fi
if [[ ${DCP:-1} != 1 ]]; then
  fail "R21 TP3 policy requires DCP=1; got ${DCP:-unset}"
fi
if [[ ${MM_ENCODER_TP_MODE:-weights} != weights ]]; then
  fail "R21 TP3 requires MM_ENCODER_TP_MODE=weights; got ${MM_ENCODER_TP_MODE:-unset}"
fi
if (($# > 0)) && [[ $1 != -* ]]; then
  fail "R21 TP3 target model is locked; positional model override $1 is forbidden"
fi
if [[ -v VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE ||
      -v VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE ]]; then
  fail 'R21 TP3 dense cache rejects split-page block-size variables'
fi

# Reject command-line values for settings owned by this launcher's
# environment contract. Emitting a setting twice lets argument order silently
# select a configuration the lock did not validate.
readonly -a locked_long_options=(
  --revision
  --speculative-config
  --moe-backend
  --model
  --config
  --enforce-eager
  --tensor-parallel-size
  --pipeline-parallel-size
  --decode-context-parallel-size
  --cp-kv-cache-interleave-size
  --dcp-kv-cache-interleave-size
  --mamba-cache-mode
  --mm-encoder-tp-mode
  --enable-expert-parallel
  --disable-expert-parallel
  --disable-custom-all-reduce
  --cudagraph-capture-sizes
  --max-model-len
  --max-num-seqs
  --max-num-batched-tokens
  --max-cudagraph-capture-size
  --gpu-memory-utilization
  --kv-cache-memory-bytes
  --num-gpu-blocks-override
  --kv-cache-dtype
  --additional-config
  --compilation-config
  --kv-transfer-config
  --kv-offloading-size
  --kv-offloading-backend
  -tp
  -dcp
  -pp
  -ep
  -sc
  -cc
)

is_locked_override() {
  local option=$1 locked
  case "${option}" in
    --speculative-config.* | --additional-config.* | \
    --compilation-config.* | --kv-transfer-config.* | \
    -tp | -dcp | -pp | -sc | -sc.* | -cc | -cc.* | -ep | -ep.*)
      return 0
      ;;
  esac
  for locked in "${locked_long_options[@]}"; do
    # FlexibleArgumentParser accepts unique long-option abbreviations. Reject
    # every prefix too, before a later caller argument can override the lock.
    if [[ ${locked} == "${option}"* ]]; then
      return 0
    fi
  done
  return 1
}

for argument in "$@"; do
  option=${argument%%=*}
  option=${option//_/-}
  if is_locked_override "${option}"; then
    fail "R21 TP3 policy rejects caller override ${argument}"
  fi
done

lock_env() {
  local name=$1 expected=$2
  if [[ -v ${name} && ${!name} != "${expected}" ]]; then
    fail "R21 TP3 ${name} is locked to ${expected}; got ${!name}"
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}

lock_env_from_parent() {
  local name=$1 expected=$2 parent_default=$3
  if [[ -v ${name} &&
        ${!name} != "${expected}" &&
        ${!name} != "${parent_default}" ]]; then
    fail "R21 TP3 ${name} is locked to ${expected}; got ${!name}"
  fi
  printf -v "${name}" '%s' "${expected}"
  export "${name}"
}

readonly locked_model=local-inference-lab/GLM-5.3-Flash-NVFP4
readonly locked_dflash_model=local-inference-lab/GLM-5.3-Flash-DFlash2
readonly locked_model_revision=378ca54585c46542bad1f3cb3ed0d73ae51cdb62
readonly locked_dflash_revision=aea0ac8a05624512ca9e106c09c16087da998426
lock_env TP 3
lock_env DCP 1
lock_env MM_ENCODER_TP_MODE weights
lock_env MODEL "${locked_model}"
lock_env DFLASH_MODEL "${locked_dflash_model}"
lock_env MODEL_REVISION "${locked_model_revision}"
lock_env DFLASH_MODEL_REVISION "${locked_dflash_revision}"
lock_env LMCACHE_ENABLED 0
lock_env GLM53_CACHE_LAYOUT dense
lock_env CP_KV_CACHE_INTERLEAVE_SIZE 4
lock_env DCP_CKV_GATHER 0
lock_env MAX_MODEL_LEN 1048576
lock_env_from_parent MAX_NUM_SEQS 8 32
lock_env_from_parent MAX_NUM_BATCHED_TOKENS 8192 4096
lock_env_from_parent PREFILL_SCHEDULE_INTERVAL 8 1
lock_env GPU_MEMORY_UTILIZATION 0.91
lock_env_from_parent MAX_CUDAGRAPH_CAPTURE_SIZE 16 256
lock_env_from_parent CUDAGRAPH_CAPTURE_SIZES '1 2 4 8 16' \
  '1 2 4 8 16 32 40 48 64 96 128 192 256'
lock_env_from_parent FAIRNESS_ENGINE none compute_share
lock_env_from_parent PREFILL_COMPUTE_SHARE none 0.4
lock_env KV_CACHE_DTYPE fp8
lock_env ATTENTION_BACKEND B12X
lock_env MOE_BACKEND auto
lock_env LINEAR_BACKEND b12x
lock_env MTP_ATTENTION_BACKEND B12X
lock_env MTP_MOE_BACKEND humming
lock_env VLLM_B12X_MOE_FP4_FORCE_A16 0
lock_env B12X_PCIE_ALLREDUCE 1
lock_env VLLM_ENABLE_PCIE_ALLREDUCE 1
lock_env VLLM_PCIE_ALLREDUCE_BACKEND b12x
lock_env VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE 84KB
lock_env B12X_PCIE_ALLREDUCE_ALGORITHM auto
lock_env GLM53_KDA_DECODE_BACKEND b12x
lock_env GLM53_KDA_PREFILL_BACKEND flashkda
lock_env CUDAGRAPH_MODE FULL
lock_env ENABLE_PREFIX_CACHING 1
lock_env GLM53_R17_REQUIRE_RUNTIME_PROOF 1

require_positive_integer CP_KV_CACHE_INTERLEAVE_SIZE \
  "${CP_KV_CACHE_INTERLEAVE_SIZE}"
require_positive_integer MAX_NUM_SEQS "${MAX_NUM_SEQS}"
require_positive_integer MAX_NUM_BATCHED_TOKENS "${MAX_NUM_BATCHED_TOKENS}"
require_positive_integer PREFILL_SCHEDULE_INTERVAL \
  "${PREFILL_SCHEDULE_INTERVAL}"
case "${GPU_MEMORY_UTILIZATION}" in
  0.91) ;;
  *) fail "R21 TP3 GPU_MEMORY_UTILIZATION is locked to 0.91; got ${GPU_MEMORY_UTILIZATION}" ;;
esac

# Dense (GPU-only) KV cache: dense target and auto recurrent pages, exactly
# the shape the base launcher resolves on its own for DCP=1. No external cache
# connector and no split-page geometry overrides.
export VLLM_GLM53_SPLIT_TARGET_BLOCK_SIZE=2048
export VLLM_GLM53_SPLIT_MAMBA_BLOCK_SIZE=auto
export LMCACHE_VLLM_KV_CACHE_DTYPE=fp8
export LMCACHE_KV_CACHE_DTYPE=fp8_ds_mla

# The dense-cache fingerprint separates the TP3 compiled artifacts from the
# qualified TP4/TP8 caches so a TP3 warmup never invalidates the R21 layout.
readonly fingerprint=cu133-torch213-glm53-r21-tp3-vllmf2d77086-b12x1e59a1f-dense-ctx1m-seq8-bt8192
readonly cache_root=/cache/jit/${fingerprint}
export LOCAL_INFERENCE_CACHE_FINGERPRINT=${fingerprint}
export XDG_CACHE_HOME=${cache_root}
export VLLM_CACHE_ROOT=${cache_root}/vllm
export VLLM_CACHE_DIR=${cache_root}/vllm
export TRITON_CACHE_DIR=${cache_root}/triton
export TORCH_EXTENSIONS_DIR=${cache_root}/torch-extensions
export TORCHINDUCTOR_CACHE_DIR=${cache_root}/torchinductor
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR=${cache_root}/flashinfer-autotune
export FLASHINFER_WORKSPACE_BASE=${cache_root}/flashinfer
export TVM_FFI_CACHE_DIR=${cache_root}/tvm-ffi
export TVM_CACHE_DIR=${cache_root}/tvm
export TILELANG_CACHE_DIR=${cache_root}/tilelang
export CUTE_DSL_CACHE_DIR=${cache_root}/cute-dsl
export B12X_CUTE_COMPILE_CACHE_DIR=${cache_root}/b12x/cute
export B12X_COMPILE_CACHE_DIR=${cache_root}/b12x/compile
export SPARKINFER_COMPILE_CACHE_DIR=${cache_root}/b12x/compile
export DG_JIT_CACHE_DIR=${cache_root}/deep-gemm
export CUDA_CACHE_PATH=${cache_root}/cuda
export CUPY_CACHE_DIR=${cache_root}/cupy

exec "${capture_launcher}" "$@" \
  --enable-expert-parallel \
  --mm-encoder-tp-mode weights
