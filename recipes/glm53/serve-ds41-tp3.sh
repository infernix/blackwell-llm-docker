#!/usr/bin/env bash
set -euo pipefail

# DS4.1 Flash TP3 serving profile for three RTX PRO 6000 GPUs with host-backed
# expert offload, the lossless prepared-scale codec, deferred WS4 expert weight
# banks, and full routed-expert promotion after the health gate.
# Capacity tier: 524,288-token context, a fixed 512 MiB KV pool per rank, four
# request slots, and a 1,024-token prefill budget, which frees the headroom
# that deferred promotion and the preparation race need.

# The common image supplies GLM tuning. Remove only those inherited values;
# preserve logging, tracing, networking, cache paths and different-value
# overrides. Docker cannot distinguish an explicit value equal to its default.
while read -r name inherited; do
    if [[ ${!name-} == "$inherited" ]]; then unset "$name"; fi
done <<'GLM_IMAGE_DEFAULTS'
VLLM_GLM53_L2_PREFETCH 1
VLLM_GLM53_L2_PREFETCH_PERSIST_MB 0
VLLM_GLM53_DFLASH_ATTN 1
VLLM_CAUSAL_CONV1D_UPDATE_HOIST 1
VLLM_GLM53_KDA_GATE_SIDE_STREAM 1
VLLM_GLM53_MTP_DRAFT_HEAD nvfp4
VLLM_MXFP8_LM_HEAD 0
VLLM_MTP_NVFP4_LM_HEAD 0
VLLM_LM_HEAD_A16 1
VLLM_PCIE_TWOSHOT_ALLREDUCE_MAX_SIZE 768KB
VLLM_GLM53_ONLINE_DENSE_MXFP8 0
VLLM_DISABLE_SHARED_EXPERTS_STREAM 0
VLLM_DISABLED_KERNELS MarlinFP8ScaledMMLinearKernel
VLLM_CPP_AR_1STAGE_NCCL_CUTOFF 56KB
VLLM_CPP_AR_IGNORE_CUTOFF_MAX_ROWS 0
VLLM_SOURCE_OVERLAY_ACTIVE 1
OMP_NUM_THREADS 1
MAX_CUDAGRAPH_CAPTURE_SIZE 256
B12X_DYNAMIC_SPLIT_ROUTE_COMPUTE 1
B12X_DYNAMIC_DIRECT_EXPERT_SCALES 1
B12X_DYNAMIC_SPLIT_LOW_SMEM 1
B12X_DYNAMIC_SKIP_SPLIT_BARRIER_RESET 1
B12X_DYNAMIC_SPLIT_FAST_PREPARE 1
B12X_DYNAMIC_WORK_SOURCE persistent_grid
B12X_DYNAMIC_SPLIT_COMPUTE_MAC 224
B12X_PCIE_ONESHOT_THREADS 512
B12X_PCIE_ONESHOT_BLOCK_LIMIT 4
B12X_PCIE_ONESHOT_PDL 1
B12X_MHC_PDL 1
GLM_IMAGE_DEFAULTS
unset NCCL_GRAPH_FILE
export VLLM_USE_BREAKABLE_CUDAGRAPH=${VLLM_USE_BREAKABLE_CUDAGRAPH:-0}

export PYTHON_BIN=${PYTHON_BIN:-/opt/venv/bin/python}
export MODEL_PATH=${MODEL_PATH:-${MODEL:-deepseek-ai/DeepSeek-V4.1-Flash}}
export HOST=${HOST:-0.0.0.0} PORT=${PORT:-8000}
export TP_SIZE=${TP_SIZE:-3}
export MAX_MODEL_LEN=${MAX_MODEL_LEN:-524288}
export MAX_NUM_SEQS=${MAX_NUM_SEQS:-4}
export MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-1024}
export MAX_CUDAGRAPH_CAPTURE_SIZE=${MAX_CUDAGRAPH_CAPTURE_SIZE:-24}
export GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.93}
export MAX_PARALLEL_PREFILLS=${MAX_PARALLEL_PREFILLS:-1}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
export MODE=${MODE:-off}
export ENGRAM_TABLE_MEMORY=${ENGRAM_TABLE_MEMORY:-disk}
export LOAD_FORMAT=${LOAD_FORMAT:-instanttensor}
export JIT_MONITOR_MODE=${JIT_MONITOR_MODE:-warn}
export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1} NCCL_P2P_LEVEL=${NCCL_P2P_LEVEL:-SYS}
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-lo}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-lo}

# Scale codec: expert prepared scales are stored losslessly compressed
# (E8M0 planes, ~4.5 GiB saved per rank) and decoded straight into the
# shared-memory stage. Deferred weight banks keep every routed expert's
# WS4 weight bank in host RAM at boot and promote them onto the GPU after
# the health gate, under a 1 GiB per-rank reserve.
export B12X_SCALE_CODEC=${B12X_SCALE_CODEC:-1}
export B12X_DEFERRED_WEIGHT_BANKS=${B12X_DEFERRED_WEIGHT_BANKS:-1}
export B12X_DEFERRED_BANK_RESERVE_BYTES=${B12X_DEFERRED_BANK_RESERVE_BYTES:-1073741824}
export B12X_DEFERRED_BANK_ARM_FILE=${B12X_DEFERRED_BANK_ARM_FILE:-/cache/deferred-banks.arm}
export B12X_COMPILE_CACHE_DIR=${B12X_COMPILE_CACHE_DIR:-/cache/b12x-compile}
export VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=${VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE:-256KB}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

# Host-backed decoder-half expert offload with contiguous VMM regions so
# promotion can reuse the same addresses; embedding/vision/aligner towers
# stream from pinned host memory.
export DS41_OFFLOAD_FROM_ORDINAL=${DS41_OFFLOAD_FROM_ORDINAL:-20}
export DS41_OFFLOAD_TO_ORDINAL=${DS41_OFFLOAD_TO_ORDINAL:-22}
export DS41_VMM_ENABLE=${DS41_VMM_ENABLE:-1}
export DS41_VMM_COLD_FRACTION=${DS41_VMM_COLD_FRACTION:-1}
export DS41_ROPE_CACHE_ROWS=${DS41_ROPE_CACHE_ROWS:-524288}
export DS41_EMBEDDING_UVA=${DS41_EMBEDDING_UVA:-1}

case "$ENGRAM_TABLE_MEMORY" in
    ram|disk) ;;
    *) echo 'ENGRAM_TABLE_MEMORY must be ram or disk.' >&2; exit 2 ;;
esac

# LMCache is the single cache owner: engine-driven L1 in host DRAM, L2 off.
# The launcher rejects a duplicated managed option, so each default applies
# only when the caller did not pass the same option.
has_option() {
    local wanted="$1"
    shift
    local argument
    for argument in "$@"; do
        case "${argument//_/-}" in
            "$wanted"|"${wanted}="*) return 0 ;;
        esac
    done
    return 1
}

cache_pairs=(
    "--cache-mode lmcache"
    "--cache-transfer-mode engine_driven"
    "--cache-l1-gib 48"
    "--cache-l1-init-gib 2"
    "--cache-l2-enabled false"
    "--cache-directory /cache/lmcache"
    "--kv-cache-memory-bytes 536870912"
    "--prefill-policy round-robin"
    "--jit-monitor-mode warn"
    "--scheduling-policy priority"
)
cache_args=()
for pair in "${cache_pairs[@]}"; do
    name="${pair%% *}"
    value="${pair#* }"
    if has_option "$name" "$@"; then
        continue
    fi
    cache_args+=("$name" "$value")
done

exec "${VLLM_SOURCE_DIR:-/opt/glm53-flash/vllm}/serve-ds41-flash.sh" \
    "${cache_args[@]}" \
    -- --cpu-offload-gb 8 --cpu-offload-params experts vision aligner "$@"
