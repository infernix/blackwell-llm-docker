#!/usr/bin/env bash
set -euo pipefail

die() {
  echo "ERROR: $*" >&2
  exit 2
}

TP="${TP:-8}"
DCP="${DCP:-1}"
DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT:-${VLLM_DCP_QUERY_SPLIT:-auto}}"
DCP_CKV_GATHER="${DCP_CKV_GATHER:-${VLLM_B12X_MLA_CKV_GATHER:-auto}}"
DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE:-${VLLM_DCP_TOPK_OWNER_MERGE:-auto}}"
DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS:-${VLLM_DCP_INDEXER_SHARDS:-auto}}"
DCP_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH:-${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH:-auto}}"
DCP_CKV_PREFETCH_WORKSPACE_MIB="${DCP_CKV_PREFETCH_WORKSPACE_MIB:-${VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB:-1024}}"
launcher_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=glm52-dcp-prefill-policy.sh
source "${launcher_dir}/glm52-dcp-prefill-policy.sh"

case "${DCP_QUERY_SPLIT}" in
  auto|0|1) ;;
  *) die "DCP_QUERY_SPLIT must be auto, 0, or 1" ;;
esac

case "${DCP_CKV_GATHER}" in
  auto|0|1) ;;
  *) die "DCP_CKV_GATHER must be auto, 0, or 1" ;;
esac

case "${DCP_TOPK_OWNER_MERGE}" in
  auto|0|1) ;;
  *) die "DCP_TOPK_OWNER_MERGE must be auto, 0, or 1" ;;
esac

[[ "${DCP_INDEXER_SHARDS}" == "auto" || "${DCP_INDEXER_SHARDS}" =~ ^[0-9]+$ ]] || \
  die "DCP_INDEXER_SHARDS must be auto or a non-negative integer"
[[ "${DCP_CKV_PREFETCH_DEPTH}" == "auto" || "${DCP_CKV_PREFETCH_DEPTH}" =~ ^[0-9]+$ ]] || \
  die "DCP_CKV_PREFETCH_DEPTH must be auto or a non-negative integer"
[[ "${DCP_CKV_PREFETCH_WORKSPACE_MIB}" =~ ^[0-9]+$ ]] || \
  die "DCP_CKV_PREFETCH_WORKSPACE_MIB must be a non-negative integer"

read -r \
  DCP_QUERY_SPLIT \
  DCP_CKV_GATHER \
  DCP_TOPK_OWNER_MERGE \
  DCP_INDEXER_SHARDS \
  DCP_CKV_PREFETCH_DEPTH < <(
    resolve_glm52_dcp_prefill_policy \
      "${TP}" \
      "${DCP}" \
      "${DCP_QUERY_SPLIT}" \
      "${DCP_CKV_GATHER}" \
      "${DCP_TOPK_OWNER_MERGE}" \
      "${DCP_INDEXER_SHARDS}" \
      "${DCP_CKV_PREFETCH_DEPTH}"
  )

export VLLM_DCP_QUERY_SPLIT="${DCP_QUERY_SPLIT}"
export VLLM_B12X_MLA_CKV_GATHER="${DCP_CKV_GATHER}"
export VLLM_DCP_TOPK_OWNER_MERGE="${DCP_TOPK_OWNER_MERGE}"
export VLLM_DCP_INDEXER_SHARDS="${DCP_INDEXER_SHARDS}"
export VLLM_B12X_MLA_CKV_PREFETCH_DEPTH="${DCP_CKV_PREFETCH_DEPTH}"
export VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB="${DCP_CKV_PREFETCH_WORKSPACE_MIB}"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  printf 'VLLM_DCP_QUERY_SPLIT=%q\n' "${VLLM_DCP_QUERY_SPLIT}"
  printf 'VLLM_B12X_MLA_CKV_GATHER=%q\n' "${VLLM_B12X_MLA_CKV_GATHER}"
  printf 'VLLM_DCP_TOPK_OWNER_MERGE=%q\n' "${VLLM_DCP_TOPK_OWNER_MERGE}"
  printf 'VLLM_DCP_INDEXER_SHARDS=%q\n' "${VLLM_DCP_INDEXER_SHARDS}"
  printf 'VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=%q\n' "${VLLM_B12X_MLA_CKV_PREFETCH_DEPTH}"
  printf 'VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB=%q\n' "${VLLM_B12X_MLA_CKV_PREFETCH_WORKSPACE_MIB}"
fi

exec /usr/local/bin/serve-glm52-v16.sh "$@"
