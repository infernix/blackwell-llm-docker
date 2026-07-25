#!/usr/bin/env bash

# Resolve only configurations with measured end-to-end gains. Explicit values
# always win so every automatic choice has a deployment kill switch.
resolve_glm52_dcp_prefill_policy() {
  local tp="$1"
  local dcp="$2"
  local query_split="$3"
  local ckv_gather="$4"
  local owner_merge="$5"
  local indexer_shards="$6"
  local prefetch_depth="$7"

  case "${tp}:${dcp}" in
    4:1|8:1)
      [[ "${query_split}" == "auto" ]] && query_split=1
      [[ "${ckv_gather}" == "auto" ]] && ckv_gather=0
      ;;
    4:2|4:4|8:2|8:4|8:8)
      [[ "${query_split}" == "auto" ]] && query_split=1
      [[ "${ckv_gather}" == "auto" ]] && ckv_gather=1
      ;;
    *)
      [[ "${query_split}" == "auto" ]] && query_split=0
      [[ "${ckv_gather}" == "auto" ]] && ckv_gather=0
      ;;
  esac

  if [[ "${owner_merge}" == "auto" ]]; then
    [[ "${dcp}" == "1" ]] && owner_merge=0 || owner_merge=1
  fi

  if [[ "${indexer_shards}" == "auto" ]]; then
    case "${tp}:${dcp}" in
      8:4) indexer_shards=2 ;;
      8:8) indexer_shards=4 ;;
      *) indexer_shards=0 ;;
    esac
  fi

  if [[ "${prefetch_depth}" == "auto" ]]; then
    [[ "${ckv_gather}" == "1" ]] && prefetch_depth=1 || prefetch_depth=0
  fi

  printf '%s %s %s %s %s\n' \
    "${query_split}" "${ckv_gather}" "${owner_merge}" \
    "${indexer_shards}" "${prefetch_depth}"
}
