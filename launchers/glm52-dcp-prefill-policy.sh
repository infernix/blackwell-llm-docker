#!/usr/bin/env bash

# Resolve only configurations with measured end-to-end gains. Explicit values
# always win so every automatic choice has a deployment kill switch.

# Classify whether a layer-ahead CKV gather can safely overlap the TP traffic
# on the selected PCIe topology. The overlap is counterproductive when TP
# crosses a NUMA boundary or when a DCP ring has no sufficiently local links.
# Input is the tab-separated output of `nvidia-smi topo -m` on stdin.
classify_glm52_ckv_prefetch_topology() {
  local tp="$1"
  local dcp="$2"
  local gpus="$3"

  awk -v tp="${tp}" -v dcp="${dcp}" -v gpu_csv="${gpus}" '
    function trim(value) {
      gsub(/^[[:space:]]+|[[:space:]]+$/, "", value)
      return value
    }

    function unsafe(reason) {
      print "unsafe:" reason
      decided = 1
      exit
    }

    BEGIN {
      FS = "\t"
      if (tp !~ /^[0-9]+$/ || dcp !~ /^[0-9]+$/ || tp < 1 || dcp < 1) {
        unsafe("invalid-parallel-sizes")
      }
      if (tp % dcp != 0) {
        unsafe("dcp-does-not-divide-tp")
      }
      gpu_count = split(gpu_csv, requested_gpus, ",")
      if (gpu_count < tp) {
        unsafe("too-few-gpus")
      }
      for (i = 1; i <= tp; ++i) {
        gpu = trim(requested_gpus[i])
        if (gpu ~ /^[0-9]+$/) {
          selected[i] = "GPU" gpu
        } else if (gpu ~ /^GPU[0-9]+$/) {
          selected[i] = gpu
        } else {
          unsafe("non-numeric-gpu-selection")
        }
      }
    }

    {
      # Strip ANSI underline/color sequences emitted by recent nvidia-smi.
      gsub(/\033\[[0-9;]*m/, "", $0)
      first = trim($1)
      if (!have_header) {
        for (i = 1; i <= NF; ++i) {
          field = trim($i)
          if (field ~ /^GPU[0-9]+$/) {
            gpu_column[field] = i
            ++header_gpu_count
          }
        }
        if (header_gpu_count > 0) {
          have_header = 1
        }
        next
      }
      if (first ~ /^GPU[0-9]+$/) {
        for (gpu in gpu_column) {
          links[first SUBSEP gpu] = trim($(gpu_column[gpu]))
        }
      }
    }

    END {
      if (decided) {
        exit
      }
      if (!have_header) {
        unsafe("topology-unavailable")
      }
      for (i = 1; i <= tp; ++i) {
        if (!(selected[i] in gpu_column)) {
          unsafe("selected-gpu-missing")
        }
      }

      # Concurrent TP traffic crossing SYS links starves the side-stream CKV
      # all-gather even when each DCP subgroup itself remains on one socket.
      for (i = 1; i <= tp; ++i) {
        for (j = i + 1; j <= tp; ++j) {
          if (links[selected[i] SUBSEP selected[j]] == "SYS") {
            unsafe("tp-crosses-numa")
          }
        }
      }

      # NCCL may reorder a ring internally, but the rank-order ring is a
      # conservative proxy. Require at least half of each DCP ring to use a
      # direct switch/NVLink path. PHB/NODE-only groups are kept synchronous.
      for (start = 1; start <= tp; start += dcp) {
        direct = 0
        for (offset = 0; offset < dcp; ++offset) {
          current = start + offset
          next_idx = start + ((offset + 1) % dcp)
          relation = links[selected[current] SUBSEP selected[next_idx]]
          if (relation ~ /^(PIX|PXB|NV[0-9]+)$/) {
            ++direct
          }
        }
        if (direct * 2 < dcp) {
          unsafe("dcp-ring-lacks-local-links")
        }
      }
      print "safe:local-dcp-rings"
    }
  '
}

resolve_glm52_dcp_prefill_policy() {
  local tp="$1"
  local dcp="$2"
  local query_split="$3"
  local ckv_gather="$4"
  local owner_merge="$5"
  local indexer_shards="$6"
  local prefetch_depth="$7"
  local prefetch_overlap_safe="${8:-1}"
  local owner_merge_topology_safe="${9:-1}"

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
    if [[ "${dcp}" == "1" ]]; then
      owner_merge=0
    elif [[ "${tp}:${dcp}" == "8:8" && \
            "${owner_merge_topology_safe}" != "1" ]]; then
      # The exact owner exchange wins on a local PCIe fabric but loses its
      # launch/transport advantage when every rank crosses a NUMA boundary.
      owner_merge=0
    else
      owner_merge=1
    fi
  fi

  if [[ "${indexer_shards}" == "auto" ]]; then
    case "${tp}:${dcp}" in
      8:4) indexer_shards=2 ;;
      8:8) indexer_shards=4 ;;
      *) indexer_shards=0 ;;
    esac
  fi

  if [[ "${prefetch_depth}" == "auto" ]]; then
    if [[ "${ckv_gather}" == "1" && "${prefetch_overlap_safe}" == "1" ]]; then
      prefetch_depth=1
    else
      prefetch_depth=0
    fi
  fi

  printf '%s %s %s %s %s\n' \
    "${query_split}" "${ckv_gather}" "${owner_merge}" \
    "${indexer_shards}" "${prefetch_depth}"
}
