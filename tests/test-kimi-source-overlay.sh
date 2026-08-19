#!/usr/bin/env bash
# Verify lazy source selection and fail-closed source identity validation.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
overlay_dir="${repo_root}/runtime/kimi-k3-qsrt/source-overlay"
scratch_dir="$(mktemp -d)"
trap 'rm -rf "${scratch_dir}"' EXIT
missing_source="${scratch_dir}/missing-vllm-source"
source_root="${scratch_dir}/source"
binary_package="${scratch_dir}/binary/vllm"

mkdir -p \
  "${source_root}/vllm/vllm_flash_attn" \
  "${binary_package}/vllm_flash_attn/cute"
printf 'SOURCE_IDENTITY = "composed"\n' >"${source_root}/vllm/__init__.py"
printf '' >"${source_root}/vllm/vllm_flash_attn/__init__.py"
printf 'GENERATED_IDENTITY = "binary"\n' \
  >"${binary_package}/vllm_flash_attn/cute/__init__.py"

output="$({
  PYTHONPATH="${overlay_dir}" \
    VLLM_SOURCE_OVERLAY_ROOT="${missing_source}" \
    python3 -c 'print("inactive overlay did not import vLLM")'
} 2>&1)"
grep -Fxq 'inactive overlay did not import vLLM' <<<"${output}"

if PYTHONPATH="${overlay_dir}" \
    VLLM_SOURCE_OVERLAY_ACTIVE=1 \
    VLLM_SOURCE_OVERLAY_ROOT="${missing_source}" \
    python3 -c 'print("invalid overlay continued")' >/dev/null 2>&1; then
  printf 'An invalid active vLLM source overlay did not abort Python startup\n' >&2
  exit 1
fi

output="$({
  PYTHONPATH="${overlay_dir}" \
    VLLM_SOURCE_OVERLAY_ACTIVE=1 \
    VLLM_SOURCE_OVERLAY_ROOT="${source_root}" \
    VLLM_BINARY_PACKAGE_DIR="${binary_package}" \
    python3 -c 'import sys; assert "vllm" not in sys.modules; import vllm; import vllm.vllm_flash_attn.cute as cute; print(vllm.SOURCE_IDENTITY, cute.GENERATED_IDENTITY)'
} 2>&1)"
grep -Fxq 'composed binary' <<<"${output}"

printf 'Kimi source overlay activation policy: PASS\n'
