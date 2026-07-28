#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Every new GG image is composed from the current clean dev/gilded-gnosis head
# plus the exact PR heads in the versioned manifest. Historical release modes
# are explicit and use immutable source artifacts.
composition_mode="${VLLM_RELEASE_COMPOSITION:-clean}"
release_date="${RELEASE_DATE:-$(date -u +%Y%m%d)}"

configure_vllm_composition() {
  local composition_dir="$1"
  local verify_base_head="$2"
  local composition_lock="${composition_dir}/integration.lock.json"

  export VLLM_REPO="$(jq -er '.base.repository' "${composition_lock}")"
  export VLLM_REF="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${composition_lock}")"
  export VLLM_COMMIT="$(jq -er '.base.commit' "${composition_lock}")"
  export VLLM_PATCH_FILE="${composition_dir#patches/}/integration.patch"
  export VLLM_PATCH_SHA256="$(jq -er '.result.patch_sha256' "${composition_lock}")"
  export VLLM_INTEGRATION_LOCK_FILE="${composition_lock}"
  export VLLM_INTEGRATION_TREE="$(jq -er '.result.tree' "${composition_lock}")"
  export REQUIRE_CLEAN_VLLM_COMPOSITION=1
  export VERIFY_VLLM_BASE_HEAD="${verify_base_head}"
}

configure_sparkinfer_composition() {
  local composition_dir="$1"
  local verify_base_head="$2"
  local composition_lock="${composition_dir}/integration.lock.json"

  export SPARKINFER_REPO="$(jq -er '.base.repository' "${composition_lock}")"
  export SPARKINFER_REF="$(jq -er '.base.ref | sub("^refs/heads/"; "")' "${composition_lock}")"
  export SPARKINFER_COMMIT="$(jq -er '.base.commit' "${composition_lock}")"
  export SPARKINFER_PATCH_FILE="${composition_dir#patches/}/integration.patch"
  export SPARKINFER_PATCH_SHA256="$(jq -er '.result.patch_sha256' "${composition_lock}")"
  export SPARKINFER_INTEGRATION_LOCK_FILE="${composition_lock}"
  export SPARKINFER_INTEGRATION_TREE="$(jq -er '.result.tree' "${composition_lock}")"
  export REQUIRE_CLEAN_SPARKINFER_COMPOSITION=1
  export VERIFY_SPARKINFER_BASE_HEAD="${verify_base_head}"
}

if [[ "${composition_mode}" == "clean" ]]; then
  composition_dir="patches/generated/gilded-gnosis-v20/vllm"
  python3 scripts/compose_vllm_release.py \
    manifests/vllm/gilded-gnosis-v20.json \
    --output-dir "${composition_dir}" >/dev/null
  configure_vllm_composition "${composition_dir}" 1

  sparkinfer_composition_dir="patches/generated/gilded-gnosis-v20/sparkinfer"
  python3 scripts/compose_vllm_release.py \
    manifests/sparkinfer/gilded-gnosis-v20.json \
    --output-dir "${sparkinfer_composition_dir}" >/dev/null
  configure_sparkinfer_composition "${sparkinfer_composition_dir}" 1

  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm${VLLM_INTEGRATION_TREE:0:7}-si${SPARKINFER_INTEGRATION_TREE:0:7}-fi801d57a-cu132-${release_date}-r8}"
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm${VLLM_INTEGRATION_TREE:0:7}.si${SPARKINFER_INTEGRATION_TREE:0:7}.fi801d57a.cu132.${release_date}.r8}"
elif [[ "${composition_mode}" == "reproduce-r5" ]]; then
  configure_vllm_composition \
    "patches/releases/gilded-gnosis-v20-r5/vllm" 0
  configure_sparkinfer_composition \
    "patches/releases/gilded-gnosis-v20-r5/sparkinfer" 0

  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r5}"
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r5}"
elif [[ "${composition_mode}" == "reproduce-r6" ]]; then
  # r6 changes only the container/runtime integration. Reuse the immutable r5
  # vLLM and SparkInfer source archives instead of consulting moving branches.
  configure_vllm_composition \
    "patches/releases/gilded-gnosis-v20-r5/vllm" 0
  configure_sparkinfer_composition \
    "patches/releases/gilded-gnosis-v20-r5/sparkinfer" 0

  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r6}"
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r6}"
elif [[ "${composition_mode}" == "reproduce-r7" ]]; then
  # r7 keeps the validated model stacks and moves the LMCache integration from
  # a release patch to its merged, immutable source commit.
  configure_vllm_composition \
    "patches/releases/gilded-gnosis-v20-r5/vllm" 0
  configure_sparkinfer_composition \
    "patches/releases/gilded-gnosis-v20-r5/sparkinfer" 0

  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r7}"
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r7}"
elif [[ "${composition_mode}" == "reproduce-r8" ]]; then
  # r8 keeps the immutable r7 compute and LMCache stacks, then replaces only
  # XGrammar with the pinned source build configured below.
  configure_vllm_composition \
    "patches/releases/gilded-gnosis-v20-r5/vllm" 0
  configure_sparkinfer_composition \
    "patches/releases/gilded-gnosis-v20-r5/sparkinfer" 0

  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r8}"
  export VLLM_BUILD_VERSION="${VLLM_BUILD_VERSION:-0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r8}"
elif [[ "${composition_mode}" == "reproduce-r4" ]]; then
  export IMAGE="${IMAGE:-voipmonitor/vllm:gilded-gnosis-v20-vllm0c79e41-sic3828fd-fi801d57a-cu132-20260727-r4}"
  export VLLM_REPO="https://github.com/voipmonitor/vllm.git"
  export VLLM_REF="build/gilded-gnosis-v20-pcie-auto-20260726"
  export VLLM_COMMIT="0c79e41db41f250ccdfc4be92d171960a5787f73"
  export VLLM_BUILD_VERSION="0.11.2.dev280+gilded.gnosis.v20.vllm0c79e41.sic3828fd.fi801d57a.cu132.20260727"
  export VLLM_PATCH_URL=
  export VLLM_PATCH_SHA256=
  export VLLM_PATCH_FILE=
  export REQUIRE_CLEAN_VLLM_COMPOSITION=0
  export SPARKINFER_REPO="https://github.com/local-inference-lab/sparkinfer.git"
  export SPARKINFER_REF="build/sparkinfer-v20-runtime-stride-20260727"
  export SPARKINFER_COMMIT="c3828fd7f807ce237a9ac36ef033659e6f6b6dd3"
  export SPARKINFER_PATCH_FILE=
  export SPARKINFER_PATCH_SHA256=
  export REQUIRE_CLEAN_SPARKINFER_COMPOSITION=0
else
  printf 'Unknown VLLM_RELEASE_COMPOSITION=%s\n' "${composition_mode}" >&2
  exit 1
fi

export SYSTEM_BASE_IMAGE="${SYSTEM_BASE_IMAGE:-voipmonitor/vllm:glm-kimi-cu132-system-base-20260626}"
export BUILD_BASE_IMAGE_TAG="${BUILD_BASE_IMAGE_TAG:-voipmonitor/vllm:glm-kimi-cu132-build-base-20260626}"
export BUILD_BASE_IMAGE="${BUILD_BASE_IMAGE:-0}"
export PUSH_BASE_IMAGE="${PUSH_BASE_IMAGE:-0}"
export MAX_JOBS="${MAX_JOBS:-64}"
export VLLM_MAX_JOBS="${VLLM_MAX_JOBS:-64}"
export NVCC_THREADS="${NVCC_THREADS:-1}"
export VLLM_NVCC_THREADS="${VLLM_NVCC_THREADS:-1}"
export PIN_SOURCE_COMMITS=1
export VALIDATION_GPU="${VALIDATION_GPU:-0}"

export NCCL_REPO="${NCCL_REPO:-https://github.com/local-inference-lab/nccl-canonical.git}"
export NCCL_REF="${NCCL_REF:-canonical/cu132-nccl2304-amd-noxml}"
export NCCL_COMMIT="${NCCL_COMMIT:-dfab7c1ace32da250ba97757879429c341b7bcf9}"

export FLASHINFER_REPO="${FLASHINFER_REPO:-https://github.com/voipmonitor/flashinfer.git}"
export FLASHINFER_REF="${FLASHINFER_REF:-codex/sm120-dspark-stack-20260711}"
export FLASHINFER_COMMIT="${FLASHINFER_COMMIT:-801d57a08958c13d375ddbb6be3be4808f48a708}"
export FLASHINFER_BUILD_CUBIN="${FLASHINFER_BUILD_CUBIN:-0}"

export DEEPGEMM_REPO="${DEEPGEMM_REPO:-https://github.com/deepseek-ai/DeepGEMM.git}"
export DEEPGEMM_REF="${DEEPGEMM_REF:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"
export DEEPGEMM_COMMIT="${DEEPGEMM_COMMIT:-a6b593d2826719dcf4892609af7b84ee23aaf32a}"

export EXLLAMAV3_REPO="${EXLLAMAV3_REPO:-https://github.com/brandonmmusic-max/exllamav3.git}"
export EXLLAMAV3_REF="${EXLLAMAV3_REF:-a1-retile-sm120}"
export EXLLAMAV3_COMMIT="${EXLLAMAV3_COMMIT:-704aefd743b390af4bd0fb429d1906f9b964c7d8}"

export VLLM_PATCH_URL=

export SPARKINFER_VERSION="${SPARKINFER_VERSION:-1.0.1}"

export LAUNCHER_REPO="${LAUNCHER_REPO:-https://github.com/local-inference-lab/blackwell-llm-docker.git}"
export LAUNCHER_REF="${LAUNCHER_REF:-a5791db0cf8daa4acab7d849e04fc036f1be00d5}"
export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-a5791db0cf8daa4acab7d849e04fc036f1be00d5}"
export VLLM_REQUIRED_LAUNCHERS="serve-gilded-gnosis.sh serve-fathomless-firmament.sh serve-glm52-v16.sh serve-glm52-v18.sh serve-glm52-v19.sh serve-glm52-hybrid-v17.sh serve-glm52-hybrid-v18.sh serve-glm52-hybrid-v19.sh glm52-dcp-prefill-policy.sh glm52-pcie-runtime-env.sh glm52-pcie-calibration.py glm52-lmcache-wrapper.sh"

export CUTLASS_REF="${CUTLASS_REF:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_COMMIT="${CUTLASS_COMMIT:-e6233cbac5d7c7a865c19c91cd684ceece19513c}"
export CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION:-4.6.0}"
export TORCH_VERSION_PREFIX="${TORCH_VERSION_PREFIX:-2.12.0+cu132}"
export TOKENSPEED_MLA_VERSION="${TOKENSPEED_MLA_VERSION:-0.1.8}"
export TVM_FFI_VERSION="${TVM_FFI_VERSION:-0.1.10}"
export TRITON_KERNELS_REF=
export TRITON_KERNELS_COMMIT=

export XGRAMMAR_REPO="${XGRAMMAR_REPO:-https://github.com/mlc-ai/xgrammar.git}"
if [[ "${composition_mode}" == "clean" || "${composition_mode}" == "reproduce-r8" ]]; then
  export XGRAMMAR_REF="${XGRAMMAR_REF:-v0.2.5}"
  export XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT:-2ea71da4ccb997a06928c9fb69b99f330da56697}"
  export XGRAMMAR_VERSION="${XGRAMMAR_VERSION:-0.2.5}"
  export XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT:-1}"
else
  # Historical composition modes retain the xgrammar supplied by their
  # original vLLM requirements instead of silently changing old releases.
  export XGRAMMAR_REF="${XGRAMMAR_REF-}"
  export XGRAMMAR_COMMIT="${XGRAMMAR_COMMIT-}"
  export XGRAMMAR_VERSION="${XGRAMMAR_VERSION-}"
  export XGRAMMAR_TRANSFORMERS5_COMPAT="${XGRAMMAR_TRANSFORMERS5_COMPAT:-0}"
fi

export INSTANTTENSOR_REPO="${INSTANTTENSOR_REPO:-https://github.com/scitix/InstantTensor.git}"
export INSTANTTENSOR_REF="${INSTANTTENSOR_REF:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
export INSTANTTENSOR_COMMIT="${INSTANTTENSOR_COMMIT:-85e7c5f5539d9c006ee0c26bc1b5233c65251b6b}"
if [[ "${composition_mode}" == "reproduce-r6" ]]; then
  export LMCACHE_REPO="${LMCACHE_REPO:-https://github.com/LMCache/LMCache.git}"
  export LMCACHE_REF="${LMCACHE_REF:-v0.5.2}"
  export LMCACHE_COMMIT="${LMCACHE_COMMIT:-cd2c0d6a6a982ec5e334bae7704e1029c06d3c97}"
  export LMCACHE_PATCH_FILE="${LMCACHE_PATCH_FILE:-lmcache/glm52-dcp-v052.patch}"
  export LMCACHE_PATCH_SHA256="${LMCACHE_PATCH_SHA256:-504434b7d0ccd7bffa2b01a716ad0471405fa6dad8b2ba78a3b0ec4b8627a255}"
  export LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.2}"
else
  export LMCACHE_REPO="${LMCACHE_REPO:-https://github.com/local-inference-lab/LMCache.git}"
  export LMCACHE_REF="${LMCACHE_REF:-release/v0.5.2-glm52-dcp-base}"
  export LMCACHE_COMMIT="${LMCACHE_COMMIT:-9cebd405d0caf4bebe01d694b5a8bf4e3e354314}"
  export LMCACHE_PATCH_FILE="${LMCACHE_PATCH_FILE:-}"
  export LMCACHE_PATCH_SHA256="${LMCACHE_PATCH_SHA256:-}"
  export LMCACHE_BUILD_VERSION="${LMCACHE_BUILD_VERSION:-0.5.2+glm52dcp.3}"
fi

if [[ "${PRINT_RELEASE_CONFIG:-0}" == 1 ]]; then
  printf 'composition=%s\nimage=%s\nversion=%s\nvllm_tree=%s\nsparkinfer_tree=%s\nlmcache_repo=%s\nlmcache_ref=%s\nlmcache_commit=%s\nlmcache_patch=%s\nlmcache_version=%s\nxgrammar_repo=%s\nxgrammar_ref=%s\nxgrammar_commit=%s\nxgrammar_version=%s\nxgrammar_transformers5_compat=%s\n' \
    "${composition_mode}" "${IMAGE}" "${VLLM_BUILD_VERSION}" \
    "${VLLM_INTEGRATION_TREE:-}" "${SPARKINFER_INTEGRATION_TREE:-}" \
    "${LMCACHE_REPO}" "${LMCACHE_REF}" "${LMCACHE_COMMIT}" \
    "${LMCACHE_PATCH_FILE}" "${LMCACHE_BUILD_VERSION}" \
    "${XGRAMMAR_REPO}" "${XGRAMMAR_REF}" "${XGRAMMAR_COMMIT}" \
    "${XGRAMMAR_VERSION}" "${XGRAMMAR_TRANSFORMERS5_COMPAT}"
  exit 0
fi
export HUMMING_KERNELS_SPEC="${HUMMING_KERNELS_SPEC:-humming-kernels[cu13]==0.1.10}"
export VLLM_RUNTIME_EXTRA_PACKAGES="${VLLM_RUNTIME_EXTRA_PACKAGES:-nvtx==0.2.15 PyNvVideoCodec==2.0.4 nccl4py==0.3.1}"

requested_push="${PUSH_IMAGE:-0}"
export PUSH_IMAGE=0

runtime_source_paths=(
  launchers
  tests/test-glm52-dcp-prefill-policy.sh
  tests/test-glm52-pcie-calibration-helper.sh
  tests/test-glm52-online-quant-policy.sh
  tests/test-glm52-exl3-helper.sh
  tests/test-glm52-lmcache-helper.sh
  tests/test-glm52-pcie-calibration.py
)
if ! git diff --quiet "${LAUNCHER_COMMIT}" -- "${runtime_source_paths[@]}" || \
   [[ -n "$(git status --porcelain --untracked-files=all -- "${runtime_source_paths[@]}")" ]]; then
  printf 'Launcher/test sources do not match pinned commit %s\n' \
    "${LAUNCHER_COMMIT}" >&2
  exit 1
fi

./tests/test-glm52-dcp-prefill-policy.sh
./tests/test-glm52-pcie-calibration-helper.sh
./tests/test-glm52-online-quant-policy.sh
./tests/test-glm52-exl3-helper.sh
./tests/test-glm52-lmcache-helper.sh
python3 -m pytest -q tests/test-glm52-pcie-calibration.py
./build-vllm-sparkinfer-cu132.sh "$@"

labels="$(docker image inspect "${IMAGE}" --format '{{json .Config.Labels}}')"
jq -e --arg value "${VLLM_COMMIT}" '."local-inference.vllm.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${SPARKINFER_COMMIT}" '."local-inference.sparkinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${FLASHINFER_COMMIT}" '."local-inference.flashinfer.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LAUNCHER_COMMIT}" '."local-inference.launcher.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${CUTLASS_DSL_VERSION}" '."local-inference.cutlass_dsl.version" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${EXLLAMAV3_COMMIT}" '."local-inference.exllamav3.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LMCACHE_COMMIT}" '."local-inference.lmcache.commit" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LMCACHE_PATCH_SHA256}" '."local-inference.lmcache.patch_sha256" == $value' <<<"${labels}" >/dev/null
jq -e --arg value "${LMCACHE_BUILD_VERSION}" '."local-inference.lmcache.version" == $value' <<<"${labels}" >/dev/null
if [[ -n "${XGRAMMAR_REF}" ]]; then
  jq -e --arg value "${XGRAMMAR_COMMIT}" '."local-inference.xgrammar.commit" == $value' <<<"${labels}" >/dev/null
  jq -e --arg value "${XGRAMMAR_VERSION}" '."local-inference.xgrammar.version" == $value' <<<"${labels}" >/dev/null
  jq -e --arg value "${XGRAMMAR_TRANSFORMERS5_COMPAT}" '."local-inference.xgrammar.transformers5_compat" == $value' <<<"${labels}" >/dev/null
fi
if [[ "${composition_mode}" != "reproduce-r4" ]]; then
  jq -e --arg value "${VLLM_INTEGRATION_TREE}" '."local-inference.vllm.integration.tree" == $value' <<<"${labels}" >/dev/null
  jq -e --arg value "${VLLM_PATCH_SHA256}" '."local-inference.vllm.patch_sha256" == $value' <<<"${labels}" >/dev/null
  jq -e '."local-inference.vllm.patch_file" != "" and ."local-inference.vllm.patch_url" == ""' <<<"${labels}" >/dev/null
  jq -e --arg value "${SPARKINFER_INTEGRATION_TREE}" '."local-inference.sparkinfer.integration.tree" == $value' <<<"${labels}" >/dev/null
  jq -e --arg value "${SPARKINFER_PATCH_SHA256}" '."local-inference.sparkinfer.patch_sha256" == $value' <<<"${labels}" >/dev/null
  jq -e '."local-inference.sparkinfer.patch_file" != ""' <<<"${labels}" >/dev/null
else
  jq -e '."local-inference.vllm.patch_file" == "" and ."local-inference.vllm.patch_url" == ""' <<<"${labels}" >/dev/null
  jq -e '."local-inference.sparkinfer.patch_file" == ""' <<<"${labels}" >/dev/null
fi

for launcher in ${VLLM_REQUIRED_LAUNCHERS}; do
  expected_launcher_sha="$(git show "${LAUNCHER_COMMIT}:launchers/${launcher}" | sha256sum | cut -d' ' -f1)"
  actual_launcher_sha="$(docker run --rm --entrypoint sha256sum "${IMAGE}" "/usr/local/bin/${launcher}" | cut -d' ' -f1)"
  if [[ "${actual_launcher_sha}" != "${expected_launcher_sha}" ]]; then
    printf 'Launcher %s does not match pinned commit %s\n' "${launcher}" "${LAUNCHER_COMMIT}" >&2
    exit 1
  fi
done

cache_fingerprint="$(jq -r '."local-inference.cache.fingerprint"' <<<"${labels}")"
vllm_cache_prefix="${VLLM_COMMIT:0:10}"
sparkinfer_cache_prefix="${SPARKINFER_COMMIT:0:10}"
[[ "${cache_fingerprint}" =~ ^vllm${vllm_cache_prefix}-b12x${sparkinfer_cache_prefix}-[0-9a-f]{16}$ ]]

image_env="$(docker image inspect "${IMAGE}" --format '{{range .Config.Env}}{{println .}}{{end}}')"
grep -Fxq "XDG_CACHE_HOME=/cache/jit/${cache_fingerprint}" <<<"${image_env}"
grep -Fxq "VLLM_CACHE_ROOT=/cache/jit/${cache_fingerprint}/vllm" <<<"${image_env}"
grep -Fxq "SPARKINFER_COMPILE_CACHE_DIR=/cache/jit/${cache_fingerprint}/sparkinfer/compile" <<<"${image_env}"

docker run --rm --gpus "device=${VALIDATION_GPU}" -i \
  -e EXPECTED_SPARKINFER_VERSION="${SPARKINFER_VERSION}" \
  -e EXPECTED_CUTLASS_DSL_VERSION="${CUTLASS_DSL_VERSION}" \
  -e EXPECTED_TORCH_VERSION_PREFIX="${TORCH_VERSION_PREFIX}" \
  -e EXPECTED_LMCACHE_VERSION="${LMCACHE_BUILD_VERSION}" \
  -e EXPECTED_XGRAMMAR_VERSION="${XGRAMMAR_VERSION}" \
  --entrypoint /opt/venv/bin/python "${IMAGE}" - <<'PY'
import importlib.metadata as md
import inspect
import os

import torch
import vllm._C_stable_libtorch  # noqa: F401
import lmcache.c_ops  # noqa: F401
from lmcache.integration.vllm.vllm_multi_process_adapter import ParallelStrategy
from sparkinfer.attention.sparse_mla._scratch import SPARKINFERSparseMLAScratchCaps
from sparkinfer.attention.nsa_indexer import tiled_topk
from sparkinfer.comm.pcie import DcpAllToAllPool, DcpTopKOwnerExchange
from sparkinfer.comm.pcie.pcie_dma import (
    PCIeDmaAllReduce,
    _normalize_fp8_mode,
)
from sparkinfer.gemm import bmm, can_implement_bmm, prewarm_bmm
from sparkinfer.moe.fused_moe import _impl as fused_moe_impl
from sparkinfer.moe._shared.kernels.w4a16 import kernel as w4a16_kernel
from vllm import envs as vllm_envs
from vllm.distributed.device_communicators.cuda_communicator import CudaCommunicator
from vllm.model_executor.layers.attention import mla_attention
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
from vllm.model_executor.layers.quantization.exl3 import _load_exl3_ext
from vllm.model_executor.layers.sparse_attn_indexer import (
    _merge_b12x_dcp_topk_by_owner,
)
from vllm.model_executor.models.deepseek_v2 import _indexer_cache_dcp_shard_count
from vllm.v1.attention.backends.mla.b12x_mla_sparse import (
    B12xMLASparseImpl,
    _ckv_prefetch_depth_within_budget,
)
from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.worker.gpu_worker import Worker

assert md.version("sparkinfer") == os.environ["EXPECTED_SPARKINFER_VERSION"]
assert md.version("lmcache") == os.environ["EXPECTED_LMCACHE_VERSION"]
if os.environ.get("EXPECTED_XGRAMMAR_VERSION"):
    assert md.version("xgrammar") == os.environ["EXPECTED_XGRAMMAR_VERSION"]
assert md.version("nvidia-cutlass-dsl") == os.environ["EXPECTED_CUTLASS_DSL_VERSION"]
assert torch.__version__.startswith(os.environ["EXPECTED_TORCH_VERSION_PREFIX"])
assert torch.version.cuda == "13.2"
assert "VLLM_EXL3_ABI_SHIM" not in os.environ
exl3_ext = _load_exl3_ext()
for exl3_export in (
    "exl3_gemm",
    "exl3_moe_fused",
    "exl3_moe_fused_retile",
    "exl3_moe_max_concurrency",
):
    assert hasattr(exl3_ext, exl3_export), exl3_export
from sparkinfer.moe import trellis_moe
for trellis_export in ("Caps", "plan", "prepare_weights", "bind", "run"):
    assert hasattr(trellis_moe, trellis_export), trellis_export
assert fused_moe_impl._dynamic_kernel_intermediate_size(352, "w4a8_mx") == 384
assert tiled_topk._COARSE_RADIX_BITS == 10
assert tiled_topk._SMEM_CANDS == 8192
assert inspect.getsource(w4a16_kernel).count("cooperative=True") >= 2
assert _normalize_fp8_mode("i8-ring") == "i8_ring"
assert _normalize_fp8_mode("mxfp8-ring") == "mx_ring"
dma_source = inspect.getsource(PCIeDmaAllReduce.all_reduce)
assert "out = torch.empty_like(inp)" in dma_source
assert "_persistent_output_view" not in inspect.getsource(PCIeDmaAllReduce)
topk_source = inspect.getsource(tiled_topk)
assert "tiled_topk_v27_runtime_page_stride" in topk_source
assert "row_topk_v7_runtime_page_stride" in topk_source
emit_source = inspect.getsource(tiled_topk._emit_global_index_virtual)
assert "Int64(row_idx) * Int64(output_page_table_row_stride)" in emit_source
assert callable(DcpTopKOwnerExchange)
assert callable(bmm) and callable(can_implement_bmm) and callable(prewarm_bmm)
assert "head_major_output" in inspect.signature(cp_lse_ag_out_rs).parameters
assert hasattr(CudaCommunicator, "reduce_scatter_head_major")
assert "head_major_output=True" in inspect.getsource(MLAAttention.forward_impl)
assert "ensure_cublas_tail_padding" not in inspect.getsource(MLAAttention._v_up_proj)
assert hasattr(torch.ops._C, "safe_mla_query_bmm")
assert "out" in inspect.signature(DcpAllToAllPool.lse_reduce_scatter).parameters
assert hasattr(mla_attention, "_preallocate_absorbed_mla_weights")
assert not hasattr(mla_attention, "_release_b12x_mxfp8_kv_b_proj")
spec_source = inspect.getsource(B12xMLASparseImpl)
assert 'os.getenv("VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE", "auto")' in spec_source
assert "attn_metadata.is_spec_decode" in spec_source
assert "use_safe_mla_query_bmm = True" in spec_source
assert hasattr(B12xMLASparseImpl, "reset_kv_cache_binding_state")
assert hasattr(Worker, "_profile_model_with_kernel_warmup")
assert hasattr(Worker, "_warmup_kernels_once")
assert callable(_merge_b12x_dcp_topk_by_owner)
assert callable(_indexer_cache_dcp_shard_count)
assert _ckv_prefetch_depth_within_budget(2, 1, 8, 1024, 576) == 0
assert hasattr(vllm_envs, "VLLM_DCP_QUERY_SPLIT")
assert hasattr(vllm_envs, "VLLM_B12X_MLA_CKV_GATHER")
assert hasattr(vllm_envs, "VLLM_DCP_TOPK_OWNER_MERGE")
assert hasattr(vllm_envs, "VLLM_DCP_INDEXER_SHARDS")
assert hasattr(vllm_envs, "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH")
caps = SPARKINFERSparseMLAScratchCaps(
    device="cuda:0",
    dtype=torch.bfloat16,
    num_q_heads=8,
    max_q_rows=6,
    max_width=2048,
    head_dim=576,
    v_head_dim=512,
    head_major_output=True,
)
assert caps.head_major_output is True
lmcache_strategy = ParallelStrategy(
    use_mla=True,
    vllm_world_size=8,
    vllm_worker_id=5,
    tp_size=8,
    pp_size=1,
    n_servers=1,
    dcp_size=4,
)
assert lmcache_strategy.kv_world_size == 4
assert lmcache_strategy.kv_worker_id == 1
assert lmcache_strategy.kv_tp_size == 2
assert not lmcache_strategy.is_kv_writer
assert __import__("pathlib").Path(
    "/opt/vllm/kv-scales/glm52-nvfp4-nf3-hybrid_mla_outer_scales_v1.json"
).is_file()
print("v20 final runtime contracts: PASS")
PY

dry_run_file="/tmp/gilded-gnosis-v20-final-dcp1.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e MAX_NUM_SEQS=1 \
  -e GRAPH=6 \
  "${IMAGE}" | tee "${dry_run_file}"

grep -q -- '--max-num-seqs 1' "${dry_run_file}"
grep -q -- '--max-cudagraph-capture-size 6' "${dry_run_file}"
grep -q -- '--load-format instanttensor' "${dry_run_file}"
grep -q -- '--max-model-len 262144' "${dry_run_file}"
grep -q -- '--gpu-memory-utilization 0.96' "${dry_run_file}"
grep -Fxq 'VLLM_DCP_QUERY_SPLIT=1' "${dry_run_file}"
grep -Fxq 'VLLM_B12X_MLA_CKV_GATHER=0' "${dry_run_file}"
grep -Fxq 'VLLM_DCP_TOPK_OWNER_MERGE=0' "${dry_run_file}"
grep -Fxq 'VLLM_DCP_INDEXER_SHARDS=0' "${dry_run_file}"
grep -Fxq 'VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=0' "${dry_run_file}"
grep -Fxq 'VLLM_PCIE_DMA_MIN_BYTES=6MB' "${dry_run_file}"
grep -Fxq 'PCIE_CALIBRATION_STATUS=skipped:dry-run' "${dry_run_file}"

exl3_dry_run_file="/tmp/gilded-gnosis-v20-final-exl3.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52-exl3 \
  "${IMAGE}" | tee "${exl3_dry_run_file}"

grep -q -- '--tensor-parallel-size 4' "${exl3_dry_run_file}"
grep -q -- '--decode-context-parallel-size 4' "${exl3_dry_run_file}"
grep -q -- '--quantization exl3' "${exl3_dry_run_file}"
grep -q -- '--load-format safetensors' "${exl3_dry_run_file}"
grep -q -- '--no-async-scheduling' "${exl3_dry_run_file}"
grep -Fq -- '\"moe_backend\":\"triton\"' "${exl3_dry_run_file}"
grep -Fq -- '\"draft_sample_method\":\"greedy\"' "${exl3_dry_run_file}"
grep -Fq -- '\"cudagraph_capture_sizes\":\[4\,8\,12\,16\,20\,24\,28\,32\]' "${exl3_dry_run_file}"
if grep -q -- '--quantization-config' "${exl3_dry_run_file}"; then
  printf 'EXL3 helper unexpectedly enabled online quantization\n' >&2
  exit 1
fi

assert_dcp_policy() {
  local name="$1"
  local tp="$2"
  local dcp="$3"
  local query_split="$4"
  local ckv_gather="$5"
  local owner_merge="$6"
  local indexer_shards="$7"
  local prefetch_depth="$8"
  local calibration_status="$9"
  shift 9
  local output_file="/tmp/gilded-gnosis-v20-final-policy-${name}.txt"

  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e DRY_RUN=1 \
    -e MODEL_FAMILY=glm52 \
    -e MODEL=/model \
    -e TP="${tp}" \
    -e DCP="${dcp}" \
    -e MTP=0 \
    -e MAX_NUM_SEQS=1 \
    -e GRAPH=6 \
    "$@" \
    "${IMAGE}" | tee "${output_file}"

  grep -Fxq "VLLM_DCP_QUERY_SPLIT=${query_split}" "${output_file}"
  grep -Fxq "VLLM_B12X_MLA_CKV_GATHER=${ckv_gather}" "${output_file}"
  grep -Fxq "VLLM_DCP_TOPK_OWNER_MERGE=${owner_merge}" "${output_file}"
  grep -Fxq "VLLM_DCP_INDEXER_SHARDS=${indexer_shards}" "${output_file}"
  grep -Fxq \
    "VLLM_B12X_MLA_CKV_PREFETCH_DEPTH=${prefetch_depth}" \
    "${output_file}"
  grep -Fxq "VLLM_PCIE_DMA_MIN_BYTES=6MB" "${output_file}"
  grep -Fxq \
    "PCIE_CALIBRATION_STATUS=${calibration_status}" \
    "${output_file}"
}

assert_dcp_policy tp8-dcp4 8 4 1 1 1 2 1 skipped:dry-run \
  -e DCP_CKV_PREFETCH_TOPOLOGY=safe
assert_dcp_policy tp8-dcp8 8 8 1 1 1 4 1 skipped:dry-run \
  -e DCP_CKV_PREFETCH_TOPOLOGY=safe
assert_dcp_policy tp8-dcp4-slow-topology 8 4 1 1 1 2 0 skipped:dry-run \
  -e DCP_CKV_PREFETCH_TOPOLOGY=unsafe
assert_dcp_policy tp6-dcp3 6 3 0 0 1 0 0 skipped:unsupported-tp-dcp
assert_dcp_policy tp8-dcp4-disabled 8 4 0 0 0 0 0 skipped:dry-run \
  -e DCP_QUERY_SPLIT=0 \
  -e DCP_CKV_GATHER=0 \
  -e DCP_TOPK_OWNER_MERGE=0 \
  -e DCP_INDEXER_SHARDS=0 \
  -e DCP_CKV_PREFETCH_DEPTH=0

mxfp8_dry_run_file="/tmp/gilded-gnosis-v20-final-mxfp8.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  "${IMAGE}" | tee "${mxfp8_dry_run_file}"

grep -Fq 'QUANTIZATION_CONFIG_JSON=\{\"linear\":\{\"weight\":\"mxfp8\"\}\}' "${mxfp8_dry_run_file}"
if grep -q 'kv_b_proj' "${mxfp8_dry_run_file}"; then
  printf 'Unexpected kv_b_proj ignore in the default MXFP8 preset\n' >&2
  exit 1
fi

mxfp8_ignore_json='{"linear":{"weight":"mxfp8"},"ignore":["re:.*[.]q_a_proj$","re:.*[.]kv_a_proj_with_mqa$"]}'
mxfp8_ignore_dry_run_file="/tmp/gilded-gnosis-v20-final-mxfp8-ignore.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=1 \
  -e MTP=0 \
  -e MOE_MODE=a16 \
  -e ONLINE_QUANT=mxfp8 \
  -e QUANTIZATION_CONFIG_JSON="${mxfp8_ignore_json}" \
  "${IMAGE}" | tee "${mxfp8_ignore_dry_run_file}"

grep -Fq 're:.\*\[.\]q_a_proj\$' "${mxfp8_ignore_dry_run_file}"
grep -Fq 're:.\*\[.\]kv_a_proj_with_mqa\$' "${mxfp8_ignore_dry_run_file}"

for dma_mode in i8_ring mx_ring; do
  dma_dry_run_file="/tmp/gilded-gnosis-v20-final-${dma_mode}.txt"
  docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
    -e DRY_RUN=1 \
    -e MODEL_FAMILY=glm52 \
    -e MODEL=/model \
    -e F8_DMA="${dma_mode}" \
    "${IMAGE}" | tee "${dma_dry_run_file}"
  grep -Fxq "VLLM_PCIE_DMA_FP8=${dma_mode}" "${dma_dry_run_file}"
  grep -Fxq "SPARKINFER_PCIE_DMA_FP8=${dma_mode}" "${dma_dry_run_file}"
done

grep -Fxq 'VLLM_B12X_ABSORB_BMM=1' "${mxfp8_dry_run_file}"

absorb_disabled_dry_run_file="/tmp/gilded-gnosis-v20-final-absorb-disabled.txt"
docker run --rm --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e ONLINE_QUANT=mxfp8 \
  -e VLLM_B12X_ABSORB_BMM=0 \
  "${IMAGE}" | tee "${absorb_disabled_dry_run_file}"

grep -Fxq 'VLLM_B12X_ABSORB_BMM=0' "${absorb_disabled_dry_run_file}"

lmcache_dry_run_file="/tmp/gilded-gnosis-v20-final-lmcache.txt"
docker run --rm --gpus "device=${VALIDATION_GPU}" --shm-size=2g \
  --entrypoint /usr/local/bin/serve-gilded-gnosis.sh \
  -e DRY_RUN=1 \
  -e MODEL_FAMILY=glm52 \
  -e MODEL=/model \
  -e TP=8 \
  -e DCP=4 \
  -e MTP=0 \
  -e LMCACHE_MODE=ram \
  -e LMCACHE_L1_GB=1 \
  -e LMCACHE_L1_INIT_GB=1 \
  "${IMAGE}" | tee "${lmcache_dry_run_file}"

grep -Fq 'LMCache ready: mode=ram' "${lmcache_dry_run_file}"
grep -Fq -- '--kv-transfer-config' "${lmcache_dry_run_file}"
grep -Fq 'LMCacheMPConnector' "${lmcache_dry_run_file}"

if [[ "${requested_push}" == "1" ]]; then
  docker push "${IMAGE}"
fi

printf 'Image: %s\n' "${IMAGE}"
