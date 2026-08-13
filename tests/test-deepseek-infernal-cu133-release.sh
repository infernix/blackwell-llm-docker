#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
builder="${repo_root}/build-deepseek-infernal-invocation-cu133-torch213.sh"
pip_check_allowlist="${repo_root}/tests/deepseek-infernal-cu133-pip-check.allowlist"
compose_file="${repo_root}/examples/docker-compose-ds4-infernal-invocation-cu133-r9.yml"
vllm_manifest="${repo_root}/manifests/vllm/infernal-invocation.json"
vllm_lock="${repo_root}/patches/releases/infernal-invocation-r9/vllm/integration.lock.json"

LC_ALL=C sort -cu "${pip_check_allowlist}"

grep -Fq 'ARG BASE_IMAGE=voipmonitor/vllm:kimi-k3-cu133-torch213-nccl2312-20260811-r2' "${dockerfile}"
grep -Fq 'local-inference.cuda.version="13.3"' "${dockerfile}"
grep -Fq 'local-inference.torch.version="2.13.0"' "${dockerfile}"
grep -Fq 'local-inference.nccl.version="2.31.2"' "${dockerfile}"
grep -Fq 'CUTLASS_DSL_VERSION=${CUTLASS_DSL_VERSION}' "${dockerfile}"
grep -Fq 'local-inference.xgrammar.version="${XGRAMMAR_VERSION}"' "${dockerfile}"
grep -Fq 'local-inference.exllamav3.commit="${EXLLAMAV3_COMMIT}"' "${dockerfile}"
grep -Fq 'compose_source lmcache /opt/infernal-invocation/lmcache' "${dockerfile}"
grep -Fq -- '--component _vllm_fa4_cutedsl_C' "${dockerfile}"
grep -Fq 'vllm_flash_attn/cute/utils.py' "${dockerfile}"
grep -Fq -- '--force-reinstall /tmp/instanttensor-src' "${dockerfile}"
grep -Fq 'local-inference.instanttensor.libaio.tree="${INSTANTTENSOR_LIBAIO_TREE}"' "${dockerfile}"
grep -Fq 'local-inference.nccl4py.version="${NCCL4PY_VERSION}"' "${dockerfile}"
grep -Fq 'deepseek-infernal-cu133-pip-check.allowlist' "${dockerfile}"
grep -Fq 'ENTRYPOINT ["/usr/local/bin/lmcache-mp-wrapper.sh", "/usr/local/bin/serve-ds4-flash.sh"]' "${dockerfile}"
grep -Fq -- '--build-arg "INSTANTTENSOR_COMMIT=${instanttensor_commit}"' "${builder}"
grep -Fq -- '--build-arg "INSTANTTENSOR_LIBAIO_TREE=${instanttensor_libaio_tree}"' "${builder}"
grep -Fq 'Process-group interfaces: GLOO_SOCKET_IFNAME=lo NCCL_SOCKET_IFNAME=lo' "${builder}"
grep -Fq 'SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}' "${compose_file}"
grep -Fq 'GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-lo}' "${compose_file}"
grep -Fq 'NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-lo}' "${compose_file}"
grep -Fq 'LOAD_FORMAT: ${LOAD_FORMAT:-instanttensor}' "${compose_file}"
grep -Fq 'INSTANTTENSOR_BACKEND: ${INSTANTTENSOR_BACKEND:-BUFFERED}' "${compose_file}"
docker compose -f "${compose_file}" config --quiet
jq -e '
  any(.pull_requests[];
    .number == 302 and
    .head == "7d1c21353cf4563b5344c83cf53acecac1f2f99c") and
  any(.pull_requests[];
    .number == 303 and
    .head == "4b297d1a07bfcc1bf0ab14c1dc25fe59c3e8f081")
' "${vllm_manifest}" >/dev/null
jq -e '
  .result.tree == "88aafbfa10cdb73adc50265a129edc0306541288" and
  any(.pull_requests[];
    .number == 302 and
    .head == "7d1c21353cf4563b5344c83cf53acecac1f2f99c" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 303 and
    .head == "4b297d1a07bfcc1bf0ab14c1dc25fe59c3e8f081" and
    .disposition == "merged")
' "${vllm_lock}" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'release=infernal-invocation-cu133-torch213' <<<"${output}"
grep -Fxq 'revision=r9' <<<"${output}"
grep -Fxq 'vllm_ref=dev/infernal-invocation' <<<"${output}"
grep -Fxq 'vllm_tree=88aafbfa10cdb73adc50265a129edc0306541288' <<<"${output}"
grep -Fxq 'b12x_ref=master' <<<"${output}"
grep -Fxq 'b12x_tree=5d648d944a047d4fac5c2035309c207b3faebd9c' <<<"${output}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output}"
grep -Fxq 'lmcache_tree=5fdf59cfa184bc15dc5414df0bd633da9e49aaae' <<<"${output}"
grep -Fxq 'torch=2.13.0' <<<"${output}"
grep -Fxq 'cuda=13.3' <<<"${output}"
grep -Fxq 'nccl=2.31.2' <<<"${output}"
grep -Fxq 'flashinfer=0.6.18+cu133' <<<"${output}"
