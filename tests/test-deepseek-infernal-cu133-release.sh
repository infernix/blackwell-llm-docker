#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
builder="${repo_root}/build-deepseek-infernal-invocation-cu133-torch213.sh"
pip_check_allowlist="${repo_root}/tests/deepseek-infernal-cu133-pip-check.allowlist"
compose_file="${repo_root}/examples/docker-compose-ds4-infernal-invocation-cu133-r15.yml"
glm_nvfp4_compose_file="${repo_root}/examples/docker-compose-glm52-nvfp4-infernal-invocation-r15.yml"
glm_exl3_compose_file="${repo_root}/examples/docker-compose-glm52-exl3-infernal-invocation-r15.yml"
glm_sqg_compose_file="${repo_root}/examples/docker-compose-glm52-sqg-infernal-invocation-r15.yml"
vllm_manifest="${repo_root}/manifests/vllm/infernal-invocation.json"
vllm_lock="${repo_root}/patches/releases/infernal-invocation-r15/vllm/integration.lock.json"
b12x_manifest="${repo_root}/manifests/b12x/infernal-invocation.json"
b12x_lock="${repo_root}/patches/releases/infernal-invocation-r15/b12x/integration.lock.json"

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
grep -Fq 'docker run --rm --gpus "\"device=${smoke_gpus}\"" --ipc=host' "${builder}"
grep -Fq 'SERVED_MODEL_NAME: ${SERVED_MODEL_NAME:-DeepSeek-V4-Flash-0731}' "${compose_file}"
grep -Fq 'GLOO_SOCKET_IFNAME: ${GLOO_SOCKET_IFNAME:-lo}' "${compose_file}"
grep -Fq 'NCCL_SOCKET_IFNAME: ${NCCL_SOCKET_IFNAME:-lo}' "${compose_file}"
grep -Fq 'LOAD_FORMAT: ${LOAD_FORMAT:-instanttensor}' "${compose_file}"
grep -Fq 'INSTANTTENSOR_BACKEND: ${INSTANTTENSOR_BACKEND:-BUFFERED}' "${compose_file}"
docker compose -f "${compose_file}" config --quiet
docker compose -f "${glm_nvfp4_compose_file}" config --quiet
docker compose -f "${glm_exl3_compose_file}" config --quiet
docker compose -f "${glm_sqg_compose_file}" config --quiet
grep -Fq 'MODEL: ${MODEL:-brandonmusic/GLM-5.2-SQG-W4A8}' "${glm_sqg_compose_file}"
grep -Fq 'MODEL_REVISION: ${MODEL_REVISION:-593dd0d2de6f79ce4e65303930c22c75e1359d44}' "${glm_sqg_compose_file}"
grep -Fq 'VLLM_GLM_SQG_W4A8_EVIDENCE_DIR: /cache/evidence/glm52-sqg-r15' "${glm_sqg_compose_file}"
jq -e '
  any(.pull_requests[];
    .number == 302 and
    .head == "7d1c21353cf4563b5344c83cf53acecac1f2f99c") and
  any(.pull_requests[];
    .number == 303 and
    .head == "4b297d1a07bfcc1bf0ab14c1dc25fe59c3e8f081") and
  any(.pull_requests[];
    .number == 304 and
    .head == "229de6270e511701045fd73af592620901c7422b") and
  any(.pull_requests[];
    .number == 305 and
    .head == "a5389deea51a64727bbca303ecabcd070517479f") and
  any(.pull_requests[];
    .number == 308 and
    .head == "053e6351d0b3b3e35c969c9e3933db64d30a7164") and
  any(.pull_requests[];
    .number == 309 and
    .head == "dc0c026df62448d1bec747d9dd6fb0a01d838f3e") and
  any(.pull_requests[];
    .number == 315 and
    .head == "ca9668472dc1dad4b99ac35fb6c34772828b81f7") and
  any(.pull_requests[];
    .number == 320 and
    .head == "e9534672129b961399b1625d33d83c79eacded30") and
  all(.pull_requests[]; .number != 291) and
  any(.reviewed_exclusions[];
    .number == 291 and
    .disposition == "superseded")
' "${vllm_manifest}" >/dev/null
jq -e '
  .result.tree == "068fc8e7270b92077ba753d002da179c865e444d" and
  any(.pull_requests[];
    .number == 302 and
    .head == "7d1c21353cf4563b5344c83cf53acecac1f2f99c" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 303 and
    .head == "4b297d1a07bfcc1bf0ab14c1dc25fe59c3e8f081" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 304 and
    .head == "229de6270e511701045fd73af592620901c7422b" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 305 and
    .head == "a5389deea51a64727bbca303ecabcd070517479f" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 308 and
    .head == "053e6351d0b3b3e35c969c9e3933db64d30a7164" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 309 and
    .head == "dc0c026df62448d1bec747d9dd6fb0a01d838f3e" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 315 and
    .head == "ca9668472dc1dad4b99ac35fb6c34772828b81f7" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 320 and
    .head == "e9534672129b961399b1625d33d83c79eacded30" and
    .disposition == "merged") and
  all(.pull_requests[]; .number != 291)
' "${vllm_lock}" >/dev/null
jq -e '
  any(.pull_requests[];
    .number == 197 and
    .head == "b234532cd35bf57c0efda1439981c72565ec4e6e") and
  any(.pull_requests[];
    .number == 214 and
    .head == "321c24a7ef60174cd6131d932f43bb84a4f3a60f")
' "${b12x_manifest}" >/dev/null
jq -e '
  .result.tree == "96e5d3d5c2057fa5d4f542e2368951ddbdcb5b42" and
  any(.pull_requests[];
    .number == 197 and
    .head == "b234532cd35bf57c0efda1439981c72565ec4e6e" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 214 and
    .head == "321c24a7ef60174cd6131d932f43bb84a4f3a60f" and
    .disposition == "merged")
' "${b12x_lock}" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'release=infernal-invocation-cu133-torch213' <<<"${output}"
grep -Fxq 'revision=r15' <<<"${output}"
grep -Fxq 'vllm_ref=dev/infernal-invocation' <<<"${output}"
grep -Fxq 'vllm_tree=068fc8e7270b92077ba753d002da179c865e444d' <<<"${output}"
grep -Fxq 'b12x_ref=master' <<<"${output}"
grep -Fxq 'b12x_tree=96e5d3d5c2057fa5d4f542e2368951ddbdcb5b42' <<<"${output}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output}"
grep -Fxq 'lmcache_tree=5fdf59cfa184bc15dc5414df0bd633da9e49aaae' <<<"${output}"
grep -Fxq 'torch=2.13.0' <<<"${output}"
grep -Fxq 'cuda=13.3' <<<"${output}"
grep -Fxq 'nccl=2.31.2' <<<"${output}"
grep -Fxq 'flashinfer=0.6.18+cu133' <<<"${output}"
