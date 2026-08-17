#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
dockerfile="${repo_root}/Dockerfile.deepseek-infernal-invocation-cu133-torch213"
builder="${repo_root}/build-deepseek-infernal-invocation-cu133-torch213.sh"
pip_check_allowlist="${repo_root}/tests/deepseek-infernal-cu133-pip-check.allowlist"
compose_file="${repo_root}/examples/docker-compose-ds4-infernal-invocation-cu133-r16.yml"
glm_nvfp4_compose_file="${repo_root}/examples/docker-compose-glm52-nvfp4-infernal-invocation-r16.yml"
glm_exl3_compose_file="${repo_root}/examples/docker-compose-glm52-exl3-infernal-invocation-r16.yml"
vllm_manifest="${repo_root}/manifests/vllm/infernal-invocation.json"
vllm_lock="${repo_root}/patches/releases/infernal-invocation-r16/vllm/integration.lock.json"
b12x_manifest="${repo_root}/manifests/b12x/infernal-invocation.json"
b12x_lock="${repo_root}/patches/releases/infernal-invocation-r16/b12x/integration.lock.json"

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
grep -Fq 'ONLINE_QUANT: ${ONLINE_QUANT:-trellis-mcg-b6}' "${glm_exl3_compose_file}"
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
    .number == 308 and
    .head == "053e6351d0b3b3e35c969c9e3933db64d30a7164") and
  any(.pull_requests[];
    .number == 309 and
    .head == "dc0c026df62448d1bec747d9dd6fb0a01d838f3e") and
  any(.pull_requests[];
    .number == 300 and
    .head == "901a7c50e5d344b5bea975c3393c4dc23c958fc1") and
  any(.pull_requests[];
    .number == 320 and
    .head == "e9534672129b961399b1625d33d83c79eacded30") and
  any(.pull_requests[];
    .number == 417 and
    .head == "2511e5df2b1e4dfd2360a28e89899c90b7b3fcc7") and
  any(.pull_requests[];
    .number == 415 and
    .head == "c805ebd0896ccfbd2569bc0b2a7944d3282106ff") and
  all(.pull_requests[]; .number != 291 and .number != 305 and .number != 315) and
  any(.reviewed_exclusions[];
    .number == 291 and
    .disposition == "superseded") and
  any(.reviewed_exclusions[];
    .number == 305 and
    .disposition == "superseded") and
  any(.reviewed_exclusions[];
    .number == 315 and
    .disposition == "unsupported")
' "${vllm_manifest}" >/dev/null
jq -e '
  .base.commit == "d6e0bb7c813264f1293288b9de0d18f277be7c9f" and
  .result.tree == "5beffc48f7cd9d4ade076e4b6d1f117ac8e79d4a" and
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
    .number == 308 and
    .head == "053e6351d0b3b3e35c969c9e3933db64d30a7164" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 309 and
    .head == "dc0c026df62448d1bec747d9dd6fb0a01d838f3e" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 300 and
    .head == "901a7c50e5d344b5bea975c3393c4dc23c958fc1" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 320 and
    .head == "e9534672129b961399b1625d33d83c79eacded30" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 417 and
    .head == "2511e5df2b1e4dfd2360a28e89899c90b7b3fcc7" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 415 and
    .head == "c805ebd0896ccfbd2569bc0b2a7944d3282106ff" and
    .disposition == "merged") and
  all(.pull_requests[]; .number != 291 and .number != 305 and .number != 315)
' "${vllm_lock}" >/dev/null
jq -e '
  any(.pull_requests[];
    .number == 221 and
    .head == "413f96e889dad1ae0752fd1f4be9d37f56849600") and
  any(.pull_requests[];
    .number == 223 and
    .head == "e99775f552c4f28cf1f345ded28bb77a57ea6a83") and
  any(.pull_requests[];
    .number == 227 and
    .head == "e38436d76a95c586c57e06646f8ea5b8c8ed11c7") and
  any(.pull_requests[];
    .number == 228 and
    .head == "50046df84a15cc5f76b94260e897fd39072b2fdf") and
  any(.pull_requests[];
    .number == 229 and
    .head == "2cdd9e265cd6c4dca43e7d42c5a8cb265c92adfb") and
  any(.pull_requests[];
    .number == 230 and
    .head == "156920046e858f413db0c51e53cd25b9020d5f40") and
  all(.pull_requests[]; .number != 146 and .number != 150 and .number != 197 and .number != 214) and
  any(.reviewed_exclusions[];
    .number == 197 and
    .disposition == "unsupported")
' "${b12x_manifest}" >/dev/null
jq -e '
  .base.commit == "c25cdba2c1df7a69b2d7771e4243e12a8fbf19d5" and
  .result.tree == "a4a0bc8a8f5e56dbef85f9b46b0d74f6e8edb491" and
  any(.pull_requests[];
    .number == 221 and
    .head == "413f96e889dad1ae0752fd1f4be9d37f56849600" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 223 and
    .head == "e99775f552c4f28cf1f345ded28bb77a57ea6a83" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 227 and
    .head == "e38436d76a95c586c57e06646f8ea5b8c8ed11c7" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 228 and
    .head == "50046df84a15cc5f76b94260e897fd39072b2fdf" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 229 and
    .head == "2cdd9e265cd6c4dca43e7d42c5a8cb265c92adfb" and
    .disposition == "merged") and
  any(.pull_requests[];
    .number == 230 and
    .head == "156920046e858f413db0c51e53cd25b9020d5f40" and
    .disposition == "merged") and
  all(.pull_requests[]; .number != 146 and .number != 150 and .number != 197 and .number != 214)
' "${b12x_lock}" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'release=infernal-invocation-cu133-torch213' <<<"${output}"
grep -Fxq 'revision=r16' <<<"${output}"
grep -Fxq 'vllm_ref=dev/infernal-invocation' <<<"${output}"
grep -Fxq 'vllm_tree=5beffc48f7cd9d4ade076e4b6d1f117ac8e79d4a' <<<"${output}"
grep -Fxq 'b12x_ref=master' <<<"${output}"
grep -Fxq 'b12x_tree=a4a0bc8a8f5e56dbef85f9b46b0d74f6e8edb491' <<<"${output}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output}"
grep -Fxq 'lmcache_tree=e045d729bc5c4c63a40e13d032f42923de97812f' <<<"${output}"
grep -Fxq 'torch=2.13.0' <<<"${output}"
grep -Fxq 'cuda=13.3' <<<"${output}"
grep -Fxq 'nccl=2.31.2' <<<"${output}"
grep -Fxq 'flashinfer=0.6.18+cu133' <<<"${output}"
