#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-infernal-invocation-cu133-torch213.sh"
compose_file="${repo_root}/examples/docker-compose-ds4-infernal-invocation-cu133-r19.yml"
composition_root="${repo_root}/patches/releases/infernal-invocation-r19"

for component in vllm b12x lmcache; do
  lock="${composition_root}/${component}/integration.lock.json"
  patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}"
  test -f "${patch}"
  jq -e '
    .schema_version == 1 and
    (.composition_strategy == "patch" or .composition_strategy == "merge") and
    (.base.repository | type == "string") and
    (.base.ref | type == "string") and
    (.base.commit | test("^[0-9a-f]{40}$")) and
    (.result.tree | test("^[0-9a-f]{40}$")) and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$"))
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
done

jq -e '
  .base.ref == "refs/heads/dev/infernal-invocation" and
  .result.tree == "174c789e09984049d0d53b261024460ca5e9c449" and
  any(.research_changes[];
    .status == "research-only" and
    (.purpose | contains("BF16 all-reduce row count")))
' "${composition_root}/vllm/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/master" and
  (.pull_requests | length) == 0 and
  .result.tree == "12c426322cc5d239023b57a4bd5ab0e60c4302e0" and
  any(.research_changes[];
    .status == "research-only" and
    (.purpose | contains("exact TP2 peer-push all-reduce")))
' "${composition_root}/b12x/integration.lock.json" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'revision=r19' <<<"${output}"
grep -Fxq 'vllm_tree=174c789e09984049d0d53b261024460ca5e9c449' <<<"${output}"
grep -Fxq 'b12x_tree=12c426322cc5d239023b57a4bd5ab0e60c4302e0' <<<"${output}"
grep -Fq '20260823-r19' <<<"${output}"
grep -Fq 'releases/infernal-invocation-r19/vllm/integration.patch' <<<"${output}"
grep -Fq 'releases/infernal-invocation-r19/b12x/integration.patch' <<<"${output}"

config="$(docker compose -f "${compose_file}" config)"
grep -Fq 'MODE: dspark-mtp0' <<<"${config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${config}"
grep -Fq 'TP_SIZE: "2"' <<<"${config}"
grep -Fq 'DCP_SIZE: "1"' <<<"${config}"
grep -Fq 'MAX_NUM_SEQS: "32"' <<<"${config}"
grep -Fq 'MAX_MODEL_LEN: "1048576"' <<<"${config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: "0.975"' <<<"${config}"
grep -Fq 'ALLREDUCE_MODE: auto' <<<"${config}"
grep -Fq 'LOAD_FORMAT: instanttensor' <<<"${config}"
grep -Fq 'INSTANTTENSOR_BACKEND: BUFFERED' <<<"${config}"
grep -Fq 'NATIVE_L2_GB: ""' <<<"${config}"
grep -Fq 'NATIVE_L2_PATH: ""' <<<"${config}"

for component in VLLM B12X LMCACHE; do
  grep -Fq -- "--build-arg \"${component}_UPSTREAM_BASE=\${${component}_UPSTREAM_BASE}\"" "${builder}"
  grep -Fq -- "--build-arg \"${component}_MERGE_HEADS=\${${component}_MERGE_HEADS}\"" "${builder}"
done

grep -Fq 'DS4 launch: mode=dspark depth=fixed backend=b12x-a8' "${builder}"
grep -Fq 'tp=2 dcp=1 max_seqs=16 graph=96' "${builder}"
