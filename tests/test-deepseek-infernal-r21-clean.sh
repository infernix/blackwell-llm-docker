#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-infernal-invocation-cu133-torch213.sh"
target_compose="${repo_root}/examples/docker-compose-ds4-infernal-invocation-cu133-r21.yml"
dspark_compose="${repo_root}/examples/docker-compose-ds4-dspark-infernal-invocation-cu133-r21.yml"
composition_root="${repo_root}/patches/releases/infernal-invocation-r21"

for component in vllm b12x lmcache; do
  lock="${composition_root}/${component}/integration.lock.json"
  patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}"
  test -f "${patch}"
  jq -e '
    .schema_version == 1 and
    .composition_strategy == "merge" and
    (.base.repository | type == "string") and
    (.base.ref | type == "string") and
    (.base.commit | test("^[0-9a-f]{40}$")) and
    (.result.tree | test("^[0-9a-f]{40}$")) and
    (.result.patch_sha256 | test("^[0-9a-f]{64}$")) and
    ((.research_changes // []) | length) == 0 and
    (.source_patches | length) == 0
  ' "${lock}" >/dev/null
  echo "$(jq -er '.result.patch_sha256' "${lock}")  ${patch}" |
    sha256sum -c - >/dev/null
done

jq -e '
  .base.ref == "refs/heads/dev/infernal-invocation" and
  .base.commit == "b5f995e73e6b7fe27c9927477e277a151ebcc9e9" and
  .result.tree == "d6cf36ae0dc30d48fd656a3c34a353ec62074922" and
  (.pull_requests | length) == 31 and
  any(.pull_requests[]; .number == 482) and
  any(.pull_requests[]; .number == 483) and
  any(.pull_requests[]; .number == 485)
' "${composition_root}/vllm/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/master" and
  .base.commit == "a71c705f1c4710f59129562d26c73e70098e29de" and
  .result.tree == "f6dc512eb13ac2c09b2bf53656c704081af64361" and
  [.pull_requests[].number] == [243, 246, 247]
' "${composition_root}/b12x/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/release/v0.5.2-glm52-dcp-base" and
  .base.commit == "a128b2e286ebb3556cb43124149e600ff99fe481" and
  .result.tree == "e045d729bc5c4c63a40e13d032f42923de97812f" and
  (.pull_requests | length) == 13
' "${composition_root}/lmcache/integration.lock.json" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'revision=r21' <<<"${output}"
grep -Fxq 'vllm_tree=d6cf36ae0dc30d48fd656a3c34a353ec62074922' <<<"${output}"
grep -Fxq 'b12x_tree=f6dc512eb13ac2c09b2bf53656c704081af64361' <<<"${output}"
grep -Fq '20260827-r21' <<<"${output}"
grep -Fq 'releases/infernal-invocation-r21/vllm/integration.patch' <<<"${output}"
grep -Fq 'releases/infernal-invocation-r21/b12x/integration.patch' <<<"${output}"
grep -Fq 'base=voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2' <<<"${output}"
grep -Fxq 'runtime_foundation=1' <<<"${output}"

target_config="$(docker compose -f "${target_compose}" config)"
grep -Fq 'MODE: dspark-mtp0' <<<"${target_config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${target_config}"
grep -Fq 'TP_SIZE: "2"' <<<"${target_config}"
grep -Fq 'MAX_NUM_SEQS: "32"' <<<"${target_config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${target_config}"
grep -Fq 'ALLREDUCE_MODE: auto' <<<"${target_config}"

dspark_config="$(docker compose -f "${dspark_compose}" config)"
grep -Fq 'MODE: dspark' <<<"${dspark_config}"
grep -Fq 'DSPARK_TOKENS: "5"' <<<"${dspark_config}"
grep -Fq 'MAX_NUM_SEQS: "8"' <<<"${dspark_config}"
grep -Fq 'GRAPH: auto' <<<"${dspark_config}"
grep -Fq 'MAX_MODEL_LEN: "1048576"' <<<"${dspark_config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: "0.975"' <<<"${dspark_config}"

for component in VLLM B12X LMCACHE; do
  grep -Fq -- "--build-arg \"${component}_UPSTREAM_BASE=\${${component}_UPSTREAM_BASE}\"" "${builder}"
  grep -Fq -- "--build-arg \"${component}_MERGE_HEADS=\${${component}_MERGE_HEADS}\"" "${builder}"
done

grep -Fq -- '--build-arg "RUNTIME_FOUNDATION=${runtime_foundation}"' "${builder}"
grep -Fq -- '--build-arg "RUNTIME_FOUNDATION_IMAGE=${runtime_foundation_image}"' "${builder}"
