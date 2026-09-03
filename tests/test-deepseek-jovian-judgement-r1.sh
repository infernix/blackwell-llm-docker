#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
builder="${repo_root}/build-deepseek-jovian-judgement-cu133-torch213.sh"
target_compose="${repo_root}/examples/docker-compose-ds4-jovian-judgement-r1.yml"
dspark_compose="${repo_root}/examples/docker-compose-ds4-dspark-jovian-judgement-r1.yml"
composition_root="${repo_root}/patches/releases/jovian-judgement-ds4-r1"

for component in vllm b12x lmcache; do
  lock="${composition_root}/${component}/integration.lock.json"
  patch="${composition_root}/${component}/integration.patch"

  test -f "${lock}"
  test -f "${patch}"
  jq -e '
    .schema_version == 1 and
    (.composition_strategy == "merge" or
      .composition_strategy == "cherry_pick") and
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
  .composition_strategy == "cherry_pick" and
  .base.ref == "refs/heads/dev/jovian-judgement" and
  .base.commit == "54371894ecaa77f2725a1c99e018f3fe93d358dd" and
  .result.tree == "3f2d4c33d221d0ba1116f077aceb98168c86d6a7" and
  [.pull_requests[].number] == [628] and
  .pull_requests[0].head ==
    "cbb66bdff1763c174ebc794a7f968930e956580f"
' "${composition_root}/vllm/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/master" and
  .base.commit == "694e7650f0ea96997ee0d056e39e5764ddd74d5c" and
  .composition_strategy == "cherry_pick" and
  .result.tree == "a7e71747e0703101d9c8359564774078455076d8" and
  [.pull_requests[].number] == [246, 302] and
  .pull_requests[0].head ==
    "2a2baaf3979c1f47f8fd139aab5c6eeea35003e4" and
  .pull_requests[1].head ==
    "f8a7b9c4070754ed4399c71fc4306d8970711e1c"
' "${composition_root}/b12x/integration.lock.json" >/dev/null

jq -e '
  .base.ref == "refs/heads/release/v0.5.2-glm52-dcp-base" and
  .base.commit == "a128b2e286ebb3556cb43124149e600ff99fe481" and
  .composition_strategy == "merge" and
  .result.tree == "1dfee6811b7b5ee735620337a8f3d93c8b560658" and
  (.pull_requests | length) == 14 and
  .pull_requests[-1].number == 44 and
  .pull_requests[-1].head ==
    "a0cb4153ab9227c6503ccc5e3f4613087c8f8862"
' "${composition_root}/lmcache/integration.lock.json" >/dev/null

output="$(PRINT_RELEASE_CONFIG=1 "${builder}")"
grep -Fxq 'release=jovian-judgement-deepseek-v4-flash-cu133-torch213' <<<"${output}"
grep -Fxq 'revision=r1' <<<"${output}"
grep -Fxq 'vllm_ref=dev/jovian-judgement' <<<"${output}"
grep -Fxq 'vllm_tree=3f2d4c33d221d0ba1116f077aceb98168c86d6a7' <<<"${output}"
grep -Fxq 'b12x_tree=a7e71747e0703101d9c8359564774078455076d8' <<<"${output}"
grep -Fq '20260903-r1' <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r1/vllm/integration.patch' <<<"${output}"
grep -Fq 'releases/jovian-judgement-ds4-r1/b12x/integration.patch' <<<"${output}"
grep -Fq 'base=voipmonitor/vllm@sha256:03b67e53dda73c3fa317d4cb529ad38a220c51c7365ee8d54c16e5063fcc54e2' <<<"${output}"
grep -Fxq 'runtime_foundation=1' <<<"${output}"

target_config="$(docker compose -f "${target_compose}" config)"
grep -Fq 'MODE: dspark-mtp0' <<<"${target_config}"
grep -Fq 'BACKEND: b12x-a8-dglin' <<<"${target_config}"
grep -Fq 'TP_SIZE: "2"' <<<"${target_config}"
grep -Fq 'MAX_NUM_SEQS: "32"' <<<"${target_config}"
grep -Fq 'MAX_NUM_BATCHED_TOKENS: "4096"' <<<"${target_config}"
grep -Fq 'ALLREDUCE_MODE: auto' <<<"${target_config}"
grep -Fq 'jovian-judgement-vllm3f2d4c3-b12xa7e7174' <<<"${target_config}"

dspark_config="$(docker compose -f "${dspark_compose}" config)"
grep -Fq 'MODE: dspark' <<<"${dspark_config}"
grep -Fq 'DSPARK_TOKENS: "5"' <<<"${dspark_config}"
grep -Fq 'MAX_NUM_SEQS: "8"' <<<"${dspark_config}"
grep -Fq 'GRAPH: auto' <<<"${dspark_config}"
grep -Fq 'MAX_MODEL_LEN: "1048576"' <<<"${dspark_config}"
grep -Fq 'GPU_MEMORY_UTILIZATION: "0.975"' <<<"${dspark_config}"
grep -Fq 'jovian-judgement-vllm3f2d4c3-b12xa7e7174' <<<"${dspark_config}"

for component in VLLM B12X LMCACHE; do
  grep -Fq -- "--build-arg \"${component}_UPSTREAM_BASE=\${${component}_UPSTREAM_BASE}\"" "${builder}"
  grep -Fq -- "--build-arg \"${component}_MERGE_HEADS=\${${component}_MERGE_HEADS}\"" "${builder}"
done

grep -Fq -- '--build-arg "RUNTIME_FOUNDATION=${runtime_foundation}"' "${builder}"
grep -Fq -- '--build-arg "RUNTIME_FOUNDATION_IMAGE=${runtime_foundation_image}"' "${builder}"
