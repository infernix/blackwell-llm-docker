#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_script="${repo_root}/build-gilded-gnosis-v20-final-cu132.sh"

# `export NAME=$(jq ...)` returns export's status and masks a failed lookup
# under `set -e`, with or without quotes. Assign each value before exporting it.
masked_export_pattern='^[[:space:]]*export[[:space:]]+[A-Z0-9_]+="?[[:space:]]*\$\([[:space:]]*(jq|sha256sum)'

for fixture in \
  'export VALUE="$(jq -er .value lock.json)"' \
  'export VALUE=$(jq -er .value lock.json)' \
  'export VALUE="$(sha256sum lock.json)"' \
  'export VALUE=$(sha256sum lock.json)'; do
  grep -Eq "${masked_export_pattern}" <<<"${fixture}" || {
    echo "fail-fast pattern missed masked export: ${fixture}" >&2
    exit 1
  }
done

if grep -Eq "${masked_export_pattern}" \
  "${build_script}"; then
  echo "release manifest lookup is masked by export" >&2
  exit 1
fi

bash -n "${build_script}"

mapfile -t launcher_refs < <(
  sed -n 's/^  export LAUNCHER_REF="${LAUNCHER_REF:-\([^}]*\)}"$/\1/p' \
    "${build_script}"
)
mapfile -t launcher_commits < <(
  sed -n 's/^  export LAUNCHER_COMMIT="${LAUNCHER_COMMIT:-\([^}]*\)}"$/\1/p' \
    "${build_script}"
)

[[ "${#launcher_refs[@]}" -gt 0 ]] || {
  echo "no default LAUNCHER_REF values found" >&2
  exit 1
}
[[ "${#launcher_refs[@]}" -eq "${#launcher_commits[@]}" ]] || {
  echo "default LAUNCHER_REF and LAUNCHER_COMMIT counts differ" >&2
  exit 1
}

for index in "${!launcher_refs[@]}"; do
  [[ "${launcher_refs[index]}" =~ ^[0-9a-f]{40}$ ]] || {
    echo "default LAUNCHER_REF is not an immutable commit" >&2
    exit 1
  }
  [[ "${launcher_refs[index]}" == "${launcher_commits[index]}" ]] || {
    echo "default LAUNCHER_REF and LAUNCHER_COMMIT differ" >&2
    exit 1
  }
done

assert_pr_set() {
  local manifest="$1"
  local expected="$2"
  local actual

  actual="$(jq -r '[.pull_requests[].number] | sort | map(tostring) | join(" ")' \
    "${repo_root}/${manifest}")"
  [[ "${actual}" == "${expected}" ]] || {
    echo "required PR set mismatch in ${manifest}: ${actual}" >&2
    exit 1
  }
}

# This explicit gate prevents a syntactically valid manifest from silently
# omitting a reviewed release dependency. Updating the set is a release action.
assert_pr_set \
  manifests/vllm/gilded-gnosis-v20.json \
  '145 188 213 214 217 218 228 229 230 234 235 245 248 251 252 253 254 255 256'
assert_pr_set \
  manifests/b12x/gilded-gnosis-v20.json \
  '125 126'
assert_pr_set \
  manifests/lmcache/gilded-gnosis-v20.json \
  '7 8 9 10 11 12 13 14 15 16 17'

echo "release manifest fail-fast contract: PASS"
