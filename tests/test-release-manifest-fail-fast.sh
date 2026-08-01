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

echo "release manifest fail-fast contract: PASS"
