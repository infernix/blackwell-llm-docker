#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_script="${repo_root}/build-gilded-gnosis-v20-final-cu132.sh"

# `export NAME="$(jq ...)"` returns export's status and masks a failed lookup
# under `set -e`. Assign each manifest value before exporting it.
if grep -Eq '^[[:space:]]*export[[:space:]]+[A-Z0-9_]+="\$\((jq|sha256sum)' \
  "${build_script}"; then
  echo "release manifest lookup is masked by export" >&2
  exit 1
fi

bash -n "${build_script}"
echo "release manifest fail-fast contract: PASS"
