#!/usr/bin/env bash
set -euo pipefail

# Preserve the GLM-specific entrypoint while all model families share one
# LMCache multiprocessing contract.
wrapper_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${wrapper_dir}/lmcache-mp-wrapper.sh" "$@"
