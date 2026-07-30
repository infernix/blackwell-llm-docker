#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d)"
trap 'rm -rf "${tmp_root}"' EXIT

output="$({
  DRY_RUN=1 \
  XDG_CACHE_HOME="${tmp_root}/cache" \
  TMPDIR="${tmp_root}/tmp" \
  MODEL=/tmp/model \
  LOAD_FORMAT=dummy \
  MAX_NUM_SEQS=1 \
  GRAPH=6 \
  bash "${repo_root}/launchers/serve-glm52-v16.sh" \
    --profiler-config.profiler=torch \
    --profiler-config.torch_profiler_dir='/tmp/profile with spaces' \
    --max-num-seqs=3
} 2>&1)"

command_line="$(grep -F 'Command:' <<<"${output}")"
grep -Fq -- '--profiler-config.profiler=torch' <<<"${command_line}"
grep -Fq -- '--profiler-config.torch_profiler_dir=/tmp/profile\ with\ spaces' \
  <<<"${command_line}"
grep -Fq -- '--max-num-seqs=3' <<<"${command_line}"

# Keep every public GLM wrapper forwarding arguments to the terminal launcher.
grep -Fq 'glm52-exl3|exl3)' \
  "${repo_root}/launchers/serve-gilded-gnosis.sh"
grep -Fq 'model_command=(/usr/local/bin/serve-glm52-hybrid-v19.sh "$@")' \
  "${repo_root}/launchers/serve-gilded-gnosis.sh"
grep -Fq 'model_command=("${glm52_server}" "$@")' \
  "${repo_root}/launchers/serve-gilded-gnosis.sh"
grep -Fq 'exec /usr/local/bin/serve-glm52-v19.sh "$@"' \
  "${repo_root}/launchers/serve-glm52-hybrid-v19.sh"
grep -Fq 'exec /usr/local/bin/serve-glm52-v16.sh "$@"' \
  "${repo_root}/launchers/serve-glm52-v19.sh"
grep -Fq 'exec /usr/local/bin/serve-glm52-v16.sh "$@"' \
  "${repo_root}/launchers/serve-glm52-v18.sh"

echo "GLM-5.2 launcher extra arguments: PASS"
