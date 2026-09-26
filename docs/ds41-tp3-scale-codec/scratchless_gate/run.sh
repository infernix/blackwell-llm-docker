#!/bin/bash
set -eu
root=$(realpath "$(dirname "$0")/../../..")
work="$root/docs/ds41-tp3-scale-codec"
b12x_source=${B12X_SOURCE_DIR:-"$root/../b12x"}
model=${MODEL_PATH:?set MODEL_PATH to the DeepSeek-V4.1-Flash checkpoint directory}
image=${IMAGE:-localhost/ds41-tp3:assembled}
mkdir -p "$work/scratchless_gate/cache"
entry=python
args=()
if [ "${SANITIZE:-0}" = 1 ]; then
  entry=compute-sanitizer
  args=(--tool memcheck --error-exitcode 86 python)
fi
exec podman run --rm --pull never --device nvidia.com/gpu=0 --security-opt seccomp=unconfined --ipc=host \
  -e B12X_AUTOTUNE=0 -e B12X_COMPILE_CACHE_DIR=/gate-cache \
  -v "$work:/work" \
  -v "$work/scratchless_gate/cache:/gate-cache" \
  -v "$model:/model:ro" \
  -v "$b12x_source/b12x/moe/fused_moe/_impl.py:/opt/venv/lib/python3.12/site-packages/b12x/moe/fused_moe/_impl.py:ro" \
  -v "$b12x_source/b12x/moe/fused_moe/api.py:/opt/venv/lib/python3.12/site-packages/b12x/moe/fused_moe/api.py:ro" \
  -v "$b12x_source/b12x/moe/_shared/kernels/dynamic.py:/opt/venv/lib/python3.12/site-packages/b12x/moe/_shared/kernels/dynamic.py:ro" \
  --entrypoint "$entry" "$image" \
  "${args[@]}" /work/scratchless_gate/gate.py --report /work/scratchless_gate/report.json "$@"
