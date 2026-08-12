#!/usr/bin/env bash
set -euo pipefail

# Unified entrypoint for model families qualified by the Infernal Invocation
# runtime. The model-family dispatcher owns the public environment interface;
# the GLM-5.2 implementation owns its DCP and quantization policy.
export GLM52_SERVER="${GLM52_SERVER:-/usr/local/bin/serve-glm52-v19.sh}"

exec /usr/local/bin/serve-gilded-gnosis.sh "$@"
