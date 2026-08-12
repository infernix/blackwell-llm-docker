#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
launcher="${repo_root}/launchers/serve-infernal-invocation.sh"

bash -n "${launcher}"
grep -Fq \
  'GLM52_SERVER="${GLM52_SERVER:-/usr/local/bin/serve-glm52-v19.sh}"' \
  "${launcher}"
grep -Fq 'exec /usr/local/bin/serve-gilded-gnosis.sh "$@"' "${launcher}"

for required in \
  serve-gilded-gnosis.sh \
  serve-glm52-v16.sh \
  serve-glm52-v19.sh \
  serve-glm52-hybrid-v19.sh \
  glm52-dcp-prefill-policy.sh \
  glm52-pcie-runtime-env.sh \
  glm52-pcie-calibration.py \
  lmcache-mp-wrapper.sh; do
  test -f "${repo_root}/launchers/${required}"
done

echo 'Infernal Invocation GLM launcher contract: PASS'
