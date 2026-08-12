#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
temporary="$(mktemp -d)"
trap 'rm -rf "${temporary}"' EXIT

venv_root="${temporary}/venv"
site_packages="${venv_root}/lib/python3.12/site-packages"
torch_nccl="${site_packages}/nvidia/nccl/lib/libnccl.so.2"
local_nccl="${temporary}/libnccl-local.so.2.30.4"
canonical_nccl="${temporary}/libnccl.so.2.30.4"
system_nccl_link="${temporary}/libnccl.so.2"
mkdir -p "$(dirname "${torch_nccl}")" "${temporary}/bin"
printf 'bundled-library\n' >"${torch_nccl}"
printf 'patched-library\n' >"${local_nccl}"

cat >"${temporary}/bin/python" <<'EOF'
#!/usr/bin/env bash
printf '2.29.7\n'
EOF
cat >"${temporary}/bin/readelf" <<'EOF'
#!/usr/bin/env bash
printf ' 0x000000000000000e (SONAME) Library soname: [libnccl.so.2]\n'
EOF
cat >"${temporary}/bin/ldconfig" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "${temporary}/bin/python" "${temporary}/bin/readelf" \
  "${temporary}/bin/ldconfig"

env \
  PYTHON_BIN="${temporary}/bin/python" \
  VENV_ROOT="${venv_root}" \
  SITE_PACKAGES="${site_packages}" \
  TORCH_BUNDLED_NCCL_VERSION=2.29.7 \
  LOCAL_NCCL="${local_nccl}" \
  CANONICAL_NCCL="${canonical_nccl}" \
  SYSTEM_NCCL_LINK="${system_nccl_link}" \
  READELF_BIN="${temporary}/bin/readelf" \
  LDCONFIG_BIN="${temporary}/bin/ldconfig" \
  VERIFY_IMPORT=0 \
  "${repo_root}/scripts/install-patched-nccl.sh"

bundled_nccl="${venv_root}/.local-inference/nccl/2.29.7/libnccl.so.2"
grep -Fxq 'bundled-library' "${bundled_nccl}"
grep -Fxq 'patched-library' "${canonical_nccl}"
grep -Fxq 'patched-library' "${local_nccl}.orig"
[[ "$(readlink "${torch_nccl}")" == "${canonical_nccl}" ]]
[[ "$(readlink "${local_nccl}")" == "${canonical_nccl}" ]]
[[ "$(readlink "${system_nccl_link}")" == "${canonical_nccl}" ]]

# Reinstallation is idempotent and keeps the separately preserved wheel file.
env \
  PYTHON_BIN="${temporary}/bin/python" \
  VENV_ROOT="${venv_root}" \
  SITE_PACKAGES="${site_packages}" \
  TORCH_BUNDLED_NCCL_VERSION=2.29.7 \
  LOCAL_NCCL="${local_nccl}" \
  CANONICAL_NCCL="${canonical_nccl}" \
  SYSTEM_NCCL_LINK="${system_nccl_link}" \
  READELF_BIN="${temporary}/bin/readelf" \
  LDCONFIG_BIN="${temporary}/bin/ldconfig" \
  VERIFY_IMPORT=0 \
  "${repo_root}/scripts/install-patched-nccl.sh"
grep -Fxq 'bundled-library' "${bundled_nccl}"
