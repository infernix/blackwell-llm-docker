#!/usr/bin/env bash
set -euo pipefail

# Install the patched NCCL library at one canonical path and make every CUDA
# consumer resolve libnccl.so.2 to that file. The PyTorch wheel is preserved in
# a pip-independent directory so later dependency transactions cannot remove
# the rollback artifact or leave PyTorch with a dangling library link.

python_bin=${PYTHON_BIN:-/opt/venv/bin/python}
venv_root=${VENV_ROOT:-/opt/venv}
site_packages=${SITE_PACKAGES:-}
expected_bundled_version=${TORCH_BUNDLED_NCCL_VERSION:?TORCH_BUNDLED_NCCL_VERSION is required}
local_nccl=${LOCAL_NCCL:-/opt/libnccl-local-inference.so.2.30.4}
canonical_nccl=${CANONICAL_NCCL:-/opt/libnccl.so.2.30.4}
system_nccl_link=${SYSTEM_NCCL_LINK:-/opt/libnccl.so.2}
readelf_bin=${READELF_BIN:-readelf}
ldconfig_bin=${LDCONFIG_BIN:-ldconfig}
verify_import=${VERIFY_IMPORT:-1}

if [[ -z "${site_packages}" ]]; then
  site_packages="$(${python_bin} -c 'import site; print(site.getsitepackages()[0])')"
fi

installed_version="$(${python_bin} -c \
  'import importlib.metadata as md; print(md.version("nvidia-nccl-cu13"))')"
[[ "${installed_version}" == "${expected_bundled_version}" ]] || {
  printf 'nvidia-nccl-cu13 version %s does not match expected version %s\n' \
    "${installed_version}" "${expected_bundled_version}" >&2
  exit 1
}

torch_nccl="${site_packages}/nvidia/nccl/lib/libnccl.so.2"
bundled_nccl="${venv_root}/.local-inference/nccl/${expected_bundled_version}/libnccl.so.2"

mkdir -p "$(dirname "${torch_nccl}")" "$(dirname "${bundled_nccl}")"
if [[ ! -e "${bundled_nccl}" ]]; then
  [[ -e "${torch_nccl}" ]] || {
    printf 'PyTorch bundled NCCL is unavailable at %s\n' "${torch_nccl}" >&2
    exit 1
  }
  cp -aL "${torch_nccl}" "${bundled_nccl}"
fi

if [[ ! -e "${canonical_nccl}" ]]; then
  [[ -e "${local_nccl}" ]] || {
    printf 'Patched NCCL is unavailable at %s\n' "${local_nccl}" >&2
    exit 1
  }
  cp -aL "${local_nccl}" "${canonical_nccl}"
fi
if [[ ! -e "${local_nccl}.orig" ]]; then
  cp -aL "${canonical_nccl}" "${local_nccl}.orig"
fi

"${readelf_bin}" -d "${canonical_nccl}" \
  | grep -Fq 'Library soname: [libnccl.so.2]'
ln -sfn "${canonical_nccl}" "${local_nccl}"
ln -sfn "${canonical_nccl}" "${system_nccl_link}"
ln -sfn "${canonical_nccl}" "${torch_nccl}"
"${ldconfig_bin}"

if [[ "${verify_import}" == 1 ]]; then
  "${python_bin}" - "${canonical_nccl}" <<'PY'
import sys

import torch

expected = sys.argv[1]
paths = []
with open("/proc/self/maps", encoding="utf-8") as maps:
    for line in maps:
        if "libnccl" in line:
            paths.append(line.split()[-1])

unique = sorted(set(paths))
print("torch nccl api", torch.cuda.nccl.version())
print("mapped libnccl paths", unique)
assert unique == [expected], unique
PY
elif [[ "${verify_import}" != 0 ]]; then
  printf 'VERIFY_IMPORT must be 0 or 1; got %s\n' "${verify_import}" >&2
  exit 2
fi
