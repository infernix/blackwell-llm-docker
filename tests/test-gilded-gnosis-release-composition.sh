#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
output_r6="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r6 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r6' <<<"${output_r6}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r6' \
  <<<"${output_r6}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r6' \
  <<<"${output_r6}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output_r6}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output_r6}"
grep -Fxq 'lmcache_repo=https://github.com/LMCache/LMCache.git' <<<"${output_r6}"
grep -Fxq 'lmcache_ref=v0.5.2' <<<"${output_r6}"
grep -Fxq \
  'lmcache_commit=cd2c0d6a6a982ec5e334bae7704e1029c06d3c97' \
  <<<"${output_r6}"
grep -Fxq 'lmcache_patch=lmcache/glm52-dcp-v052.patch' <<<"${output_r6}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.2' <<<"${output_r6}"

output_r7="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r7 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r7' <<<"${output_r7}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r7' \
  <<<"${output_r7}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r7' \
  <<<"${output_r7}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output_r7}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output_r7}"
grep -Fxq \
  'lmcache_repo=https://github.com/local-inference-lab/LMCache.git' \
  <<<"${output_r7}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output_r7}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r7}"
grep -Fxq 'lmcache_patch=' <<<"${output_r7}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.3' <<<"${output_r7}"
grep -Fxq 'xgrammar_ref=' <<<"${output_r7}"
grep -Fxq 'xgrammar_commit=' <<<"${output_r7}"
grep -Fxq 'xgrammar_version=' <<<"${output_r7}"
grep -Fxq 'xgrammar_transformers5_compat=0' <<<"${output_r7}"

output_r8="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r8 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r8' <<<"${output_r8}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm936ed48-sif532ec9-fi801d57a-cu132-20260728-r8' \
  <<<"${output_r8}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm936ed48.sif532ec9.fi801d57a.cu132.20260728.r8' \
  <<<"${output_r8}"
grep -Fxq \
  'vllm_tree=936ed4829ed6b6a34b9052a7a2614333ee3b2623' \
  <<<"${output_r8}"
grep -Fxq \
  'sparkinfer_tree=f532ec965a70b710ba45e6f751fe5d7135001108' \
  <<<"${output_r8}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r8}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r8}"
grep -Fxq \
  'xgrammar_commit=2ea71da4ccb997a06928c9fb69b99f330da56697' \
  <<<"${output_r8}"
grep -Fxq 'xgrammar_version=0.2.5' <<<"${output_r8}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r8}"

output_r11="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r11 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r11' <<<"${output_r11}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm9502cc7-side7739a-fi801d57a-cu132-20260729-r11' \
  <<<"${output_r11}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm9502cc7.side7739a.fi801d57a.cu132.20260729.r11' \
  <<<"${output_r11}"
grep -Fxq \
  'vllm_tree=9502cc7ee9dbc060899a2b1d30fac6916d3a4a95' \
  <<<"${output_r11}"
grep -Fxq \
  'sparkinfer_tree=de7739aa7ebc8a52eb1f1997367eee1d0a6bab79' \
  <<<"${output_r11}"
grep -Fxq \
  'lmcache_tree=175e59294436a02861e9e3ebedf7358edea4ed36' \
  <<<"${output_r11}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r11}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r11/lmcache/integration.patch' \
  <<<"${output_r11}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r11}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r11}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r11}"

output_r12="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r12 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r12' <<<"${output_r12}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllmb46c3aa-si35aebc6-fi801d57a-cu132-20260730-r12' \
  <<<"${output_r12}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllmb46c3aa.si35aebc6.fi801d57a.cu132.20260730.r12' \
  <<<"${output_r12}"
grep -Fxq \
  'vllm_tree=b46c3aac9421ec6c03b9dfcb5aacc8c1eb11f09b' \
  <<<"${output_r12}"
grep -Fxq \
  'sparkinfer_tree=35aebc6d46a0d3d4f7275b81251211864f410269' \
  <<<"${output_r12}"
grep -Fxq \
  'launcher_ref=513bd84a1d8f4b834ca343abb4189e82acb1df52' \
  <<<"${output_r12}"
grep -Fxq \
  'launcher_commit=513bd84a1d8f4b834ca343abb4189e82acb1df52' \
  <<<"${output_r12}"
grep -Fxq \
  'lmcache_tree=175e59294436a02861e9e3ebedf7358edea4ed36' \
  <<<"${output_r12}"
grep -Fxq \
  'lmcache_repo=https://github.com/local-inference-lab/LMCache.git' \
  <<<"${output_r12}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output_r12}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r12}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r12/lmcache/integration.patch' \
  <<<"${output_r12}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r12}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r12}"
grep -Fxq \
  'xgrammar_commit=2ea71da4ccb997a06928c9fb69b99f330da56697' \
  <<<"${output_r12}"
grep -Fxq 'xgrammar_version=0.2.5' <<<"${output_r12}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r12}"

output_r13="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r13 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r13' <<<"${output_r13}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm69ba80b-sia2ea608-fi801d57a-cu132-20260730-r13' \
  <<<"${output_r13}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm69ba80b.sia2ea608.fi801d57a.cu132.20260730.r13' \
  <<<"${output_r13}"
grep -Fxq \
  'vllm_tree=69ba80b9e49b5ad6740f8b3d5d0e592213b25959' \
  <<<"${output_r13}"
grep -Fxq \
  'sparkinfer_tree=a2ea6083713c15dcf7e2d2bcc74fbece837ff84d' \
  <<<"${output_r13}"
grep -Fxq \
  'launcher_ref=513bd84a1d8f4b834ca343abb4189e82acb1df52' \
  <<<"${output_r13}"
grep -Fxq \
  'launcher_commit=513bd84a1d8f4b834ca343abb4189e82acb1df52' \
  <<<"${output_r13}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r13}"
grep -Fxq \
  'lmcache_repo=https://github.com/local-inference-lab/LMCache.git' \
  <<<"${output_r13}"
grep -Fxq 'lmcache_ref=release/v0.5.2-glm52-dcp-base' <<<"${output_r13}"
grep -Fxq \
  'lmcache_commit=9cebd405d0caf4bebe01d694b5a8bf4e3e354314' \
  <<<"${output_r13}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r13/lmcache/integration.patch' \
  <<<"${output_r13}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r13}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r13}"
grep -Fxq \
  'xgrammar_commit=2ea71da4ccb997a06928c9fb69b99f330da56697' \
  <<<"${output_r13}"
grep -Fxq 'xgrammar_version=0.2.5' <<<"${output_r13}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r13}"

output_r14="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r14 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r14' <<<"${output_r14}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm749050e-si8110e3e-fi801d57a-cu132-20260730-r14' \
  <<<"${output_r14}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm749050e.si8110e3e.fi801d57a.cu132.20260730.r14' \
  <<<"${output_r14}"
grep -Fxq \
  'vllm_tree=749050edab1b6664937c52fa1b0be360be632c1e' \
  <<<"${output_r14}"
grep -Fxq \
  'sparkinfer_tree=8110e3ea417794bfb08aff1fba20135102e5536b' \
  <<<"${output_r14}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r14}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r14/lmcache/integration.patch' \
  <<<"${output_r14}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r14}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r14}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r14}"

output_r15="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r15 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r15' <<<"${output_r15}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm0bc48c5-sieec30ff-fi801d57a-cu132-20260731-r15' \
  <<<"${output_r15}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm0bc48c5.sieec30ff.fi801d57a.cu132.20260731.r15' \
  <<<"${output_r15}"
grep -Fxq \
  'vllm_tree=0bc48c5943561c56353ce1f8047f81d5e0517237' \
  <<<"${output_r15}"
grep -Fxq \
  'sparkinfer_tree=eec30ff294c1870b59a04686fff6608fddb62089' \
  <<<"${output_r15}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r15}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r15/lmcache/integration.patch' \
  <<<"${output_r15}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r15}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r15}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r15}"

output_r16="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r16 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r16' <<<"${output_r16}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllm1e9c9c3-sieec30ff-fi801d57a-cu132-20260731-r16' \
  <<<"${output_r16}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllm1e9c9c3.sieec30ff.fi801d57a.cu132.20260731.r16' \
  <<<"${output_r16}"
grep -Fxq \
  'vllm_tree=1e9c9c3475fa30ab48d5639f8882f1e93bb552bf' \
  <<<"${output_r16}"
grep -Fxq \
  'sparkinfer_tree=eec30ff294c1870b59a04686fff6608fddb62089' \
  <<<"${output_r16}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r16}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r16/lmcache/integration.patch' \
  <<<"${output_r16}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r16}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r16}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r16}"

output_r18="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r18 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r18' <<<"${output_r18}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllmab358b1-sib2bff71-fi801d57a-cu132-20260801-r18' \
  <<<"${output_r18}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllmab358b1.sib2bff71.fi801d57a.cu132.20260801.r18' \
  <<<"${output_r18}"
grep -Fxq \
  'vllm_tree=ab358b11844ab626ca227be455165e336d5f855a' \
  <<<"${output_r18}"
grep -Fxq \
  'sparkinfer_tree=b2bff719ba1be0a5d30cb39cba795f0812db0f3d' \
  <<<"${output_r18}"
grep -Fxq \
  'launcher_ref=3ce8fc75c1bef3f6c1204ddb9bb133bc1c31245f' \
  <<<"${output_r18}"
grep -Fxq \
  'launcher_commit=3ce8fc75c1bef3f6c1204ddb9bb133bc1c31245f' \
  <<<"${output_r18}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r18}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r18/lmcache/integration.patch' \
  <<<"${output_r18}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r18}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r18}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r18}"

output_r17="$(
  cd "${repo_root}"
  PRINT_RELEASE_CONFIG=1 VLLM_RELEASE_COMPOSITION=reproduce-r17 \
    ./build-gilded-gnosis-v20-final-cu132.sh
)"

grep -Fxq 'composition=reproduce-r17' <<<"${output_r17}"
grep -Fxq \
  'image=voipmonitor/vllm:gilded-gnosis-v20-vllmdb29328-sib2bff71-fi801d57a-cu132-20260801-r17' \
  <<<"${output_r17}"
grep -Fxq \
  'version=0.11.2.dev280+gilded.gnosis.v20.vllmdb29328.sib2bff71.fi801d57a.cu132.20260801.r17' \
  <<<"${output_r17}"
grep -Fxq \
  'vllm_tree=db293280d021d32db0552f3f6e4b95abbd9c69a1' \
  <<<"${output_r17}"
grep -Fxq \
  'sparkinfer_tree=b2bff719ba1be0a5d30cb39cba795f0812db0f3d' \
  <<<"${output_r17}"
grep -Fxq \
  'launcher_ref=ec0279f1c2ccf06656df21d65c7a18984c45fcd8' \
  <<<"${output_r17}"
grep -Fxq \
  'launcher_commit=ec0279f1c2ccf06656df21d65c7a18984c45fcd8' \
  <<<"${output_r17}"
grep -Fxq \
  'lmcache_tree=a5aa59cc8edca462a3f4c198d17fd2b9c1a7ffaa' \
  <<<"${output_r17}"
grep -Fxq \
  'lmcache_patch=releases/gilded-gnosis-v20-r17/lmcache/integration.patch' \
  <<<"${output_r17}"
grep -Fxq 'lmcache_version=0.5.2+glm52dcp.4' <<<"${output_r17}"
grep -Fxq 'xgrammar_ref=v0.2.5' <<<"${output_r17}"
grep -Fxq 'xgrammar_transformers5_compat=1' <<<"${output_r17}"

echo 'Gilded Gnosis release composition: PASS'
