# Blackwell LLM Docker

Docker images for LLM inference on NVIDIA Blackwell GPUs (SM120).

## Images

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/sglang:cu130` | `Dockerfile.sglang-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), SGLang + b12x + PCIe allreduce |
| `voipmonitor/sglang:cu132` | `Dockerfile.sglang-cu132` | CUDA 13.2, torch 2.12 from source, FlashInfer source (PR #2913), SGLang + b12x |
| `voipmonitor/vllm:cu130` | `Dockerfile.vllm-cu130` | CUDA 13.0, torch 2.11 stable cu130, FlashInfer source (PR #2913), vLLM + cherry-picks |
| `voipmonitor/vllm:vllm-b12x-cu132` | `Dockerfile.vllm-b12x-cu132` | Clean CUDA 13.2.1, PyTorch 2.12 cu132 wheels, patched NCCL 2.30.4, FlashInfer, DeepGEMM, B12X, vLLM |
| `voipmonitor/vllm:lucifer` | `Dockerfile.vllm-b12x-cu132` | Lucifer DS4 Flash/CUTLASS vLLM branch on the same CUDA 13.2.1 base, FlashInfer, DeepGEMM, and Triton kernels source hook |

Base image for cu132 (torch + FlashInfer compiled from source):

| Image | Dockerfile | Stack |
|-------|-----------|-------|
| `voipmonitor/torch:cu132` | `Dockerfile.torch-cu132` | CUDA 13.2, torch 2.12 from source (no pip nvidia-*), FlashInfer from source |

## Quick start

```bash
# Qwen3.5-397B NVFP4 on 4x Blackwell GPUs
docker compose -f examples/docker-compose-qwen35.yml up -d

# GLM-5 NVFP4 on 8x Blackwell GPUs
docker compose -f examples/docker-compose-glm5.yml up -d
```

See `examples/` for full docker-compose files with hardware requirements and configuration options.

## Run

### With model profile

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 \
  -e MODEL_PROFILE=qwen35-b12x \
  voipmonitor/sglang:cu130
```

Available profiles: `qwen35-b12x`, `glm5-nvfp4` (see `profiles/` directory).

### Direct command

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v jit-cache:/cache/jit -p 5000:5000 \
  voipmonitor/sglang:cu130 \
  python -m sglang.launch_server --model-path <model> --tp 8 --host 0.0.0.0 --port 5000
```

### vLLM

```bash
docker run --gpus all --ipc=host --shm-size=8g \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 5000:5000 \
  voipmonitor/vllm:cu130 \
  --model <model> --tensor-parallel-size 4 --host 0.0.0.0 --port 5000
```

## Build

```bash
# SGLang cu130
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang-cu130 -t voipmonitor/sglang:cu130 .

# SGLang cu132 (requires torch base first)
docker build -f Dockerfile.torch-cu132 -t voipmonitor/torch:cu132 .
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.sglang-cu132 -t voipmonitor/sglang:cu132 .

# vLLM cu130
docker build --build-arg CACHEBUST=$(date +%s) -f Dockerfile.vllm-cu130 -t voipmonitor/vllm:cu130 .

# Clean vLLM+B12X cu132. This builds the reusable system/build base images
# first, then builds the final vLLM image from those base images.
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 ./build-vllm-b12x-cu132.sh

# Reproduce the pushed black-benediction PR11 image exactly.
./build-black-benediction-b12xpr11-cu132.sh

# Build the Lucifer DS4 Flash/CUTLASS image from local-inference-lab/vllm:lucifer.
./build-lucifer-cu132.sh

# Build the unified GLM-5.2 and DS4/DSpark v16 image from immutable vLLM,
# B12X, FlashInfer, DeepGEMM, CUTLASS, InstantTensor, and NCCL commits.
./build-fathomless-firmament-v16-cu132.sh

# Build the current unified v17 image with NF3/NVFP4-KV support and the
# validated TP4/TP6/TP8 sparse-MLA DCP prefill workspace paths.
./build-fathomless-firmament-v17-cu132.sh

# Build a new GG v20 candidate. This always resolves the current clean
# dev/gilded-gnosis and SparkInfer master heads and composes both pinned PR
# manifests from scratch.
./build-gilded-gnosis-v20-final-cu132.sh
```

### Clean GG release composition

New Gilded Gnosis images must not use an earlier `build/*` integration branch
as their source. `build-gilded-gnosis-v20-final-cu132.sh` reads the manifests
under `manifests/vllm/` and `manifests/sparkinfer/`, resolves the current
`dev/gilded-gnosis` and SparkInfer `master` heads, verifies every pinned PR
head, and creates fresh integration patches and lockfiles. The build stops if
either base advances, a PR changes, or a PR conflicts. The Dockerfile
independently verifies that applying each patch produces its locked Git tree
and records both bases, PR heads, trees, patch hashes, and lock hashes in image
labels.

The old r4 source can only be selected explicitly for reproducibility:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r4 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The validated r5 image is reproducible from its archived, hash-verified source
locks and integration patches even after either upstream branch advances:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r5 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r6 LMCache runtime uses the same immutable vLLM and SparkInfer trees as r5:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r6 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r7 candidate keeps those model stacks, but builds LMCache directly from
the merged `local-inference-lab/LMCache` release commit instead of applying a
container-local source patch:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r7 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

This historical mode still verifies the pinned base commits, patch hashes, and
resulting Git trees. It only skips the normal requirement that the current
remote branch heads remain equal to the archived base commits.

Current clean r8 candidates also build XGrammar `0.2.5` from the immutable
`v0.2.5` source commit. The image build verifies GLM `tool_choice=required`
semantics: at least one tool call is required, while multiple calls and normal
termination after a call remain valid. XGrammar caps Transformers below 5 for
tokenizer regressions in other model families; this GLM image removes only that
package-metadata cap and validates the pinned GLM tokenizer with its
Transformers 5 runtime. The override is recorded in the image labels.
Historical r4-r7 reproduction modes keep the XGrammar version and metadata
supplied by their original vLLM requirements. Reproduce the exact r8 source
composition with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r8 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

The r9 release retains the r8 runtime and adds three independently verifiable
changes:

- optional dynamic per-token NVFP4 MLA KV scaling from the paired vLLM #189
  and SparkInfer #86 ABI change;
- exact adaptive sparse-indexer folding from SparkInfer #87, which keeps the
  two-level reduction when it fits the configured workspace budget and falls
  back to exact streaming carry otherwise;
- `pytest==8.4.1` in the final `/opt/venv`, so focused tests can run against a
  deployed image without copying the repository test tree into the image.

The default cache format is unchanged. Dynamic NVFP4 scaling is enabled only
when all three settings below are selected, and it must not be combined with a
static outer-scale file:

```bash
KV_CACHE_DTYPE=nvfp4_ds_mla \
KV_FP8_ROPE=1 \
VLLM_NVFP4_MLA_DYNAMIC_SCALE=1 \
VLLM_NVFP4_MLA_SCALES_FILE= \
  docker compose up -d
```

Adaptive folding defaults to `auto` with a 256 MiB temporary-workspace budget.
Override it only for diagnosis with
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD=0|1` or change the budget with
`SPARKINFER_INDEXER_TWO_LEVEL_FOLD_MAX_MIB`.

Reproduce the exact r9 source composition from the archived lock files and
hash-verified integration patches with:

```bash
VLLM_RELEASE_COMPOSITION=reproduce-r9 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r9 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm34f26c2-side7739a-fi801d57a-cu132-20260728-r9
```

The r15 release adds the `DeepSeek-V4-Flash-0731` DSpark serving profile and
keeps the standard-checkpoint MTP modes separate. It composes vLLM #212, #213,
#214 and SparkInfer #106 over current clean GG/master sources. The paired cache
changes accept exact and padded compressed-MLA pages without copying, while the
V2 warmup change keeps FlashInfer autotune enabled before KV initialization.

Build or reproduce the exact release with:

```bash
./build-gilded-gnosis-v20-final-cu132.sh

VLLM_RELEASE_COMPOSITION=reproduce-r15 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

Published r15 image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm0bc48c5-sieec30ff-fi801d57a-cu132-20260731-r15
```

Start the pinned 0731 checkpoint in the measured fixed-K7 mode:

```bash
GPUS=0,1 docker compose -f examples/docker-compose-ds4-v20-r15.yml up -d
```

Set `DSPARK_DEPTH_MODE=dynamic` for load-aware draft depth, or `MODE=dspark-mtp0`
for a no-speculation baseline on the same 0731 checkpoint. `MODE=mtp2|mtp3`
selects the historical standard checkpoint because 0731 does not provide the
standard MTP serving contract.

The r16 release adds native CPU KV offload for DS4 without the pinned-host
power-of-two allocation restriction and preserves SWA, MTP, and shared-prefix
replay boundaries. It also changes the DS4 Compose profile to fixed K5, which
was faster in sustained decode and more reliable than K7 in local validation.

Build or reproduce the exact release with:

```bash
./build-gilded-gnosis-v20-final-cu132.sh

VLLM_RELEASE_COMPOSITION=reproduce-r16 \
  ./build-gilded-gnosis-v20-final-cu132.sh
```

r16 release image:

```text
voipmonitor/vllm:gilded-gnosis-v20-vllm1e9c9c3-sieec30ff-fi801d57a-cu132-20260731-r16
```

Start DSpark K5 on two GPUs:

```bash
GPUS=0,1 docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

Native offload is opt-in. `KV_OFFLOADING_SIZE` is the total host capacity in
GiB across all TP ranks; decimal, non-power-of-two values are supported:

```bash
KV_OFFLOADING_SIZE=48.5 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v20-r16.yml up -d
```

The exact r16 image passed TP2 K5 E2E with and without native offload. The
no-offload baseline reached 220.6 tok/s; a repeated 5.5 GiB offload run reached
222.9 tok/s. A 70k/80k/100k prefix sequence transferred 5.22 GB from GPU to
CPU, then restored 635.5 MB and 69,888 prefix tokens from CPU on replay.

The current unified image installs
`/usr/local/bin/serve-fathomless-firmament.sh`, which dispatches to the GLM or
DS4 helper through `MODEL_FAMILY`. Start either model with a minimal
environment-only Compose file and override only the serving choices you need:

```text
voipmonitor/vllm:fathomless-firmament-v17-vllm05f50ae-b12x1377d5f-fi801d57a-cu132-20260715
sha256:9b6f1ab6db4d3a7b7b786481eb32abe82e86d185648d62c3ac1cfa6d72a55e47
```

```bash
MODE=dspark BACKEND=lucifer-cutlass TP_SIZE=2 GPUS=0,1 \
  docker compose -f examples/docker-compose-ds4-v10.yml up -d

MTP=0 DCP=1 MOE_MODE=a16 ONLINE_QUANT=mxfp8 \
  docker compose -f examples/docker-compose-glm52-v17.yml up -d
```

Supported modes are `mtp0`, `mtp2`, `mtp3`, and `dspark`. Supported backend
profiles are `b12x-a16`, `b12x-a8`, `b12x-a8-dglin`, `lucifer-default`, and
`lucifer-cutlass`; the helper derives the CUDA graph cap from
`MAX_NUM_SEQS`. The GLM helper likewise derives `GRAPH=4*MAX_NUM_SEQS` unless
explicitly overridden. Both helpers default to InstantTensor with the
page-cache-aware `BUFFERED` backend. The GLM v17 Compose also defaults
`DCP_PREFILL_WORKSPACE=auto`, which enables the optimized eager prefill path
only for its validated TP/DCP topology list.

### Automatic PCIe calibration

The v20 GLM helper runs a lossless PCIe preflight before the first model load
for each GPU order, TP/DCP geometry, CPU/NUMA placement, image fingerprint,
NCCL configuration, and probe revision. It measures the real collectives and
caches four independent decisions under
`${XDG_CACHE_HOME}/pcie-calibration`:

- `VLLM_B12X_MLA_CKV_PREFETCH_DEPTH`
- `VLLM_DCP_QUERY_SPLIT`
- `VLLM_DCP_QUERY_SPLIT_MIN_CONTEXT_TOKENS`
- `VLLM_PCIE_DMA_MIN_BYTES`, including `off` when NCCL wins the full ladder

Later starts use the cache. Explicit values always take precedence. Set
`PCIE_CALIBRATION=force` to remeasure or `PCIE_CALIBRATION=off` to retain the
conservative static policy. FP8, INT8, and MXFP8 DMA wire modes remain
explicit choices; selecting one through `F8_DMA` never enables a compressed
mode through calibration.

GPU order is resolved as `GPUS`, then an existing `CUDA_VISIBLE_DEVICES`, then
the launcher default. This ensures the probe measures the same ordered devices
that vLLM serves on, including Compose files that leave `GPUS` empty. A cold
probe may compile kernels and has a 600-second startup limit; override it with
`PCIE_CALIBRATION_TIMEOUT` when required.

### Current vLLM+B12X CUDA 13.2 base image

The vLLM+B12X build uses two reusable base images:

- `voipmonitor/vllm:vllm-b12x-cu132-system-base`: CUDA 13.2.1 cuDNN devel base, cuBLAS 13.4.1, cuDNN 9.22, Python 3.12, build/runtime OS packages, and patched NCCL 2.30.4.
- `voipmonitor/vllm:vllm-b12x-cu132-build-base`: the system base plus `/opt/venv` with PyTorch `2.12.0+cu132`, torchvision `0.27.0+cu132`, CUDA tile, and CUTLASS DSL `4.5.2`.

The final image is built `FROM` the system base and copies the completed vLLM
venv from the build stages. This keeps the final image from carrying a stale
base venv while avoiding repeated apt/PyTorch downloads on normal source-only
rebases. The historical 2026-06-08 black-benediction build reused the already
published `glm-kimi-cu132-system-base-20260608` and
`glm-kimi-cu132-build-base-20260608` tags; the preset below preserves that exact
input stack.

```bash
git clone https://github.com/local-inference-lab/blackwell-llm-docker.git
cd blackwell-llm-docker

SYSTEM_BASE_IMAGE=voipmonitor/vllm:vllm-b12x-cu132-system-base \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:vllm-b12x-cu132-build-base \
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 \
./build-vllm-b12x-cu132.sh

# Push the reusable base images when publishing a new stack baseline.
SYSTEM_BASE_IMAGE=voipmonitor/vllm:vllm-b12x-cu132-system-base \
BUILD_BASE_IMAGE_TAG=voipmonitor/vllm:vllm-b12x-cu132-build-base \
IMAGE=voipmonitor/vllm:vllm-b12x-cu132 \
PUSH_BASE_IMAGE=1 \
./build-vllm-b12x-cu132.sh

# Exact black-benediction PR11 image from 2026-06-08.
./build-black-benediction-b12xpr11-cu132.sh

# Lucifer DS4 Flash/CUTLASS image. This reuses the same cu132 system/build bases
# and builds vLLM from local-inference-lab/vllm branch `lucifer`, which contains
# the rebased Lucifer SM120 sparse MLA patch and CUTLASS MoE fix from
# procr1337/llm-bench. It also enables the Triton kernels source hook used by
# that stack.
./build-lucifer-cu132.sh
```

Useful sanity check after the build:

```bash
docker run --rm voipmonitor/vllm:vllm-b12x-cu132-system-base bash -lc '
python --version
nvcc --version | tail -n 1
strings /opt/libnccl-local-inference.so.2.30.4 | grep "NCCL version 2.30.4 compiled with CUDA 13.2"
dpkg-query -W \
  "cuda-compat-13-2" \
  "cublas-cuda-13" \
  "libcublas13-cuda-13" \
  "libcublas13-dev-cuda-13" \
  "libcudnn9-cuda-13" \
  "libcudnn9-dev-cuda-13" \
  "libcudnn9-headers-cuda-13"
'

docker run --rm voipmonitor/vllm:vllm-b12x-cu132-build-base bash -lc '
python - <<PY
import torch
import cutlass
print(torch.__version__, torch.version.cuda)
print(cutlass.__file__)
PY
'
```

## Hardware

- NVIDIA RTX PRO 6000 Blackwell Server Edition (SM120) or compatible
- CUDA driver 575+
- 96 GB VRAM per GPU

## Key features

- **FlashInfer from source** with PR #2913 (GDC for SM120) — no prebuilt cubin/jit-cache that would override patched kernels
- **b12x backend** (lukealonso) — TP-only NVFP4 MoE/GEMM for SM120
- **PCIe allreduce** — custom allreduce for PCIe topologies (cu130 only)
- **nvidia-cublas pinned to 13.1** (cu130) — 13.3 causes illegal memory access on CUDA 13.0 toolkit
- **Model profiles** — preconfigured launch configs via `MODEL_PROFILE` env var
- **Adaptive speculative decoding** (PR #21599) — dynamically adjusts num_steps
- Pre-tuned Triton MoE configs for RTX PRO 6000 Blackwell

## vLLM+B12X CUDA 13.2 Image

`Dockerfile.vllm-b12x-cu132` is intentionally based on reusable base images that
are themselves built from `nvidia/cuda:13.2.1-cudnn-devel-ubuntu24.04`, not from
an older `voipmonitor/vllm` image. The system base keeps the CUDA toolkit on
13.2.1, overlays the latest CUDA 13 library packages currently used by this
image (`cuBLAS` 13.4.1, `cuDNN` 9.22, `cuda-compat-13-2` 595.71), and includes
patched NCCL `2.30.4` from `local-inference-lab/nccl-canonical`. The build base
adds PyTorch `2.12.0+cu132` from the official PyTorch wheel index and CUTLASS
DSL. The final image then builds FlashInfer, DeepGEMM, B12X and the selected
vLLM branch on top of those bases.

The final image defaults to `/usr/local/bin/run-kimi26-vllm`; GLM is available
through `/usr/local/bin/run-glm51-vllm`.
