# DS4.1 Flash TP3 migration ledger

This ledger maps the TP3 work originally developed as llama-swap bind-mount
overlays to its durable source, recipe, and evidence repositories. It is the
pruning gate for removing TP3 implementation files from llama-swap: every
runtime behavior must remain reproducible from the destinations below.

## Runtime implementation

The current llama-swap overlays and destination files were compared byte for
byte before this ledger was written.

### vLLM (`6c679489a0444f141adfba785de537741b60f497`)

| llama-swap overlay | durable path | owning commit |
|---|---|---|
| `config/ds41-tp3-offload/uva.py` | `vllm/model_executor/offloader/uva.py` | `2ead0dce92` |
| `config/ds41-tp3-offload/uva_p4.py` | `vllm/model_executor/offloader/uva_p4.py` | `2ead0dce92` |
| `config/ds41-tp3-offload/vmm_policy.py` | `vllm/model_executor/offloader/vmm_policy.py` | `b4b85f6670` |
| `config/ds41-tp3-offload/vmm_shim.py` | `vllm/model_executor/offloader/vmm_shim.py` | `b4b85f6670` |
| `config/ds41-tp3-offload/b12x_layers.py` | `vllm/models/deepseek_v4_1/b12x_layers.py` | `f7cbb83fec` |
| `config/ds41-tp3-offload/rope.py` | `vllm/models/deepseek_v4/common/rope.py` | `f7cbb83fec` |
| `config/ds41-tp3-offload/bank/b12x.py` | `vllm/model_executor/layers/fused_moe/b12x.py` | `b532d2a8f2`, `6c679489a0` |
| `config/ds41-tp3-offload/bank/expert_residency.py` | `vllm/model_executor/layers/fused_moe/expert_residency.py` | `b532d2a8f2` |

These commits provide ordinal-gated startup offload, contiguous VMM expert
regions, post-health promotion, input-embedding UVA, a bounded RoPE cache,
route monitoring, deferred WS4 banks, and the scale-codec tuning pin.

### B12X (`d7944fe29abf8f2cf2ac1d22d99df425150c765d`)

| llama-swap overlay | durable path | owning commit |
|---|---|---|
| `config/ds41-tp3-offload/pcie_oneshot.py` | `b12x/comm/pcie/pcie_oneshot.py` | `0ff26731` |
| `config/ds41-tp3-offload/pcie_tuning.py` | `b12x/comm/pcie/_tuning.py` | `0ff26731` |
| `config/ds41-tp3-offload/bank/dynamic.py` | `b12x/moe/_shared/kernels/dynamic.py` | `ede0920c`, `e94d91b6` |
| `config/ds41-tp3-offload/bank/_impl.py` | `b12x/moe/fused_moe/_impl.py` | `ede0920c`, `e94d91b6` |
| `config/ds41-tp3-offload/bank/api.py` | `b12x/moe/fused_moe/api.py` | `d7944fe2` |

These commits provide TP3 PCIe one-shot all-reduce, bank-addressed WS4
staging, the lossless E8M0 prepared-scale codec, and its public attachment API.

## Launch and image assembly

The host-specific llama-swap placement is replaced by repository-owned recipe
inputs:

- `recipes/glm53/serve-ds41-tp3.sh`: environment, cache defaults, ordinal
  staging, codec, deferred bank promotion, and native vLLM arguments.
- `recipes/glm53/ds41-tp3-launch.yaml`: TP3 scheduler, context, KV, LMCache,
  Engram, and graph settings.
- `recipes/glm53/tests/test_serve_ds41_tp3.py`: recipe behavior coverage.
- `docs/ds41-tp3-operating-point.md`: exact launch contract, expected
  `hot=384 cold=0` proof, source pins, and operational recovery notes.
- `tools/jovian_wheel_runtime/`: source-locked wheel bundle assembly and the
  model-neutral 68-layer runtime image.

The verified image uses these exact inputs:

| Component | Source commit |
|---|---|
| blackwell-llm-docker | branch `ds41f-tp3` |
| vLLM | `6c679489a0444f141adfba785de537741b60f497` |
| B12X | `d7944fe29abf8f2cf2ac1d22d99df425150c765d` |
| FlashInfer | `2206a14e46387a56c093860a46bbbdd00596b75b` |
| InstantTensor | `95d4729b6d6a991bb8de61877147a9d9d9100b23` |
| LMCache | `0b8082a5b559d3fe882898c10fed055c91d773f2` |
| NCCL canonical | `93fe05d9f9b6963ef841166a69cd0b30e4efe97b` |

The foundation is
`nvcr.io/nvidia/pytorch@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`.

## Evidence and design history

- `docs/ds41-tp3-vram/00-investigation-plan.md`: original acceptance plan.
- `docs/ds41-tp3-vram/01-*` through `29-*`: the complete 31-file measured
  VRAM, VMM, all-reduce, and full-residency investigation trail.
- `docs/ds41-tp3-scale-codec/`: all 54 committed codec prototypes, gates,
  reports, and handoff files. Generated compile caches, object files, lock
  files, and Python bytecode were deliberately excluded.

The final component implementations supersede the prototype modules. The
prototype tree remains evidence and a reproducible numerical/measurement
reference, not production importable code.

## Intentionally not transferred as product source

- `config.yaml`, matrix policy, placement names, checkpoint/cache host paths,
  ports, and GPU assignment are llama-swap host orchestration state.
- `config/models/host-cuda-03-*` changes only isolated companion workloads on
  GPU 3; the durable recipe states that TP3 consumes GPUs 0-2.
- `config/ds41-karmic-launch.yaml` is the separate TP4 operating point.
- `WATCHDOG.yml` selects local advisor models and is unrelated to TP3 runtime
  behavior.
- `__pycache__/*.pyc`, compile caches, native objects, transient logs, and
  `/dev/shm` state are generated artifacts and must not be migrated.
- Historical overlay revisions that were deleted or superseded remain in
  llama-swap Git history and in the investigation record; only the final
  source state belongs in vLLM/B12X.

## Pruning gate

Do not prune llama-swap until all three conditions hold:

1. `ds41f-tp3` is published in every referenced `github.com/infernix` fork.
2. The exact vLLM and B12X wheel bundles and the assembled container are
   published with immutable source refs and SHA-256 metadata.
3. A blackwell-llm-docker GitHub release records the image digest, component
   revisions, qualification scope, changelog, and release assets.
