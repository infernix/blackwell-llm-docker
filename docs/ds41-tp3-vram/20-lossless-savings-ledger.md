# DS4.1 TP3 — 20: lossless VRAM-savings ledger (supported options)

Date 2026-09-24. Compiled by Main from scouts 14/15/16/17/18/12 evidence + live-log verification; the assigned scout could not write files, its verified summary is at agent://LosslessLedger.

## 1. FI-workspace contradiction: RESOLVED AS PHANTOM

Dispatch proof: checkpoint arch DeepseekV41ForCausalLM -> registry (model_executor/models/registry.py:383-386) -> vllm/models/deepseek_v41/__init__.py:20-23 re-exports the NATIVE b12x package (vllm.models.deepseek_v4_1) whenever is_device_capability_family(120) (interface.py:482-496; GB202 = capability 12.0). The native package has ZERO flashinfer imports (grep-verified) and raises 'DeepSeek V4.1 requires B12X' (deepseek_v4_1/nvidia/model.py:77-81).

Live-log proof (Main grep on running ds41_3gpu_karmic_r1): zero occurrences of trtllm_workspace/_fi_sparse; attention_backend='B12X' in non-default args; all-reduce dispatch line 'Using [PYNCCL] ... out of potential backends: [FLASHINFER_PCIE_IPC, B12X_PCIE, ...]' proves the native communicator stack; FlashInfer present only in autotune cache paths + top-p/top-k sampler ('Using FlashInfer for top-p & top-k sampling').

Consequence: both 394 MiB workspaces are never constructed. Runs B/B2 (06) neutrality was structural. The native stack's own workspaces are WorkspaceManager shared arenas sized by scratch_specs (vllm/v1/worker/workspace.py:198-208; deepseek_v4_1/attention.py:74-77, :1482-1485) - MiB-scale, no env lever. 15-study sections E/H (FI ~0.8-0.9 GiB attribution) SUPERSEDED; post-load residual re-attributed to ~2.0 GiB lazy module data/JIT cubins/runtime pools.

## 2. Ledger table

| # | item | mechanism | supported by installed tree | savings GiB/rank | perf risk | effort |
|---|---|---|---|---:|---|---|
| L1 | KV right-size | kv_cache_memory_bytes 1124073472 -> 536870912 | yes - honored at gpu_worker.py:621-643; auto-fit re-derives max_model_len; min bounded by check_enough_kv_cache_memory | +0.547 | none at real workloads (2x~203k = 0.375 GiB); max_model_len 1M -> ~520k | config line + restart |
| L2 | ws3 PCIe one-shot all-reduce | patched b12x/comm/pcie/pcie_oneshot.py:25 SUPPORTED_WORLD_SIZES=(2,3,4,6,8,10) bind-mounted ro | yes - pure Python; wrapper already accepts ws3 | 0 (perf: +2-3% steps/s decode) | unqualified odd-ws correctness - gate on functional suite | 1-line patch + bind mount + restart |
| L3 | UVA offloader hardening | port #54972 empty_cache + #56413 skip-double-pin into config/ds41-tp3-offload/uva.py | yes - our overlay is the delivery path | +0 (removes mid-walk OOM ceiling) | none | ~5 LOC + restart |
| L4 | NCCL buffer shrink | NCCL_BUFFSIZE / NCCL_MAX_NCHANNELS env | yes (standard NCCL envs) | +0.10-0.15 | prefill throughput risk at 2048-token chunks | env flags |
| L5 | FI workspace trim | VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE | PHANTOM - workspaces never allocated (section 1) | 0 | n/a | none - do not spend |
| L6 | DMA capacity trim | VLLM_PCIE_DMA_MIN_BYTES=off | PHANTOM - _initialize_dma never runs on ws3 (b12x_pcie_all_reduce.py:295-302 early return) | 0 | n/a | none |
| L7 | Graph knobs | max-cudagraph-capture-size / breakable / piecewise | pool is 0.02 GiB; piecewise n/a (model cannot torch.compile) | 0 | - | measured (06 run C) |
| L8 | Allocator knobs | PYTORCH_CUDA_ALLOC_CONF extras | reserved-unallocated measured 38.9 MiB; expandable_segments already on | ~0 | none to gain | none |
| L9 | max-num-batched-tokens 2048->1024 | launch config | yes | +0.012-0.014 | prefill chunking throughput - LOSSY for prefill | config |
| L10 | P4 VMM mixed-residency | HOST-LOCATION cuMemCreate pages for cold experts inside single VA range; cp.async path needs no kernel change | yes at allocation layer (~300 LOC shim, zero b12x change); platform probe required first | flexible 6-8 MiB quanta (converts L1 into cold-expert re-residency) | 4-5x PCIe REDUCTION on offloaded layers | ~300 LOC + probe |
| L11 | nvfp4_ds_mla KV | upstream #57463 family | NO - dtype not accepted by deepseek_v4_1 sparse_mla (supported: auto/fp8/fp8_ds_mla); nvfp4 absent; sliding-window record unmerged | future ~0.3-0.35 at equal tokens | - | blocked: image upgrade + upstream |
| L12 | fp8_ds_mla KV dtype | _resolve_dsv4_kv_cache_dtype supports it (deepseek_v4_1) | unproven saving for this arch (already fp8 base) | unknown, likely ~0 | unknown | config, low priority |
| L13 | embed/head FP8 | no upstream implementation; B12xEmbeddingMethod keeps weight.dtype (b12x_layers.py:152-158) | NO | ~0.4 theoretical | accuracy risk; violates checkpoint-integrity standard | hand port - skip |

## 3. Sums

Credible lossless total (config/env/patch, excluding P4): L1 0.547 + L4 0.10-0.15 + L2/L3/L7/L8 0 = 0.65-0.70 GiB/rank = ~10% of the 6.72 needed to fully re-reside ords 20-22. NOT reachable losslessly with config alone.

With P4 (L10): the 0.547 KV trim converts into ~8-9% of ords 20-22's expert bytes moved back to GPU (cold experts, 6-8 MiB quanta), cutting PCIe ~4-5x on the remainder. That is the only structural reduction of 'experts in RAM' available without upstream kernel work.

## 4. Supported-options classification

- config-only: L1 (kv_cache_memory_bytes), L9
- env-only (must be -e flags on podman run line; placement env-list does not reach container - proxy/process.go:444): L4, plus L5/L6 if they were not phantoms
- overlay-patch (bind-mounted ro file, our delivery path next to uva.py): L2 (pcie_oneshot.py), L3 (uva.py), L10 (new shim file)
- image-rebuild/upstream: L11, L13, b12x expert-map kernel work (P2)
