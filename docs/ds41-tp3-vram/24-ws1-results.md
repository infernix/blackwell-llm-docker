# DS4.1 TP3 — WS1 config batch results (24)

Applied (single batched managed restart, 2026-09-24 15:40):
- L1: kv_cache_memory_bytes 1124073472 -> 536870912 (0.5 GiB)
- L2: pcie_oneshot.py SUPPORTED_WORLD_SIZES += 3 AND b12x/comm/pcie/_tuning.py SURFACES OneshotAllReduce* rows += 3 (second gate found live: 'OneshotAllReduce.all_reduce supports world sizes (2, 4, 6, 8, 10)' aborted the first restart at b12x preparation; both files bind-mounted ro)
- L3: UVA hardening in overlay uva.py: 54972 empty_cache after each offloaded module; 56413 pin only on fallback path (verified the native op accepts unpinned: get_cuda_view_from_cpu_tensor(UNPINNED) -> cuda view OK, in-container test)

## Results

| gate | expected | observed |
|---|---|---|
| validate-config | OK | OK (before restart) |
| offload walk | 7.63 (towers+ords 20-22) | 7.63 unchanged |
| KV pool | ~567k tokens | kv_cache_size_tokens=480972, kv_cache_memory_bytes=536870912, max_concurrency 1.0004 @ ~480k |
| ws3 one-shot | init line + B12X_PCIE first in dispatch | Using B12X PCIe all-reduce (algorithm=oneshot, one-shot max=98304...); dispatch [B12X_PCIE, PYNCCL]; NO preparation failure |
| steady used (GPUs 0-2) | ~94.5-95 GiB | 94900/94900/94920 MiB, free ~2392 MiB |
| graphs | 2/2 captured | (in startup log) |
| functional suite | 5/5 | REASONING 3/10 stop; TOOL get_weather Utrecht; BASIC Hello; IMAGE Red; ALIAS ALIAS READY |
| TTFT (streaming, warm) | <20ms | 2.7-16.2ms |

## Notes
- KV 480,972 tokens covers 2 concurrent ~240k contexts (observed max 67; adapter worst ~203k).
- First restart attempt failed: the runtime preparation gate lives in _tuning.py SURFACES, separate from the SUPPORTED_WORLD_SIZES constant. Both staged files patched; error text pinpointed it.
- One-shot max=98304 (96KB) per hardware profile; DMA stays off (ws3 gate retained there intentionally - unreviewed).
