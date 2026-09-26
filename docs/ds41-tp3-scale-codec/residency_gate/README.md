# Post-start expert residency gates

Two promotion mechanisms were implemented for DeepSeek V4.1 Flash's N128-repacked B12X path.

## Shared lifecycle

Startup retains ordinals 20–22 in immutable HOST_NUMA VMM backing so InstantTensor loading and B12X preparation fit. Vision and aligner remain host-backed. Prepared scale planes are compressed once at startup; FP4 weight bytes remain unchanged.

Promotion is explicitly armed only after API health by creating `/cache/deferred-banks.arm`. Until then, B12X preparation and CUDA graph capture use all-cold pointer banks. The first real request after arming copies selected expert weight slices into VRAM, synchronizes, and updates stable device pointer tables. Host backing remains alive, so eviction or hot-set replacement does not reread the checkpoint.

The placement retains 1 GiB/rank as an operational reserve. With the measured 512K-context steady state, promotion made layers 20 and 21 fully GPU-resident and cached 153/384 experts in layer 22. Remaining host tier: 231 expert instances in layer 22.

## WS4 hybrid path

`DeferredWeightBanks` maintains stable `w13` and `down` pointer tables. Compressed scales stay contiguous and use scratchless shared-memory decoding; there are no scale slabs. `replace_hot()` supports arbitrary logical expert IDs and keeps RAM authoritative. Focused native tests cover all-cold, hot-prefix, non-prefix replacement, eviction, graph execution, and output parity.

## Same-VA VMM path

`MixedResidencyAllocation.promote_up_to()` can replace a complete cold VMM mapping with DEVICE backing without changing its canonical address. The retained HOST_NUMA handle supports copy-free eviction. CUDA requires complete original mappings to be unmapped and allocation offset zero; the current two-handle allocation therefore has a whole-region, not page, promotion quantum. This path passes same-address graph replay and eviction tests but is less granular than WS4.

## Evidence

- `source_gate.py`: codec+bank specialization and raw-path exclusion checks.
- `driver_gate.py`: transactional VMM bookkeeping and rollback.
- `bank_gate.py`: byte-exact expert slabs, stable tables, arbitrary hot sets, eviction, rollback.
- `vmm_gpu_gate.py`: same VA, host-to-device promotion, graph replay, eviction.
- Live TP3 run: nine all-cold bank sets attached; after post-health arm, layers 20/21 promoted 384 experts each and layer 22 promoted 153 experts per rank. Exact promoted bytes/rank: 2,264,924,160 + 2,264,924,160 + 902,430,720 = 5,432,279,040. Physical free VRAM after promotion was 1,026 MiB/GPU and health remained 200. Warmup returned `CACHE READY`.

The fallback hot order is logical IDs ascending. It is correctness-safe but not routing-aware; measured benchmark traffic remained high. Dynamic replacement is supported by the ownership model, but route-histogram collection and policy are separate tuning work.

## Opt-in routed expert accounting

Set `B12X_EXPERT_ROUTES=1` **before worker initialization** to allocate per-bank
384-entry int64 counters and compile int32/int64 ID kernels. Default is off.
Captured graphs include the counter kernel with its device enable scalar zero;
the existing arm-triggered eager promotion enables it after table publication.
Preparation and disabled graph replays do not count. Arming still requires an
eager apply to reach the existing promotion hook; touching the arm file alone
cannot enable a process doing only graph replay. Already captured live graphs
cannot acquire this instrumentation without a later worker load; no live
placement change or restart is part of this implementation.

Each enabled bank starts a daemon writing
`/cache/expert-routes-rank{rank}-layer{ordinal}.json` every
`B12X_EXPERT_ROUTES_INTERVAL` seconds (default 30, strictly positive).
No signals are intercepted. Snapshots use a dedicated CUDA stream, preallocated
sample/pinned-host buffers, int64 atomic sampling, and atomic file replacement.
There is no serving-step synchronization or tensor allocation. Sampling is
per-counter atomic, not a simultaneous global cutoff of concurrent routes.
Counts are cumulative; `cold_routes`, `hot_hit_ratio`, and `cold_hit_ratio` are
recomputed against the **current** `hot_ids`, not historical residency at the
time each route occurred. The remaining schema fields are `rank`, `layer`,
`total_routes`, `counts` (384 integers), and `timestamp` (Unix seconds).
Padding IDs outside [0,384) are ignored. Owner ordinals identify layers 20–22.

`routes_gpu_gate.py` exercises disabled eager/capture, both ID widths, duplicate
IDs, five enabled graph replays, replacement accounting without counter reset,
daemon file refresh, exact schema, and stable replay GPU allocation. On RTX PRO
6000 Blackwell (CC 12.0), the isolated gate passed and compute-sanitizer memcheck
reported zero errors. A two-counter-kernel replay measured 4.10 us for seven
IDs; this is a synthetic measurement, not serving throughput or large-prefill
overhead. Counters + sample consume 6148 logical GPU bytes per bank, plus
allocator rounding, stream/event/runtime overhead and 3072 pinned-host bytes.
Before production use, compare identical request workloads with the gate off/on
for TTFT, decode tokens/s, and large-prefill latency; contention grows with
duplicate routes. Rollback is disabling `B12X_EXPERT_ROUTES` on the next load.
No model weights, routing outputs, or pointer-bank ABI are changed.

## Context-sized DeepSeek RoPE caches

DeepSeek owns two process-global FP32 RoPE tables, each `[rows, 64]`. The YaRN compressor table uses original length 65,536 × factor 16 = 1,048,576 rows; it is not a second 16× multiplier over 1M. Exact source probe results:

| Serving rows | Two FP32 caches | Saving vs 1M |
|---:|---:|---:|
| 1,048,576 | 512 MiB | 0 |
| 786,432 | 384 MiB | 128 MiB |
| 524,288 | 256 MiB | 256 MiB |

`DS41_ROPE_CACHE_ROWS` caps only materialized lookup positions. Checkpoint max-position, YaRN frequency correction, factor, and mscale remain unchanged. Every retained FP32 row was byte-identical to the baseline. Live cold boots measured 4,848 MiB free/rank at 1M, 5,630 MiB at 768K (includes the larger KV tier delta), and 6,430 MiB at 512K versus 6,210 MiB before allocation-time capping: 220 MiB measured physical reclaim/rank. The 512K placement reached health and returned `ROPE READY`.

Precision experiments on sampled rows: BF16 max absolute error about 0.00195 and mean about 0.0005; FP8 E4M3 max about 0.0312 and mean about 0.0074. DeepSeek deliberately stores FP32 for precision, so neither lossy dtype is deployed.

## Full expert residency operating point

At the 524,288-token tier, lossless input-embedding UVA reclaimed 418 MiB/rank. B12X embedding gathers only selected 10 KiB BF16 rows; the checkpoint and table bytes remain unchanged in retained pinned RAM. The context-sized FP32 RoPE allocation cap separately reclaimed 220 MiB/rank. Together they raised pre-promotion free memory to 6,854 MiB/rank.

With `B12X_DEFERRED_BANK_RESERVE_BYTES=268435456`, all 384 experts in each of layers 20, 21 and 22 promoted after health: 6,794,772,480 bytes/rank total. RAM backing remains retained for eviction. Live physical free memory after promotion was 370-374 MiB/GPU. Health remained 200 and chat returned `FULLY RESIDENT`. Prefill smoke requests completed for 1,055 prompt tokens in 0.389 s and 8,223 prompt tokens in 1.599 s; these are single-run end-to-end checks, not an A/B performance claim.
