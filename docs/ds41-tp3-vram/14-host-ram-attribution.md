# 14 — Host RAM attribution & real sizing levers (DS4.1 TP3 placement)

Date: 2026-09-24 16:23–16:58 CEST. Read-only; builds on `14-host-ram-composition.txt` (meminfo, /dev/shm listing, worker smaps_rollup), does not redo it.
All numbers are from captured command output on the live host (meminfo/df/smaps/pagemap/cgroup/HTTP probes); process ids are stable for this boot of the placement.

**Scope note (per Main, 16:58):** priority shifted — the reconciliation table (what holds the 184 GiB; elastic vs pinned) and the sglang-79-GiB breakdown are the deliverables; the LMCache-cap / shrink-lever material is kept brief. One new boot-state finding that matters for the re-residency goal is flagged in §4 and the verdict.

## 1. Top-level reconciliation (meminfo-anchored)

Coherent snapshot (hostram10 analyzer, 16:44): `/proc/meminfo`, `df -k /dev/shm`, global smaps_rollup sums taken in one pass.

| Component | Source | GiB |
|---|---|---:|
| MemTotal | /proc/meminfo | 184.80 |
| MemFree | /proc/meminfo | 1.84 |
| **MemAvailable** | /proc/meminfo | **2.58** |
| Shmem (resident shared) | /proc/meminfo | 156.06 |
| AnonPages | /proc/meminfo | 16.95 |
| File page cache (Cached − Shmem) | /proc/meminfo | 1.57 |
| Slab (SReclaimable 1.14 + SUnreclaim 1.08) | /proc/meminfo | 2.21 |
| PageTables (+SecPageTables) | /proc/meminfo | 0.64 |
| KernelStack + Percpu | /proc/meminfo | 0.06 |
| VmallocUsed | /proc/meminfo | 3.24 |
| **Sum** | | **182.58** |
| Residual (struct pages, firmware, GPU-reserved) | derived | ~2.22 |

Sum + residual = MemTotal. ✔ Reconciles.

Corrections to earlier assumptions:
- **File page cache is NOT the hidden consumer.** Cached 157.68 GiB is almost entirely Shmem; true file cache is only 1.57 GiB (the model safetensors are streamed through and evicted — instanttensor lazy load; ds41 container `file` stat is cgroup-v2 `file` which *includes* shmem, do not read it as page cache).
- **Shmem is fully resident**: `nr_shmem` 40,919,130 pages = 156.06 GiB (vmstat) — none of it is "mapped to files on disk".
- Swap: 26.64 GiB used (SwapTotal 192.00 − SwapFree 165.36); SwapCached 6.80 GiB (pages swapped out and brought back).
- `MemAvailable 2.58` is consistent with the components: free 1.84 + ~0.7 GiB reclaimable file/slab. Dirty shmem does not count toward MemAvailable without swap-out, which is why 156 GiB of shmem still leaves only ~2.5 GiB available.

## 2. Shmem attribution — who owns the 156.06 GiB

cgroup v2 `memory.stat` `shmem` (charged pages, resident + swapped-out), containers under `user.slice/user-1000.slice/user@1000.service/.../libpod-<id>.scope/container`:

| Container | shmem charged | matches |
|---|---:|---|
| ds41_3gpu_karmic_r1 (28a0766…) | 80.55 GiB | pool 48.00 + psm 0.62 + **~31.9 deleted-arena residual** (see below) |
| qwen38_flash_next_gpu3_sglang (ba7555c…) | 75.55 GiB | PLE 64.00 + HiCache ~11.5 + misc ✔ exact |
| qwen3-embedding-4b (319fb89…) | ~0.00 GiB | ✔ (its 6.6 GiB heap is anon, mostly swapped: swapcached 6.60) |
| **Sum** | **156.10 GiB** | ≈ global Shmem 156.06 ✔ (anchor) |

Named /dev/shm files (df `Used` 48.64 GiB = physically allocated tmpfs blocks):
- `lmcache_l1_pool_lmcache-ds41-flash-5807-15807`: 51,539,607,552 B = **48.00 GiB fallocated** (st_blocks 100,663,296×512 exact; inode 2196, dev 1b).
- 91 `psm_*` files: 0.62 GiB real (blocks×512) — **86 stale** (no live opener via fuser) = 0.62 GiB, 5 live files ≈ 2 MiB real.
- Remainder: 23 `cuda.shm.*` 80 KiB files, `sem.mp-*`, `sglang_loads_*.shm`, jack/libpod locks — ~2.5 MiB total.

**Residual flag [INFERENCE]:** ds41 cgroup shmem (80.55) exceeds named files (48.62) by **~31.9 GiB**. Worker smaps show only ~0.54 GiB each mapped from `/dev/zero (deleted)` rw-s arenas (MAP_SHARED /dev/zero creates unlinked shmem objects; workers carry 157 deleted mappings each, 1.42 TiB virtual). Best decomposition: ~1.6 GiB mapped + up to 8.17 GiB swapped out (container `swap.current`) + remainder resident-unmapped page cache of those deleted arenas (staging buffers touched once at startup). Cannot close exactly without new probes (out of scope); it does not change any verdict below.

Cross-check, resident-unmapped shmem: unique mapped shmem ≈ 75.52 (sglang sched) + 23.10 (pool, shared by 3 workers, pagemap-verified) + ~1.6 (worker arenas) ≈ 100.2 GiB vs 156.06 resident → ~56 GiB sits unmapped in shmem page cache (pool remnants + deleted arenas) [INFERENCE, ±9 GiB given the cgroup sum quirk above].

## 3. sglang 79 GiB RSS breakdown (pid 2404178 `sglang::scheduler`)

smaps_rollup: Rss 82,816,480 kB = 78.98 GiB; Pss 78.97; Private_Dirty 82.78; Swap 0. Composition from per-mapping smaps (hostram5-C, hostram9-I, hostram10 buckets):

| Component | GiB | Evidence |
|---|---:|---|
| **PLE embedding table** (`/dev/zero (deleted)` rw-s, 67,108,864 kB VMA) | **64.00 exact** | smaps; config comment "The model's PLE table is pinned in RAM" (config/models/host-cuda-03-multimodal.yaml:19-21) |
| **HiCache host pool** (remaining rw-s arenas: 8,045,440 + 3,538,944 kB + small) | **~11.5** | `--hicache-size 12`; `hicache_host_used_tokens` 617,216 / 618,880 = **99.7% full** (:5818/metrics) |
| Heap | 2.34 | smaps [heap] Private_Dirty |
| Anonymous | ~1.0 | smaps ANON bucket |
| Code/libs | ~0.1 | smaps |

- Shrinkable: the **HiCache ~11.5 GiB** (a restart of the sglang placement with a smaller `--hicache-size` would free it; it is 99.7% full, so shrinking costs cache capacity it is actively using). The **PLE 64.00 GiB is pinned-by-design** — the Qwen3.8-Flash-Next PLE table must stay host-resident for this placement; not a lever.
- Note the 79 GiB is one process's *mapping* of shared-class shmem; at kernel-charge level it belongs to the sglang container's 75.55 GiB (§2). It coexists with the TP3 placement on this host by design (policy `ds41_3gpu_qwen38`, config/matrix.yaml:79).

## 4. ds41 placement processes (per-process attribution)

smaps_rollup at 16:44 (hostram10), GiB:

| Process | Pid | Rss | Pss | Swap | Notes |
|---|---|---:|---:|---:|---|
| sglang::scheduler | 2404178 | 78.98 | 78.97 | 0 | §3 |
| VLLM::Worker_TP0/1/2 | 2620683-85 | 24.2 ea | 9.3 ea | ~2.05 ea | pool VMA 23.10 shared-dirty each (unique 23.10, pagemap-verified); private ~1.55 |
| VLLM::EngineCore | 2620622 | 0.45 | 0.43 | 0.65 | |
| lmcache server | 2620255 | 0.05 | 0.04 | 0.45 | |
| api server (python) | 2620117 | 0.00 | 0.00 | 0.05 | mostly swapped |
| VLLM::Worker (embed) | 1286668 | 6.59 | 6.74 | 4.84 | private heap 6.62 |
| llama-swap | 1280076 | 0.26 | 0.27 | 0.06 | |
| omp + runner | 1963539/1977782 | 0.65/2.9 | — | 1.35/2.9 | investigation tooling |

Global (51 procs, 16:44): sumPss 118.25 + sumSwapPss 19.13 = **137.4 GiB unique** mapped memory; the rest of resident RAM is unmapped shmem page cache (§2) + file cache + kernel.

**Boot-state finding (matters for the re-residency goal):** the placement was restarted at 16:00:19 (`podman inspect` StartedAt). This boot's engine log shows the UVA offload walk offloaded **only the vision/aligner, 0.77 GiB** — not the expert ords 20-22:

```
(Worker_TP0 pid=508) INFO 09-24 14:00:51 [uva.py:82] ds41-tp3 overlay: offload walk from ordinal 20, (ordinal, bytes) = [(0, '0.77')]
(Worker_TP0 pid=508) INFO 09-24 14:00:51 [uva.py:89] Total CPU offloaded parameters: 0.77
```

(the single hit is the vision **tower** walk — stock behaviour for `prefix != ""` per the deployed `config/ds41-tp3-offload/uva.py:66-95`; a main-walk hit at ordinals 20-22 would log tuples `(20|21|22, …)` and a ~7.5 GiB total). The `DS41_OFFLOAD_FROM_ORDINAL=20/TO=22` env vars are set in the container cmd, so either module enumeration changed this boot or the expert offload silently did not trigger. **Expert ords 20-22 appear to be GPU-resident already in the running instance** — GranularOffload should verify against the engine before planning any re-residency change; host RAM for ~20.2 GiB of new pinned memory is only needed if that finding is wrong.

Where UVA-offloaded weights live (for the general case): `uva.py` pins via `cpu_data.pin_memory()` (cudaHostAlloc, page-locked) and mounts a UVA view (`get_accelerator_view_from_cpu_tensor`, vllm/utils/torch_utils.py:915) — pinned host memory is unswappable and must be supplied by MemAvailable at allocation time.

## 5. LMCache actual-vs-cap

- **Cap/allocated:** 48.00 GiB — `l1-size-gb 48.0` (ps cmd line), `memory_total_bytes` 51,539,607,552 (`/status`, `/config`), fallocated in /dev/shm (st_blocks exact), charged to the ds41 cgroup.
- **Mapped-resident:** **23.10 GiB** — pagemap present-bytes over the pool VMA in each worker (identical 23.100 GiB ×3, shared). Worker RSS for the pool fell 44.4 → 23.1 GiB between 16:31 and 16:44 while df/Shmem stayed constant: pages were unmapped under pressure, pool page cache remains resident-unmapped (§2).
- **Used by objects:** **0 bytes / 0 objects at 16:42; 5,009,408 B / 11 objects at 16:52** (first production stores during the observation window). `lmcache_mp_l1_memory_usage_bytes 0.0`, `l1_usage_ratio 0.0`, allocator "Total free size: 49152.000000 MB" (engine log 14:31). Content sample: only the first 64 MiB of the 48 GiB file holds data; 6/12/24/40 GiB offsets are all zeros.
- **Hit-rate:** `lookup_requested_tokens_total 0`, `lookup_hit_tokens_total 0`; engine `External prefix cache hit rate: 0.0%` across 17 requests this boot; no L2 adapters.
- Methodology note: mincore on this kernel reports *per-VMA faulted* residency, not page-cache presence (selftest: 32 MiB written via fd showed 2/8192 pages resident in a fresh mapping) — worker pagemap is the ground truth used above. `SHM pinned=True` is logged per worker (shm.py:145) but the 44.4→23.1 GiB RSS shrink proves the pool is **not** mlock-pinned (host-wide Mlocked = 48 kB).

## 6. Elastic vs pinned

| Class | Pools | GiB |
|---|---|---:|
| **Pinned (cannot be reclaimed)** | sglang PLE table (host-pinned by design) 64.00; CUDA-pinned host buffers (UVA offloaded weights, currently only 0.77 vision/aligner); mlock total 0.00005 | ~65 |
| **Dirty-shmem (reclaimable only via swap-out)** | sglang HiCache ~11.5; pool written pages (23.1 mapped + page cache); deleted /dev/zero arenas | ~35-90 |
| **Elastic (cheaply reclaimable)** | file page cache 1.57; SReclaimable 1.14; LMCache pool *capacity* — 48.00 charged but ~0 used (fallocated, mostly untouched) | ~51 potential |
| Anon (working sets, swappable) | 16.95 resident; 19.13 SwapPss already out | — |

## 7. Verdict

**VERDICT: Host RAM is fully subscribed (184.8 GiB, MemAvailable 2.58) but it is NOT the binding constraint for putting expert ords 20-22 back in VRAM — they are already GPU-resident in the running instance (engine log: total CPU-offloaded = 0.77 GiB = vision/aligner only); the only consumer that matters is sglang's pinned PLE table (64.0 GiB, untouchable) plus LMCache's 48 GiB L1 pool, which is dead weight today (0 bytes used until 5 MB at 16:52, hit-rate 0/0) — lowering the LMCache L1 cap (48→8 GiB) is the single effective lever (frees ~40 GiB charged / 23.1 GiB mapped) and would be prerequisite headroom if ~20.2 GiB of new pinned host memory is ever needed for a re-enabled expert offload; stale psm_* cleanup (0.62 GiB, 86 files) is free but trivial; sglang HiCache (~11.5 GiB) requires a sglang restart and is not worth it.**

Lever ranking (reclaimable × cost):
1. **LMCache L1 cap reduction** — frees up to ~40 GiB charged (48.0 cap − 0.005 used); cost: one placement restart via llama-swap; risk: nil today (hit-rate 0, no L2). Only lever big enough to matter for pinned-host expansion.
2. **psm_* stale cleanup** — 0.62 GiB real; `rm /dev/shm/psm_*` of the 86 files with no live opener; risk: none (fuser-verified live set).
3. **sglang HiCache reduction** — ~11.5 GiB; cost: sglang placement restart, loses an actively-used (99.7% full) host cache.
4. **mamba lazy slots** — no host-RAM effect measured (mamba pools are GPU-side; host pool classes in sglang source are not among the big arenas).
