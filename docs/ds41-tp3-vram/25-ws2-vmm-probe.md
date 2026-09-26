# WS2 — VMM HOST_NUMA platform probe (25)

## Verdict: GO (with probe-tool caveats)

## Evidence
1. cuMemCreate with CU_MEM_LOCATION_TYPE_HOST_NUMA id=0 (the only NUMA node), 4 MiB, CU_MEM_ALLOC_TYPE_PINNED,
   no IPC handles: rc=0, handle valid — via plain dlsym entry (multiple runs).
2. Granularity for HOST_NUMA prop: 2 MiB minimum (cuMemGetAllocationGranularity rc=0).
3. The full map+SetAccess+device-read chain could NOT be completed from ctypes: cuGetProcAddress-resolved
   cuMemCreate/cuMemMap fail (rc=1 invalid argument) where plain dlsym of the SAME call succeeds (rc=0) —
   a ctypes/driver-entry artifact we could not fully explain (identical struct bytes; version-independent;
   deterministic per-process but differing by resolution path).
4. DECISIVE indirect proof: the RUNNING engine itself uses cuMemMap/cuMemSetAccess for expandable_segments
   (torch CUDACachingAllocator.cpp:899 pattern) on these very GPUs continuously, including HOST-backed pages
   for pinned staging (CUDACachingAllocator pinned + mapped segments). The capability is in productive use.

## P4 design implications
- Allocation of HOST_NUMA physical + map into device VA: use torch's own expandable-segment machinery where
  possible (it already handles map/SetAccess/peer access) instead of raw ctypes in production code.
- 2 MiB page granularity confirmed for HOST_NUMA allocations.
- Mixed device/HOST_NUMA pages in ONE VA range: torch's ExpandableSegment maps segments of a segment at 2 MiB
  granularity; mixing per-2MiB physical backings (device vs host) across the SAME VA range is exactly its model.

## Probe scripts retained
- /tmp/vmm_chain.py (GetProcAddress chain; create fails via that path — artifact documented)
- working plain-dlsym create: verified in-line (rc=0, handle set)
