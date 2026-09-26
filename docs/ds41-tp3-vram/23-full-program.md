# DS4.1 TP3 — FULL lossless program, nothing excluded (23)

Supersedes 22. User directive: include EVERYTHING, even hard work. Kernel work (P2) is now IN.

## The complete program — five workstreams

### WS1 - Config/overlay restart batch (L1+L2+L3) — Phase 1
- L1 KV right-size: kv_cache_memory_bytes 1124073472 -> 536870912 (+0.547 GiB/rank GPU)
- L2 ws3 one-shot: patched pcie_oneshot.py:25 SUPPORTED_WORLD_SIZES=(2,3,4,6,8,10), bind-mounted ro
- L3 UVA hardening: #54972 empty_cache-after-module + #56413 skip-double-pin in config/ds41-tp3-offload/uva.py (~5 LOC)
- Gate: validate-config, functional suite, graphs 2/2, KV ~567k tokens, one-shot init line, no expandable_segments OOM warnings

### WS2 - VMM platform probe — Phase 2
- 20-line cuda-python probe: cuMemCreate HOST_NUMA location + cuMemMap into device VA + cuMemSetAccess + device-side read on GB202, granularity check (2 MiB)
- Go/no-go for P4. If NO-GO: P2 becomes the primary path instead of fallback.

### WS3 - P4 VMM mixed-residency shim (~300 LOC) — Phase 3
- Allocation shim: VMM-backed torch tensor wrap (cuMemCreate/cuMemMap/cuMemSetAccess per 2 MiB page range)
- Placement policy: cold-expert selection (page-aligned expert ranges; initial policy: routing histogram from RoutedExpertsCapturer or 20-line topk_ids capture_fn)
- uva.py walk variant: allocate via shim for ords in [FROM,TO], hot pages device-backed, cold pages HOST-LOCATION
- Order of operations: repack first (all device-backed), then remap cold ranges to host pages before serving
- Gate: 1-GPU numerics microbench vs resident baseline; functional suite; graphs 2/2; TTFT A/B (alternating 1-token streams, matched queue depth); steady MiB delta

### WS4 - P2 KERNEL WORK (~400-500 LOC, b12x upstream-aligned) — Phase 4, IN SCOPE
Deliverable: per-expert pointer-table staging mode in the w4a8 kernels.
- 4 new kernel params: device Int64 banks w13_bank/down_bank/sfbw13_bank/sfbdown_bank[E] (byte base per expert; null => legacy arithmetic mode via compile-time const_expr flag - keeps today specialization intact)
- Staging sites: replace Int64(task_expert_idx)*expert_stride with ld_global_u64(bank+idx*8)+intra_expert_offset at ~8-12 sites (dynamic.py:8602-8625, 8749-8786, 8848-8870, 9129-9141, 9246-9250, 2006-2009 + SFB/residual variants ~308-345, 8917-8933); one extra 8B load per 3.75 MiB tile (noise)
- Launch ABI: _impl.py:11619-11723 operand list + _encode_dynamic_launch_policy :11572-11592
- Canonical-storage contract extension: _impl.py:492-514 must accept per-expert bank backing (~40 LOC, upstream-facing change)
- vllm wiring: b12x.py prepare/apply + policy passthrough (~150-200 LOC)
- TMA recipes (nvfp4/w4a8_b_tma): NOT touched - if ever needed, use the two-bank dual-descriptor form (18 sec 4), separate decision
- Purpose even with P4 working: (a) removes the 2 MiB page co-location slack (~6-8 MiB quanta -> true 5.98 MiB expert quanta); (b) enables EXPERT-MAP semantics in b12x (supports_expert_map() False today, b12x.py:790-791) which unlocks vLLM #57943 StagingBank convergence; (c) upstream contribution aligned with #57794 RFC
- Gate: b12x test corpus (if present in image), numerics vs legacy mode on synthetic hot/cold split, then full placement gate as WS3

### WS5 - Tracking (upstream-gated, keep on radar)
- nvfp4_ds_mla KV (#57463 family; needs SM120 sliding-window record + image upgrade) - future +0.3-0.35 GiB at equal tokens
- HostBank/StagingBank (#57943/#57794) - converges with WS4 outcome
- Fork integration/karmic-kraken-beta updates (29 commits ahead of pin; #887 DCP fix among them)

## Held items (explicit, with revival conditions)
- L4 NCCL buff shrink (+0.10-0.15): revive only if VRAM-critical; prefill A/B required
- LMCache L1 cap 48->8 (host RAM ~40 GiB): revive if MemAvailable < ~1 GiB sustained
- Lossy items (KV 0.25, batched-tokens, embed/head FP8, ord 23 offload): permanently out per user constraint

## Final achievable state (WS1+WS3, and WS4 as enhancer)
- ords 20-22: only COLD experts host-backed (~2-5% of routes); ~0.5 GiB of expert bytes return to GPU via L1 funding
- PCIe: ~215 -> ~50 MiB/step on offloaded layers; all-reduce 44.9 -> 6.0us
- GPU free: ~+0.5 GiB/rank net (KV trim spent on cold experts); more if routing skew allows deeper cold selection
- With WS4: quanta tighten to 5.98 MiB and expert-map APIs unlock; host cold tier can shrink further via StagingBank-style hot pooling later

## Execution order
WS1 (1 restart) -> WS2 probe -> WS3 (1 restart) -> WS4 (kernel branch + upstream PR) -> WS5 watch
Every phase independently revertible; functional suite is the gate at every restart.
