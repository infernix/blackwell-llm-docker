# DS4.1 TP3 — complete findings report (21)

Date 2026-09-24. Consolidates investigations 12-20 (eight subagents) plus 01-11 (earlier session work).
Constraint established by user: text-path performance must not degrade; vision may live in host RAM; minimize or eliminate experts in host RAM and/or minimize PCIe traffic.

## A. Established facts (all measured or code-verified)

### A1. Memory map (per rank, GPUs 0-2, 95.01 GiB cards)

| component | GiB/rank | state |
|---|---:|---|
| model load (with ords 20-22 + towers offloaded) | 87.58 | measured 11 |
| of which: resident decoder-half experts (ords 23-39) | 38.15 | 17 layers x 2.2413 |
| of which: encoder half (ords 0-19) | 0 experts | encoder half is dense+shared experts |
| expert ords 20-22 OFFLOADED (UVA host-backed) | 6.72 | 3 x 2.2413, 04/09 |
| vision tower + aligner OFFLOADED | 0.904 | 09/10 |
| KV pool (fixed, kv_cache_memory_bytes) | 1.05 | 1,131,122 tokens fp8 |
| CUDA graphs (FULL_DECODE_ONLY) | 0.02 | 06 |
| context+NCCL+runtime floor (non-reclaimable) | ~3.4-4.3 | 15 (post-20 correction below) |
| steady free | ~1.5 | live nvidia-smi after suite traffic |

### A2. Corrections made this round (subagent-verified)

1. FI workspaces are PHANTOM: the native b12x package (deepseek_v4_1, the only one loaded on capability 120) has ZERO flashinfer imports; live log has zero trtllm_workspace/_fi_sparse allocations; FlashInfer appears only in the sampler and autotune cache. Runs B/B2 neutrality was structural. (20; 15 sections E/H SUPERSEDED)
2. The 03-runtime-screening inventory was built against the WRONG package (deepseek_v41 generic). The native deepseek_v4_1 stack uses WorkspaceManager shared arenas (MiB-scale, no env lever). Fixed-buffer inventory shrinks to ~0.05 GiB.
3. Context-gap re-attribution: post-load residual ~2.0 GiB is lazy module data/JIT cubins/runtime pools, not workspaces. Reducible: ~0 (expandable_segments already on; reserved-unallocated only 38.9 MiB).
4. Prior-scout 13 BLOCKED verdict was too strong: per-expert POINTER indirection is absent (needs ~400-500 LOC kernel change), but expert-granular MIXED RESIDENCY is achievable at the ALLOCATION layer with ZERO kernel changes (18): the executed w4a8 path stages B weights via cp.async with base+idx*stride arithmetic - single-pointer, so one VMM VA range can mix device-backed pages (hot experts) and HOST-LOCATION cuMemCreate pages (cold experts). ~300 LOC host-side shim, no b12x change. At equal VRAM freed: ~4-5x less PCIe on the offloaded layers (cold experts carry ~2-5% of routes).
5. DCP3: hard no. Cannot start (three independent blockers: hybrid-KV asserts, no DCP wiring in deepseek_v4_1 native package, pin predates fork PR #887 Triton power-of-two fix). Even patched, frees only ~0.65-0.7 GiB/rank KV at 1M capability; experts untouched by DCP. (19)
6. Host RAM: 184.8 GiB fully subscribed (MemAvailable 2.58) but composition is: sglang PLE table 64.0 (pinned by design, untouchable) + LMCache L1 pool 48.0 (allocated, only 23.1 GiB mapped-resident, used-by-objects ~0, hit-rate 0) + HiCache ~11.5 (99.7% full, restart needed) + workers ~18.5 Pss each. LMCache L1 48->8 GiB is a real ~40 GiB host-RAM lever if needed. (14)
7. Upstream: pin is FRESH (2b86d3d80 = fork PR #866 merge 2026-09-23; upstream sync ~Sep 19/20; 465 upstream commits ahead). No upstream PR hands this placement a large VRAM win. Real finds: fork 29 commits ahead on integration/karmic-kraken-beta (KV-connector fixes, GLM-5.3 TP3, DCP LSE fix); #54972/#56413 small UVA-offloader hardening (open, portable to our overlay ~5 LOC total); #57463 nvfp4 KV (needs sliding-window follow-up + image upgrade); #57943/#57794 HostBank/StagingBank RFC (needs b12x expert-map work = P2). (12)

### A3. The binding arithmetic

- Goal: ords 20-22 fully resident = free 6.72 GiB/rank. All lossless actions sum 0.65-0.70 GiB/rank (KV trim 0.547 + NCCL 0.10-0.15). = 10% of goal. NOT reachable losslessly.
- Smallest config-expressible offload quantum: w13_weight = 1.406 GiB (single parameter). KV trim (0.547) alone cannot fund even one quantum.
- Smallest possible quantum with P4 (VMM page-granular): ~6-8 MiB (one expert slice, 2 MiB pages). Then KV trim funds ~8-9% of ords 20-22 experts moving back to GPU.
- True floor: expert decoder-half total 44.83 GiB/rank vs ~38-40 GiB GPU budget after everything else -> ~5-7 GiB must stay somewhere non-GPU. A capacity fact, not a kernel fact.

## B. Options ranked (lossless first, then lossy; single and combined)

### Tier 1 - lossless, config-only, low effort

| # | option | VRAM/rank | host RAM | PCIe traffic | effort | verdict |
|---|---|---:|---|---|---|---|
| 1 | KV right-size: kv_cache_memory_bytes 1124073472 -> 536870912 (0.5 GiB) | +0.547 | 0 | 0 | config line + restart | BEST SINGLE ACTION. Covers worst real workload (2x~203k); max_model_len auto-drops 1M->~520k. Funds nothing alone (0.547 < 1.406) but is prerequisite headroom for #4/#6 |
| 2 | ws3 PCIe one-shot patch (b12x pcie_oneshot.py:25 SUPPORTED_WORLD_SIZES += 3, bind-mounted) | 0 | 0 | -6.0us vs 44.9us per all-reduce | 1-line patched file + bind mount + restart | PURE PERF: +2-3% steps/s decode at batch>=8, +1-2% at seqs=2. Does not change memory, but REDUCES THE COST of any residual offload. Combine with anything |
| 3 | UVA offloader hardening ports (#54972 empty_cache + #56413 skip-double-pin) into overlay | +0 (removes mid-walk OOM ceiling) | -3.4 GiB transient (halves walk spike) | 0 | ~5 LOC overlay | Enables SAFER deeper offload experiments; fixes observed expandable_segments warning. Do together with any restart |

### Tier 2 - lossless, code patch (moderate), high payoff

| # | option | VRAM/rank | host RAM | PCIe | effort | verdict |
|---|---|---:|---|---|---|---|
| 4 | P4: VMM mixed-residency expert offload (device pages = hot experts, HOST-LOCATION cuMemCreate pages = cold experts; single VA range; zero kernel change) | any target <=6.72 at 6-8 MiB quanta | same-or-less (converts KV trim into -0.5 GiB host) | -4-5x on offloaded layers (~215 -> ~50 MiB/step) | ~300 LOC shim + 20-line platform probe | THE STRUCTURAL FIX for "experts in RAM" cost. Keeps total offload bytes if desired but slashes PCIe; or offload FEWER bytes for the same traffic. Platform probe first (HOST-location VMM device access on GB202) |
| 5 | P4 + #1 + #3 combined: KV trim funds moving ~0.5 GiB of COLD experts back to GPU via P4 quanta | +0.5 | -0.5 | -4-5x on remaining | ~300 LOC + config | The honest optimum reachable without kernel changes: less expert bytes in RAM AND ~50 MiB/step PCIe AND +0.5 GiB GPU headroom |

### Tier 3 - lossless but upstream-gated (wait/track)

| # | option | verdict |
|---|---|---|
| 6 | nvfp4_ds_mla KV dtype (#57463 family; needs SM120 sliding-window record + image upgrade) | Track. Would shrink KV ~0.3-0.35 GiB at equal tokens on top of #1. Blocked upstream today |
| 7 | b12x expert-map/pointer-table kernel work (P2, ~400-500 LOC) aligned with vLLM #57794/#57943 HostBank | Upstream contribution, not critical path. Only if P4 probe fails |

### Tier 4 - lossy (only if user accepts capability/perf cost)

| # | option | VRAM | cost | verdict |
|---|---|---:|---|---|
| 8 | KV trim to 0.25 GiB (2x~100k) | +0.30 more | context cap ~100k | Probably wrong: real adapter worst case is 203k |
| 9 | max_num_batched_tokens 2048->1024 | +0.012 | prefill throughput | Not worth it (12-14 MiB) |
| 10 | embed/head FP8 conversion (derived checkpoint) | +0.4 | accuracy risk + violates checkpoint-integrity standard | No upstream support; hand port; skip |
| 11 | Reduce expert layers resident (offload ord 23 too) | -1.41 GPU | +PCIe on text path | REJECTED by constraint (text perf) |

### Rejected/phantom (do not spend time)

- FI workspace env trims (phantom - never allocated in native b12x stack)
- DMA capacity trims (never initialized on ws3)
- Graph pool knobs (0.02 GiB pool)
- Allocator knobs (38.9 MiB waste, expandable_segments already optimal)
- DCP3 (cannot start on this pin/model; perf-negative at this traffic)
- max_concurrent_batches/UVA pool knobs (host-side)
- Dedup/tying (no duplicates exist; embed/head genuinely differ)

## C. Recommended plan (ordered)

1. NOW (config-only, one restart each or batched): #1 KV right-size + #3 UVA hardening ports + #2 ws3 one-shot patch. Expected: +0.547 GiB/rank GPU, +2-3% decode steps/s, safer deeper offload, all lossless. Batch all three into ONE placement restart with the functional suite as gate.
2. NEXT (code, ~1 day): P4 platform probe (20-line script: cuMemCreate HOST-location + device-read test on GB202). If green: implement the ~300 LOC VMM shim, A/B the PCIe win with the 1-token TTFT probe. This is the only path that structurally reduces "experts in RAM" pain beyond one ordinal.
3. TRACK: upstream #57463 (nvfp4 KV), #57943/#57794 (HostBank), fork integration branch updates. Re-run this ledger when the pin moves.

## D. Honest bottom line
- Full expert residency (drop ALL expert offload) is NOT achievable on 3x95 GiB with this checkpoint: needs +6.72 GiB/rank vs 0.65-0.70 achievable. The 5-7 GiB host cold-tier floor is a capacity fact.
- The COST of the residual offload (PCIe on text decode) IS reducible: #2 (all-reduce speed) + #4/#5 (cold-expert residency, 4-5x traffic cut). That is the real win available.
