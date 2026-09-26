# 19 — DCP3 study: would decode_context_parallel_size=3 help the DS4.1 TP3 placement?

2026-09-24, read-only static analysis of the installed tree in the running container
`ds41_3gpu_karmic_r1` (engine :5807, pin `2b86d3d80` = vllm 0.1.dev21735+g2b86d3d80, b12x 1.3.0).
No restart, no config change, no probe traffic. Question: does DCP3 (or any DCP>1 valid on 3 GPUs)
restructure KV/attention memory enough to bring the 6.72 GiB/rank of host-backed expert ordinals
20-22 back into VRAM (vision/aligner stay host-resident)?

**Verdict up front: no — DCP3 is a hard no on this placement.** Three independent blockers mean it
cannot even reach weight load on this pin and model (§3, §4, §6); even with all of them patched,
the VRAM relief is bounded at ~0.65-0.70 GiB/rank and exists only in the 1M-context decode regime
this host never routes to DS4.1 (§7), it never touches expert weights (§6), and it adds per-layer
collectives that would slow the actual workload (§8). The 6.72 GiB goal is a weight-budget problem;
DCP is a KV-placement feature. Details and citations below.

## 1. What DCP is in this engine: KV sharding across the *existing* TP ranks

- `vllm/config/parallel.py:353-356` — `decode_context_parallel_size`: "Number of ranks that shard
  the decode KV cache. DCP does not expand the process world size. Without PCP, DCP reuses TP
  ranks." So TP3+DCP3 is a 3-process job, not 9; the same 3 GPUs 0-2 stay in use.
- Validity for 3 ranks (`parallel.py:559-566`): with PCP disabled, `tp % dcp != 0` raises;
  `3 % 3 == 0` passes. **TP3+DCP3 is a structurally valid config** in this engine version.
  (DCP can also run as TP1+DCP3 — both reuse the same 3 ranks.)
- Group construction: `vllm/distributed/parallel_state.py:2165-2178` builds the DCP group from the
  same ranks (`dcp_size = decode_context_model_parallel_size or 1`), backend from
  `dcp_comm_backend`.
- Sharding granularity (`parallel.py:388-395`): KV is token-interleaved — with
  `cp_kv_cache_interleave_size=1`, "token `i` is stored on dcp_rank `i % dcp_world_size`". So the
  answer to "does each rank hold 1/dcp of the KV per token?" is **yes: each rank persistently holds
  ~1/3 of the tokens' KV** (block geometry is scaled too: `vllm/v1/core/kv_cache_utils.py:667-676`,
  `resolve_dcp_kv_block_size` returns `spec.block_size * dcp_world_size` for full-attention specs).
- At decode, each rank computes attention for its query rows over its local shard only; partial
  outputs are combined with per-rank LSE correction (`vllm/v1/attention/ops/dcp.py:375-423`,
  `correct_attn_out` / `CorrectAttnCPOutKernel`).
- Comm backend (`parallel.py:364-373`): `"ag_rs"` = AllGather + ReduceScatter (3 NCCL calls/layer),
  `"a2a"` = all-to-all of partial outputs + LSE (2 calls/layer). `None` resolves via
  `set_dcp_defaults` (`parallel.py:575-586`, called unconditionally at `vllm/vllm.py:1474-1475`) →
  default ag_rs. Only GlmMoeDsa overrides to a2a (`vllm/model_executor/models/config.py:52-56`);
  **DS4.1 sets no override** — matches the live banner `dcp_comm_backend=ag_rs` (01b-startup-accounting.txt:27).

## 2. Which KV groups would shard and which cannot (the memory-relevant split)

DCP shards **full-attention groups only**; everything else must be replicated per rank:

- `vllm/v1/kv_cache_interface.py:543-552` — `AttentionSpec.max_num_blocks_per_req` divides max_len
  by `block_size * kv_shard_count`, where `kv_shard_count = 1 if dcp_replicated else dcp_size`.
- `kv_cache_interface.py:581-586` — `FullAttentionSpec.max_memory_usage_bytes` divides
  `max_model_len` by dcp when not replicated. `MLAAttentionSpec(FullAttentionSpec)`
  (`kv_cache_interface.py:668-669`) inherits this → **the DS4.1 latent KV group is the shardable part**.
- `kv_cache_interface.py:896-899` — `SlidingWindowSpec.max_memory_usage_bytes` **asserts**
  `decode_context_parallel_size == 1 or dcp_replicated`: "DCP only supports sliding-window KV when
  it is replicated."
- `kv_cache_interface.py:937-960` — `CircularBufferSpec` (compressor ring) has no DCP division at
  all (1 block per request, `max_num_blocks_per_req` ignores dcp).
- `vllm/v1/core/kv_cache_coordinator.py:726-735` — for dcp>1 with a hybrid (multi-group) layout:
  "DCP with hybrid KV cache layouts only supports full-attention, Mamba, and replicated draft
  groups" — asserts every group spec is `FullAttentionSpec`, `MambaSpec`, or `dcp_replicated`.

DS4.1's group composition (from its own spec providers):

- latent KV: `MLAAttentionSpec` — `vllm/models/deepseek_v4_1/attention.py:608-620`
  (head_size 512, state_content_bytes 288, uint8, tokens_per_state=compress_ratio),
  kv-source layers only (4 layers, config `kv_source_layer_ids: [2,8,14,20]`,
  16-kv-rightsizing.md:38).
- SWA caches: `SlidingWindowMLASpec` — `attention.py:286-293` (window 128, 528 B/state);
  `dcp_replicated` is never set → False.
- compressor ring: `CircularBufferSpec` — `vllm/models/deepseek_v4_1/compressor.py:107-117`.
- recurrent/hc groups: empty at runtime (LMCache: "recurrent KV cache groups: []",
  01b-startup-accounting.txt:26).

≥2 groups → `HybridKVCacheCoordinator` (`kv_cache_coordinator.py:1128-1145`) → the §2 asserts apply.

## 3. Blocker A — DCP3 cannot start on DS4.1's KV topology (two fatal asserts)

With `--decode-context-parallel-size 3` on this model, KV-cache init hits at least two independent
AssertionErrors (ordering immaterial; both are in the startup path):

1. `kv_cache_coordinator.py:726-735` — the `SlidingWindowMLASpec` group and the
   `CircularBufferSpec` group are neither `FullAttentionSpec`/`MambaSpec` nor `dcp_replicated`.
2. `kv_cache_interface.py:896-899` — `SlidingWindowSpec.max_memory_usage_bytes` with dcp=3 and
   `dcp_replicated=False`.

No code between config validation (`parallel.py:559-563`, which passes) and these asserts offers an
escape hatch for DS4.1: its spec providers (`attention.py:269-293,608-620`, `compressor.py:107-117`)
have no `dcp_replicated` or DCP-conditional logic whatsoever (grep `dcp|DCP` over
`vllm/models/deepseek_v4_1/` → **zero matches**).

## 4. Blocker B — the DS4.1-native attention/indexer path has no DCP implementation

The placement runs the native b12x topology ("V4.1 Full/Reindex/Reuse topology with native b12x
compute only", `attention.py:2`):

- `DeepseekV41B12xBackend` / `DeepseekV41B12xMetadataBuilder` / `DeepseekV41B12xMetadata`
  (`vllm/models/deepseek_v4_1/sparse_mla.py:150-252`) carry no DCP fields, no
  `dcp_local_seq_lens` consumption, no LSE-correction call, no query gather — nothing to execute
  under DCP>1. The generic MLA DCP machinery (`vllm/v1/attention/backends/mla/
  sparse_mla_attention.py:199-343`, `mla_attention.py:839-845,1343-1371`,
  `vllm/v1/attention/ops/dcp.py:1459+` `MLADCPManager`) is simply never wired for this backend.
- The generic indexer path that other DeepSeek models use is *asserted off* for this design:
  `vllm/model_executor/layers/sparse_attn_indexer.py:354-361` — "Candidate blocks are
  request-local; the DCP-sharded logits layout would need per-rank translation that is not
  implemented" → `assert dcp_world_size == 1` ("v4.1 two-level candidate filtering is not supported
  with DCP"). DS4.1's native equivalent hard-codes the same two-level scheme
  (`candidate_topk_blocks=2048` publish/mask, `attention.py:1061-1064,1176-1182`; b12x
  `dsa_indexer.Caps` max_candidates=16384 / candidate_topk_blocks=2048,
  `attention.py:1164-1182`) — and `b12x/attention/dsa_indexer/` + `compressed_sparse_mla/`
  contain **zero** DCP handling (grep → 0; the only b12x DCP code is QSA and the PCIe comm
  surfaces, neither used by DS4.1 attention).

So even if the §3 asserts were patched, DS4.1 under DCP3 would run KV-sharded storage with no code
to localize top-k candidates or combine partial outputs — i.e., a silent-corruption or crash surface,
not a feature.

## 5. Blocker C — the pin predates fork PR #887: DCP3 fails Triton compilation in generic DCP

Independently of DS4.1, the pin's generic DCP LSE kernel cannot compile at world size 3:

- `vllm/v1/attention/ops/dcp.py:283` — warmup key passes `n_rounded=dcp_world_size` (3) as
  `N_ROUNDED`; the kernel does `tl.arange(0, N_ROUNDED)` (`dcp.py:155`) — Triton requires a
  power-of-two range. Verified: no `next_power_of_2` wraps this value in the pin (the only such use
  is `NUM_SEQS_POW2` at `dcp.py:896`).
- Fork **PR #887** (github.com/local-inference-lab/vllm, opened 2026-09-24, titled "fix: DCP LSE
  correction for non-power-of-two ranks; pad aligned state rows with NULL_BLOCK_ID"): switches to
  `next_power_of_2(N)` lanes with extra lanes masked to -inf, plus a mamba CUDA-graph padding fix;
  tests cover world sizes 2/3/4/6; E2E on GLM-5.3-Flash **TP3** MTP and Qwen passed (PR page, E2E
  table 12:50Z). It post-dates the pin (pin = PR #866 merge `2b86d3d80`; the PR targets
  `integration/karmic-kraken-beta` at a later beta). Its existence confirms 3-rank DCP is a real
  intended use case upstream — but for dense-MLA models (GLM/Qwen), not DS4.1's two-level sparse
  indexer.
- b12x context (KernelGranular scout, PR #297): that PR does not touch DCP/LSE (PCIe oneshot
  world-size-3 + GLM KDA geometry + nvfp4 alignment). b12x *does* ship DCP collectives
  (`b12x/comm/pcie/pcie_dcp_a2a.py:1010-1290`, `api.py:9-22` DcpAllToAll/DcpTopKOwnerExchange),
  but the PCIe DCP fast path supports world sizes 2/4/8 only — 3 falls back to NCCL
  (local-inference-lab/rtx6kpro docs, glm5.2_v11.md/v15.md). Irrelevant to DS4.1 anyway (§4).

## 6. Expert weights: DCP never frees them; attention weights can only gain bytes

- DCP does not expand the process world size and does not re-shard MoE experts
  (`parallel.py:353-356`; expert sharding follows the TP/EP axis, unchanged by dcp). The offloaded
  ords 20-22 are host-backed because the **total weight budget** (0.98 × 95 GiB minus KV, graphs,
  transients) cannot hold them — 16-kv-rightsizing.md:76: one ordinal step = 2.24 GiB. DCP changes
  none of the weight terms.
- `dcp_q_replicate` defaults to False (`parallel.py:375-383`; `vllm.py:1474-1475`; no DS4.1
  override) → attention projections stay TP-sharded as today, but decode adds a per-layer query
  all-gather (`mla_attention.py:1343-1349` path). Setting `dcp_q_replicate=True` instead replicates
  the full padded head set (72 heads at TP3) on every rank — effective TP per DCP group =
  `tp_size // dcp_world_size` = 1 (`vllm/model_executor/layers/linear.py:685-701`) — i.e. **added**
  weight VRAM per rank [INFERENCE on magnitude: order 1-3 GiB/rank across ~61 layers in fp8].
- Net: under any DCP>1 config valid on 3 GPUs, expert ords 20-22 remain host-backed. The only VRAM
  DCP frees is persistent KV pool, below.

## 7. KV memory math: DCP1 vs DCP3 for the same tokens

Anchors: fixed pool `kv_cache_memory_bytes=1,124,073,472` (1.05 GiB) = 1,131,122 tokens,
blended 0.9938 B/token, num_gpu_blocks=4933, block 128 (16-kv-rightsizing.md:28,42;
01b-startup-accounting.txt:20-21). MLA latent KV is single-latent-head → replicated across TP3
today, so the pool is per-rank identical (1.08× concurrency for a 1M request,
01b-startup-accounting.txt:21). A 1M request costs ≈0.97 GiB — latent-dominated
(16-kv-rightsizing.md:38); SWA is window-capped at 128 tokens × 528 B ≈ 67.5 KiB/layer/request and
the ring is one block/request — both noise at any context length.

| quantity (per rank) | DCP1 (today) | DCP3 (hypothetical, post-patch) |
|---|---|---|
| latent/full-attn KV for T tokens | T × ~1 B | T/3 × ~1 B (÷3, `kv_cache_interface.py:581-586`) |
| SWA + ring + recurrent groups | unchanged | **unchanged — assert-forced replicated** (§2) |
| pool for 1,131,122-token capacity | 1.05 GiB | ≈ 0.36-0.40 GiB → frees **~0.65-0.70 GiB/rank** |
| pool at realistic trim (540k tokens, 16-kv-rightsizing §3) | 0.5 GiB | ≈ 0.19-0.22 GiB → frees ~0.3 GiB more |
| pool at observed workload (67-token contexts) | 1.05 GiB fixed | same bytes (pool is pre-allocated; sharding changes capacity, not idle usage) |

Reading: DCP3's KV relief is real but (a) proportional to the *shardable* share and (b) only
materializes as a smaller *allocation* if someone re-tunes `kv_cache_memory_bytes` downward — and
the honest ceiling (~0.7 GiB at 1M-token capability, ~0.3 GiB at the already-recommended 0.5 GiB
trim, ~0 GiB at observed 67-token traffic) never approaches the **2.24 GiB per ordinal step**
needed for ordinals 20/21/22 (16-kv-rightsizing.md:76). The plain config trim already recommended
in 16-kv-rightsizing.md:74-84 frees 0.547 GiB/rank with one line and zero code risk — DCP3's
entire theoretical benefit at realistic scale is smaller than that.

## 8. Throughput/latency at max_num_seqs=2 (§5 of the brief)

- ag_rs adds ~3 NCCL collectives per layer per step (query all-gather + LSE all-gather +
  reduce-scatter; a2a would be 2 — `parallel.py:364-373`); decode FLOPs per rank drop ~3× because
  each rank attends only its 1/3 shard (`dcp.py:375-423`).
- At max_num_seqs=2 with observed contexts (max 67 tokens; 16-kv-rightsizing.md:27) the attention
  compute being split is microseconds, while ~61 layers × 3 collectives of NCCL latency per step on
  PCIe-attached workstation GPUs is not — **net slowdown** for this host's traffic. DCP is a
  long-context decode-latency feature (upstream DCP blog, vllm.ai 2026-08-07); the longest
  consumer-producible context here is ≈203k and it goes to qwen3.8, not DS4.1
  (16-kv-rightsizing.md:30,37).

## 9. Verdict

**DCP3 (or any DCP>1 valid on 3 GPUs) yields nothing toward the 6.72 GiB goal on this placement:**

1. **Cannot start**: `kv_cache_coordinator.py:726-735` + `kv_cache_interface.py:896-899` asserts fire
   on DS4.1's SWA/ring groups (§3).
2. **Not implemented for DS4.1**: zero DCP wiring in `deepseek_v4_1/*` and b12x
   dsa_indexer/compressed_sparse_mla; two-level candidate filtering is explicitly DCP-incompatible
   in-tree (`sparse_attn_indexer.py:354-361`) (§4).
3. **Pin too old**: generic DCP3 needs fork PR #887 (post-pin) to compile the LSE kernel
   (`dcp.py:283` pre-fix) (§5).
4. **Even fully patched**: ≤~0.7 GiB/rank theoretical KV relief at 1M-token capability, ~0.3 GiB at
   realistic scale, ~0 at observed traffic — vs 2.24 GiB per ordinal step; expert weights untouched;
   q_replicate variants add weight bytes (§6-7).

Relevant levers for the goal remain the ones already identified: `kv_cache_memory_bytes` trim
(0.547 GiB, 16-kv-rightsizing.md:74) and the offload-walk weight budget (04-instrument-inventory.md),
not context parallelism. Revisit DCP only if the placement is ever repurposed for sustained
100k+ token decode — and then only after rebasing onto karmic-kraken-beta ≥ #887 **and** a
DS4.1-native DCP implementation exists (it does not today).

### Key files

- vllm/config/parallel.py — DCP fields + validation (353-395, 559-586)
- vllm/v1/kv_cache_interface.py — shard/replicate semantics + SWA assert (506-508, 543-552, 581-586, 668-669, 896-899, 937-960)
- vllm/v1/core/kv_cache_coordinator.py — hybrid-layout DCP assert (726-735), coordinator selection (1128-1145)
- vllm/v1/core/kv_cache_utils.py — DCP block geometry (667-716)
- vllm/v1/attention/ops/dcp.py — LSE kernel, pre-#887 (103-173, 283, 155, 375-423)
- vllm/models/deepseek_v4_1/attention.py — DS4.1 specs + two-level candidates (269-293, 608-620, 1061-1064, 1164-1182)
- vllm/models/deepseek_v4_1/sparse_mla.py — native backend, no DCP (150-252)
- vllm/models/deepseek_v4_1/compressor.py — CircularBufferSpec (107-117)
- vllm/model_executor/layers/sparse_attn_indexer.py — two-level DCP assert (354-361)
- vllm/model_executor/layers/linear.py — q_replicate semantics (685-701)
- vllm/model_executor/models/config.py — who overrides DCP defaults (52-56)
- b12x/comm/pcie/* — DCP collectives exist, ws 2/4/8 fast path (api.py:9-22, pcie_dcp_a2a.py)
- Siblings: 16-kv-rightsizing.md (pool anchors), 01b-startup-accounting.txt (live config),
  04-instrument-inventory.md (ordinal step cost)
