# WS4 completion handoff: put experts back in VRAM

## 0. Instructions for the successor LLM

The user is benchmarking the running model. DO NOT restart/unload it, send inference or profiling requests, run GPU probes, attach to workers, change watched configuration, or edit any bind-mounted source. Do not run CPU-heavy project suites during that benchmark either. Work offline in an unmounted checkout/directory. Obtain explicit permission before GPU experiments or deployment. This handoff changes documentation only.

Primary objective: **increase persistent GPU residency of expert weights, reducing expert-weight RAM offload and host reads during decode.** WS4 is explicitly in scope. Boot success, a larger KV pool, valid chat output, and low HTTP first-byte time are not substitutes for that objective.

Retain the requested capacity targets: 1,048,576-token context, max_num_seqs=4, max_num_batched_tokens=4096. Do not silently trim them, increase offload of other expert layers, change precision, or drop WS4 to manufacture a success. If the combined targets cannot fit, present the measured shortfall and tradeoffs for a user decision.

User-observed current decode: about 64 tokens/s at C1 and 80 at C2. Accept this observation; establish the benchmark's aggregate/per-request convention when consuming its results, not by rerunning it now. The task is expert residency, not finding an alternative explanation to dismiss the slowdown.

## 1. Sources actually reviewed for this handoff

Current source:
- `config/ds41-tp3-offload/uva.py`
- `config/ds41-tp3-offload/uva_p4.py`
- `config/ds41-tp3-offload/vmm_policy.py`
- `config/ds41-tp3-offload/vmm_shim.py`
- `config/ds41-tp3-launch.yaml`

Recorded read-only container mount destinations (relative source paths above):
- `uva.py` -> `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py`
- `uva_p4.py` -> `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/uva_p4.py`
- `vmm_policy.py` -> `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/vmm_policy.py`
- `vmm_shim.py` -> `/opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/vmm_shim.py`
- `config/ds41-tp3-offload/pcie_oneshot.py` -> `/opt/venv/lib/python3.12/site-packages/b12x/comm/pcie/pcie_oneshot.py`
- `config/ds41-tp3-offload/pcie_tuning.py` -> `/opt/venv/lib/python3.12/site-packages/b12x/comm/pcie/_tuning.py` (destination basename differs)

The launch settings file is separately mounted at `/etc/lil-launch.yaml`. The placement command also mounts PCIe overlays; preserve them unchanged during WS4 comparisons. No candidate file may replace these mount sources during the user's benchmark.

Original design and investigation:
- `investigations/ds41-tp3-vram/23-full-program.md`, especially lines 24–33: WS4 is explicitly IN SCOPE, even if P4 works.
- `investigations/ds41-tp3-vram/18-kernel-granular-offload.md`, especially sections 0, 3, 6, 7.
- Also consult `13-granular-offload.md`, `20-lossless-savings-ledger.md`, `22-lossless-combination.md`, `24-ws1-results.md`, `25-ws2-vmm-probe.md` before repeating earlier investigations.

Historical b12x source was recovered read-only from Git, not from the running worker:

```sh
git show 08c1570:config/ds41-tp3-offload/bank/dynamic.py
git show 08c1570:config/ds41-tp3-offload/bank/_impl.py
```

These are historical reference snapshots, NOT implemented WS4 patches or proof of the presently imported image contents. Refresh a version-pinned source checkout and compare hashes before implementing. Old investigation line numbers are navigation hints, not safe patch anchors.

## 2. What Git history establishes

| Commit | Actual significance |
|---|---|
| `c79024e` | Expanded program explicitly includes kernel work; WS4 was not optional. |
| `93f81fa` | WS1 batch: KV reduction, PCIe world-size patch, UVA hardening. |
| `10fb14e` | VMM platform probe report. Its speculative ctypes explanations must be rechecked against headers. |
| `31b4db5` | Added initial shim, policy, conversion path and hook in UVA. Not a completed kernel-bank implementation. |
| `1ad945d` | Four-line shim fix: annotations and None hot-byte value. Commit prose overstates the scope of this individual diff. |
| `a178165` | Mounted/enabled initial P4 modules. Early green requests were later found to have used stock fallback. |
| `bb817cf` | Scale-name matching and live-shape padded layout; disabled VMM after bad generation. |
| `08c1570` | Replaced padded inter-expert storage with contiguous per-parameter regions, enabled partial residency. Also added two large unmodified kernel staging copies. Removed the active histogram-selection call from conversion. |
| `7218cd5` | Deleted those unused kernel copies. No bank ABI or kernel implementation was delivered. |
| `995c8ad` | KV 512 MiB -> 1 GiB/rank; sequences 2 -> 4; prefill 2048 -> 4096; cold fraction 0.75 -> 1. This relinquished the achieved partial expert residency. |

Review diffs, not commit messages as proof of runtime results. In particular, 969,421 KV tokens is NOT 1,048,576; reporting concurrency 1.00x at the automatically reduced 969,216 limit does not meet the requested context target.

The prior assistant dropped WS4 without user approval. Reopen all seven deliverables: kernel parameters/specialization; staging addressing; launch ABI/policy; canonical-storage ownership; vLLM policy wiring; numerical tests; upstream PR draft.

## 3. Current configuration and evidence boundaries

Recorded committed operating point (`995c8ad`):
- TP3 placement `ds41_3gpu_karmic_r1`; targeted ordinals 20–22.
- `DS41_VMM_ENABLE=1`, `DS41_VMM_COLD_FRACTION=1`.
- KV budget 1,073,741,824 bytes per worker.
- Launch YAML: 4 sequences, 4096 batched tokens, one parallel prefill, capture size 24, LMCache L1 48 GiB, L2 disabled, Engram disk mode.
- Recorded successful all-host boot reported 969,421 KV tokens and ~1.6 GiB free/rank after short requests.
- Earlier partial-residency boots with the enlarged capacity failed: one 1024 MiB allocation and a later 256 MiB allocation during b12x preparation. The latter failed with only 43.31 MiB free on rank 0. These are real startup failures, not drain-window artifacts.

At cold fraction 1 the converter requests zero device-hot pages. Assuming those allocations remain canonical through serving, it restores NO GPU residency for the targeted expert tensors. Other expert layers already resident on GPU are separate; do not say the whole model's experts are all offloaded.

Recorded page counts per targeted layer/rank were 720 + 45 + 360 + 23 pages, at 2 MiB/page: 2296 MiB/layer, 6888 MiB for three layers. These are allocation sizes including rounding, not measured process RSS. Towers were separately reported around 0.9 GiB/rank.

IMPORTANT unresolved accounting contradiction: conversation logs later showed stock `Total CPU offloaded parameters: 7.63` while claiming VMM conversion ownership. Current `uva.py:88–111` increments that counter only on the stock path; successful P4 conversion sets delta=0. Do not add this counter to VMM totals or call it authoritative VMM accounting. Investigate duplicate walks, later replacement, partial fallback, and exact boot identity during the approved verification window. A success log proves a swap happened at that point, not that the same backing survives preparation and serving.

Past evidence does NOT prove:
- decode parity or speedup; curl `time_starttransfer` is HTTP first-byte time, not first nonempty generated-token time;
- kernel numerical parity; filling and reading a tensor through from_blob is not a MoE-output comparison;
- actual 1M context execution;
- 4 simultaneous long generations or mixed 4K-prefill/decode stability;
- the historical estimated 4–5x PCIe reduction or 2–5% cold-route fraction. Those were projections, not measured achievements.

## 4. Concrete gaps in the code already written

These are planning findings, not instructions to patch live mounted files now.

### 4.1 `uva_p4.py`

- `_load_histogram()` at lines 44–54 is never called by `maybe_vmm_wrap_layer()`. The active loop (84–105) only calls `region_plan()`. No measured hot-expert selection exists.
- It allocates a separate contiguous byte range for each parameter and selects a prefix independently in page units. A page boundary may cut an expert row, and rounding differs across weights/scales. This is not a coherent arbitrary per-expert residency policy.
- Matching keeps the first suffix match per logical key across the whole module. Preflight must identify a single intended expert submodule, reject ambiguous matches, and validate actual names, dtypes, rank-local shapes and counts.
- It mutates `p.data` after each allocation/copy. If a later parameter fails, the module is partially converted; `uva.py` catches the exception and continues with stock offload. There is no transaction/rollback.
- `alloc` is local and not retained on the module or storage owner. `from_blob` has a no-op deleter. Current code therefore has no lifetime-coupled cleanup (not evidence of immediate GC unmapping: there is no destructor). Adding a destructor without fixing ownership would introduce use-after-free.
- Conversion does not mark residency ownership or idempotence. Stock code only recognizes `_vllm_is_uva_offloaded`, set by its own path (`uva.py:166`), not by P4. Establish protection against later walks replacing bank-backed parameters.
- No proof that preparation preserves these allocations; no explicit prepare-then-publish contract. Original WS3 plan required repack-first then cold placement, but current code converts during the module-construction walk.

### 4.2 `vmm_policy.py`

- `region_plan()` (28–37) validates neither finite fraction in [0,1] nor positive sizes/counts. Negative fractions can be silently capped to all-hot; >1 can produce negative hot-page counts. Reject malformed policy before allocating or mutating.
- Dormant `cold_experts()` sorts descending then takes `ranked[n_cold:]`: for a complete 384-entry histogram at fraction .75 it returns 96 entries, not 288. Missing expert IDs are also not completed consistently. This is not a working selection algorithm.
- Fixed `NUM_EXPERTS=384` and `BLOCK_BYTES=6 MiB` remain in dormant helpers/summary, while active conversion derives sizes live. The summary is not actual region accounting.
- Replace the split dead/active policy with a single tested per-layer expert-residency plan under a byte budget. Preserve rank-local tensor layout and logical expert identity.

### 4.3 `vmm_shim.py`

- None semantics are still broken: `hot_bytes=self.size` for None (239–243), but `_all_hot` is false because `bool(None)` is false (228–231). The cold branch then tries to create zero cold bytes (264–274). Fix all-hot, all-cold and partial states consistently; test small allocations, not multi-GiB probes next to production.
- `hot_pages` (305–309) returns an empty set for partial-hot allocations, not the actual prefix.
- Prefix validation does not enforce k <= num_pages; `n` is calculated but unused. Allocation size and wrap bounds are not validated.
- Partial allocation/map failure leaks resources: no unwind; handles are only recorded after mapping. `free()` ignores driver errors, is not clearly idempotent, and can invalidate still-live tensors or graph replay.
- There is no per-page remapping: `set_page_residency()` raises (311–315). Do not claim original repack-first/remap-later policy is implemented.
- Driver ABI must be re-derived from the pinned image's CUDA headers: `_Prop`, `_Acc`, HOST_NUMA enum, attribute 149, cuGetProcAddress version/signature and status handling, size_t/count/flag widths. A 32-byte struct size alone does not prove field layout correctness. Prefer supported bindings or a small header-compiled bridge rather than guessed ctypes structs.
- Original dlsym `cuMemMap` has no argtypes declaration in this source; the CFUNCTYPE-resolved replacement does. Thus past experiments do not establish a driver compatibility-stub bug: integer truncation is a confound. Correct the documentation only after controlled ABI verification.
- Module docstring says allocate before importing torch, while `ensure_init()` relies on caller context. Reconcile with an explicit context contract; do not create private contexts behind torch's back.
- NUMA ID is cached globally rather than by device and failures silently map to node 0. Use queried supported topology or a clear failure, not an invented fallback.

### 4.4 `uva.py`

- Broad exception fallback (76–93) can disguise failed/partial P4 conversion as a working stock placement.
- `empty_cache()` currently runs only when stock offload delta > 0 (94–99), never on successful P4 replacement even though original GPU tensor storage was released to PyTorch's allocator. This is a candidate startup-headroom issue, NOT a proven explanation of prior OOMs. Measure allocated/reserved/external bytes before deciding the fix; do not sprinkle empty_cache on the decode path.
- CPU-offload byte limit/filter accounting and P4 ownership are not unified. The current P4 branch runs before the stock filters/budget. Establish one authoritative residency ledger and prevent accidental double offload.

## 5. WS4 target architecture

### Four pointer tables, not one ambiguous bank

Original contract: four GPU-resident Int64 arrays of length E, holding absolute **byte addresses** for logical expert e:
- `w13_bank[e]`: prepared FC1/gate-up weights;
- `down_bank[e]`: prepared FC2 weights;
- `sfbw13_bank[e]`: matching prepared FC1 scales;
- `sfbdown_bank[e]`: matching prepared FC2 scales.

Each expert can point to a GPU-resident or device-readable host allocation. Tables are indexed by the logical/local expert identity expected by routing. Preserve gate/up ordering, scale encoding, TP-local shapes, alphas and routing weights exactly.

A compile-time bank/legacy specialization keeps existing arithmetic and no extra bank loads when disabled. Bank mode must cover all required FC1/FC2 and scale paths. Unsupported recipes must reject bank mode explicitly, not silently ignore it.

**Pointer indirection does not create VRAM.** It enables arbitrary hot-expert selection and tighter allocation granularity, turning available bytes into useful persistent expert residency. The plan must independently fund that residency through real memory savings and safe preparation peaks. Do not promise all model experts fit based on bank support alone.

### Ownership and lifecycle

Extend the existing canonical expert-weights descriptor rather than building a detached collection of pointers. It must own:
- the four pointer tables;
- GPU/host backing allocations and tensor views;
- exact layouts, expert-ID mapping, per-expert byte sizes, residency mask;
- preparation-completion events and graph/plan lifetime relationships.

Preparation must produce final canonical storage without permanently retaining original full GPU tensors plus repacks plus new banks. Release superseded storage only after the dependent copies/kernels complete. Use bounded expert/chunk preparation if a whole-layer or all-device staging phase exceeds peak budget. The original 'repack all on GPU first' suggestion is conditional on headroom, not a requirement to OOM.

Static placement at model initialization first. No per-token migration or adaptive live rebalance is required to complete this work. If expert_map is implemented for upstream convergence, specify logical-ID/local-slot semantics and test them; four pointer tables alone do not automatically implement every vLLM expert_map contract.

Before integration, present the addressing decision for review: explicit expert strides can describe a regular padded layout, but cannot express arbitrary independently located expert bases. The approved WS4 contract is four pointer tables; do not replace it with stride-only support without approval and evidence that the required placement semantics remain covered. State whether the opt-in bank representation replaces prefix-only placement for supported recipes; retain legacy contiguous execution as the comparison/control, not a second silently competing selection policy.

## 6. Implementation sequence and acceptance criteria

### A. Establish exact source/version and restore WS4 tracking

1. Open a separate unmounted worktree/source checkout. Do not edit current `config/ds41-tp3-offload/uva.py`, `config/ds41-tp3-offload/uva_p4.py`, `config/ds41-tp3-offload/vmm_policy.py`, or `config/ds41-tp3-offload/vmm_shim.py` during the benchmark: they are bind-mounted. Do not edit `config/ds41-tp3-launch.yaml` or `config/models/host-cuda-00-01-02-03-ds41-karmic.yaml` either.
2. Preserve original program scope and reopen the seven WS4 deliverables. Save source hashes, image digest, package versions and relevant commit IDs in the handoff for the implementation branch. No secrets, host environment dumps or checkpoint copies.
3. Recover historical kernel snapshots using the Git commands above only as navigation aids. Confirm actual source compatibility against the pinned image when safe; no imports that initialize CUDA during the benchmark.
4. Trace real vLLM prepare/apply -> b12x API -> prepared descriptor -> launch -> staging for the current shape and C1/C2/C4 plus prefill.

Acceptance: an exact source map and a supported-path matrix, with no speculative filename/branch guessing.

### B. Fix policy/ownership prerequisites in the offline candidate

1. Define a byte-budgeted per-layer plan using measured expert route counts (or an explicitly named deterministic baseline policy when no counts exist). Counts must carry model/layer/expert identities, sample size and workload provenance.
2. For equal-size experts, prioritize higher route counts; for differing sizes use explicit benefit/cost accounting. Keep all four tensors of an expert coherently placed. Avoid claiming that low numeric IDs are hotter.
3. Validate missing counts, ties, zero/full budgets, expert counts and finite fractions. Remove or migrate dormant helpers rather than maintaining two policy conventions.
4. Implement transaction-safe conversion/publication and ownership cleanup. Fix shim defects only if retaining that shim for candidate backing; otherwise remove obsolete candidate paths after a clean cutover.
5. Record bytes of each canonical allocation exactly once: GPU expert bytes, host expert bytes, padding, pointer tables, persistent repacks, temporary peak allocations.

Acceptance: deterministic placement under a budget; no partial module mutation on failure; no stale pointers, double frees or unowned allocations.

### C. Implement kernel parameters and staging addressing

Historical source in `08c1570`:
- `b12x/moe/_shared/kernels/dynamic.py`: `_w4a8_stage_repacked_b_half` 251, `_full` 288, scale half/full 308/330; `DynamicLaunchParams` 581; `_ld_global_u64` 664.
- Original investigation identifies **N128 repack as the DS4 TP3 path**: intermediate 768, `n64_repack=False`, `w4a8_b_tma=False`. Confirm on the pinned candidate. Do NOT implement only the six N64 helper call sites examined late in the prior session.

1. Inventory every expert-dependent address: FC1 gate/up halves, full-tile paths, FC2/down, materialized intermediate paths, scales and applicable residual branches. Include decode and prefill specializations selected by the requested capacities.
2. Replace only the inter-expert address term in bank mode: `base_bytes = load(bank[e])`; add the existing correctly scaled intra-expert offset. Preserve packed layout within each expert.
3. Maintain an explicit units ledger: byte pointer vs uint32 index vs element index. Scale offsets and weight offsets do not necessarily share units.
4. Load each needed base once per reusable task/pair where safe, not redundantly inside each transfer iteration. Verify generated code rather than asserting overhead is negligible.
5. Keep legacy specializations unchanged. Keep TMA nvfp4/non-repacked recipes out of scope per the original program; reject bank mode there unless separately approved. Cover N64 if claiming it is supported; otherwise explicitly reject it while completing the actual N128 path.
6. If banks travel through `DynamicLaunchParams`, update constructor, `__extract_mlir_values__`, `__new_from_mlir_values__` and every construction site together. Choose the actual entrypoint plumbing after tracing the complete call chain, not from a single helper signature.

Acceptance: supported legacy and bank kernels compile; all address sites accounted for; nontrivial expert-ID access produces matching results, including scale and FC2 paths.

### D. Extend launch ABI, preparation contract and caches

Historical `_impl.py` anchors from the recovered snapshot:
- `B12XFP4ExpertWeights` 425; canonical storage checks 492–515.
- N64 selection in preparation 6643.
- `_launch_dynamic_flat` 11163.
- `_encode_dynamic_launch_policy` 11566, decoder 11595.
- `_tp_moe_dynamic_launch_op` 11642; weight operands 11668–11679.

1. Extend the descriptor/representation and validation for bank-owned canonical storage. Do not just delete existing pointer-equality checks: legacy guarantees remain, bank mode needs equivalent ownership/shape/layout guarantees.
2. Thread four banks through public/internal APIs, custom-op schemas, fake/meta paths, compile-planning fake pointers, real launch bindings and prewarmed wrappers.
3. Version or extend policy encoding safely: existing bits 0–6 are occupied and active-cluster encoding starts at bit 7. Update encoder/decoder/all consumers together; do not steal an in-use bit.
4. Include bank mode/layout in compilation cache identity. Ensure runtime plans bind the correct table/storage identity and retain ownership; invalidate or rebind plans after any approved relocation. Never reuse a cached graph bound to retired addresses.
5. Validate device placement, table length, address alignment required by vector loads (at least existing 16-byte contracts where used), complete expert coverage and dtype/layout compatibility.
6. Remove source/full-repack duplicates only at a verified lifetime boundary. Prove bank mode does not silently allocate a full contiguous GPU copy for an existing helper.

Acceptance: cold compile, warm cache, custom-op/fake planning, prewarm and CUDA graph replay all consume the same owned representation correctly; legacy and bank plans cannot cross-bind.

### E. Wire vLLM and achieve GPU residency

1. Trace `vllm/model_executor/layers/fused_moe/b12x.py` and the native DS4 implementation identified by the pinned source (original investigation: `vllm/models/deepseek_v4_1/quant_config.py`). Avoid generic-model paths not used by this image.
2. Wire plan/prepare/apply to the bank representation at the right point in weight loading. Preserve parameter metadata needed by loaders and quantization. Do not merely replace raw `p.data` and assume downstream preparation honors it.
3. Introduce one explicit policy/config path consistent with existing b12x/vLLM conventions. Respect launcher argument separation; podman `-e` is required for container env delivery. Stage launcher/config changes inertly until approved.
4. Build a stage-by-stage memory ledger: load, source retirement, expert preparation, attention/dsa_indexer preparation, KV allocation, graph capture, first decode and mixed load. Distinguish PyTorch allocated/reserved bytes from external VMM allocations.
5. Investigate b12x preparation buffer lifetimes/reuse and lost allocator headroom before sacrificing GPU experts. Do not assume expert preparation caused the reported `attention.dsa_indexer` OOM: trace that specific preparation path.
6. Spend verified recovered memory on selected experts. Report resident expert count and bytes per layer/rank before and after. If zero experts return to VRAM, this stage has NOT achieved the goal.

Acceptance: a positive, directly verified expert-residency increase that survives cold startup and serving at the approved capacity, without a hidden extra full GPU copy or silent fallback.

### F. Numerical, memory and performance gates (only after permission)

Use small fixtures first in an isolated GPU window; no full-size duplicate model allocations alongside production. Derive fixtures from actual packed formats, not oversized guessed BF16 shapes.

Required numerical cases:
- legacy vs bank mode with identical prepared weights, inputs and routing;
- all-device bank mode (isolates indirection from memory residency);
- mixed device/host backing with non-prefix hot IDs and alternating residency;
- first/middle/last expert, duplicate IDs, top-k weighting, gate/up order, FC2 and scales;
- supported decode/prefill tile variants, TP-local dimensions and rank mapping;
- graph capture/replay, plan-cache reuse, source retirement and failure cleanup.

Compare expert outputs and fused MoE outputs with a justified tolerance; compare packed payload bytes where exact identity is expected. A fill/readback shim test is necessary allocator proof but insufficient kernel proof. Keep meaningful regressions; no source-text tests or tautological forwarding tests.

Measure performance after correctness:
- Same image, checkpoint, parallel topology, prompts, output length, sampling, scheduler/capacity and cache state.
- Separately report C1/C2/C4 per-request and aggregate decode tok/s. Decode interval must exclude prefill; use emitted-token timestamps and verified counts. HTTP first-byte time is not TTFT.
- Record actual first nonempty generated-token time and prefill throughput independently.
- Record cached_tokens and controlled warm/cold cache state. Do not infer decode speed from end-to-end throughput affected by prefix reuse.
- Capture expert residency, routed hot/cold frequency, PCIe traffic and peak/free VRAM. Profile in a separate pass if collection affects timing.
- Exercise a full 4K prefill overlapping decode, four long-running requests, a cold start, and the full 1,048,576 context target when resources/test window are approved. A startup KV-capacity line alone is not that long-context test.
- Full functional suite: basic answer, reasoning `3/10`, `get_weather({"city":"Utrecht"})`, valid PNG recognized as red, alias `ALIAS READY`.

Acceptance: no numerical regression, no repeatable decode regression beyond documented measurement variation, a demonstrated residency improvement, and successful requested-capacity operation. Quantify the gain, not just 'green'.

### G. Deliver, deploy and upstream

1. Produce focused implementation commits in the proper b12x/vLLM sources, with patch provenance and baseline hashes. Do not commit untouched 24k-line staging copies as if they were an implementation.
2. Provide an upstream b12x PR draft: pointer-table contract, supported/excluded recipes, canonical ownership, plan/cache implications, numerical evidence and measured performance. Do not claim expert_map/StagingBank compatibility without its contract tests. Do not publish a PR without authorization.
3. Stage exact immutable overlay files, digests and an inert placement candidate. No modification to current mounted sources until the user ends benchmarking and approves deployment.
4. Validate staged config before activating watched files; deploy in one coordinated batch. Verify `Configuration Reloaded`, exact managed command/env/mounts and real backend readiness. Proxy `/v1/models` and LMCache's own Uvicorn port are not vLLM readiness.
5. Stop on `EngineCore failed to start`; cancel waiting client jobs, preserve logs and restore the approved known-good state. Do not repeatedly poll a dead worker or call OOM-related 502s harmless drain windows.
6. Run relevant project gates when permitted: config validation, formatting/whitespace, `make test-all`; `make test-dev` only for proxy-code changes. Kernel/backend numerical/performance gates remain separate and mandatory. Commit reusable config; leave local `config.yaml` uncommitted.

## 7. Suggested task ownership after the design is fixed

Do not split a shared ABI before agreeing exact field types, units, supported paths and ownership.
- Integration owner: descriptor/bank ownership, policy schema, memory budget, final acceptance and deployment.
- Kernel owner: `dynamic.py`, all staging branches and generated-code checks.
- Host ABI owner: `_impl.py` planning/real/fake signatures, cache identity and canonical contract; coordinate with kernel owner on a frozen contract.
- vLLM owner: prepare/apply/loading integration and expert selection using the agreed descriptor.
- Validation owner: fixtures, numerical parity and benchmark harness consuming the agreed API; no live experiments until approved.

All agents work in offline copies and skip project-wide validation during concurrent edits. Integrate once, validate once per coherent candidate. No agent may change production to make its local test pass.

### WS5 remains tracking-only, not cancelled

Per `23-full-program.md:35–38`, track the upstream-gated nvfp4 DS MLA KV format, HostBank/StagingBank work, and pinned-fork updates. Recheck upstream status before any future proposal; historical issue statuses and projected memory savings are not current evidence. WS5 neither replaces WS4 nor authorizes an image upgrade, cache-format change, or runtime experiment during this task.

## 8. Exit report template (required)

- Exact commits/image/patch hashes and effective runtime settings.
- Before/after GPU-resident expert count and bytes per layer/rank.
- Before/after host expert allocation bytes, including rounding and any duplicates.
- Startup peak and minimum headroom, including attention/dsa_indexer preparation.
- Actual context capacity and long-context verification status; 4/4096 verification status.
- C1/C2/C4 decode, TTFT, prefill, mixed-load results and cache-state provenance.
- Numerical parity results and supported kernel paths.
- Upstream draft location and rollback instructions.
- Anything not finished explicitly listed; do not drop WS4 or relabel lower capacity as success.

**Definition of done: more useful experts remain in VRAM, the cold tier shrinks or is accessed less because of that residency, decode benefits are measured, and the agreed capacity/startup/correctness constraints still hold. A working all-host allocation shim alone is not completion.**
