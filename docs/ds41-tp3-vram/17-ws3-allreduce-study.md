# 17 — WS3 all-reduce study: b12x PCIe one-shot at world size 3

Date: 2026-09-24. Scope: why the b12x PCIe one-shot all-reduce never initializes on the TP3
placement (`ds41_3gpu_karmic_r1`), upstream status, why 3 is excluded, and the exact
patch/delivery/verification plan. Read-only investigation; no probe requests sent, no config changed.

## 1. Where ws3 is excluded — pure Python, one constant

Warning source (exact text match to the observed failure):

- `b12x/comm/pcie/pcie_allreduce.py:92-101` — `_algorithm_for_world_size()` raises
  `ValueError(f"unsupported PCIe all-reduce world size {world_size}; supported world sizes are {SUPPORTED_WORLD_SIZES}")`.
- The tuple it prints is assembled at `pcie_allreduce.py:83-89`:
  - `MAX_DIRECT_WORLD_SIZE = 8` (:83)
  - `DIRECT_WORLD_SIZES = tuple(ws for ws in ONESHOT_WORLD_SIZES if ws <= MAX_DIRECT_WORLD_SIZE)` (:84-88)
  - `SUPPORTED_WORLD_SIZES = (*DIRECT_WORLD_SIZES, *HIERARCHICAL_WORLD_SIZES)` (:89)
- Root constant: `b12x/comm/pcie/pcie_oneshot.py:25` — `SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)`
  (b12x 1.3.0; no 3).
- Hierarchical adds only TP12/16: `pcie_hierarchical.py:29` — `SUPPORTED_WORLD_SIZES = (12, 16)`.
  Printed set at ws3: `(2, 4, 6, 8, 12, 16)` — matches the assignment's warning verbatim.

The same constant is re-validated inside the runtime; the single-line patch clears all gates:

- `pcie_oneshot.py:1448-1450` (`_CuTeOneshotBackend.init_custom_ar`), `:2144-2145`, `:2470-2471`,
  `:3426-3427` (channel/pool constructors): `if world_size not in SUPPORTED_WORLD_SIZES: raise
  ValueError(f"unsupported world size {world_size}")`.

vLLM side — no vLLM patch needed:

- `vllm/distributed/device_communicators/b12x_pcie_all_reduce.py:278-305`: the wrapper constructs
  `allreduce_cls.from_exchange_group(...)`, catches the ValueError, `_all_ranks_succeeded` → warning
  `B12X PCIe all-reduce initialization failed on rank %d: %s` → `self.disabled = True`. That is how
  the b12x message reaches our log.
- `vllm/distributed/device_communicators/cuda_communicator.py:75-86,131-139`: `use_b12x_allreduce`
  is gated on the global custom-allreduce enable flag + envs, not on ws3; the adapter object IS
  constructed at ws3 and disables itself. Backend selection then logs `['PYNCCL']` (:357-364).
- `vllm/distributed/device_communicators/custom_all_reduce.py:108,183-191`: stock CA
  `_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8, 16]` → "Custom allreduce is disabled due to an unsupported
  world size: 3" (seen in `01b-startup-accounting.txt:24`). Separate gate, uninvolved in this plan.

Compiled? **No.** `glob b12x/**/*.so*` → no files. `b12x/comm/pcie/api.py:52-56` `is_supported()`
docstring: "Device programs are compiled and primed by PreparationSession before use; no
repo-authored C++/CUDA extension or runtime nvcc build is involved" — kernels are CuTe-DSL JIT
(`b12x/_lib/compiler.py`; `KernelCompileSpec.from_key("comm.pcie.oneshot", ...)` at
`_oneshot_cute.py:1268`). The ws gate is a Python tuple ⇒ patchable via read-only bind mount,
same mechanism as the existing `uva.py` override.

DMA has its own, separate gate — deliberately left closed (see §4): `pcie_dma.py:36`
`SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)`, raised at `:189-193`.

## 2. Upstream status

- Package origin: `b12x-1.3.0.dist-info/METADATA:3-5` — b12x 1.3.0, "DGX Spark and RTX 6000-focused
  inference kernel library", Author Luke Alonso; no repo URL in metadata. Public repo confirmed:
  `github.com/local-inference-lab/b12x`.
- Upstream master today still excludes 3: raw
  `master/b12x/comm/pcie/pcie_oneshot.py:24` = `SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)`.
- **Issue #410** — "feat(comm): PCIe oneshot all-reduce for world size 3 (TP3)", OPEN, opened
  2026-09-21 by mark-yong. Identical failure on LIL r38 (DeepSeek-V4.1-Flash TP3 + decoder-half
  UVA); Pete's gist rig measured PYNCCL TP all-reduce ≈40% of decode kernel time (Gen5
  x16/x16/x8). Hypothesis: "allowlist/specialization/qualification gap rather than something that
  needs a new collective". Acceptance ordering: GPU oracle on rank-distinct inputs → CUDA graph
  capture+replay → torture suite at ws3 → only then perf.
- **PR #297** — "pcie: world size 3 for the one-shot/DMA all-reduce; GLM-5.3-Flash TP3 policy
  corpus", OPEN, label `status:deferred`, opened 2026-09-02 by MadeBy561. Core diff = add `3` to
  `SUPPORTED_WORLD_SIZES` in `pcie_oneshot.py` and `pcie_dma.py`; PR text: "the generic one-shot
  CuTe kernel iterates `(rank + i) % world_size` over the 16-slot peer tables and the DMA ring is
  `2 x (world - 1)` steps; nothing else gated 3". Pre-merge checks FAILED (performance-claim
  evidence + prose) → deferred, unmerged. Claimed TP3 C1 56.4→62.0 steps/s (unevidenced in-repo).
- Pinned image (vllm g2b86d3d80, b12x 1.3.0, `ghcr.io/local-inference-lab/vllm@sha256:d9d273c6…`)
  predates any fix — and no merged fix exists anywhere yet (neither repo).
- vllm-project/vllm: not applicable — stock custom AR ws3 disabled upstream (compiled dispatch
  2/4/6/8; FlashInfer fused AR ∈ {2,4,8,16}); issue #410 explicitly scopes vLLM custom AR out.

## 3. Why 3 is excluded; correctness risk of adding it

Not mathematical. The generic pull path is world-size-generic:

- Barrier: `_oneshot_cute.py:254` `if tidx < Int32(self._world_size)` — one thread per rank, any
  ws ≤ 16.
- Reduce: `_oneshot_cute.py:350-352` `for peer_index in cutlass.range_constexpr(self._world_size):
  peer_rank = (self._rank + peer_index) % self._world_size`; fused `stage_pull` equivalent at
  `:862-863`.
- Pointer tables padded to `_MAX_RANKS = 16` (`pcie_oneshot.py:1332`, zero-padded `:1436`) — three
  entries fit trivially.
- FP32 accumulation per 16-byte pack (`_load_accumulate`, `_oneshot_cute.py:178-214`) — same
  algorithm class as ws4/6/8; no evenness assumption anywhere in pull.

What IS ws-gated (and stays gated at 3): topology transports — TP2 remote-push (`:1691`), TP4
remote-push (`:1685-1690`), TP8 owner (`:1677-1683`); launcher-level rejection
`_oneshot_cute.py:509-512` ("fused remote push requires TP2 or TP4", "fused owner transport
requires TP8"). At ws3 `_fused_topology_mode` returns None → generic `stage_pull`. Two-shot gate
`(2,4,8)` (`pcie_twoshot.py:42`) untouched; bf16 two-shot is ws4-only (`pcie_twoshot_bf16.py:52`).

So the even-only allowlist is a **qualification** statement, not an algorithmic one: #297
"nothing else gated 3"; #410 "WS3 has never been through the actual qualification path" (the GPU
torture test hardcodes the even allowlist).

Correctness risks to gate on:

1. Numeric — low. Pull accumulates all ranks in FP32; only reduction ordering differs vs NCCL
   (bf16 tolerance). Verify with rank-distinct/adversarial inputs, not zeros (a stale or missing
   rank silently passes on zeros — #410's point).
2. CUDA graph — decode runs FULL_DECODE_ONLY graphs (`01b-startup-accounting.txt:25`). Cold capture
   is already rejected by design (`_oneshot_cute.py` `is_oneshot_launcher_prepared` check → "cold
   PCIe oneshot CUDA graph capture is not allowed; call prepare_graph_all_reduce() before
   capture"), i.e. a preparation miss at ws3 fails startup loudly, not silently. The kernel
   compile key includes `world_size` (`_oneshot_cute.py:1164-1166`) → fresh JIT specialization for
   ws3 on first boot, cached afterward.
3. Unknown unknowns — compile-cache / IPC-layout / planner assumptions at odd ws (#410's caveat).
   Mitigation: keep the patch byte-identical to #297's core so upstream qualification transfers
   when it merges.

## 4. Delivery plan (convention: read-only bind mount of a patched host file + placement restart)

Precedent in this exact placement: `config/ds41-tp3-offload/uva.py` is already bind-mounted
read-only into site-packages (`config/models/host-cuda-00-01-02-03-ds41-karmic.yaml:70`, cmd tail:
`-v …/config/ds41-tp3-offload/uva.py:/opt/venv/lib/python3.12/site-packages/vllm/model_executor/offloader/uva.py:ro
ghcr.io/local-inference-lab/vllm@sha256:d9d273c6890060eb5eae511a34efd724dd5c29b6dd767c55f149e0da04a7a4c3 …`).

1. **Stage the patched file**: copy the container's `b12x/comm/pcie/pcie_oneshot.py` →
   `/home/gerben/ai/llama-swap/config/ds41-tp3-offload/pcie_oneshot.py`, one-line edit (line 25):

   ```
   SUPPORTED_WORLD_SIZES = (2, 4, 6, 8, 10)   →   SUPPORTED_WORLD_SIZES = (2, 3, 4, 6, 8, 10)
   ```

2. **Bind-mount line** to insert into the placement `cmd` (next to the uva.py mount, before the
   image ref):

   ```
   -v /home/gerben/ai/llama-swap/config/ds41-tp3-offload/pcie_oneshot.py:/opt/venv/lib/python3.12/site-packages/b12x/comm/pcie/pcie_oneshot.py:ro
   ```

3. **Env / max size.** Current chain: vLLM default `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: str =
   "84KB"` (`vllm/envs.py:243`, resolver `:2010-2012`); hardware profile
   `/opt/lil/runtime/hardware/rtx-pro-6000-pcie.yaml` `model_environment: ds41-flash:`
   `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE: 96KB` (fused 96KB) — the live 96KB. Precedence:
   hardware `model_environment` is applied after hardware env
   (`/opt/lil/runtime/launcher.py:435-436`); placement `-e` passthrough wins over both (launcher
   `explicit_env` merge `:540-556`; VLLM_/B12X_/NCCL_ prefixes accepted; empirically placement
   `-e DS41_OFFLOAD_*` land in PID 1 env — `04-instrument-inventory.md` §g).
   - Required env change: **none** for correctness — 96KB already covers steady decode payloads
     (rows ≤ 9 at hidden 5120 bf16).
   - Recommended optional: `-e VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=512KB` so every captured
     decode graph batch (rows ≤ 24 → 240 KiB) routes one-shot instead of capturing PYNCCL for
     rows > 9. 512KB is the measured crossover where one-shot stops winning (upstream rtx6kpro
     doc: 1 MB → NCCL wins); slab cost ~2×max_size per channel/rank (~1 MiB) vs ~2.0 GiB free —
     negligible. Do NOT raise further: full-prefill reduces (2048 × 5120 × 2 B = 20 MiB) belong
     on NCCL.
   - Do NOT patch `pcie_dma.py`: DMA admission additionally requires
     `nbytes % (world_size × 8) elems == 0` (`b12x_pcie_all_reduce.py:499-501`); at ws3/hidden
     5120 that demands token counts divisible by 3 — the fixed 2048-token chunk fails, so DMA
     would never engage at ws3. Its init will warn `PCIe DMA all-reduce initialization failed …
     will use PyNCCL` — expected and benign.
4. **Apply**: edit the placement cmd in `config/models/host-cuda-00-01-02-03-ds41-karmic.yaml`,
   reload llama-swap config (the stored cmd lives in llama-swap's process memory — a bare
   `podman stop` would relaunch the OLD cmd), let llama-swap restart the placement
   (`--replace --rm` recreates it). GPU 3 companions untouched.

## 5. Verification plan

Init proof (engine log, rank 0):

- PASS line — `b12x_pcie_all_reduce.py:319-326`:
  `Using B12X PCIe all-reduce (algorithm=oneshot, one-shot max=<bytes>, fused max=<bytes>, two-shot bf16 max=off, DMA min=6291456).`
  grep: `grep -E "Using B12X PCIe all-reduce"` → must show `algorithm=oneshot`.
- Backend selection — `cuda_communicator.py:357-364`:
  `Using ['B12X_PCIE', 'PYNCCL'] all-reduce backends (in dispatch order) for group 'tp:0'`
  (today: `['PYNCCL']`).
- Negative: absence of `unsupported PCIe all-reduce world size 3` and of `B12X PCIe all-reduce
  initialization failed`. The DMA warning above is expected.
- Startup must still capture CUDA graphs `2/2` (`11-functional-verification.md:20`) — a ws3
  preparation miss aborts capture loudly (§3.2), which is the failure mode to watch on first boot.

Functional + numeric:

- Re-run the enforced functional suite (`11-functional-verification.md`) through the llama-swap
  proxy (medium/multimodal alias; chat + vision).
- Numeric drift check: identical fixed prompts before/after patch; expect near-identical output
  (FP32-accumulated reduction, order-only differences). Gross garbage ⇒ revert the mount.

Perf proof (the prior +3% method — alternating 1-token streaming arms at matched queue depth):

1. Arms: patch off (baseline) vs patch on; alternate A/B/A/B ≥ 20 pairs.
2. Saturation: ≥ 16 concurrent `min_tokens=600, ignore_eos=true` generations (concurrencyLimit 128
   keeps llama-swap's gate open; `max_num_seqs 2` ⇒ deep backend queue).
3. Probe: streamed `max_tokens=1` request fired mid-burst from a separate client; metric = TTFT of
   the stream, never total wall time.
4. Scrape `vllm:num_requests_running` / `vllm:num_requests_waiting` during the burst — if waiting
   ≈ 0, the burst exercised nothing.
5. Idle control round: both arms near-identical with no burst.
6. Throughput cross-check: 30 s c=8 fixed-length run per arm; compare tok/s and steps/s (engine
   iteration log). Prior rig measured +3% steps/s from exactly this swap.
7. Memory gate: `nvidia-smi` per-rank used ≈ 95.3 GiB unchanged after init.

## 6. Verdict and expected gain

**Feasible: yes — a one-line Python-constant patch delivered as a read-only bind mount next to
the existing uva.py override, plus one optional env raise; no vLLM-side change, no image rebuild,
no compiled code.**

Arithmetic (hidden 5120, bf16):

- Decode payload = rows × 10,240 B. Steady decode rows ≤ 2 (`max-num-seqs: 2`) → 20 KiB; captured
  graphs to rows 24 → 240 KiB. All inside the one-shot band (≤ 512 KiB).
- ~2 all-reduces per layer × 40 layers = 80 reduces/step (attention out + MoE out).
- Measured on this rig class at 256 KiB: ring fallback 44.9 µs vs one-shot 6.0 µs → Δ = 38.9 µs
  per reduce. Full-graph step: 80 × 38.9 µs ≈ 3.1 ms/step at the 256 KiB point. At the 20 KiB
  steady-decode point ring ≈ 13.5 µs vs one-shot ≈ 6.1 µs (TP4 table; TP3 ring is one hop
  shorter, so treat as an upper bound) → ≈ 0.6 ms/step.
- Net expectation: **+2–3% steps/s at batch ≥ 8, +1–2% at the max_num_seqs=2 steady state** — same
  sign and class as the prior rig's +3%. Pete's ~40%-of-decode-kernel-time figure (issue #410,
  Gen5 x16/x16/x8) is the pessimistic ceiling; our measured 44.9 µs ring number is operative.

Risks: ws3 never upstream-qualified (torture suite is even-only) — gate on the functional+numeric
suite before production traffic; graph capture at ws3 must flow through preparation (fails loudly
if not); PR #297 is deferred upstream with failed evidence checks — keep our patch identical to
its core so qualification transfers when it merges; do not expect the pinned image to carry this
soon.
