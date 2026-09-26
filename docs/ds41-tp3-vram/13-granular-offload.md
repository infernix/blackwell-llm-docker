# DS4.1 TP3 — 13: expert-tensor-granular CPU offload feasibility

Question: can individual expert tensors (or a selected subset, e.g. least-routed experts) be offloaded instead of whole MoE layers?

**VERDICT: BLOCKED for per-expert granularity** — the b12x grouped-MoE kernel consumes expert weights as single contiguous stacked tensors and addresses each expert by arithmetic offset from ONE base pointer (no per-expert pointer table, expert maps rejected), so mixed host/device residency *within* one layer's expert set is not expressible. **Feasible-as-config for whole-tensor granularity**: the existing budget walk already supports partial-ordinal offload, but the quantum is one parameter (w13_weight = 1.41 GiB), not 0.47 GiB. A routing predicate could only choose *which tensors*, never *which experts*, without a kernel change.

Notation: `<SP>` = `/home/gerben/.local/share/containers/storage/overlay/254d59abfab2…/diff/opt/venv/lib/python3.12/site-packages` (installed fork code).

## 1. Parameter structure per decoder-half MoE layer per rank

**Stacked, not per-expert.** The routed experts live in `RoutedExperts` (PluggableLayer) whose quant method for this checkpoint is `Mxfp4MoEMethod` (`<SP>/vllm/models/deepseek_v4_1/quant_config.py:45-46`). `create_weights` registers **four stacked parameters** with a leading expert dim (`<SP>/vllm/model_executor/layers/quantization/mxfp4.py:589-646`):

| runtime param | shape (per rank, TP3) | dtype | bytes | GiB |
|---|---|---|---|---|
| `w13_weight` | [384, 2·768=1536, 2560] | uint8 (fp4-packed) | 1,509,949,440 | 1.4063 |
| `w13_weight_scale` | [384, 1536, 160] | uint8 (UE8M0) | 94,371,840 | 0.0879 |
| `w2_weight` | [384, 5120, 384] | uint8 | 754,974,720 | 0.7031 |
| `w2_weight_scale` | [384, 5120, 24] | uint8 | 47,185,920 | 0.0439 |
| **sum/layer/rank** | | | **2,406,481,920** | **2.2413** |

There are **no separate w1/w3 runtime tensors**: w13 fuses gate+up (`w13_num_shards = 2 if is_act_and_mul`, `<SP>/vllm/model_executor/layers/fused_moe/config.py:1397-1399`); checkpoint `w1`/`w3` names (`.experts.{i}.w{1,2,3}` + `.scale`, `<SP>/vllm/models/deepseek_v4_1/nvidia/dspark.py:75-77`) are fan-in'd by the weight loader into the w13 shards. No biases (`has_bias` false for this checkpoint; b12x rejects biases, `<SP>/vllm/model_executor/layers/fused_moe/b12x.py:497-499`).

Geometry: hidden 5120, `moe_intermediate_size` 2304, TP3 → per-rank intermediate 768, fp4 2/byte → hidden/2 = 2560, scale cols = /32 (config.json `text_config`; `<SP>/vllm/model_executor/layers/fused_moe/oracle/mxfp4.py:735-743` b12x returns sizes unrounded; alignment gates pass: 768%32=0, 5120%256=0).

**The ~26 MiB expectation is wrong.** Cross-checks:
- per expert per rank: w13 slice 1536·2560 = 3.75 MiB + w2 slice 5120·384 = 1.875 MiB + scales 368,640 B = 0.352 MiB → **5.98 MiB**;
- whole unsharded expert (checkpoint): w1 (2304,2560) + w3 (2304,2560) + w2 (5120,1152) I8 = 16.875 MiB + scales 1.055 MiB → **17.93 MiB**;
- 384 × 5.98 MiB = **2.24 GiB/layer/rank — exactly the measured walk deltas** (ords 20/21/22 = 2.24 each, investigations/04, 09; 20 decoder layers × 2.24 = 44.8 GiB, matches 02 §4).

EP is off: b12x requires `not use_ep and ep_size == 1` (`<SP>/vllm/model_executor/layers/fused_moe/b12x.py:557-565`), so every rank holds all 384 experts, intermediate-sharded by 3.

## 2. Runtime read path — how the kernel gets expert weights

Chain: `Mxfp4MoEMethod.process_weights_after_loading` → `_setup_kernel`/`_build_moe_kernel` (`mxfp4.py:787-835, 745-757`) → `B12xExperts.process_weights_after_loading` → `_prepare_experts(w1=layer.w13_weight, w2=layer.w2_weight, …)` (`b12x.py:462-524`) → `fused_moe.prepare_weights(PackedWeights(w13=…, w2=…))` (`b12x.py:363-393`) → `prepare_b12x_fp4_moe_weights` (`<SP>/b12x/moe/fused_moe/planning.py:305-361` → `_impl.py:6395+`), which for `w4a8_mx` **repacks in place** into the same storage (`_prepare_w4a8_from_e8m0_source`, `_impl.py:6108-6245`; the `B12XFP4ExpertWeights` contract demands prepared tensors reuse the canonical allocations — `_impl.py:492-514`). Per-step `apply` passes only `prepared` + activations; it deletes w1/w2 (`b12x.py:760-763`) and binds via `fused_moe.bind(... experts=prepared ...)` (`b12x.py:812-825`).

**Pointer model: single base pointer, arithmetic expert addressing.**
- Views are permutes of one stacked tensor, no copy: `w13 = w1_fp4.permute(1,2,0)` `[w1_n, k//2, E]`, `down = w2_fp4.permute(1,2,0)` (`_impl.py:5868-5870`); pointers built once via `make_ptr(..., t.data_ptr(), AddressSpace.gmem)` (`_impl.py:5910-5917`, launch `_gptr` at `_impl.py:11466-11469`).
- Inside the kernels, each expert is reached as `base + expert_idx * expert_stride`: dynamic kernel staging `dynamic.py:381-385` and `:435-437`; w4a8 phase1 `:290`; phase2 `_stage_n64_b` `:212-216` and byte variant `:260-264`. TMA loads select the expert as a tensor mode: `tBgB_w13[(None, intermediate_slice, None, task_expert_idx)]` (`<SP>/b12x/moe/_shared/kernels/dynamic.py:8533-8564`). The TMA descriptor is built once over the whole tensor (`dynamic.py:2590-2601`).
- **No per-expert pointer array exists anywhere in the launch ABI** (`b12x::tp_moe_dynamic_launch` operand list, `_impl.py:11642-11715`; `_launch_dynamic_flat` call `_impl.py:11466-11560`). Routing enters only as `topk_ids`/`row_counts`/`expert_tile_base` metadata.
- `expert_map` is explicitly rejected: `raise ValueError("b12x TP MoE does not support expert maps")` (`b12x.py:790-791`; `supports_expert_map() -> False`, `:576-577`).

**Host-backed UVA already works — for whole layers.** Layers 20-22 are fully offloaded (storage replaced by `get_accelerator_view_from_cpu_tensor(cpu_data)`, pinned host memory, device-side view — config/ds41-tp3-offload/uva.py:140-146; helper `<SP>/vllm/utils/torch_utils.py:915-935`) and run through this same kernel: functional suite passed with graphs captured (`Capturing CUDA graphs (FULL): 2/2`, investigations/11). The UVA view's `data_ptr()` is a device-addressable pointer to host pages, so every expert of those layers is read over PCIe by the same arithmetic addressing. **Mixed residency at the TENSOR level also already works**: run 09 (budget 8) left ord 23 with w13 host-backed and w2 + scales on GPU, and the engine passed smoke checks.

**Mixed residency at the EXPERT level does not work**: host and device pointers live in different UVA VA segments, so a stacked `[E,…]` tensor cannot have some experts' bytes on host and others on device while remaining one contiguous storage. The kernel has no indirection to fix up per-expert pointers, and `cuMemMap`-style VMM cannot alias host pages into a device VA range. The only theoretical in-place path — one `cudaMallocManaged` allocation with per-range `cudaMemAdvise(preferred-location=CPU)` on cold-expert byte ranges — keeps the single base pointer but is a kernel/platform change (TMA + managed-memory support is not established, and the w4a8 in-place repack at load would churn every page). Not a small overlay.

## 3. Overlay mechanics (config/ds41-tp3-offload/uva.py)

The walk (`wrap_modules`, uva.py:55-93) enumerates decoder layers via `make_layers` (`<SP>/vllm/model_executor/models/utils.py:874-884`), gates ordinals `min<=ordinal<=max` (uva.py:68-79), and `_maybe_offload_to_cpu` walks `module.named_parameters()` with a **pre-parameter budget check**: `if self.cpu_offload_bytes >= self.cpu_offload_max_bytes: break` (uva.py:111-115), accumulating per parameter (uva.py:145). Consequences, verified against measured runs:

1. **Partial-ordinal offload is already expressible — but the quantum is one parameter.** Parameter order per layer is registration order: `w13_weight` (1.406 GiB) first (`mxfp4.py:589-598`). The check is a break-before, not a fits-check, so a parameter larger than the remaining budget is *still offloaded* (overshoot by one param, then break).
2. **The "8.1 → ~0.47 GiB of ord-23 experts" prediction is falsified by the code and by measurements.** After towers (0.904) + ords 20-22 (6.72) = 7.624 GiB, ANY budget in (7.624, 9.03] offloads exactly ord-23 `w13_weight` → total 9.03 GiB. Measured: budget 8 → 9.03 (09), budget 7.7 → 9.03 (10). The remaining-budget arithmetic (8 − 7.624 = 0.376) never caps the offload because w13 (1.406) is a single parameter. Note run 10's "ord 23 expert params are 3-6 MiB each" inference is wrong — the bleed is the single 1.406 GiB w13 tensor; per-expert params would have bled only ~0.38 GiB.
3. **Achievable budget points:** ≤7.624 → ords 20-22 only (7.63, today's vis4 state); (7.624, 9.03] → +ord-23 w13 (9.03, free +1.41 GiB, w2 of ord 23 stays on GPU); (9.03, 9.12] → +its scales; then ord-24 w13 → 10.44. No w2-only or sub-1.4-GiB increment is reachable via budget alone (w13 precedes w2 in registration order).
4. **What a routing-aware predicate would need:** a per-parameter hook in the walk (e.g. `DS41_OFFLOAD_EXPERT_PREDICATE` consulted in the uva.py:111 loop, replacing the budget as the gate) that decides per *tensor* (w13 vs w2 vs none, per layer) — expert-index filtering cannot be honored downstream (§2). Cold-expert selection as a *choice of tensors* would only make sense if layers differed in skew; all decoder layers here are architecturally identical and every rank holds all 384 experts, so a per-layer predicate degenerates to the existing ordinal gate.
5. **RoutedExpertsCapturer can measure routing, but cannot drive per-expert offload.** It exists and is wired: `--enable-return-routed-experts` → `init_routed_experts_capturer` (`<SP>/vllm/v1/worker/gpu_worker.py:894-895`, `gpu/model_runner.py:384-391`), `RoutedExpertsCapturer.device_buffer [max_tokens, 40, 6] int32` (`routed_experts_capturer.py:96-102`), capture at `base_router.py:296-297` / DSv4 router `models/deepseek_v4/nvidia/model.py:703-704`, scheduler-side `RoutedExpertsManager` slot buffer (`routed_experts_capturer.py:315+`). It exports per-token/per-slot routing → a per-expert-per-layer histogram is a trivial post-process. Flag is off by default (`ModelConfig.enable_return_routed_experts = False`, `<SP>/vllm/config/model.py:251`); cheaper alternative for a histogram: a ~20-line `capture_fn` counting `topk_ids` into a `[E]` device histogram (no scheduler dependency). **Prerequisite only for the (blocked) per-expert design; not useful for tensor-granular offload.**

## 4. Risks

| risk | finding |
|---|---|
| b12x plan cache keyed on data_ptr | MoE plan key is `(counts, activation, route_on_input, id(prepared))` (`b12x.py:640`); weight-view cache keys on the six `data_ptr()`s (`_impl.py:5850-5854`, `_WEIGHT_CACHE` :2135). UVA view `data_ptr` is stable for the process lifetime (host allocation pinned once). On restart everything is rebuilt fresh (preparation → plan → views), so invalidation is a non-issue. Attention-side paged plans key on KV-pool `data_ptr` (`<SP>/b12x/attention/paged/_preparation.py:255`) — same process-local reasoning. |
| CUDA graph capture with mixed residency | Already exercised: preparation must complete before capture (`b12x.py:332-334` rejects capture-time preparation); graphs captured 2/2 with ords 20-22 host-backed and with run-09's mixed ord 23. Capture binds fixed pointers; residency does not change the launch ABI. |
| Per-step pointer-array construction | None — no pointer arrays exist; base pointers are fixed at preparation. The only per-step cost is the PCIe reads themselves (paid today for ords 20-22). |
| Grouped-GEMM scheduling degradation | Residency is invisible to scheduling: per-step `row_counts`/`expert_tile_base` are built from `topk_ids` (dynamic.py:3476-3518) regardless of where bytes live. Cold tiles simply take longer (cp.async/TMA over PCIe), and the resident-grid barrier waits for all experts' tasks — one cold expert stalls the layer tail. Fully-offloaded layers already pay the worst case; tensor-level partial offload (w13 host, w2 device) splits the stall across FC1 only. |
| w4a8 in-place repack on host memory | One-time load cost: `_prepare_w4a8_from_e8m0_source` repack kernels write through the UVA view over PCIe (`_impl.py:6108-6245`). Already paid for ords 20-22; no new mechanism. |

## 5. Verdict + VRAM-vs-PCIe tradeoff math

**Verdict: BLOCKED (kernel assumption: single contiguous stacked tensor, arithmetic expert addressing, no per-expert pointer indirection, expert maps rejected).** Feasible today as-config: whole-tensor partial-ordinal offload in 1.406 GiB (w13) quanta via the budget walk; a small overlay could add a per-tensor predicate or w2-only increments (0.747 GiB quantum), but that is tensor-granular, not expert-granular.

Per-rank constants: 2.2413 GiB expert bytes per decoder layer; 5.98 MiB per expert per rank; decode step at max-num-seqs 2 → 2×6 = 12 routed rows/layer (≈12 distinct experts touched); prefill chunk 2048 tokens → 12,288 routed rows.

**(a) Uniform random partial offload** (hypothetical per-expert granularity, fraction f of each layer's experts on host):
- VRAM freed: f × 2.2413 GiB/layer (linear, no surprise).
- PCIe per decode step/layer: expected cold routed experts = 12f → 12f × 5.98 MiB ≈ 71.7·f MiB. Probability a step touches zero cold experts = (1−f)^12 (bursty, not steady). Per GiB saved, PCIe cost is f-proportional — i.e. partial per-expert offload is *strictly better* than whole-layer offload per GiB (f=1 costs 71.7 MiB/step/layer for the same 2.24 GiB that f=0.25 delivers for ~17.9 MiB/step). This is exactly the win the kernel design forbids.

**(b) Cold-expert offload assuming routing skew** (offload the coldest fraction f):
- Same VRAM freed: f × 2.2413 GiB/layer.
- PCIe = (share of routed rows hitting cold experts) × full-row cost. With typical MoE skew the coldest quartile serves well under its uniform 25% share; PCIe add scales with the *cold mass* m·f (m = fraction of routes on cold experts, m ≪ 1 for well-trained routers) instead of f. E.g. f=0.25 with m=0.08 → ~5.7 MiB/step/layer vs 17.9 uniform — ~3× cheaper per GiB.
- **Measurement needed first:** per-expert per-layer selection histogram over representative traffic. Sources: `--enable-return-routed-experts` + `RoutedExpertsManager` export (routed_experts_capturer.py:315-432), or a capture_fn histogram hook (model.py:703-704). Skew without kernel support is, however, only a number — it cannot be acted on per expert in this build.

**Practical recommendation:** keep vis4 (ords 20-22 + towers, 7.63 GiB, ~2.1 GiB free/rank). If more VRAM is ever needed and +1.41 GiB PCIe reads on one layer are acceptable, the config-only lever is raising the budget past 7.624 (land on 9.03 GiB total, run-09 state); a w2-only or per-tensor overlay is the next rung, and per-expert offload would require b12x kernel work (per-expert pointer tables or managed-memory residency) that is out of proportion to the ~2 GiB/rank at stake."
}

[You have received this identical output 3 times. Re-reading 'agent://GranularOffload' will not change it — use a narrower selector (path:A-B), or proceed with the edit.]