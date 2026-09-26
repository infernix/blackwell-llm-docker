# 16 — KV pool right-sizing: is the fixed 1.05 GiB / 1,131,122-token pool over-provisioned?

2026-09-24, read-only investigation of the live placement `ds41_3gpu_karmic_r1` (engine :5807, LMCache MP :15807/:15808). No config was changed; the two small probe bursts visible in the logs are sibling-agent traffic, not consumer load.

## Verdict (TL;DR)

**Yes — the pool is over-provisioned for every workload this host actually routes to DS4.1.** Observed maximum context on this engine is 67 tokens (50 in + 17 out). The engine *can* accept 1,048,576-token requests today (auto-fit kept `max_model_len` at the model ceiling because a 1M request fits the pool), but the largest *realistic* inherited workload — Hindsight/Honcho adapter fallback traffic — is ≈203k tokens. A **0.5 GiB pool (`kv_cache_memory_bytes: 536870912`)** covers two concurrent ≈203k requests with ~30% headroom and frees **0.547 GiB (560 MiB) per rank**.

Honest framing: freed KV alone cannot fund an expert ordinal step (2.24 GiB/rank needed vs ~2.0 GiB free now; the trim adds ≤0.55). Its real value is startup/transient-allocation headroom, plus proof that KV is not the binding constraint for realistic concurrency. See §5.

## 1. Live evidence (two engine /metrics samples + llama-swap activity log)

Two `curl 127.0.0.1:5807/metrics` samples separated by intervening tool work (~1–2 min); every request-scoped counter is identical between them (only `process_cpu_seconds_total` moved 12.37 → 12.68), confirming a still-idle engine:

| metric | sample 1 | sample 2 |
|---|---|---|
| `vllm:kv_cache_usage_perc` | 0.0 | 0.0 |
| `vllm:num_requests_running` / `waiting` | 0 / 0 | 0 / 0 |
| `vllm:num_requests_waiting_by_reason{capacity,deferred}` | 0 / 0 | 0 / 0 |
| `vllm:num_preemptions_total` | 0 | 0 |
| `vllm:prompt_tokens_total` | 1026 | 1026 |
| `vllm:generation_tokens_total` | 150 | 150 |
| `vllm:prefix_cache_queries_total` / `hits` | 1026 / 0 | 1026 / 0 |
| `vllm:external_prefix_cache_queries_total` / `hits` | 1026 / 0 | 1026 / 0 |

- Engine process booted 2026-09-24T14:00:25Z (`process_start_time_seconds=1.79025842591e9`); all 17 recorded `POST /v1/chat/completions` are post-boot probes (`http_requests_total{handler="/v1/chat/completions"} 17.0`).
- Observed maxima over those 17 requests (histogram buckets, both samples): prompt **≤500 tokens** (`le=500` cumulative 17; sum 1026, mean 60.4), generation **≤100** (`le=100` cumulative 17; sum 150), request `max_tokens` param **≤1000** (`le=1000` cumulative 17; sum 2012). No request has ever exceeded the 500-token prompt bucket on this engine. 12 of 17 ended `finished_reason=length` (probe caps), 5 `stop`.
- `vllm:cache_config_info` (identical both samples): `kv_cache_memory_bytes=1124073472`, `kv_cache_size_tokens=1131122`, `kv_cache_max_concurrency=1.0787229389897224`, `num_gpu_blocks=4933`, `block_size=128`, `cache_dtype=fp8`, `sliding_window=128`, `gpu_memory_utilization=0.98`, `enable_prefix_caching=True`.
- llama-swap `/api/metrics` (:8080, window ids 45752–46568 ≈ 817 requests over 15:04:17→16:13:18): **DS4.1 appears only in two probe bursts** (16:02:35–16:02:51 and 16:12:50–16:12:51) — 11 records total, input 31–50 tokens, output 1–17, user agents `curl/8.13.0` and `Python-urllib/3.11` (sibling probes; the 16:12 burst is 6 × input 31 / output 1 at ~230–280 ms; one 16:02 record shows `duration_ms=135901`, a cold-start outlier — the engine had booted ~2 min earlier). [INFERENCE] Engine counted 17 POSTs vs 11 llama-swap-logged records; the extra 6 are likely engine-direct probes by siblings (valid API keys, bypassing the proxy).
- All real adapter traffic in the window went to `qwen38_flash_next_gpu3_sglang`: Hindsight (`hindsight/0.9.2`) and Honcho (`Python-urllib/3.11`) requests had input 11–**197,985** tokens, output ≤4,574. Largest prompts host-wide: 197,985 (hindsight→qwen3.8), 196,921 (omp/18.2.0 interactive→qwen3.8).

## 2. Consumer caps: what is the largest context any consumer can produce?

- **No token limit is declared anywhere in llama-swap config.** The adapters manage sampling params only — `config.yaml:45-141`: `hindsight` (:49, managedParams temperature/top_p/top_k/min_p/penalties/reasoning_effort/priority), `honcho-instruct` (:84), `honcho-dialectic` (:111); none touch `max_tokens`/`max_model_len`. The model entry sets only `top_p: 0.95` (`config/models/host-cuda-00-01-02-03-ds41-karmic.yaml:3-9`). The placement sets `concurrencyLimit: 128` (:53) — request concurrency, not tokens.
- Launch options: `max-model-len: -1`, `max-num-seqs: 2`, `max-num-batched-tokens: 2048` (`config/ds41-tp3-launch.yaml:16-18`).
- Therefore the only hard cap is the engine's resolved `max_model_len = 1,048,576` (`GET /v1/models` → `"max_model_len": 1048576`), auto-fit from `-1` up to the model ceiling (`max_position_embeddings: 1048576`, model config.json).
- Exposure: `deepseek-v4.1-flash` is the **second fallback candidate** for `hindsight` (`config.yaml:73`), `honcho-instruct` (:100) and `honcho-dialectic` (:127). If qwen3.8 is unavailable, adapter traffic — observed worst case ≈198k input + 4.6k output ≈ **203k tokens** — lands on DS4.1. No consumer can produce more than that today without a code change on their side; the 1M engine ceiling is capability nobody exercises (0 requests >500 tokens ever on this engine).
- The 1M capability is cheap for this checkpoint by architecture: `config.json` shows `sliding_window: 128`, `kv_source_layer_ids: [2,8,14,20]` (only 4 layers hold full KV), `index_topk: 512`, and per-layer `compress_ratios` — hence ≈1 byte/token KV and a 1M request needing only ~0.97 GiB.

## 3. Bytes/token and scenario sizing

`bytes/token = 1,124,073,472 / 1,131,122 = 0.9938` (1.0063 tokens/byte).

| scenario | tokens | KV bytes | GiB | fits 1.047 GiB pool? |
|---|---|---|---|---|
| observed max single request (50+17) | 67 | ~67 | ~0 | trivially |
| adapter worst case single (198k+4.6k) | 202,570 | 201,312,962 | 0.188 | yes (18% of pool) |
| 128k single | 131,072 | 130,252,965 | 0.121 | yes |
| 262k single | 262,144 | 260,505,931 | 0.243 | yes |
| 512k single | 524,288 | 521,011,862 | 0.485 | yes |
| 1M single | 1,048,576 | 1,042,023,724 | 0.971 | yes (93% of pool) — why auto-fit kept the 1M ceiling |
| **2 × 1M concurrent** (`max_num_seqs=2`) | 2,097,152 | 2,084,047,448 | 1.941 | **NO — second request waits** |
| 2 × 262k concurrent | 524,288 | 521,011,862 | 0.485 | yes |
| 2 × 203k concurrent (adapter worst pair) | 405,140 | 402,625,924 | 0.375 | yes (72% of pool) |

### Recommendation

**`kv_cache_memory_bytes: 536870912` (0.5 GiB / 512 MiB)** in `config/ds41-tp3-launch.yaml` → ≈540,000 tokens at 0.9938 B/token. Covers the 2 × 203k adapter-worst-case pair (0.375 GiB) with ~30% headroom, or a single 512k request. Frees `1,124,073,472 − 536,870,912 = 587,202,560 B` = **0.547 GiB (560 MiB) per rank**, all three ranks (GPUs 0-2 steady free ~2.0 → ~2.55 GiB).

Variants:
- Conservative `805306368` (0.75 GiB ≈ 810k tokens): keeps a single ~786k request possible; frees 0.297 GiB.
- Aggressive `419430400` (400 MiB ≈ 422k tokens): still fits the 2 × 203k pair (4% headroom); frees 0.656 GiB.
- Any pool below 0.971 GiB caps the single-request length below 1M — the 1M capability and any meaningful trim are mutually exclusive. With `max-model-len: -1` the same knob re-fits the API cap automatically (§4), so a 0.5 GiB pool yields `max_model_len ≈ 520k` at next start.

## 4. Scheduler behavior when tokens exceed KV capacity

- **Above `max_model_len`**: rejected at validation, never scheduled — `"This model's maximum context length is N tokens. However, you requested ..."` (`vllm/renderers/params.py:364-368`, `:507-511`).
- **Within `max_model_len` but larger than free KV at admission**: neither rejected nor preempted on arrival — it **waits**. Admission is gated by `scheduler_reserve_full_isl` (default `True`; `vllm/config/scheduler.py:180-183`: "checks whether the full input sequence length fits in the KV cache before admitting a new request ... Prevents over-admission and KV cache thrashing"), passed as `full_sequence_must_fit` into `allocate_slots` (`vllm/v1/core/sched/scheduler.py:1694`), which first checks the full sequence fits (`min(request.num_tokens, self.max_model_len)`, `vllm/v1/core/kv_cache_manager.py:630-633`). On failure the request stays queued (`scheduler.py:1725-1743` "The request cannot be scheduled" → break/requeue; surfaces as `num_requests_waiting_by_reason{reason="capacity"}`).
- **Already-running requests** that need new blocks: the `allocate_slots` loop preempts victims — priority-aware (`max(self.running, key=(priority, arrival_time))`, `scheduler.py:1133-1146`) or last-requested (`:1180`) — via `_preempt_request` (`:2304-2345`: frees blocks, re-queues, `num_preemptions += 1`). With `max_num_seqs=2` and full-ISL admission, a 1M + 1M pair means the second waits at admission rather than both thrashing; observed `num_preemptions_total=0` and `request_num_preemptions` histogram flat at 0 for all 17 requests.
- **Does `max_model_len=-1` trust `kv_cache_memory_bytes` only?** No — the two are coupled by construction. With explicit `kv_cache_memory_bytes`, the GPU worker skips memory profiling and returns those bytes as the KV budget (`vllm/v1/worker/gpu_worker.py:621-643`, log line "reserved ... for KV Cache as specified by kv_cache_memory_bytes config and skipped memory profiling"); then `_auto_fit_max_model_len` runs because `original_max_model_len == -1` (`vllm/v1/core/kv_cache_utils.py:3138-3139`) and binary-searches the largest length that fits that budget (`fits()` = `_max_memory_usage_bytes_from_groups <= available_memory`, `:2950-2960`). Shrinking the pool therefore automatically lowers the API-level length cap; `check_enough_kv_cache_memory` (`:876-930`) still guards the floor (startup ValueError if one max-len request doesn't fit — unreachable under auto-fit since the cap is derived from the same budget).

## 5. Verdict

**Recommended: `kv_cache_memory_bytes: 1124073472` → `536870912` (one line in `config/ds41-tp3-launch.yaml`). Frees 0.547 GiB (560 MiB) per rank.** Evidence basis: observed engine max context 67 tokens and 0% KV usage across both samples; 0/1026 prefix hits; largest consumer-producible realistic context ≈203k (adapter fallback worst case); no configured token caps anywhere; 2 × 203k = 0.375 GiB.

**Ordinal-step honesty (do not oversell):** the offload walk allocates ordinal residency from the GPU weight budget — `requested_memory` (0.98 × 95.01 GiB ≈ 93.11) minus KV, graphs and misc [INFERENCE on mechanism; measured figures grounded]. Freeing 0.547 GiB of KV raises that budget by 0.547 — but one ordinal step is **2.24 GiB** (ords 20/21/22 = 2.24/2.24/2.20, engine overlay walk, `04-instrument-inventory.md:29,55-57`) and even the best-case stack of all other measured trims totals ≤0.8 GiB (`05-attribution.md:25`). **KV trim alone, or KV + every other measured trim, still cannot fund one full ordinal step** (0.8 ≪ 2.24; even the smallest adjacent ordinal, 23 at 1.41 GiB per `09-vision-offload.md:8`, is out of reach). It also does nothing for host-RAM pressure (LMCache L1 48 GiB pinned pool is a separate host-side allocation, `config/ds41-tp3-launch.yaml:12-15`).

**What the freed bytes ARE good for:**
1. **Startup resilience** — removes the expandable-segments first-map OOM warning class ("map 20971520 B, free 3473408", `04-instrument-inventory.md:29`) and adds slack under the 0.98 startup free-memory gate (0.985 already fails it; comment at `config/ds41-tp3-launch.yaml:20-21`).
2. **Runtime transients** — NCCL/SymmMem ring-fallback buffers (ring fallback on ws3: 44.9 µs vs 6.0 µs one-shot), FlashInfer workspace (394 MiB default, `06-candidates.md`), and multimodal image-prefill activation spikes all draw from free VRAM; +0.55 GiB turns near-OOM retries into clean scheduling under mixed load.
3. **Concurrency proof** — with 0.5 GiB, the binding constraint for realistic concurrent workloads is `max-num-seqs: 2` (`config/ds41-tp3-launch.yaml:17`), not KV (2 × 203k fits; even 4 × 100k = 0.368 GiB fits). For the observed short requests KV was never the binder (2 × 67 tokens ≈ 134 bytes of pool).
4. **No cache- reuse loss** — durable prefix reuse lives in LMCache L1 (48 GiB host DRAM; `cache-mode: lmcache`, single cache owner, L2 disabled, `config/ds41-tp3-launch.yaml:9-15`); GPU KV is paged scratch. External prefix hits observed 0/1026, so nothing is being lost today.

**Trade-off to sign off:** the API-level length cap drops from 1,048,576 to ~520k on next start (auto-fit, §4). A client sending >~520k would newly get a validation 400 instead of a queued run. Nothing observed on this host comes within 2.5× of that boundary; if unknown future consumers matter, take the conservative 0.75 GiB variant instead.

**Rollout note:** requires a managed-container restart of `ds41_3gpu_karmic_r1` (config file is read via bind mount at container start) — production traffic, so schedule it; do not touch the running placement now. Post-restart checks: `vllm:cache_config_info` shows `kv_cache_memory_bytes=536870912`, `kv_cache_size_tokens≈540k`; `/v1/models` `max_model_len≈520k`; nvidia-smi steady free +≈560 MiB on GPUs 0-2; `num_preemptions_total` stays 0; smoke probe 200. Recipe: `verify-placement-config-change`.
