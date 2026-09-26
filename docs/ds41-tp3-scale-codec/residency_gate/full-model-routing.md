# Full-model routed-expert instrumentation

The B12X overlay can count logical routed expert IDs for every transformer layer without enabling vLLM's routed-expert response path, which is incompatible with the active LMCache connector.

## Mechanism

Each layer owns a CUDA `int64[384]` counter and a device enable scalar. The counter kernel is included in CUDA graph capture while disabled; after `/cache/deferred-banks.arm` appears, counters are enabled and serving graph replays increment them. Preparation routes remain excluded. A daemon snapshots counters every configured interval to `/cache/all-expert-routes-rank{rank}-layer{layer}.json`. Deferred layers 20–22 continue to write residency-aware snapshots separately.

Focused validation proved exact duplicate ID counts for eager and graph replay, zero steady replay allocation delta, and about 4.1 microseconds for two counter kernels on the test shape. Logical counter storage is about 240 KiB across 40 layers, excluding allocator rounding and stream/event resources.

## Preliminary full-model trace

A quick representative run produced rank-identical histograms for all 40 layers. The globally coldest 231 `(layer, expert)` instances had zero observed routes and were distributed across layers 0–10. The current policy instead keeps those resident while leaving 231 layer-22 experts host-backed.

This result is preliminary. Continue representative workload collection before changing startup storage. The final policy should rank `(layer, expert)` pairs globally, preserve a broad-workload minimum observation window, and apply hysteresis before evicting an expert that becomes active.

## Prior art

- vLLM EPLB state: https://github.com/vllm-project/vllm/blob/main/vllm/distributed/eplb/eplb_state.py
- vLLM LFRU expert cache PR: https://github.com/vllm-project/vllm/pull/37190
- vLLM hot/cold UVA residency RFC: https://github.com/vllm-project/vllm/issues/57794
- SGLang expert distribution: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/eplb/expert_distribution.py
- Graph-safe hot cache: https://github.com/ap03906101/moe-hotcache

These support a two-timescale policy: graph-safe counters in the inference path, then a slow control loop with decayed counts, hysteresis, retained RAM backing, copy-before-publication, and slot quarantine.
