# DeepSeek V4.1 TP3 — VRAM investigation plan

Scope: **GPUs 0–2 only**. No fourth-GPU spill, TP4, CPU/disk weight offload counted as a solution.
Goal: recover enough VRAM per rank to drop the decoder-half expert UVA offload (currently ~6.68 GiB/rank).
Practical target ≈ 6.7–7.2 GiB/rank additional (≈6.2 GiB just to fit the offloaded experts, plus 0.5–1 GiB headroom). Cold-start peaks must fit too.

Preserve unchanged: numerical weights, KV capacity, context limit, concurrency, prefill settings, Engram disk mode, LMCache config, throughput.

## Current baseline (measured)

| Quantity | Value | Source |
|---|---:|---|
| Card capacity / used / free (GPU 0/1/2) | ~95.6 GiB / ~94.5 GiB / 0.49–0.54 GiB | `nvidia-smi --query-gpu=...` |
| Model loading per rank | 88.52 GiB | worker log `model_runner.py:444` |
| Initial free memory / KV reserved | 93.53 GiB / 1.05 GiB (`kv-cache-memory-bytes 1124073472`) | worker log `gpu_worker.py:638` |
| GPU KV cache tokens | 1,131,122 | log `kv_cache_utils.py:2798` |
| CUDA graphs captured | FULL, 2 graphs; mode overridden to FULL_DECODE_ONLY | logs |
| All-reduce backend | `['PYNCCL']` for `tp:0` and `ep:0` (world size 3 unsupported by B12X PCIe) | logs `cuda_communicator.py:365` |
| b12x version | 1.3.0 exactly | dist-info |
| vLLM version | 0.1.dev21735+g2b86d3d80 (newer than patch base dev21537) | engine banner |
| Checkpoint storage | ~497 GiB total: expert payload I8-packed 259.45 GiB; expert scales F8_E8M0 16.216 GiB; engram F8_E4M3 183.11 + scales 5.72 GiB; BF16 survivors 3.68 GiB | safetensors `data_offsets` census |

Notes:
- Experts are already MXFP4 packed inside `I8` containers (`w1.weight [2304,2560]` for logical width 5120; `Mxfp4MoEMethod` in `models/deepseek_v41/nvidia/model.py`). An FP8→FP4 conversion therefore yields no automatic saving.
- Scale plane is ≈5.41 GiB/rank equal-share; payload ≈86.48 GiB/rank equal-share. Full decompression into another resident buffer defeats any compressed-storage gain.

## Phase 1 — Baseline without restart

Record image digest + overlay checksums, effective worker env from `/proc/<worker>/environ` (not `podman exec env`), GPU UUID→rank map + processes, and startup accounting (loading, KV allocation, graph capture, B12X prep, actual offloaded bytes).

Measure physical VRAM during: idle after requests; one short generation; one prefill larger than the chunk budget; concurrent requests at the current sequence limit. Reuse identical prompts/lengths/cache state for comparisons.

Separate unrelated GPU-process usage on GPUs 0–2 and report it as operational cleanup, not model-memory tuning. Do not stop anything automatically.

Deliverable: baseline table splitting startup accounting vs idle residency vs observed serving peaks. Historical log numbers alone are not a baseline.

## Phase 2 — Read-only screening

| Candidate | Measurement | Advance only if |
|---|---|---|
| Embedding/head sharing | Compare exact tensors via safetensors index: dtype, shape, raw bytes | Byte-identical AND loader/runtime can share one compatible storage allocation |
| Other exact duplicates | Group by dtype+shape, hash byte ranges, compare matches | Duplicates are actually resident, not already aliased, not needing distinct transformed layouts |
| Inactive modules | Trace which vision/draft/aux modules are constructed and used | An allocation is provably unused without dropping a supported feature |
| Workspace allocations | Trace allocation sites, consumers, sizes, config parsing | Persistent workspace exists, is unused or oversized vs consumers |
| Graph buffers | Inspect captured shapes + graph-pool accounting | Unreachable shapes or redundant storage actually allocated |
| Allocator slack | Check worker-local allocated/reserved stats | Material reserved-but-unused gap exists |

Screening rules: use `data_offsets`, never a guessed dtype-size map; start with embed/head and repeated scales before hashing bulk expert weights; equal shape/name ≠ duplicate; an env default is proof of configuration size, not allocated residency; validate option parsing before proposing values; do not mistake a separate process's `torch.cuda.memory_summary()` for the serving worker's allocator.

Deliverable: candidate list with evidence, estimated reclaimable bytes, pass/reject. Reject unsupported items immediately.

## Phase 3 — Instrumented restart (only if telemetry is insufficient)

Temporary opt-in worker diagnostic; maintenance window; restart the placement only, never llama-swap.

Capture at boundaries (before model construction, after model load, after B12X preparation, after temporary tuning release, after KV+graph capture, after warmup): PyTorch allocated/reserved, device free, unique param/buffer storage sizes, B12X prepared weight/scale/workspace sizes, CUDA-resident vs UVA host view. Group tensors by underlying storage so aliases/views are not double-counted; report opaque non-PyTorch memory separately. Diagnostics must not copy tensors to GPU, create extra CUDA contexts, retain references, or sync every step.

Deliverable: rank-by-rank allocation inventory of duplicates, unused workspace, allocator slack.

## Phase 4 — Test candidates individually, easiest first

1. Alias byte-identical compatible resident tensors (share after required transforms; ensure preparation cannot mutate one consumer independently).
2. Avoid unused persistent allocations (prove unreachable on every supported request path, not just a short text smoke).
3. Right-size overallocated workspace (derive required size from consumers/supported shapes; don't blindly halve).
4. Remove unreachable graph shapes / redundant graph storage (keep coverage for supported decode shapes).
5. Allocator configuration only if measured slack is material and compatible with graph/runtime setup.

Use temporary checksummed overlays/config with precise rollback. Keep the expert-offload budget fixed during tests to isolate each change.

Per-candidate verification: cold-start to readiness → record per-rank residency/startup → short generation, chunked prefill, concurrency at current limit → tool calling + image input for shared-storage/workspace changes → deterministic-output comparison → matched-condition TTFT/decode throughput/peak-vs-free VRAM. For exact aliases also verify shared storage and unchanged values post-preparation. Reject candidates that save memory only by forcing a slower path; within-noise differences must be repeated, not declared wins.

## Phase 5 — Combine survivors, then shrink offload

Combine only independently-passing candidates; measure combined effect, don't sum estimates.

Reduce expert offload one actual tensor-sized step at a time: change the byte budget (not merely `DS41_OFFLOAD_FROM_ORDINAL`), preserve the decoder-half boundary (first `compress_ratio == 1` entry in `kv_source_layer_ids`), record actual offloaded tensors/bytes on every rank (nominal budget can overshoot by one tensor), cold-start and re-run serving workload each step.

Drop expert offload fully only when measured combined saving covers remaining offloaded bytes **and** leaves demonstrated startup/serving margin. If cheap candidates fall short, stop — do not expand into speculative kernel work.

## Final report table

| Experiment | Verified saving/rank | Startup result | Performance result | Expert offload remaining | Decision |
|---|---:|---|---|---:|---|

Keep proven changes, remove instrumentation, `go run . --config config.yaml --validate-config` before watched reload, verify `Configuration Reloaded`, run `make test-all`, commit only proven config files.

Expected outcome: either a small defensible set of lossless savings permitting less offload, or measured evidence that cheap options cannot reach full residency. The plan does not presume a 6 GiB saving exists.
