# Scratchless prepared-scale native gate

**Decision: functional isolated prototype passes; do not enable in serving.**
96 real checkpoint experts, layer 0, hidden 5120 / intermediate 2304 (not TP-sliced), GPU0 RTX PRO 6000 Blackwell SM120. Exact pinned image is in `run.sh` and reports. No model/service/config changes.

## Results

`parent-report.json`, 200 CUDA-graph replays after warmup:

| Tokens | Raw prepared native ms | Compressed native ms | A/B max abs | Raw repeatability range |
|---:|---:|---:|---:|---:|
| 1 | 0.083469 | 0.120701 | 0.00048828125 | 0.00048828125 |
| 16 | 0.715627 | 0.717660 | 0.00244140625 | 0.0029296875 |
| 128 | 0.952898 | 0.940532 | 0.0029296875 | 0.00390625 |

64 interleaved samples per implementation. Tolerance is the maximum
elementwise range across raw A samples only. Native B directly reads compressed
planes; no decode launch or decoded global scratch. Both graphs together add
**zero allocated bytes** in every case; reserved deltas include allocator
releases and are not codec savings.

Raw scales: **106,168,320 B**. Compact resident planes: **15,094,152 B**, saving **91,074,168 B (85.78%)**. A remains resident for comparison; this is per-representation accounting, not process peak savings. Probe output is bounded to 256 tiles (at most 256 KiB), released before native execution.

Every prepared 512B half and 1KiB tile across all experts passed byte-exact compiled CuTe staging: FC1 138,240 halves / 69,120 tiles; FC2 69,120 halves / 34,560 tiles. This calls the same helper as the native kernel, not a Triton decoder.

## ABI and ownership

One contiguous CUDA Uint32 tensor per plane opts in; legacy Int32 scales stay raw. Word 0 is group count G; next 8G words are little-endian bits (0=120, 1=121); next G+1 words are uint32 exception prefixes; remaining bytes are uint8 `(position,value)` records, padded only to a four-byte boundary. Groups are 256 raw bytes. Expert-major offsets are unchanged. No expanded persistent int64 positions. Four-byte output ownership avoids shared write races; existing ready/done barriers retain stage ownership. Bank, N64, other quant modes and split-materialized paths are rejected for compact scales.

The existing fixed activation-wrapper ABI is retained: the backend specialization is set before compilation and included in the compile cache key. No extra kernel pointer arguments or per-launch codec allocation. `banks=None` parent fix is retained.

**Prerequisite discovered:** the local `b12x_moe_fp4` call computed but did not forward `dynamic_w4a8_prepared`. Added that missing argument so both A and B actually use prepared scales. Before this fix, memcheck showed native B OOB raw reads at kernel offsets `0x12630` / `0x132b0`; merely replacing scale storage was not a valid prepared-native test. This also means older native-gate latency numbers are not the corrected prepared-path baseline.

## Commands

```sh
bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --repeats 100
SANITIZE=1 bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --report /work/scratchless_gate/memcheck-report.json
bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --probe-only --report /work/scratchless_gate/probe-report.json
```

First command passed. Second completed all staging/native parity cases (`memcheck-report.json`), with **no reported invalid memory accesses**, but exited 86 with four CuTe-runtime CUDA API fabric-handle/IMEX errors. This is **not a clean sanitizer pass**. `sanitizer-summary.json` records that qualification limitation. Third command is the focused-only invocation (full runs already execute that probe first).

## Risk, rollback, next measurement

Single-token latency regresses about 44%; 16-token latency is approximately neutral; the ~1% 128-token improvement is not a robust throughput claim. Keep opt-in and repeat interleaved timing/soaks before deployment. Next measure the same graphs over multiple independent runs and exception-density distributions, then TP-sliced geometry separately; expected memory delta is the exact representation saving above, latency benefit is not assumed. Kill switch/rollback: retain ordinary Int32 prepared planes (default); no service configuration was changed. Sanitizer environment must be qualified before treating this as production-safe.

## vLLM preparation integration follow-up

The codec heuristic now selects dynamic M16 for every N128 W4A8 token
capacity. For this geometry, 1–4 tokens use direct routing; larger verification
and prefill counts use grouped routing. The ordinary 1024-token heuristic
selected M64, whose external materialized FC1 is raw-only; a focused run
confirmed the existing compact-scale guard rejects it. Explicit/cached micro
bindings now fail before launch rather than treating Uint32 containers as raw
scale planes. Raw Int32 micro bindings are unchanged.

The gate uses the bounded production encoder and checks it against the CPU
reference before byte-exact compiled staging. `--route-pattern 0` selects the
same route corpus and softmax weights as vLLM's preparation factory;
`--route-mode direct` and `--tile-m` select exact launch variants.
`--deterministic` opts into B12X's existing deterministic output path for
bitwise A/B qualification (direct routing does not support that mode).

`preparation-grouped-deterministic.json`: 96 experts, tokens 1/16/128/1024,
all native A/B outputs bitwise equal. `preparation-direct-serving.json`:
tokens 1/2/3/4, all A/B errors within measured raw repeatability. Both retain
exact compiled staging and save 91,074,168 bytes. The first nondeterministic
grouped corpus run exceeded its measured repeatability bound; it is retained
as `preparation-grouped.json`, not relabeled as a pass.

The existing isolated container reached 331/331 weight units, CUDA graph
capture, HTTP health, and returned `codec ready` from a real chat request.
Its Python codec logger emitted no receipts, so this does not prove compressed
residency. The vLLM overlay now logs prepared plane dtypes and encoded bytes
through its configured logger. A parent-owned cold boot is still required to
collect those receipts and verify the final M16 policy on the full model.
The isolated container was stopped after the successful request to free GPU0
for the focused gates; no managed service was restarted.

Measure final startup residency receipts, per-rank allocated/reserved VRAM,
and 1024-token prefill latency against raw M64: forcing M16 can regress
prefill performance. Rollback remains `B12X_SCALE_CODEC=0` on the next boot.

Focused sanitizer invocation (tokens 1/16/1024, deterministic M16) completed
all staging and parity checks in `preparation-memcheck.json`. Compute-sanitizer
reported no invalid memory access, but exited 86 with four CuTe runtime IMEX
fabric-handle CUDA API errors, matching the earlier environment limitation.
This is not a clean sanitizer pass.

Reproduction commands (the corpus runs above used pattern 0):

```sh
bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --tokens 1 16 128 1024 --route-pattern 0 --deterministic --report /work/scratchless_gate/preparation-grouped-deterministic.json
bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --tokens 1 2 3 4 --route-pattern 0 --route-mode direct --report /work/scratchless_gate/preparation-direct-serving.json
SANITIZE=1 bash docs/ds41-tp3-scale-codec/scratchless_gate/run.sh --experts 96 --tokens 1 16 1024 --route-pattern 0 --deterministic --report /work/scratchless_gate/preparation-memcheck.json
```
