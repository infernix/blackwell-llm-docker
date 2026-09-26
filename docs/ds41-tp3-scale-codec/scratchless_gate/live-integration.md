# Live vLLM/B12X integration

The production integration is deliberately scoped to DeepSeek V4.1 Flash's
N128-repacked `w4a8_mx` path. At startup, after B12X creates prepared runtime
weights, `B12X_SCALE_CODEC=1` encodes only `w13_sfb` and `w2_sfb`. The FP4
payloads (`w13_rp`, `w2_rp`) and checkpoint files remain unchanged.

The startup encoder uses bounded 4 MiB chunks and emits one compact CUDA
`uint32` container per prepared scale plane. It replaces the canonical scale
owner only after encoding completes, allowing the raw prepared scale storage to
be released normally. Runtime decoding writes directly into existing shared
SFB stages; there is no global decoded-scale tensor.

Codec owners are constrained to the monolithic dynamic M16 implementation.
Micro, N64, pointer-bank, and external-materialized-FC1 variants reject compact
scales before launch. This is required because those variants retain raw scale
loads.

The TP3 placement passes `B12X_SCALE_CODEC=1` directly to Podman and removes the
mutually-exclusive expert-bank settings. Startup logs include one
`b12x scale codec` receipt per prepared layer and rank with dtype and encoded
bytes.

## Verified live result

Exact placement cold start on GPUs 0-2 completed with 120 codec receipts (40
layers x 3 ranks). Every receipt reported `w13_dtype=torch.uint32` and
`w2_dtype=torch.uint32`. Encoded resident scales totalled 2,419,565,892 bytes
across all ranks, or 806,521,964 bytes/rank. The raw prepared scale total is
16,986,931,200 bytes checkpoint-wide, or 5,662,310,400 bytes/rank under TP3;
this saves 4,855,788,436 bytes/rank (4.5223 GiB) in the prepared scale
representation.

The container reached 331/331 B12X preparation units, CUDA graph capture, and
HTTP health 200. A real Chat Completions request returned exactly `CODEC READY`
with `finish_reason=stop` from `DeepSeek-V4.1-Flash`. Live GPU process memory was
92,314 MiB/rank with 4,848 MiB free per physical GPU. The isolated container is
left running as `ds41_codec_live` on `127.0.0.1:5807` for external benchmarks.

