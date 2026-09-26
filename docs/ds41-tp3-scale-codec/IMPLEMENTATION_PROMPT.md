# Build a lossless expert-scale codec prototype

You are implementing a small, runnable research prototype, not writing another design proposal. Build and test the code in the working directory. Keep the current checkpoint and serving environment unchanged.

## Objective and boundaries

Determine whether exact compression of routed-expert E8M0 scale bytes can become a useful GPU-resident representation for a DeepSeek-V4.1-Flash-derived checkpoint. The deployment goal is to free VRAM for DSpark and/or more GPU-resident experts while retaining a 1,048,576-token context window.

Implement CPU encoding, on-disk sidecars, exhaustive round-trip verification, and a standalone GPU byte-decoding benchmark. Do not start by rewriting vLLM, implementing FP4 GEMM, or launching the complete model. A later experiment must measure decompression plus the real MoE kernel; decoder throughput alone does not prove faster inference.

The primary candidate is a one-bit stream plus exact exceptions. No REAP, requantization, discarded outliers, altered scale values, or changes to FP4 payloads are allowed. Engram, attention, vision, and draft/MTP tensors are out of scope initially.

## Evidence and inputs

Read `target-scales-g256.json` and the existing `checkpoint_audit.py` first. `audit_facts.json`, when present, is a compact derived summary, not a replacement for the original report or checkpoint bytes.

The audited checkpoint directory was:

    /path/to/DeepSeek-V4.1-Flash

Make this configurable with `--model`; do not assume it exists in your environment. The checkpoint itself is not included in the handoff. The JSON report contains counts, not the scale sequences; it cannot be used to reconstruct actual tensors.

Select only:

    ^layers\.[0-9]+\.ffn\.experts\.[0-9]+\.w[123]\.scale$

The supplied inventory identifies their dtype as F8_E8M0. The report covers 46,080 tensors totaling 16,986,931,200 raw bytes. Every reported tensor has 368,640 scale bytes, but read actual shapes/dtypes/offsets from checkpoint headers; do not derive matrix orientation or TP slicing from byte counts.

Observed byte counts:

    117:               450
    118:            14,256
    119:         5,122,574
    120:     9,300,877,845
    121:     7,670,665,097
    122:         9,163,405
    123:         1,040,117
    124:            46,232
    125:             1,183
    126:                39
    127:                 2

Bytes 120 and 121 account for approximately 99.9094% of entries; there are 15,388,258 other entries. The existing 256-entry palette audit models 3,102,451,392 encoded bytes, saving 13,884,479,808 bytes. That is roughly 4.3103 GiB per rank if simply normalized over TP3, NOT measured VRAM recovery.

The report excludes GPU alignment, execution-layout changes and scratch. It records only 128 encoder/decoder smoke tests, not exhaustive round-trip validation. Do not inherit its validation claim as proof of your implementation.

The serving stack discussed is a local B12X/vLLM TP3 setup. Detect actual GPU models, compute capabilities, free memory, driver/toolkit and installed package versions. Do not infer support from the GPU family. Do not upgrade the environment or fetch another checkpoint automatically.

## Project shape

Provide one command-line entry point, with small importable modules:

    scale_codec.py              CLI, report writing, benchmark orchestration
    scale_codec_core.py         CPU codecs, safe byte I/O, sidecar format
    scale_codec_gpu.py          lazy-loaded GPU decoding kernels
    tests/test_scale_codec.py   CPU and GPU tests, GPU skips explicit
    requirements-cpu.txt        minimal CPU dependencies
    README.md                  format specification and runnable commands

Use Python and NumPy for CPU functionality. Prefer PyTorch plus Triton for standalone GPU decoding, after verifying the installed environment. A small CUDA extension is acceptable if justified. GPU dependencies must be optional and lazily imported: CPU commands and tests must work without CUDA, PyTorch, Triton, or vLLM.

Avoid a framework, service, notebook-only implementation, or a large plugin system. A thin optional native-MoE adapter belongs in a later stage.

## Implement the representation

Operate on raw uint8 byte patterns, not floating-point scale values. Read safetensors byte ranges directly or use a proven bit-preserving view. A numerical conversion to uint8 is not a byte view.

Implement `binary_exceptions256` with these semantics:

1. Flatten the original scale bytes in their recorded order. Use groups of 256 entries and handle the final partial group.
2. Store one bit per entry, least-significant-bit first within each byte. Set the bit to 1 only when the source byte is 121. Bit 0 reconstructs byte 120. Set unused tail bits to zero.
3. For each source byte other than 120 or 121, store its exact local position in the 256-entry group and its original byte. Each exception is two uint8 bytes. Exceptional entries have primary bit 0 and are overwritten during reconstruction.
4. Store group prefix offsets into the exception-record array, with an end sentinel. Use little-endian uint32 offsets local to each tensor, measured in exception records, not ambiguous byte offsets. Reject overflow or use an explicitly versioned wider representation; never silently truncate. A group may have 256 exceptions, so an 8-bit exception count is insufficient.
5. Include a versioned header, original byte length, codec/mode, section lengths and enough information to validate offsets. Preserve original name, dtype, shape, layout identity and checksums in a manifest. Use uint64 offsets for an aggregate blob that may exceed 4 GiB.
6. Provide explicit raw-tensor fallback when the encoded representation is larger. Include all real headers, manifest bytes and alignment in storage reporting. Do not force the implementation to match an estimate by omitting metadata.

For orientation only, using a hypothetical 16-byte per-tensor header, this dataset would occupy:

    ceil(N/8) + 4*(groups + tensors) + 2*exceptions + 16*tensors
    = 2,420,485,316 bytes

This is an unaligned design estimate. Your actual file and GPU representations may differ; report their actual sizes and explain differences.

Make encode/decode support arbitrary byte values 0..255, not merely 117..127. Unusual or reserved E8M0 patterns are still bytes that must round-trip exactly.

Retain the existing palette256 reference as a CPU comparison where practical. An optional nibble4 comparison can store `byte - min_byte` in four bits only after validating that the entire tensor's range fits; otherwise use raw fallback. Neither comparison should delay a working primary codec.

Document the binary format precisely enough for an independent decoder. Reject truncated input, inconsistent lengths, invalid modes, invalid/non-monotonic offsets and out-of-range exception positions before launching GPU work. Define duplicate exception positions as invalid.

## CPU pipeline and correctness

Use bounded-memory reads and deterministic tensor ordering. Reuse the audit reader only after reviewing its assumptions. Do not load a whole checkpoint shard or the complete model to obtain selected scales.

Provide these commands with coherent `--help` documentation:

    inspect       reconcile audit totals and checkpoint headers when available
    self-test     execute synthetic CPU tests; --gpu explicitly enables GPU tests
    roundtrip     encode, decode and byte-compare selected checkpoint tensors
    encode        create a separate encoded sidecar and manifest
    verify        reread the written sidecar and compare against source bytes
    bench-decode  standalone GPU decoding and memory experiments

Support `--include`, `--limit-tensors`, `--max-bytes`, `--seed`, `--output` and explicit `--all` where relevant. Distinguish a bounded subset from the complete target pool in every report. Full mode must check the exact selected population; mismatches with the supplied report must be surfaced, not silently ignored.

Default to a small, deterministic subset. `--all` must stream across every selected byte. Emit progress and separate I/O time, encoding time, decoding time and verification time. A useful streaming no-persist roundtrip mode should not require writing the entire sidecar.

Write sidecars only outside the checkpoint directory. Use temporary files and atomic completion, refuse overwrites by default, and avoid one tiny filesystem file per tile. Identify the source checkpoint and source tensor content hashes in the manifest. Verify actual serialized bytes, not just an in-memory encode/decode pair.

Test byte equality directly; floating-point allclose is not a substitute. Record per-tensor mismatch counts, first mismatch locations, expected/actual bytes, and SHA-256 hashes. Fail nonzero on mismatches. The report must say exactly how many tensors/bytes were verified.

Tests must cover empty input, lengths 1/7/8/9/255/256/257 and larger, every possible byte value, both common values, all-constant inputs, no exceptions, every entry exceptional, clustered exceptions, tile-boundary exceptions, odd-length tails, random distributions, raw fallback, corruption, offset overflow and CPU file serialization. Include realistic 368,640-byte fixtures, but label histogram-derived synthetic fixtures as synthetic, not checkpoint data.

## GPU decoder

Implement a byte decoder, not floating-point dequantization. Input compressed bytes and metadata reside in GPU memory; output is the original uint8 scale bytes in preallocated output/scratch buffers.

Support sequential decoding and selected-tile decoding. A selected-tile request contains tensor ID and tile ID; GPU-resident descriptors must select data at runtime. Repeated requests are allowed and write to distinct output slots. Test changed descriptors across CUDA-graph replays. Do not copy route IDs to CPU or use `.item()` inside the timed decoding path.

Start with a clearly correct ordered implementation: expand primary bits, then apply exceptions in a second kernel on the same stream. Time both kernels together. Do not concurrently perform base writes and scatter patches to the same output locations without correct synchronization. A later fused per-tile kernel is welcome, but its race-free ownership and synchronization must be explained and tested.

Use preallocated, bounded output buffers; avoid allocation per tensor/tile in the timed path. All pointer arithmetic must safely handle large offsets and masked tails. Raw fallback must work on GPU too. Validate buffers/metadata before timing, and mask invalid dynamic requests safely rather than permitting out-of-bounds accesses.

Run GPU versus CPU byte comparisons on real selected tensors where available and on hostile synthetic cases. Add an explicit full-GPU-verification mode that checks all selected tensors in bounded batches. When CUDA is unavailable, execute CPU work and mark GPU results not_run with the precise reason; do not invent measurements or call an unexecuted test passed.

## Benchmarks that answer the actual question

Report latency distributions, working-set size, bytes read/written, output bytes per second, launch count, persistent encoded bytes, output/scratch bytes, peak allocated memory and reserved memory. Do not call output-throughput a measured DRAM bandwidth.

Prewarm compilation and kernels. Use synchronized GPU-event timing on the relevant stream; report eager and graph replay separately where supported. Keep disk reads, CPU encoding, H2D upload, allocations and correctness checks outside decoder-only timings and disclose this boundary. Also provide a separate end-to-end setup measurement. Never hide exception processing, descriptor preparation performed by the measured path, or copies inside omitted timing regions.

Compare against a byte-copy/gather baseline producing the same selected output. Explicitly state that this copy baseline is not the native MoE baseline: the native kernel may read its resident scales directly without an intermediate copy.

Test sequential and random tile access; repeated/cache-hot and rotating working sets; and synthetic expert-selection workloads with 1, 6, 12, 24, 48 and 96 distinct experts, where the loaded pool permits. Each expert request must include its w1/w2/w3 scale tensors. Label these as synthetic selections, not measured DSpark routes. Do not quietly shrink an unavailable workload. Use observed routing traces when available in the later native experiment.

Separate persistent-memory tests from correctness tests that deliberately retain both raw and encoded copies. Use isolated runs or clean baselines for memory comparisons. A compressed representation plus a full-size persistent decoded copy is not a saving. One reusable scratch pool across a larger compressed set is acceptable; account for it, double buffering and graph-private pools.

For a selected resident set, calculate:

    net_resident_saving = original resident scale allocation
                         - encoded resident allocation
                         - extra resident metadata and scratch

Record this only for the measured set. A one-third normalization is not measured TP3 savings. Do not count CPU/UVA-host storage as GPU-resident simply because it is exposed through a CUDA tensor.

## Native-kernel integration: separate follow-on gate

After the standalone prototype, inspect the user's exact local vLLM and B12X revisions. Record commit IDs and local changes; do not replace them with an assumed upstream build.

A historical source starting point is `local-inference-lab/vllm`, commit `66c29357`, particularly `Mxfp4MoEMethod` in:

    vllm/model_executor/layers/quantization/mxfp4.py

Inspect actual TP slicing, gate/up fusion, scale permutation/swizzling, padding, allocation ownership and the native scale-consumption sites. Do not copy GPT-OSS-specific logic or divide a flattened checkpoint tensor into three arbitrary pieces and call it TP3.

First compare, on the same native layer inputs, weights, routing and kernel specialization:

    A: native prepared raw scales -> native MoE operation
    B: encoded prepared scales -> bounded decode scratch -> same native MoE

Decode only the needed experts where the native interface safely permits. Include GPU-side selection/deduplication, ID mapping, scratch layout and all decompression work in B's timing. If the unchanged interface requires full-layer scratch, implement and account for that explicitly rather than pretending it accepts a compact selected-expert array. Do not run two MoE operations to assemble one result.

Compressing already-prepared scale bytes can isolate layout risks, but startup preparation peak and later repacking need separate accounting. Never assume the checkpoint histogram is identical to a padded/transformed execution layout; measure it.

Require byte-exact reconstructed prepared scales. Compare layer outputs with the baseline's repeatability; do not silently relax numerical tolerances to hide unexplained differences. Test ordinary decode and actual verification/prefill paths separately.

Only then consider decoding inside the existing compute kernel. Do not label standalone decode or a toy matmul as a fused MXFP4/B12X result. Missing native dependencies should leave this stage explicitly unverified, without blocking delivery of the standalone prototype.

Full DSpark throughput and near-1M prompt-plus-generation stability are subsequent serving tests, not claims to make from this prototype.

## Deliverables and completion criteria

Deliver runnable source, tests, format documentation, exact commands and machine-readable JSON reports. Implement the standalone work, run the tests supported by your environment, and distinguish passed, failed, skipped and not_run. Do not finish with only a plan.

Acceptance gates are: byte-exact CPU roundtrip; verification of persisted sidecars; byte-exact GPU decoding where executable; measured persistent/scratch allocation; and separately identified native-MoE regression/latency results when the real backend is available. No predetermined performance improvement or compression target is required to pass correctness.

Finish with actual encoded bytes and net savings for the measured selection, test coverage, GPU latency and memory results or explicit unavailability, limitations and the smallest concrete next integration change. Do not claim that DSpark now fits or that 1M serving is proven.

Reference documentation (verify APIs against the locally installed versions):

    https://triton-lang.org/main/python-api/generated/triton.jit.html
    https://docs.pytorch.org/docs/stable/notes/cuda.html
    https://raw.githubusercontent.com/local-inference-lab/vllm/66c29357/vllm/model_executor/layers/quantization/mxfp4.py
