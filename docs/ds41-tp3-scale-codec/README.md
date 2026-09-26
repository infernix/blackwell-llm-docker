# Scale codec — standalone prototype (binary_exceptions256)

Lossless codec for routed-expert `F8_E8M0` scale bytes of a safetensors
checkpoint. Everything operates on raw uint8 byte patterns; no floating-point
conversion is used as a byte view. CPU works with plain NumPy only; GPU code
(`scale_codec_gpu.py`) imports torch/triton lazily and is fully optional.

This directory preserves the complete committed prototype and qualification
record that preceded the production B12X/vLLM implementation. Production
source lives in sibling `b12x` and `vllm` checkouts at the revisions in
`../ds41-tp3-migration-ledger.md`. GPU gate scripts default to those sibling
checkouts and `localhost/ds41-tp3:assembled`; override `B12X_SOURCE_DIR`,
`VLLM_SOURCE_DIR`, `MODEL_PATH`, or `IMAGE` as needed.

## Format (per tensor blob, little-endian)

```
offset 0   magic "SCD1"                     4 bytes
offset 4   mode uint8: 0 = raw, 1 = binary_exceptions256
offset 5   reserved, must be zero           3 bytes
offset 8   uint64 LE original byte length N
mode 0: followed by exactly N raw source bytes
mode 1: followed by, in order:
  bits        ceil(N/8) bytes. One bit per entry, LSB-first inside each
              byte. Bit = 1 iff the source byte equals 121; bit 0
              reconstructs byte 120. Unused tail bits are zero.
  prefixes    4*(ceil(N/256)+1) bytes of LE uint32 offsets into the
              exception-record array, measured in exception RECORDS, local
              to this tensor. prefixes[0]=0, monotonic, last entry is the
              end sentinel == total record count. Groups hold up to 256
              exceptions, so counts use uint32, never uint8.
  exceptions  2*E bytes of (position_in_group uint8, original byte uint8)
              pairs in group order. Exceptional entries have primary bit 0
              and overwrite the reconstructed base byte.
```

Decoder validation order (all before any output): magic/mode/reserved,
declared length vs payload sizes, prefix monotonicity and sentinel-vs-length
consistency, exception positions within `N`, duplicate positions are invalid.
When the binary blob would not be smaller than the raw blob including the
same 16-byte header (and N>0), mode 0 raw fallback is used. Header is exactly
16 bytes per tensor.

The aggregate sidecar `<prefix>.sbuf` concatenates blobs; the JSON manifest
`<prefix>.smanifest.json` records each tensor's name, dtype, shape, source
data offsets, layout identity, uint64 blob offset/length, raw length, and
SHA-256 for source and expected decoded bytes.
Sidecars are written atomically (temp + rename) OUTSIDE the checkpoint
directory; overwrite is refused unless `--force`.

Estimated vs actual: the design estimate `ceil(N/8) + 4*(groups+tensors) +
2*exceptions + 16*tensors` = 2,420,485,316 bytes assumes the audit-average
exception count. Actual measured sizes come from real checkpoint bytes and
include every per-tensor header. Reports account for the sidecar and manifest
JSON separately and report their total serialized bytes.

## Commands

```
python3 scale_codec.py inspect                    # bounded subset (default)
python3 scale_codec.py --all inspect              # complete selected population
python3 scale_codec.py self-test [--gpu]          # synthetic CPU tests
python3 scale_codec.py roundtrip --limit-tensors 4
python3 scale_codec.py encode --limit-tensors 12 --output out/scales-subset
python3 scale_codec.py verify --output out/scales-subset
python3 scale_codec.py bench-decode --limit-tensors 2
```

Shared flags: `--model DIR` (checkpoint directory; default
`/path/to/DeepSeek-V4.1-Flash`,
not assumed to exist elsewhere), `--include REGEX` (default
`^layers\.[0-9]+\.ffn\.experts\.[0-9]+\.w[123]\.scale$`),
`--limit-tensors`, `--max-bytes`, `--seed`, `--output PREFIX`, `--all`,
`--force`. Reports distinguish `bounded-subset` from `full` mode; full mode
cross-checks selected-tensor counts/raw bytes against `audit_facts.json` and
surfaces mismatches instead of hiding them. Checkpoint files are only ever
read; sidecars/outputs go under `out/` beside the code.

`roundtrip` streams source read → encode → decode → byte-compare with no
sidecar and reports separate io/encode/decode/verify timings, SHA-256 per
tensor, mismatch counts, first mismatch index and expected/actual bytes; it
exits nonzero on any mismatch.

## Tests

```
python3 tests/test_scale_codec.py            # CPU only, no pytest needed
python3 tests/test_scale_codec.py --gpu      # adds GPU decode checks when CUDA exists
```

Coverage: empty input; lengths 1/7/8/9/255/256/257/4096; every byte value
0..255; both-common-only and all-constant inputs; no exceptions; every-entry
exceptional; clustered and tile-boundary exceptions; odd tails; seeded random
distributions; raw fallback; corruption rejection (truncated, bad magic, bad
mode, non-monotonic offsets); duplicate exception positions; out-of-range
positions; 256 exceptions in one group (uint32 prefix); realistic 368,640-byte
fixture labelled synthetic (histogram-derived, not checkpoint bytes); sidecar
file serialization incl. refuse-overwrite; palette256/nibble4 reference
codecs. GPU status without CUDA is reported as `not_run` with the exact
reason — never as passed.

GPU decoding supports both complete tensors and 256-byte selected tiles. The
selected-tile path reads mutable GPU-resident `(tensor_id, tile_id)` request
descriptors, writes repeated requests to distinct output slots, masks invalid
tensor/tile IDs to zero, and uses ordered base-expand then exception-patch
kernels. Tests mutate descriptors between CUDA graph replays. Raw-fallback
blobs use a byte-exact device copy and binary blobs use expand/patch kernels.
Standalone decode timing excludes disk I/O, CPU encoding, H2D upload,
allocation, and correctness checks; it is not native MoE throughput.
