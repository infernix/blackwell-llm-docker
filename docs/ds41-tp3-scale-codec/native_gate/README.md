# Prepared-scale native gate

## Outcome

Native A/B passes on the exact pinned image and local B12X overlay after fixing
the non-bank custom-op launch to pass `banks=None`. The custom operation has no
bank operand; bank-enabled launches already bypass it and call
`_launch_dynamic_flat` directly with their four pointer tables.

The gate uses real layer-0 checkpoint experts with hidden size 5120 and
intermediate size 2304. The 96-expert run exercises the full checkpoint expert
count, but weights are not TP-sliced. Synthetic BF16 activations and routes use
the same prepared weights, dynamic native operation, inputs, and kernel
configuration for A and B:

- A: canonical prepared scales -> native `b12x_moe_fp4`.
- B: encoded prepared scales -> preallocated decoded scratch -> the same native
  `b12x_moe_fp4`.

Prepared `w13_sfb` and `w2_sfb` bytes are exact. Across 32 interleaved samples
per shape, A/B maximum error stayed within the combined A/A and B/B native
repeatability envelope. Dynamic reduction is not bitwise deterministic.

## Reproduction

From the repository root:

```sh
bash docs/ds41-tp3-scale-codec/native_gate/run.sh
bash docs/ds41-tp3-scale-codec/native_gate/run.sh \
  --experts 96 --report /work/native_gate/native-96-report.json
```

Only GPU0 is exposed by CDI. The image is pinned by digest in `run.sh`; the
runner mounts the exact local B12X `_impl.py`, `api.py`, and `dynamic.py`
overlays read-only. No complete model, service, live configuration, or
checkpoint write is involved.

## Full 96-expert result

| Metric | Result |
|---|---:|
| Prepared scale bytes | 106,168,320 |
| Encoded file bytes | 15,094,172 |
| Persistent GPU decoder bytes | 14,009,994 |
| Full-layer decode scratch | 106,168,320 |
| CUDA-graph decode | 0.05504 ms |
| CUDA-graph raw D2D copy | 0.11354 ms |

| Tokens | Native A | Decode + native B | Added latency | A/B max | Repeatability envelope |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.08044 ms | 0.16819 ms | 0.08775 ms | 0.001343 | 0.001343 |
| 16 | 0.70549 ms | 0.78803 ms | 0.08254 ms | 0.001282 | 0.001709 |
| 128 | 0.95174 ms | 1.03000 ms | 0.07826 ms | 0.001465 | 0.001465 |

Decoder graph replay and every native graph capture allocated zero additional
live bytes after their buffers were prepared. Encoding, validation, H2D,
exception-position expansion, and descriptor preparation occur before timing.

## Decision

The codec is correct and the prepared representation is ~13.2% of raw scale
bytes, but full-layer scratch equals the original scale allocation. Replacing
resident raw scales with compressed bytes plus full-layer scratch would consume
about 120.2 MB versus 106.2 MB raw and save no VRAM. It also adds roughly
0.08-0.09 ms to these native operations.

The next viable experiment is selected-expert decode into compact scratch with
a native kernel interface that accepts compact expert remapping. Until that
interface exists, do not integrate this full-layer path into serving and do not
claim capacity or throughput improvement.

