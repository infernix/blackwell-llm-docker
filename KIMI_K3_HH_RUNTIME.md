# Kimi K3 HH runtime image

This branch builds one CUDA 13.2/InstantTensor image from vLLM PR #238 at
`3846d740fcfe566e821c574892574f7797b85008` and SparkInfer PR #118 at
`5a46e5b5a8a87012a5b8261b81130ee057591d5d`. It contains TP16 launchers for
DCP8 and DCP16, with and without the Inferact DSpark draft. The published r5
image records Docker recipe revision
`c95180778127cbcb1c4c15fb3ffd52593bf67d42` in its OCI labels.

Published image:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r5-20260805@sha256:d5b7e0160ebdb8202237d378bfddd2343fcb4dbd3b537e026f55bec122bf2181
```

For non-speculative CC1 decode, use `dcp8-no-dspark`. It keeps the original
MXFP4 experts, BF16 dense weights, physical 1M FP8 target KV cache, and is now
faster than the equivalent DCP16 profile.

Matched full-model CC1 decode, 256 prompt tokens and 1,024 forced output
tokens, six runs per profile:

| Runtime | Runs (tok/s) | All-run median | Last-three median | Physical KV |
| --- | --- | ---: | ---: | ---: |
| DCP8 r4 | 46.635, 47.117, 47.849, 48.631, 48.364, 48.162 | 48.006 | 48.364 | 1,054,602 |
| DCP16 r4 | 47.748, 48.470, 49.110, 49.342, 49.196, 49.419 | 49.153 | 49.342 | 1,460,937 |
| DCP8 r5 source-byte candidate | 51.265, 51.289, 52.439, 52.826, 52.782, 52.939 | 52.610 | 52.826 | 1,054,602 |
| DCP16 r5 source-byte candidate | 51.798, 51.902, 52.120, 51.985, 51.932, 52.033 | 51.959 | 51.985 | 1,460,937 |

The stable DCP8 r5 result is 9.23% faster than DCP8 r4 and 1.62% faster than
the equivalently patched DCP16 r5 result. The change does not modify model
weights, KV format, reduction order, or active-split policy.

The candidates above used the exact source bytes later copied into r5. A second
validation from the immutable r5 image, with no source mounts, produced a
52.358 tok/s median over the stable six-run window (range 51.808-52.655). The
image loaded all 497,218 InstantTensor tensors in 157 seconds; complete model
loading took 185.9 seconds and 90.96 GiB/rank.

The r4 profiler showed that DCP8's smaller query gather exposed eager Python
work between the gather and dense-MLA launch. DCP8 spent 39.7 us/layer in that
gap across ranks versus 23.1 us/layer for DCP16, approximately 0.40 ms/token.
`dense_mla.bind` itself measured 28.47 us/call, or 0.683 ms across 24 full
attention layers. r5 keeps the gathered query and output addresses stable and
reuses a validated SparkInfer binding while all tensor views and the active
split count match. A changed pointer, layout, scale, or active split rebuilds
the binding.

The isolated production-geometry SparkInfer graph benchmark confirms that the
DCP8 kernels are faster. At 256 threads, batch 1, BF16 query width 576 and
output width 512, gather+reduce measured 25.631 us/layer for DCP8 versus
57.176 us/layer for DCP16. Across 24 full-attention layers that is a 0.757
ms/token DCP8 kernel advantage. A DCP8 512-thread launch measured 24.989
us/layer, only 0.015 ms/token better across the whole model, so r5 retains the
tested 256-thread default.

Validation gates: vLLM B12X MLA tests 48/48; DCP A2A tests 48/48; SparkInfer
dense MLA tests 11/11; full-model forced-token output completed for all twelve
r5 runs. InstantTensor loaded 497,218 tensors in 157-168 seconds; complete
per-rank model loading took 185.9-189 seconds and 90.96 GiB/rank for DCP8 or
91.15 GiB/rank for DCP16.

The r4 image remains the clean rollback:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r4-20260805@sha256:7e205e5c7c54cb750a480169ee346787e5b09d23351be6e90754ed9440645150
```

The previous r3 image remains the clean rollback:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r3-20260805@sha256:b9c780a20346caf05c1ad449e5ff319c432e117532cb0bef356aede20222b803
```

The rejected r2 packaging attempt
(`sha256:5b52837eac512b0500e547bd5e99940e1243678eca1c7f59e6261ba4a5a4c923`)
must not be used: its inherited `_moe_C` exposed the old seven-argument
`topk_sigmoid` ABI and failed after weight loading. The r3 build pins a
compatible immutable base, verifies the required `is_padding` argument while
building, and passed the 19 focused projection/graph-lifecycle tests from the
finished image. The previous working immutable image remains a rollback at
`voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-20260805@sha256:e029cab81df9ef35cf55bf3caed6e62acaeabe87ad72a62722d10b5e07d3e66d`.

Build:

```bash
./build-kimi-k3-hh-dcp16.sh
```

Run the selected profile against a host Hugging Face cache:

```bash
./run-kimi-k3-hh.sh dcp16-dspark
./run-kimi-k3-hh.sh dcp16-dspark-full
./run-kimi-k3-hh.sh dcp16-no-dspark
./run-kimi-k3-hh.sh dcp16-no-dspark-batch8
./run-kimi-k3-hh.sh dcp8-no-dspark
./run-kimi-k3-hh.sh dcp8-dspark
```

The default `dcp16-dspark` launcher uses the selective KDA-input MXFP8
profile. `dcp16-dspark-full`, `dcp16-no-dspark`, and `dcp8-no-dspark` retain
the target checkpoint's original MXFP4 experts and BF16 dense weights. The
historical `dcp8-dspark` profile quantizes target shared experts to MXFP8 and
is retained only for reproduction.

See `models/kimi-k3/` in the `local-inference-lab/rtx6kpro` repository for
the exact cache sizes, throughput measurements, KLD notes, and profile
selection guidance.
