# Kimi K3 HH runtime image

This branch builds one CUDA 13.2/InstantTensor image from vLLM PR #238 at
`91a81414c72d0633e5d7292702c1f4611d5b7e4d` and SparkInfer PR #118 at
`34bb490b28fd0742006a611c83c6b9883ed3d453`. It contains TP16 launchers for
DCP8 and DCP16, with and without the Inferact DSpark draft. The published r3
image records Docker recipe revision
`af5e90ef5cbc12e88dba67230d6cc3a85c6f31f9` in its OCI labels.

Published image:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-r3-20260805@sha256:b9c780a20346caf05c1ad449e5ff319c432e117532cb0bef356aede20222b803
```

The r3 image enables the same lossless TP16 projection transport for DCP8
that DCP16 already used. It was tested from the published image without source
bind mounts. The clean-image DCP8 CC1 decode results were 45.725, 45.234, and
45.438 tok/s (median 45.438 tok/s) with 1,054,602 physical FP8 KV tokens. The
InstantTensor pass loaded 497,218 tensors in 162 seconds; complete per-rank
model loading took about 193.5 seconds and 90.96 GiB/rank.

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
