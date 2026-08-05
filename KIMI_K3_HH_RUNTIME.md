# Kimi K3 HH runtime image

This branch builds one CUDA 13.2/InstantTensor image from vLLM PR #238 at
`b7e203d0bb6e1456b858277a388726f93f0d1ff6` and SparkInfer PR #118 at
`34bb490b28fd0742006a611c83c6b9883ed3d453`. It contains TP16 launchers for
DCP8 and DCP16, with and without the Inferact DSpark draft.

Published image:

```text
voipmonitor/vllm:kimi-k3-hh-runtime-pr238-pr118-20260805@sha256:2407e74fba03d5074acc3d2a7b1a1340bb21905fcacc560186bb3322c4e8f125
```

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
