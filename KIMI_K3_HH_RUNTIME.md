# Kimi K3 HH runtime image

This branch builds one CUDA 13.2/InstantTensor image from vLLM PR #238 and
SparkInfer PR #118. It contains full-checkpoint TP16 launchers for DCP8 and
DCP16, with and without the Inferact DSpark draft.

Build:

```bash
./build-kimi-k3-hh-dcp16.sh
```

Run the selected profile against a host Hugging Face cache:

```bash
./run-kimi-k3-hh.sh dcp16-dspark
./run-kimi-k3-hh.sh dcp16-no-dspark
./run-kimi-k3-hh.sh dcp16-no-dspark-batch8
./run-kimi-k3-hh.sh dcp8-no-dspark
```

The default `dcp16-dspark` launcher uses the selective KDA-input MXFP8
profile. `dcp16-dspark-full`, `dcp16-no-dspark`, and `dcp8-no-dspark` retain
the target checkpoint's original MXFP4 experts and BF16 dense weights.

See `models/kimi-k3/` in the `local-inference-lab/rtx6kpro` repository for
the exact cache sizes, throughput measurements, KLD notes, and profile
selection guidance.
