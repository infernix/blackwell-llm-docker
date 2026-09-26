# TP3 B12X PCIe all-reduce qualification

Date: 2026-09-25

## Result

DeepSeek V4.1 TP3 already selected `B12X_PCIE` first, but the inherited 96 KiB
plain all-reduce threshold routed captured 16-row and 24-row BF16 decode tensors
to PyNCCL. Set `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB` on the TP3 placement.
The fused add/RMSNorm threshold remains unchanged because it was not benchmarked.

## Evidence

Pinned image: `ghcr.io/local-inference-lab/vllm@sha256:d9d273c6890060eb5eae511a34efd724dd5c29b6dd767c55f149e0da04a7a4c3`.
Hardware: GPUs 0-2, TP3, BF16 tensors with hidden width 5120.

The isolated probe prepared the same B12X native plans used by vLLM, checked exact
output against the expected three-rank sum, captured both B12X and NCCL paths in
CUDA graphs, and timed 1,000 replays. All tested shapes passed correctness.

| Rows | Bytes | B12X one-shot | NCCL | Delta |
|---:|---:|---:|---:|---:|
| 1 | 10,240 | 7.24 us | 11.10 us | -3.87 us |
| 2 | 20,480 | 7.21 us | 12.64 us | -5.43 us |
| 4 | 40,960 | 8.85 us | 13.06 us | -4.21 us |
| 8 | 81,920 | 10.62 us | 14.30 us | -3.67 us |
| 9 | 92,160 | 11.54 us | 14.76 us | -3.22 us |
| 10 | 102,400 | 12.37 us | 15.36 us | -2.99 us |
| 16 | 163,840 | 15.45 us | 19.31 us | -3.86 us |
| 24 | 245,760 | 20.33 us | 24.04 us | -3.71 us |

Eager Python dispatch is slower and is not the model execution path. CUDA-graph
replay is the relevant comparison because vLLM captured rows 1, 2, 4, 8, 16, and
24 during startup.

## Managed live verification

The placement was restarted through llama-swap after `config OK` and a watched
`Configuration Reloaded`. The resulting container reports
`one-shot max=262144` and dispatch order `['B12X_PCIE', 'PYNCCL']` for TP group
`tp:0`. `podman inspect` confirms
`VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE=256KB` in the live container. Backend
`/health` returned 200, and both a direct completion and an authenticated
llama-swap completion succeeded. The startup also exposed incomplete expert-bank
overlay wiring; the placement now mounts the matching `api.py`, `_impl.py`, and
`dynamic.py` together rather than combining a bank-aware API with the image's
older dynamic-kernel ABI.

## Reproduction

The temporary probe was `investigations/scale_codec_handoff/residency_gate/tp3_allreduce_probe.py`.
It is intentionally not part of the committed runtime change.
