# DS4.1 TP3 — direct no-offload startup test (08)

Run 2026-09-24 13:25 UTC. Same placement cmd as the managed config minus
'--cpu-offload-gb 6 --cpu-offload-params experts' (PLACEMENT_ID=ds41_nooff, PORT=5807).

## Result: full residency does NOT fit.

All three TP workers failed identically during startup:

- torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.41 GiB.
- GPU total capacity 95.01 GiB, 1.12 GiB free at failure;
  process at 93.88 GiB (PyTorch allocated 92.37 GiB, 38.91 MiB reserved-unallocated).

Cross-check against instrumented offload run (04): model load there was 88.52 GiB/rank
WITH experts off. Adding back the 6.68 GiB of expert tables -> ~95.2 GiB weights,
exceeds the 95.01 GiB card capacity minus ~1.4 GiB of unavoidable runtime
(KV 1.05 + graphs 0.02 + context/NCCL) — matches the observed 1.41 GiB failing allocation.

Health 200 seen in log window was the LMCache MP server on its own port; the engine
workers died before serving. Container removed itself (--rm); GPUs 0-2 back to idle.

Final answer: keep DS41_OFFLOAD_FROM_ORDINAL=20 (6.68 GiB off). No candidate env change
(06) produced reclaim, so there is no headroom to shrink the offload further.
