# DS4.1 TP3 — vision+aligner offload candidate run (09)

Change vs control: --cpu-offload-gb 6 -> 8; --cpu-offload-params 'experts vision aligner'
(config/models/host-cuda-00-01-02-03-ds41-karmic.yaml line 69, validate-config OK before run).

## Offload walk (identical on all 3 ranks)
- tower walks first: vision 0.77 GiB + aligner 0.14 GiB (walk ordinals logged as 0, prefix "vision"/"aligner")
- layer walk: ords 20,21,22 = 2.24+2.24+2.24 GiB AND NEW ordinal 23 = 1.41 GiB (budget 8 now covers 4 ords)
- Total CPU offloaded parameters: 9.03 GiB per rank

## Measured deltas vs control (04/06)

| metric | control | vision-offload run | delta |
|---|---|---|---|
| Model loading took | 88.52 GiB | 86.17 GiB | -2.35 GiB |
| Total offloaded | 6.68 GiB | 9.03 GiB | +2.35 GiB |
| steady used (GPUs 0-2) | 96190/96190/96190 MiB | 93796/93796/93776 MiB | -2394 MiB per GPU |
| steady free | ~1102 MiB | ~3496 MiB | +~2394 MiB |
| GPU KV tokens | 1,131,122 (unchanged) | 1,131,122 | 0 |

Offload freed ~2.35 GiB/rank = vision 0.767 + aligner 0.137 + expert ord 23 remainder ~1.41 + rounding.

## Correctness
- /health 200 in ~150 s (same ballpark as control)
- smoke chat x2: HTTP 200, finish_reason=length (8-token cap hit during reasoning, same as control run 04), latency ~0.7 s class
- llama-swap proxy untouched: /health 200 on :8080 throughout
- container stopped after run; GPUs 0-2 back to 0 MiB

## Interpretation
Offloading vision+aligner is WORKABLE and frees 2.35 GiB/rank without touching text-path weights.
Note: run also shifted expert offload from 3 ords (6.68) to 4 ords (8.09 GiB of experts) because
the raised budget let the walk continue to ord 23 — a throughput-relevant side effect, not vision-related.
To isolate vision-only savings: keep --cpu-offload-gb 8 but the extra 1.41 GiB of experts back on GPU
would require capping the walk (DS41_OFFLOAD_FROM_ORDINAL or budget). Follow-up run measured separately (10).

## Correction (post-run code check)
The ~60 MiB DMA capacity buffers (03 item: VLLM_PCIE_DMA_MIN_BYTES) are NEVER allocated
in this topology: b12x_pcie_all_reduce.py:295-302 returns before _initialize_dma (:309)
when the PCIe one-shot init fails on world size 3. DMA trim is a phantom lever here —
removing item 9 from the untested ledger.
