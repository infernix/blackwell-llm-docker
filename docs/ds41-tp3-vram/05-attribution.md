# DS4.1 TP3 — per-rank attribution (05)

Inputs: 03-runtime-screening.md (fixed-buffer inventory), 04-instrument-inventory.md (live measurement of ds41_3gpu_karmic_r1, 2026-09-24).

Per-rank steady-state accounting (identical on ranks 0-2):

| layer | source | GiB |
|---|---|---:|
| Model load (weights + runtime objects incl. UVA views) | engine log `Model loading took` | 88.52 |
| Explicit KV pool | log `Initial free memory ... reserved` / kv_cache_memory_bytes 1124073472 | 1.05 |
| CUDA graph capture | log `CUDA graph capture finished, took` | ~0.02 |
| Claimed subtotal | sum | ~89.59 |
| nvidia-smi per-process | compute-apps row | 93.81 (96060 MiB) |
| Allocator/context/NCCL gap | difference | ~4.2 |
| Driver overhead (per-GPU minus per-process) | 96202-96060 MiB | ~0.14 |

Steady free per GPU: ~1.06 GiB (1090 MiB). Card per-process ceiling reached; nothing else on GPUs 0-2.

Fixed non-weight buffers inside the claim (from 03): two FlashInfer workspaces ~394 MiB each (~0.77 GiB), candidate_block 16 MiB, topk_indices 4 MiB, routed-experts buffer ~1.88 MiB, sampler bins ~0.99 MiB, DMA capacity ~60 MiB when enabled, PCIe exchange ~96 KiB → ~0.9 GiB total. Token-proportional buffers are tens of MiB at max_num_batched_tokens=2048.

Duplicates: embed.weight vs head.weight NOT byte-identical (sha256 differ, checkpoint 02); tie_word_embeddings=false; BF16 duplicate scan 520/520 distinct; expert sample-hash 80/80 distinct → no dedup win.

Slack interpretation: the ~4.2 GiB per-rank gap between nvidia per-process and engine-claimed totals is CUDA context + NCCL 2.31.2 buffers + allocator fragmentation + JIT/graph pools. Engine log shows expandable_segments first-map warnings during offload walk, so the allocator is already in expandable mode; residual slack is mostly context + comm buffers, not free chunks. Reclaiming it requires trimming comm/JIT footprint, not allocator flags.

Reconciliation with goal: to host the currently-offloaded experts (6.68 GiB/rank claimed; measured offload hits ords 20/21/22 = 2.24+2.24+2.20 = 6.68) fully on GPU, need >=6.68 GiB while steady free is ~1.06 GiB. Best-case stack of proven items: FI workspace pair trim (~0.5 GiB realistic given sparse-MLA floor), graph/misc headroom (~0.1), comm-buffer/DMA (~0.1), embed/head keep-until-measured — total ≤ ~0.8 Gi. Does not reach target; partial un-offload (ords >=23 ≈ +0.3 GiB host->GPU shift) remains the realistic landing zone.
