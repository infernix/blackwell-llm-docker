# DS4.1 Flash TP3 — instrumented startup inventory (placement ds41_3gpu_karmic_r1)

Run date 2026-09-24. Container recreated manually from the exact placement `cmd:` in
`config/models/host-cuda-00-01-02-03-ds41-karmic.yaml` (${PLACEMENT_ID} -> ds41_3gpu_karmic_r1,
--port 5807, all bind mounts/env identical: PROFILE=ds41-flash HARDWARE_PROFILE=rtx-pro-6000-pcie
TP=3 CUDA_DEVICE_ORDER=PCI_BUS_ID DS41_OFFLOAD_FROM_ORDINAL=20, extra args --kv-cache-memory-bytes
1124073472 --cpu-offload-gb 6 --cpu-offload-params experts). llama-swap proxy untouched.
Engine-log timestamps are HH:MM:SS from container logs; host-clock stamps noted separately.

## (a) t0 / idle baseline (before engine start; host clock 14:46:04)

| GPU | used MiB | free MiB |
|---|---|---|
| 0 | 0 | 97291 |
| 1 | 0 | 97291 |
| 2 | 0 | 97291 |
| 3 (companions, untouched) | 96591 | 701 |

llama-swap http://127.0.0.1:8080/health -> 200 before start. Card total per GPU 0-2 = ~97292 MiB
(95.0 GiB) per driver: total 102016548864 bytes.

## (b) Lifecycle milestones -> GPU used (identical on ranks/GPUs 0-2)

| Engine log time | Milestone | GPU used MiB each | Note |
|---|---|---|---|
| 12:46:42-46 | LMCache MP server up; shm L1 free 49152 MB | (t0 = 0) | ZMQ 15807, HTTP 15808 |
| 12:47:09-11 | NCCL init world_size=3; PYNCCL all-reduce fallback (ws 3 unsupported by B12X PCIe/custom allreduce) | — | |
| 12:47:12 | Offloader set to UVAOffloader; loading model from scratch | — | |
| 12:47:19 | ds41-tp3 overlay: offload walk from ordinal 20, [(20,'2.24'),(21,'2.24'),(22,'2.2')]; Total CPU offloaded parameters: 6.68 (every rank) | — | expandable_segments first-map OOM warnings per rank (map 20971520 B, free 3473408) |
| 12:48:11 | Loading weights took 51.90 s (InstantTensor, 286 GiB @ ~6.1 GB/s) | — | |
| 12:48:14 | Model loading took 88.52 GiB memory and ~62.28 s (per rank) | — | TP0 62.2786 s / TP1 62.2792 s / TP2 62.2788 s |
| 12:48:22 | Initial free memory 93.53 GiB; reserved 1.05 GiB KV via kv_cache_memory_bytes; GPU KV cache size 1,131,122 tokens; concurrency 1.08x @1M; 4933 serving blocks (+32 autotune) | — | |
| 12:48:39 | JIT kernel warmup finished 0.07 s; FlashInfer autotune no-op (0 configs) | — | |
| 12:48:42 | Capturing CUDA graphs (FULL) finished in 2 s, took 0.02 GiB | — | |
| 12:48:43 | init engine (profile, create kv cache, warmup) took 29.23 s | — | |
| 12:48:47 | vLLM server on http://0.0.0.0:5807; /health 200 observed | — | |

GPU polls (nvidia-smi):

| Host clock | GPU 0-2 used MiB each | free MiB each |
|---|---|---|
| 14:46:04 (t0) | 0 | 97291 |
| 14:48:59 (ready; health=200 t=0.001 s) | 96190 | 1102 |
| 14:49:05 (post-smoke steady) | 96202 | 1090 |
| 14:49:51 (after stop) | 0 | 97291 |

## (c) Final steady-state per GPU 0-2

used 96202 MiB, free 1090 MiB each (GPUs 0-2). GPU 3 companions unchanged (~96595 MiB used).

## (d) Per-rank offload / load accounting

| Rank (Worker pid) | Model loading took | seconds | Total CPU offloaded parameters |
|---|---|---|---|
| TP0 (507) | 88.52 GiB | 62.2786 | 6.68 GiB (ords 20=2.24, 21=2.24, 22=2.20) |
| TP1 (508) | 88.52 GiB | 62.2792 | 6.68 GiB (same walk) |
| TP2 (509) | 88.52 GiB | 62.2788 | 6.68 GiB (same walk) |

Weights stream: 286 GiB safetensors, load 51.90 s (~6.1 GB/s).

## (e) Smoke request (steady state)

POST http://127.0.0.1:5807/v1/chat/completions, model DeepSeek-V4.1-Flash,
"Reply with exactly: 3/10", max_tokens=8:
HTTP 200; content=null, reasoning="The user wants exactly \"3/10"; finish_reason=length;
usage prompt=38 completion=8 (reasoning=8); x-vllm fingerprint vllm-0.1.dev21735+g2b86d3d80-tp3.
curl time_total = 0.466665 s.

## (f) Worker RSS vs nvidia per-process — allocator slack estimate

| Worker (host pid) | RSS (ps) | RSS GiB | nvidia compute-apps used | claimed (model+KV+graphs) | slack |
|---|---|---|---|---|---|
| TP0 2563128 | 53884196 KiB | ~51.39 | 96060 MiB = 93.81 GiB | 88.52+1.05+0.02 = 89.59 GiB | ~4.2 GiB |
| TP1 2563129 | 53879796 KiB | ~51.38 | 96060 MiB = 93.81 GiB | 89.59 GiB | ~4.2 GiB |
| TP2 2563130 | 53887128 KiB | ~51.39 | 96060 MiB = 93.81 GiB | 89.59 GiB | ~4.2 GiB |

Driver/context overhead: GPU used 96202 - compute-apps 96060 = ~142 MiB per GPU.
Host RSS (~51.4 GiB/worker) covers pinned UVA host pools + InstantTensor staging; it does not
bound device allocations.
Decision arithmetic: steady-state spare is only ~1090 MiB per rank. Restoring the 6.68 GiB
(~6840 MiB) decoder-half experts to GPU would need ~102.9 GiB-equivalent vs ~95 GiB capacity —
dropping the UVA expert offload does NOT fit at gpu_memory_utilization=0.98 with KV fixed at
1.05 GiB unless another ~5.5 GiB/rank is trimmed elsewhere. The 12:47:19 expandable_segments
first-map OOM warnings confirm tight headroom already during weight load.

## (g) Env knobs observed (podman exec ... tr "\0" "\n" < /proc/1/environ)

Placement knobs on PID 1: PROFILE=ds41-flash, HARDWARE_PROFILE=rtx-pro-6000-pcie, TP=3,
CUDA_DEVICE_ORDER=PCI_BUS_ID, DS41_OFFLOAD_FROM_ORDINAL=20, NVIDIA_VISIBLE_DEVICES=void,
VIRTUAL_ENV=/opt/venv, PYTHONIOENCODING=utf-8, CUDA_VERSION=13.4.1.012,
CUDA_MODULE_LOADING=LAZY, TORCH_CUDA_ARCH_LIST="7.5 8.0 8.6 9.0 10.0 12.0+PTX".
No other VLLM_* vars in PID 1 environ; workers log VLLM_NCCL_SO_PATH=
/opt/venv/lib/python3.12/site-packages/local_inference_nccl/lib/libnccl.so.2.31.2, and vLLM
warned "Unknown vLLM environment variable detected: VLLM_CACHE_DIR". Non-placement NVIDIA_*
version pins (CUDNN 9.25.0.28, NCCL 2.30.7+cuda13.3, torch 2.14.0a0) come from the base image.

## Provenance

Container started detached (podman run, --replace --rm --init, GPUs 0-2 only), reached
ready in ~2m10s, then stopped via podman stop (exit 143; --rm removed the container).
After stop: GPUs 0-2 used 0 MiB / free 97291 MiB; llama-swap /health 200 on :8080.
