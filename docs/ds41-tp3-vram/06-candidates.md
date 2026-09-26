# DS4.1 TP3 — candidate VRAM runs (06)

Manual `podman run -d` of placement `ds41_3gpu_karmic_r1` (image ghcr.io/local-inference-lab/vllm@sha256:d9d273c6890..., `--port 5807`), one container at a time, GPUs 0-2 only. Readiness polled at http://127.0.0.1:5807/health → 200. Each run stopped afterwards with `podman stop ds41_3gpu_karmic_r1`.

## Comparison table

| run | env / setting changes | Model loading GiB (per rank) | Initial free GiB / reserved | GPU KV cache size | Graph capture | Total CPU offloaded params | steady used/free MiB (GPU 0,1,2) | smoke reply + finish_reason + curl time_total | OOM/warning lines |
|---|---|---|---|---|---|---|---|---|---|
| A (control) | none (file env verbatim: PROFILE, HARDWARE_PROFILE, TP=3, CUDA_DEVICE_ORDER, DS41_OFFLOAD_FROM_ORDINAL=20) | 88.52 GiB / 61.7s ×3 ranks | 93.53 GiB free, reserved 1.05 GiB KV (kv_cache_memory_bytes) | 1,131,122 tokens; max concurrency 1.08x @ 1,048,576 | finished in 2 secs, took 0.02 GiB (all ranks); sizes [1,2,4,8,16,24], max_cudagraph_capture_size 24 | 6.68 ×3 | used 96190/free 1102 on each of 0,1,2 (GPU3 companion: 96595/697) | \"0.3\", finish_reason=stop, 0.729447s | no OOM; see warning list below |
| B (+FI workspace trim) | `-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=205520896` (=196 MiB, plain integer bytes) | 88.52 GiB / 62.3s ×3 | 93.53 GiB free, reserved 1.05 GiB KV | 1,131,122 tokens; 1.08x | finished in 2 secs, took 0.02 GiB ×3 | 6.68 ×3 | used 96190/free 1102 each on 0,1,2 (GPU3 96635/657) | \"0.3\", stop, 0.724718s | no OOM; same warning set |
| B2 (third run, deeper trim) | `-e VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=167772160` (=160 MiB) | 88.52 GiB / 61.7s ×3 | 93.53 GiB free, reserved 1.05 GiB KV | 1,131,122 tokens; 1.08x | finished in 2 secs, took 0.02 GiB ×3 | 6.68 ×3 | used 96192/free 1100 each on 0,1,2 | \"0.3\", stop, 0.757985s | no OOM; same warning set |
| C (graph-capture trim) | `config/ds41-tp3-launch.yaml`: max-cudagraph-capture-size 24 → 16 (file read through bind mount at container start) | 88.52 GiB / 61.9s ×3 | 93.53 GiB free, reserved 1.05 GiB KV | 1,131,122 tokens; 1.08x | finished in 2 secs, took 0.02 GiB ×3; live args confirmed max_cudagraph_capture_size 16, cudagraph_capture_sizes [1,2,4,8,16] | 6.68 ×3 | used 96190/free 1102 each on 0,1,2 | \"0.3\", stop, 0.768628s | no OOM; same warning set |

Warning/error lines present identically in every run (none blocked readiness or smoke):
- `ERROR [config.py:29] Failed to import Triton kernels ... No module named 'triton_kernels.matmul_ogs'` (and same from mxfp4.py:61)
- `WARNING [envs.py:2409] Unknown vLLM environment variable detected: VLLM_CACHE_DIR`
- `WARNING [indexer_topk.py:29] Failed to import the DeepSelect extension (vllm._deepselect_C)`
- `WARNING [config.py:428] Padded DeepSeek V4.1 attention for TP3: heads 64 -> 72, output groups 8 -> 9.`
- `WARNING [b12x_pcie_all_reduce.py:296] B12X PCIe all-reduce initialization failed ... unsupported PCIe all-reduce world size 3` (×3 ranks)
- `WARNING [symm_mem.py:67] SymmMemCommunicator: Device capability 12.0 not supported`
- `WARNING [custom_all_reduce.py:184] Custom allreduce is disabled due to an unsupported world size: 3`
- `WARNING [torch_utils.py:275] OMP_NUM_THREADS=8 is set; leaving Torch threads at 8`
- `WARNING [compilation.py:1480] Cudagraph mode FULL_AND_PIECEWISE requires piecewise capture ... Overriding to FULL_DECODE_ONLY.`
- `WARNING [compilation.py:1361] Op 'quant_fp8' not present in model` / `WARNING [vllm.py:3614] torch.compile is turned on, but the model /model does not support it`
- `WARNING [base.py:219] Initializing KVConnectorBase_V1 ... experimental`
- LMCache `Session ... has no lookup ipc key, skipping touch` (first request only)
- During run B's first attempt only: `[rankN] expandable_segments: memory mapping failed with OOM on device N ... (free: 3473408)` — that attempt died on a config error, see below; it is not steady-state behavior.

## Exact diff made for run C

```
--- a/config/ds41-tp3-launch.yaml
+++ b/config/ds41-tp3-launch.yaml
@@ -19 @@
-  max-cudagraph-capture-size: 24
+  max-cudagraph-capture-size: 16
```
Applied with sed on `config/ds41-tp3-launch.yaml` before launching run C; the container read it through the read-only bind mount `/etc/lil-launch.yaml` at start, so no llama-swap reload was involved. After measuring, the file was restored to `max-cudagraph-capture-size: 24` (confirmed by grep: line 19 reads 24) — REVERTED, only the two values 24 → 16 → 24 were ever present.

## Env-value finding (run B, first attempt)

`VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=196MiB` fails startup: the value is parsed with `int()`, so suffixed strings abort the workers —
`ValueError: invalid literal for int() with base 10: '196MiB'` → `Exception: WorkerProc initialization failed due to an exception in a background process` → `RuntimeError: Engine core initialization failed`, container exit status 1. Passing plain byte counts (205520896, 167772160) starts fine. Container envs.py confirms the default is expressed as `VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE: int = 394 * 1024 * 1024`.

## Config validation

`go run . --config config.yaml --validate-config` needs the API-key env vars the live process has; inherited from pid 1280076 `/proc/<pid>/environ`:

```
config OK
VALIDATE-EXIT=0
```

First attempt without them failed with `Error loading config: environment variable 'LLAMA_SWAP_API_KEY_OBOT' is not set` (then `LLAMA_SWAP_API_KEY_OMP_BELIEVE`), exit 1 — an env-shape artifact, not a config defect.

Reload confirmation: `journalctl --no-pager -u llama-swap` returns `-- No entries --` (no systemd unit); the proxy runs as pid 1280076 `build/llama-swap-linux-amd64 --config config.yaml --watch-config --listen 0.0.0.0:8080` with stdout not captured to a readable log file, so no 'Configuration Reloaded' line was reachable. Per the assignment, the passing `--validate-config` output above is the recorded evidence, and this paragraph states which method was used.

## Verdict per candidate

- A (baseline): reference — steady used 96190 MiB / free 1102 MiB per GPU 0-2, health 200, smoke \"0.3\` stop in ~0.73s.
- B (FI workspace 196 MiB): NEUTRAL — steady used identical at 96190 MiB (free 1102), engine-side totals unchanged (model load 88.52 GiB, initial free 93.53 GiB, KV 1,131,122 tokens, graph pool 0.02 GiB). No reclaim measured; health/smoke fine.
- B2 (FI workspace 160 MiB): NEUTRAL/slightly worse — steady used 96192 MiB (+2 MiB), free 1100 MiB. Still health 200 and smoke OK, so the smaller value is harmless, but it buys nothing measurable.
- C (max-cudagraph-capture-size 16): NEUTRAL — steady used unchanged at 96190 MiB / free 1102 MiB, while dropping the 24-size graph capture (sizes become [1,2,4,8,16]). The graph pool is only ~0.02 GiB in this FULL_DECODE_ONLY path, so trimming it has no headroom to give.

Combined recommendation: keep the current settings exactly as they are (baseline A, max-cudagraph-capture-size 24, no FI workspace override) — none of the three candidates reduced steady used MiB on GPUs 0-2 beyond ±2 MiB of noise, and two of them cost something (C loses batch-24 graph coverage; B2 measured 2 MiB higher). To actually reclaim VRAM for a smaller decoder-half expert offload, the lever is the offload budget itself (`--cpu-offload-gb` / `DS41_OFFLOAD_FROM_ORDINAL`), moving weight bytes host→GPU one tensor step at a time, not these fixed buffers. If a future run does set the workspace env, remember it must be a plain integer byte count.

## Run B2 note

Working env form: VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE must be a plain integer byte count (e.g. 205520896). Suffix forms like '196MiB' crash worker init with ValueError from int().
