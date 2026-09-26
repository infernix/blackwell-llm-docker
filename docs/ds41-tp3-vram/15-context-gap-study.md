# 15 — Context-gap study: engine-claimed vs nvidia per-process memory

Date: 2026-09-24 (~14:26). Read-only live probes against RUNNING placement `ds41_3gpu_karmic_r1`
(container `ds41_3gpu_karmic_r1`, engine :5807). No config changed; one transient 1-KiB-tensor
probe process was started inside the container and exited (measured 144 MiB, see §C).

## 0. Important: the placement was restarted since notes 01b/04

`podman top` ELAPSED 26m at probe; container logs start 14:00:26. The 12:46–12:48 run behind
01b/04 (load 88.52, per-process 96060 MiB) is gone. **This file uses the live run only**;
older numbers are quoted where they corroborate structure, not magnitude.

## A. Live measurement (per rank)

```
$ podman exec ds41_3gpu_karmic_r1 nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
508, 95176 MiB      # Worker_TP0 (host pid 2620683 via podman top)
509, 95176 MiB      # Worker_TP1 (host 2620684)
510, 95176 MiB      # Worker_TP2 (host 2620685)
$ nvidia-smi --query-gpu=index,memory.total,memory.used,memory.free --format=csv,noheader
0/1/2: 97887 MiB total, 95308 used, 1984 free   (GPU3 = companion sglang, PIDs 2404093/2404178)
```

Engine claims this run (`podman logs ds41_3gpu_karmic_r1`):

- `Model loading took 87.58 GiB memory and 62.594022 seconds` (all three ranks, 14:01:53)
- `Initial free memory 93.53 GiB, reserved 1.05 GiB memory for KV Cache as specified by
  kv_cache_memory_bytes` (14:02:08); metrics `kv_cache_memory_bytes="1124073472"` = 1072.0 MiB
- `Graph capturing finished in 2 secs, took 0.02 GiB` (14:02:25)

| quantity | MiB |
|---|---|
| nvidia per-process | 95,176 |
| engine-claimed: load 87.58 GiB (89,682) + KV 1,072 + graphs 20 | 90,774 |
| **GAP** | **4,402 ≈ 4.30 GiB** |

(Old run: 96,060 − 91,736 = 4,324 MiB ≈ 4.22 GiB — same structure, so the gap is restart-stable.)

## B. Accounting semantics (why the gap exists at all) — installed tree

`<SP>` = installed site-packages under overlay 254d59ab…/diff/opt/venv/lib/python3.12/site-packages.

1. **The init snapshot is taken AFTER NCCL init** — gpu_worker.py:465–467:
   `"Initialize the distributed environment BEFORE taking memory snapshot / This ensures NCCL
   buffers are allocated before we measure available memory"`, snapshot at :481–483. So the
   pre-load floor (CUDA context + NCCL init + runtime module staging) is **excluded** from every
   engine claim; it only shows in nvidia per-process.
2. **`Model loading took` = mem_get_info delta across the weight-copy window only**
   (mem_utils.py:297–300 `total_consumed = before_create.free_memory − after_profile.free_memory`,
   consumed by model_runner.py:443–448). FI workspaces, KV pool, NCCL first-collective buffers and
   graphs are all allocated **after** that window → also excluded from the claim.
3. **Driver reservation is invisible to CUDA**: torch `mem_get_info` total 97,290 MiB vs nvidia
   97,887 MiB → 597 MiB per GPU that no CUDA-side accounting can ever see (measured via transient
   probe, §C).

## C. Decomposition of the 4,402 MiB/rank gap

Anchors (all per rank):

- Free at init snapshot: 93.53 GiB = 95,775 MiB → nvidia-view used at snapshot = 97,887 − 95,775
  = **2,112 MiB pre-load floor**; torch-view used = 97,290 − 95,775 = 1,515 MiB; difference
  597 MiB = driver reservation.
- Post-load growth (nvidia view) = 95,176 − 2,112 = 93,064 MiB; engine-claimed post-snapshot
  items = 89,682 + 1,072 + 20 = 90,774 → **2,290 MiB post-load unnamed**.
- Bare CUDA context (transient in-container probe, `torch.cuda.mem_get_info` before/after
  `torch.cuda.init()` + one 1-KiB tensor, CUDA_MODULE_LOADING=LAZY): free 1425→1281 MiB
  = **144 MiB**; init alone = 0 delta (LAZY defers module load to first kernel). Worker has 39
  /dev/nvidia fds each (host /proc/2620683–85/fd), so its real context+module data exceeds this.

| # | component | MiB/rank | basis | confidence |
|---|---|---|---|---|
| 1 | Driver reservation (nvidia total − CUDA total) | 597 | §B.3 measured | measured |
| 2 | CUDA context + runtime/NCCL module staging, pre-load | ~1,515 | snapshot anchor minus 144 bare-context floor | measured total; composition [INFERENCE] |
| 3 | NCCL init (pre-snapshot) | ~200–300 (inside #2) | PYNCCL 2.31.2 init, 1 communicator, NCCL_MAX_NCHANNELS=16; no NCCL_DEBUG lines available | [INFERENCE] |
| 4 | Post-load: FlashInfer workspaces, 2 × 394 MiB | 788 | flashinfer.py:96 default `394*1024*1024`, :109–112 `trtllm_workspace_buffer`; mla/flashinfer_mla_sparse.py:326 base 394 MiB, dcp_world_size=1 → no softmax slab; both live per 03-runtime-screening.md:43 | high |
| 5 | Post-load: KV pool metadata/rounding over claim | +52 | block pool 4,419 blk × 166,912 B = 703.4 MiB + 420.7 metadata = 1,124.1 vs claimed 1,072.0 | high |
| 6 | Post-load: NCCL first-collective ring buffers | ~128–256 | 16 ch × 4 MiB (NCCL_BUFFSIZE default) × 2 (send+recv); first all-reduce happens in post-load warmup; no NCCL_DEBUG to confirm | [INFERENCE] |
| 7 | Post-load: CUDA graphs | 20 | claimed and measured (capture_model start/end free delta, gpu/model_runner.py:1250–1256) | measured |
| 8 | Post-load residual: lazy module data + JIT kernels + runtime pools | ~1,250–1,350 | 2,290 − (788 + 52 + ~200 + 20) | [INFERENCE] residual |

Check: 597 + 2,112-floor remainder vs post-load: 2,112 (pre-load) + 2,290 (post-load) = 4,402 ✓.

**The single biggest slab is the pre-load floor (2,112 MiB = driver 597 + CUDA-visible 1,515).**
Its composition beyond the 144 MiB bare context is not resolvable live (no NCCL_DEBUG; allocator
stats gauges absent from :5807/metrics — only `vllm:cache_config_info` and kv usage gauges exist).
Follow-up that would nail it: instrumented cold load with per-phase `nvidia-smi` polls and
NCCL_DEBUG=INFO (needs a restart → out of scope here).

## D. NCCL footprint and shrink knobs

Worker environ (host /proc/2620683/environ, all ranks identical): `NCCL_IB_DISABLE=1`,
`NCCL_MAX_NCHANNELS=16`, `NCCL_PROTO=LL,LL128,Simple`; **no** `NCCL_BUFFSIZE` (default 4 MiB),
**no** `NCCL_NVLS_ENABLE`/`NCCL_P2P_DISABLE`. Lib: `VLLM_NCCL_SO_PATH=…local_inference_nccl/lib/
libnccl.so.2.31.2` (logs :132–135, `vLLM is using nccl==2.31.2`). Fallback context: B12X PCIe and
custom allreduce both refuse ws=3 (01b:23–24), so allreduce = PYNCCL ring on PCIe.

Ring buffer math (ws3, per rank): channels × buffsize × 2 ≈ 16 × 4 MiB × 2 = **128 MiB**, plus
init-time internal staging (~200–300 MiB [INFERENCE], component #3).

| knob | saving (est) | risk |
|---|---|---|
| `NCCL_BUFFSIZE=1048576` (4→1 MiB) | ~96 MiB | Simple-protocol large prefills chunk into more round trips; LL/LL128 small-message decode largely unaffected. Prefill all-reduce hit est 5–15% of comm time |
| `NCCL_MAX_NCHANNELS=8` | ~64–128 MiB | decode messages here are small (hidden 5120 × bf16 × ≤24 tokens ≈ 240 KB → LL path); low decode risk, mild prefill risk |
| `NCCL_NVLS_ENABLE=0` | 0 | no-op: NVLS needs NVLink multicast; this box is PCIe (HARDWARE_PROFILE=rtx-pro-6000-pcie) |
| `NCCL_P2P_DISABLE=1` | negative | forces SHM host bounce — never |
| drop `Simple` from NCCL_PROTO | ~0 | hurts large prefill all-reduce badly |

Realistic NCCL reclaim: **~0.10–0.15 GiB/rank** with prefill-side throughput risk; decode at TP3
PCIe is already on the slow ring (44.9 µs vs 6.0 µs one-shot at 256 KiB, prior finding) so do not
cut further without an A/B.

## E. FlashInfer workspace attribution (why earlier env trims were neutral)

- Main prefill workspace: flashinfer.py:1018–1041 `_get_workspace_buffer` sets
  `buffer_size = max(VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE, est)` with
  `est = max_num_batched_tokens × num_qo_heads × head_dim × 16` (:96, 16 B/elem). For this shape:
  2048 × 22 heads/rank × 512 × 16 ≈ 352 MiB < 394 MiB default → **the env var is only a floor**;
  runs B (196 MiB) and B2 (160 MiB) could not shrink it (flashinfer.py:1037). This run sets no env
  at all → 394 MiB.
- Sparse MLA workspace: mla/flashinfer_mla_sparse.py:326 `_DEFAULT_WORKSPACE_BUFFER_SIZE` = 394 MiB;
  :370–372 honors an env override below the computed size **with an overflow warning**; with
  dcp_world_size=1 (`--decode-context-parallel-size 1`) no softmax slab is added → 394 MiB.
- Both buffers are live for this model (03-runtime-screening.md:43) → 788 MiB total (#4).
- Open inconsistency: B/B2 nvidia used was identical to control despite the sparse-ws shrink those
  runs should have produced (~200 MiB). Either the sparse workspace was not allocated in those runs
  or the saving was reabsorbed elsewhere; not resolvable from saved notes. Treat the 788 MiB as the
  live-run measurement, and treat B/B2 neutrality as unresolved.
- Reducible: **code-only** — honor the env override in flashinfer.py:1037 (or lower the default for
  this shape). Floor est 352 MiB bounds the main-ws saving at ~42 MiB; sparse ws 394→~196 MiB is
  plausible (~198 MiB) since dcp=1 needs no slab. Crash risk if underprovisioned (source warns).
  Realistic **~0.20–0.25 GiB/rank**.

## F. Allocator state

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` confirmed in worker environ; expandable first-map
OOM warnings during load (logs :165–167, free 1,376,256 B at that moment). `/metrics` exposes **no**
allocator reserved/allocated gauges (only cache_config_info / kv usage), so reserved-unallocated
waste is not observable live. Bounded indirectly: 08-no-offload-test.md:12 measured PyTorch
allocated 92.37 GiB with only 38.9 MiB reserved-unallocated at OOM → fragmentation waste on this
workload is tens of MiB, not GiB. No allocator knob to pull.

## G. Graphs / JIT / standalone-compile artifacts

- `VLLM_USE_STANDALONE_COMPILE=1` and `VLLM_USE_MEGA_AOT_ARTIFACT=1` in worker environ; no
  artifact-load lines in this run's logs; no measurable deserialized-artifact device footprint
  identified (jit cache on host = 497 MB, binaries 23.7 MB — host-side, not device).
- JIT warmup 0.07 s, 28 compile keys (logs :457–463); graph capture 2 graphs/rank, 0.02 GiB
  (#7). `torch.compile` is OFF for the model (logs :174–177: "does not support it"). No knob with
  materializable device savings.

## H. Verdict

| slice | GiB/rank | reducible? |
|---|---|---|
| Driver reservation | 0.58 | no |
| Pre-load floor: context + NCCL init + module staging | 1.48 (144 measured bare + rest [INFERENCE]) | no knob identified; needs instrumented cold load to attribute |
| FI workspaces | 0.77 | ~0.20–0.25 via code patch (env floor + sparse override) |
| KV metadata/rounding | 0.05 | no |
| NCCL collective ring (post-load) | ~0.13–0.25 | ~0.10–0.15 via BUFFSIZE/MAX_NCHANNELS, prefill risk |
| Graphs | 0.02 | already minimal |
| Post-load residual (module data / JIT / pools) | ~1.25 | no knob identified (LAZY already on) |

**Total realistic bound: ~0.3–0.4 GiB/rank** (FI patch + NCCL knobs), stretch to ~1.5–2.5 only if
instrumented cold-load attribution finds the two ~1.2–1.5 GiB staging residuals tamable — no
current evidence for that.

**Plain statement:** over half the gap (≈2.1 of 4.3 GiB/rank) is pre-load floor — driver
reservation + CUDA context + NCCL init + module staging — plus ~1.25 GiB of post-load lazy module
data; none of it reachable by env or config. The "context+NCCL floor" explanation is essentially
correct, and the last big phantom lever is dead: realistic reclaim 0.3–0.4 GiB/rank cannot fund the
2.24 GiB/rank partial-ordinal step (needs ≥2.24 with only ~2.0 GiB free today).

## Provenance

Live: podman exec/top/logs on ds41_3gpu_karmic_r1 (2026-09-24 14:2x); nvidia-smi host+container;
/proc/2620683–85 environ+fd; transient in-container torch probe (144 MiB, exited). Source:
installed tree `<SP>/vllm/v1/worker/gpu_worker.py:465-483,627-638`, `<SP>/vllm/utils/mem_utils.py:104-176,297-300`,
`<SP>/vllm/v1/worker/gpu/model_runner.py:1250-1256`, `<SP>/vllm/v1/attention/backends/flashinfer.py:96,1018-1041`,
`<SP>/vllm/v1/attention/backends/mla/flashinfer_mla_sparse.py:324-408`, `<SP>/vllm/multimodal/gpu_ipc_memory.py:155-223`.
Prior notes: 01b, 03:39-43, 04, 05, 06 (runs A/B/B2/C), 08.
