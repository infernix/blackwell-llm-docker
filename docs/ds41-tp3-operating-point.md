# DS4.1 Flash TP3 operating point

DeepSeek-V4.1-Flash served on three RTX PRO 6000 (SM120) GPUs with tensor
parallelism 3. This is the authoritative description of the verified serving
configuration; the llama-swap placement file that carries it is a thin
wrapper around the command and environment below.

## Launch

    podman run --replace --rm --init \
      --device nvidia.com/gpu=0 --device nvidia.com/gpu=1 --device nvidia.com/gpu=2 \
      --network host --ipc=host --security-opt seccomp=unconfined \
      --ulimit memlock=-1 --ulimit nofile=1048576 --ulimit stack=67108864 \
      -v <checkpoint-dir>:/model:ro -v <cache-dir>:/cache \
      -v $PWD/recipes/glm53/ds41-tp3-launch.yaml:/etc/lil-launch.yaml:ro \
      <image> \
      --profile ds41-flash --hardware rtx-pro-6000-pcie \
      --settings /etc/lil-launch.yaml --port 5807 \
      --scheduling-policy priority \
      -- --kv-cache-memory-bytes 536870912 \
         --cpu-offload-gb 8 --cpu-offload-params experts vision aligner

`recipes/glm53/serve-ds41-tp3.sh` applies the profile defaults and passes the
native flags after `--`. The launcher rejects a duplicated managed option, so
each cache default applies only when the caller did not pass the same option.

## Environment (all `-e` on the run command)

| variable | value | purpose |
|---|---|---|
| `DS41_OFFLOAD_FROM_ORDINAL` | 20 | decoder-half boundary; only layers >= 20 are eligible for host-backed staging during boot |
| `DS41_OFFLOAD_TO_ORDINAL` | 22 | upper ordinal window for staged residency |
| `DS41_VMM_ENABLE` | 1 | contiguous VMM regions per routed-expert parameter so promotion reuses addresses |
| `DS41_VMM_COLD_FRACTION` | 1 | fraction of experts initially cold under the VMM walk |
| `DS41_ROPE_CACHE_ROWS` | 524288 | size the RoPE cos/sin cache to the 524,288-token context |
| `DS41_EMBEDDING_UVA` | 1 | keep the input embedding weight pinned in host RAM, accelerator view on GPU |
| `B12X_SCALE_CODEC` | 1 | lossless E8M0 prepared-scale planes; ~4.5 GiB saved per rank |
| `B12X_DEFERRED_WEIGHT_BANKS` | 1 | every routed expert's WS4 bank starts in host RAM |
| `B12X_DEFERRED_BANK_RESERVE_BYTES` | 1073741824 | per-rank VRAM reserve kept free during promotion |
| `B12X_DEFERRED_BANK_ARM_FILE` | /cache/deferred-banks.arm | arm file for the post-health promotion |
| `B12X_COMPILE_CACHE_DIR` | /cache/b12x-compile | persistent B12X compile cache on fast NVMe |
| `VLLM_PCIE_ONESHOT_ALLREDUCE_MAX_SIZE` | 256KB | one-shot PCIe all-reduce threshold, TP3 |
| `PYTORCH_CUDA_ALLOC_CONF` | expandable_segments:True | allocator fragmentation control |

Expert ordinals 20-22 must stay host-backed during startup to prevent OOM
before scale compression completes; promotion happens after the health gate.

## Expected steady state

Every TP rank logs after the health gate:

    b12x deferred weight promotion free_before=6185877504 reserve=1073741824 promoted=2264924160 hot=384 cold=0
    b12x deferred weight promotion free_before=3920953344 reserve=1073741824 promoted=2264924160 hot=384 cold=0

`hot=384 cold=0` on TP0/TP1/TP2 is the full-residency signature: all routed
experts end GPU-resident; nothing stays host-backed in steady state. Only the
input embedding, vision tower and aligner remain in pinned host RAM. Measured
decode ~127-150 tok/s at these settings.

## Capacity tier (recipes/glm53/ds41-tp3-launch.yaml)

Fixed 512 MiB KV pool per rank, max-model-len 524288, max-num-seqs 4,
max-num-batched-tokens 1024, max-cudagraph-capture-size 24, gpu-memory-
utilization 0.93, engram tables on disk. This tier trades the verified 1M
context for post-start expert-residency headroom while keeping 524,288-token
requests. LMCache is the single cache owner: engine-driven L1 48 GiB pinned
host DRAM, lazily grown, L2 disabled; native KV offload is not combined with
it. Boot to health ~110-165 s with warm compile caches.

## Verification

1. Entrypoint verifier: `python /opt/venv/libexec/verify_qwen38_runtime.py
   --foundation ngc --defer-native-imports` returns
   `qwen38_runtime=PASS` (generic stack gate, not model-specific).
2. Health: `GET /health` on the served port after boot.
3. Residency: grep container logs for `hot=384 cold=0` on all three ranks.
4. Serving: `POST /v1/chat/completions` with served name
   `DeepSeek-V4.1-Flash` returns a completion; throughput ~127 tok/s+.

## Operational pitfalls

- If a manual run leaves `/dev/shm/lmcache_l1_pool_lmcache-ds41-flash-<port>-<port>[.lock]`
  behind, the next boot fails with "Refusing to reuse an existing cache SHM
  arena". Delete the stale shm files; behavior is intended isolation, not a bug.
- Compile caches (B12X, Triton, TorchInductor) live on the host NVMe under the
  `/cache` mount; keeping them makes boot ~110 s instead of a cold JIT build.
- The scale codec requires pinning `moe.decode` to its heuristic default with
  autotuning enabled so preparation settles through the pinned path.

## Component revisions (assembled image)

Wheels installed from source-locked releases whose tags embed the exact commit:
vllm `6c679489a0444f141adfba785de537741b60f497`, b12x
`d7944fe29abf8f2cf2ac1d22d99df425150c765d`, lmcache
`0b8082a5b559d3fe882898c10fed055c91d773f2`, flashinfer
`2206a14e46387a56c093860a46bbbdd00596b75b`, nccl-canonical
`93fe05d9f9b6963ef841166a69cd0b30e4efe97b`, InstantTensor
`95d4729b6d6a991bb8de61877147a9d9d9100b23`, all on branch `ds41f-tp3`.
Foundation: `nvcr.io/nvidia/pytorch@sha256:33ef5fc15e8937602d64022209cdb2777b32dadf742f41332023d946041b3c14`
(CUDA 13.4.1, torch 2.14.0a0+4fdf77b940), neutralized of profile-owned env
per `runtime/platform-environment.json`. Build via
`tools/jovian_wheel_runtime/assemble_qwen38_runtime_bundle.py` then
`Dockerfile.runtime`; final image is the 67 foundation layers plus one
application layer.
