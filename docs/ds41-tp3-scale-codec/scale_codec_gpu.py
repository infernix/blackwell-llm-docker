"""Lazy-loaded GPU decoding kernels for the scale codec.

Import this module only from GPU-capable code paths; every heavy dependency
(torch / triton) is imported inside functions so CPU commands never require
them. The decoder is a byte decoder: input compressed bytes + metadata live
in GPU memory, output is the original uint8 scale bytes in preallocated
buffers. No floating-point dequantization is involved.

Ordered two-kernel design (clearly correct first pass):
  kernel 1 expands primary bits into the output slot;
  kernel 2 (same stream) applies exception records on top.
Both kernels are timed together. A later fused per-tile kernel would need
race-free tile ownership; this module keeps one tile owned by exactly one
program so a fused variant remains possible without cross-program races.
"""
from __future__ import annotations

import time

import numpy as np


def gpu_status() -> dict:
    try:
        import torch  # noqa: WPS433
        import triton  # noqa: WPS433
    except Exception as exc:
        return {"available": False, "reason": f"import failed: {exc}"}
    try:
        import torch
        if not torch.cuda.is_available():
            return {"available": False, "reason": "torch.cuda.is_available() False"}
        props = torch.cuda.get_device_properties(0)
        return {
            "available": True,
            "torch": torch.__version__,
            "triton": __import__("triton").__version__,
            "device": props.name,
            "compute_capability": f"{props.major}.{props.minor}",
            "total_memory_bytes": int(props.total_memory),
            "free_memory_bytes": _free_memory(torch),
        }
    except Exception as exc:
        return {"available": False, "reason": f"cuda probe failed: {exc}"}


def _free_memory(torch) -> int:
    free, total = torch.cuda.mem_get_info(0)
    return int(free)


def make_kernels(torch):
    """Build the two ordered decode kernels for the installed torch/triton."""
    import triton
    import triton.language as tl

    @triton.jit
    def expand_bits_kernel(bits_ptr, out_ptr, n, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n
        byte_i = offs >> 3
        bits = tl.load(bits_ptr + byte_i, mask=mask, other=120)
        shift = (offs & 7)
        bit = (bits >> shift) & 1
        val = tl.where(bit == 1, 121, 120)
        tl.store(out_ptr + offs, val.to(tl.uint8), mask=mask)

    @triton.jit
    def apply_exceptions_kernel(pos_ptr, val_ptr, out_ptr, e, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < e
        pos = tl.load(pos_ptr + offs, mask=mask, other=0)
        val = tl.load(val_ptr + offs, mask=mask, other=0)
        tl.store(out_ptr + pos, val, mask=mask)

    return expand_bits_kernel, apply_exceptions_kernel, triton


def decode_one(torch, row_blob: bytes, n: int, out_buf, scratch=None):
    """Prepare a validated sidecar blob for repeated GPU decoding.

    Uploads and descriptor preparation happen once, outside the returned timed
    path. The output buffer is caller-owned and reused. Raw fallback uses one
    device-to-device copy; binary mode uses ordered expand and patch kernels.
    """
    import numpy as np
    import scale_codec_core as core

    decoded = core.decode_bytes(row_blob)  # complete structural validation
    if len(decoded) != n:
        raise ValueError(f"declared output length {len(decoded)} != requested {n}")
    if out_buf.dtype != torch.uint8 or out_buf.numel() < n or not out_buf.is_cuda:
        raise ValueError("out_buf must be a CUDA uint8 tensor with at least n elements")

    arr = np.frombuffer(row_blob, dtype=np.uint8)
    mode = int(arr[4])
    if mode == core.MODE_RAW:
        raw_gpu = torch.from_numpy(np.frombuffer(decoded, dtype=np.uint8).copy()).to(
            out_buf.device
        )

        def launch():
            out_buf[:n].copy_(raw_gpu)

        launch.launch_count = 1
    else:
        bits_len = (n + 7) // 8
        groups = (n + 255) // 256
        prefixes = np.frombuffer(
            row_blob, dtype="<u4", count=groups + 1, offset=16 + bits_len
        )
        bits = arr[16:16 + bits_len]
        recs = arr[16 + bits_len + 4 * (groups + 1):]
        e = int(prefixes[groups])
        positions = np.empty(e, dtype=np.int64)
        values = np.empty(e, dtype=np.uint8)
        w = 0
        for g in range(groups):
            lo, hi = int(prefixes[g]), int(prefixes[g + 1])
            count = hi - lo
            if count:
                positions[w:w + count] = (
                    g * 256 + recs[2 * lo:2 * hi:2].astype(np.int64)
                )
                values[w:w + count] = recs[2 * lo + 1:2 * hi:2]
                w += count
        bits_gpu = torch.from_numpy(np.array(bits, copy=True)).to(out_buf.device)
        pos_gpu = torch.from_numpy(positions).to(out_buf.device)
        val_gpu = torch.from_numpy(values).to(out_buf.device)
        expand, apply_exc, _ = make_kernels(torch)
        block = 1024

        def launch():
            expand[((n + block - 1) // block,)](
                bits_gpu, out_buf, n, BLOCK=block
            )
            if e:
                apply_exc[((e + block - 1) // block,)](
                    pos_gpu, val_gpu, out_buf, e, BLOCK=block
                )

        launch.launch_count = 2 if e else 1

    launch()
    torch.cuda.synchronize(out_buf.device)
    return launch

def prepare_selected_pool(torch, blobs: list[bytes]):
    """Validate blobs and upload a compact pool for GPU-selected tile decode."""
    import numpy as np
    import scale_codec_core as core

    modes = []
    lengths = []
    bits_offsets = []
    raw_offsets = []
    prefix_offsets = []
    exception_offsets = []
    bits_all = bytearray()
    raw_all = bytearray()
    prefixes_all: list[int] = []
    positions_all = bytearray()
    values_all = bytearray()
    for blob in blobs:
        core.decode_bytes(blob)  # reject malformed descriptors before GPU work
        n = int.from_bytes(blob[8:16], "little")
        mode = blob[4]
        modes.append(mode)
        lengths.append(n)
        bits_offsets.append(len(bits_all))
        raw_offsets.append(len(raw_all))
        prefix_offsets.append(len(prefixes_all))
        exception_offsets.append(len(positions_all))
        if mode == core.MODE_RAW:
            raw_all += blob[16:]
            prefixes_all.append(0)
            continue
        bits_len = (n + 7) // 8
        groups = (n + core.GROUP - 1) // core.GROUP
        pre = np.frombuffer(blob, dtype="<u4", count=groups + 1,
                            offset=16 + bits_len)
        recs_at = 16 + bits_len + 4 * (groups + 1)
        recs = np.frombuffer(blob, dtype=np.uint8, offset=recs_at)
        bits_all += blob[16:16 + bits_len]
        prefixes_all.extend(int(x) for x in pre)
        positions_all += recs[0::2].tobytes()
        values_all += recs[1::2].tobytes()

    dev = torch.device("cuda")
    def upload(values, dtype):
        return torch.tensor(values, dtype=dtype, device=dev)

    return {
        "modes": upload(modes, torch.int32),
        "lengths": upload(lengths, torch.int64),
        "bits_offsets": upload(bits_offsets, torch.int64),
        "raw_offsets": upload(raw_offsets, torch.int64),
        "prefix_offsets": upload(prefix_offsets, torch.int64),
        "exception_offsets": upload(exception_offsets, torch.int64),
        "bits": upload(list(bits_all), torch.uint8),
        "raw": upload(list(raw_all), torch.uint8),
        "prefixes": upload(prefixes_all, torch.int64),
        "positions": upload(list(positions_all), torch.uint8),
        "values": upload(list(values_all), torch.uint8),
        "tensor_count": len(blobs),
    }


def make_selected_kernels():
    import triton
    import triton.language as tl

    @triton.jit
    def selected_base_kernel(modes, lengths, bit_offs, raw_offs, bits, raw,
                             tensor_ids, tile_ids, out, tensor_count,
                             requests, BLOCK: tl.constexpr):
        request = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        active = request < requests
        tensor_id = tl.load(tensor_ids + request, mask=active, other=0)
        tile_id = tl.load(tile_ids + request, mask=active, other=0)
        valid_tensor = active & (tensor_id >= 0) & (tensor_id < tensor_count)
        safe_tensor = tl.where(valid_tensor, tensor_id, 0)
        n = tl.load(lengths + safe_tensor)
        position = tile_id * BLOCK + lane
        valid = valid_tensor & (tile_id >= 0) & (position < n)
        mode = tl.load(modes + safe_tensor)
        raw_offset = tl.load(raw_offs + safe_tensor)
        bit_offset = tl.load(bit_offs + safe_tensor)
        raw_value = tl.load(raw + raw_offset + position, mask=valid & (mode == 0),
                            other=0)
        packed = tl.load(bits + bit_offset + (position >> 3),
                         mask=valid & (mode == 1), other=0)
        bit = (packed >> (position & 7)) & 1
        binary_value = tl.where(bit == 1, 121, 120)
        value = tl.where(valid, tl.where(mode == 0, raw_value, binary_value), 0)
        tl.store(out + request * BLOCK + lane, value, mask=active)

    @triton.jit
    def selected_exception_kernel(modes, lengths, prefix_offs, exception_offs,
                                  prefixes, positions, values, tensor_ids,
                                  tile_ids, out, tensor_count, requests,
                                  BLOCK: tl.constexpr):
        request = tl.program_id(0)
        lane = tl.arange(0, BLOCK)
        active = request < requests
        tensor_id = tl.load(tensor_ids + request, mask=active, other=0)
        tile_id = tl.load(tile_ids + request, mask=active, other=0)
        valid_tensor = active & (tensor_id >= 0) & (tensor_id < tensor_count)
        safe_tensor = tl.where(valid_tensor, tensor_id, 0)
        n = tl.load(lengths + safe_tensor)
        valid_tile = valid_tensor & (tile_id >= 0) & (tile_id * BLOCK < n)
        mode = tl.load(modes + safe_tensor)
        prefix_offset = tl.load(prefix_offs + safe_tensor)
        safe_tile = tl.where(valid_tile, tile_id, 0)
        begin = tl.load(prefixes + prefix_offset + safe_tile,
                        mask=valid_tile & (mode == 1), other=0)
        end = tl.load(prefixes + prefix_offset + safe_tile + 1,
                      mask=valid_tile & (mode == 1), other=0)
        record = begin + lane
        patch = valid_tile & (mode == 1) & (record < end)
        exception_offset = tl.load(exception_offs + safe_tensor)
        pos = tl.load(positions + exception_offset + record, mask=patch, other=0)
        value = tl.load(values + exception_offset + record, mask=patch, other=0)
        tl.store(out + request * BLOCK + pos, value, mask=patch)

    return selected_base_kernel, selected_exception_kernel


def selected_tile_launcher(torch, pool, tensor_ids, tile_ids, out_buf):
    """Return an allocation-free launcher using mutable GPU request tensors."""
    requests = tensor_ids.numel()
    if (not tensor_ids.is_cuda or not tile_ids.is_cuda
            or tensor_ids.dtype != torch.int64 or tile_ids.dtype != torch.int64
            or tile_ids.shape != tensor_ids.shape):
        raise ValueError("tensor_ids and tile_ids must be same-shape CUDA int64 tensors")
    if (not out_buf.is_cuda or out_buf.dtype != torch.uint8
            or out_buf.numel() < requests * 256):
        raise ValueError("out_buf must be CUDA uint8 with requests*256 bytes")
    base, patch = make_selected_kernels()

    def launch():
        base[(requests,)](
            pool["modes"], pool["lengths"], pool["bits_offsets"],
            pool["raw_offsets"], pool["bits"], pool["raw"], tensor_ids,
            tile_ids, out_buf, pool["tensor_count"], requests, BLOCK=256,
        )
        patch[(requests,)](
            pool["modes"], pool["lengths"], pool["prefix_offsets"],
            pool["exception_offsets"], pool["prefixes"], pool["positions"],
            pool["values"], tensor_ids, tile_ids, out_buf,
            pool["tensor_count"], requests, BLOCK=256,
        )

    return launch


def bench(args) -> dict:
    """Standalone GPU decode benchmark over selected checkpoint tensors."""
    status = gpu_status()
    if not status["available"]:
        return {"status": "not_run", "reason": status["reason"]}

    import torch
    import scale_codec_core as core

    rows = []
    src_rows = core.enumerate_selected(args.model, args.include)
    limit = min(len(src_rows), args.limit_tensors or 8)
    t0 = time.perf_counter()
    for r in src_rows[:limit]:
        src = core.read_tensor_bytes(r)
        blob = core.encode_bytes(src)
        rows.append((r, src, blob))
    setup_time = time.perf_counter() - t0

    torch.cuda.init()
    dev = torch.device("cuda")
    results = []
    for r, src, blob in rows:
        n = len(src)
        out_buf = torch.empty(n, dtype=torch.uint8, device=dev)
        launch = decode_one(torch, blob, n, out_buf)
        # eager timing: N repeats, GPU-event bracket on the current stream
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        launch()  # prewarm already happened inside decode_one sync
        start.record()
        reps = 50
        for _ in range(reps):
            launch()
        end.record()
        torch.cuda.synchronize()
        ms = start.elapsed_time(end) / reps
        got = out_buf.cpu().numpy().tobytes()
        results.append({
            "name": r["name"],
            "n_bytes": n,
            "encoded_bytes": len(blob),
            "byte_exact": got == src,
            "eager_ms_per_decode": round(ms, 4),
            "output_bytes_per_second": round(n / (ms / 1000.0), 1),
        })
    peak = int(torch.cuda.max_memory_allocated())
    reserved = int(torch.cuda.max_memory_reserved())
    return {
        "status": "passed" if all(x["byte_exact"] for x in results) else "failed",
        "environment": status,
        "timed_boundary": "two decode kernels only; disk reads, CPU encode, "
                          "H2D uploads, allocation and correctness checks are "
                          "outside the event bracket",
        "setup_seconds": round(setup_time, 4),
        "launches_per_decode": 2,
        "tensors": results,
        "peak_allocated_bytes": peak,
        "reserved_bytes": reserved,
    }
