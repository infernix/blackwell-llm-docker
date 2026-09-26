import numpy as np
import torch
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import Int32, Uint32
from cutlass.cute.runtime import from_dlpack
from b12x.moe._shared.kernels.dynamic import _w4a8_stage_codec_sfb
from b12x._lib.intrinsics import ld_shared_u32


def encode(src):
    raw = src.view(torch.uint8).flatten().cpu().numpy()
    assert src.is_contiguous() and raw.size % 1024 == 0
    groups = raw.size // 256
    bits = np.packbits(raw == 121, bitorder='little')
    positions = np.flatnonzero((raw != 120) & (raw != 121))
    if groups > np.iinfo(np.uint32).max or positions.size > np.iinfo(np.uint32).max:
        raise OverflowError("codec exceeds uint32 group/prefix range")
    counts = np.bincount(positions // 256, minlength=groups)
    prefix = np.concatenate(([0], np.cumsum(counts))).astype(np.uint32)
    records = np.empty(positions.size * 2, dtype=np.uint8)
    records[0::2] = positions % 256
    records[1::2] = raw[positions]
    blob = np.array([groups], dtype=np.uint32).tobytes() + bits.tobytes() + prefix.tobytes() + records.tobytes()
    blob += bytes((-len(blob)) % 4)
    return torch.from_numpy(np.frombuffer(blob, dtype=np.uint32).copy()).cuda()


@cute.kernel
def probe_kernel(encoded: cute.Tensor, offsets: cute.Tensor, output: cute.Tensor, stage_bytes: cutlass.Constexpr):
    tid, _, _ = cute.arch.thread_idx()
    block, _, _ = cute.arch.block_idx()
    smem = utils.SmemAllocator()
    shared = smem.allocate_tensor(Uint32, cute.make_layout((stage_bytes // 4,)), 16)
    base = Int32(shared.iterator.toint())
    _w4a8_stage_codec_sfb(encoded, base, offsets[block], Int32(tid), 32, stage_bytes)
    cute.arch.sync_threads()
    for i in cutlass.range_constexpr(stage_bytes // 128):
        word = Int32(tid) + Int32(i * 32)
        output[block, word] = ld_shared_u32(base + word * Int32(4))


@cute.jit
def probe(encoded: cute.Tensor, offsets: cute.Tensor, output: cute.Tensor, stage_bytes: cutlass.Constexpr):
    probe_kernel(encoded, offsets, output, stage_bytes).launch(grid=(offsets.shape[0], 1, 1), block=(32, 1, 1))


def validate(src, encoded):
    rows = []
    # Every tile/half in all experts; output is only a bounded probe batch.
    raw = src.view(torch.uint8).flatten()
    for size in (512, 1024):
        output = torch.empty((256, size // 4), device='cuda', dtype=torch.uint32)
        offsets = torch.empty(256, device='cuda', dtype=torch.int64)
        compiled = cute.compile(probe, from_dlpack(encoded), from_dlpack(offsets), from_dlpack(output), size)
        for start in range(0, raw.numel(), size * 256):
            count = min(256, (raw.numel() - start) // size)
            cpu_offsets = torch.arange(256, dtype=torch.int64).clamp(max=count-1) * size + start
            offsets.copy_(cpu_offsets)
            compiled(from_dlpack(encoded), from_dlpack(offsets), from_dlpack(output))
            assert torch.equal(output[:count].view(torch.uint8).flatten(), raw[start:start+count*size])
        rows.append(dict(stage_bytes=size, tiles=raw.numel() // size, exact=True))
    return rows
