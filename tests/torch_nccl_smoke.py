#!/usr/bin/env python3
"""Verify that PyTorch collectives use the packaged NCCL library on every GPU."""

from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.distributed as dist


def _mapped_nccl_libraries() -> set[Path]:
    libraries: set[Path] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rsplit(maxsplit=1)[-1]
            if "libnccl" in path and path.startswith("/"):
                libraries.add(Path(path))
    return libraries


def main() -> None:
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    expected_library = Path(os.environ["NCCL_LOCAL_INFERENCE_PATH"]).resolve(strict=True)

    assert torch.cuda.device_count() == world_size
    assert torch.cuda.nccl.version() == (2, 31, 2)

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")

    value = torch.tensor(rank + 1, device=f"cuda:{local_rank}", dtype=torch.float64)
    dist.all_reduce(value)
    torch.cuda.synchronize(local_rank)

    expected_value = world_size * (world_size + 1) // 2
    assert value.item() == expected_value, (rank, value.item(), expected_value)
    assert any(
        path.exists() and os.path.samefile(path, expected_library)
        for path in _mapped_nccl_libraries()
    ), f"rank {rank} did not map {expected_library}"

    dist.barrier(device_ids=[local_rank])
    if rank == 0:
        print(
            "Kimi-K3 PyTorch NCCL collective: PASS "
            f"ranks={world_size} all_reduce={expected_value} library={expected_library}"
        )
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
