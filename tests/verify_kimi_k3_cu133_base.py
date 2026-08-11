#!/usr/bin/env python3
"""Verify the CUDA, PyTorch, NCCL, XGrammar, and Transformers base contract."""

from __future__ import annotations

import argparse
import ctypes
import importlib.metadata as metadata
import os
from pathlib import Path

from packaging.version import Version


FORBIDDEN_CUDA_WHEEL_PREFIXES = (
    "nvidia-cublas-",
    "nvidia-cuda-cupti-",
    "nvidia-cuda-nvrtc-",
    "nvidia-cuda-runtime-",
    "nvidia-cudnn-",
    "nvidia-cufft-",
    "nvidia-curand-",
    "nvidia-cusolver-",
    "nvidia-cusparse-",
    "nvidia-nccl-",
    "nvidia-nvjitlink-",
)


def _mapped_libraries(fragment: str) -> list[Path]:
    libraries: set[Path] = set()
    with open("/proc/self/maps", encoding="utf-8") as maps:
        for line in maps:
            path = line.rsplit(maxsplit=1)[-1]
            if fragment in path and path.startswith("/"):
                libraries.add(Path(path))
    return sorted(libraries)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-version", required=True)
    parser.add_argument("--cuda-version", required=True)
    parser.add_argument("--nccl-version-code", required=True, type=int)
    parser.add_argument("--nccl-library", required=True)
    parser.add_argument("--minimum-xgrammar-version", required=True)
    parser.add_argument("--minimum-transformers-version", required=True)
    args = parser.parse_args()

    import torch
    import transformers
    import xgrammar

    assert Version(torch.__version__).base_version == args.torch_version
    assert not torch.__version__.startswith(f"{args.torch_version}a")
    assert torch.version.cuda == args.cuda_version
    assert "USE_NCCL=1" in torch.__config__.show()

    nccl_path = Path(args.nccl_library).resolve(strict=True)
    nccl = ctypes.CDLL(str(nccl_path))
    version = ctypes.c_int()
    result = nccl.ncclGetVersion(ctypes.byref(version))
    assert result == 0, f"ncclGetVersion returned {result}"
    assert version.value == args.nccl_version_code
    assert any(
        path.exists() and os.path.samefile(path, nccl_path)
        for path in _mapped_libraries("libnccl")
    ), f"{nccl_path} is not mapped in the validation process"

    xgrammar_version = Version(metadata.version("xgrammar"))
    transformers_version = Version(metadata.version("transformers"))
    assert xgrammar_version >= Version(args.minimum_xgrammar_version)
    assert transformers_version >= Version(args.minimum_transformers_version)
    assert xgrammar.__file__
    assert transformers.__file__

    installed = {distribution.metadata["Name"].lower() for distribution in metadata.distributions()}
    forbidden = sorted(
        package
        for package in installed
        if package.startswith(FORBIDDEN_CUDA_WHEEL_PREFIXES)
    )
    assert not forbidden, f"pip CUDA runtime overlays are installed: {forbidden}"

    print(
        "Kimi-K3 CUDA base contract: PASS "
        f"torch={torch.__version__} cuda={torch.version.cuda} "
        f"nccl={version.value} xgrammar={xgrammar_version} "
        f"transformers={transformers_version}"
    )


if __name__ == "__main__":
    main()
